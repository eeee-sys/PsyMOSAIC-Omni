"""Official TRL 0.23.1 standard GRPO for the Diagnosis decoder LoRA.

Only the memory-efficient completion log-probability calculation and greedy
validation are adapted for the Omni Thinker. TRL owns rollout sampling, grouped
advantages, clipped GRPO objective, reference KL, accumulation and checkpointing.
No SFT initialization is performed.
"""
import sys as _sys
from pathlib import Path as _CorePath
for _core_root in _CorePath(__file__).resolve().parents:
    if (_core_root / "project_paths.py").is_file():
        if str(_core_root) not in _sys.path:
            _sys.path.insert(0, str(_core_root))
        break
from project_paths import (load_config, expand_paths, get_api_key, LORA_ROOT,
                           MODEL_ROOT, OUTPUT_ROOT, RAW_DATA_ROOT, VENDOR_ROOT,
                           CODE_ROOT as PROJECT_CODE_ROOT, DATA_ROOT as PROJECT_DATA_ROOT)


import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.utils.checkpoint import checkpoint
from datasets import Dataset
from transformers import TrainerCallback, set_seed
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import GRPOConfig, GRPOTrainer

from model import (add_decoder_lora, generate_text_batch,
                           load_text_thinker, load_tokenizer)
from rewards import build_reward_function, parse_response


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def token_audit(rows, tokenizer, config):
    lengths = []
    for row in rows:
        if not isinstance(row["prompt"], list):
            raise ValueError("Each prompt must be a list of chat messages")
        if row["ground_truth_answer"] not in ("POSITIVE", "NEGATIVE"):
            raise ValueError("Training target must be POSITIVE or NEGATIVE")
        length = len(tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                                  add_generation_prompt=True))
        if length + config["max_completion_length"] > config["context_limit"]:
            raise ValueError(f"Full profile exceeds context limit: {length} prompt tokens")
        lengths.append(length)
    return {"rows": len(rows), "min_prompt_tokens": min(lengths),
            "max_prompt_tokens": max(lengths), "mean_prompt_tokens": sum(lengths) / len(lengths),
            "prompt_truncation": False,
            "targets": dict(Counter(row["target_disorder"] for row in rows)),
            "labels": dict(Counter(row["ground_truth_answer"] for row in rows))}, lengths


def to_arrow_rows(rows):
    # JSON preserves each heterogeneous clinical profile exactly; Arrow otherwise
    # merges nested clinical field schemas and inserts extra null fields.
    return [{**{key: value for key, value in row.items() if key != "patient_record"},
             "patient_record_json": json.dumps(row["patient_record"], ensure_ascii=False,
                                                separators=(",", ":"))} for row in rows]


def wrap_reward(config):
    reward = build_reward_function(config)

    def diagnosis_gated_reward(prompts, completions, patient_record_json=None, **kwargs):
        if patient_record_json is None:
            raise ValueError("Missing exact clinical profile metadata for reward scoring")
        records = [json.loads(item) for item in patient_record_json]
        result = reward(prompts=prompts, completions=completions, patient_record=records, **kwargs)
        if len(result) != len(completions) or any(not math.isfinite(float(x)) for x in result):
            raise ValueError("Reward must return one finite scalar per completion")
        return result

    return diagnosis_gated_reward


def binary_metrics(pairs):
    tp = sum(g == "POSITIVE" and p == "POSITIVE" for g, p in pairs)
    # Keep failed generations in the denominator using the same incorrect-answer
    # policy as the final paired test; never reward malformed negative examples.
    fp = sum(g == "NEGATIVE" and p != "NEGATIVE" for g, p in pairs)
    fn = sum(g == "POSITIVE" and p != "POSITIVE" for g, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "samples": len(pairs)}


class OmniGRPOTrainer(GRPOTrainer):
    """Keep the official standard GRPO objective; project completion tokens only."""
    def __init__(self, *args, diagnostic_config, validation_rows, **kwargs):
        self.diagnostic_config = diagnostic_config
        self.validation_rows = validation_rows
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs):
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            # TRL's generation context restores checkpointing with its default
            # kwargs. Restore this run's non-reentrant choice after each rollout;
            # the official sampling, reward and GRPO objective are unchanged.
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})

    def _get_per_token_logps_and_entropies(
            self, model, input_ids, attention_mask, logits_to_keep,
            batch_size=None, compute_entropy=False, pixel_values=None,
            image_grid_thw=None, pixel_attention_mask=None, image_sizes=None):
        if any(x is not None for x in (pixel_values, image_grid_thw, pixel_attention_mask, image_sizes)):
            raise ValueError("This run consumes textual feature evidence, not raw media tensors")
        microbatch = batch_size or input_ids.size(0)
        chunk_size = self.diagnostic_config.get("projection_chunk_size", 64)
        logps_all, entropies_all = [], []
        thinker = self.accelerator.unwrap_model(model).get_base_model()
        for start in range(0, input_ids.size(0), microbatch):
            ids = input_ids[start:start + microbatch]
            mask = attention_mask[start:start + microbatch]
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 1)
            hidden = thinker.model(
                input_ids=ids, attention_mask=mask,
                position_ids=positions.unsqueeze(0).expand(3, -1, -1),
                use_cache=False, return_dict=True).last_hidden_state
            # Hidden state immediately before each completion token predicts it.
            hidden = hidden[:, -(logits_to_keep + 1):-1, :]
            targets = ids[:, -logits_to_keep:]
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            targets_flat = targets.reshape(-1)
            logps_chunks, entropy_chunks = [], []

            def project(h, token_ids):
                logits = thinker.lm_head(h).float() / self.temperature
                log_normalizer = torch.logsumexp(logits, dim=-1)
                selected = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
                logps = selected - log_normalizer
                if compute_entropy:
                    with torch.no_grad():
                        entropy = log_normalizer - (logits.softmax(-1) * logits).sum(-1)
                else:
                    entropy = torch.empty(0, device=h.device)
                return logps, entropy

            for token_start in range(0, hidden_flat.shape[0], chunk_size):
                h = hidden_flat[token_start:token_start + chunk_size]
                token_ids = targets_flat[token_start:token_start + chunk_size]
                if torch.is_grad_enabled() and h.requires_grad:
                    # Recompute vocab projection during backward rather than retain
                    # a [all completion tokens, vocabulary] activation tensor.
                    lp, ent = checkpoint(project, h, token_ids, use_reentrant=False)
                else:
                    lp, ent = project(h, token_ids)
                logps_chunks.append(lp)
                if compute_entropy:
                    entropy_chunks.append(ent)
            logps_all.append(torch.cat(logps_chunks).reshape(targets.shape))
            if compute_entropy:
                entropies_all.append(torch.cat(entropy_chunks).reshape(targets.shape))
        return torch.cat(logps_all), torch.cat(entropies_all) if compute_entropy else None

    def _save(self, output_dir=None, state_dict=None):
        destination = Path(output_dir or self.args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination, state_dict=state_dict,
                                   safe_serialization=True, save_embedding_layers=False)
        self.processing_class.save_pretrained(destination)
        torch.save(self.args, destination / "training_args.bin")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Validation is greedy binary prediction on teacher profiles. This does not
        # call the API judge or generate six training candidates for dev examples.
        started = time.time()
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        pairs, outputs = {}, []
        batch_size = self.diagnostic_config.get("validation_batch_size", 4)
        try:
            for start in range(0, len(self.validation_rows), batch_size):
                batch = self.validation_rows[start:start + batch_size]
                responses = generate_text_batch(
                    model, self.processing_class, [row["prompt"] for row in batch],
                    max_new_tokens=self.diagnostic_config["max_completion_length"],
                    context_limit=self.diagnostic_config["context_limit"], do_sample=False)
                for row, response in zip(batch, responses):
                    parsed = parse_response(response["text"])
                    valid = bool(parsed["format_valid"] and response["ended_with_eos"])
                    prediction = parsed["answer"] if valid else None
                    pairs.setdefault(row["target_disorder"], []).append(
                        (row["ground_truth_answer"], prediction))
                    outputs.append({"patient_id": row.get("patient_id"),
                                    "target_disorder": row["target_disorder"],
                                    "ground_truth_answer": row["ground_truth_answer"],
                                    "prediction": prediction, "format_valid": valid, **response})
                print(f"Dev greedy: {min(start + batch_size, len(self.validation_rows))}/{len(self.validation_rows)}", flush=True)
        finally:
            model.train(was_training)
        per_target = {key: binary_metrics(value) for key, value in pairs.items()}
        metrics = {f"{metric_key_prefix}_macro_f1": sum(x["f1"] for x in per_target.values()) / len(per_target),
                   f"{metric_key_prefix}_format_valid_rate": sum(x["format_valid"] for x in outputs) / len(outputs),
                   f"{metric_key_prefix}_runtime": time.time() - started}
        for target, values in per_target.items():
            for name in ("precision", "recall", "f1"):
                metrics[f"{metric_key_prefix}_{target.lower()}_{name}"] = values[name]
        step = self.state.global_step
        destination = Path(self.diagnostic_config["run_dir"]) / "dev_predictions" / f"step-{step}.json"
        atomic_json(destination, {"global_step": step, "metrics": metrics,
                                  "per_target": per_target, "outputs": outputs,
                                  "input_source": "teacher", "evidence_mode": "text_audio_video",
                                  "api_judge_calls": 0})
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


class RunStatus(TrainerCallback):
    def __init__(self, config):
        self.config = config
        self.recovery_save_steps = int(config.get("recovery_save_steps", 5))
        if self.recovery_save_steps < 1:
            raise ValueError("recovery_save_steps must be a positive optimizer-step interval")
        self.started = time.time()
        self.latest = {}

    def write(self, state, status):
        atomic_json(Path(self.config["run_dir"]) / "status.json", {
            "status": status, "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "global_step": state.global_step, "max_steps": state.max_steps,
            "epoch": state.epoch, "elapsed_seconds": round(time.time() - self.started, 1),
            "best_model_checkpoint": state.best_model_checkpoint,
            "best_metric": state.best_metric, "latest_metrics": self.latest,
            "output_dir": self.config["output_dir"],
            "recovery_save_steps": self.recovery_save_steps,
            "gpu_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1)})

    def on_train_begin(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.latest.update(logs or {})
        self.write(state, "training")

    def on_step_end(self, args, state, control, **kwargs):
        # This event follows an optimizer update, not an accumulation microstep.
        # Request Trainer's native full-state checkpoint so optimizer, scheduler,
        # RNG and resume position are preserved. Evaluation remains on its own
        # cadence; native checkpoint rotation also keeps the current best model.
        if state.global_step > 0 and state.global_step % self.recovery_save_steps == 0:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_train_end(self, args, state, control, **kwargs):
        self.write(state, "saving_final_adapter")


def make_training_args(config, tokenizer):
    if config.get("num_generations", 6) != 6:
        raise ValueError("This run requires exactly six candidates per prompt")
    return GRPOConfig(
        output_dir=config["output_dir"], num_train_epochs=config.get("num_train_epochs", 3),
        learning_rate=config.get("learning_rate", 5e-6),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 24),
        steps_per_generation=config.get("steps_per_generation", 6),
        num_generations=6, max_prompt_length=None,
        max_completion_length=config["max_completion_length"],
        temperature=config.get("temperature", 1.0), top_p=1.0, top_k=None,
        generation_kwargs={"eos_token_id": tokenizer.eos_token_id,
                           "pad_token_id": tokenizer.pad_token_id, "use_cache": True},
        num_iterations=1, loss_type="grpo", scale_rewards="group",
        importance_sampling_level="token", epsilon=0.2, epsilon_high=0.2,
        beta=config.get("beta", 0.04), delta=None, top_entropy_quantile=1.0,
        mask_truncated_completions=False, sync_ref_model=False,
        use_vllm=False, use_liger_loss=False, disable_dropout=True,
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=config.get("optim", "adamw_torch_fused"),
        lr_scheduler_type="cosine", warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        logging_steps=config.get("logging_steps", 1),
        eval_strategy="steps", eval_steps=config.get("eval_steps", 25),
        # Native GRPO's eval batch divisibility check still applies. The overridden
        # greedy evaluation uses validation_batch_size instead of this field.
        per_device_eval_batch_size=6,
        save_strategy="steps", save_steps=config.get("save_steps", 25),
        save_total_limit=config.get("save_total_limit", 5),
        load_best_model_at_end=True, metric_for_best_model="eval_macro_f1",
        greater_is_better=True, remove_unused_columns=False, report_to="none",
        log_completions=False, seed=config.get("seed", 42), data_seed=config.get("seed", 42),
        dataloader_num_workers=0, dataloader_pin_memory=True, disable_tqdm=True)


def preflight(trainer, rows, lengths, config):
    """Disposable full-size longest-profile rollout/backward/Adam smoke test."""
    generation_batch_size = trainer.args.generation_batch_size
    prompt_count = generation_batch_size // 6
    indices = sorted(range(len(rows)), key=lengths.__getitem__, reverse=True)[:prompt_count]
    if len(indices) != prompt_count:
        raise ValueError("Not enough rows to preflight the configured generation batch")
    samples = to_arrow_rows([rows[index] for index in indices])
    inputs = [dict(sample) for sample in samples for _ in range(6)]
    trainer.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer.model.enable_input_require_grads()
    trainer.model.train()
    trainer.current_gradient_accumulation_steps = math.ceil(
        generation_batch_size / trainer.args.per_device_train_batch_size)
    prepared = trainer._generate_and_score_completions(inputs)
    real_advantages = prepared["advantages"].detach().clone()
    # Equal real rewards legitimately produce zero advantages. A deterministic,
    # centered probe verifies nonzero adapter gradients in that case only. It is
    # never used by production training and no preflight weights are saved.
    gradient_probe = not bool(real_advantages.abs().sum().item())
    if gradient_probe:
        prepared["advantages"] = torch.tensor(
            [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5] * prompt_count,
            device=real_advantages.device)
    trainer.create_optimizer()
    loss_sum = 0.0
    # Backward microbatch matches production training; detached rollout/reference
    # tensors retain all six completions per group through the accumulated update.
    for start in range(0, generation_batch_size, trainer.args.per_device_train_batch_size):
        end = start + trainer.args.per_device_train_batch_size
        batch = {key: (value[start:end] if isinstance(value, torch.Tensor)
                       and value.ndim > 0 and value.shape[0] == generation_batch_size else value)
                 for key, value in prepared.items()}
        loss = trainer.compute_loss(trainer.model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("GRPO preflight loss is nonfinite")
        loss.backward()
        loss_sum += loss.item()
    grad_sum = sum(p.grad.float().abs().sum().item() for p in trainer.model.parameters()
                   if p.requires_grad and p.grad is not None)
    if not grad_sum > 0:
        raise RuntimeError("GRPO longest-profile backward failed")
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.get("max_grad_norm", 1.0))
    trainer.optimizer.step()
    optimizer_bytes = sum(value.numel() * value.element_size()
                          for state in trainer.optimizer.state.values() for value in state.values()
                          if isinstance(value, torch.Tensor))
    report = {"status": "passed", "sample_indices": indices,
              "full_prompt_tokens": [lengths[index] for index in indices],
              "num_generations": 6, "generation_batch_size": generation_batch_size,
              "backward_microbatch_size": trainer.args.per_device_train_batch_size,
              "completion_tokens": prepared["completion_mask"].sum(-1).tolist(),
              "real_group_advantages": real_advantages.tolist(),
              "used_disposable_gradient_probe": gradient_probe,
              "loss": loss_sum, "adapter_gradient_abs_sum": grad_sum,
              "optimizer_state_mib": round(optimizer_bytes / 1024**2, 1),
              "gpu_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
              "weights_saved": False, "production_training_started": False}
    atomic_json(Path(config["run_dir"]) / "gpu_preflight.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    options = parser.parse_args()
    if trl.__version__ != "0.23.1":
        raise RuntimeError(f"The reviewed official trainer version is TRL0.23.1; got {trl.__version__}")
    config = load_config(options.config)
    config.setdefault("num_generations", 6)
    config.setdefault("recovery_save_steps", 5)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.get("seed", 42))
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = load_tokenizer(config["base_model"])
    train_path = Path(config["data_root"]) / "train.jsonl"
    dev_path = Path(config["data_root"]) / "dev.jsonl"
    train_rows, dev_rows = read_jsonl(train_path), read_jsonl(dev_path)
    train_audit, train_lengths = token_audit(train_rows, tokenizer, config)
    dev_audit, _ = token_audit(dev_rows, tokenizer, config)
    for audit, path in ((train_audit, train_path), (dev_audit, dev_path)):
        audit["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    train_patients = {(r["dataset"], r["patient_id"]) for r in train_rows}
    dev_patients = {(r["dataset"], r["patient_id"]) for r in dev_rows}
    if train_patients & dev_patients:
        raise ValueError("Patient leakage between training and validation")
    atomic_json(run_dir / "training_data_audit.json", {"train": train_audit, "dev": dev_audit,
                "eos_token_id": tokenizer.eos_token_id, "teacher_profiles_only": True,
                "patient_overlap": 0})
    if options.audit_only:
        print(json.dumps({"train": train_audit, "dev": dev_audit}), flush=True)
        return
    model = load_text_thinker(config["base_model"], config.get("cpu_offload_embedding", False))
    removed = model._text_only_removed_encoder_bytes
    model, setup = add_decoder_lora(model, config)
    model.config.use_cache = False
    # Decoder-only log-probability evaluation bypasses the PEFT forward wrapper.
    # Frozen embedding outputs must still enter the checkpointed decoder with a
    # gradient graph so adapter parameters receive gradients in production too.
    model.enable_input_require_grads()
    model.to("cuda")
    training_args = make_training_args(config, tokenizer)
    reward_func = wrap_reward(config)
    trainer = OmniGRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=reward_func,
        args=training_args, train_dataset=Dataset.from_list(to_arrow_rows(train_rows)),
        eval_dataset=Dataset.from_list(to_arrow_rows(dev_rows)),
        diagnostic_config=config, validation_rows=dev_rows,
        callbacks=[RunStatus(config)])
    atomic_json(run_dir / "model_setup.json", {**setup,
                "removed_unused_encoder_bytes": removed,
                "precision": "bfloat16", "attention": "sdpa", "quantization": None,
                "dropout_disabled_for_grpo": True, "reference": "base_model_adapter_disabled",
                "input_embedding_outputs_require_grad": True,
                "gradient_checkpointing_use_reentrant": False,
                "implementation": "trl.GRPOTrainer", "trl_version": trl.__version__})
    effective = {**config, "official_grpo_arguments": training_args.to_dict(),
                 "prompt_truncation": False, "eos_token_id": tokenizer.eos_token_id,
                 "resume_from_checkpoint": options.resume}
    atomic_json(run_dir / "effective_config.json", effective)
    if options.preflight:
        preflight(trainer, train_rows, train_lengths, config)
        return
    resume = options.resume
    if resume == "auto":
        resume = get_last_checkpoint(config["output_dir"])
        if resume is None:
            raise ValueError("No checkpoint is available to resume")
    try:
        result = trainer.train(resume_from_checkpoint=resume)
        final = str(Path(config["output_dir"]) / "final_adapter")
        trainer.save_model(final)
        trainer.save_state()
        dev_metrics = trainer.evaluate()
        report = {"status": "complete", "global_step": trainer.state.global_step,
                  "train_metrics": result.metrics, "dev_metrics": dev_metrics,
                  "best_model_checkpoint": trainer.state.best_model_checkpoint,
                  "best_metric": trainer.state.best_metric, "final_adapter": final}
        atomic_json(run_dir / "training_result.json", report)
        atomic_json(run_dir / "status.json", {**json.loads((run_dir / "status.json").read_text()),
                    **report})
        print(f"Diagnosis GRPO complete: {final}", flush=True)
    except Exception as error:
        status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
        atomic_json(run_dir / "status.json", {**status, "status": "failed",
                    "exception_type": type(error).__name__, "error": str(error)[:1000]})
        raise


if __name__ == "__main__":
    main()
