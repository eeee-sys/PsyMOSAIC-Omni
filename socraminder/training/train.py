"""Stage 1 teacher-guided, assistant-only LoRA SFT of the original Omni Thinker."""
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
import json
import math
import os
from pathlib import Path
import re
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from project_paths import load_config

try:
    from .model import load_text_thinker
    from .data import (AssistantCollator, TokenizedDataset, atomic_json, audit_split_isolation,
                           length_stats, read_jsonl, selected_causal_loss, sha256_file)
except ImportError:
    from model import load_text_thinker
    from data import (AssistantCollator, TokenizedDataset, atomic_json, audit_split_isolation,
                          length_stats, read_jsonl, selected_causal_loss, sha256_file)


LORA_TARGETS = (r"model\.layers\.\d+\."
                r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def text_forward_loss(model, inputs, chunk_size=128):
    thinker = model.get_base_model()
    mask = inputs["attention_mask"]
    positions = mask.long().cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 1)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = thinker.model(input_ids=inputs["input_ids"], attention_mask=mask,
                               position_ids=positions.unsqueeze(0).expand(3, -1, -1),
                               use_cache=False, return_dict=True)
        loss = selected_causal_loss(output.last_hidden_state, thinker.lm_head,
                                    inputs["labels"], chunk_size=chunk_size)
    return loss


class TargetOnlyTrainer(Trainer):
    def __init__(self, *args, loss_chunk_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        # Loss is already a per-batch mean; Trainer applies normal accumulation scaling.
        self.model_accepts_loss_kwargs = False
        self.loss_chunk_size = loss_chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = text_forward_loss(model, inputs, self.loss_chunk_size)
        return (loss, {"loss": loss}) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir, state_dict=state_dict,
                                   safe_serialization=True, save_embedding_layers=False)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, Path(output_dir) / "training_args.bin")


class RunStatus(TrainerCallback):
    def __init__(self, path, config):
        self.path, self.config = Path(path), config
        self.started = time.time()
        self.latest = {}

    def write(self, state, status):
        atomic_json(self.path, {"status": status, "pid": os.getpid(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "global_step": state.global_step, "max_steps": state.max_steps,
                    "epoch": state.epoch, "elapsed_seconds": round(time.time() - self.started, 1),
                    "best_model_checkpoint": state.best_model_checkpoint,
                    "latest_metrics": self.latest, "output_dir": self.config["output_dir"],
                    "gpu_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1)})

    def on_train_begin(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.latest.update(logs or {})
        self.write(state, "training")

    def on_save(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_train_end(self, args, state, control, **kwargs):
        self.write(state, "saving_final_adapter")


class FirstOptimizerUpdateCheck(TrainerCallback):
    """Verify gradients immediately and weights at the first positive-LR update."""
    def __init__(self, report_path, max_zero_lr_steps=8):
        self.report_path = Path(report_path)
        self.before = None
        self.checked = False
        self.zero_lr_optimizer_steps = []
        self.max_zero_lr_steps = max_zero_lr_steps
        self.gradient_check_step = None
        self.pending_zero_lr = False

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self.checked:
            return
        if optimizer is None:
            raise RuntimeError("Optimizer update verification requires optimizer parameter groups")
        learning_rates = {id(parameter): float(group["lr"])
                          for group in optimizer.param_groups for parameter in group["params"]}
        if any(not math.isfinite(value) or value < 0 for value in learning_rates.values()):
            raise RuntimeError("Invalid optimizer learning rate")
        candidates = []
        gradient_abs_sum = 0.0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grad = parameter.grad.detach()
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"Non-finite gradient in {name}")
                magnitude = grad.float().abs().sum().item()
                gradient_abs_sum += magnitude
                if magnitude > 0:
                    candidates.append((magnitude, name, parameter, learning_rates.get(id(parameter), 0.0)))
        if not candidates or not math.isfinite(gradient_abs_sum) or gradient_abs_sum <= 0:
            raise RuntimeError("The optimizer step has no nonzero adapter gradient")
        self.gradient_abs_sum = gradient_abs_sum
        if self.gradient_check_step is None:
            self.gradient_check_step = state.global_step + 1
        positive_lr_candidates = [item for item in candidates if item[3] > 0]
        self.pending_zero_lr = not positive_lr_candidates
        if self.pending_zero_lr:
            if any(rate > 0 for rate in learning_rates.values()):
                raise RuntimeError("Positive-LR optimizer groups have no nonzero adapter gradients")
            self.before = None
            return
        _, self.name, self.parameter, self.learning_rate = max(positive_lr_candidates, key=lambda item: item[0])
        self.before = self.parameter.detach().float().cpu().clone()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if self.checked:
            return
        if self.pending_zero_lr:
            self.zero_lr_optimizer_steps.append(state.global_step + 1)
            report = {"status": "waiting_for_positive_learning_rate",
                      "gradient_check_step": self.gradient_check_step,
                      "gradient_abs_sum": self.gradient_abs_sum,
                      "zero_lr_optimizer_steps": self.zero_lr_optimizer_steps,
                      "gradient_accumulation_steps": args.gradient_accumulation_steps}
            atomic_json(self.report_path, report)
            print("Finite nonzero gradients verified; zero-LR warmup step has no weight update: "
                  + json.dumps(report), flush=True)
            if len(self.zero_lr_optimizer_steps) > self.max_zero_lr_steps:
                raise RuntimeError("No positive-LR optimizer update within the bounded startup check")
            return
        if self.before is None:
            raise RuntimeError("Optimizer update callback did not observe gradients")
        delta = (self.parameter.detach().float().cpu() - self.before).abs()
        delta_sum = delta.sum().item()
        if not math.isfinite(delta_sum) or delta_sum <= 0:
            raise RuntimeError("The first positive-LR optimizer step did not change the adapter weights")
        report = {"status": "passed", "optimizer_step": state.global_step + 1,
                  "gradient_check_step": self.gradient_check_step,
                  "zero_lr_optimizer_steps": self.zero_lr_optimizer_steps,
                  "learning_rate_at_verified_update": self.learning_rate,
                  "parameter": self.name, "gradient_abs_sum": self.gradient_abs_sum,
                  "weight_delta_abs_sum": delta_sum, "weight_delta_abs_max": delta.max().item(),
                  "gradient_accumulation_steps": args.gradient_accumulation_steps}
        atomic_json(self.report_path, report)
        print("First optimizer update verified: " + json.dumps(report), flush=True)
        self.checked = True
        self.before = self.parameter = None

    def on_train_end(self, args, state, control, **kwargs):
        if not self.checked:
            raise RuntimeError("Training ended before a positive-LR adapter weight update was verified")


def audit_data(config, tokenizer, run_dir):
    data_root = Path(config["data_root"])
    train_path, dev_path = data_root / "train.jsonl", data_root / "dev.jsonl"
    train_records, dev_records = read_jsonl(train_path), read_jsonl(dev_path)
    split_audit = audit_split_isolation(
        train_records, dev_records, require_patient_ids=config.get("require_patient_ids", True),
        expected_train_patients=config.get("expected_train_patients"),
        expected_dev_patients=config.get("expected_dev_patients"))
    train_data = TokenizedDataset(train_records, tokenizer, config["context_limit"])
    dev_data = TokenizedDataset(dev_records, tokenizer, config["context_limit"])
    report = {"train": {**split_audit["train"], **length_stats(train_data.items),
                         "sha256": sha256_file(train_path)},
              "dev": {**split_audit["dev"], **length_stats(dev_data.items),
                       "sha256": sha256_file(dev_path)},
              "test_loaded": False, "assistant_only_loss": True,
              "context_limit": config["context_limit"], "target_blocks": ["state_update", "response", "stage_status"]}
    atomic_json(run_dir / "training_data_audit.json", report)
    print("Complete-context data audit: " + json.dumps(report), flush=True)
    return train_data, dev_data


def build_lora(config, run_dir):
    print("Loading the original BF16 Qwen2.5-Omni Thinker (no prior adapter)", flush=True)
    model = load_text_thinker(config["base_model"], config.get("cpu_offload_embedding", False))
    removed_encoder_bytes = model._text_only_removed_encoder_bytes
    decoder_targets = [module for name, module in model.named_modules() if re.fullmatch(LORA_TARGETS, name)]
    if len(decoder_targets) != 196:
        raise RuntimeError(f"Unexpected decoder target count: {len(decoder_targets)}")
    expected_count = config["lora_rank"] * sum(module.in_features + module.out_features for module in decoder_targets)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config.get("lora_dropout", .05),
        target_modules=LORA_TARGETS, bias="none"))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".model.layers." not in name or ".lora_" not in name for name in trainable):
        raise RuntimeError("Unexpected non-decoder trainable parameters")
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_count != expected_count:
        raise RuntimeError(f"Unexpected LoRA parameter count: {trainable_count} != {expected_count}")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to("cuda")
    atomic_json(run_dir / "model_setup.json", {"base_model": config["base_model"],
        "fresh_base_without_existing_adapter": True, "trainable_parameters": trainable_count,
        "trainable_tensors": len(trainable), "decoder_lora_modules": len(decoder_targets),
        "lora_rank": config["lora_rank"], "lora_alpha": config["lora_alpha"],
        "removed_unused_encoder_bytes": removed_encoder_bytes,
        "input_embedding_device": str(model.get_input_embeddings().weight.device),
        "trainable_parameter_names": trainable, "quantization": None,
        "precision": "bfloat16", "attention": "sdpa", "lora_target_regex": LORA_TARGETS})
    return model


def gpu_smoke_test(model, train_data, tokenizer, config, run_dir):
    # Restoring RNG preserves the main run's post-initialization random state.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model.train()
        indices = sorted(range(len(train_data)), key=lambda index: len(train_data[index]["input_ids"]),
                         reverse=True)[:config["per_device_train_batch_size"]]
        batch = {key: value.to("cuda") for key, value in AssistantCollator(tokenizer.pad_token_id)(
                 [train_data[index] for index in indices]).items()}
        moment_reserve = [torch.empty(parameter.shape, dtype=torch.float32, device=parameter.device)
                          for parameter in model.parameters() if parameter.requires_grad for _ in range(2)]
        loss = text_forward_loss(model, batch, config.get("projection_chunk_size", 128))
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()
                     if parameter.requires_grad and parameter.grad is not None]
        if any(not torch.isfinite(grad).all() for grad in gradients):
            raise RuntimeError("Non-finite gradient in the complete-context backward smoke test")
        grad_sum = sum(grad.float().abs().sum().item() for grad in gradients)
        if not torch.isfinite(loss) or not grad_sum > 0:
            raise RuntimeError("The complete-context backward smoke test failed")
        report = {"status": "passed", "sample_indices": indices, "batch_size": len(indices),
                  "padded_sequence_tokens": batch["input_ids"].shape[1],
                  "total_padded_tokens": batch["input_ids"].numel(),
                  "full_tokens_per_sample": [len(train_data[index]["input_ids"]) for index in indices],
                  "loss": loss.item(), "adapter_gradient_abs_sum": grad_sum,
                  "optimizer_steps_performed": 0, "main_run_rng_preserved": True,
                  "adam_moment_reserve_mib": round(sum(p.numel() * p.element_size() for p in moment_reserve) / 1024**2, 1),
                  "gpu_max_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1)}
        atomic_json(run_dir / "gpu_smoke_test.json", report)
        print("Complete-context GPU backward smoke test: " + json.dumps(report), flush=True)
        model.zero_grad(set_to_none=True)
        del batch, loss, gradients, moment_reserve
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/train.json")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(config["run_dir"]) / "smoke" if args.smoke_test else Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.resume and args.smoke_test:
        raise ValueError("Smoke tests always use a fresh adapter and cannot resume training")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_data, dev_data = audit_data(config, tokenizer, run_dir)
    if args.audit_only:
        return
    model = build_lora(config, run_dir)
    if args.smoke_test or (config.get("startup_smoke_test", True) and not args.resume):
        gpu_smoke_test(model, train_data, tokenizer, config, run_dir)
        if args.smoke_test:
            return
    output_dir = config["output_dir"]
    eval_strategy = config.get("eval_strategy", "epoch")
    save_strategy = config.get("save_strategy", "epoch")
    if eval_strategy != save_strategy:
        raise ValueError("Best-dev checkpoint selection requires matching eval/save strategies")
    options = {}
    if eval_strategy == "steps":
        options.update(eval_steps=config["eval_steps"], save_steps=config["save_steps"])
    elif eval_strategy != "epoch":
        raise ValueError("Evaluation/save strategy must be epoch or steps")
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine", warmup_ratio=config.get("warmup_ratio", .03),
        weight_decay=config.get("weight_decay", .01), max_grad_norm=config.get("max_grad_norm", 1.),
        optim=config.get("optim", "adamw_torch_fused"), logging_steps=config.get("logging_steps", 1),
        eval_strategy=eval_strategy, save_strategy=save_strategy,
        save_total_limit=config.get("save_total_limit", 5),
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, remove_unused_columns=False, report_to="none",
        seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0,
        dataloader_pin_memory=True, disable_tqdm=True,
        group_by_length=config.get("group_by_length", True),
        torch_empty_cache_steps=config.get("torch_empty_cache_steps"), **options)
    trainer = TargetOnlyTrainer(model=model, args=training_args, train_dataset=train_data,
        eval_dataset=dev_data, processing_class=tokenizer,
        data_collator=AssistantCollator(tokenizer.pad_token_id),
        loss_chunk_size=config.get("projection_chunk_size", 128),
        callbacks=[RunStatus(run_dir / "status.json", config),
                   FirstOptimizerUpdateCheck(run_dir / "first_optimizer_update.json")])
    resume = args.resume
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
        if resume is None:
            raise ValueError("No checkpoint is available to resume")
    if not resume and Path(output_dir).is_dir() and any(Path(output_dir).glob("checkpoint-*")):
        raise ValueError("Output already contains checkpoints; use --resume auto or a new output directory")
    atomic_json(run_dir / "effective_config.json", {**config, "resume_from_checkpoint": resume})
    result = trainer.train(resume_from_checkpoint=resume)
    update_report = run_dir / "first_optimizer_update.json"
    if not update_report.is_file() or json.loads(update_report.read_text()).get("status") != "passed":
        raise RuntimeError("Training did not produce a verified optimizer update")
    trainer.save_model(str(Path(output_dir) / "final_adapter"))
    trainer.save_state()
    # The final adapter is the best dev-loss checkpoint restored by Trainer.
    dev_metrics = trainer.evaluate()
    atomic_json(run_dir / "training_result.json", {"status": "complete", "train_metrics": result.metrics,
                "dev_metrics": dev_metrics, "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "final_adapter": str(Path(output_dir) / "final_adapter"),
                "global_step": trainer.state.global_step, "test_evaluated": False})
    status = json.loads((run_dir / "status.json").read_text())
    atomic_json(run_dir / "status.json", {**status, "status": "complete",
                "final_adapter": str(Path(output_dir) / "final_adapter"), "dev_metrics": dev_metrics})
    print("Training complete: " + str(Path(output_dir) / "final_adapter"), flush=True)


if __name__ == "__main__":
    main()
