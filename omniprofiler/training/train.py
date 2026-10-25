"""Text-only, full-transcript LoRA SFT of the Qwen2.5-Omni Thinker."""
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
import os
from pathlib import Path
import re
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)
from data import (AssistantCollator, TokenizedDataset, atomic_json,
                      length_stats, read_jsonl, selected_causal_loss, sha256_file)
from model import load_text_thinker


LORA_TARGETS = (r"model\.layers\.\d+\."
                r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")


class TargetOnlyTrainer(Trainer):
    def __init__(self, *args, loss_chunk_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss uses a mean CE and does not consume num_items_in_batch.
        # Trainer must therefore apply its ordinary accumulation scaling.
        self.model_accepts_loss_kwargs = False
        self.loss_chunk_size = loss_chunk_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        thinker = model.get_base_model()
        mask = inputs["attention_mask"]
        positions = mask.long().cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = thinker.model(input_ids=inputs["input_ids"], attention_mask=mask,
                                   position_ids=positions.unsqueeze(0).expand(3, -1, -1),
                                   use_cache=False, return_dict=True)
            loss = selected_causal_loss(output.last_hidden_state, thinker.lm_head, inputs["labels"],
                                       chunk_size=self.loss_chunk_size)
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
                   "epoch": state.epoch, "elapsed_seconds": round(time.time()-self.started, 1),
                   "best_model_checkpoint": state.best_model_checkpoint,
                   "latest_metrics": self.latest,
                   "output_dir": self.config["output_dir"],
                   "gpu_max_allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1)})

    def on_train_begin(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.latest.update(logs or {})
        self.write(state, "training")

    def on_save(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_train_end(self, args, state, control, **kwargs):
        self.write(state, "saving_final_adapter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    print("Loading the BF16 Thinker and decoder LoRA", flush=True)
    model = load_text_thinker(config["base_model"], config.get("cpu_offload_embedding", False))
    removed_encoder_bytes = model._text_only_removed_encoder_bytes
    model.config.use_cache = False
    decoder_targets = [module for name, module in model.named_modules()
                       if re.fullmatch(LORA_TARGETS, name)]
    if len(decoder_targets) != 196:
        raise RuntimeError(f"Unexpected decoder target count: {len(decoder_targets)}")
    expected_trainable_count = config["lora_rank"] * sum(
        module.in_features + module.out_features for module in decoder_targets)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=LORA_TARGETS, bias="none"))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(".model.layers." not in name or ".lora_" not in name for name in trainable):
        raise RuntimeError("Unexpected non-decoder trainable parameters")
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_count != expected_trainable_count:
        raise RuntimeError(f"Unexpected LoRA parameter count: {trainable_count}")
    model.to("cuda")
    atomic_json(run_dir / "model_setup.json", {"trainable_parameters": trainable_count,
        "trainable_tensors": len(trainable), "decoder_lora_modules": len(trainable)//2,
        "lora_rank": config["lora_rank"], "lora_alpha": config["lora_alpha"],
        "removed_unused_encoder_bytes": removed_encoder_bytes,
        "input_embedding_device": str(model.get_input_embeddings().weight.device),
        "trainable_parameter_names": trainable, "quantization": None,
        "precision": "bfloat16", "attention": "sdpa", "lora_target_regex": LORA_TARGETS})
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_path, dev_path = [Path(config["data_root"]) / name for name in ("train.jsonl", "dev.jsonl")]
    train_data = TokenizedDataset(read_jsonl(train_path), tokenizer, config["context_limit"])
    dev_data = TokenizedDataset(read_jsonl(dev_path), tokenizer, config["context_limit"])
    atomic_json(run_dir / "training_data_audit.json", {
        "train": {**length_stats(train_data.items), "sha256": sha256_file(train_path)},
        "dev": {**length_stats(dev_data.items), "sha256": sha256_file(dev_path)}})
    output_dir = config["output_dir"]
    if args.smoke_test or config.get("startup_smoke_test", False):
        # Test the longest full-transcript batch, including backward and Adam memory.
        model.to("cuda")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
        indices = sorted(range(len(train_data)),
                         key=lambda i: len(train_data[i]["input_ids"]), reverse=True
                         )[:config["per_device_train_batch_size"]]
        batch = {key: value.to("cuda") for key, value in
                 AssistantCollator(tokenizer.pad_token_id)([train_data[i] for i in indices]).items()}
        thinker = model.get_base_model()
        pos = batch["attention_mask"].long().cumsum(-1)-1
        pos.masked_fill_(batch["attention_mask"] == 0, 1)
        # Reserve the two FP32 Adam moments to include optimizer memory in preflight.
        moment_reserve = [torch.empty_like(p) for p in model.parameters()
                          if p.requires_grad for _ in range(2)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = thinker.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                position_ids=pos.unsqueeze(0).expand(3, -1, -1), use_cache=False)
            loss = selected_causal_loss(out.last_hidden_state, thinker.lm_head, batch["labels"],
                                       chunk_size=config.get("projection_chunk_size", 128))
        loss.backward()
        grad_sum = sum(p.grad.float().abs().sum().item() for p in model.parameters()
                       if p.requires_grad and p.grad is not None)
        if not torch.isfinite(loss) or not grad_sum > 0:
            raise RuntimeError("The full-transcript backward smoke test failed")
        report = {"status": "passed", "sample_indices": indices,
                  "batch_size": len(indices), "padded_sequence_tokens": batch["input_ids"].shape[1],
                  "total_padded_tokens": batch["input_ids"].numel(),
                  "full_tokens_per_sample": [len(train_data[i]["input_ids"]) for i in indices],
                  "loss": loss.item(), "adapter_gradient_abs_sum": grad_sum,
                  "adam_moment_reserve_mib": round(sum(p.numel()*p.element_size() for p in moment_reserve)/1024**2, 1),
                  "gpu_max_allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1)}
        atomic_json(run_dir / "gpu_smoke_test.json", report)
        print(json.dumps(report), flush=True)
        if args.smoke_test:
            return
        model.zero_grad(set_to_none=True)
        del batch, out, loss, pos, moment_reserve
        torch.cuda.empty_cache()
        set_seed(config["seed"])
    training_args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine", warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"], max_grad_norm=config["max_grad_norm"],
        optim=config.get("optim", "adamw_torch"), logging_steps=config["logging_steps"],
        eval_strategy="steps", eval_steps=config["eval_steps"],
        save_strategy="steps", save_steps=config["save_steps"], save_total_limit=config["save_total_limit"],
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, remove_unused_columns=False, report_to="none",
        seed=config["seed"], data_seed=config["seed"], dataloader_num_workers=0,
        dataloader_pin_memory=True, disable_tqdm=True,
        group_by_length=config.get("group_by_length", False),
        torch_empty_cache_steps=config.get("torch_empty_cache_steps"))
    trainer = TargetOnlyTrainer(model=model, args=training_args, train_dataset=train_data,
                               eval_dataset=dev_data, processing_class=tokenizer,
                               data_collator=AssistantCollator(tokenizer.pad_token_id),
                               loss_chunk_size=config.get("projection_chunk_size", 128),
                               callbacks=[RunStatus(run_dir / "status.json", config)])
    resume = args.resume
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(output_dir)
        if resume is None:
            raise ValueError("No checkpoint is available to resume")
    atomic_json(run_dir / "effective_config.json", {**config, "resume_from_checkpoint": resume})
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(Path(output_dir) / "final_adapter"))
    trainer.save_state()
    dev_metrics = trainer.evaluate()
    atomic_json(run_dir / "training_result.json", {"status": "complete", "train_metrics": result.metrics,
                "dev_metrics": dev_metrics, "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "final_adapter": str(Path(output_dir) / "final_adapter"),
                "global_step": trainer.state.global_step})
    status = json.loads((run_dir / "status.json").read_text())
    atomic_json(run_dir / "status.json", {**status, "status": "complete",
                "final_adapter": str(Path(output_dir) / "final_adapter"), "dev_metrics": dev_metrics})
    print("Training complete: " + str(Path(output_dir) / "final_adapter"), flush=True)


if __name__ == "__main__":
    main()
