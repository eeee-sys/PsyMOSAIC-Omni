"""Text-only Omni, bounded CPU audits and fully resumable adapter checkpoints.

No code in this module discovers/selects a GPU. Model construction is lazy so
--audit-only can validate all 30 training cases without allocating CUDA memory.
"""
from __future__ import annotations

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
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time

from common import ROOT, load_json, read_jsonl, score_completion, write_json

LORA_TARGETS = (r"model\.layers\.\d+\."
                r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "config/train.json"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def setup_device(gpu):
    if gpu is None or gpu < 0:
        raise ValueError("An explicitly reviewed --gpu INDEX is required for training")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The explicitly selected CUDA device is unavailable")
    torch.cuda.set_device(0)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(config):
    data = Path(config["data_dir"]) / "train.jsonl"
    prompt = Path(config.get("student_system_prompt", ROOT / "prompts/student_system.txt"))
    assets = {"config": config, "train_sha256": sha256(data),
              "student_system_sha256": sha256(prompt)}
    # Protect the actual runtime as well as input data/configuration on resume.
    assets["runtime_sha256"] = {name: sha256(ROOT / name) for name in
                               ("common.py", "training_utils.py", "train_sft.py", "train_grpo.py")}
    return hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()


def append_event(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": time.time(), **value}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def tokenizer_for(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id != 151645:
        raise ValueError("Unexpected Omni ChatML EOS token")
    return tokenizer


def training_rows(config, tokenizer):
    """Read only train.jsonl; test records are never opened by either trainer."""
    path = Path(config["data_dir"]) / "train.jsonl"
    rows = read_jsonl(path)
    expected = {f"Case{i}" for i in range(1, 31)}
    actual = [row.get("case_id") for row in rows]
    if len(rows) != 30 or len(set(actual)) != 30 or set(actual) != expected:
        raise ValueError("Training must contain exactly Case1–Case30, once each")
    system = Path(config.get("student_system_prompt", ROOT / "prompts/student_system.txt")).read_text()
    audited, lengths = [], []
    completion_budget = int(config.get("grpo", {}).get("max_completion_tokens", 768))
    for row in rows:
        if row.get("split") != "train":
            raise ValueError(f"Non-training row: {row['case_id']}")
        source = row["model_input"]
        if set(source) != {"patient_historical_record", "current_situation", "candidate_medications"}:
            raise ValueError("Student input must contain exactly the three authorized fields")
        candidates = source["candidate_medications"]
        if len(candidates) < 30 or len(candidates) != len(set(candidates)):
            raise ValueError("At least 30 distinct canonical candidate medications are required")
        if any(not isinstance(x, str) or not x.strip() or x != x.lower().strip() for x in candidates):
            raise ValueError("Candidates must be lowercase canonical generic medication names")
        target = row["target"]
        if set(target) != {"recommendations"} or not target["recommendations"]:
            raise ValueError("Target must contain only a nonempty recommendations array")
        target_text = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
        reference = row["reference_medications"]
        if reference != [item["medication"] for item in target["recommendations"]]:
            raise ValueError("Reference ranking must match the supervised target exactly")
        reward = score_completion(target_text, reference, candidates)
        if not reward["valid"] or abs(reward["reward"] - 1.0) > 1e-8:
            raise ValueError(f"Invalid self-consistent teacher target: {row['case_id']}: {reward}")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(source, ensure_ascii=False, separators=(",", ":"))}]
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full_ids = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": target_text}],
                                                 tokenize=True, add_generation_prompt=False)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template prefix mismatch; cannot mask assistant-only loss")
        limit = int(config.get("context_limit", 16384))
        if max(len(full_ids), len(prompt_ids) + completion_budget) > limit:
            raise ValueError(f"Complete {row['case_id']} exceeds context_limit={limit}; no truncation is allowed")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        audited.append({**row, "messages": messages, "prompt_ids": prompt_ids,
                        "input_ids": full_ids, "labels": labels})
        lengths.append({"case_id": row["case_id"], "prompt_tokens": len(prompt_ids),
                        "target_tokens": len(full_ids) - len(prompt_ids), "total_tokens": len(full_ids)})
    report = {"status": "passed", "rows": len(rows), "case_ids": actual,
              "test_loaded": False, "test_used_for_selection": False, "prompt_truncation": False,
              "assistant_only_sft_loss": True, "train_sha256": sha256(path), "lengths": lengths,
              "max_total_tokens": max(x["total_tokens"] for x in lengths),
              "max_prompt_tokens": max(x["prompt_tokens"] for x in lengths)}
    return audited, report


@contextmanager
def last_token_generation(model):
    """Avoid projecting the complete prompt to the 152k vocabulary during generation."""
    thinker = model.get_base_model() if hasattr(model, "get_base_model") else model
    def hook(module, inputs):
        import torch
        if not torch.is_grad_enabled() and len(inputs) == 1 and inputs[0].ndim == 3:
            return (inputs[0][:, -1:, :],)
        return None
    handle = thinker.lm_head.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def load_model(config, adapter_path=None, reference_path=None):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        config["base_model"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True)
    for name in ("audio_tower", "visual"):
        if hasattr(model, name):
            setattr(model, name, None)
    targets = [name for name, _ in model.named_modules() if re.fullmatch(LORA_TARGETS, name)]
    if len(targets) != 196:
        raise ValueError(f"Expected 196 language decoder projections, found {len(targets)}")
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name="default", is_trainable=True)
    else:
        lora = config.get("lora", {})
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                              r=int(lora.get("rank", 32)), lora_alpha=int(lora.get("alpha", 64)),
                              lora_dropout=float(lora.get("dropout", .05)),
                              target_modules=LORA_TARGETS, bias="none"))
    if reference_path:
        model.load_adapter(str(reference_path), adapter_name="frozen_sft_reference", is_trainable=False)
        # RL likelihoods and rollout policy must be deterministic w.r.t. dropout.
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    model.to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    activate_adapter(model, "default")
    return model


def activate_adapter(model, name):
    model.set_adapter(name)
    # PEFT set_adapter can toggle requires_grad: enforce reference freezing after
    # every switch. Only the policy adapter is ever passed to the optimizer.
    for pname, parameter in model.named_parameters():
        parameter.requires_grad_(name == "default" and ".lora_" in pname and ".default." in pname)


def trainable_report(model):
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names or any(".model.layers." not in n or ".lora_" not in n or ".default." not in n for n in names):
        raise RuntimeError("Non-decoder or reference parameter was marked trainable")
    return {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trainable_tensors": len(names), "trainable_parameter_names": names,
            "decoder_lora_modules": 196, "precision": "bfloat16", "attention": "sdpa",
            "quantization": None, "base_model_frozen": True}


def tensor_adapter_hash(model, name):
    digest = hashlib.sha256()
    for pname, parameter in model.named_parameters():
        if ".lora_" in pname and f".{name}." in pname:
            digest.update(pname.encode())
            digest.update(parameter.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def collate_sft(items, pad_id):
    import torch
    length = max(len(x["input_ids"]) for x in items)
    ids, labels, masks = [], [], []
    for item in items:
        padding = length - len(item["input_ids"])
        ids.append([pad_id] * padding + item["input_ids"])
        labels.append([-100] * padding + item["labels"])
        masks.append([0] * padding + [1] * len(item["input_ids"]))
    return {"input_ids": torch.tensor(ids, device="cuda"), "attention_mask": torch.tensor(masks, device="cuda"),
            "labels": torch.tensor(labels, device="cuda")}


def decoder_hidden(model, ids, attention_mask):
    thinker = model.get_base_model()
    positions = attention_mask.long().cumsum(-1) - 1
    positions.masked_fill_(attention_mask == 0, 1)
    return thinker.model(input_ids=ids, attention_mask=attention_mask,
                         position_ids=positions.unsqueeze(0).expand(3, -1, -1),
                         use_cache=False, return_dict=True).last_hidden_state


def selected_logps(model, hidden, targets, chunk_size=64, temperature=1.0):
    """Vocabulary projection is checkpointed in small chunks; gradients stay exact."""
    import torch
    from torch.utils.checkpoint import checkpoint
    head = model.get_base_model().lm_head
    hflat, tflat = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
    outputs = []
    def project(h, tokens):
        logits = head(h).float() / temperature
        return logits.gather(-1, tokens[:, None]).squeeze(-1) - torch.logsumexp(logits, -1)
    for start in range(0, hflat.shape[0], chunk_size):
        h, tokens = hflat[start:start + chunk_size], tflat[start:start + chunk_size]
        if torch.is_grad_enabled() and h.requires_grad:
            values = checkpoint(project, h, tokens, use_reentrant=False)
        else:
            values = project(h, tokens)
        outputs.append(values)
    return torch.cat(outputs).reshape(targets.shape)


def sft_loss(model, batch, chunk_size=64):
    import torch
    hidden = decoder_hidden(model, batch["input_ids"], batch["attention_mask"])
    labels = batch["labels"][:, 1:]
    selected = labels != -100
    logps = selected_logps(model, hidden[:, :-1][selected], labels[selected], chunk_size)
    return -logps.mean()


def completion_logps(model, prompt_ids, completion_ids, chunk_size=64, temperature=1.0):
    import torch
    ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device="cuda")
    mask = torch.ones_like(ids)
    hidden = decoder_hidden(model, ids, mask)
    completion_hidden = hidden[:, len(prompt_ids) - 1:-1]
    targets = ids[:, len(prompt_ids):]
    return selected_logps(model, completion_hidden, targets, chunk_size, temperature).squeeze(0)


def sample_completions(model, tokenizer, row, count, config):
    import torch
    from transformers import GenerationConfig
    policy = config["grpo"]
    activate_adapter(model, "default")
    model.eval()
    if count < 1:
        raise ValueError("At least one completion is required")
    generation_batch_size = int(policy.get("generation_batch_size", 3))
    if generation_batch_size < 1:
        raise ValueError("generation_batch_size must be positive")
    device = next(model.parameters()).device
    ids = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
    results = []
    # Start from an explicit sampling configuration rather than inherit the
    # checkpoint's top-k, repetition penalty, or other hidden logits warpers.
    generation = GenerationConfig(max_new_tokens=int(policy.get("max_completion_tokens", 768)),
        do_sample=True, temperature=float(policy.get("temperature", .8)),
        top_p=float(policy.get("top_p", .95)), top_k=0, repetition_penalty=1.0,
        no_repeat_ngram_size=0, num_beams=1, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True)
    # Sampling can use a few independent candidates at once, while old/ref
    # likelihoods and policy backward remain separate one-candidate microbatches.
    with torch.inference_mode(), last_token_generation(model):
        for start in range(0, count, generation_batch_size):
            batch_count = min(generation_batch_size, count - start)
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       generation_config=generation,
                                       use_model_defaults=False,
                                       num_return_sequences=batch_count)
            if generated.shape[0] != batch_count:
                raise RuntimeError("Native generation returned an unexpected candidate count")
            for generated_row in generated:
                tokens = generated_row[len(row["prompt_ids"]):].tolist()
                ended = tokenizer.eos_token_id in tokens
                if ended:
                    # HF right-pads candidates that finish earlier in the batch;
                    # retain the first EOS, then discard every later pad/token.
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                score = score_completion(text, row["reference_medications"], row["model_input"]["candidate_medications"])
                # A complete JSON without EOS is retained and recorded, but a
                # partial JSON naturally receives invalid=0 via the shared parser.
                results.append({"text": text, "completion_ids": tokens,
                                "completion_mask": [1] * len(tokens), "ended_with_eos": ended, **score})
    return results


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def latest_checkpoint(output_dir, requested):
    if not requested:
        return None
    if requested != "latest":
        path = Path(requested)
        if not (path / "training_state.pt").is_file():
            raise ValueError("Resume checkpoint must include full optimizer/RNG/trainer state")
        return path
    available = sorted(Path(output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    available = [p for p in available if (p / "training_state.pt").is_file()]
    if not available:
        raise ValueError("No complete checkpoint is available to resume")
    return available[-1]


def read_checkpoint(path, expected_fingerprint):
    import torch
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    if state["fingerprint"] != expected_fingerprint:
        raise ValueError("Training data, prompts, configuration or runtime differ from the checkpoint")
    return state


def save_checkpoint(model, tokenizer, optimizer, scheduler, state, output_dir, keep=5):
    import torch
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / f"checkpoint-{state['global_step']}"
    if (final / "training_state.pt").is_file():
        # Re-evaluation and zero-variance skips change curriculum/RNG state, but
        # never adapter weights at the same optimizer step. Update only the full
        # state atomically, preserving a usable checkpoint even if interrupted.
        temporary_state = final / f".training_state.tmp-{os.getpid()}.pt"
        torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng": rng_state()}, temporary_state)
        temporary_state.replace(final / "training_state.pt")
        write_json(final / "resume_summary.json", {key: value for key, value in state.items()
                   if key not in ("last_rollout",)})
        return final
    temporary = output_dir / f".checkpoint-{state['global_step']}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    activate_adapter(model, "default")
    model.save_pretrained(temporary, safe_serialization=True, selected_adapters=["default"], save_embedding_layers=False)
    tokenizer.save_pretrained(temporary)
    torch.save({**state, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_state()}, temporary / "training_state.pt")
    write_json(temporary / "resume_summary.json", {key: value for key, value in state.items()
               if key not in ("last_rollout",)})
    if final.exists():
        shutil.rmtree(final)
    temporary.rename(final)
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    for old in checkpoints[:-keep]:
        shutil.rmtree(old)
    return final


def optimizer_step(model, optimizer, scheduler, max_grad_norm, verification_path, global_step):
    import torch
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad and p.grad is not None]
    if not parameters:
        raise RuntimeError("No policy gradients were produced")
    gradient_abs_sum, best = 0.0, None
    for name, p in parameters:
        if not torch.isfinite(p.grad).all():
            raise RuntimeError(f"Non-finite gradient: {name}")
        magnitude = p.grad.detach().float().abs().sum().item()
        gradient_abs_sum += magnitude
        if best is None or magnitude > best[0]:
            best = (magnitude, name, p)
    if not math.isfinite(gradient_abs_sum) or gradient_abs_sum <= 0:
        raise RuntimeError("Optimizer step has zero or non-finite gradients")
    grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in parameters], max_grad_norm))
    learning_rate = float(optimizer.param_groups[0]["lr"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise RuntimeError("Counted optimizer updates require a positive finite learning rate")
    before = best[2].detach().float().cpu().clone()
    optimizer.step()
    scheduler.step()
    delta = (best[2].detach().float().cpu() - before).abs()
    change = delta.sum().item()
    if not math.isfinite(change) or change <= 0:
        raise RuntimeError("Positive-learning-rate optimizer step did not change LoRA weights")
    report = {"status": "passed", "optimizer_step": global_step, "learning_rate": learning_rate,
              "positive_learning_rate": learning_rate > 0, "finite_nonzero_gradients": True,
              "gradient_abs_sum": gradient_abs_sum, "gradient_norm": grad_norm, "parameter": best[1],
              "weight_delta_abs_sum": change, "weight_delta_abs_max": delta.max().item(),
              "weight_delta_abs_mean": delta.mean().item()}
    if not Path(verification_path).exists():
        write_json(verification_path, report)
    optimizer.zero_grad(set_to_none=True)
    return report


def make_optimizer(model, settings, total_steps):
    import torch
    from torch.optim.lr_scheduler import LambdaLR
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=float(settings["learning_rate"]), betas=(.9, .999), weight_decay=float(settings.get("weight_decay", 0.0)))
    warmup = max(0, int(total_steps * float(settings.get("warmup_ratio", .05))))
    def multiplier(index):
        # First optimizer step is already positive; no fake zero-LR update.
        if index < warmup:
            return (index + 1) / max(1, warmup)
        progress = (index - warmup) / max(1, total_steps - warmup)
        return max(.1, .5 * (1 + math.cos(math.pi * min(1., progress))))
    return optimizer, LambdaLR(optimizer, multiplier)
