"""Full-transcript tokenization and assistant-only causal loss for text SFT."""
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


import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize_record(record, tokenizer, context_limit):
    messages = record["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected exactly system, user and assistant messages")
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True,
                                           add_generation_prompt=True)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                                           add_generation_prompt=False)
    if tokens[:len(prompt)] != prompt:
        raise ValueError("The assistant prefix is inconsistent with the chat template")
    if len(tokens) > context_limit:
        raise ValueError(f"Full transcript exceeds context: {len(tokens)} > {context_limit}")
    if len(tokens) <= len(prompt):
        raise ValueError("Empty assistant target")
    target = json.loads(messages[-1]["content"])
    dataset_name = record["dataset"].lower().replace("-", "")
    if dataset_name != "edaic":
        raise ValueError(f"Unknown dataset: {record['dataset']}")
    expected = {"stage", "record_update", "source_segment_ids"}
    if set(target) != expected:
        raise ValueError(f"Unexpected target fields: {set(target)}")
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": [-100] * len(prompt) + tokens[len(prompt):]}


class TokenizedDataset(Dataset):
    def __init__(self, records, tokenizer, context_limit):
        self.items = [tokenize_record(r, tokenizer, context_limit) for r in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        length = max(len(e["input_ids"]) for e in examples)
        result = {}
        for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([e[key] + [pad] * (length - len(e[key]))
                                        for e in examples], dtype=torch.long)
        return result


def selected_causal_loss(hidden_states, lm_head, labels, chunk_size=128):
    """Same masked causal CE as full-vocabulary logits, computed only for targets."""
    batch_indices, token_indices = torch.where(labels[:, 1:] != -100)
    if token_indices.numel() == 0:
        raise ValueError("Batch has no supervised causal targets")
    if chunk_size < 1:
        raise ValueError("Projection chunk size must be positive")
    def chunk_loss(states, targets):
        logits = lm_head(states)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-100, reduction="sum")
    total_loss = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, token_indices.numel(), chunk_size):
        rows = batch_indices[start:start+chunk_size]
        columns = token_indices[start:start+chunk_size]
        states, targets = hidden_states[rows, columns, :], labels[rows, columns+1]
        if torch.is_grad_enabled():
            value = checkpoint(chunk_loss, states, targets, use_reentrant=False,
                               preserve_rng_state=False)
        else:
            value = chunk_loss(states, targets)
        total_loss = total_loss + value
    return total_loss / token_indices.numel()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def length_stats(items):
    lengths = sorted(len(item["input_ids"]) for item in items)
    target_lengths = sorted(sum(t != -100 for t in item["labels"]) for item in items)
    def percentiles(values):
        return {"min": values[0], "p50": values[int((len(values)-1)*.5)],
                "p95": values[int((len(values)-1)*.95)], "max": values[-1]}
    return {"samples": len(items), "tokens": percentiles(lengths),
            "target_tokens": percentiles(target_lengths), "truncated_samples": 0}
