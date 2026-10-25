"""Turn-level SFT validation, complete-context tokenization and masked causal CE."""
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
import re


TARGET_PATTERN = re.compile(
    r"\A\s*<state_update>(.*?)</state_update>\s*"
    r"<response>(.*?)</response>\s*"
    r"<stage_status>\s*(stay|advance|complete)\s*</stage_status>\s*\Z",
    re.DOTALL,
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_target(content):
    """Accept only the three externally trained blocks, never teacher auxiliaries."""
    if not isinstance(content, str):
        raise ValueError("Assistant target must be text")
    match = TARGET_PATTERN.fullmatch(content)
    if match is None:
        raise ValueError("Target must contain exactly state_update, response, stage_status in that order")
    try:
        state_update = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("state_update must be valid JSON") from exc
    if not isinstance(state_update, dict):
        raise ValueError("state_update must be a JSON object")
    response = match.group(2).strip()
    if not response:
        raise ValueError("Therapist response must not be empty")
    forbidden = ("<patient_understanding", "<question_plan", "<state_update", "<response", "<stage_status")
    if any(tag in response.lower() for tag in forbidden):
        raise ValueError("Therapist response must not contain nested target or auxiliary blocks")
    return {"state_update": state_update, "response": response, "stage_status": match.group(3)}


def audit_split_isolation(train_records, dev_records, require_patient_ids=True,
                          expected_train_patients=None, expected_dev_patients=None):
    """Audit patient boundaries; this interface deliberately has no test-data input."""
    patients = {}
    sample_ids = set()
    for split, records in (("train", train_records), ("dev", dev_records)):
        if not records:
            raise ValueError(f"The {split} split is empty")
        patient_ids = set()
        for row in records:
            if row.get("split", split) != split:
                raise ValueError(f"Unexpected split {row.get('split')!r} in {split} data")
            patient_id = row.get("patient_id")
            if patient_id is None:
                if require_patient_ids:
                    raise ValueError(f"Missing patient_id in {split} SFT record")
            else:
                patient_ids.add(str(patient_id))
            sample_id = row.get("sample_id")
            if sample_id is not None:
                if str(sample_id) in sample_ids:
                    raise ValueError(f"Duplicate SFT sample_id {sample_id}")
                sample_ids.add(str(sample_id))
        patients[split] = patient_ids
    overlap = patients["train"] & patients["dev"]
    if overlap:
        raise ValueError(f"Train/dev patient overlap: {sorted(overlap)}")
    for split, expected in (("train", expected_train_patients), ("dev", expected_dev_patients)):
        if expected is not None and len(patients[split]) != expected:
            raise ValueError(f"Unexpected {split} patient count: {len(patients[split])} != {expected}")
    return {split: {"samples": len(records), "patients": len(patients[split]),
                    "patient_ids": sorted(patients[split])}
            for split, records in (("train", train_records), ("dev", dev_records))}


def tokenize_record(record, tokenizer, context_limit):
    messages = record["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("Expected exactly system, user and assistant messages")
    if any(not isinstance(m["content"], str) or not m["content"].strip() for m in messages):
        raise ValueError("All SFT messages must contain nonempty text")
    validate_target(messages[-1]["content"])
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if tokens[:len(prompt)] != prompt:
        raise ValueError("The assistant prefix is inconsistent with the chat template")
    if len(tokens) > context_limit:
        raise ValueError(f"Full dialogue context exceeds limit: {len(tokens)} > {context_limit}")
    if len(tokens) <= len(prompt):
        raise ValueError("Empty assistant token target")
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens),
            "labels": [-100] * len(prompt) + tokens[len(prompt):]}


class TokenizedDataset:
    """The minimal map-style dataset contract consumed by Trainer."""
    def __init__(self, records, tokenizer, context_limit):
        self.items = [tokenize_record(row, tokenizer, context_limit) for row in records]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        import torch
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        length = max(len(example["input_ids"]) for example in examples)
        return {key: torch.tensor([example[key] + [pad] * (length - len(example[key]))
                                   for example in examples], dtype=torch.long)
                for key, pad in (("input_ids", self.pad_token_id), ("attention_mask", 0), ("labels", -100))}


def selected_causal_loss(hidden_states, lm_head, labels, chunk_size=128):
    """Exact masked next-token CE; project only supervised hidden states in chunks."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    if chunk_size < 1:
        raise ValueError("Projection chunk size must be positive")
    batch_indices, token_indices = torch.where(labels[:, 1:] != -100)
    if token_indices.numel() == 0:
        raise ValueError("Batch has no supervised causal targets")

    def chunk_loss(states, targets):
        logits = lm_head(states)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100, reduction="sum")

    total_loss = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, token_indices.numel(), chunk_size):
        rows = batch_indices[start:start + chunk_size]
        columns = token_indices[start:start + chunk_size]
        states, targets = hidden_states[rows, columns, :], labels[rows, columns + 1]
        if torch.is_grad_enabled():
            value = checkpoint(chunk_loss, states, targets, use_reentrant=False, preserve_rng_state=False)
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
    if not items:
        raise ValueError("Cannot summarize an empty dataset")
    lengths = sorted(len(item["input_ids"]) for item in items)
    target_lengths = sorted(sum(token != -100 for token in item["labels"]) for item in items)

    def percentiles(values):
        return {"min": values[0], "p50": values[int((len(values) - 1) * .5)],
                "p95": values[int((len(values) - 1) * .95)], "max": values[-1]}

    return {"samples": len(items), "tokens": percentiles(lengths),
            "target_tokens": percentiles(target_lengths), "truncated_samples": 0}
