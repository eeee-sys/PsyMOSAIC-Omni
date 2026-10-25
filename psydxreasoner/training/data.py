"""Canonical teacher profiles and label-disjoint target-conditioned prompts."""
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
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

STAGES = ("presenting_problem", "symptoms", "course", "functioning", "context_history")
TARGETS = ("DEPRESSION", "PTSD")
MODES = {
    "text": ("text",),
    "text_audio": ("text", "audio"),
    "text_video": ("text", "video"),
    "text_audio_video": ("text", "audio", "video"),
}
FIELDS = {
    "presenting_problem": ("main_concern", "current_state", "stated_needs"),
    "symptoms": ("mood", "interest", "anxiety", "sleep", "energy", "concentration", "appetite", "psychomotor", "self_evaluation", "trauma_related", "irritability", "other"),
    "course": ("onset", "duration", "frequency", "trend", "episodes", "temporal_context"),
    "functioning": ("impairment", "affected_domains", "work_study", "relationships", "daily_life", "coping_activities"),
    "context_history": ("recent_stressor", "social_support", "previous_treatment", "current_medication", "mental_health_history", "trauma_history", "medical_history", "substance_use", "background"),
}
PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _evidence_text(value):
    if not isinstance(value, list):
        raise ValueError("Each evidence modality must be a list")
    result = []
    for item in value:
        # Preserve the original quote; discard IDs, time ranges, and provenance.
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Evidence must contain nonempty strings or original text objects")
        result.append(text)
    return result


def sanitize_profile(profile, evidence_mode="text_audio_video", dataset=None):
    """Whitelist clinical fields and textual evidence; never emit labels or IDs."""
    if evidence_mode not in MODES:
        raise ValueError(f"Unsupported evidence mode {evidence_mode}")
    if not isinstance(profile, dict) or set(profile) != set(STAGES):
        raise ValueError("Expected exactly five clinical stages")
    clean = {}
    for stage in STAGES:
        value = profile[stage]
        if isinstance(value, str) and stage == "presenting_problem":
            update, evidence = {"main_concern": value}, {}
        elif isinstance(value, dict) and "record_update" in value:
            update, evidence = value["record_update"], value.get("multimodal_evidence", {})
        elif isinstance(value, dict):
            update, evidence = value, {}
        else:
            raise ValueError(f"Invalid clinical stage {stage}")
        if not isinstance(update, dict) or not update or not set(update) <= set(FIELDS[stage]):
            raise ValueError(f"Unexpected or empty clinical fields in {stage}")
        for key, item in update.items():
            if key == "affected_domains":
                if not isinstance(item, list) or any(not isinstance(v, str) for v in item):
                    raise ValueError("affected_domains must be a string list")
            elif not isinstance(item, str) or not item.strip():
                raise ValueError(f"{stage}/{key} must be a nonempty clinical string")
        if not isinstance(evidence, dict):
            raise ValueError("multimodal_evidence must be a dictionary")
        clean[stage] = {
            "record_update": dict(update),
            "multimodal_evidence": {
                modality: _evidence_text(evidence.get(modality, [])) if modality in MODES[evidence_mode] else []
                for modality in ("text", "audio", "video")
            },
        }
    return clean


def make_messages(profile, target_disorder):
    target = str(target_disorder).upper()
    if target not in TARGETS:
        raise ValueError(f"Invalid target disorder {target}")
    # Check canonical shape without injecting or removing any evidence.
    canonical = sanitize_profile(profile)
    if canonical != profile:
        raise ValueError("make_messages expects an already sanitized profile")
    return [
        {"role": "system", "content": (PROMPT_ROOT / "diagnosis_system.txt").read_text(encoding="utf-8").strip()},
        {"role": "user", "content": "PATIENT RECORD:\n" + json.dumps(profile, ensure_ascii=False, separators=(",", ":")) + "\n\nTARGET DISORDER:\n" + target},
    ]


def prepare(source_data, output):
    source_data, output = Path(source_data), Path(output)
    sources = [source_data / "edaic/patient_records_multimodal.jsonl"]
    source_hashes = {str(path): sha256(path) for path in sources}
    edaic = read_jsonl(sources[0])
    assert len(edaic) == len({str(r["patient_id"]) for r in edaic}) == 275
    split_ids = {s: set() for s in ("train", "dev", "test")}
    rows = {s: [] for s in split_ids}
    profiles = {s: [] for s in split_ids}

    def add(row, split):
        dataset = "E-DAIC"
        pid = str(row["patient_id"])
        key = (dataset, pid)
        assert key not in split_ids[split]
        split_ids[split].add(key)
        profile = sanitize_profile(row["patient_record"], dataset=dataset)
        profiles[split].append({"patient_id": pid, "dataset": dataset, "split": split, "patient_record": profile, "labels": row["labels"]})
        for target in TARGETS:
            label = row["labels"].get(target.lower())
            if label is None:
                continue
            if type(label) is not int or label not in (0, 1):
                raise ValueError(f"Invalid label {dataset}/{pid}/{target}")
            rows[split].append({"patient_id": pid, "dataset": dataset, "split": split,
                                "patient_record": profile, "target_disorder": target,
                                "ground_truth_answer": "POSITIVE" if label else "NEGATIVE",
                                "prompt": make_messages(profile, target)})

    for row in edaic:
        add(row, row["split"])
    assert not (split_ids["train"] & split_ids["dev"] or split_ids["train"] & split_ids["test"] or split_ids["dev"] & split_ids["test"])
    assert {split: len(value) for split, value in split_ids.items()} == {"train": 163, "dev": 56, "test": 56}
    assert {split: len(value) for split, value in rows.items()} == {"train": 326, "dev": 112, "test": 112}
    for split in rows:
        write_jsonl(output / f"{split}.jsonl", rows[split])
        write_jsonl(output / f"{split}_patient_profiles.jsonl", profiles[split])
    report = {
        "input_source": "teacher annotations, never Generator predictions during training",
        "training_samples": 326, "training_patients": {"E-DAIC": 163},
        "dev_samples": 112, "dev_patients": {"E-DAIC": 56},
        "test_samples": 112, "test_patients": {"E-DAIC": 56},
        "target_counts": {split: dict(Counter(r["target_disorder"] for r in samples)) for split, samples in rows.items()},
        "label_counts": {split: {target: dict(Counter(r["ground_truth_answer"] for r in samples if r["target_disorder"] == target)) for target in TARGETS} for split, samples in rows.items()},
        "missing_labels_used_as_negative": False,
        "patient_disjoint_splits": True,
        "model_input_fields": "five stage record_update + text/audio/video evidence; Target Disorder",
        "metadata_in_model_input": False,
        "source_sha256": source_hashes,
        "prompt_sha256": sha256(PROMPT_ROOT / "diagnosis_system.txt"),
    }
    assert source_hashes == {str(path): sha256(path) for path in sources}
    write_json(output / "preparation_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_data, args.output), indent=2, ensure_ascii=False))
