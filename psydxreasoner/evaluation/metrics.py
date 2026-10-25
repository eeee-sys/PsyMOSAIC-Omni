"""Exact answer parsing and patient-complete positive-class metrics."""
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


import re

DISORDERS = ("DEPRESSION", "PTSD")
MODES = ("text", "text_audio", "text_video", "text_audio_video")
SOURCES = ("teacher", "generator")
SETTINGS = tuple((source, mode) for source in SOURCES for mode in MODES)
ANSWER_PATTERN = re.compile(
    r"\s*<rationale>(.*?)</rationale>\s*<answer>(POSITIVE|NEGATIVE)</answer>\s*",
    flags=re.DOTALL,
)
INVALID_POLICY = (
    "Every test patient is retained. An invalid or unavailable prediction is scored as "
    "incorrect: it contributes FN for positive ground truth and FP for negative ground "
    "truth. Invalid counts and valid-answer rates are also reported. Zero-denominator "
    "precision, recall, and F1 are defined as zero."
)


def parse_answer(text):
    """Accept one complete rationale/answer response; never extract a valid prefix."""
    match = ANSWER_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("Response must contain exactly one rationale and one binary answer")
    rationale, answer = match.groups()
    if not rationale.strip() or any(tag in rationale for tag in
            ("<rationale>", "</rationale>", "<answer>", "</answer>")):
        raise ValueError("Rationale is empty or contains additional output tags")
    return {"rationale": rationale.strip(), "answer": answer}


def binary_metrics(rows):
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    invalid_positive = invalid_negative = valid = 0
    for row in rows:
        truth = row["ground_truth_answer"]
        if truth not in ("POSITIVE", "NEGATIVE"):
            raise ValueError("Ground truth must be POSITIVE or NEGATIVE")
        prediction = row.get("answer")
        is_valid = row.get("valid", False) and prediction in ("POSITIVE", "NEGATIVE")
        if not is_valid:
            if truth == "POSITIVE":
                counts["fn"] += 1
                invalid_positive += 1
            else:
                counts["fp"] += 1
                invalid_negative += 1
            continue
        valid += 1
        counts[{("POSITIVE", "POSITIVE"): "tp", ("NEGATIVE", "POSITIVE"): "fp",
                ("NEGATIVE", "NEGATIVE"): "tn", ("POSITIVE", "NEGATIVE"): "fn"}
               [(truth, prediction)]] += 1
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"samples": len(rows), "positive_support": sum(r["ground_truth_answer"] == "POSITIVE" for r in rows),
            "precision": precision, "recall": recall, "f1": f1, **counts,
            "valid_answers": valid, "invalid_answers": invalid_positive + invalid_negative,
            "invalid_positive": invalid_positive, "invalid_negative": invalid_negative,
            "valid_answer_rate": valid / len(rows) if rows else 0.0}


def setting_metrics(rows, patient_ids, settings=SETTINGS):
    """Require a prediction for every patient/target/setting before reporting."""
    ids = set(map(str, patient_ids))
    if len(rows) != len(ids) * len(DISORDERS) * len(settings):
        raise ValueError("Evaluation is incomplete or contains duplicate predictions")
    reports = []
    for source, mode in settings:
        selected = [r for r in rows if (r["source"], r["evidence_mode"]) == (source, mode)]
        per_disorder = {}
        for disorder in DISORDERS:
            subset = [r for r in selected if r["target_disorder"] == disorder]
            if len(subset) != len(ids) or {str(r["patient_id"]) for r in subset} != ids:
                raise ValueError(f"Missing or duplicate patients for {source}/{mode}/{disorder}")
            per_disorder[disorder] = binary_metrics(subset)
        reports.append({"source": source, "evidence_mode": mode, "per_disorder": per_disorder,
                        "macro_average": {metric: sum(x[metric] for x in per_disorder.values()) / 2
                                          for metric in ("precision", "recall", "f1")},
                        "valid_answer_rate": sum(x["valid_answers"] for x in per_disorder.values()) / len(selected)})
    return reports
