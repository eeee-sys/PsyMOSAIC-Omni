"""Reconstruct Generator-selected evidence from the complete segment caches."""
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


import importlib
import json
from pathlib import Path
import sys

STAGES = ("presenting_problem", "symptoms", "course", "functioning", "context_history")


def load_feature_rules(code_root):
    """Reuse the exact feature conversion implementation used for Teacher evidence."""
    code_root = Path(code_root).resolve()
    if not (code_root / "data_processing/features.py").is_file():
        raise FileNotFoundError(f"Feature conversion implementation absent at {code_root}")
    sys.path.insert(0, str(code_root))
    module = importlib.import_module("data_processing.features")
    if Path(module.__file__).resolve() != code_root / "data_processing/features.py":
        raise RuntimeError("A different feature conversion implementation was already imported")
    return module


def validate_generator_output(text, stage, field_schema, allowed_ids):
    """Strict whole-response schema check, including citation vocabulary."""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"stage", "record_update", "source_segment_ids"}:
        raise ValueError("Generator output must have exactly stage, record_update, source_segment_ids")
    if value["stage"] != stage or not isinstance(value["record_update"], dict):
        raise ValueError("Generator stage or record_update is invalid")
    if set(value["record_update"]) != set(field_schema):
        raise ValueError("Generator record_update does not have the exact stage fields")
    for key, kind in field_schema.items():
        item = value["record_update"][key]
        if kind == "str" and not isinstance(item, str):
            raise ValueError(f"Stage field {key} must be a string")
        if kind == "list" and (not isinstance(item, list) or not all(isinstance(x, str) for x in item)):
            raise ValueError(f"Stage field {key} must be a list of strings")
        if kind not in ("str", "list"):
            raise ValueError("Unsupported clinical field schema")
    ids = value["source_segment_ids"]
    if not isinstance(ids, list) or not all(isinstance(x, str) and x in allowed_ids for x in ids):
        raise ValueError("Generator citations contain an unknown segment ID")
    if len(ids) != len(set(ids)):
        raise ValueError("Generator citations contain duplicate segment IDs")
    return value


def load_patient_cache(edaic_root, patient_id, rules):
    root = Path(edaic_root)
    transcript = json.loads((root / "transcripts" / f"{patient_id}.json").read_text())
    features = json.loads((root / "feature_statistics" / f"{patient_id}.json").read_text())
    text_by_id = {row["segment_id"]: row for row in transcript["segments"]}
    feature_by_id = {row["segment_id"]: row for row in features["segments"]}
    if len(text_by_id) != len(transcript["segments"]) or len(feature_by_id) != len(features["segments"]):
        raise ValueError("Transcript or feature cache has duplicate segment IDs")
    if set(text_by_id) != set(feature_by_id):
        raise ValueError("Feature cache must cover all transcript segments")
    if features["rules_version"] != rules.RULES_VERSION:
        raise ValueError("Feature statistics version differs from the saved conversion rules")
    if features["alignment_tolerance_seconds"] != rules.ALIGNMENT_TOLERANCE_SECONDS:
        raise ValueError("Feature alignment tolerance differs from the Teacher conversion")
    return text_by_id, feature_by_id, features["session_baseline"]


def assemble_generator_profile(stage_outputs, text_by_id, feature_by_id, baseline, calibration, rules):
    """IDs stay internal; Diagnosis sees only clinical fields and evidence strings."""
    if set(stage_outputs) != set(STAGES):
        raise ValueError("A Generator profile requires all five successful stages")
    if calibration["version"] != rules.RULES_VERSION or calibration["fit_split"] != "train":
        raise ValueError("Expected the unchanged train-fitted feature calibration")
    profile = {}
    for stage in STAGES:
        output = stage_outputs[stage]
        ids = output["source_segment_ids"]
        stats = rules.aggregate_segments([feature_by_id[x] for x in ids])
        converted = rules.evidence_strings(stats, baseline, calibration)
        profile[stage] = {"record_update": output["record_update"],
                          "multimodal_evidence": {"text": [text_by_id[x]["text"] for x in ids],
                                                  "audio": converted["audio"], "video": converted["video"]}}
    return profile
