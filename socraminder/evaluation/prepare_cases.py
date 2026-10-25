#!/usr/bin/env python3
"""Prepare held-out, text-only E-DAIC cases without therapy targets or labels.

The published Stage 1 profile is the preferred source: it includes exactly the
five clinical record updates and their original English transcript quotations.
The ``patient_record`` field is the complete model payload. IDs, split, profile
hashes, and source hashes stay outside that object for bookkeeping only.
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
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STAGES = ("presenting_problem", "symptoms", "course", "functioning", "context_history")
RECORD_FIELDS = {
    "presenting_problem": ("main_concern", "current_state", "stated_needs"),
    "symptoms": (
        "mood", "interest", "anxiety", "sleep", "energy", "concentration",
        "appetite", "psychomotor", "self_evaluation", "trauma_related",
        "irritability", "other",
    ),
    "course": ("onset", "duration", "frequency", "trend", "episodes", "temporal_context"),
    "functioning": (
        "impairment", "affected_domains", "work_study", "relationships",
        "daily_life", "coping_activities",
    ),
    "context_history": (
        "recent_stressor", "social_support", "previous_treatment",
        "current_medication", "mental_health_history", "trauma_history",
        "medical_history", "substance_use", "background",
    ),
}
SCHEMA_VERSION = "psychotherapy_edaic_test_cases_v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sanitize_patient_record(source: Mapping[str, Any]) -> dict[str, Any]:
    """Match Stage 1's summary allowlist and retain quotations verbatim.

    Evidence dictionaries contribute only their text. Audio/video features,
    segment references, timestamps, provenance, diagnosis labels, simulation
    targets, and every other unlisted field never enter the model payload.
    """
    record = source.get("patient_record", source)
    if not isinstance(record, Mapping):
        raise ValueError("patient_record must be an object")
    result: dict[str, Any] = {}
    for stage_name in STAGES:
        stage = record.get(stage_name)
        if not isinstance(stage, Mapping):
            raise ValueError(f"Missing or invalid clinical stage: {stage_name}")
        update = stage.get("record_update")
        if not isinstance(update, Mapping):
            raise ValueError(f"{stage_name}.record_update must be an object")
        clean_update: dict[str, Any] = {}
        for field in RECORD_FIELDS[stage_name]:
            if field not in update:
                raise ValueError(f"Missing clinical field: {stage_name}.{field}")
            value = update[field]
            if stage_name == "functioning" and field == "affected_domains":
                if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                    raise ValueError("functioning.affected_domains must be a list of strings")
                clean_update[field] = list(value)
            elif not isinstance(value, str):
                raise ValueError(f"{stage_name}.{field} must be a string")
            else:
                clean_update[field] = value
        evidence = stage.get("text_evidence")
        if evidence is None:
            multimodal = stage.get("multimodal_evidence", {})
            if not isinstance(multimodal, Mapping):
                raise ValueError(f"{stage_name}.multimodal_evidence must be an object")
            evidence = multimodal.get("text", [])
        if not isinstance(evidence, list):
            raise ValueError(f"{stage_name}.text_evidence must be a list")
        quotations: list[str] = []
        for item in evidence:
            text = item.get("text") if isinstance(item, Mapping) else item
            if not isinstance(text, str):
                raise ValueError(f"{stage_name} quotation must be a string")
            if text.strip():
                quotations.append(text)
        result[stage_name] = {"record_update": clean_update, "text_evidence": quotations}
    return result


def _patient_id(row: Mapping[str, Any]) -> str:
    patient_id = row.get("patient_id")
    if isinstance(patient_id, bool) or not isinstance(patient_id, (str, int)):
        raise ValueError("patient_id must be a string or integer")
    patient_id = str(patient_id)
    if not patient_id.strip():
        raise ValueError("patient_id must not be empty")
    return patient_id


def _read_rows(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return raw, rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_cases(
    source_profiles: str | Path,
    output: str | Path,
    expected_patients: int = 56,
    official_records: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all cases before writing deterministic JSONL and an audit file."""
    source = Path(source_profiles)
    destination = Path(output)
    if source.resolve() == destination.resolve():
        raise ValueError("Output must not overwrite the published source profiles")
    if type(expected_patients) is not int or expected_patients < 1:
        raise ValueError("expected_patients must be a positive integer")
    raw, rows = _read_rows(source)
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing_quotes = dict.fromkeys(STAGES, 0)
    quote_counts = []
    for row in rows:
        patient_id = _patient_id(row)
        if patient_id in seen:
            raise ValueError(f"Duplicate patient ID: {patient_id}")
        if row.get("split") != "test":
            raise ValueError(f"Patient {patient_id} is not in the test split")
        record = sanitize_patient_record(row)
        profile_hash = stable_hash(record)
        if row.get("profile_sha256") is not None and row["profile_sha256"] != profile_hash:
            raise ValueError(f"Published profile hash mismatch for patient {patient_id}")
        profile = {
            "patient_id": patient_id,
            "split": "test",
            "patient_record": record,
            "profile_sha256": profile_hash,
        }
        cases.append({
            "patient_id": patient_id,
            "split": "test",
            "profile": profile,
            "patient_record": record,
            "source_sha256": stable_hash(row),
        })
        quote_counts.append(sum(len(record[stage]["text_evidence"]) for stage in STAGES))
        for stage in STAGES:
            missing_quotes[stage] += not bool(record[stage]["text_evidence"])
        seen.add(patient_id)
    if len(cases) != expected_patients:
        raise ValueError(f"Test count is {len(cases)}, expected {expected_patients}")

    official_audit: dict[str, Any] | None = None
    if official_records is not None:
        official_path = Path(official_records)
        if official_path.resolve() == destination.resolve():
            raise ValueError("Output must not overwrite the official source records")
        official_raw, official_rows = _read_rows(official_path)
        official_test = [_patient_id(row) for row in official_rows if row.get("split") == "test"]
        if len(official_test) != len(set(official_test)):
            raise ValueError("Duplicate patient IDs in official test records")
        if set(official_test) != seen:
            raise ValueError("Prepared patient IDs differ from the official E-DAIC test split")
        all_ids = [_patient_id(row) for row in official_rows]
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Official patient IDs overlap across source splits")
        official_audit = {
            "path": str(official_path.resolve()),
            "sha256": hashlib.sha256(official_raw).hexdigest(),
            "test_patient_ids_match": True,
            "test_patients": len(official_test),
        }
    content = "".join(canonical_json(case) + "\n" for case in cases)
    audit_path = destination.with_suffix(".audit.json")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_profiles": str(source.resolve()),
        "source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "output": str(destination.resolve()),
        "cases_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "cases": len(cases),
        "split": "test",
        "patient_ids": [case["patient_id"] for case in cases],
        "profile_sha256": {case["patient_id"]: case["profile"]["profile_sha256"] for case in cases},
        "model_payload": "patient_record",
        "payload_fields_per_stage": ["record_update", "text_evidence"],
        "text_policy": "Preserve original English summaries and quotations verbatim.",
        "diagnosis_labels_in_payload": False,
        "multimodal_features_in_payload": False,
        "therapy_targets_in_payload": False,
        "missing_text_evidence_by_stage": missing_quotes,
        "quotations_per_patient": {"min": min(quote_counts), "max": max(quote_counts)},
        "official_split_validation": official_audit,
    }
    _atomic_text(destination, content)
    _atomic_text(audit_path, json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-patients", type=int, default=56)
    parser.add_argument("--official-records", type=Path, help="Optional official all-split JSONL for exact held-out ID verification")
    args = parser.parse_args()
    audit = prepare_cases(args.source_profiles, args.output, args.expected_patients, args.official_records)
    print(json.dumps({key: audit[key] for key in ("cases", "split", "cases_sha256", "output", "official_split_validation")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
