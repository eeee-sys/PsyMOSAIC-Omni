#!/usr/bin/env python3
"""Prepare text-only teacher inputs from the official E-DAIC patient splits.

Only ``patient_record`` is a model payload. Patient IDs, split membership, and
hashes are cache/audit metadata outside that object. Existing English clinical
summaries and transcript quotations are preserved verbatim.
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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


STAGES = (
    "presenting_problem", "symptoms", "course", "functioning", "context_history"
)
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
EXPECTED_COUNTS = {"train": 163, "dev": 56, "test": 56}
SCHEMA_VERSION = "edaic_text_patient_record_v1"


def canonical_json(value: Any) -> str:
    """Return stable JSON for audit hashing, independent of dictionary order."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def profile_sha256(patient_record: Mapping[str, Any]) -> str:
    """Hash exactly the text payload sent to the teacher, excluding metadata."""
    return hashlib.sha256(canonical_json(patient_record).encode("utf-8")).hexdigest()


def sanitize_patient_record(source_record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the five-stage text-only patient record with a field allowlist.

    ``source_record`` can be a complete source row or its ``patient_record``.
    Empty quotations are omitted; nonempty quotation strings are never stripped,
    normalized, translated, or deduplicated. Unexpected structured values inside
    clinical summary fields are rejected rather than converted to text.
    """
    source = source_record.get("patient_record", source_record)
    if not isinstance(source, Mapping):
        raise ValueError("patient_record must be an object")
    result: dict[str, Any] = {}
    for stage_name in STAGES:
        stage = source.get(stage_name)
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
            elif isinstance(value, str):
                clean_update[field] = value
            else:
                raise ValueError(f"{stage_name}.{field} must be a string")

        # Prefer the original multimodal_evidence.text source. Accept the clean
        # text_evidence form as well so preparation is idempotent.
        evidence = stage.get("text_evidence")
        if evidence is None:
            multimodal = stage.get("multimodal_evidence", {})
            if not isinstance(multimodal, Mapping):
                raise ValueError(f"{stage_name}.multimodal_evidence must be an object")
            evidence = multimodal.get("text", [])
        if not isinstance(evidence, list):
            raise ValueError(f"{stage_name} text evidence must be a list")
        quotes: list[str] = []
        for item in evidence:
            quotation = item.get("text") if isinstance(item, Mapping) else item
            if not isinstance(quotation, str):
                raise ValueError(f"{stage_name} evidence quotation must be a string")
            if quotation.strip():
                quotes.append(quotation)
        result[stage_name] = {"record_update": clean_update, "text_evidence": quotes}
    return result


def _source_split_dir(source_root: Path) -> Path:
    for candidate in (source_root, source_root / "splits", source_root / "edaic" / "splits"):
        if all((candidate / f"{split}_patient_records.jsonl").is_file() for split in EXPECTED_COUNTS):
            return candidate
    raise FileNotFoundError("Could not find all three official *_patient_records.jsonl split files")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_profiles(
    source_root: str | Path,
    output_root: str | Path,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate all splits, then write sanitized JSONL profiles and a manifest.

    Official counts are mandatory by default. ``expected_counts`` allows small
    synthetic fixtures in contract tests; the split names remain train/dev/test.
    No output is written until counts, IDs, split membership, and all records
    have passed validation.
    """
    expected = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    if set(expected) != set(EXPECTED_COUNTS):
        raise ValueError("Expected split names must be exactly train, dev, and test")
    source_dir = _source_split_dir(Path(source_root))
    destination = Path(output_root)
    prepared: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    missing_evidence: dict[str, dict[str, int]] = {}
    for split in EXPECTED_COUNTS:
        source_path = source_dir / f"{split}_patient_records.jsonl"
        raw = source_path.read_bytes()
        source_hashes[split] = hashlib.sha256(raw).hexdigest()
        rows: list[dict[str, Any]] = []
        missing = Counter({stage: 0 for stage in STAGES})
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{split}:{line_number} is not an object")
            patient_id = item.get("patient_id")
            if not isinstance(patient_id, (str, int)) or isinstance(patient_id, bool):
                raise ValueError(f"{split}:{line_number} has an invalid patient ID")
            patient_id = str(patient_id)
            if not patient_id.strip():
                raise ValueError(f"{split}:{line_number} has an empty patient ID")
            if patient_id in seen_ids:
                raise ValueError(
                    f"Duplicate patient ID across {seen_ids[patient_id]} and {split} splits"
                )
            if item.get("split") != split:
                raise ValueError(f"{split}:{line_number} has inconsistent source split metadata")
            payload = sanitize_patient_record(item)
            for stage in STAGES:
                if not payload[stage]["text_evidence"]:
                    missing[stage] += 1
            rows.append({
                "patient_id": patient_id,
                "split": split,
                "patient_record": payload,
                "profile_sha256": profile_sha256(payload),
            })
            seen_ids[patient_id] = split
        if len(rows) != expected[split]:
            raise ValueError(f"{split} count is {len(rows)}, expected {expected[split]}")
        prepared[split] = rows
        missing_evidence[split] = dict(missing)

    outputs: dict[str, dict[str, Any]] = {}
    for split, rows in prepared.items():
        content = "".join(canonical_json(row) + "\n" for row in rows)
        target = destination / f"{split}.jsonl"
        _atomic_text(target, content)
        outputs[split] = {
            "path": str(target.resolve()),
            "patients": len(rows),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "profile_hashes": {row["patient_id"]: row["profile_sha256"] for row in rows},
            "missing_text_evidence_by_stage": missing_evidence[split],
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_split_directory": str(source_dir.resolve()),
        "source_sha256": source_hashes,
        "stages": list(STAGES),
        "payload_fields_per_stage": ["record_update", "text_evidence"],
        "model_payload_field": "patient_record",
        "excluded_metadata": [
            "patient_id", "split", "source_segment_ids", "segment_id", "timestamps",
            "audio", "video", "feature_statistics", "evidence_provenance",
            "labels", "scores", "diagnoses", "provenance",
        ],
        "text_policy": "Preserve existing English summary and quotation strings verbatim.",
        "split_policy": "Official disjoint patient splits; only train is eligible for SFT.",
        "counts": {split: len(rows) for split, rows in prepared.items()},
        "total_patients": len(seen_ids),
        "patient_ids_disjoint": True,
        "outputs": outputs,
    }
    _atomic_text(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare_profiles(args.source_root, args.output_root)
    print(json.dumps({
        "counts": manifest["counts"],
        "total_patients": manifest["total_patients"],
        "manifest": str((args.output_root / "manifest.json").resolve()),
        "missing_text_evidence": {
            split: meta["missing_text_evidence_by_stage"]
            for split, meta in manifest["outputs"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
