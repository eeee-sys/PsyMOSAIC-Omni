"""Prepare caption-only MidiCaps train data and independent reward calibration."""
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
import random
from collections import Counter
from pathlib import Path
from project_paths import RAW_DATA_ROOT
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_dataset(
    source_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    calibration_size: int = 32,
    train_size: int = 1000,
) -> dict[str, Any]:
    """Keep only official training rows; preserve caption text and duplicates.

    ``train.json`` is JSONL containing BOTH official splits. The bool-valued
    ``test_set`` field is authoritative. Row IDs are original physical line
    numbers (1-based). Calibration rows and their identical-caption copies are
    excluded from optimization; other repeated captions remain unchanged.
    """
    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()
    if calibration_size < 2 or train_size < 1:
        raise ValueError("calibration_size must be >= 2 and train_size >= 1")
    training: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    source_digest = hashlib.sha256()
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            source_digest.update(raw)
            if not raw.strip():
                counts["blank_lines"] += 1
                continue
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            split = record.get("test_set")
            if split is not False and split is not True:
                raise ValueError(f"Line {line_number}: test_set must be a JSON boolean")
            counts["total_records"] += 1
            counts["official_test_records" if split is True else "official_train_records"] += 1
            caption = record.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Line {line_number}: caption must be a nonempty string")
            if split is False:
                training.append({"row_id": line_number, "caption": caption})
    if len(training) < calibration_size + train_size:
        raise ValueError("Not enough official training rows for the requested partitions")

    random.Random(seed).shuffle(training)
    calibration = training[:calibration_size]
    calibration_captions = {row["caption"] for row in calibration}
    optimization: list[dict[str, Any]] = []
    excluded_copies: list[int] = []
    for row in training[calibration_size:]:
        if row["caption"] in calibration_captions:
            excluded_copies.append(row["row_id"])
        elif len(optimization) < train_size:
            optimization.append(row)
    if len(optimization) != train_size:
        raise ValueError("Not enough caption-disjoint training rows after calibration exclusion")
    assert not ({r["row_id"] for r in calibration} & {r["row_id"] for r in optimization})
    assert not (calibration_captions & {r["caption"] for r in optimization})

    destination.mkdir(parents=True, exist_ok=True)
    calibration_path = destination / "calibration_prompts.jsonl"
    train_path = destination / "train_prompts.jsonl"
    _write_jsonl(calibration_path, calibration)
    _write_jsonl(train_path, optimization)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": source_digest.hexdigest(),
        "format": "jsonl",
        "official_train_filter": "test_set is False",
        "counts": dict(counts),
        "official_train_unique_captions": len({r["caption"] for r in training}),
        "seed": seed,
        "row_id_definition": "1-based physical line number in the source JSONL",
        "caption_transform": "none",
        "training_deduplication": False,
        "calibration_size": len(calibration),
        "train_size": len(optimization),
        "calibration_rows": [r["row_id"] for r in calibration],
        "train_rows": [r["row_id"] for r in optimization],
        "calibration_copy_excluded_rows": excluded_copies,
        "calibration_copy_excluded_count": len(excluded_copies),
        "calibration_train_row_overlap": 0,
        "calibration_train_caption_overlap": 0,
        "outputs": {
            "calibration": {"path": str(calibration_path), "sha256": _sha256(calibration_path)},
            "train": {"path": str(train_path), "sha256": _sha256(train_path)},
        },
    }
    manifest_path = destination / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(RAW_DATA_ROOT / "MidiCaps" / "train.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=1000)
    args = parser.parse_args()
    manifest = prepare_dataset(args.source, args.output_dir, args.seed, args.calibration_size, args.train_size)
    print(json.dumps({k: manifest[k] for k in ("counts", "calibration_size", "train_size", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
