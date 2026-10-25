"""Materialize every MusicCaps row and its already-cropped embedded audio.

The source start/end fields describe the original YouTube interval.  They are
provenance, not offsets to apply to the embedded ten-second waveform.
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
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from project_paths import RAW_DATA_ROOT
import re
import statistics
import sys
import tempfile


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as f:
        tmp = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def part_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"part_(\d+)", path.stem)
    return (int(match.group(1)) if match else 10**9, path.name)


def parse_aspects(value: object) -> list[str]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value):
        raise ValueError("aspect_list must be a nonempty list of nonempty strings")
    return value


def prepare_dataset(source_dir: str | Path, output_dir: str | Path,
                    verify_decode: bool = True) -> dict:
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    audio_dir = output / "reference_audio"
    audio_dir.mkdir(exist_ok=True)
    parts = sorted(source.glob("*.parquet"), key=part_sort_key)
    if not parts:
        raise FileNotFoundError(f"No parquet parts in {source}")

    counters: Counter = Counter()
    ids: Counter = Counter()
    captions: Counter = Counter()
    sample_rates: Counter = Counter()
    channels: Counter = Counter()
    formats: Counter = Counter()
    durations: list[float] = []
    failures: list[dict] = []
    sources: list[dict] = []
    row_index = 0
    records_path = output / "records.jsonl"
    temporary_records = records_path.with_suffix(".jsonl.preparing")
    extension_by_format = {
        "WAV": ".wav", "WAVEX": ".wav", "RF64": ".wav",
        "FLAC": ".flac", "OGG": ".ogg", "AIFF": ".aiff", "AU": ".au",
    }

    # Audio files are individually resumable.  records.jsonl is replaced only
    # after all source rows have passed validation, preserving an earlier good run.
    with temporary_records.open("w", encoding="utf-8") as records:
        for part in parts:
            parquet = pq.ParquetFile(part)
            source_row = 0
            # Each source row group contains only 100 rows, but about 200 MB of
            # audio.  Reading it once avoids re-decoding giant binary pages for
            # many small record batches on older pyarrow releases.
            for group_index in range(parquet.num_row_groups):
                group = parquet.read_row_group(group_index)
                for row in group.to_pylist():
                    current_index = row_index
                    row_index += 1
                    counters["source_rows"] += 1
                    source_id = str(row.get("index", ""))
                    ids[source_id] += 1
                    try:
                        caption = row.get("caption")
                        if not isinstance(caption, str) or not caption.strip():
                            counters["missing_caption"] += 1
                            raise ValueError("Missing caption")
                        captions[caption] += 1
                        try:
                            aspects = parse_aspects(row.get("aspect_list"))
                        except Exception:
                            counters["invalid_aspects"] += 1
                            raise
                        audio_struct = row.get("audio") or {}
                        audio_bytes = audio_struct.get("bytes")
                        if not isinstance(audio_bytes, bytes) or not audio_bytes:
                            counters["missing_embedded_audio"] += 1
                            raise ValueError("Missing embedded audio bytes")
                        try:
                            info = sf.info(io.BytesIO(audio_bytes))
                            if info.frames <= 0:
                                raise ValueError("Audio contains zero frames")
                            if verify_decode:
                                wave, decoded_rate = sf.read(io.BytesIO(audio_bytes),
                                    dtype="float32", always_2d=True)
                                if (decoded_rate != info.samplerate
                                        or len(wave) != info.frames
                                        or not np.isfinite(wave).all()):
                                    raise ValueError("Invalid or non-finite decoded audio")
                        except Exception:
                            counters["invalid_audio"] += 1
                            raise
                        suffix = extension_by_format.get(info.format)
                        if suffix is None:
                            raise ValueError(f"Unsupported embedded format: {info.format}")
                        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(source_id).stem)
                        sample_id = f"{current_index:06d}_{slug or 'sample'}"
                        destination = audio_dir / (sample_id + suffix)
                        digest = hashlib.sha256(audio_bytes).hexdigest()
                        existing_ok = (destination.is_file()
                            and destination.stat().st_size == len(audio_bytes)
                            and file_sha256(destination) == digest)
                        if not existing_ok:
                            temporary_audio = destination.with_suffix(suffix + ".preparing")
                            with temporary_audio.open("wb") as f:
                                f.write(audio_bytes)
                            os.replace(temporary_audio, destination)
                            counters["audio_written"] += 1
                        else:
                            counters["audio_reused"] += 1

                        duration = float(info.duration)
                        durations.append(duration)
                        sample_rates[info.samplerate] += 1
                        channels[info.channels] += 1
                        formats[f"{info.format}/{info.subtype}"] += 1
                        if abs(duration - 10.0) > .01:
                            counters["duration_not_10_seconds"] += 1
                        if abs(duration - float(row["audio_len"])) > .01:
                            counters["audio_len_header_mismatch"] += 1
                        record = {
                            "sample_id": sample_id,
                            "row_index": current_index,
                            "caption": caption,
                            "aspects": aspects,
                            "aspect_prompt": ", ".join(aspects),
                            "reference_audio": str(destination),
                            "reference_sha256": digest,
                            "source_part": part.name,
                            "source_row": source_row,
                            "source_dataset_index": source_id,
                            "duration": duration,
                            "sample_rate": int(info.samplerate),
                            "channels": int(info.channels),
                        }
                        records.write(json.dumps(record, ensure_ascii=False,
                                                 allow_nan=False) + "\n")
                        counters["valid_records"] += 1
                    except Exception as error:
                        failures.append({"row_index": current_index,
                            "source_part": part.name, "source_row": source_row,
                            "source_dataset_index": source_id,
                            "error": f"{type(error).__name__}: {error}"})
                    finally:
                        source_row += 1
                records.flush()
                print(json.dumps({"part": part.name, "row_group": group_index,
                    "processed_rows": row_index, "failures": len(failures)}), flush=True)
            stat = part.stat()
            sources.append({"path": str(part), "rows": parquet.metadata.num_rows,
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": file_sha256(part)})
            print(json.dumps({"part": part.name, "processed_rows": row_index,
                "valid_records": counters["valid_records"], "failures": len(failures)}),
                flush=True)
        records.flush()
        os.fsync(records.fileno())

    manifest = {
        "dataset": "MusicCaps", "source_dir": str(source),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_file": str(records_path), "counts": dict(counters),
        "all_rows_retained": len(failures) == 0,
        "audio_validation": "full_decode_and_finite" if verify_decode else "headers_only",
        "audio_cropping": "Embedded audio is already cropped; no start_s/end_s slicing applied.",
        "ordering": "Numeric parquet part order, followed by original row order.",
        "prompt_rules": {"aspect_list": "Parse list literal and join with comma+space.",
                         "caption": "Preserve the original caption string.",
                         "aspect_list+caption": "aspect_prompt + newline + caption"},
        "unique_source_dataset_indices": len(ids),
        "duplicate_source_dataset_index_rows": sum(n - 1 for n in ids.values()),
        "unique_captions": len(captions),
        "duplicate_caption_rows": sum(n - 1 for n in captions.values()),
        "sample_rates": dict(sample_rates), "channels": dict(channels),
        "formats": dict(formats),
        "duration_seconds": {"min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "mean": statistics.mean(durations) if durations else None,
            "sum": sum(durations)},
        "sources": sources, "failures": failures,
    }
    if failures:
        atomic_json(output / "data_manifest.failed.json", manifest)
        raise RuntimeError(f"{len(failures)} source rows failed validation; see data_manifest.failed.json")
    os.replace(temporary_records, records_path)
    manifest["records_sha256"] = file_sha256(records_path)
    atomic_json(output / "data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(RAW_DATA_ROOT / "MusicCaps"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--headers-only", action="store_true")
    args = parser.parse_args()
    manifest = prepare_dataset(args.source_dir, args.output_dir,
                               verify_decode=not args.headers_only)
    print(json.dumps({"status": "complete", "records": manifest["counts"]["valid_records"],
        "manifest": str(Path(args.output_dir) / "data_manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
