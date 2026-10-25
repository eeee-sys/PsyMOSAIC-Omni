#!/usr/bin/env python3
"""Export every AudioCaps test caption with its own original encoded reference.

Preparation is CPU-only. Audio is decoded only to inspect its actual metadata and
check finite samples; the exported reference contains the original encoded bytes.
No source start_time offset is applied to an already extracted AudioCaps clip.
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
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
from project_paths import RAW_DATA_ROOT
import re
import tempfile
from typing import Any
from urllib.parse import quote


FORMAT_VERSION = 2
SHARD_PATTERN = re.compile(r"test-(\d+)-of-(\d+)\.parquet\Z")
REQUIRED_COLUMNS = {"audiocap_id", "youtube_id", "start_time", "caption", "audio"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        # Make the directory entry durable as well as the file contents.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _output_lock(output_root: Path):
    import fcntl

    with (output_root / ".preparation.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another preparation process owns {output_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _discover_shards(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data" if (dataset_root / "data").is_dir() else dataset_root
    files = list(data_dir.glob("*.parquet"))
    if not files:
        raise ValueError(f"No Parquet shards found under {data_dir}")
    parsed = []
    for path in files:
        match = SHARD_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected non-test or unrecognized Parquet shard: {path}")
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    totals = {total for _, total, _ in parsed}
    if len(totals) != 1:
        raise ValueError("Test shards disagree about their declared shard count")
    declared = totals.pop()
    indices = sorted(index for index, _, _ in parsed)
    if declared < 1 or indices != list(range(declared)):
        raise ValueError(f"Incomplete or duplicate test shards: found {indices}, expected 0..{declared - 1}")
    return [path for _, _, path in sorted(parsed)]


def _inspect_audio(encoded: bytes) -> tuple[dict[str, Any], str]:
    import numpy as np
    import soundfile as sf

    if not encoded:
        raise ValueError("Reference audio bytes are empty")
    with sf.SoundFile(io.BytesIO(encoded)) as handle:
        metadata = {
            "sample_rate": int(handle.samplerate),
            "frames": int(handle.frames),
            "channels": int(handle.channels),
            "format": handle.format,
            "subtype": handle.subtype,
        }
        samples = handle.read(dtype="float64", always_2d=True)
    if metadata["sample_rate"] <= 0 or metadata["frames"] <= 0 or metadata["channels"] <= 0:
        raise ValueError(f"Reference has invalid audio dimensions: {metadata}")
    if samples.shape != (metadata["frames"], metadata["channels"]):
        raise ValueError("Decoded reference shape differs from its audio header")
    if not np.isfinite(samples).all():
        raise ValueError("Reference contains non-finite decoded samples")
    metadata["duration_s"] = metadata["frames"] / metadata["sample_rate"]
    metadata["finite"] = True
    digest = hashlib.sha256()
    digest.update(_json_bytes({key: metadata[key] for key in ["sample_rate", "frames", "channels"]}))
    # An explicit byte order keeps this comparison stable across CPU platforms.
    digest.update(np.asarray(samples, dtype="<f8", order="C").tobytes(order="C"))
    return metadata, digest.hexdigest()


def _audio_suffix(audio_format: str) -> str:
    return {"WAV": ".wav", "WAVEX": ".wav", "FLAC": ".flac", "OGG": ".ogg", "AIFF": ".aiff"}.get(audio_format, ".audio")


def _source_inventory(dataset_root: Path, shards: list[Path]) -> tuple[list[dict[str, Any]], str]:
    import pyarrow.parquet as pq

    inventory = []
    for path in shards:
        parquet = pq.ParquetFile(path)
        missing = REQUIRED_COLUMNS - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        inventory.append({
            "path": str(path.relative_to(dataset_root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": parquet.metadata.num_rows,
        })
    return inventory, hashlib.sha256(_json_bytes(inventory)).hexdigest()


def _validate_identity(output_root: Path, dataset_root: Path, fingerprint: str) -> list[dict[str, Any]]:
    identities = [output_root / "preparation_state.json", output_root / "dataset_manifest.json"]
    found_identity = False
    repairs = []
    for path in identities:
        if not path.exists():
            continue
        found_identity = True
        document = json.loads(path.read_text(encoding="utf-8"))
        if (document.get("source_fingerprint") != fingerprint
                or document.get("dataset_root") != str(dataset_root)):
            raise ValueError(f"Dataset identity differs from {path}; choose a new output root")
        if document.get("format_version") != FORMAT_VERSION:
            # Version 1 could stop after finding two different waveforms for one
            # semantic clip. Only this incomplete, same-source state can migrate.
            if (path.name != "preparation_state.json" or document.get("format_version") != 1
                    or document.get("status") != "preparing"
                    or (output_root / "dataset_manifest.json").exists()):
                raise ValueError(f"Prepared dataset format differs from {path}; choose a new output root")
            old_content = path.read_bytes()
            old_sha = hashlib.sha256(old_content).hexdigest()
            archive = output_root / "preparation_history" / f"state_v1_{old_sha[:16]}.json"
            if archive.exists() and archive.read_bytes() != old_content:
                raise ValueError(f"Preparation history hash collision: {archive}")
            if not archive.exists():
                _atomic_bytes(archive, old_content)
            repairs.append({
                "from_format_version": 1,
                "to_format_version": FORMAT_VERSION,
                "reason": "compatible data-fidelity repair: per-caption-source-audio",
                "previous_validated_rows": document.get("validated_rows", 0),
                "previous_validated_clips": document.get("validated_clips", 0),
                "previous_state_sha256": old_sha,
                "previous_state_archive": str(archive),
                "existing_reference_files_preserved": True,
            })
        else:
            for repair in document.get("compatibility_repairs", []):
                if repair not in repairs:
                    repairs.append(repair)
    if not found_identity:
        references = output_root / "references"
        if (output_root / "records.jsonl").exists() or (references.exists() and any(references.iterdir())):
            raise ValueError("Existing records/references have no trusted preparation identity; choose a new output root")
    return repairs


def prepare_dataset(dataset_root: str | Path, output_root: str | Path, *,
                    expected_rows: int | None = None, expected_clips: int | None = None) -> dict[str, Any]:
    """Prepare all rows in numeric shard order; reruns verify and reuse references."""
    import pyarrow.parquet as pq

    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root == dataset_root or dataset_root in output_root.parents:
        raise ValueError("Output root must be outside the source dataset directory")
    shards = _discover_shards(dataset_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_root):
        inventory, fingerprint = _source_inventory(dataset_root, shards)
        total_rows = sum(item["rows"] for item in inventory)
        if expected_rows is not None and total_rows != expected_rows:
            raise ValueError(f"Expected {expected_rows} caption rows, found {total_rows}")
        repairs = _validate_identity(output_root, dataset_root, fingerprint)
        state = {
            "format_version": FORMAT_VERSION,
            "dataset_root": str(dataset_root),
            "source_fingerprint": fingerprint,
            "status": "preparing",
            "validated_rows": 0,
            "validated_clips": 0,
            "validated_reference_files": 0,
            "compatibility_repairs": repairs,
        }
        _atomic_bytes(output_root / "preparation_state.json", _json_bytes(state))
        references_dir = output_root / "references"
        references_dir.mkdir(exist_ok=True)
        references: dict[str, dict[str, Any]] = {}
        clips: dict[tuple[str, int], dict[str, Any]] = {}
        caption_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
        records: list[dict[str, Any]] = []
        sample_ids: set[str] = set()
        for shard, shard_info in zip(shards, inventory):
            parquet = pq.ParquetFile(shard)
            columns = [name for name in parquet.schema_arrow.names if name in REQUIRED_COLUMNS or name == "audio_length"]
            source_row = 0
            for batch in parquet.iter_batches(batch_size=16, columns=columns):
                for row in batch.to_pylist():
                    audiocap_id, youtube_id, start_time, caption = (row[name] for name in ["audiocap_id", "youtube_id", "start_time", "caption"])
                    if (not isinstance(audiocap_id, int) or isinstance(audiocap_id, bool)
                            or not isinstance(youtube_id, str) or not youtube_id
                            or not isinstance(start_time, int) or isinstance(start_time, bool) or start_time < 0
                            or not isinstance(caption, str) or not caption.strip()):
                        raise ValueError(f"Invalid caption metadata in {shard.name} row {source_row}")
                    sample_id = f"ac_{audiocap_id}"
                    if sample_id in sample_ids:
                        raise ValueError(f"Duplicate sample_id: {sample_id}")
                    sample_ids.add(sample_id)
                    audio = row["audio"]
                    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes) or not audio["bytes"]:
                        raise ValueError(f"Missing embedded audio bytes for {sample_id}")
                    encoded = audio["bytes"]
                    raw_hash = hashlib.sha256(encoded).hexdigest()
                    clip = (youtube_id, start_time)
                    first_row_for_clip = clip not in clips
                    if first_row_for_clip:
                        clips[clip] = {"youtube_id": youtube_id, "start_time": start_time,
                                       "caption_rows": 0, "reference_variants": {}}
                    if raw_hash not in references:
                        try:
                            metadata, decoded_hash = _inspect_audio(encoded)
                        except (ValueError, RuntimeError, OSError) as exc:
                            raise ValueError(f"Invalid reference audio for {sample_id}: {exc}") from exc
                        suffix = _audio_suffix(metadata["format"])
                        filename = f"ac_ref_sha256_{raw_hash}{suffix}"
                        reference_path = references_dir / filename
                        legacy_path = references_dir / f"ac_ref_{quote(youtube_id, safe='')}_{start_time}{suffix}"
                        if first_row_for_clip and legacy_path.exists():
                            # Preserve and reuse our verified incomplete v1 export.
                            # Later variants of this clip must use their own bytes.
                            if sha256_file(legacy_path) != raw_hash:
                                raise ValueError(f"Existing reference hash mismatch: {legacy_path}")
                            reference_path = legacy_path
                        elif reference_path.exists():
                            if sha256_file(reference_path) != raw_hash:
                                raise ValueError(f"Existing reference hash mismatch: {reference_path}")
                        else:
                            _atomic_bytes(reference_path, encoded)
                        references[raw_hash] = {
                            "reference_audio": str(reference_path),
                            "reference_sha256": raw_hash,
                            "decoded_sha256": decoded_hash,
                            **metadata,
                            "caption_rows": 0,
                            "source_clips": [],
                        }
                    reference = references[raw_hash]
                    clip_info = clips[clip]
                    variants = clip_info["reference_variants"]
                    if raw_hash not in variants:
                        variants[raw_hash] = {
                            "reference_sha256": raw_hash,
                            "reference_audio": reference["reference_audio"],
                            "decoded_sha256": reference["decoded_sha256"],
                            "caption_rows": 0,
                        }
                        reference["source_clips"].append({"youtube_id": youtube_id, "start_time": start_time})
                    variants[raw_hash]["caption_rows"] += 1
                    clip_info["caption_rows"] += 1
                    reference["caption_rows"] += 1
                    caption_sets[clip].add(caption)
                    record = {
                        "sample_id": sample_id,
                        "row_index": len(records),
                        "split": "test",
                        "caption": caption,
                        "youtube_id": youtube_id,
                        "start_time": start_time,
                        "audiocap_id": audiocap_id,
                        "reference_audio": reference["reference_audio"],
                        "reference_sha256": reference["reference_sha256"],
                        "reference_sample_rate": reference["sample_rate"],
                        "reference_frames": reference["frames"],
                        "reference_channels": reference["channels"],
                        "reference_duration_s": reference["duration_s"],
                        "source_shard": shard_info["path"],
                        "source_row_index": source_row,
                        "source_audio_path": audio.get("path"),
                        "source_audio_bytes_sha256": raw_hash,
                    }
                    if "audio_length" in row:
                        record["source_audio_length"] = row["audio_length"]
                    records.append(record)
                    source_row += 1
                state.update(validated_rows=len(records), validated_clips=len(clips),
                             validated_reference_files=len(references))
                _atomic_bytes(output_root / "preparation_state.json", _json_bytes(state))
            if source_row != shard_info["rows"]:
                raise ValueError(f"Parquet row count changed during preparation: {shard}")
        if expected_clips is not None and len(clips) != expected_clips:
            raise ValueError(f"Expected {expected_clips} clips, found {len(clips)}")
        # Catch source changes that occurred after the inventory was computed.
        if _discover_shards(dataset_root) != shards:
            raise ValueError("Source shard inventory changed during preparation")
        for shard, original in zip(shards, inventory):
            if shard.stat().st_size != original["size_bytes"] or sha256_file(shard) != original["sha256"]:
                raise ValueError(f"Source shard changed during preparation: {shard}")
        records_content = b"".join(_json_bytes(record) for record in records)
        records_hash = hashlib.sha256(records_content).hexdigest()
        references_list = list(references.values())
        semantic_clips = []
        for item in clips.values():
            item = dict(item)
            item["reference_variants"] = list(item["reference_variants"].values())
            item["unique_encoded_variants"] = len(item["reference_variants"])
            item["unique_decoded_variants"] = len({variant["decoded_sha256"] for variant in item["reference_variants"]})
            semantic_clips.append(item)
        manifest = {
            "format_version": FORMAT_VERSION,
            "dataset_root": str(dataset_root),
            "source_fingerprint": fingerprint,
            "split": "test",
            "evaluation_unit": "caption_row",
            "reference_policy": "per-caption-source-audio",
            "reference_weighting": "one_original_reference_per_caption_row",
            "row_order": "numeric_parquet_shard_then_row",
            "caption_rows": len(records),
            "unique_clips": len(clips),
            "unique_reference_files": len(references),
            "unique_decoded_audio_variants": len({item["decoded_sha256"] for item in references_list}),
            "caption_rows_per_clip": {str(count): count_clips for count, count_clips in sorted(Counter(item["caption_rows"] for item in clips.values()).items())},
            "duplicate_caption_rows_within_clip": sum(item["caption_rows"] - len(caption_sets[clip]) for clip, item in clips.items()),
            "clips_with_multiple_encoded_byte_variants": sum(item["unique_encoded_variants"] > 1 for item in semantic_clips),
            "clips_with_multiple_decoded_variants": sum(item["unique_decoded_variants"] > 1 for item in semantic_clips),
            "records_file": "records.jsonl",
            "records_sha256": records_hash,
            "sources": inventory,
            "references": references_list,
            "semantic_clips": semantic_clips,
            "compatibility_repairs": repairs,
        }
        manifest_content = _json_bytes(manifest)
        manifest_hash = hashlib.sha256(manifest_content).hexdigest()
        _atomic_bytes(output_root / "records.jsonl", records_content)
        _atomic_bytes(output_root / "records.jsonl.sha256", f"{records_hash}  records.jsonl\n".encode("ascii"))
        _atomic_bytes(output_root / "dataset_manifest.json", manifest_content)
        _atomic_bytes(output_root / "dataset_manifest.sha256", f"{manifest_hash}  dataset_manifest.json\n".encode("ascii"))
        state.update(status="complete", records_sha256=records_hash, manifest_sha256=manifest_hash)
        _atomic_bytes(output_root / "preparation_state.json", _json_bytes(state))
        return manifest


def verify_prepared_dataset(output_root: str | Path, *, verify_references: bool = False) -> dict[str, Any]:
    """Reject incomplete/mixed manifests and optionally hash all exported references."""
    output_root = Path(output_root).expanduser().resolve()
    manifest_path = output_root / "dataset_manifest.json"
    checksum = (output_root / "dataset_manifest.sha256").read_text(encoding="ascii").split()[0]
    if sha256_file(manifest_path) != checksum:
        raise ValueError("Dataset manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported prepared dataset format")
    records_path = output_root / "records.jsonl"
    records_hash = sha256_file(records_path)
    sidecar_hash = (output_root / "records.jsonl.sha256").read_text(encoding="ascii").split()[0]
    if records_hash != manifest["records_sha256"] or records_hash != sidecar_hash:
        raise ValueError("Caption records SHA256 mismatch")
    references = {item["reference_sha256"]: item for item in manifest["references"]}
    if len(references) != manifest["unique_reference_files"]:
        raise ValueError("Unique reference file count differs from manifest")
    samples = set()
    row_references = Counter()
    semantic_clips = set()
    count = 0
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("Blank caption record encountered")
            row = json.loads(line)
            if row["row_index"] != count or row["sample_id"] in samples or row["split"] != "test":
                raise ValueError("Caption row order, identity, or split is inconsistent")
            raw_hash = row["source_audio_bytes_sha256"]
            if row["reference_sha256"] != raw_hash or raw_hash not in references:
                raise ValueError("Caption source audio does not match its original reference")
            reference = references[raw_hash]
            if row["reference_audio"] != reference["reference_audio"]:
                raise ValueError("Caption reference path differs from manifest")
            for suffix, field in [("sample_rate", "sample_rate"), ("frames", "frames"),
                                  ("channels", "channels"), ("duration_s", "duration_s")]:
                if row[f"reference_{suffix}"] != reference[field]:
                    raise ValueError("Caption reference audio metadata differs from manifest")
            samples.add(row["sample_id"])
            semantic_clips.add((row["youtube_id"], row["start_time"]))
            row_references[raw_hash] += 1
            count += 1
    if count != manifest["caption_rows"] or len(semantic_clips) != manifest["unique_clips"]:
        raise ValueError("Caption row or semantic clip count differs from manifest")
    if set(row_references) != set(references) or any(row_references[key] != item["caption_rows"] for key, item in references.items()):
        raise ValueError("Caption reference weighting differs from manifest")
    if verify_references:
        for reference in manifest["references"]:
            if sha256_file(Path(reference["reference_audio"])) != reference["reference_sha256"]:
                raise ValueError(f"Reference SHA256 mismatch: {reference['reference_audio']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(RAW_DATA_ROOT / "AudioCaps"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-clips", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_prepared_dataset(args.output_root, verify_references=True)
    else:
        manifest = prepare_dataset(args.dataset_root, args.output_root,
                                   expected_rows=args.expected_rows, expected_clips=args.expected_clips)
    print(json.dumps({key: manifest[key] for key in ["caption_rows", "unique_clips", "records_sha256", "source_fingerprint"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
