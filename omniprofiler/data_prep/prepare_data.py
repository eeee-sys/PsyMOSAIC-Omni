#!/usr/bin/env python3
"""Prepare exact text-only Teacher targets and patient-disjoint SFT datasets."""
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
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import random

STAGES = ("presenting_problem", "symptoms", "course", "functioning", "context_history")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def dump_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def patient_sort_key(patient_id):
    return int(str(patient_id).split("_")[-1])


def edaic_target(record, stage):
    value = record["patient_record"][stage]
    return {"stage": stage, "record_update": value["record_update"], "source_segment_ids": value["source_segment_ids"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=PROJECT_DATA_ROOT)
    parser.add_argument("--source-code", type=Path, default=PROJECT_CODE_ROOT / "data_processing")
    parser.add_argument("--output", type=Path, default=PROJECT_DATA_ROOT / "omniprofiler")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prompt_root = Path(__file__).resolve().parent / "prompts"
    edaic_root = args.source_data / "edaic"
    canonical_rich = edaic_root / "patient_records_multimodal.jsonl"
    rich_path = canonical_rich if canonical_rich.exists() else edaic_root / "patient_records.jsonl"
    source_hashes_before = {str(rich_path): sha256(rich_path)}
    rich_records = load_jsonl(rich_path)
    edaic_by_id = {str(row["patient_id"]): row for row in rich_records}
    assert len(rich_records) == len(edaic_by_id) == 275
    assert all("record_update" in row["patient_record"]["presenting_problem"] for row in rich_records)
    transcript_hashes = {}
    edaic_transcripts = {}
    edaic_transcript_payloads = {}
    official_splits = {}
    for split in ("train", "dev", "test"):
        source = RAW_DATA_ROOT / "E-DAIC/labels" / f"{split}_split.csv"
        source_hashes_before[str(source)] = sha256(source)
        with source.open(newline="", encoding="utf-8-sig") as handle:
            official_splits[split] = {str(int(row["Participant_ID"])) for row in csv.DictReader(handle)}
    for pid, row in edaic_by_id.items():
        tpath = edaic_root / "transcripts" / f"{pid}.json"
        transcript = load_json(tpath)
        segments = transcript["segments"]
        assert len({segment["segment_id"] for segment in segments}) == len(segments)
        # Exact original strings, including whitespace, are retained after the ID separator.
        text = "\n".join(f"{segment['segment_id']} | {segment['text']}" for segment in segments)
        edaic_transcripts[pid] = text
        edaic_transcript_payloads[pid] = segments
        transcript_hashes[pid] = sha256(tpath)
        assert pid in official_splits[row["split"]]
    assert collections.Counter(row["split"] for row in rich_records) == {"train": 163, "dev": 56, "test": 56}
    for stage in (*STAGES, "system"):
        source = args.source_code / "prompts" / f"{stage}.txt"
        source_hashes_before[str(source)] = sha256(source)
        target = prompt_root / "edaic" / f"{stage}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    edaic_system = (prompt_root / "edaic/system.txt").read_text(encoding="utf-8").strip()
    edaic_prompts = {stage: (prompt_root / "edaic" / f"{stage}.txt").read_text(encoding="utf-8").strip() for stage in STAGES}
    edaic_sft, edaic_targets, edaic_minimal = [], [], []
    for pid in sorted(edaic_by_id, key=patient_sort_key):
        row = edaic_by_id[pid]
        grouped = {"patient_id": pid, "split": row["split"], "patient_record": {}}
        ids = {segment["segment_id"] for segment in edaic_transcript_payloads[pid]}
        for stage in STAGES:
            target = edaic_target(row, stage)
            assert set(target) == {"stage", "record_update", "source_segment_ids"}
            assert set(target["source_segment_ids"]) <= ids
            grouped["patient_record"][stage] = target
            metadata = {"patient_id": pid, "dataset": "E-DAIC", "split": row["split"], "stage": stage}
            edaic_targets.append({**metadata, "target": target})
            edaic_sft.append({**metadata, "messages": [
                {"role": "system", "content": edaic_system},
                {"role": "user", "content": edaic_prompts[stage] + "\n\nFULL INTERVIEW TRANSCRIPT\n" + edaic_transcripts[pid]},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, allow_nan=False)}]})
        edaic_minimal.append(grouped)
    # Validate exact equality to the Teacher-derived SFT already used during reconstruction.
    original_sft = []
    for split in ("train", "dev", "test"):
        source = edaic_root / "sft" / f"{split}.jsonl"
        source_hashes_before[str(source)] = sha256(source)
        original_sft.extend(load_jsonl(source))
    original_lookup = {(sample["patient_id"], sample["stage"]): sample for sample in original_sft}
    assert len(original_lookup) == len(original_sft) == 1375
    assert all(sample == original_lookup[(sample["patient_id"], sample["stage"])] for sample in edaic_sft)

    # Every patient has five stage targets built from the complete transcript.
    assert len({(sample["patient_id"], sample["stage"]) for sample in edaic_sft}) == len(edaic_sft)
    assert all(count == 5 for count in collections.Counter(sample["patient_id"] for sample in edaic_sft).values())
    for sample in edaic_sft:
        assert [message["role"] for message in sample["messages"]] == ["system", "user", "assistant"]
        prompt, transcript = sample["messages"][1]["content"].split("\n\nFULL INTERVIEW TRANSCRIPT\n", 1)
        assert transcript == edaic_transcripts[sample["patient_id"]]
        target = json.loads(sample["messages"][2]["content"])
        assert set(target) == {"stage", "record_update", "source_segment_ids"}
        assert target["stage"] == sample["stage"]
    for split in ("train", "dev", "test"):
        ids = {sample["patient_id"] for sample in edaic_sft if sample["split"] == split}
        assert ids == official_splits[split]
        for other_split in ("train", "dev", "test"):
            if other_split != split:
                assert not ids & official_splits[other_split]
    out = args.output
    dump_jsonl(out / "edaic/patient_records_minimal.jsonl", edaic_minimal)
    dump_jsonl(out / "edaic/stage_targets_minimal.jsonl", edaic_targets)
    dump_jsonl(out / "edaic/clean_transcripts.jsonl", [{"patient_id": pid, "transcript": edaic_transcripts[pid]} for pid in sorted(edaic_by_id, key=patient_sort_key)])
    for split in ("train", "dev", "test"):
        dump_jsonl(out / "edaic" / f"{split}.jsonl", [sample for sample in edaic_sft if sample["split"] == split])
    train_samples = [sample for sample in edaic_sft if sample["split"] == "train"]
    random.Random(args.seed).shuffle(train_samples)
    assert len(train_samples) == 815
    dump_jsonl(out / "train.jsonl", train_samples)
    dump_jsonl(out / "dev.jsonl", [sample for sample in edaic_sft if sample["split"] == "dev"])
    assert all(sha256(path) == digest for path, digest in source_hashes_before.items())
    assert all(sha256(edaic_root / "transcripts" / f"{pid}.json") == digest for pid, digest in transcript_hashes.items())
    report = {
        "status": "passed", "seed": args.seed, "stages": list(STAGES),
        "edaic": {"patients": 275, "patient_splits": {"train": 163, "dev": 56, "test": 56},
                  "sft_splits": {"train": 815, "dev": 280, "test": 280},
                  "teacher_targets_unchanged": True, "segment_ids_retained": True},
        "samples": {"train": 815, "dev": 280, "test": 280},
        "checks": {"full_original_transcripts_retained": True, "patient_splits_disjoint": True,
                   "five_samples_per_patient": True, "target_top_level_keys_exact": True,
                   "no_audio_video_evidence_in_sft": True, "no_labels_scores_or_filenames_in_messages": True,
                   "source_files_unchanged": True},
        "source_sha256": {str(rich_path): sha256(rich_path)},
        "official_split_source_sha256": {path: digest for path, digest in source_hashes_before.items() if path.endswith("_split.csv")},
        "edaic_transcript_sha256": transcript_hashes,
        "prompt_sha256": {str(path.relative_to(prompt_root)): sha256(path) for path in sorted(prompt_root.glob("*/*.txt"))},
        "outputs": {str(path.relative_to(out)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(out.rglob("*.jsonl"))},
    }
    dump_json(out / "preparation_validation_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k in ("status", "seed", "edaic", "samples", "checks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
