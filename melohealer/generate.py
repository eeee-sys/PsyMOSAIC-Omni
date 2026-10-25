#!/usr/bin/env python3
"""Generate all paired model/input experiments, with atomic per-item resume."""
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
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import soundfile as sf
import torch

from project_paths import load_config
from evaluation.audio_normalization import normalize_waveform
from evaluation.inference import PairedAceInference, atomic_json, input_prompt, stable_seed

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def valid_output(path, row, mode, model, config):
    metadata = path.with_suffix(".json")
    if not path.exists() or not metadata.exists():
        return False
    try:
        value = json.loads(metadata.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    expected = {"sample_id": row["sample_id"], "model": model, "input": mode,
                "protocol_id": config["protocol_id"],
                "prompt_sha256": hashlib.sha256(input_prompt(row, mode).encode()).hexdigest()}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Existing output belongs to another experiment: {path}")
    if value.get("normalization") != normalization_policy(config):
        raise ValueError(f"Existing output uses another audio normalization policy: {path}")
    try:
        info = sf.info(path)
    except (OSError, sf.LibsndfileError):
        return False
    return info.frames == config["duration_seconds"] * 48000 and info.samplerate == 48000 and info.channels == 2


def normalization_policy(config):
    options = config.get("normalization", {})
    return {"version": "ace_native_peak_v2", "decode_peak_limit": 1.0,
            "enable_normalization": True,
            "normalization_db": options.get("normalization_db", -1.0),
            "near_silence_threshold": 1e-6, "channel_gain": "shared_stereo"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        raise ValueError("Require workers > 0 and 0 <= worker-index < workers")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    config = load_config(args.config)
    rows = [json.loads(line) for line in Path(config["data_records"]).read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    rows = [r for r in rows if r["row_index"] % args.workers == args.worker_index]
    run = Path(config["run_root"])
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / f"generation-worker-{args.worker_index}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    torch.set_num_threads(config.get("cpu_threads", 6))
    torch.manual_seed(config["seed"])
    runtime = None
    status_path = run / f"generation-worker-{args.worker_index}.json"
    total = len(rows) * len(config["input_kinds"]) * len(config["model_kinds"])
    completed = 0
    started = time.monotonic()

    def status(state, **values):
        value = {"state": state, "worker": args.worker_index, "pid": os.getpid(),
            "protocol_id": config["protocol_id"],
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "completed": completed, "total": total,
            "elapsed_seconds": time.monotonic() - started,
            "time_utc": datetime.now(timezone.utc).isoformat(), **values}
        atomic_json(status_path, value)
        print(json.dumps(value, allow_nan=False), flush=True)

    status("validating_cached_outputs")
    for mode in config["input_kinds"]:
        for offset in range(0, len(rows), config["generation_batch_size"]):
            batch = rows[offset:offset + config["generation_batch_size"]]
            paths = {(r["sample_id"], model): Path(config["audio_root"]) / model / mode / f"{r['sample_id']}.wav"
                     for r in batch for model in config["model_kinds"]}
            needed = []
            for row in batch:
                existing = [valid_output(paths[row["sample_id"], model], row, mode, model, config)
                            for model in config["model_kinds"]]
                completed += sum(existing)
                if not all(existing):
                    needed.append(row)
            if not needed:
                status("generating", input=mode, resumed=True)
                continue
            if runtime is None:
                status("loading_models")
                runtime = PairedAceInference(config).load()
                status("generating")
            prepared = [runtime.condition(row, mode) for row in needed]
            conditions = [value[0] for value in prepared]
            tokens = [value[1] for value in prepared]
            for model in config["model_kinds"]:
                waves = runtime.generate(needed, conditions, model)
                for row, wave, token_count in zip(needed, waves, tokens):
                    path = paths[row["sample_id"], model]
                    if valid_output(path, row, mode, model, config):
                        continue
                    if wave.shape != (2, config["duration_seconds"] * 48000) or not torch.isfinite(wave).all():
                        raise RuntimeError(f"Invalid generated waveform: {row['sample_id']}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(f".{os.getpid()}.tmp.wav")
                    normalization = normalization_policy(config)
                    audio = normalize_waveform(wave.numpy().T, normalization["normalization_db"])
                    sf.write(temporary, audio, 48000, subtype="FLOAT")
                    temporary.replace(path)
                    metadata = {"sample_id": row["sample_id"], "row_index": row["row_index"],
                        "model": model, "input": mode, "protocol_id": config["protocol_id"],
                        "seed": stable_seed(row, config), "prompt": input_prompt(row, mode),
                        "prompt_sha256": hashlib.sha256(input_prompt(row, mode).encode()).hexdigest(),
                        "text_tokens": token_count, "duration_seconds": config["duration_seconds"],
                        "sample_rate": 48000, "channels": 2, "subtype": "FLOAT",
                        "normalization": normalization,
                        "rms": float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))),
                        "peak": float(np.max(np.abs(audio))),
                        "generation_batch_size": len(needed), "file_bytes": path.stat().st_size}
                    atomic_json(path.with_suffix(".json"), metadata)
                    completed += 1
                status("generating", input=mode, model=model,
                       gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            if STOP:
                status("stopped")
                return
    if completed != total:
        raise RuntimeError(f"Incomplete generation: {completed}/{total}")
    status("complete")


if __name__ == "__main__":
    main()
