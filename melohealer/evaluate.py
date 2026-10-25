#!/usr/bin/env python3
"""Run paired MusicCaps generation, metric extraction, and final comparison."""
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
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from project_paths import load_config

STOP = False
ACTIVE = []


def request_stop(signum, frame):
    global STOP
    STOP = True
    for job in ACTIVE:
        if job.poll() is None:
            job.terminate()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = Path(__file__).resolve().parent
    run = Path(config["run_root"])
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    def status(state, **extra):
        value = {"state": state, "pid": os.getpid(), "protocol_id": config["protocol_id"],
            "time_utc": datetime.now(timezone.utc).isoformat(), **extra}
        atomic_json(run / "pipeline_status.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    def launch(name, gpu, command):
        env = os.environ.copy()
        if gpu is None:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LOGURU_LEVEL="WARNING",
                   TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(config["cpu_threads"]),
                   OPENBLAS_NUM_THREADS=str(config["cpu_threads"]),
                   MALLOC_ARENA_MAX="2", PYTORCH_ALLOC_CONF="expandable_segments:True")
        with (run / f"{name}.log").open("ab", buffering=0) as out:
            process = subprocess.Popen([sys.executable, "-u", *command], cwd=root, env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        ACTIVE.append(process)
        (run / f"{name}.pid").write_text(str(process.pid) + "\n")
        status("running", stage=name, child_pid=process.pid, gpu=gpu)
        return process

    def wait_jobs(jobs):
        while jobs:
            for name, job in list(jobs.items()):
                code = job.poll()
                if code is None:
                    continue
                del jobs[name]
                ACTIVE.remove(job)
                if code != 0:
                    raise RuntimeError(f"{name} exited {code}; see {run / (name + '.log')}")
            if STOP:
                raise InterruptedError("Pipeline stop requested")
            if jobs:
                time.sleep(3)

    try:
        gpus = config["generation_gpus"]
        jobs = {}
        for index, gpu in enumerate(gpus):
            name = f"generation-worker-{index}"
            jobs[name] = launch(name, gpu, [str(root / "generate.py"), "--config", str(config_path),
                                           "--worker-index", str(index), "--workers", str(len(gpus))])
        wait_jobs(jobs)
        status("generation_complete", expected_audio=len(config["model_kinds"]) *
               len(config["input_kinds"]) * config["expected_samples"])
        # Dedicated metric phases run after all generated audio exists.
        # Each backend writes independent resumable feature shards.
        for pair in (("panns", "vggish"), ("aesthetics", "clap")):
            jobs = {}
            for index, backend in enumerate(pair):
                gpu = gpus[index % len(gpus)]
                name = f"metrics-{backend}"
                jobs[name] = launch(name, gpu, ["-m", "evaluation.metric_runner", "--config",
                    str(config_path), "--backend", backend, "--device", "cuda:0"])
            wait_jobs(jobs)
        wait_jobs({"metrics-aggregate": launch("metrics-aggregate", None,
            ["-m", "evaluation.metric_runner", "--config", str(config_path), "--backend", "aggregate"])})
        metrics = json.loads((Path(config["result_root"]) / "metrics.json").read_text())
        expected_groups = len(config["model_kinds"]) * len(config["input_kinds"])
        if len(metrics) != expected_groups or any(row["count"] != config["expected_samples"] for row in metrics):
            raise RuntimeError("The final result does not contain all configured full-sample experiments")
        directions = {"KL": -1, "IS": 1, "FD": -1, "FAD": -1, "PC": 1, "PQ": 1, "Align": 1}
        comparison = []
        for input_name in config["input_kinds"]:
            if not {"base", "finetuned"}.issubset(config["model_kinds"]):
                break
            base = next(row for row in metrics if row["model"] == "base" and row["input"] == input_name)
            trained = next(row for row in metrics if row["model"] == "finetuned" and row["input"] == input_name)
            comparison.append({"input": input_name, "count": base["count"],
                "delta_finetuned_minus_base": {key: trained[key] - base[key] for key in directions},
                "improved": {key: directions[key] * (trained[key] - base[key]) > 0 for key in directions}})
        atomic_json(Path(config["result_root"]) / "comparison.json", comparison)
        status("complete", samples_per_experiment=config["expected_samples"], experiments=expected_groups,
               metrics=str(Path(config["result_root"]) / "metrics.csv"))
    except BaseException as error:
        for job in ACTIVE:
            if job.poll() is None:
                job.terminate()
        status("stopped" if STOP else "failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
