"""Resumable feature extraction and CPU aggregation for six MusicCaps runs.

Each atomic NPZ shard records its ordered sample IDs, audio file stat manifest,
checkpoint fingerprint, and extractor code hash. No missing audio or feature
shard is silently dropped. Only one backbone is loaded at a time. Importing
this module or running ``--backend aggregate`` does not import PyTorch.
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
import contextlib
import csv
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.special import log_expit, log_softmax

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from project_paths import load_config, expand_paths
from evaluation.statistics import calculate_metrics


BACKENDS = ("panns", "vggish", "aesthetics", "clap")
MODELS = ("base", "finetuned")
INPUTS = ("aspect", "caption", "aspect_caption")
CACHE_VERSION = 1


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False,
                     allow_nan=False), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _stat(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    info = target.stat()
    if not target.is_file():
        raise ValueError(f"Expected a regular file: {target}")
    return {"path": str(target.resolve()), "size": info.st_size,
            "mtime_ns": info.st_mtime_ns}


def _checkpoint_fingerprint(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if target.is_file():
        return [_stat(target)]
    if not target.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {target}")
    files = sorted(item for item in target.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Checkpoint directory is empty: {target}")
    return [_stat(item) for item in files]


def _configuration(config) -> dict:
    if isinstance(config, (str, Path)):
        config = load_config(config)
    config = expand_paths(dict(config))
    required = ("data_records", "audio_root", "feature_root", "result_root",
                "vendor_root", "assets_root", "checkpoints")
    for key in required:
        if key not in config:
            raise ValueError(f"Missing evaluation configuration field: {key}")
    for key in BACKENDS:
        if key not in config["checkpoints"]:
            raise ValueError(f"Missing checkpoints.{key}")
    return config


def _records(config: dict) -> list[dict]:
    result = []
    seen = set()
    with Path(config["data_records"]).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("sample_id", "row_index", "caption", "aspect_prompt", "reference_audio"):
                if key not in record:
                    raise ValueError(f"Record line {line_number} has no {key}")
            sample_id = record["sample_id"]
            if not isinstance(sample_id, str) or not sample_id or Path(sample_id).name != sample_id:
                raise ValueError(f"Invalid sample_id at line {line_number}")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            if not all(isinstance(record[key], str) and record[key].strip()
                       for key in ("caption", "aspect_prompt")):
                raise ValueError(f"Missing prompt text for {sample_id}")
            seen.add(sample_id)
            result.append(record)
    if not result:
        raise ValueError("The evaluation manifest has no records")
    return result


def _batch_size(config: dict, backend: str) -> int:
    value = config.get("batch_sizes", {}).get(backend, config.get("batch_size", 16))
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Invalid {backend} batch size: {value}")
    return value


def _prompt(record: dict, input_name: str) -> str:
    if input_name == "aspect":
        return record["aspect_prompt"]
    if input_name == "caption":
        return record["caption"]
    if input_name == "aspect_caption":
        return record["aspect_prompt"] + "\n" + record["caption"]
    raise ValueError(f"Unknown input condition: {input_name}")


def _signature(config: dict, backend: str) -> dict:
    source = Path(__file__).with_name("metrics_backbones.py")
    return {"checkpoint": _checkpoint_fingerprint(config["checkpoints"][backend]),
            "extractor_sha256": _sha256(source),
            "records_sha256": _sha256(Path(config["data_records"])),
            "vendor_root": str(Path(config["vendor_root"]).resolve()),
            "assets_root": str(Path(config["assets_root"]).resolve()),
            "batch_size": _batch_size(config, backend)}


def _audio_path(config: dict, record: dict, group: tuple[str, str] | None) -> Path:
    if group is None:
        return Path(record["reference_audio"])
    model, input_name = group
    return Path(config["audio_root"]) / model / input_name / (record["sample_id"] + ".wav")


def _jobs(config: dict, records: list[dict], backend: str) -> list[dict]:
    size = _batch_size(config, backend)
    signature = _signature(config, backend)
    root = Path(config["feature_root"])
    groups = [(model, input_name) for model in config.get("model_kinds", MODELS)
              for input_name in config.get("input_kinds", INPUTS)]
    if backend in ("panns", "vggish"):
        groups.insert(0, None)
    result = []
    for group in groups:
        group_parts = ("reference",) if group is None else group
        for start in range(0, len(records), size):
            rows = records[start:start + size]
            paths = [_audio_path(config, row, group) for row in rows]
            try:
                sources = [_stat(path) for path in paths]
            except FileNotFoundError:
                missing = [str(path) for path in paths if not path.is_file()]
                raise FileNotFoundError(f"Missing required {backend} audio files: {missing[:10]}")
            ids = [row["sample_id"] for row in rows]
            metadata = {"version": CACHE_VERSION, "backend": backend, "kind": "audio",
                        "signature": signature, "ids": ids, "sources": sources}
            result.append({"path": root.joinpath(backend, *group_parts, f"batch-{start:06d}.npz"),
                           "metadata": metadata, "ids": ids, "inputs": [str(p) for p in paths],
                           "count": len(rows), "group": group, "start": start})
    if backend == "clap":
        for input_name in dict.fromkeys(("caption", *config.get("input_kinds", INPUTS))):
            for start in range(0, len(records), size):
                rows = records[start:start + size]
                texts = [_prompt(row, input_name) for row in rows]
                ids = [row["sample_id"] for row in rows]
                metadata = {"version": CACHE_VERSION, "backend": backend, "kind": "text",
                            "signature": signature, "ids": ids,
                            "text_sha256": hashlib.sha256(_json(texts).encode()).hexdigest()}
                result.append({"path": root / "clap_text" / input_name / f"batch-{start:06d}.npz",
                               "metadata": metadata, "ids": ids, "inputs": texts,
                               "count": len(rows), "group": input_name, "start": start})
    return result


def _validate_payload(job: dict, arrays: dict[str, np.ndarray]) -> None:
    count = job["count"]
    backend, kind = job["metadata"]["backend"], job["metadata"]["kind"]
    shapes = {}
    if backend == "panns":
        shapes = {"logits": (count, 527), "embedding": (count, 2048)}
    elif backend == "aesthetics":
        shapes = {key: (count,) for key in ("PC", "PQ", "CE", "CU")}
    elif backend == "clap":
        shapes = {"text_embedding" if kind == "text" else "audio_embedding": (count, 512)}
    elif backend == "vggish":
        if set(arrays) != {"patches", "offsets"}:
            raise ValueError("VGGish cache needs patches and offsets")
        patches, offsets = arrays["patches"], arrays["offsets"]
        if patches.ndim != 2 or patches.shape[1] != 128 or offsets.shape != (count + 1,):
            raise ValueError("Invalid VGGish patch/offset shape")
        if (not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0
                or offsets[-1] != len(patches) or not np.all(np.diff(offsets) > 0)):
            raise ValueError("Invalid VGGish patch offsets")
    else:
        raise ValueError(f"Unknown backbone: {backend}")
    if shapes and set(arrays) != set(shapes):
        raise ValueError(f"Unexpected {backend} feature fields: {sorted(arrays)}")
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Invalid {backend} {key} shape: {arrays[key].shape}, expected {shape}")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Nonfinite {backend} feature value")


def _read_cache(job: dict, *, required: bool = False) -> dict[str, np.ndarray] | None:
    path = job["path"]
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing feature shard; run its backbone first: {path}")
        return None
    try:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["_metadata"].item()))
            ids = stored["_sample_ids"].tolist()
            if metadata != job["metadata"] or ids != job["ids"]:
                raise ValueError("Cache metadata or sample order differs")
            arrays = {name: stored[name].copy() for name in stored.files if not name.startswith("_")}
        _validate_payload(job, arrays)
        return arrays
    except Exception as error:
        if required:
            raise ValueError(f"Incomplete or stale cache {path}: {error}") from error
        _log("cache_rebuild", path=str(path), reason=str(error))
        return None


def _write_cache(job: dict, arrays: dict[str, np.ndarray]) -> None:
    _validate_payload(job, arrays)
    path = job["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, _metadata=np.array(_json(job["metadata"])),
                                _sample_ids=np.asarray(job["ids"], dtype=np.str_), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another metric process owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _extract(backbone, job: dict) -> dict[str, np.ndarray]:
    backend, kind = job["metadata"]["backend"], job["metadata"]["kind"]
    if backend == "clap":
        if kind == "text":
            return {"text_embedding": np.asarray(backbone.text_features(job["inputs"]))}
        return {"audio_embedding": np.asarray(backbone.audio_features(job["inputs"]))}
    values = backbone.extract(job["inputs"])
    if backend != "vggish":
        return {key: np.asarray(value) for key, value in values.items()}
    patches = [np.asarray(value) for value in values["patches"]]
    if len(patches) != job["count"]:
        raise ValueError("VGGish output count differs from input count")
    return {"patches": np.concatenate(patches, axis=0),
            "offsets": np.concatenate(([0], np.cumsum([len(value) for value in patches]))).astype(np.int64)}


def run_backend(config, backend: str, device: str = "cuda:0") -> dict:
    """Extract one backbone's reference and generated shards, resuming valid ones."""
    config = _configuration(config)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown extraction backend: {backend}")
    with _lock(Path(config["feature_root"]) / f".{backend}.lock"):
        records = _records(config)
        jobs = _jobs(config, records, backend)
        pending = [job for job in jobs if _read_cache(job) is None]
        _log("backend_plan", backend=backend, total_batches=len(jobs),
             cached_batches=len(jobs) - len(pending), pending_batches=len(pending))
        if not pending:
            return {"backend": backend, "batches": len(jobs), "computed": 0}
        module = importlib.import_module("evaluation.metrics_backbones")
        classes = {"panns": "PannsBackbone", "vggish": "VGGishBackbone",
                   "aesthetics": "AestheticsBackbone", "clap": "CLAPBackbone"}
        kwargs = {"checkpoint": config["checkpoints"][backend],
                  "vendor_root": config["vendor_root"], "device": device}
        if backend == "clap":
            kwargs["assets_root"] = config["assets_root"]
        backbone = getattr(module, classes[backend])(**kwargs)
        try:
            for index, job in enumerate(pending, 1):
                _write_cache(job, _extract(backbone, job))
                _log("feature_batch", backend=backend, kind=job["metadata"]["kind"],
                     group=job["group"], completed=index, pending_total=len(pending),
                     first_sample=job["ids"][0], samples=job["count"])
        finally:
            backbone.close()
            del backbone
            gc.collect()
            # torch has already been imported by a backbone, never by aggregation.
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
        return {"backend": backend, "batches": len(jobs), "computed": len(pending)}


def _assembled(jobs: list[dict], group, *, kind: str = "audio") -> dict[str, np.ndarray]:
    selected = [job for job in jobs if job["group"] == group and job["metadata"]["kind"] == kind]
    selected.sort(key=lambda job: job["start"])
    if not selected:
        raise ValueError(f"No feature shards for {group}, {kind}")
    parts = [_read_cache(job, required=True) for job in selected]
    if "patches" in parts[0]:
        return {"patches": np.concatenate([part["patches"] for part in parts], axis=0)}
    return {name: np.concatenate([part[name] for part in parts], axis=0) for name in parts[0]}


def _cosine_rows(audio: np.ndarray, text: np.ndarray) -> np.ndarray:
    if audio.shape != text.shape:
        raise ValueError("CLAP audio/text embedding shapes differ")
    audio = np.asarray(audio, dtype=np.float64)
    text = np.asarray(text, dtype=np.float64)
    denominator = np.linalg.norm(audio, axis=1) * np.linalg.norm(text, axis=1)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("CLAP embeddings have a zero or nonfinite norm")
    return np.clip(np.sum(audio * text, axis=1) / denominator, -1.0, 1.0)


def _paired_kl_rows(reference: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    log_ref, log_gen = log_softmax(reference, axis=1), log_softmax(generated, axis=1)
    primary = np.maximum(0.0, np.sum(np.exp(log_ref) * (log_ref - log_gen), axis=1))
    sig_ref, sig_gen = log_expit(reference), log_expit(generated)
    sigmoid = np.sum(np.exp(sig_ref) * (sig_ref - np.logaddexp(sig_gen, np.log(1e-6))), axis=1)
    return primary, sigmoid


@contextlib.contextmanager
def _atomic_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def aggregate(config) -> list[dict]:
    """Aggregate all six complete experiments on CPU, without loading a backbone."""
    config = _configuration(config)
    records = _records(config)
    root = Path(config["result_root"])
    with _lock(root / ".aggregate.lock"):
        jobs = {backend: _jobs(config, records, backend) for backend in BACKENDS}
        # Check every shard before writing any experiment result.
        for backend_jobs in jobs.values():
            for job in backend_jobs:
                _read_cache(job, required=True)
        reference_panns = _assembled(jobs["panns"], None)
        reference_vggish = _assembled(jobs["vggish"], None)["patches"]
        caption_text = _assembled(jobs["clap"], "caption", kind="text")["text_embedding"]
        rows = []
        for model in config.get("model_kinds", MODELS):
            for input_name in config.get("input_kinds", INPUTS):
                group = (model, input_name)
                panns = _assembled(jobs["panns"], group)
                vggish = _assembled(jobs["vggish"], group)["patches"]
                aesthetics = _assembled(jobs["aesthetics"], group)
                audio_embedding = _assembled(jobs["clap"], group)["audio_embedding"]
                prompt_text = _assembled(jobs["clap"], input_name, kind="text")["text_embedding"]
                align = _cosine_rows(audio_embedding, caption_text)
                align_prompt = _cosine_rows(audio_embedding, prompt_text)
                metrics = calculate_metrics(reference_panns["embedding"], panns["embedding"],
                    reference_panns["logits"], panns["logits"], reference_vggish, vggish,
                    aesthetics["PC"], aesthetics["PQ"], align, align_prompt)
                row = {"model": model, "input": input_name, "count": len(records), **metrics}
                rows.append(row)
                kl, kl_sigmoid = _paired_kl_rows(reference_panns["logits"], panns["logits"])
                output = root / "per_sample" / f"{model}_{input_name}.jsonl"
                with _atomic_text(output) as stream:
                    for index, record in enumerate(records):
                        sample = {"sample_id": record["sample_id"], "row_index": record["row_index"],
                                  "model": model, "input": input_name,
                                  **{key: float(aesthetics[key][index]) for key in ("PC", "PQ", "CE", "CU")},
                                  "Align": float(align[index]), "Align_prompt": float(align_prompt[index]),
                                  "KL": float(kl[index]), "KL_sigmoid": float(kl_sigmoid[index])}
                        stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                _log("metrics_group", **row)
        with _atomic_text(root / "metrics.json") as stream:
            json.dump(rows, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
        fields = ["model", "input", "count", "KL", "IS", "FD", "FAD", "PC", "PQ", "Align",
                  "IS_std", "KL_sigmoid", "Align_prompt"]
        with _atomic_text(root / "metrics.csv") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        _log("aggregation_complete", groups=len(rows), samples_per_group=len(records), result_root=str(root))
        return rows


def run(config, *, backend: str = "all", device: str = "cuda:0"):
    """Public pipeline entry point; aggregate is CPU-only."""
    config = _configuration(config)
    if backend == "aggregate":
        return aggregate(config)
    if backend == "all":
        for name in BACKENDS:
            run_backend(config, name, device)
        return aggregate(config)
    return run_backend(config, backend, device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=(*BACKENDS, "all", "aggregate"), default="all")
    arguments = parser.parse_args(argv)
    run(arguments.config, backend=arguments.backend, device=arguments.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
