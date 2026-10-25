"""Atomic LoRA checkpoints with optimizer and random state for exact resumption."""
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


import json
import os
from pathlib import Path
import random
import shutil

import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)


def checkpoint_complete(path):
    """Require every file needed for adapter loading and exact resumption."""
    path = Path(path)
    required = ("trainer_state.json", "training_config.json", "optimizer.pt",
                "rng_state.pt", "adapter/adapter_model.safetensors",
                "adapter/adapter_config.json")
    return path.is_dir() and all((path / name).is_file() for name in required)


def latest_checkpoint(output_dir):
    paths = [path for path in Path(output_dir).glob("checkpoint-*")
             if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
             and checkpoint_complete(path)]
    return max(paths, key=lambda path: int(path.name.split("-")[-1]), default=None)


def save_checkpoint(decoder, optimizer, state, config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"checkpoint-{state['global_step']}"
    if destination.exists():
        if checkpoint_complete(destination):
            return destination
        raise FileExistsError(f"Existing checkpoint is incomplete: {destination}")
    temporary = output / f".checkpoint-{state['global_step']}.{os.getpid()}.tmp"
    temporary.mkdir()
    decoder.save_pretrained(temporary / "adapter", safe_serialization=True)
    torch.save(cpu_tree(optimizer.state_dict()), temporary / "optimizer.pt")
    torch.save({"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()}, temporary / "rng_state.pt")
    atomic_json(temporary / "trainer_state.json", state)
    atomic_json(temporary / "training_config.json", config)
    if config.get("run_dir"):
        calibration = Path(config["run_dir"]) / "reward_calibration.json"
        if calibration.is_file():
            shutil.copyfile(calibration, temporary / "reward_calibration.json")
    temporary.rename(destination)
    checkpoints = sorted((path for path in output.glob("checkpoint-*")
                          if path.name.removeprefix("checkpoint-").isdigit()
                          and checkpoint_complete(path)),
                         key=lambda path: int(path.name.split("-")[-1]))
    for path in checkpoints[:-config.get("save_total_limit", 5)]:
        shutil.rmtree(path)
    return destination


def restore_checkpoint(path, decoder, optimizer):
    path = Path(path)
    if not checkpoint_complete(path):
        raise FileNotFoundError(f"Checkpoint is incomplete: {path}")
    weights = load_file(path / "adapter/adapter_model.safetensors", device="cpu")
    result = set_peft_model_state_dict(decoder, weights, adapter_name="default")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA checkpoint tensors: {result.unexpected_keys}")
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    rng = torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return json.loads((path / "trainer_state.json").read_text())
