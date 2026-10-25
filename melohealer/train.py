#!/usr/bin/env python3
"""Train ACE-Step DiT LoRA with caption-only, deterministic Neighbor GRPO."""
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
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from music_training.ace_runtime import AceRuntime
from music_training.checkpointing import (append_jsonl, atomic_json, latest_checkpoint,
                                           optimizer_to, restore_checkpoint, save_checkpoint)
from music_training.dataset import prepare_dataset
from music_training.neighbor import (group_advantages, neighbor_grpo_loss,
                                     sample_correlated_noise)
from music_training.reward import CMIReward

from project_paths import load_config, OUTPUT_ROOT

ROOT = Path(__file__).resolve().parent
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def event(name, **values):
    print(json.dumps({"event": name, "time_utc": utc_now(), **values},
                     allow_nan=False), flush=True)


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    complete = data.rfind(b"\n") + 1
    if complete != len(data):
        with path.open("r+b") as handle:
            handle.truncate(complete)
    return [json.loads(line) for line in data[:complete].splitlines() if line.strip()]


def scalar(value):
    return float(value.detach().float().cpu()) if isinstance(value, torch.Tensor) else float(value)


def set_status(config, status, **kwargs):
    value = {"status": status, "pid": os.getpid(), "updated_at_utc": utc_now(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "max_steps": config["max_steps"], **kwargs}
    atomic_json(Path(config["run_dir"]) / "status.json", value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def verify_output_directory(directory):
    """Check real checkpoint-directory I/O before loading any model weights."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=".write-preflight-", dir=directory)
    temporary = Path(filename)
    payload = b"PsyMOSAIC checkpoint output write preflight\n"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.read_bytes() != payload:
            raise OSError(f"Output-directory write verification failed: {directory}")
    finally:
        # Only remove the unique file created by this check.
        temporary.unlink(missing_ok=True)
    result = {"output_dir": str(directory), "checked_at_utc": utc_now(),
              "write_fsync_read_delete": "passed"}
    event("output_directory_write_check_passed", **result)
    return result


def time_grid(config):
    times = torch.linspace(1, 0, config["sampling_steps"] + 1, dtype=torch.float32)
    shift = config["timestep_shift"]
    return shift * times / (1 + (shift - 1) * times)


def cached_condition(runtime, row, config):
    directory = Path(config["run_dir"]) / "conditions"
    directory.mkdir(exist_ok=True)
    path = directory / f"row-{row['row_id']}.pt"
    caption_hash = hashlib.sha256(row["caption"].encode()).hexdigest()
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["caption_sha256"] != caption_hash:
            raise ValueError("Cached condition belongs to a different caption")
        return cached["condition"]
    # One frozen LM plan per caption, reused by all neighbors and selected anchors.
    started = time.monotonic()
    event("planning_started", row_id=row["row_id"])
    plan_path = directory / f"row-{row['row_id']}_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["caption"] != row["caption"] or plan["seed"] != config["seed"] + row["row_id"]:
            raise ValueError("Cached LM plan belongs to a different caption or seed")
    else:
        plan = runtime.plan_caption(row["caption"], seed=config["seed"] + row["row_id"])
        atomic_json(plan_path, plan)
    condition = runtime.condition_caption(row["caption"], plan)
    temporary = path.with_suffix(".tmp")
    torch.save({"caption_sha256": caption_hash, "condition": condition}, temporary)
    temporary.replace(path)
    event("planning_complete", row_id=row["row_id"], elapsed_seconds=time.monotonic() - started)
    return condition


def rollout(runtime, condition, config, group_index):
    started = time.monotonic()
    event("rollout_started", group_index=group_index)
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 100003 * group_index)
    shape = tuple(condition.context_latents.shape[-2:-1]) + (64,)
    noise = sample_correlated_noise(config["num_generations"], shape,
                                    sigma=config["neighbor_sigma"], generator=generator,
                                    device="cpu", dtype=torch.float32)
    times = time_grid(config)
    states = [noise.to(runtime.device)]
    runtime.activate_decoder(training=False)
    with torch.no_grad():
        for index in range(len(times) - 1):
            delta = float(times[index] - times[index + 1])
            following = []
            for current in states[-1].split(config.get("rollout_batch_size", 1)):
                velocity = runtime.velocity(current, float(times[index]), condition, grad=False)
                following.append((current - delta * velocity.float()).detach())
            states.append(torch.cat(following))
    trajectory = torch.stack(states).cpu()
    event("rollout_complete", group_index=group_index,
          elapsed_seconds=time.monotonic() - started)
    return trajectory, times


def score_rollout(runtime, reward, row, trajectory, config):
    started = time.monotonic()
    event("decoding_started", row_id=row["row_id"])
    waveforms = []
    for latent in trajectory[-1].split(config.get("vae_batch_size", 1)):
        waveform = runtime.decode(latent)
        if isinstance(waveform, tuple):
            waveform = waveform[0]
        waveforms.extend(waveform.detach().float().cpu().unbind(0))
    runtime.offload()
    event("reward_scoring_started", row_id=row["row_id"],
          decoding_seconds=time.monotonic() - started)
    started = time.monotonic()
    scores = reward.score(waveforms, [row["caption"]] * len(waveforms),
                          source_sample_rate=48000, batch_size=config.get("reward_batch_size", 1))
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(waveforms), 2) or not np.isfinite(scores).all():
        raise ValueError("CMI-RM must return finite alignment/quality scores for every neighbor")
    event("reward_scoring_complete", row_id=row["row_id"],
          elapsed_seconds=time.monotonic() - started)
    return scores, waveforms


def save_audio(waveforms, destination):
    import soundfile as sf
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for index, waveform in enumerate(waveforms):
        sf.write(destination / f"candidate-{index:02d}.wav", waveform.numpy().T,
                 48000, subtype="PCM_16")


def update(runtime, optimizer, condition, trajectory, times, rewards, config, group_index):
    advantage = group_advantages(torch.as_tensor(rewards, dtype=torch.float32),
                                  p=config["quasi_norm_p"],
                                  std_epsilon=config["advantage_epsilon"])
    if advantage.skip_group:
        return {"skipped": True, "reason": "zero_group_advantage"}
    runtime.activate_decoder(training=True)
    parameters = [(name, parameter) for name, parameter in runtime.decoder.named_parameters()
                  if parameter.requires_grad]
    if not parameters or any("lora_" not in name for name, _ in parameters):
        raise RuntimeError("Only DiT LoRA parameters may be optimized")
    optimizer_to(optimizer, "cuda")
    optimizer.zero_grad(set_to_none=True)
    sampler = random.Random(config["seed"] + group_index * 15485863)
    anchors = sampler.sample(range(config["num_generations"]), config["anchor_count"])
    selected_times = sampler.sample(range(config["sampling_steps"]),
                                    config["training_timesteps_per_group"])
    losses, entropies, clipping, log_ratio = [], [], [], []
    if config.get("batched_anchor_updates", False):
        # Use the same full-group batch shape as the rollout. This keeps old
        # and current BF16 kernels consistent when forming raw-L2 ratios.
        if config.get("rollout_batch_size", 1) != config["num_generations"]:
            raise ValueError("Batched anchor updates require full-group rollout batches")
        update_batches = [(anchors, index, True) for index in selected_times]
    else:
        update_batches = [([anchor], index, False) for anchor in anchors for index in selected_times]
    update_started = time.monotonic()
    for selected_anchors, index, full_group in update_batches:
        current = (trajectory[index] if full_group else
                   trajectory[index, selected_anchors]).to("cuda")
        velocity = runtime.velocity(current, float(times[index]), condition, grad=True)
        predicted = current - float(times[index] - times[index + 1]) * velocity.float()
        if full_group:
            predicted = predicted[selected_anchors]
        candidates = trajectory[index + 1].to("cuda")
        # There is exactly one optimizer step after every anchor/time backward.
        # All predictions here therefore still use theta_old. Reuse its detached
        # value for the denominator so BF16 no-grad/grad kernel differences do
        # not create a spurious off-policy ratio before the first update.
        old_anchor = (predicted.detach() if full_group else
                      trajectory[index + 1, selected_anchors].to("cuda"))
        result = neighbor_grpo_loss(predicted, old_anchor, candidates,
                                     advantage.reweighted.to("cuda"),
                                     clip_epsilon=config["clip_epsilon"])
        loss = result.loss / len(update_batches)
        if not torch.isfinite(loss):
            raise FloatingPointError("Neighbor GRPO loss is not finite")
        loss.backward()
        losses.append(scalar(result.loss))
        entropies.append(scalar(result.entropy.mean()))
        clipping.append(scalar(result.clip_fraction))
        log_ratio.append(scalar((result.log_probs - result.old_log_probs).abs().max()))
        del result, loss, predicted, velocity, current, candidates, old_anchor
    present = [parameter for _, parameter in parameters if parameter.grad is not None]
    if not present or not all(torch.isfinite(parameter.grad).all() for parameter in present):
        raise FloatingPointError("LoRA gradients are absent or not finite")
    grad_norm = scalar(torch.nn.utils.clip_grad_norm_(present, config["max_grad_norm"]))
    if grad_norm == 0:
        raise FloatingPointError("A nonzero-advantage group produced zero LoRA gradients")
    audit_name, audit_parameter = next((name, parameter) for name, parameter in parameters
                                       if "lora_B" in name and parameter.grad is not None)
    before = audit_parameter.detach().clone()
    optimizer.step()
    delta = scalar((audit_parameter.detach() - before).norm())
    optimizer.zero_grad(set_to_none=True)
    return {"skipped": False, "loss": float(np.mean(losses)), "grad_norm": grad_norm,
            "lora_parameter_delta_l2": delta, "audited_parameter": audit_name,
            "gradient_tensors": len(present), "trainable_tensors": len(parameters),
            "trainable_parameters": sum(parameter.numel() for _, parameter in parameters),
            "neighbor_entropy": float(np.mean(entropies)), "clip_fraction": float(np.mean(clipping)),
            "optimizer_update_seconds": time.monotonic() - update_started,
            "maximum_abs_log_ratio": max(log_ratio), "anchors": anchors,
            "training_timestep_indices": selected_times}


def calibrate(runtime, reward, rows, config):
    run = Path(config["run_dir"])
    path = run / "reward_calibration.json"
    if path.exists():
        reward.load_calibration(path)
        return
    ledger = run / "calibration_scores.jsonl"
    cached = {row["row_id"]: row for row in read_rows(ledger)}
    if not set(cached) <= {row["row_id"] for row in rows}:
        raise ValueError("Calibration ledger has unexpected caption rows")
    for index, row in enumerate(rows):
        if row["row_id"] in cached:
            continue
        set_status(config, "calibrating", calibration_completed=len(cached),
                   calibration_total=len(rows), current_row_id=row["row_id"])
        condition = cached_condition(runtime, row, config)
        trajectory, _ = rollout(runtime, condition, config, group_index=-index - 1)
        scores, audio = score_rollout(runtime, reward, row, trajectory, config)
        value = {"row_id": row["row_id"], "scores_alignment_quality": scores.tolist()}
        append_jsonl(ledger, value)
        cached[row["row_id"]] = value
        if index == 0:
            save_audio(audio, Path(config.get("sample_dir", OUTPUT_ROOT / "melohealer" / "samples")) / "calibration_initial")
        del trajectory, audio, condition
        if STOP_REQUESTED:
            set_status(config, "stopped_during_calibration", calibration_completed=len(cached))
            raise SystemExit(0)
    all_scores = np.asarray([score for row in rows
                             for score in cached[row["row_id"]]["scores_alignment_quality"]])
    calibration = reward.fit_calibration(all_scores)
    atomic_json(path, calibration)
    print(json.dumps({"event": "reward_calibration_complete", "samples": len(all_scores),
                      "calibration": calibration}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run = Path(config["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output_preflight = verify_output_directory(config["output_dir"])
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    torch.set_num_threads(8)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    set_status(config, "preparing", preflight=args.preflight,
               output_write_preflight=output_preflight)
    data_dir = Path(config["data_dir"])
    if not (data_dir / "train_prompts.jsonl").exists():
        prepare_dataset(config["source_data"], data_dir, seed=config["seed"],
                        calibration_size=config["calibration_prompts"],
                        train_size=config["training_prompts"])
    train_rows = read_rows(data_dir / "train_prompts.jsonl")
    calibration_rows = read_rows(data_dir / "calibration_prompts.jsonl")
    runtime = AceRuntime(config["base_model"], config["ace_code_root"], device="cuda",
                         duration=config["duration_seconds"], lora_rank=config["lora_rank"],
                         lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                         keep_models_on_device=config.get("keep_models_on_device", False))
    runtime.load()
    reward = CMIReward(config["reward_model"], config["reward_code_root"],
                       config["reward_assets"], device="cuda", max_duration=None,
                       keep_model_on_device=config.get("keep_models_on_device", False))
    parameters = [parameter for parameter in runtime.decoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], foreach=False)
    state = {"global_step": 0, "group_index": 0, "skipped_groups": 0,
             "started_at_utc": utc_now()}
    if args.resume:
        checkpoint = latest_checkpoint(config["output_dir"]) if args.resume == "auto" else Path(args.resume)
        if checkpoint is not None:
            old_config = json.loads((checkpoint / "training_config.json").read_text())
            for key in ("base_model", "reward_model", "seed", "lora_rank", "lora_alpha",
                        "duration_seconds", "num_generations", "neighbor_sigma", "quasi_norm_p"):
                if old_config[key] != config[key]:
                    raise ValueError(f"Resume protocol differs: {key}")
            state = restore_checkpoint(checkpoint, runtime.decoder, optimizer)
            if state["global_step"] > 0:
                saved_calibration = checkpoint / "reward_calibration.json"
                active_calibration = run / "reward_calibration.json"
                if saved_calibration.exists():
                    saved_values = json.loads(saved_calibration.read_text())
                    if active_calibration.exists() and json.loads(active_calibration.read_text()) != saved_values:
                        raise ValueError("The reward calibration differs from the resumed checkpoint")
                    atomic_json(active_calibration, saved_values)
                elif not active_calibration.exists():
                    raise FileNotFoundError("Resuming trained weights requires the original reward calibration")
            print(json.dumps({"event": "resumed", "checkpoint": str(checkpoint), **state}), flush=True)
    elif latest_checkpoint(config["output_dir"]) is not None and not args.preflight:
        raise RuntimeError("Existing checkpoints require explicit --resume")

    if args.preflight:
        plan_each = config.get("preflight_plan_each_caption", False)
        if plan_each:
            runtime._load_lm()
        reward.load()
        calibration_path = run / "reward_calibration.json"
        if calibration_path.exists():
            reward.load_calibration(calibration_path)
        cycles = []
        for iteration in range(config.get("preflight_iterations", 1)):
            row = calibration_rows[iteration]
            started = time.monotonic()
            if plan_each:
                plan = runtime.plan_caption(row["caption"], config["seed"] + row["row_id"])
                condition = runtime.condition_caption(row["caption"], plan)
            else:
                condition = cached_condition(runtime, row, config)
            trajectory, times = rollout(runtime, condition, config, group_index=-iteration - 1)
            scores, audio = score_rollout(runtime, reward, row, trajectory, config)
            # Isolated optimizer tests; these weights are never committed.
            test_rewards = (reward.normalized_reward(scores)[0] if reward.calibration is not None
                            else scores.mean(axis=1))
            metrics = update(runtime, optimizer, condition, trajectory, times,
                             test_rewards, config, group_index=-iteration - 1)
            if metrics.get("skipped") or metrics["lora_parameter_delta_l2"] == 0:
                raise RuntimeError("Preflight did not produce a real LoRA optimizer update")
            if config.get("batched_anchor_updates", False) and metrics["maximum_abs_log_ratio"] > 1e-6:
                raise RuntimeError("The pre-update on-policy probability ratio must equal one")
            cycle = {"iteration": iteration, "elapsed_seconds": time.monotonic() - started,
                     "row_id": row["row_id"], **metrics}
            cycles.append(cycle)
            event("preflight_cycle_complete", **cycle)
            if iteration == 0 and config.get("keep_models_on_device", False):
                serial_scores = reward.score(audio, [row["caption"]] * len(audio),
                                             source_sample_rate=48000, batch_size=1)
                score_difference = float(np.max(np.abs(scores - serial_scores)))
                if not np.allclose(scores, serial_scores, rtol=0.02, atol=0.01):
                    raise RuntimeError(f"Batched CMI score mismatch: max difference {score_difference}")
                serial_audio = runtime.decode(trajectory[-1, :1])[0]
                audio_difference = float((serial_audio - audio[0]).norm() /
                                         serial_audio.norm().clamp_min(1e-8))
                if audio_difference > 1e-3:
                    raise RuntimeError(f"Batched VAE relative error {audio_difference}")
                parity = {"cmi_max_absolute_difference": score_difference,
                          "vae_relative_l2_difference": audio_difference}
                event("batch_parity_passed", **parity)
        result = {"status": "passed", "scores_alignment_quality": scores.tolist(), **metrics,
                  "cycles": cycles, "checkpoint_step": state["global_step"],
                  "batch_parity": parity if config.get("keep_models_on_device", False) else None,
                  "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                  "production_weights_changed": False, "train_updates_committed": 0}
        atomic_json(run / "preflight.json", result)
        save_audio(audio, Path(config.get("sample_dir", OUTPUT_ROOT / "melohealer" / "samples")) / "preflight")
        set_status(config, "preflight_passed", preflight_result=result)
        return

    calibrate(runtime, reward, calibration_rows, config)
    clock_started = time.monotonic()
    starting_step = state["global_step"]
    while state["global_step"] < config["max_steps"]:
        if state["group_index"] >= config.get("max_attempted_groups", 2 * config["max_steps"]):
            raise RuntimeError("Too many zero-advantage groups to reach the requested update budget")
        index = state["group_index"]
        row = train_rows[index % len(train_rows)]
        group_started = time.monotonic()
        set_status(config, "training", global_step=state["global_step"],
                   group_index=index, current_row_id=row["row_id"])
        condition = cached_condition(runtime, row, config)
        trajectory, times = rollout(runtime, condition, config, group_index=index)
        scores, audio = score_rollout(runtime, reward, row, trajectory, config)
        normalized, zscores = reward.normalized_reward(scores)
        step = state["global_step"]
        if step < config["warmup_steps"]:
            lr = config["learning_rate"] * (step + 1) / config["warmup_steps"]
        else:
            progress = (step - config["warmup_steps"]) / max(1, config["max_steps"] - config["warmup_steps"])
            lr = config["learning_rate"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        metrics = update(runtime, optimizer, condition, trajectory, times, normalized, config, index)
        state["group_index"] += 1
        if metrics["skipped"]:
            state["skipped_groups"] += 1
        else:
            state["global_step"] += 1
        record = {**state, **metrics, "row_id": row["row_id"], "learning_rate": lr,
                  "step_seconds": time.monotonic() - group_started,
                  "execution_revision": config.get("execution_revision", "serial_offload_v1"),
                  "reward": float(np.mean(normalized)), "reward_std": float(np.std(normalized)),
                  "raw_alignment_mean": float(scores[:, 0].mean()),
                  "raw_quality_mean": float(scores[:, 1].mean()),
                  "scores_alignment_quality": scores.tolist(), "normalized_rewards": normalized.tolist(),
                  "elapsed_seconds": time.monotonic() - clock_started,
                  "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
        updates = state["global_step"] - starting_step
        if updates:
            record["seconds_per_update"] = record["elapsed_seconds"] / updates
            record["estimated_remaining_seconds"] = record["seconds_per_update"] * (config["max_steps"] - state["global_step"])
        append_jsonl(run / "train_metrics.jsonl", record)
        set_status(config, "training", **record)
        if not metrics["skipped"] and (state["global_step"] == starting_step + 1 or
                                          state["global_step"] % config["save_steps"] == 0):
            checkpoint = save_checkpoint(runtime.decoder, optimizer, state, config)
            print(json.dumps({"event": "checkpoint_saved", "path": str(checkpoint)}), flush=True)
        if state["global_step"] and state["global_step"] % config["save_audio_steps"] == 0:
            save_audio(audio, Path(config.get("sample_dir", OUTPUT_ROOT / "melohealer" / "samples")) / f"step-{state['global_step']:06d}")
        del trajectory, audio, condition
        if STOP_REQUESTED:
            save_checkpoint(runtime.decoder, optimizer, state, config)
            set_status(config, "stopped_by_user", **state)
            return
    save_checkpoint(runtime.decoder, optimizer, state, config)
    runtime.offload()
    final_adapter = Path(config["output_dir"]) / "final_adapter"
    runtime.decoder.save_pretrained(final_adapter, safe_serialization=True)
    if config["export_complete_merged_model"]:
        set_status(config, "exporting_complete_model", **state)
        runtime.export_complete(Path(config["output_dir"]) / "final_model")
    set_status(config, "complete", **state, final_adapter=str(final_adapter))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--config" in sys.argv:
            try:
                filename = sys.argv[sys.argv.index("--config") + 1]
                set_status(load_config(filename), "failed",
                           error_type=type(error).__name__, error=str(error))
            except Exception:
                pass
        raise
