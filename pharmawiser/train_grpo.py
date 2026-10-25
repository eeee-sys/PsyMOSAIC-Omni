"""Fixed initial difficulty bins, dynamic group choice and standard token GRPO.

Six candidates per patient, group-relative reward advantages, detached rollout
log probabilities, token-level clipped policy surrogate and KL to a separately
loaded frozen SFT adapter. The three positive reward terms are shared with data
validation; there is no formatting bonus, synthetic advantage or test selection.
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


import math
from pathlib import Path
import random
import time

from common import load_json, write_json
from training_utils import (ROOT, activate_adapter, append_event, arguments, completion_logps,
    fingerprint, latest_checkpoint, load_model, make_optimizer, optimizer_step, read_checkpoint,
    restore_rng, sample_completions, save_checkpoint, seed_everything, setup_device, sha256,
    tensor_adapter_hash, tokenizer_for, trainable_report, training_rows)


def evaluate_rows(model, tokenizer, rows, config, run_dir, tag):
    settings = config["grpo"]
    results = {}
    for row in rows:
        samples = sample_completions(model, tokenizer, row, int(settings.get("M", 6)), config)
        rewards = [float(item["reward"]) for item in samples]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in rewards):
            raise RuntimeError("Reward is outside the normalized [0,1] interval")
        result = {"case_id": row["case_id"], "mean_reward": sum(rewards) / len(rewards),
                  "pass_rate": sum(x >= float(settings.get("pass_reward", .8)) for x in rewards) / len(rewards),
                  "samples": samples}
        results[row["case_id"]] = result
        write_json(run_dir / "evaluations" / tag / f"{row['case_id']}.json", result)
        append_event(run_dir / "evaluation_progress.jsonl", {"tag": tag, "case_id": row["case_id"],
                     "done": len(results), "total": len(rows), "mean_reward": result["mean_reward"]})
    return results


def fixed_groups(initial_results, count=3):
    ordered = sorted(initial_results, key=lambda cid: (1 - initial_results[cid]["mean_reward"],
                                                      int(cid.removeprefix("Case"))))
    if len(ordered) % count:
        raise ValueError("Initial equal-sized quantile groups require divisible case count")
    width = len(ordered) // count
    return [ordered[start:start + width] for start in range(0, len(ordered), width)]


def group_rates(groups, evaluations):
    return [sum(evaluations[cid]["pass_rate"] for cid in group) / len(group) for group in groups]


def select_group(rates, lower=.2, upper=.8):
    eligible = [i for i, rate in enumerate(rates) if lower <= rate < upper]
    if eligible:
        return max(eligible, key=lambda i: (rates[i], -i)), "zpd"
    unfinished = [i for i, rate in enumerate(rates) if rate < upper]
    if unfinished:
        return max(unfinished, key=lambda i: (rates[i], -i)), "highest_unmastered_fallback"
    return None, "all_mastered"


def grpo_candidate_loss(logps, old_logps, reference_logps, advantage, clip_epsilon, beta, completion_mask=None):
    """Per-completion mean, then the caller averages the six completions equally."""
    import torch
    ratio = torch.exp(logps - old_logps.detach())
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    # Nonnegative k3 estimator used by standard GRPO; reference is frozen SFT.
    difference = reference_logps.detach() - logps
    kl = torch.exp(difference) - difference - 1
    mask = torch.ones_like(logps) if completion_mask is None else completion_mask.to(logps)
    denominator = mask.sum().clamp_min(1)
    return ((-surrogate + beta * kl) * mask).sum() / denominator, (kl * mask).sum().detach() / denominator, (ratio * mask).sum().detach() / denominator


def group_advantages(values, device="cpu"):
    """Exactly equal rewards are a skip, even if float32 reduction has roundoff.

    For example std(float32([0.3]*6)) can be 3.26e-8 although no preference
    exists. Check Python reward equality first and normalize in float64.
    """
    import torch
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Each reward must be finite and normalized to [0,1]")
    rewards = torch.tensor(values, dtype=torch.float64, device=device)
    if max(values) - min(values) <= 1e-12:
        return rewards, torch.zeros((), dtype=torch.float64, device=device), None
    std = rewards.std(unbiased=True)
    if float(std) <= 1e-12:
        return rewards, std, None
    advantages = ((rewards - rewards.mean()) / (std + 1e-4)).to(torch.float32).detach()
    return rewards, std, advantages


def main():
    args = arguments(__doc__)
    config = load_json(args.config)
    settings = config["grpo"]
    run_dir = Path(config.get("run_dir", ROOT / "run")) / "grpo"
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_for(config)
    rows, audit = training_rows(config, tokenizer)
    write_json(run_dir / "data_audit.json", audit)
    if int(settings.get("M", 6)) != 6 or int(settings.get("G", 6)) != 6 or int(settings.get("n_groups", 3)) != 3:
        raise ValueError("This experiment requires M=G=6 and three fixed difficulty groups")
    if args.audit_only:
        print("CPU GRPO audit passed: Case1–Case30 only, M=G=6, 3 groups; test not loaded", flush=True)
        return
    setup_device(args.gpu)
    import torch
    seed_everything(int(config.get("seed", 42)))
    reference_path = Path(config.get("sft_reference_adapter", Path(config["sft_output_dir"]) / "final_adapter"))
    reference_file = reference_path / "adapter_model.safetensors"
    if not reference_file.is_file():
        raise ValueError("Complete SFT final adapter is required before GRPO")
    reference_file_hash = sha256(reference_file)
    output_dir = Path(config["grpo_output_dir"])
    identity = fingerprint(config)
    resume = latest_checkpoint(output_dir, args.resume)
    state = read_checkpoint(resume, identity) if resume else {"global_step": 0, "attempts": 0,
        "zero_variance_skips": 0, "fingerprint": identity, "stage": "grpo", "groups": None,
        "rates": None, "mastered_streak": 0, "last_evaluation_step": -1,
        "reference_adapter_sha256": reference_file_hash, "group_sampling_cycles": {}}
    if state["reference_adapter_sha256"] != reference_file_hash:
        raise ValueError("Frozen SFT reference changed since the resume checkpoint")
    model = load_model(config, adapter_path=resume or reference_path, reference_path=reference_path)
    reference_tensor_hash = tensor_adapter_hash(model, "frozen_sft_reference")
    write_json(run_dir / "model_setup.json", {**trainable_report(model),
        "reference_is_frozen_sft_adapter": True, "reference_adapter": str(reference_path),
        "reference_adapter_sha256": reference_file_hash, "reference_tensor_sha256": reference_tensor_hash,
        "disable_adapter_used_as_reference": False, "policy_initialization": str(resume or reference_path)})
    maximum = int(settings.get("max_updates", 200))
    optimizer, scheduler = make_optimizer(model, settings, maximum)
    if resume:
        optimizer.load_state_dict(state.pop("optimizer"))
        scheduler.load_state_dict(state.pop("scheduler"))
        restore_rng(state.pop("rng"))
    started = time.time()
    if state["groups"] is None:
        initial = evaluate_rows(model, tokenizer, rows, config, run_dir, "initial_sft_difficulty")
        state["groups"] = fixed_groups(initial, int(settings.get("n_groups", 3)))
        state["rates"] = group_rates(state["groups"], initial)
        state["initial_difficulty"] = {cid: 1 - item["mean_reward"] for cid, item in initial.items()}
        state["last_evaluation_step"] = 0
        state["mastered_streak"] = int(all(rate >= float(settings.get("zpd_high", .8)) for rate in state["rates"]))
        write_json(run_dir / "fixed_difficulty_groups.json", {"groups": state["groups"],
            "initial_difficulty": state["initial_difficulty"], "immutable_groups": True,
            "difficulty_definition": "1-mean(0.4*TCAS+0.3*MMS+0.3*RCR) over M=6 SFT samples"})
        save_checkpoint(model, tokenizer, optimizer, scheduler, state, output_dir,
                        int(settings.get("save_total_limit", 5)))
    row_by_id = {row["case_id"]: row for row in rows}
    stop_reason = "max_updates"
    consecutive_confirmations = int(settings.get("mastered_confirmations", 2))
    evaluation_every = int(settings.get("reevaluate_every", 10))
    maximum_attempts = int(settings.get("max_attempts", 1000))
    while state["global_step"] < maximum and state["attempts"] < maximum_attempts:
        # Independent confirmation is required even if every group is mastered
        # initially; it does not fabricate a policy update just to reach step10.
        need_confirmation = all(rate >= float(settings.get("zpd_high", .8)) for rate in state["rates"])
        due = state["global_step"] - state["last_evaluation_step"] >= evaluation_every
        if due or (need_confirmation and state["mastered_streak"] < consecutive_confirmations):
            tag = f"step-{state['global_step']}-attempt-{state['attempts']}-confirm-{state['mastered_streak']}"
            evaluation = evaluate_rows(model, tokenizer, rows, config, run_dir, tag)
            state["rates"] = group_rates(state["groups"], evaluation)
            state["last_evaluation_step"] = state["global_step"]
            state["mastered_streak"] = state["mastered_streak"] + 1 if all(
                rate >= float(settings.get("zpd_high", .8)) for rate in state["rates"]) else 0
            append_event(run_dir / "metrics.jsonl", {"event": "group_reevaluation", "global_step": state["global_step"],
                "rates": state["rates"], "mastered_streak": state["mastered_streak"]})
            save_checkpoint(model, tokenizer, optimizer, scheduler, state, output_dir,
                            int(settings.get("save_total_limit", 5)))
        if state["mastered_streak"] >= consecutive_confirmations:
            stop_reason = "all_groups_mastered_twice"
            break
        selected, basis = select_group(state["rates"], float(settings.get("zpd_low", .2)),
                                       float(settings.get("zpd_high", .8)))
        if selected is None:
            raise RuntimeError("All groups mastered but confirmation/stop logic did not resolve")
        # Visit every case once in a freshly shuffled cycle inside the chosen
        # fixed group, rather than repeatedly selecting one convenient patient.
        key = str(selected)
        queue = state["group_sampling_cycles"].get(key, [])
        if not queue:
            queue = list(state["groups"][selected])
            random.shuffle(queue)
        case_id = queue.pop(0)
        state["group_sampling_cycles"][key] = queue
        row = row_by_id[case_id]
        state["attempts"] += 1
        samples = None
        for retry in range(int(settings.get("zero_variance_retries", 2)) + 1):
            sampled = sample_completions(model, tokenizer, row, 6, config)
            rewards, std, advantages = group_advantages([item["reward"] for item in sampled], device="cuda")
            write_json(run_dir / "rollouts" / f"attempt-{state['attempts']}-retry-{retry}.json",
                       {"case_id": case_id, "group": selected, "selection_basis": basis,
                        "global_step_before": state["global_step"], "samples": sampled})
            if advantages is not None:
                samples = sampled
                break
        if samples is None:
            state["zero_variance_skips"] += 1
            append_event(run_dir / "metrics.jsonl", {"event": "zero_variance_skipped", "case_id": case_id,
                         "attempts": state["attempts"], "global_step": state["global_step"],
                         "reward_mean": float(rewards.mean()), "group": selected})
            # Persist finite attempt accounting even when no gradients exist.
            if state["attempts"] % int(settings.get("save_steps", 10)) == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, state, output_dir,
                                int(settings.get("save_total_limit", 5)))
            continue
        temperature = float(settings.get("temperature", .8))
        projection_chunk = int(config.get("projection_chunk_size", 64))
        old_logps, reference_logps = [], []
        model.eval()
        with torch.no_grad():
            activate_adapter(model, "default")
            for sample in samples:
                old_logps.append(completion_logps(model, row["prompt_ids"], sample["completion_ids"],
                                                  projection_chunk, temperature).detach())
            activate_adapter(model, "frozen_sft_reference")
            for sample in samples:
                reference_logps.append(completion_logps(model, row["prompt_ids"], sample["completion_ids"],
                                                        projection_chunk, temperature).detach())
        activate_adapter(model, "default")
        losses, kls, ratios = [], [], []
        # Reusing a rollout for PPO epochs is supported, although this run's
        # default is one. Every actual optimizer update counts toward the budget.
        for ppo_epoch in range(int(settings.get("ppo_epochs", 1))):
            if state["global_step"] >= maximum:
                break
            optimizer.zero_grad(set_to_none=True)
            model.train()  # dropout modules are explicitly p=0 for GRPO
            for index, sample in enumerate(samples):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logps = completion_logps(model, row["prompt_ids"], sample["completion_ids"],
                                             projection_chunk, temperature)
                    loss, kl, ratio = grpo_candidate_loss(logps, old_logps[index], reference_logps[index],
                        advantages[index], float(settings.get("clip_epsilon", .2)), float(settings.get("beta", .04)),
                        torch.tensor(sample["completion_mask"], device=logps.device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite GRPO objective")
                (loss / 6).backward()
                losses.append(float(loss.detach()))
                kls.append(float(kl))
                ratios.append(float(ratio))
                del loss, logps
            state["global_step"] += 1
            health = optimizer_step(model, optimizer, scheduler, float(settings.get("max_grad_norm", 1.)),
                                    run_dir / "first_optimizer_update.json", state["global_step"])
            event = {"stage": "grpo", "status": "training", "global_step": state["global_step"],
                     "max_steps": maximum, "attempts": state["attempts"], "case_id": case_id,
                     "selected_group": selected, "selection_basis": basis, "group_rates": state["rates"],
                     "reward": float(rewards.mean()), "reward_std": float(std),
                     "advantages": advantages.cpu().tolist(), "loss": sum(losses) / len(losses),
                     "kl": sum(kls) / len(kls), "ratio": sum(ratios) / len(ratios),
                     "health": health, "elapsed_seconds": time.time() - started,
                     "gpu_allocated_mib": torch.cuda.memory_allocated() / 1024**2}
            append_event(run_dir / "metrics.jsonl", event)
            write_json(run_dir / "status.json", event)
            if state["global_step"] % int(settings.get("save_steps", 10)) == 0:
                save_checkpoint(model, tokenizer, optimizer, scheduler, state, output_dir,
                                int(settings.get("save_total_limit", 5)))
        del old_logps, reference_logps, samples
    if state["attempts"] >= maximum_attempts and state["global_step"] < maximum:
        stop_reason = "bounded_attempt_budget_exhausted"
    if tensor_adapter_hash(model, "frozen_sft_reference") != reference_tensor_hash:
        raise RuntimeError("The frozen SFT reference adapter changed during GRPO")
    save_checkpoint(model, tokenizer, optimizer, scheduler, state, output_dir,
                    int(settings.get("save_total_limit", 5)))
    final = output_dir / "final_adapter"
    activate_adapter(model, "default")
    model.save_pretrained(final, safe_serialization=True, selected_adapters=["default"], save_embedding_layers=False)
    tokenizer.save_pretrained(final)
    write_json(run_dir / "status.json", {"status": "completed" if stop_reason != "bounded_attempt_budget_exhausted" else "stopped_attempt_budget",
        "stop_reason": stop_reason, "global_step": state["global_step"], "attempts": state["attempts"],
        "zero_variance_skips": state["zero_variance_skips"], "rates": state["rates"],
        "final_adapter": str(final), "reference_adapter_unchanged": True,
        "test_loaded": False, "elapsed_seconds": time.time() - started, "fingerprint": identity})


if __name__ == "__main__":
    main()
