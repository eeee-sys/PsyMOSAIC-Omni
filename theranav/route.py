"""Base-only Controller routing of 56 E-DAIC patients in four evidence modes.

Reads existing Generator profiles and existing predicted diagnoses. Official
labels and patient identifiers never enter the model input. Missing diagnoses
remain UNAVAILABLE. There is no training, adapter loading, or diagnosis repair.
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
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODES = ("text", "text_audio", "text_video", "text_audio_video")
ROUTES = ("psychotherapy", "drug_therapy", "musical_treatment")
TARGETS = ("DEPRESSION", "PTSD")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return load_config(path)


def write_text(path, text):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as stream:
        os.chmod(tmp, 0o600);stream.write(text);stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def status(config, phase, **extra):
    value = {"state": phase, "pid": os.getpid(), "time_utc": datetime.now(timezone.utc).isoformat(), **extra}
    write_json(Path(config["run_dir"]) / "status.json", value)
    write_json(Path(config["output_dir"]) / "run/status.json", value)
    print(canonical(value), flush=True)


def source_data(config):
    project = Path(config["diagnosis_project"])
    sys.path.insert(0, str(project / "training"))
    from data import STAGES, sanitize_profile
    source = Path(config["source_results"])
    manifest = read_json(source / "evaluation_manifest.json")
    if manifest["scope"] != "full_official_test":
        raise ValueError("Source is not the full official E-DAIC test evaluation")
    official = [json.loads(s) for s in Path(config["official_test_records"]).read_text().splitlines() if s.strip()]
    ids = sorted((str(r["patient_id"]) for r in official), key=int)
    if len(ids) != 56 or len(set(ids)) != 56 or any(r["split"] != "test" for r in official):
        raise ValueError("Exactly 56 unique official test patients are required")
    if set(ids) != set(manifest["patient_ids"]):
        raise ValueError("Diagnosis result and official test IDs disagree")
    source_rows = [json.loads(s) for s in (source / "diagnosis_predictions.jsonl").read_text().splitlines() if s.strip()]
    diagnoses = {}
    for row in source_rows:
        if row["source"] == "generator":
            key = (str(row["patient_id"]), row["evidence_mode"], row["target_disorder"])
            if key in diagnoses:
                raise ValueError("Duplicate Generator diagnosis key")
            diagnoses[key] = row
    expected = {(pid, mode, target) for pid in ids for mode in MODES for target in TARGETS}
    if set(diagnoses) != expected:
        raise ValueError("Expected all 448 Generator-based target diagnosis results")
    rows, missing, hashes = [], [], {}
    for path in [source / "evaluation_manifest.json", source / "diagnosis_predictions.jsonl",
                 Path(config["official_test_records"]), project / "training/data.py"]:
        hashes[str(path)] = sha(path)
    for pid in ids:
        path = source / "generator_profiles" / f"{pid}.json"
        profile = read_json(path);hashes[str(path)] = sha(path)
        if str(profile["patient_id"]) != pid or profile["profile_error"] is not None:
            raise ValueError("A Generator profile is invalid")
        if set(profile["patient_record"]) != set(STAGES):
            raise ValueError("A Generator profile is missing a clinical stage")
        for mode in MODES:
            clean = sanitize_profile(profile["patient_record"], evidence_mode=mode, dataset="E-DAIC")
            # Match the original evaluator's exact JSON hashing convention.
            profile_hash = hashlib.sha256(json.dumps(clean, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            combined, provenance = {}, {}
            for target in TARGETS:
                original = diagnoses[(pid, mode, target)]
                if original["profile_sha256"] != profile_hash:
                    raise ValueError("Clinical profile does not match the actual original diagnosis input")
                valid = original.get("valid") is True
                if valid:
                    answer, rationale = original["answer"], original["rationale"]
                    if answer not in ("POSITIVE", "NEGATIVE") or not isinstance(rationale, str) or not rationale.strip():
                        raise ValueError("A valid diagnosis has an incomplete decision or reason")
                else:
                    answer = "UNAVAILABLE"
                    rationale = "The preceding diagnosis output did not provide a completed screening decision."
                    missing.append({"patient_id": pid, "evidence_mode": mode, "target_disorder": target,
                                    "source_error": original.get("error")})
                combined[target.lower()] = {"answer": answer, "rationale": rationale}
                provenance[target.lower()] = {"source_row_hash": digest(original), "original_valid": valid}
            model_input = {"patient_record": clean, "diagnosis": combined}
            rows.append({"patient_id": pid, "evidence_mode": mode, "model_input": model_input,
                         "input_sha256": digest(model_input), "diagnosis_provenance": provenance,
                         "generator_profile_sha256": hashes[str(path)]})
    return rows, missing, hashes


def allowed_next(prefix, sequences, eos):
    if eos in prefix:
        return [eos]
    choices = set()
    for seq in sequences:
        if list(seq[:len(prefix)]) == list(prefix):
            choices.add(eos if len(prefix) == len(seq) else seq[len(prefix)])
    if not choices:
        raise RuntimeError("Route decoding left the three-route token trie")
    return sorted(choices)


@contextmanager
def last_token_projection(model):
    def hook(module, inputs):
        if len(inputs) == 1 and inputs[0].ndim == 3:
            return (inputs[0][:, -1:, :],)
        return None
    handle = model.lm_head.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def report(config, rows, predictions, missing, fingerprint):
    output = Path(config["output_dir"]);groups = []
    expected = {(r["patient_id"], r["evidence_mode"]) for r in rows}
    actual = [(r["patient_id"], r["evidence_mode"]) for r in predictions]
    if len(actual) != len(set(actual)) or not set(actual) <= expected:
        raise ValueError("Routing results contain duplicate or unexpected patient/mode keys")
    for mode in MODES:
        chosen = [r for r in predictions if r["evidence_mode"] == mode]
        counts = Counter(r["route"] for r in chosen)
        groups.append({"evidence_mode": mode, "expected_patients": 56, "completed": len(chosen),
                       "counts": {route: counts[route] for route in ROUTES},
                       "proportions": {route: counts[route] / len(chosen) if chosen else None for route in ROUTES}})
    result = {"state": "completed" if len(predictions) == 224 else "partial", "fingerprint": fingerprint,
              "unique_test_patients": 56, "evidence_modes": list(MODES), "expected_choices": 224,
              "completed_choices": len(predictions), "model": config["base_model"],
              "sft_performed": False, "controller_adapter_loaded": False,
              "missing_diagnoses_preserved": missing, "by_mode": groups}
    write_json(output / "routing_summary.json", result)
    stream = io.StringIO();writer = csv.writer(stream)
    writer.writerow(["evidence_mode", "completed", *[part for route in ROUTES for part in (route + "_count", route + "_percent")]])
    lines = ["| Evidence | Completed | Psychotherapy | Drug therapy | Musical treatment |",
             "|---|---:|---:|---:|---:|"]
    for group in groups:
        cells = []
        for route in ROUTES:
            prop = group["proportions"][route]
            cells += [group["counts"][route], prop * 100 if prop is not None else None]
        writer.writerow([group["evidence_mode"], group["completed"], *cells])
        lines.append("| " + " | ".join([group["evidence_mode"], str(group["completed"])] + [
            f"{group['counts'][route]} ({100*group['proportions'][route]:.2f}%)" if group["completed"] else "—" for route in ROUTES]) + " |")
    write_text(output / "routing_summary.csv", stream.getvalue())
    write_text(output / "routing_summary.txt", "\n".join(lines) + "\n")
    ordered = sorted(predictions, key=lambda x: (MODES.index(x["evidence_mode"]), int(x["patient_id"])))
    write_text(output / "routing_predictions.jsonl", "".join(canonical(row) + "\n" for row in ordered))
    return result


def share_owned(output):
    """Keep routing output under the current user's configured output location."""
    Path(output).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/inference.json"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args();config = read_json(args.config)
    if not args.audit_only:
        if args.gpu is None:
            raise ValueError("An explicitly reviewed GPU is required")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output = Path(config["output_dir"]);output.mkdir(parents=True, exist_ok=True)
    run = Path(config["run_dir"]);run.mkdir(parents=True, exist_ok=True)
    rows, missing, hashes = source_data(config)
    prompt_path = ROOT / "prompts/controller_system.txt";system = prompt_path.read_text()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token;tokenizer.padding_side = "left"
    sequence_by_route = {route: tokenizer.encode(route, add_special_tokens=False) for route in ROUTES}
    assert all(tokenizer.decode(ids) == route for route, ids in sequence_by_route.items())
    max_new_tokens = max(map(len, sequence_by_route.values())) + 1
    def messages(row):
        return [{"role": "system", "content": system}, {"role": "user", "content": canonical(row["model_input"])}]
    lengths = [len(tokenizer.apply_chat_template(messages(row), tokenize=True, add_generation_prompt=True)) for row in rows]
    if max(lengths) + max_new_tokens > int(config["context_limit"]):
        raise ValueError("A complete profile exceeds the context limit; no truncation is allowed")
    assets = {"config": config, "sources": hashes, "runtime_sha256": sha(ROOT / "route.py"),
              "prompt_sha256": sha(prompt_path), "base_config_sha256": sha(Path(config["base_model"]) / "config.json"),
              "base_index_sha256": sha(Path(config["base_model"]) / "model.safetensors.index.json"),
              "routes_token_ids": sequence_by_route}
    fingerprint = digest(assets)
    write_json(run / "cpu_audit.json", {"state": "passed", "patients": 56, "routing_inputs": len(rows),
               "reused_diagnoses": 448 - len(missing), "unavailable_diagnoses": missing,
               "min_prompt_tokens": min(lengths), "max_prompt_tokens": max(lengths), "input_fields": ["patient_record", "diagnosis"],
               "ground_truth_in_model_input": False, "segment_ids_in_model_input": False, "fingerprint": fingerprint})
    if args.audit_only:
        print(canonical({"audit": "passed", "choices": len(rows), "missing_diagnoses": len(missing), "max_prompt_tokens": max(lengths)}), flush=True)
        return
    lock = (run / "controller.lock").open("a");fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        manifest_path = output / "run_manifest.json"
        if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
            raise ValueError("Controller configuration, source data, or runtime changed on resume")
        write_json(manifest_path, {"fingerprint": fingerprint, "assets": assets, "decoding": "Greedy, constrained to the three route strings and EOS"})
        write_text(output / "controller_inputs.jsonl", "".join(canonical(row) + "\n" for row in rows))
        predictions, todo = [], []
        for row in rows:
            path = output / "predictions" / row["evidence_mode"] / f"{row['patient_id']}.json"
            if path.exists():
                value = read_json(path)
                if (value["fingerprint"] != fingerprint or value["input_sha256"] != row["input_sha256"] or value["route"] not in ROUTES
                        or value["patient_id"] != row["patient_id"] or value["evidence_mode"] != row["evidence_mode"]):
                    raise ValueError("Cached route identity or answer is invalid")
                predictions.append(value)
            else:
                todo.append(row)
        if todo:
            import torch
            from transformers import GenerationConfig, Qwen2_5OmniThinkerForConditionalGeneration
            torch.manual_seed(int(config["seed"]));torch.cuda.manual_seed_all(int(config["seed"]))
            model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(config["base_model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
            for name in ("audio_tower", "visual"):
                if hasattr(model, name):setattr(model, name, None)
            for p in model.parameters():p.requires_grad_(False)
            assert not any("lora_" in name for name, _ in model.named_parameters())
            model.to("cuda").eval()
            write_json(run / "model_setup.json", {"base_model": config["base_model"], "adapter_loaded": False, "trainable_parameters": 0,
                       "parameter_count": sum(p.numel() for p in model.parameters()), "dtype": "bfloat16", "gpu": args.gpu})
            gen = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                   repetition_penalty=1.0, use_cache=True)
            sequences = list(sequence_by_route.values());started = time.time()
            for start in range(0, len(todo), int(config["batch_size"])):
                batch = todo[start:start + int(config["batch_size"])]
                encoded = tokenizer.apply_chat_template([messages(row) for row in batch], tokenize=True, add_generation_prompt=True,
                                                        padding=True, truncation=False, return_tensors="pt", return_dict=True)
                width = encoded["input_ids"].shape[1]
                encoded = {k:v.to("cuda") for k,v in encoded.items()}
                def prefix_allowed(batch_id, ids):
                    return allowed_next(ids[width:].tolist(), sequences, tokenizer.eos_token_id)
                with torch.inference_mode(), last_token_projection(model):
                    generated = model.generate(**encoded, generation_config=gen, use_model_defaults=False, prefix_allowed_tokens_fn=prefix_allowed)
                for i, row in enumerate(batch):
                    tokens = generated[i, width:].tolist()
                    if tokenizer.eos_token_id not in tokens:raise RuntimeError("A constrained route failed to emit EOS")
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                    route = tokenizer.decode(tokens, skip_special_tokens=True)
                    if route not in ROUTES:raise RuntimeError("Controller output is outside the route vocabulary")
                    value = {"patient_id": row["patient_id"], "evidence_mode": row["evidence_mode"], "route": route,
                             "raw_output": route, "input_sha256": row["input_sha256"], "fingerprint": fingerprint,
                             "prompt_tokens": int(encoded["attention_mask"][i].sum()), "completion_ids": tokens,
                             "generated_tokens": len(tokens), "ended_with_eos": True}
                    write_json(output / "predictions" / row["evidence_mode"] / f"{row['patient_id']}.json", value)
                    predictions.append(value)
                status(config, "running", completed=len(predictions), total=224, gpu=args.gpu, elapsed_seconds=time.time()-started,
                       gpu_allocated_mib=torch.cuda.memory_allocated()/1024**2)
                if start == 0:
                    write_json(run / "first_inference_verified.json", {"valid_routes": [v["route"] for v in predictions[-len(batch):]],
                               "gpu": args.gpu, "adapter_loaded": False, "complete_inputs_untruncated": True,
                               "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated()/1024**2})
                del encoded, generated
            del model
        result = report(config, rows, predictions, missing, fingerprint)
        if result["state"] != "completed":raise RuntimeError("Expected all 224 route choices")
        if any(sha(path) != expected for path, expected in hashes.items()):
            raise RuntimeError("Original source files changed during routing")
        status(config, "completed", completed=len(predictions), unique_patients=56, by_mode=result["by_mode"], missing_diagnoses=len(missing))
        share_owned(output)
    except BaseException as error:
        status(config, "failed", error_type=type(error).__name__, error=str(error)[:600]);raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN);lock.close()


if __name__ == "__main__":
    main()
