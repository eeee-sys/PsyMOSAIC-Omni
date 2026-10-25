#!/usr/bin/env python3
"""Resumable, anonymous held-out medication evaluation; no training is performed.

Run --audit-only on CPU, --gpu INDEX for predictions plus judging, or
--judge-only to score existing predictions without loading a local model.
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
import csv
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parent
DIMENSIONS = ("Context", "Medication", "Explanation", "Response Quality", "Format")
MODEL_IDS = ("pharmawiser",)
SCHEMA_VERSION = "anonymous-medication-five-dimensions-0-3-5-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_at_root(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def private_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_text(path, text):
    path = Path(path)
    private_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sealed_artifact(value):
    return {**value, "artifact_hash": digest(value)}


def cached_artifact(path, expected):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = value.pop("artifact_hash", None)
    if stored_hash != digest(value):
        raise ValueError(f"Artifact checksum mismatch: {path}")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"Cached {key} differs from the current evaluation: {path}")
    return {**value, "artifact_hash": stored_hash}


def load_cases(config):
    """Read the ten ORIGINAL held-out records; never open train.jsonl."""
    path = path_at_root(config["test_file"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("case_id") for row in rows]
    if len(rows) != 10 or len(set(ids)) != 10 or set(ids) != {f"Case{i}" for i in range(31, 41)}:
        raise ValueError("Evaluation requires each original Case31--40 exactly once.")
    candidates = config["candidate_medications"]
    if (len(candidates) < 30 or len({x.casefold() for x in candidates}) != len(candidates)
            or any(not isinstance(x, str) or x != x.strip().lower() or not x for x in candidates)):
        raise ValueError("A common list of at least 30 unique canonical lowercase drug names is required.")
    results = []
    for row in sorted(rows, key=lambda row: int(row["case_id"].removeprefix("Case"))):
        if row.get("split") != "test" or row.get("teacher_reconstruction") is True:
            raise ValueError("Held-out records must be original test cases, not teacher reconstructions.")
        if "target" in row or "reference_medications" in row:
            raise ValueError("Held-out original records cannot contain reconstructed ranked targets.")
        for field in ("patient_historical_record", "current_situation"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing original field {field}: {row['case_id']}")
        if row.get("case_number", int(row["case_id"][4:])) != int(row["case_id"][4:]):
            raise ValueError("Inconsistent source case identity.")
        model_input = {"patient_historical_record": row["patient_historical_record"],
                       "current_situation": row["current_situation"],
                       "candidate_medications": list(candidates)}
        results.append({"case_id": row["case_id"], "model_input": model_input,
                        "input_hash": digest(model_input),
                        "original_reference": row.get("original_reference", ""),
                        "source_record_hash": digest(row)})
    return results


def protocol_for(config):
    # Credentials are never hashed or copied. The path is deliberately excluded
    # from semantic identity so a key can be rotated without changing scores.
    semantic_config = json.loads(canonical(config))
    semantic_config["judge"].pop("api_key_file", None)
    assets = {"schema": SCHEMA_VERSION, "config": semantic_config,
              "test_sha256": file_hash(path_at_root(config["test_file"])),
              "student_prompt_sha256": file_hash(path_at_root(config["student_system_prompt"])),
              "judge_prompt_sha256": file_hash(path_at_root(config["judge_system_prompt"])),
              "runtime_sha256": {name: file_hash(ROOT / name) for name in
                                  ("evaluate.py", "common.py", "training_utils.py")}}
    base = path_at_root(config["base_model"])
    assets["base_metadata_sha256"] = {name: file_hash(base / name) for name in
        ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json") if (base / name).is_file()}
    return {"protocol_id": digest(assets), "assets": assets,
            "judge_anonymous": True, "uniform_rubric": True,
            "score_values": [0, 3, 5], "mean_computed_locally": True,
            "train_loaded": False, "test_case_ids": [f"Case{i}" for i in range(31, 41)]}


def adapter_identity(config, model_id, required=True):
    directory = path_at_root(config["models"][model_id]["adapter_path"])
    files = ("adapter_model.safetensors", "adapter_config.json")
    missing = [name for name in files if not (directory / name).is_file()]
    if missing:
        if required:
            raise FileNotFoundError(f"Final adapter is not complete: {directory}; missing {missing}")
        return {"available": False, "adapter_path": str(directory), "missing": missing}
    hashes = {name: file_hash(directory / name) for name in files}
    return {"available": True, "adapter_path": str(directory), "files_sha256": hashes,
            "adapter_sha256": digest(hashes)}


def prediction_identity(protocol, case, model_id, adapter, config):
    return {"protocol_id": protocol["protocol_id"], "case_id": case["case_id"],
            "model_id": model_id, "input_hash": case["input_hash"],
            "source_record_hash": case["source_record_hash"],
            "adapter_sha256": adapter["adapter_sha256"],
            "decode_hash": digest(config["generation"])}


def validate_judgment(value):
    if not isinstance(value, dict) or set(value) != {"scores", "rationales"}:
        raise ValueError("Judge output must contain exactly scores and rationales.")
    scores, rationales = value["scores"], value["rationales"]
    if not isinstance(scores, dict) or set(scores) != set(DIMENSIONS):
        raise ValueError("Judge scores must contain exactly the five required dimensions.")
    if not isinstance(rationales, dict) or set(rationales) != set(DIMENSIONS):
        raise ValueError("Judge rationales must contain exactly the five required dimensions.")
    if any(type(scores[name]) is not int or scores[name] not in (0, 3, 5) for name in DIMENSIONS):
        raise ValueError("Each judge score must be an integer in {0,3,5}.")
    if any(not isinstance(rationales[name], str) or not rationales[name].strip() for name in DIMENSIONS):
        raise ValueError("Each judge rationale must be a nonempty English sentence.")
    return {"scores": scores, "rationales": rationales,
            "mean_score": sum(scores[name] for name in DIMENSIONS) / len(DIMENSIONS)}


class AccountError(RuntimeError):
    pass


def anonymous_judge_payload(config, case, prediction):
    payload = {"patient_context": case["model_input"], "assistant_output": prediction["raw_text"]}
    if config["judge"].get("include_original_reference_advisory", True):
        payload["original_reference_advisory"] = case["original_reference"]
    return payload


def judge_prediction(config, protocol, case, prediction, output, api_key):
    model_id, case_id = prediction["model_id"], case["case_id"]
    identity = {"protocol_id": protocol["protocol_id"], "model_id": model_id, "case_id": case_id,
                "input_hash": case["input_hash"], "prediction_hash": prediction["artifact_hash"],
                "adapter_sha256": prediction["adapter_sha256"]}
    path = output / "judgments" / model_id / f"{case_id}.json"
    cached = cached_artifact(path, identity)
    if cached:
        validated = validate_judgment({"scores": cached["scores"], "rationales": cached["rationales"]})
        if cached["mean_score"] != validated["mean_score"]:
            raise ValueError(f"Cached judgment has an incorrect mean: {path}")
        return cached
    system = path_at_root(config["judge_system_prompt"]).read_text(encoding="utf-8")
    payload = anonymous_judge_payload(config, case, prediction)
    user = canonical(payload)
    settings = config["judge"]
    feedback = ""
    raw_dir = private_directory(output / "api_raw" / model_id / case_id)
    run_stamp = f"{time.time_ns()}-{os.getpid()}"
    errors = []
    for attempt in range(int(settings.get("max_attempts_per_run", 5))):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if feedback:
            messages.append({"role": "user", "content":
                "Your previous response did not match the required JSON schema. "
                "Re-evaluate the same anonymous output with the same rubric. "
                "Use only integer scores 0, 3, or 5 and the exact keys. Parsing issue: " + feedback})
        body = {"model": settings["model"], "messages": messages,
                "temperature": float(settings.get("temperature", 0)),
                "max_tokens": int(settings.get("max_tokens", 1536)),
                "response_format": {"type": "json_object"}, "enable_thinking": False}
        request_hash = digest({"protocol_id": protocol["protocol_id"], "body": body})
        record = {**identity, "request_hash": request_hash, "request_body": body,
                  "attempt": attempt + 1, "created_at_unix": time.time()}
        raw_path = raw_dir / f"{run_stamp}-attempt-{attempt + 1}.json"
        status = None
        try:
            request = urllib.request.Request(settings["base_url"].rstrip("/") + "/chat/completions",
                data=canonical(body).encode("utf-8"), method="POST",
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=float(settings.get("timeout_seconds", 120))) as response:
                    status = response.status
                    raw_response = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as error:
                status = error.code
                raw_response = error.read().decode("utf-8", errors="replace")
            record.update(http_status=status, response_text=raw_response.replace(api_key, "[REDACTED_API_KEY]"))
            # Always preserve raw responses before attempting nested JSON parsing.
            write_json(raw_path, record)
            if status in (401, 403) or any(term in raw_response.casefold() for term in
                ("arrearage", "insufficientquota", "invalidapikey", "invalid_api_key", "insufficient balance")):
                raise AccountError("Qwen account, credential, or balance error; resolve the account and resume.")
            if status != 200:
                raise RuntimeError(f"Qwen HTTP status {status}")
            envelope = json.loads(raw_response)
            choice = envelope["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("Judge output hit its token limit and is incomplete.")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Judge message content must be a JSON string.")
            valid = validate_judgment(json.loads(content))
            value = sealed_artifact({**identity, **valid, "judge_model": settings["model"],
                "anonymous_payload_hash": digest(payload), "request_hash": request_hash,
                "raw_response_file": str(raw_path), "scoring_schema": SCHEMA_VERSION,
                "created_at_unix": time.time()})
            write_json(path, value)
            pending = output / "pending_judgments" / model_id / f"{case_id}.json"
            if pending.exists():
                pending.unlink()
            return value
        except AccountError:
            raise
        except Exception as error:
            # No clinical answer or API credential is included in the error log.
            detail = str(error).replace(api_key, "[REDACTED_API_KEY]")[:600]
            errors.append({"attempt": attempt + 1, "type": type(error).__name__, "detail": detail})
            if not raw_path.exists():
                record.update(error_type=type(error).__name__, error_detail=detail)
                write_json(raw_path, record)
            feedback = detail if isinstance(error, (ValueError, KeyError, IndexError, TypeError)) else ""
            if attempt + 1 < int(settings.get("max_attempts_per_run", 5)):
                time.sleep(min(15.0, float(settings.get("retry_delay_seconds", 2)) * 2**attempt))
    write_json(output / "pending_judgments" / model_id / f"{case_id}.json",
               {**identity, "state": "unscored_api_or_schema_error", "errors": errors})
    return None


def generate_model(config, protocol, cases, model_id, adapter, output, gpu):
    from common import parse_recommendations
    from training_utils import last_token_generation, load_model, seed_everything, setup_device, tokenizer_for
    identity_for = lambda case: prediction_identity(protocol, case, model_id, adapter, config)
    todo = []
    for case in cases:
        path = output / "predictions" / model_id / f"{case['case_id']}.json"
        if cached_artifact(path, identity_for(case)) is None:
            todo.append(case)
    if not todo:
        return
    setup_device(gpu)
    import torch
    from transformers import GenerationConfig
    tokenizer = tokenizer_for(config)
    system = path_at_root(config["student_system_prompt"]).read_text(encoding="utf-8")
    model = load_model(config, adapter_path=adapter["adapter_path"])
    model.gradient_checkpointing_disable()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    settings = config["generation"]
    if settings.get("do_sample") is not False or settings.get("num_beams", 1) != 1:
        raise ValueError("Both models require the same deterministic greedy decoding.")
    generation = GenerationConfig(max_new_tokens=int(settings["max_new_tokens"]), do_sample=False,
        num_beams=1, repetition_penalty=float(settings.get("repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(settings.get("no_repeat_ngram_size", 0)),
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True)
    try:
        for case in todo:
            seed_everything(int(config.get("seed", 42)))
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(case["model_input"], ensure_ascii=False, separators=(",", ":"))}]
            prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if len(prompt_ids) + generation.max_new_tokens > int(config.get("context_limit", 16384)):
                raise ValueError(f"Untruncated {case['case_id']} exceeds the configured context limit.")
            ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            started = time.time()
            with torch.inference_mode(), last_token_generation(model):
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        generation_config=generation, use_model_defaults=False)
            tokens = result[0, len(prompt_ids):].tolist()
            raw_text = tokenizer.decode(tokens, skip_special_tokens=True)
            try:
                parsed, parse_error = parse_recommendations(raw_text, config["candidate_medications"]), None
            except (ValueError, TypeError, KeyError) as error:
                parsed, parse_error = None, str(error)
            value = sealed_artifact({**identity_for(case), "model_input": case["model_input"],
                "raw_text": raw_text, "completion_ids": tokens, "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(tokens), "ended_with_eos": tokenizer.eos_token_id in tokens,
                "hit_generation_limit": len(tokens) >= generation.max_new_tokens and tokenizer.eos_token_id not in tokens,
                "parsed_output": parsed, "parse_error": parse_error,
                "generation_config": generation.to_dict(), "adapter_identity": adapter,
                "elapsed_seconds": time.time() - started, "created_at_unix": time.time()})
            write_json(output / "predictions" / model_id / f"{case['case_id']}.json", value)
            print(canonical({"event": "prediction_saved", "model": model_id, "case_id": case["case_id"],
                             "tokens": len(tokens), "format_valid": parsed is not None}), flush=True)
            del ids, result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def aggregate(config, protocol, cases, adapters, output):
    metrics, per_case = [], []
    for model_id in MODEL_IDS:
        adapter = adapters.get(model_id)
        predictions, judgments, missing_predictions, missing_judgments = [], [], [], []
        for case in cases:
            prediction_file = output / "predictions" / model_id / f"{case['case_id']}.json"
            if not prediction_file.exists():
                missing_predictions.append(case["case_id"])
                continue
            if adapter is None or not adapter.get("available"):
                raise ValueError("An existing prediction requires its adapter files for identity verification.")
            prediction = cached_artifact(prediction_file, prediction_identity(protocol, case, model_id, adapter, config))
            predictions.append(prediction)
            identity = {"protocol_id": protocol["protocol_id"], "model_id": model_id,
                "case_id": case["case_id"], "input_hash": case["input_hash"],
                "prediction_hash": prediction["artifact_hash"], "adapter_sha256": adapter["adapter_sha256"]}
            path = output / "judgments" / model_id / f"{case['case_id']}.json"
            judgment = cached_artifact(path, identity)
            if judgment is None:
                missing_judgments.append(case["case_id"])
                continue
            validated = validate_judgment({"scores": judgment["scores"], "rationales": judgment["rationales"]})
            if validated["mean_score"] != judgment["mean_score"]:
                raise ValueError("A saved judgment mean does not equal its five-dimension average.")
            judgments.append(judgment)
            per_case.append({"model": model_id, "case_id": case["case_id"], **judgment["scores"],
                "mean_score": judgment["mean_score"], "prediction_file": str(prediction_file),
                "judgment_file": str(path), "prediction_format_valid": prediction["parsed_output"] is not None})
        count = len(judgments)
        metrics.append({"model": model_id, "label": config["models"][model_id]["label"],
            "expected_cases": 10, "predictions": len(predictions), "scored_cases": count,
            "format_valid_predictions": sum(p["parsed_output"] is not None for p in predictions),
            "dimension_means": {name: sum(j["scores"][name] for j in judgments) / count if count else None for name in DIMENSIONS},
            "mean_score": sum(j["mean_score"] for j in judgments) / count if count else None,
            "missing_predictions": missing_predictions, "missing_judgments": missing_judgments})
    complete = all(row["scored_cases"] == 10 for row in metrics)
    report = {"state": "completed" if complete else "partial", "protocol_id": protocol["protocol_id"],
        "score_values": [0, 3, 5], "score_scale": "0--5", "uniform_anonymous_judge": True,
        "test_cases": 10, "train_loaded": False, "metrics": metrics, "per_case": per_case,
        "adapter_identities": adapters, "created_at_unix": time.time(),
        "aggregation": "Unscored infrastructure failures are excluded, never converted to clinical zeros. Original invalid model answers are judged and retained."}
    write_json(output / "results" / "summary.json", report)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Model", "Scored/10", *DIMENSIONS, "Mean"])
    for row in metrics:
        writer.writerow([row["label"], f"{row['scored_cases']}/10", *[row["dimension_means"][name] for name in DIMENSIONS], row["mean_score"]])
    write_text(output / "results" / "summary.csv", stream.getvalue())
    format_number = lambda value: "—" if value is None else f"{value:.3f}"
    lines = ["| Model | Scored/10 | Context | Medication | Explanation | Response Quality | Format | Mean |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        lines.append("| " + " | ".join([row["label"], f"{row['scored_cases']}/10",
            *[format_number(row["dimension_means"][name]) for name in DIMENSIONS], format_number(row["mean_score"])]) + " |")
    lines.extend(["", "Same anonymous Qwen3.8-flash rubric for all models; dimension scores are 0, 3, or 5. Means are computed locally.",
                  "Partial coverage is reported explicitly. Infrastructure failures are unscored; invalid model answers remain eligible for content/format scores."])
    write_text(output / "results" / "summary.txt", "\n".join(lines) + "\n")
    return report


def main():
    global MODEL_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/eval.json")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    if args.judge_only and args.predict_only:
        raise ValueError("--judge-only and --predict-only cannot be combined.")
    config = load_config(path_at_root(args.config))
    MODEL_IDS = tuple(config["models"])
    args.models = args.models or list(MODEL_IDS)
    if not set(args.models) <= set(MODEL_IDS):
        raise ValueError("Requested model is absent from the evaluation configuration.")
    if config["generation"].get("do_sample") is not False or config["generation"].get("num_beams", 1) != 1:
        raise ValueError("The shared decode configuration must be deterministic greedy.")
    cases = load_cases(config)
    protocol = protocol_for(config)
    output = private_directory(path_at_root(config["output_dir"]))
    adapters = {model_id: adapter_identity(config, model_id, required=False) for model_id in MODEL_IDS}
    if args.audit_only:
        audit = {"state": "cpu_data_audit_passed", "case_count": len(cases),
            "case_ids": [case["case_id"] for case in cases], "candidate_count": len(config["candidate_medications"]),
            "model_input_fields": list(cases[0]["model_input"]), "train_loaded": False,
            "gpu_loaded": False, "protocol": protocol, "adapters": adapters}
        write_json(output / "run" / "cpu_audit.json", audit)
        print(canonical({key: audit[key] for key in ("state", "case_count", "case_ids", "candidate_count", "gpu_loaded")}), flush=True)
        return 0
    if not args.judge_only and args.gpu is None:
        raise ValueError("An explicitly reviewed --gpu INDEX is required for local inference.")
    # One writer per output directory; this prevents a judge-only resume racing
    # another process on the same model/case artifacts.
    lock = open(output / ".evaluation.lock", "a+", encoding="utf-8")
    os.chmod(output / ".evaluation.lock", 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another evaluation process currently owns this output directory.")
    try:
        protocol_path = output / "run" / "evaluation_protocol.json"
        if protocol_path.exists():
            previous = json.loads(protocol_path.read_text(encoding="utf-8"))
            if previous["protocol_id"] != protocol["protocol_id"]:
                raise ValueError("Evaluation protocol changed; use a separate output directory.")
        else:
            write_json(protocol_path, protocol)
        api_key = None
        if not args.predict_only:
            api_key = get_api_key(config["judge"])
            if not api_key:
                raise ValueError("The existing private Qwen API credential file is empty.")
        for model_id in args.models:
            adapters[model_id] = adapter_identity(config, model_id)
            if not args.judge_only:
                generate_model(config, protocol, cases, model_id, adapters[model_id], output, args.gpu)
            if args.predict_only:
                continue
            for case in cases:
                identity = prediction_identity(protocol, case, model_id, adapters[model_id], config)
                prediction = cached_artifact(output / "predictions" / model_id / f"{case['case_id']}.json", identity)
                if prediction is None:
                    continue
                judgment = judge_prediction(config, protocol, case, prediction, output, api_key)
                print(canonical({"event": "judgment_saved" if judgment else "judgment_deferred",
                                 "model": model_id, "case_id": case["case_id"]}), flush=True)
                aggregate(config, protocol, cases, adapters, output)
        report = aggregate(config, protocol, cases, adapters, output)
        requested_complete = all(row["scored_cases"] == 10 for row in report["metrics"]
                                 if row["model"] in args.models)
        return_code = 0 if args.predict_only or requested_complete else 2
        write_json(output / "run" / "status.json", {"state": report["state"],
            "protocol_id": protocol["protocol_id"], "models": report["metrics"],
            "requested_models": args.models, "all_requested_judged": requested_complete,
            "predict_only": args.predict_only, "return_code": return_code, "updated_at_unix": time.time()})
        print(canonical({"state": report["state"], "return_code": return_code,
            "scored_cases": {row["model"]: row["scored_cases"] for row in report["metrics"]}}), flush=True)
        return return_code
    except BaseException as error:
        write_json(output / "run" / "status.json", {"state": "stopped_error", "error_type": type(error).__name__,
            "error_detail": str(error)[:800], "updated_at_unix": time.time()})
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
