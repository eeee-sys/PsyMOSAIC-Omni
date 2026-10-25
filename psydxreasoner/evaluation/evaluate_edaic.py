#!/usr/bin/env python3
"""Evaluate one Diagnosis adapter under eight paired E-DAIC test conditions.

Generator citations select evidence from complete segment caches. They never appear
in Diagnosis prompts. Test labels are used only after generation to compute metrics.
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
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
from evaluation.evidence import (assemble_generator_profile, load_feature_rules,
                                 load_patient_cache, validate_generator_output)
from evaluation.metrics import DISORDERS, INVALID_POLICY, SETTINGS, setting_metrics, parse_answer


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read_cache(path, key):
    """Recover a killed final append; reject duplicate completed entries."""
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    complete = data.rfind(b"\n") + 1
    if complete != len(data):
        with path.open("r+b") as handle:
            handle.truncate(complete)
        data = data[:complete]
    result = {}
    for line in data.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_key = key(row)
        if row_key in result:
            raise ValueError(f"Duplicate evaluation cache key in {path.name}")
        result[row_key] = row
    return result


def append_cache(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def checked_profile(profile, mode, sanitize_profile):
    clean = sanitize_profile(profile, evidence_mode=mode, dataset="E-DAIC")
    for value in clean.values():
        if set(value) != {"record_update", "multimodal_evidence"}:
            raise ValueError("Diagnosis stages may only contain clinical fields and evidence")
        if set(value["multimodal_evidence"]) != {"text", "audio", "video"}:
            raise ValueError("Diagnosis evidence must have text, audio, and video lists")
        if not all(isinstance(items, list) and all(isinstance(item, str) for item in items)
                   for items in value["multimodal_evidence"].values()):
            raise ValueError("Diagnosis evidence lists must contain only strings")
    return clean


def write_reports(output, rows, patient_ids, settings, manifest):
    metrics = setting_metrics(rows, patient_ids, settings)
    report = {"status": "complete", "scope": manifest["scope"], "patients": len(patient_ids),
              "target_disorders": list(DISORDERS), "predictions": len(rows),
              "settings": metrics, "invalid_prediction_policy": INVALID_POLICY,
              "macro_average_definition": "Arithmetic mean of the two per-disorder positive-class metrics",
              "decode": "Deterministic greedy decoding, explicit EOS, no transcript/profile truncation",
              "test_judge_api_calls": 0, "run_fingerprint": stable_hash(manifest),
              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "metrics.json", report)
    csv_fields = ("source", "evidence_mode", "target_disorder", "patients", "precision", "recall", "f1",
                  "tp", "fp", "tn", "fn", "invalid_answers", "valid_answer_rate")
    temporary = output / f"metrics.csv.{os.getpid()}.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for setting in metrics:
            for disorder, values in setting["per_disorder"].items():
                writer.writerow({"source": setting["source"], "evidence_mode": setting["evidence_mode"],
                                 "target_disorder": disorder, "patients": values["samples"],
                                 **{key: values[key] for key in csv_fields[4:]}})
            writer.writerow({"source": setting["source"], "evidence_mode": setting["evidence_mode"],
                             "target_disorder": "MACRO_AVERAGE", "patients": len(patient_ids),
                             **setting["macro_average"], "valid_answer_rate": setting["valid_answer_rate"]})
    temporary.replace(output / "metrics.csv")
    return report


def evaluate(config, args):
    from data import STAGES, read_jsonl, sanitize_profile, make_messages

    evaluation = config["evaluation"]
    data_root = Path(config.get("source_data_root", config["data_root"]))
    teacher_path = data_root / "edaic/splits/test_patient_records.jsonl"
    teacher_rows = read_jsonl(teacher_path)
    teacher_by_id = {str(row["patient_id"]): row for row in teacher_rows}
    if len(teacher_rows) != len(teacher_by_id) or len(teacher_rows) != 56:
        raise ValueError("Expected all 56 unique official E-DAIC test patients")
    if any(row["split"] != "test" for row in teacher_rows):
        raise ValueError("Non-test patients appeared in the E-DAIC test profiles")
    for row in teacher_rows:
        if set(row["patient_record"]) != set(STAGES):
            raise ValueError("A Teacher test profile does not contain all five clinical stages")
        for disorder in DISORDERS:
            if row["labels"].get(disorder.lower()) not in (0, 1):
                raise ValueError("All E-DAIC test patients must have both valid target labels")
        for mode in {mode for _, mode in SETTINGS}:
            checked_profile(row["patient_record"], mode, sanitize_profile)
    patient_ids = sorted(teacher_by_id, key=lambda x: int(x))
    settings = SETTINGS
    preflight = args.max_patients is not None or args.max_settings is not None
    if args.audit_only and preflight:
        raise ValueError("--audit-only validates the complete official test set; omit preflight limits")
    if args.max_patients is not None:
        if not 1 <= args.max_patients <= 56:
            raise ValueError("--max-patients must be between 1 and 56")
        patient_ids = patient_ids[:args.max_patients]
    if args.max_settings is not None:
        if not 1 <= args.max_settings <= 8:
            raise ValueError("--max-settings must be between 1 and 8")
        settings = SETTINGS[:args.max_settings]
    output = Path(evaluation["output_dir"])
    if preflight:
        output /= f"preflight_patients{len(patient_ids)}_settings{len(settings)}"
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "evaluation.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    adapter = Path(args.adapter or Path(config.get("adapter", LORA_ROOT / "psydxreasoner")))
    generator_adapter = Path(config["generator_adapter"])
    generator_sft_path = Path(config["generator_data_root"]) / "edaic/test.jsonl"
    generator_rows = read_jsonl(generator_sft_path)
    generator_by_key = {(str(row["patient_id"]), row["stage"]): row for row in generator_rows}
    if len(generator_rows) != 280 or len(generator_by_key) != 280:
        raise ValueError("Expected all 280 E-DAIC test stage prompts without duplicate keys")
    if {pid for pid, _ in generator_by_key} != set(teacher_by_id):
        raise ValueError("Teacher and Generator test patient sets differ")
    schemas = {}
    for row in generator_rows:
        if row["split"] != "test" or row["stage"] not in STAGES:
            raise ValueError("Generator prompts must use official test patients and known stages")
        # Only structural schema is extracted; Teacher target values never enter Generator inference.
        target = json.loads(row["messages"][-1]["content"])
        schema = {key: type(value).__name__ for key, value in target["record_update"].items()}
        if row["stage"] in schemas and schemas[row["stage"]] != schema:
            raise ValueError("Inconsistent Generator stage field schemas")
        schemas[row["stage"]] = schema
    generator_retry_policy = evaluation.get("generator_retry_policy")
    retry_stage_keys = set()
    if generator_retry_policy is not None:
        if set(generator_retry_policy) != {"repetition_penalty", "stage_keys"}:
            raise ValueError("Generator retry policy must contain penalty and stage keys")
        penalty = generator_retry_policy["repetition_penalty"]
        if not isinstance(penalty, (int, float)) or not 1.0 < penalty <= 2.0:
            raise ValueError("Generator retry repetition penalty must be within (1, 2]")
        retry_stage_keys = {tuple(key) for key in generator_retry_policy["stage_keys"]}
        if len(retry_stage_keys) != len(generator_retry_policy["stage_keys"]) or not retry_stage_keys <= set(generator_by_key):
            raise ValueError("Generator retry stage keys are duplicate or unknown")
    rules = load_feature_rules(config["reconstruction_code_root"])
    calibration_path = data_root / "edaic/feature_calibration.json"
    calibration = json.loads(calibration_path.read_text())
    if set(map(str, calibration.get("fit_patient_ids", []))) & set(teacher_by_id):
        raise ValueError("Feature calibration contains official test patients")
    caches = {pid: load_patient_cache(data_root / "edaic", pid, rules) for pid in patient_ids}
    if args.audit_only:
        audited_stages = 0
        quote_count = audio_count = video_count = 0
        for pid in patient_ids:
            text_by_id, feature_by_id, baseline = caches[pid]
            outputs = {}
            for stage in STAGES:
                source = teacher_by_id[pid]["patient_record"][stage]
                # This is a Teacher-evidence round-trip audit, never a Generator cache.
                outputs[stage] = validate_generator_output(json.dumps({
                    "stage": stage, "record_update": source["record_update"],
                    "source_segment_ids": source["source_segment_ids"]}),
                    stage, schemas[stage], set(text_by_id))
                audited_stages += 1
            reconstructed = assemble_generator_profile(outputs, text_by_id, feature_by_id,
                                                        baseline, calibration, rules)
            expected = checked_profile(teacher_by_id[pid]["patient_record"], "text_audio_video", sanitize_profile)
            reconstructed = checked_profile(reconstructed, "text_audio_video", sanitize_profile)
            if reconstructed != expected:
                raise ValueError("Teacher evidence differs from the same saved feature conversion rules")
            for stage in STAGES:
                evidence = reconstructed[stage]["multimodal_evidence"]
                quote_count += len(evidence["text"])
                audio_count += len(evidence["audio"])
                video_count += len(evidence["video"])
            for audited_profile in (expected, reconstructed):
                for mode in {mode for _, mode in SETTINGS}:
                    profile = checked_profile(audited_profile, mode, sanitize_profile)
                    for disorder in DISORDERS:
                        messages = make_messages(profile, disorder)
                        if len(messages) != 2 or [message["role"] for message in messages] != ["system", "user"]:
                            raise ValueError("Diagnosis prompts must be one system and one user message")
        report = {"status": "passed", "scope": "cpu_data_evidence_audit", "gpu_calls": 0,
                  "api_calls": 0, "patients": len(patient_ids), "stage_prompts": len(generator_rows),
                  "stages_audited": audited_stages, "settings": len(SETTINGS),
                  "diagnosis_prompt_contracts_checked": len(patient_ids) * len(SETTINGS) * len(DISORDERS),
                  "all_segment_cache_coverage": "exact", "teacher_evidence_round_trip": "exact",
                  "evidence_items": {"text": quote_count, "audio": audio_count, "video": video_count},
                  "feature_rules_version": rules.RULES_VERSION,
                  "alignment_tolerance_seconds": rules.ALIGNMENT_TOLERANCE_SECONDS,
                  "source_sha256": {str(path.resolve()): sha256(path) for path in
                                    (teacher_path, generator_sft_path, calibration_path)},
                  "completed_at_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(output / "audit_report.json", report)
        print(json.dumps(report), flush=True)
        return
    files = [teacher_path, generator_sft_path, calibration_path,
             adapter / "adapter_model.safetensors", adapter / "adapter_config.json",
             generator_adapter / "adapter_model.safetensors", generator_adapter / "adapter_config.json",
             ROOT / "training/data.py", ROOT / "training/model.py",
             Path(rules.__file__), *(Path(__file__).parent / name for name in
                                      ("evaluate_edaic.py", "metrics.py", "evidence.py"))]
    for pid in patient_ids:
        files.extend((data_root / "edaic/transcripts" / f"{pid}.json",
                      data_root / "edaic/feature_statistics" / f"{pid}.json"))
    files.append(Path(rules.__file__).parent / "common.py")
    files.extend(sorted((ROOT / "prompts").glob("*.txt")))
    for tokenizer_root in (Path(config["base_model"]), generator_adapter, adapter):
        files.extend(path for name in ("config.json", "tokenizer_config.json", "tokenizer.json",
                                       "chat_template.jinja", "generation_config.json")
                     if (path := tokenizer_root / name).exists())
    manifest = {"scope": "preflight" if preflight else "full_official_test",
                "patient_ids": patient_ids, "settings": [list(x) for x in settings],
                "base_model": config["base_model"], "diagnosis_adapter": str(adapter),
                "generator_adapter": str(generator_adapter),
                "source_sha256": {str(path.resolve()): sha256(path) for path in files},
                "max_new_tokens": evaluation.get("max_new_tokens", 256),
                "generator_max_new_tokens": evaluation.get("generator_max_new_tokens", 1536),
                "context_limit": config.get("context_limit", 32768),
                "seed": evaluation.get("seed", 42), "feature_rules_version": rules.RULES_VERSION,
                "alignment_tolerance_seconds": rules.ALIGNMENT_TOLERANCE_SECONDS}
    if generator_retry_policy is not None:
        manifest["generator_retry_policy"] = generator_retry_policy
    manifest_path = output / "evaluation_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Existing evaluation cache belongs to different code, data, adapter, or settings")
    atomic_json(manifest_path, manifest)
    status_path = output / "status.json"
    atomic_json(status_path, {"status": "evaluating", "scope": manifest["scope"],
                              "patients": len(patient_ids), "settings": len(settings)})

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from model import load_text_thinker, generate_text
    random.seed(manifest["seed"])
    torch.manual_seed(manifest["seed"])
    torch.cuda.manual_seed_all(manifest["seed"])
    model = None
    tokenizers = {}

    def inference(adapter_name, messages, max_tokens, **generation_kwargs):
        nonlocal model
        if adapter_name not in tokenizers:
            adapter_path = generator_adapter if adapter_name == "generator" else adapter
            tokenizer_root = adapter_path if (adapter_path / "tokenizer_config.json").exists() else Path(config["base_model"])
            tokenizers[adapter_name] = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
        tokenizer = tokenizers[adapter_name]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(input_ids) + max_tokens > manifest["context_limit"]:
            raise ValueError("Complete prompt plus completion budget exceeds the context limit")
        if model is None:
            base = load_text_thinker(config["base_model"], config.get("cpu_offload_embedding", False))
            adapter_path = generator_adapter if adapter_name == "generator" else adapter
            model = PeftModel.from_pretrained(base, adapter_path, adapter_name=adapter_name,
                                              is_trainable=False).to("cuda").eval()
        elif adapter_name not in model.peft_config:
            adapter_path = generator_adapter if adapter_name == "generator" else adapter
            model.load_adapter(str(adapter_path), adapter_name=adapter_name, is_trainable=False)
        return generate_text(model, tokenizer, messages, max_new_tokens=max_tokens,
                             adapter_name=adapter_name, do_sample=False,
                             context_limit=manifest["context_limit"], **generation_kwargs)

    generation_path = output / "generator_stage_predictions.jsonl"
    stage_cache = read_cache(generation_path, lambda row: (str(row["patient_id"]), row["stage"]))
    expected_stage_keys = {(pid, stage) for pid in patient_ids for stage in STAGES}
    if not set(stage_cache) <= expected_stage_keys:
        raise ValueError("Generator cache contains unexpected patients or stages")
    profiles = {}
    needs_generator = any(source == "generator" for source, _ in settings)
    if needs_generator:
        for patient_index, pid in enumerate(patient_ids, 1):
            text_by_id, feature_by_id, baseline = caches[pid]
            successful_stages = {}
            failed_stages = {}
            for stage in STAGES:
                key = (pid, stage)
                if key not in stage_cache:
                    saved = {"patient_id": pid, "stage": stage, "parsed": None, "schema_error": None}
                    try:
                        messages = generator_by_key[key]["messages"][:2]
                        generation_kwargs = {}
                        if key in retry_stage_keys:
                            generation_kwargs["repetition_penalty"] = generator_retry_policy["repetition_penalty"]
                            saved["retry_generation_policy"] = dict(generation_kwargs)
                        result = inference("generator", messages, manifest["generator_max_new_tokens"],
                                           **generation_kwargs)
                        saved.update(result)
                        if not result["ended_with_eos"]:
                            raise ValueError("Generator did not terminate naturally with EOS")
                        saved["parsed"] = validate_generator_output(result["text"], stage, schemas[stage], set(text_by_id))
                    except (ValueError, KeyError, TypeError) as error:
                        saved["schema_error"] = str(error)
                    append_cache(generation_path, saved)
                    stage_cache[key] = saved
                saved = stage_cache[key]
                if saved["schema_error"] is None:
                    successful_stages[stage] = saved["parsed"]
                else:
                    failed_stages[stage] = saved["schema_error"]
            if failed_stages:
                profiles[pid] = {"patient_id": pid, "patient_record": None,
                                 "profile_error": {"failed_stages": failed_stages}}
            else:
                try:
                    profile = assemble_generator_profile(successful_stages, text_by_id, feature_by_id,
                                                         baseline, calibration, rules)
                    profiles[pid] = {"patient_id": pid, "patient_record": checked_profile(profile, "text_audio_video", sanitize_profile),
                                     "profile_error": None}
                except (ValueError, KeyError, TypeError) as error:
                    profiles[pid] = {"patient_id": pid, "patient_record": None,
                                     "profile_error": {"evidence_assembly_error": str(error)}}
            atomic_json(output / "generator_profiles" / f"{pid}.json", profiles[pid])
            print(json.dumps({"phase": "generator", "patients_completed": patient_index,
                              "patients_total": len(patient_ids), "profile_valid": profiles[pid]["profile_error"] is None}), flush=True)

    prediction_path = output / "diagnosis_predictions.jsonl"
    prediction_cache = read_cache(prediction_path, lambda row: (row["source"], str(row["patient_id"]),
                                                               row["evidence_mode"], row["target_disorder"]))
    expected_keys = {(source, pid, mode, disorder) for source, mode in settings
                     for pid in patient_ids for disorder in DISORDERS}
    if not set(prediction_cache) <= expected_keys:
        raise ValueError("Diagnosis cache contains unexpected test conditions")
    for source, mode in settings:
        for pid in patient_ids:
            profile_error = profiles[pid]["profile_error"] if source == "generator" else None
            raw_profile = profiles[pid]["patient_record"] if source == "generator" else teacher_by_id[pid]["patient_record"]
            clean_profile = checked_profile(raw_profile, mode, sanitize_profile) if profile_error is None else None
            for disorder in DISORDERS:
                key = (source, pid, mode, disorder)
                if key in prediction_cache:
                    continue
                saved = {"source": source, "patient_id": pid, "evidence_mode": mode,
                         "target_disorder": disorder,
                         "ground_truth_answer": "POSITIVE" if teacher_by_id[pid]["labels"][disorder.lower()] else "NEGATIVE",
                         "answer": None, "rationale": None, "valid": False, "error": None,
                         "profile_sha256": stable_hash(clean_profile) if clean_profile is not None else None}
                if profile_error is not None:
                    saved["error"] = {"generator_profile_error": profile_error}
                else:
                    try:
                        result = inference("diagnosis", make_messages(clean_profile, disorder), manifest["max_new_tokens"])
                        saved.update(result)
                        if not result["ended_with_eos"]:
                            raise ValueError("Diagnosis did not terminate naturally with EOS")
                        saved.update(parse_answer(result["text"]))
                        saved["valid"] = True
                    except (ValueError, KeyError, TypeError) as error:
                        saved["error"] = str(error)
                append_cache(prediction_path, saved)
                prediction_cache[key] = saved
                atomic_json(status_path, {"status": "evaluating", "scope": manifest["scope"],
                                          "predictions_completed": len(prediction_cache),
                                          "predictions_total": len(expected_keys), "source": source, "evidence_mode": mode})
        print(json.dumps({"phase": "diagnosis", "source": source, "evidence_mode": mode,
                          "predictions_completed": len(prediction_cache), "predictions_total": len(expected_keys)}), flush=True)
    if set(prediction_cache) != expected_keys:
        raise RuntimeError("Patient-complete evaluation did not produce all expected predictions")
    ordered = [prediction_cache[(source, pid, mode, disorder)] for source, mode in settings
               for pid in patient_ids for disorder in DISORDERS]
    report = write_reports(output, ordered, patient_ids, settings, manifest)
    report["generator_profiles_valid"] = sum(row["profile_error"] is None for row in profiles.values()) if needs_generator else None
    report["generator_profiles_total"] = len(profiles) if needs_generator else 0
    atomic_json(output / "metrics.json", report)
    atomic_json(status_path, {"status": "complete", "scope": manifest["scope"],
                              "predictions_completed": len(ordered), "metrics": str(output / "metrics.json")})
    print(json.dumps({"status": "complete", "scope": manifest["scope"], "patients": len(patient_ids),
                      "predictions": len(ordered), "metrics": str(output / "metrics.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", help="Diagnosis adapter override; defaults to final_adapter")
    parser.add_argument("--audit-only", action="store_true",
                        help="Validate all test profiles, caches, feature rules, and prompts without GPU/API calls")
    parser.add_argument("--max-patients", type=int, help="Preflight subset, always saved separately from final results")
    parser.add_argument("--max-settings", type=int, help="Preflight setting subset, always saved separately")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config, args)


if __name__ == "__main__":
    main()
