"""Document-defined gated rewards with a cached, normalized Qwen API judge."""
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


from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import threading
import time

_FORMAT = re.compile(r"\s*<rationale>([^<>]+)</rationale>\s*<answer>\s*(POSITIVE|NEGATIVE)\s*</answer>\s*", re.DOTALL)


def parse_response(response):
    match = _FORMAT.fullmatch(response) if isinstance(response, str) else None
    valid = bool(match and match.group(1).strip())
    return {"format_valid": valid, "rationale": match.group(1).strip() if valid else "", "answer": match.group(2) if valid else None}


def validate_score(content):
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"score"}:
        raise ValueError('Judge must return exactly {"score": integer}')
    score = value["score"]
    if type(score) is not int or not 0 <= score <= 10:
        raise ValueError("Judge score must be an integer in [0, 10]")
    return score


class QwenRationaleJudge:
    def __init__(self, config):
        self.config = config
        self.options = config["judge"]
        self.prompt = Path(self.options["system_prompt"]).read_text(encoding="utf-8").strip()
        self.key = get_api_key(self.options)
        if len(self.key) < 20:
            raise ValueError("Qwen credential file is missing or empty")
        self.cache = Path(self.options["cache_path"])
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.events = Path(config["run_dir"]) / "judge_api_events.jsonl"
        self.events.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache) as db:
            db.execute("CREATE TABLE IF NOT EXISTS scores (fingerprint TEXT PRIMARY KEY, score INTEGER NOT NULL, metadata TEXT NOT NULL)")

    def _event(self, event):
        with self.lock:
            with self.events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"unix_time": time.time(), **event}, ensure_ascii=False) + "\n")

    def request(self, profile, target, truth, parsed):
        content = {"PATIENT RECORD": profile, "TARGET DISORDER": target,
                   "GROUND TRUTH SCREENING LABEL": truth, "PREDICTED ANSWER": parsed["answer"],
                   "GENERATED RATIONALE": parsed["rationale"]}
        return {"model": self.options["model"],
                "messages": [{"role": "system", "content": self.prompt},
                             {"role": "user", "content": json.dumps(content, ensure_ascii=False, separators=(",", ":"))}],
                "temperature": 0, "enable_thinking": False, "max_tokens": 64,
                "response_format": {"type": "json_object"}}

    def score_request(self, payload):
        import requests
        fingerprint = hashlib.sha256(json.dumps({"url": self.options["base_url"], "payload": payload}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.lock, sqlite3.connect(self.cache) as db:
            cached = db.execute("SELECT score FROM scores WHERE fingerprint=?", (fingerprint,)).fetchone()
        if cached:
            return cached[0] / 10.0, True
        endpoint = self.options["base_url"].rstrip("/") + "/chat/completions"
        last_error = ""
        for attempt in range(1, self.options.get("max_attempts", 4) + 1):
            if self.stop.is_set():
                raise RuntimeError("Qwen scoring stopped after an authentication, model-access, or account error")
            started = time.monotonic()
            try:
                response = requests.post(endpoint, headers={"Authorization": "Bearer " + self.key}, json=payload,
                                         timeout=(15, self.options.get("timeout_seconds", 90)))
                body = response.json()
                error = body.get("error", {}) if isinstance(body, dict) else {}
                code = error.get("code", "") if isinstance(error, dict) else ""
                if response.status_code in (401, 403, 404) or code == "Arrearage":
                    self.stop.set()
                if not response.ok:
                    raise RuntimeError(f"Qwen HTTP {response.status_code}, code={code}")
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Judge JSON response was truncated")
                score = validate_score(choice["message"]["content"])
                metadata = {"model": body.get("model"), "response_id": body.get("id"),
                            "usage": body.get("usage", {}), "latency_seconds": time.monotonic() - started,
                            "attempts": attempt, "raw_score": score, "normalized_score": score / 10.0}
                with self.lock, sqlite3.connect(self.cache) as db:
                    db.execute("INSERT OR REPLACE INTO scores VALUES (?,?,?)", (fingerprint, score, json.dumps(metadata)))
                self._event({"status": "scored", "fingerprint": fingerprint, **metadata})
                return score / 10.0, False
            except Exception as exc:
                last_error = str(exc).replace(self.key, "[redacted]")
                self._event({"status": "error", "fingerprint": fingerprint, "attempt": attempt, "error": last_error})
                if self.stop.is_set() or attempt == self.options.get("max_attempts", 4):
                    break
                time.sleep(min(10, 2 ** attempt) + random.random())
        # An infrastructure failure is not a low-quality rationale. Do not fake a zero.
        raise RuntimeError("Qwen rationale scoring failed: " + last_error)

    def score_many(self, requests_to_score):
        with ThreadPoolExecutor(max_workers=self.options.get("workers", 6)) as pool:
            return list(pool.map(self.score_request, requests_to_score))


def completion_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]["content"]
    raise ValueError("Expected one assistant completion per candidate")


def compute_gated_rewards(completions, patient_record, target_disorder, ground_truth_answer, weights, judge):
    if not (len(completions) == len(patient_record) == len(target_disorder) == len(ground_truth_answer)):
        raise ValueError("Completion and reward metadata lengths differ")
    components, payloads, indices = [], [], []
    for i, (completion, profile, target, truth) in enumerate(zip(completions, patient_record, target_disorder, ground_truth_answer)):
        if truth not in ("POSITIVE", "NEGATIVE"):
            raise ValueError("Missing or invalid classification ground truth")
        parsed = parse_response(completion_text(completion))
        fmt = int(parsed["format_valid"])
        correct = int(fmt and parsed["answer"] == truth)
        components.append({"format": fmt, "label": correct, "rationale": 0.0, "judge_used": False, "judge_cached": False})
        if correct:
            indices.append(i)
            payloads.append(judge.request(profile, target, truth, parsed))
    for i, (score, cached) in zip(indices, judge.score_many(payloads)):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Normalized rationale reward must lie in [0, 1]")
        components[i].update(rationale=score, judge_used=True, judge_cached=cached)
    rewards = [weights["format"] * c["format"] + weights["label"] * c["format"] * c["label"]
               + weights["rationale"] * c["format"] * c["label"] * c["rationale"] for c in components]
    return rewards, components


def build_reward_function(config):
    judge = QwenRationaleJudge(config)
    weights = config.get("reward_weights", {"format": 0.5, "label": 1.0, "rationale": 0.5})
    if set(weights) != {"format", "label", "rationale"} or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("Expected three positive finite reward weights")
    trace = Path(config["run_dir"]) / "reward_components.jsonl"

    def gated_diagnosis_reward(prompts, completions, patient_record=None, target_disorder=None, ground_truth_answer=None, **kwargs):
        result, components = compute_gated_rewards(completions, patient_record, target_disorder, ground_truth_answer, weights, judge)
        stats = {"candidates": len(result), "format_valid": sum(x["format"] for x in components),
                 "label_correct": sum(x["label"] for x in components),
                 "judge_scored": sum(x["judge_used"] for x in components),
                 "judge_cache_hits": sum(x["judge_cached"] for x in components),
                 "mean_reward": sum(result) / len(result)}
        gated_diagnosis_reward.last_stats = stats
        with trace.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"unix_time": time.time(), **stats, "components": components, "rewards": result}, ensure_ascii=False) + "\n")
        return result

    gated_diagnosis_reward.last_stats = {}
    return gated_diagnosis_reward
