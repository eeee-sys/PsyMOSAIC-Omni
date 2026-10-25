"""Held-out teacher-target loss and deterministic JSON-format generation checks."""
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
import json
from pathlib import Path
import re
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration
from data import AssistantCollator, TokenizedDataset, atomic_json, read_jsonl, selected_causal_loss
from model import load_text_thinker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--generation-patients-per-dataset", type=int, default=1)
    parser.add_argument("--generation-only", action="store_true",
                        help="Recheck generation while retaining existing held-out target losses")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(config["run_dir"])
    status_path = run_dir / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    atomic_json(status_path, {**previous, "status": "evaluating"})
    adapter = args.adapter or str(config.get("adapter", LORA_ROOT / "omniprofiler"))
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    base = load_text_thinker(config["base_model"], config.get("cpu_offload_embedding", False))
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to("cuda").eval()
    thinker = model.get_base_model()
    if args.generation_only:
        report = json.loads((run_dir / "heldout_evaluation.json").read_text())
        if report["adapter"] != adapter:
            raise ValueError("Cached losses belong to a different adapter")
        report["generation"] = {}
    else:
        report = {"adapter": adapter, "splits": {}, "generation": {},
                  "interpretation": "Held-out target loss and output-schema checks; clinical quality is not scored by these checks."}
    report["generation_eos_token_id"] = tokenizer.eos_token_id
    report["generation_only_recheck"] = args.generation_only
    generation_rows = []
    for name in ("edaic",):
        records = read_jsonl(Path(config["data_root"]) / name / "test.jsonl")
        if not args.generation_only:
            dataset = TokenizedDataset(records, tokenizer, config["context_limit"])
            weighted_loss, supervised_tokens = 0., 0
            batch_size = config.get("per_device_eval_batch_size", 1)
            ordered_items = sorted(dataset.items, key=lambda item: len(item["input_ids"]))
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for start in range(0, len(ordered_items), batch_size):
                    items = ordered_items[start:start+batch_size]
                    batch = {key: value.to("cuda") for key, value in AssistantCollator(tokenizer.pad_token_id)(items).items()}
                    pos = batch["attention_mask"].long().cumsum(-1)-1
                    pos.masked_fill_(batch["attention_mask"] == 0, 1)
                    out = thinker.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                        position_ids=pos.unsqueeze(0).expand(3, -1, -1), use_cache=False)
                    loss = selected_causal_loss(out.last_hidden_state, thinker.lm_head, batch["labels"],
                                               chunk_size=config.get("projection_chunk_size", 128))
                    n = int((batch["labels"][:, 1:] != -100).sum())
                    weighted_loss += loss.item()*n
                    supervised_tokens += n
            report["splits"][name] = {"samples": len(records), "target_tokens": supervised_tokens,
                                      "token_weighted_loss": weighted_loss/supervised_tokens}
        patients = sorted({row["patient_id"] for row in records}, key=str)[:args.generation_patients_per_dataset]
        chosen = [row for row in records if row["patient_id"] in patients]
        valid = 0
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for row in chosen:
                ids = tokenizer.apply_chat_template(row["messages"][:2], tokenize=True, add_generation_prompt=True)
                input_ids = torch.tensor([ids], device="cuda")
                generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=2048, do_sample=False, use_cache=True,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                text = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
                parsed, error = None, None
                try:
                    parsed = json.loads(text)
                    expected = {"stage", "record_update", "source_segment_ids"}
                    assert isinstance(parsed, dict) and set(parsed) == expected
                    assert parsed["stage"] == row["stage"] and isinstance(parsed["record_update"], dict)
                    expected_update = json.loads(row["messages"][2]["content"])["record_update"]
                    assert set(parsed["record_update"]) == set(expected_update)
                    for key, expected_value in expected_update.items():
                        value = parsed["record_update"][key]
                        assert isinstance(value, type(expected_value))
                        if isinstance(value, list):
                            assert all(isinstance(item, str) for item in value)
                    if name == "edaic":
                        transcript = row["messages"][1]["content"].split("\n\nFULL INTERVIEW TRANSCRIPT\n", 1)[1]
                        allowed = set(re.findall(r"(?m)^(S\d+)\s*\|", transcript))
                        assert isinstance(parsed["source_segment_ids"], list)
                        assert all(isinstance(s, str) and s in allowed for s in parsed["source_segment_ids"])
                        assert len(parsed["source_segment_ids"]) == len(set(parsed["source_segment_ids"]))
                    valid += 1
                except (ValueError, AssertionError, TypeError) as exc:
                    error = str(exc) or "Output did not match the stage schema or source-ID vocabulary"
                generation_rows.append({"dataset": name, "patient_id": row["patient_id"],
                    "stage": row["stage"], "output": text, "parsed": parsed, "schema_error": error,
                    "generated_tokens": int(generated.shape[1] - len(ids)),
                    "ended_with_eos": int(generated[0, -1]) == tokenizer.eos_token_id})
                print(json.dumps({"dataset": name, "stage": row["stage"],
                    "schema_valid": error is None, "ended_with_eos": int(generated[0, -1]) == tokenizer.eos_token_id}), flush=True)
        report["generation"][name] = {"patients": len(patients), "samples": len(chosen), "schema_valid": valid}
    with (run_dir / "heldout_generation_examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in generation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(run_dir / "heldout_evaluation.json", report)
    atomic_json(status_path, {**previous, "status": "complete", "heldout_evaluation": str(run_dir / "heldout_evaluation.json")})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import sys
        if "--config" in sys.argv:
            config = load_config(sys.argv[sys.argv.index("--config")+1])
            status_path = Path(config["run_dir"]) / "status.json"
            previous = json.loads(status_path.read_text()) if status_path.exists() else {}
            atomic_json(status_path, {**previous, "status": "evaluation_failed",
                "log": str(Path(config["run_dir"]) / "evaluation.log")})
        raise
