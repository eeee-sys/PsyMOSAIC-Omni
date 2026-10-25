"""Generate one clinical stage from a complete transcript using the saved adapter."""
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
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration
from model import load_text_thinker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--dataset", default="edaic", choices=["edaic"])
    parser.add_argument("--stage", required=True, choices=["presenting_problem", "symptoms", "course", "functioning", "context_history"])
    parser.add_argument("--transcript", required=True, help="UTF-8 complete transcript; E-DAIC lines retain Segment IDs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    adapter = args.adapter or str(config.get("adapter", LORA_ROOT / "omniprofiler"))
    prompt_dir = Path(__file__).parent.parent / "data_prep" / "prompts" / args.dataset
    messages = [{"role": "system", "content": (prompt_dir / "system.txt").read_text()},
                {"role": "user", "content": (prompt_dir / f"{args.stage}.txt").read_text()
                 + "\n\nFULL INTERVIEW TRANSCRIPT\n" + Path(args.transcript).read_text(encoding="utf-8")}]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(ids) + 2048 > config["context_limit"]:
        raise ValueError("The complete transcript and reserved output exceed the context limit")
    base = load_text_thinker(config["base_model"], config.get("cpu_offload_embedding", False))
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to("cuda").eval()
    input_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                max_new_tokens=2048, do_sample=False,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    text = tokenizer.decode(result[0, len(ids):], skip_special_tokens=True)
    target = json.loads(text)
    expected = {"stage", "record_update", "source_segment_ids"}
    if set(target) != expected or target["stage"] != args.stage:
        raise ValueError("Generated object did not match the requested stage schema")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(target, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(str(destination))


if __name__ == "__main__":
    main()
