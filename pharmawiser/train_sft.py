"""Fresh decoder-only LoRA SFT on Case1–Case30; no held-out case is opened."""
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
from training_utils import (ROOT, append_event, arguments, collate_sft, fingerprint,
    latest_checkpoint, load_model, make_optimizer, optimizer_step, read_checkpoint,
    restore_rng, save_checkpoint, seed_everything, setup_device, sft_loss,
    tokenizer_for, trainable_report, training_rows)


def main():
    args = arguments(__doc__)
    config = load_json(args.config)
    settings = config["sft"]
    run_dir = Path(config.get("run_dir", ROOT / "run")) / "sft"
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_for(config)
    rows, audit = training_rows(config, tokenizer)
    write_json(run_dir / "data_audit.json", audit)
    if args.audit_only:
        print(f"CPU audit passed: {len(rows)} training cases; test not loaded", flush=True)
        return
    setup_device(args.gpu)
    import torch
    seed_everything(int(config.get("seed", 42)))
    output_dir = Path(config["sft_output_dir"])
    identity = fingerprint(config)
    resume = latest_checkpoint(output_dir, args.resume)
    state = read_checkpoint(resume, identity) if resume else {"global_step": 0, "epoch": 0,
        "next_batch": 0, "fingerprint": identity, "stage": "sft"}
    model = load_model(config, adapter_path=resume)
    write_json(run_dir / "model_setup.json", {**trainable_report(model),
        "fresh_base_and_new_lora": resume is None, "base_model": config["base_model"],
        "resume_checkpoint": str(resume) if resume else None})
    batch_size = int(settings.get("batch_size", 1))
    accumulation = int(settings.get("gradient_accumulation_steps", 4))
    epochs = int(settings.get("epochs", 3))
    batches_per_epoch = math.ceil(len(rows) / batch_size)
    total_updates = epochs * math.ceil(batches_per_epoch / accumulation)
    optimizer, scheduler = make_optimizer(model, settings, total_updates)
    if resume:
        optimizer.load_state_dict(state.pop("optimizer"))
        scheduler.load_state_dict(state.pop("scheduler"))
        restore_rng(state.pop("rng"))
    started = time.time()
    model.train()
    for epoch in range(state["epoch"], epochs):
        order = list(range(len(rows)))
        random.Random(int(config.get("seed", 42)) + epoch).shuffle(order)
        batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)]
        cursor = state["next_batch"] if epoch == state["epoch"] else 0
        while cursor < len(batches):
            group = batches[cursor:cursor + accumulation]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for indices in group:
                batch = collate_sft([rows[index] for index in indices], tokenizer.pad_token_id)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = sft_loss(model, batch, int(config.get("projection_chunk_size", 64)))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite supervised loss")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
                del batch, loss
            state["global_step"] += 1
            health = optimizer_step(model, optimizer, scheduler, float(settings.get("max_grad_norm", 1.0)),
                                    run_dir / "first_optimizer_update.json", state["global_step"])
            cursor += len(group)
            state.update(epoch=epoch, next_batch=cursor)
            if cursor == len(batches):
                state.update(epoch=epoch + 1, next_batch=0)
            event = {"stage": "sft", "status": "training", "global_step": state["global_step"],
                     "max_steps": total_updates, "epoch": epoch + cursor / len(batches),
                     "loss": sum(losses) / len(losses), "elapsed_seconds": time.time() - started,
                     "health": health, "gpu_allocated_mib": torch.cuda.memory_allocated() / 1024**2}
            append_event(run_dir / "metrics.jsonl", event)
            write_json(run_dir / "status.json", event)
            if state["global_step"] % int(settings.get("save_steps", 5)) == 0 or cursor == len(batches):
                save_checkpoint(model, tokenizer, optimizer, scheduler, state, output_dir,
                                int(settings.get("save_total_limit", 5)))
    final = output_dir / "final_adapter"
    model.save_pretrained(final, safe_serialization=True, selected_adapters=["default"], save_embedding_layers=False)
    tokenizer.save_pretrained(final)
    write_json(run_dir / "status.json", {"status": "completed", "global_step": state["global_step"],
               "final_adapter": str(final), "epochs": epochs, "test_loaded": False,
               "elapsed_seconds": time.time() - started, "fingerprint": identity})


if __name__ == "__main__":
    main()
