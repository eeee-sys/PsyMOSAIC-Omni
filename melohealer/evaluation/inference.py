"""Shared frozen conditioning and paired native ACE-Step decoder inference."""
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


from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from project_paths import CODE_ROOT, VENDOR_ROOT, expand_paths


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def input_prompt(row, input_kind):
    if input_kind == "aspect":
        return row["aspect_prompt"]
    if input_kind == "caption":
        return row["caption"]
    if input_kind == "aspect_caption":
        return row["aspect_prompt"] + "\n" + row["caption"]
    raise ValueError(input_kind)


def stable_seed(row, config):
    # Every input format and both model variants share this row's latent seed.
    return int(config["seed"]) + int(row["row_index"]) * 1009


class PairedAceInference:
    def __init__(self, config, device="cuda:0"):
        self.config = expand_paths(config)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.decoders = {}
        self.runtime = None
        self.condition_class = None

    def load(self):
        training_root = Path(self.config.get("training_code_root", CODE_ROOT / "melohealer"))
        sys.path.insert(0, str(training_root))
        ace_code_root = str(self.config.get("ace_code_root", VENDOR_ROOT / "ACE-Step-1.5"))
        sys.path.insert(0, ace_code_root)
        from music_training.ace_runtime import AceRuntime, CaptionCondition
        from transformers import AutoModel, AutoTokenizer
        self.condition_class = CaptionCondition
        base = Path(self.config["base_model"])
        # Inference uses native decoder weights, without adding fresh adapters.
        runtime = AceRuntime(base, ace_code_root, device=str(self.device),
                             duration=self.config["duration_seconds"], keep_models_on_device=True)
        runtime.model = AutoModel.from_pretrained(base / "acestep-v15-turbo",
            trust_remote_code=True, local_files_only=True, dtype=self.dtype,
            attn_implementation="sdpa").eval().requires_grad_(False).to(self.device)
        runtime.model.config.use_cache = False
        runtime.text_tokenizer = AutoTokenizer.from_pretrained(base / "Qwen3-Embedding-0.6B",
                                                               local_files_only=True)
        runtime.text_encoder = AutoModel.from_pretrained(base / "Qwen3-Embedding-0.6B",
            local_files_only=True, dtype=self.dtype).eval().requires_grad_(False).to(self.device)
        runtime.silence_latent = torch.load(base / "acestep-v15-turbo/silence_latent.pt",
            map_location="cpu", weights_only=True).transpose(1, 2).to(self.dtype)
        self.decoders["base"] = runtime.model.decoder
        if "finetuned" in self.config.get("model_kinds", ("base", "finetuned")):
            adapter = self.config.get("finetuned_adapter")
            model_root = base if adapter else Path(self.config["finetuned_model"])
            trained = AutoModel.from_pretrained(model_root / "acestep-v15-turbo",
                trust_remote_code=True, local_files_only=True, dtype=self.dtype,
                attn_implementation="sdpa").eval().requires_grad_(False)
            if adapter:
                from peft import PeftModel
                decoder = PeftModel.from_pretrained(
                    trained.decoder, adapter, is_trainable=False, local_files_only=True
                ).merge_and_unload()
            else:
                decoder = trained.decoder
            self.decoders["finetuned"] = decoder.eval().requires_grad_(False).to(self.device)
            del trained
        self.runtime = runtime
        runtime._load_lm()
        gc.collect()
        if any(p.requires_grad for model in self.decoders.values() for p in model.parameters()):
            raise RuntimeError("Inference decoder parameters must be frozen")
        return self

    def encode(self, prompt, plan):
        """Preserve the complete input text; do not truncate combined fields."""
        from acestep.constants import DEFAULT_DIT_INSTRUCTION, SFT_GEN_PROMPT
        from acestep.core.generation.handler.metadata_utils import MetadataMixin
        runtime = self.runtime
        metadata = dict(plan.get("metadata", {}))
        metadata["duration"] = int(self.config["duration_seconds"])
        text_prompt = SFT_GEN_PROMPT.format(DEFAULT_DIT_INSTRUCTION, prompt,
                         MetadataMixin()._dict_to_meta_string(metadata))
        lyric_prompt = "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>"
        text = runtime.text_tokenizer(text_prompt, return_tensors="pt", truncation=False)
        lyric = runtime.text_tokenizer(lyric_prompt, return_tensors="pt", truncation=False)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype):
            text_hidden = runtime.text_encoder(input_ids=text.input_ids.to(self.device)).last_hidden_state
            lyric_hidden = runtime.text_encoder.embed_tokens(lyric.input_ids.to(self.device))
            hidden, mask = runtime.model.encoder(
                text_hidden_states=text_hidden,
                text_attention_mask=text.attention_mask.to(self.device),
                lyric_hidden_states=lyric_hidden,
                lyric_attention_mask=lyric.attention_mask.to(self.device),
                refer_audio_acoustic_hidden_states_packed=runtime.silence_latent[:, :750].to(self.device),
                refer_audio_order_mask=torch.zeros(1, dtype=torch.long, device=self.device))
            indices = torch.tensor(plan["audio_code_ids"], device=self.device).reshape(1, -1, 1)
            quantized = runtime.model.tokenizer.quantizer.get_output_from_indices(indices).to(self.dtype)
            latent = runtime.model.detokenizer(quantized)[:, :runtime.latent_frames]
            if latent.shape[1] != runtime.latent_frames:
                raise RuntimeError("Shared LM plan has the wrong latent duration")
            context = torch.cat((latent, torch.ones_like(latent)), dim=-1)
        condition = self.condition_class(hidden.detach().cpu(), mask.detach().cpu(),
            context.detach().cpu(), torch.ones(1, runtime.latent_frames, dtype=torch.bool))
        return condition, int(text.input_ids.shape[1])

    def condition(self, row, input_kind):
        prompt = input_prompt(row, input_kind)
        seed = stable_seed(row, self.config)
        fingerprint = hashlib.sha256(json.dumps({"prompt": prompt, "seed": seed,
            "duration": self.config["duration_seconds"], "protocol": self.config["protocol_id"]},
            sort_keys=True).encode()).hexdigest()
        root = Path(self.config["condition_root"]) / input_kind
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{row['sample_id']}.pt"
        plan_path = root / f"{row['sample_id']}.json"
        if path.exists():
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value["fingerprint"] != fingerprint:
                raise ValueError(f"Condition cache mismatch: {path}")
            return value["condition"], value["text_tokens"]
        if plan_path.exists():
            value = json.loads(plan_path.read_text())
            if value["fingerprint"] != fingerprint:
                raise ValueError(f"LM plan cache mismatch: {plan_path}")
            plan = value["plan"]
        else:
            plan = self.runtime.plan_caption(prompt, seed)
            atomic_json(plan_path, {"fingerprint": fingerprint, "plan": plan})
        condition, tokens = self.encode(prompt, plan)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"fingerprint": fingerprint, "condition": condition, "text_tokens": tokens}, temporary)
        temporary.replace(path)
        return condition, tokens

    def collate(self, conditions):
        def padded(field):
            return pad_sequence([getattr(c, field)[0] for c in conditions], batch_first=True)
        return self.condition_class(padded("encoder_hidden_states").to(self.device),
            padded("encoder_attention_mask").to(self.device),
            torch.cat([c.context_latents for c in conditions]).to(self.device),
            torch.cat([c.attention_mask for c in conditions]).to(self.device))

    def generate(self, rows, conditions, model_kind):
        runtime = self.runtime
        runtime.model.decoder = self.decoders[model_kind]
        runtime.model.decoder.eval()
        noises = []
        for row in rows:
            generator = torch.Generator(device="cpu").manual_seed(stable_seed(row, self.config))
            noises.append(torch.randn(runtime.latent_frames, 64, generator=generator))
        current = torch.stack(noises).to(self.device)
        condition = self.collate(conditions)
        times = torch.linspace(1, 0, self.config["sampling_steps"] + 1)
        shift = self.config["timestep_shift"]
        times = shift * times / (1 + (shift - 1) * times)
        with torch.no_grad():
            for index in range(len(times) - 1):
                velocity = runtime.velocity(current, float(times[index]), condition, grad=False)
                current = current - float(times[index] - times[index + 1]) * velocity.float()
        waves = []
        for latent in current.split(self.config["vae_batch_size"]):
            waves.extend(runtime.decode(latent).cpu().unbind(0))
        return waves
