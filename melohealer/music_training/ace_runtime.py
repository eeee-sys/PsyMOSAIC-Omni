"""Frozen ACE-Step planning/decoding and differentiable DiT LoRA execution.

Only the caption is supplied by the dataset. The fixed instrumental setting and
duration are run settings; one frozen LM plan is reused by every neighbour in a
group. Components enter the GPU one stage at a time.
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


from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any

import torch


from project_paths import VENDOR_ROOT

DEFAULT_CODE_ROOT = VENDOR_ROOT / "ACE-Step-1.5"
TURBO_SHIFT3 = (1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
                0.75, 0.6428571428571429, 0.5, 0.3, 0.0)


@dataclass
class CaptionCondition:
    """CPU-resident, detached conditioning shared by an entire rollout group."""

    encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor
    context_latents: torch.Tensor
    attention_mask: torch.Tensor


def shifted_timesteps(device: str | torch.device = "cpu") -> torch.Tensor:
    """Return the official eight-step shift-three ODE schedule, including zero."""
    return torch.tensor(TURBO_SHIFT3, device=device, dtype=torch.float32)


class AceRuntime:
    """ACE-Step components with frozen planning and trainable decoder adapters."""

    def __init__(self, model_base: str | Path, code_root: str | Path = DEFAULT_CODE_ROOT,
                 device: str = "cuda", duration: float = 30.0, lora_rank: int = 32,
                 lora_alpha: int = 64, lora_dropout: float = 0.0,
                 keep_models_on_device: bool = False):
        self.model_base = Path(model_base)
        self.code_root = Path(code_root)
        self.device = torch.device(device)
        self.duration = float(duration)
        self.lora_rank, self.lora_alpha = int(lora_rank), int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.keep_models_on_device = bool(keep_models_on_device)
        self.dtype = torch.bfloat16
        self.sample_rate = 48000
        self.latent_frames = round(self.duration * 25)
        self.model = self.text_encoder = self.text_tokenizer = self.vae = None
        self.lm_handler = None
        self.silence_latent = None
        self.lora_targets: list[str] = []

    @property
    def decoder(self):
        """Expose the PEFT decoder for optimizers and adapter checkpoints."""
        return self.model.decoder

    def load(self) -> "AceRuntime":
        """Load local checkpoints on CPU, freezing everything except DiT LoRA."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoTokenizer
        if str(self.code_root) not in sys.path:
            sys.path.insert(0, str(self.code_root))
        self.model = AutoModel.from_pretrained(
            self.model_base / "acestep-v15-turbo", trust_remote_code=True,
            local_files_only=True, dtype=self.dtype, attn_implementation="sdpa",
        ).eval()
        self.model.requires_grad_(False)
        target_pattern = re.compile(
            r"layers\.\d+\.(?:(?:self_attn|cross_attn)\.(?:q_proj|k_proj|v_proj|o_proj)"
            r"|mlp\.(?:gate_proj|up_proj|down_proj))"
        )
        self.lora_targets = [name for name, module in self.model.decoder.named_modules()
                             if target_pattern.fullmatch(name)
                             and isinstance(module, torch.nn.Linear)]
        expected = int(self.model.config.num_hidden_layers) * 11
        if len(self.lora_targets) != expected:
            raise RuntimeError(f"Expected {expected} DiT LoRA projections, got {len(self.lora_targets)}")
        # No causal-LM task type: the native diffusion forward must remain intact.
        self.model.decoder = get_peft_model(self.model.decoder, LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
            bias="none", target_modules=self.lora_targets,
        ))
        for parameter in self.decoder.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        text_path = self.model_base / "Qwen3-Embedding-0.6B"
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_path, local_files_only=True)
        self.text_encoder = AutoModel.from_pretrained(
            text_path, local_files_only=True, dtype=self.dtype).eval().requires_grad_(False)
        self.silence_latent = torch.load(
            self.model_base / "acestep-v15-turbo/silence_latent.pt",
            map_location="cpu", weights_only=True).transpose(1, 2).to(self.dtype)
        self.validate_trainable_parameters()
        if self.keep_models_on_device:
            self.model.to(self.device)
            self.text_encoder.to(self.device)
        return self

    def validate_trainable_parameters(self) -> dict[str, Any]:
        """Require exactly FP32 adapter parameters, without training other modules."""
        trainable = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        if not trainable or any(not name.startswith("decoder.") or "lora_" not in name
                                or p.dtype != torch.float32 for name, p in trainable):
            raise RuntimeError("Only FP32 decoder LoRA parameters may be trainable")
        return {"trainable_parameters": sum(p.numel() for _, p in trainable),
                "target_count": len(self.lora_targets), "targets": self.lora_targets}

    def _empty_cache(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def offload(self, force: bool = False) -> None:
        """Move all components to CPU after a completed phase/backward pass."""
        if self.keep_models_on_device and not force:
            return
        for module in (self.model, self.text_encoder, self.vae):
            if module is not None:
                module.to("cpu")
        if self.lm_handler is not None and self.lm_handler.llm is not None:
            self.lm_handler.llm.to("cpu")
        self._empty_cache()

    def _load_lm(self) -> None:
        if self.lm_handler is not None:
            return
        from acestep.llm_inference import LLMHandler
        self.lm_handler = LLMHandler()
        message, success = self.lm_handler.initialize(
            checkpoint_dir=str(self.model_base), lm_model_path="acestep-5Hz-lm-1.7B",
            backend="pt", device=str(self.device),
            offload_to_cpu=not self.keep_models_on_device, dtype=self.dtype)
        if not success:
            self.lm_handler = None
            raise RuntimeError(f"Frozen ACE-Step LM initialization failed: {message}")
        self.lm_handler.llm.requires_grad_(False).eval()

    def plan_caption(self, caption: str, seed: int) -> dict[str, Any]:
        """Generate one reproducible frozen 5-Hz plan for all six neighbours."""
        self.offload()
        self._load_lm()
        random_state = random.getstate()
        devices = [self.device.index or torch.cuda.current_device()] if self.device.type == "cuda" else []
        try:
            random.seed(int(seed))
            with torch.random.fork_rng(devices=devices), torch.no_grad(), self.lm_handler._load_model_context():
                torch.manual_seed(int(seed))
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(int(seed))
                result = self.lm_handler.generate_with_stop_condition(
                    caption=caption, lyrics="[Instrumental]", infer_type="llm_dit",
                    temperature=0.85, cfg_scale=1.0, top_k=None, top_p=0.9,
                    target_duration=self.duration, user_metadata={"duration": int(self.duration)},
                    use_cot_caption=False, use_cot_language=False, use_cot_metas=True,
                    use_constrained_decoding=True, batch_size=1,
                )
        finally:
            random.setstate(random_state)
            if self.lm_handler.llm is not None and not self.keep_models_on_device:
                self.lm_handler.llm.to("cpu")
            if not self.keep_models_on_device:
                self._empty_cache()
        if not result.get("success"):
            raise RuntimeError(f"Frozen LM planning failed: {result.get('error')}")
        codes = [int(value) for value in re.findall(r"<\|audio_code_(\d+)\|>", result["audio_codes"])]
        required = round(self.duration * 5)
        if len(codes) < required or any(value < 0 or value > 63999 for value in codes):
            raise RuntimeError(f"Invalid LM plan: {len(codes)} codes, expected at least {required}")
        return {"caption": caption, "seed": int(seed), "duration": self.duration,
                "metadata": result.get("metadata", {}), "audio_code_ids": codes[:required],
                "generated_code_count": len(codes), "group_shared": True}

    def condition_caption(self, caption: str, plan: dict[str, Any]) -> CaptionCondition:
        """Encode the caption and shared LM codes with frozen ACE components."""
        if caption != plan["caption"]:
            raise ValueError("The cached LM plan belongs to another caption")
        from acestep.constants import DEFAULT_DIT_INSTRUCTION, SFT_GEN_PROMPT
        from acestep.core.generation.handler.metadata_utils import MetadataMixin
        self.offload()
        metadata = dict(plan.get("metadata", {}))
        metadata["duration"] = int(self.duration)
        metadata_text = MetadataMixin()._dict_to_meta_string(metadata)
        text_prompt = SFT_GEN_PROMPT.format(DEFAULT_DIT_INSTRUCTION, caption, metadata_text)
        lyric_prompt = "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>"
        text = self.text_tokenizer(text_prompt, return_tensors="pt", truncation=True, max_length=256)
        lyric = self.text_tokenizer(lyric_prompt, return_tensors="pt", truncation=True, max_length=512)
        self.text_encoder.to(self.device)
        try:
            with torch.no_grad():
                text_hidden = self.text_encoder(input_ids=text.input_ids.to(self.device)).last_hidden_state.cpu()
                lyric_hidden = self.text_encoder.embed_tokens(lyric.input_ids.to(self.device)).cpu()
        finally:
            if not self.keep_models_on_device:
                self.text_encoder.to("cpu")
                self._empty_cache()
        # Match official text-to-music: no external reference audio, silence timbre.
        self.model.encoder.to(self.device)
        self.model.tokenizer.to(self.device)
        self.model.detokenizer.to(self.device)
        try:
            # FSQ reconstructs code vectors in FP32 internally while its frozen
            # projection weights are BF16; autocast reconciles that projection.
            with torch.no_grad(), torch.autocast(self.device.type, dtype=self.dtype):
                reference = self.silence_latent[:, :750, :].to(self.device)
                enc, mask = self.model.encoder(
                    text_hidden_states=text_hidden.to(self.device),
                    text_attention_mask=text.attention_mask.to(self.device),
                    lyric_hidden_states=lyric_hidden.to(self.device),
                    lyric_attention_mask=lyric.attention_mask.to(self.device),
                    refer_audio_acoustic_hidden_states_packed=reference,
                    refer_audio_order_mask=torch.zeros(1, device=self.device, dtype=torch.long))
                indices = torch.tensor(plan["audio_code_ids"], device=self.device).reshape(1, -1, 1)
                quantized = self.model.tokenizer.quantizer.get_output_from_indices(indices).to(self.dtype)
                lm_latent = self.model.detokenizer(quantized)[:, :self.latent_frames, :]
                if lm_latent.shape[1] != self.latent_frames:
                    raise RuntimeError("LM detokenized plan has the wrong duration")
                context = torch.cat((lm_latent, torch.ones_like(lm_latent)), dim=-1)
                condition = CaptionCondition(enc.detach().cpu().clone(), mask.detach().cpu().clone(),
                    context.detach().cpu().clone(), torch.ones(1, self.latent_frames, dtype=torch.bool))
        finally:
            if not self.keep_models_on_device:
                self.model.encoder.to("cpu")
                self.model.tokenizer.to("cpu")
                self.model.detokenizer.to("cpu")
                self._empty_cache()
        return condition

    def activate_decoder(self, training: bool = True):
        """Place only the decoder on the device, preserving FP32 adapters."""
        self.offload()
        self.decoder.to(self.device)
        self.decoder.train(bool(training))
        return self.decoder

    def velocity(self, x: torch.Tensor, t: float | torch.Tensor,
                 condition: CaptionCondition, grad: bool = True) -> torch.Tensor:
        """Return the differentiable velocity field without an LM-style wrapper."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(device=self.device, dtype=self.dtype)
        batch = x.shape[0]
        timestep = torch.as_tensor(t, device=self.device, dtype=self.dtype).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch)
        def move(value):
            value = value.to(self.device)
            return value.expand(batch, *value.shape[1:]) if value.shape[0] == 1 else value
        amp = torch.autocast("cuda", dtype=self.dtype) if self.device.type == "cuda" else nullcontext()
        with (torch.enable_grad() if grad else torch.no_grad()), amp:
            return self.decoder(
                hidden_states=x, timestep=timestep, timestep_r=timestep,
                attention_mask=move(condition.attention_mask),
                encoder_hidden_states=move(condition.encoder_hidden_states).to(self.dtype),
                encoder_attention_mask=move(condition.encoder_attention_mask),
                context_latents=move(condition.context_latents).to(self.dtype), use_cache=False)[0]

    def decode(self, latents: torch.Tensor, chunk_frames: int = 128,
               overlap: int = 16) -> torch.Tensor:
        """Decode a latent batch to CPU stereo 48-kHz audio using overlap-discard."""
        from diffusers.models import AutoencoderOobleck
        self.offload()
        if self.vae is None:
            self.vae = AutoencoderOobleck.from_pretrained(
                self.model_base / "vae", local_files_only=True).eval().requires_grad_(False)
        latent = latents.detach().cpu()
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        if latent.shape[-1] != 64:
            raise ValueError("Latents must have shape [B, T, 64]")
        latent = latent.transpose(1, 2).float()
        self.vae.to(self.device)
        cores = []
        try:
            with torch.no_grad():
                for start in range(0, latent.shape[-1], chunk_frames):
                    end = min(start + chunk_frames, latent.shape[-1])
                    left, right = max(0, start - overlap), min(latent.shape[-1], end + overlap)
                    wav = self.vae.decode(latent[:, :, left:right].to(self.device)).sample
                    scale = wav.shape[-1] / (right - left)
                    begin = round((start - left) * scale)
                    finish = round((end - left) * scale)
                    cores.append(wav[:, :, begin:finish].float().cpu())
                    del wav
        finally:
            if not self.keep_models_on_device:
                self.vae.to("cpu")
                self._empty_cache()
        expected_samples = round(self.duration * self.sample_rate)
        waveform = torch.cat(cores, dim=-1)[..., :expected_samples]
        if waveform.shape[-1] != expected_samples:
            raise RuntimeError(f"VAE returned {waveform.shape[-1]} samples, expected {expected_samples}")
        if not torch.isfinite(waveform).all():
            raise RuntimeError("VAE returned nonfinite audio")
        return waveform

    def export_complete(self, destination: str | Path) -> Path:
        """Export a portable ACE-Step tree after merging the decoder adapters.

        This finalization operation removes the adapters from this runtime.
        Save the separate adapter checkpoint before calling it.
        """
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"Complete model export already exists: {destination}")
        self.offload(force=True)
        shutil.copytree(self.model_base, destination, symlinks=False,
                        ignore=shutil.ignore_patterns(".cache", ".git"))
        self.model.decoder = self.decoder.merge_and_unload(safe_merge=True)
        self.model.save_pretrained(destination / "acestep-v15-turbo", safe_serialization=True)
        (destination / "training_export.json").write_text(json.dumps({
            "method": "ODE Neighbour GRPO", "merged_decoder_lora": True,
            "lora_rank": self.lora_rank, "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout, "duration": self.duration,
            "caption_only_external_input": True, "frozen_lm": True,
        }, indent=2) + "\n", encoding="utf-8")
        return destination
