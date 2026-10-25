"""Offline, frozen metric backbones for the MusicCaps comparison.

All evaluation inference uses FP32. Audio is averaged to mono without loudness
normalization, then resampled to the backbone's native sampling rate. The caller
provides the already selected 10-second evaluation clips.
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


import gc
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def _add_path(path):
    path = str(Path(path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_checkpoint(path):
    # Some official PANNs checkpoints use the legacy serialization format.
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def _read_mono(path, sample_rate):
    import torchaudio.functional

    audio, original_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    waveform = torch.from_numpy(audio.mean(axis=1).copy())
    if original_sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, original_sr, sample_rate)
    return waveform.contiguous()


def _stack_equal(waveforms):
    lengths = [len(x) for x in waveforms]
    if len(set(lengths)) != 1:
        raise ValueError(f"Expected equal-duration clips in an inference batch: {lengths}")
    return torch.stack(waveforms)


def _array(tensor):
    result = tensor.detach().float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite metric feature")
    return result


class _Backbone:
    def close(self):
        if getattr(self, "model", None) is not None:
            self.model.to("cpu")
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PannsBackbone(_Backbone):
    """Cnn14 32 kHz; true pre-sigmoid logits and native 2048-D embedding."""
    sample_rate = 32000

    def __init__(self, checkpoint, vendor_root, device="cuda"):
        self.device = torch.device(device)
        source = Path(vendor_root) / "audioset_tagging_cnn" / "pytorch"
        _add_path(source)
        spec = importlib.util.spec_from_file_location("psymosaic_official_panns", source / "models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.Cnn14(sample_rate=32000, window_size=1024,
            hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        checkpoint = _load_checkpoint(checkpoint)
        self.model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        self.model.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

    @torch.inference_mode()
    def extract(self, paths):
        # MusicCaps has small source-duration discrepancies. Fix the classifier
        # input to 10 seconds so padding cannot depend on the surrounding batch.
        clips = [_read_mono(p, self.sample_rate)[:320000] for p in paths]
        clips = [F.pad(x, (0, 320000 - len(x))) for x in clips]
        waveforms = _stack_equal(clips).to(self.device)
        captured = []
        # The official model returns only sigmoid scores. Hook the linear layer
        # rather than treating its sigmoid output as logits or inverting it.
        hook = self.model.fc_audioset.register_forward_hook(lambda _m, _i, out: captured.append(out))
        try:
            result = self.model(waveforms)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Unexpected PANNs classifier invocation count")
        return {"logits": _array(captured[0]), "embedding": _array(result["embedding"])}


class VGGishBackbone(_Backbone):
    """FAD raw 128-D patches; final ReLU, PCA and quantization disabled."""
    sample_rate = 16000

    def __init__(self, checkpoint, vendor_root, device="cuda"):
        self.device = torch.device(device)
        _add_path(Path(vendor_root) / "torchvggish")
        from torchvggish.vggish import VGGish
        from torchvggish.vggish_input import waveform_to_examples
        self.waveform_to_examples = waveform_to_examples
        self.model = VGGish(urls={}, device=torch.device("cpu"), pretrained=False,
                            preprocess=False, postprocess=False)
        state = _load_checkpoint(checkpoint)
        self.model.load_state_dict(state, strict=True)
        self.model.embeddings = torch.nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.device = self.device
        self.model.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

    @torch.inference_mode()
    def extract(self, paths):
        patches = []
        counts = []
        for path in paths:
            waveform = _read_mono(path, self.sample_rate).numpy()
            examples = self.waveform_to_examples(waveform, self.sample_rate)
            if not len(examples):
                raise ValueError(f"No VGGish patches: {path}")
            patches.append(examples)
            counts.append(len(examples))
        joined = torch.cat(patches).to(self.device)
        # Keep patch-level features; averaging within each clip changes FAD.
        features = torch.cat([self.model(part) for part in joined.split(64)])
        return {"patches": list(np.split(_array(features), np.cumsum(counts)[:-1]))}


class AestheticsBackbone(_Backbone):
    """Audiobox aesthetics raw, de-standardized CE/CU/PC/PQ predictions."""
    sample_rate = 16000

    def __init__(self, checkpoint, vendor_root, device="cuda"):
        self.device = torch.device(device)
        _add_path(Path(vendor_root) / "audiobox-aesthetics" / "src")
        from audiobox_aesthetics.model.aes import AesMultiOutput
        from audiobox_aesthetics.infer import make_inference_batch
        self.make_inference_batch = make_inference_batch
        state = _load_checkpoint(checkpoint)
        cfg = dict(state["model_cfg"])
        self.model = AesMultiOutput(**cfg, target_transform=state["target_transform"])
        weights = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
        self.model.load_state_dict(weights, strict=True)
        self.transforms = state["target_transform"]
        self.model.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

    @torch.inference_mode()
    def extract(self, paths):
        waveforms = [_read_mono(p, self.sample_rate).unsqueeze(0) for p in paths]
        wavs, masks, weights, bids = self.make_inference_batch(waveforms,
            hop_size=10, window_size=10, sample_rate=self.sample_rate)
        wavs = torch.stack(wavs).to(self.device)
        masks = torch.stack(masks).to(self.device)
        weights = torch.tensor(weights, device=self.device)
        bids = torch.tensor(bids, device=self.device)
        predictions = self.model({"wav": wavs, "mask": masks})
        result = {}
        for axis in ("CE", "CU", "PC", "PQ"):
            transform = self.transforms[axis]
            raw = predictions[axis] * transform["std"] + transform["mean"]
            pooled = [(raw[bids == i] * weights[bids == i]).sum() / weights[bids == i].sum()
                      for i in range(len(paths))]
            result[axis] = _array(torch.stack(pooled))
        return result


class CLAPBackbone(_Backbone):
    """Music/audio-set HTSAT-base CLAP; normalized 512-D embeddings."""
    sample_rate = 48000

    def __init__(self, checkpoint, vendor_root, assets_root, device="cuda"):
        self.device = torch.device(device)
        source = Path(vendor_root) / "CLAP" / "src" / "laion_clap" / "clap_module"
        # Import only the inference model, avoiding training.data's unrelated
        # BERT/BART tokenizer downloads and unused training dependencies.
        package = types.ModuleType("psymosaic_clap_module")
        package.__path__ = [str(source)]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".model")
        from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
        assets = Path(assets_root)
        if not (assets / "config.json").is_file():
            nested = assets / "clap_roberta_base"
            if nested.is_dir():
                assets = nested
        roberta_config = RobertaConfig.from_pretrained(str(assets), local_files_only=True)
        self.tokenizer = RobertaTokenizer.from_pretrained(str(assets), local_files_only=True)
        cfg = json.loads((source / "model_configs" / "HTSAT-base.json").read_text())
        cfg["text_cfg"]["model_type"] = "roberta"
        cfg.update(enable_fusion=False, fusion_type="None")
        # The supplied CLAP checkpoint contains the complete text backbone.
        # Build from local config, then strict-load those trained parameters.
        with patch.object(module.RobertaModel, "from_pretrained",
                          side_effect=lambda *a, **kw: RobertaModel(roberta_config)):
            self.model = module.CLAP(**cfg)
        checkpoint = _load_checkpoint(checkpoint)
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        if "text_branch.embeddings.position_ids" not in self.model.state_dict():
            state.pop("text_branch.embeddings.position_ids", None)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device, dtype=torch.float32)

    @torch.inference_mode()
    def audio_features(self, paths):
        audio = []
        for path in paths:
            x = _read_mono(path, self.sample_rate)
            # This matches the official CLAP inference API's int16 round trip.
            x = (x.clamp(-1, 1) * 32767).to(torch.int16).float() / 32767
            max_samples = 480000
            if len(x) > max_samples:
                # Canonical first 10 seconds, with no random crop; a few source
                # files exceed their intended MusicCaps duration by milliseconds.
                x = x[:max_samples]
            if len(x) < max_samples:
                repeat_count = max_samples // len(x)
                x = F.pad(x.repeat(repeat_count), (0, max_samples - len(x) * repeat_count))
            audio.append(x)
        inputs = {"waveform": torch.stack(audio).to(self.device),
                  "longer": torch.zeros(len(audio), 1, dtype=torch.bool, device=self.device)}
        embedding = self.model.audio_projection(self.model.encode_audio(inputs, self.device)["embedding"])
        return _array(F.normalize(embedding, dim=-1))

    @torch.inference_mode()
    def text_features(self, texts):
        tokens = self.tokenizer(list(texts), padding="max_length", truncation=True,
                                max_length=77, return_tensors="pt")
        embedding = self.model.encode_text(tokens, self.device)
        return _array(F.normalize(embedding, dim=-1))
