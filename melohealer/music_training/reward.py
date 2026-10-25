"""Frozen CMI-RM scoring with fixed, independent two-branch calibration.

The official baseline returns *raw*, unbounded scores in column order
``[alignment, quality]``. No sigmoid is applied. The training reward is
``0.5 * z_alignment + 0.5 * z_quality`` using a fixed calibration distribution.
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
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import torch


SCORE_COLUMNS = ("alignment", "quality")


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) == {"config"}:
        value = value["config"]
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def prepare_assets(assets_dir: str | Path) -> dict[str, Any]:
    """Download only architecture configs and the frozen text tokenizer.

    CMI-RM's full checkpoint already contains its audio and text backbones.
    Therefore no MuQ, MuQ-MuLan, or RoBERTa model weights are downloaded here.
    Hugging Face's configured endpoint/cache settings remain available.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    destination = Path(assets_dir).resolve()
    resources = [
        ("OpenMuQ/MuQ-MuLan-large", "muq_mulan", ("config.json",)),
        ("OpenMuQ/MuQ-large-msd-iter", "muq", ("config.json",)),
        ("FacebookAI/xlm-roberta-base", "xlm_roberta", (
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "sentencepiece.bpe.model", "special_tokens_map.json")),
    ]
    outputs = []
    for repository, directory, filenames in resources:
        for filename in filenames:
            try:
                path = Path(hf_hub_download(
                    repo_id=repository, filename=filename,
                    local_dir=str(destination / directory)))
            except EntryNotFoundError:
                if filename in ("special_tokens_map.json", "tokenizer_config.json"):
                    continue
                raise
            outputs.append({"repo_id": repository, "filename": filename,
                            "relative_path": str(path.relative_to(destination)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes": path.stat().st_size})
    manifest = {"schema_version": 1, "backbone_weights_downloaded": False, "files": outputs}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


@contextmanager
def _local_backbone_constructors(assets_dir: Path):
    """Instantiate upstream architectures, then load the full CMI checkpoint.

    Temporary class-method replacements affect construction only. They are
    restored immediately; no shared source or pretrained file is modified.
    This preserves CMI's official model and scoring code while avoiding its
    otherwise redundant nested ``from_pretrained`` weight downloads.
    """
    from muq import MuQ, MuQMuLan
    from transformers import XLMRobertaConfig, XLMRobertaModel

    mulan = _read_config(assets_dir / "muq_mulan/config.json")
    muq = _read_config(assets_dir / "muq/config.json")
    roberta = XLMRobertaConfig.from_pretrained(
        str(assets_dir / "xlm_roberta"), local_files_only=True)

    def make_muq(cls, model_name, *args, **kwargs):
        if str(model_name) != "OpenMuQ/MuQ-large-msd-iter":
            raise ValueError(f"Unexpected MuQ architecture: {model_name}")
        return cls(config=muq)

    def make_mulan(cls, model_name, *args, **kwargs):
        if str(model_name) != "OpenMuQ/MuQ-MuLan-large":
            raise ValueError(f"Unexpected MuQ-MuLan architecture: {model_name}")
        return cls(config=mulan)

    def make_roberta(cls, model_name, *args, **kwargs):
        if str(model_name) not in ("xlm-roberta-base", "FacebookAI/xlm-roberta-base"):
            raise ValueError(f"Unexpected text architecture: {model_name}")
        return cls(roberta)

    with patch.object(MuQ, "from_pretrained", classmethod(make_muq)), \
         patch.object(MuQMuLan, "from_pretrained", classmethod(make_mulan)), \
         patch.object(XLMRobertaModel, "from_pretrained", classmethod(make_roberta)):
        yield


class CMIReward:
    """Official frozen CMI-RM, CPU-resident outside each scoring call."""

    def __init__(
        self,
        checkpoint: str | Path,
        repo_root: str | Path,
        assets_dir: str | Path,
        device: str = "cuda:0",
        sample_rate: int = 24000,
        max_duration: float | None = None,
        keep_model_on_device: bool = False,
    ) -> None:
        if sample_rate != 24000:
            raise ValueError("The supplied CMI-RM was trained at 24000 Hz")
        if max_duration is not None:
            raise ValueError("This training scores complete generated audio; use max_duration=None")
        self.checkpoint = Path(checkpoint).resolve()
        if self.checkpoint.is_dir():
            self.checkpoint = self.checkpoint / "model.safetensors"
        self.repo_root = Path(repo_root).resolve()
        self.assets_dir = Path(assets_dir).resolve()
        self.device = str(device)
        self.sample_rate = sample_rate
        self.max_duration = None
        self.keep_model_on_device = bool(keep_model_on_device)
        self.calibration: dict[str, Any] | None = None
        self.inference = None
        self.weight_validation: dict[str, Any] | None = None

    def load(self) -> None:
        if self.inference is not None:
            return
        for directory in (self.repo_root, self.repo_root / "models/cmi-rm/src"):
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        for filename in ("muq_mulan/config.json", "muq/config.json", "xlm_roberta/config.json"):
            if not (self.assets_dir / filename).is_file():
                raise FileNotFoundError(f"Missing CMI asset: {self.assets_dir / filename}")
        module = importlib.import_module("baselines.inference")
        if Path(module.__file__).resolve() != self.repo_root / "baselines/inference.py":
            raise RuntimeError("A different baselines.inference module is already imported")
        with _local_backbone_constructors(self.assets_dir):
            inference = module.RewardModelInference(
                checkpoint=str(self.checkpoint), device="cpu", sr=self.sample_rate,
                mode="final", bf16=False)
        # Tokenization is lazy upstream; bind the matching local tokenizer now.
        from transformers import AutoTokenizer
        inference.model.text_module._tokenizer = AutoTokenizer.from_pretrained(
            str(self.assets_dir / "xlm_roberta"), local_files_only=True,
            use_fast=True)
        inference.model.requires_grad_(False)
        inference.model.eval()

        # Upstream loads strict=False. Require complete key/shape coverage here
        # so newly initialized parameters can never silently score rewards.
        from safetensors import safe_open
        expected = inference.model.state_dict()
        with safe_open(str(self.checkpoint), framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            missing = sorted(set(expected) - actual)
            unexpected = sorted(actual - set(expected))
            mismatches = [key for key in set(expected) & actual
                          if tuple(expected[key].shape) != tuple(handle.get_slice(key).get_shape())]
        if missing or unexpected or mismatches:
            raise RuntimeError(f"CMI weight mismatch: missing={missing}, unexpected={unexpected}, "
                               f"shape_mismatches={mismatches}")
        self.weight_validation = {
            "checkpoint": str(self.checkpoint), "keys": len(expected),
            "missing": [], "unexpected": [], "shape_mismatches": [],
            "parameters": sum(p.numel() for p in inference.model.parameters()),
            "frozen": all(not p.requires_grad for p in inference.model.parameters()),
            "score_columns": list(SCORE_COLUMNS),
            "backbone_loading": "architecture configs + full CMI checkpoint; no extra pretrained weights",
        }
        self.inference = inference

    @contextmanager
    def _active(self):
        self.load()
        inference = self.inference
        assert inference is not None
        inference.device = self.device
        inference.bf16 = self.device.startswith("cuda")
        try:
            inference.model.to(self.device)
            inference.model.eval()
            yield inference
        finally:
            if not self.keep_model_on_device:
                inference.model.to("cpu")
                inference.device = "cpu"
                inference.bf16 = False
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()

    def score(
        self,
        waveforms: list[torch.Tensor] | torch.Tensor,
        captions: list[str],
        source_sample_rate: int = 48000,
        batch_size: int = 1,
    ) -> np.ndarray:
        """Return raw [N,2] alignment/quality; retain every generated sample.

        ACE outputs channel-first 48 kHz waveforms. Channel averaging and
        resampling occur on CPU before CMI's 30-second chunk encoder; longer
        songs are evaluated in full, rather than cropped to its default 30 s.
        """
        if source_sample_rate <= 0 or batch_size < 1:
            raise ValueError("source_sample_rate and batch_size must be positive")
        if isinstance(waveforms, torch.Tensor):
            if waveforms.ndim in (1, 2) and len(captions) == 1:
                waveforms = [waveforms]
            else:
                waveforms = list(waveforms)
        if len(waveforms) != len(captions):
            raise ValueError("Every waveform needs exactly one original caption")
        if not waveforms:
            return np.empty((0, 2), dtype=np.float64)
        from torchaudio.functional import resample
        inputs = []
        for waveform, caption in zip(waveforms, captions):
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("CMI text must be a nonempty original caption")
            waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
            if waveform.ndim == 2:
                if waveform.shape[0] > 8:
                    raise ValueError("Waveform must be channel-first [channels, samples]")
                waveform = waveform.mean(dim=0)
            if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
                raise ValueError("CMI audio must be a nonempty finite 1-D waveform")
            if source_sample_rate != self.sample_rate:
                waveform = resample(waveform, source_sample_rate, self.sample_rate)
            inputs.append({"audio": waveform.contiguous(), "text": caption,
                           "lyrics": "", "ref_audio": None})
        with self._active() as inference, torch.inference_mode():
            scores = inference.score_batch(inputs, batch_size=batch_size,
                                           max_dur=None, show_progress=False)
        result = np.asarray(scores, dtype=np.float64)
        if result.shape != (len(inputs), 2) or not np.isfinite(result).all():
            raise ValueError("CMI must return finite [alignment, quality] scores")
        return result

    def fit_calibration(self, scores: np.ndarray) -> dict[str, Any]:
        values = self._validate_scores(scores)
        if len(values) < 2:
            raise ValueError("At least two independent calibration scores are required")
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        if (std <= 1e-8).any():
            raise ValueError("A reward branch has zero calibration variance")
        self.calibration = {
            "schema_version": 1,
            "score_columns": list(SCORE_COLUMNS),
            "count": len(values), "mean": mean.tolist(), "std": std.tolist(),
            "ddof": 0, "weights": [0.5, 0.5],
            "normalization": "fixed independent calibration z-score; no clipping or sigmoid",
        }
        return self.calibration

    def load_calibration(self, path: str | Path) -> dict[str, Any]:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("score_columns") != list(SCORE_COLUMNS):
            raise ValueError("Calibration columns must be alignment, quality")
        if value.get("weights") != [0.5, 0.5]:
            raise ValueError("This run uses equal alignment and quality reward weights")
        mean, std = np.asarray(value["mean"]), np.asarray(value["std"])
        if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() \
                or not np.isfinite(std).all() or (std <= 1e-8).any():
            raise ValueError("Invalid fixed reward calibration")
        self.calibration = value
        return value

    @staticmethod
    def _validate_scores(scores: np.ndarray) -> np.ndarray:
        value = np.asarray(scores, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
            raise ValueError("Expected finite raw scores of shape [N,2]")
        return value

    def normalized_reward(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.calibration is None:
            raise RuntimeError("Fit or load the independent fixed calibration first")
        value = self._validate_scores(scores)
        zscores = (value - np.asarray(self.calibration["mean"])) / np.asarray(self.calibration["std"])
        rewards = zscores @ np.asarray(self.calibration["weights"])
        return rewards, zscores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-assets", required=True, metavar="DIRECTORY")
    args = parser.parse_args()
    print(json.dumps(prepare_assets(args.prepare_assets), indent=2))


if __name__ == "__main__":
    main()
