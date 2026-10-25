"""Load the original Omni Thinker for text-only SFT without unused encoders."""
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


import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration


class FrozenCPUEmbedding(torch.nn.Embedding):
    """Optional exact frozen lookup that transfers selected vectors from CPU."""
    def __init__(self, original):
        super().__init__(original.num_embeddings, original.embedding_dim,
                         padding_idx=original.padding_idx, max_norm=original.max_norm,
                         norm_type=original.norm_type, scale_grad_by_freq=original.scale_grad_by_freq,
                         sparse=original.sparse, _weight=original.weight.detach().cpu(), _freeze=True)

    def _apply(self, fn, recurse=True):
        return self

    def forward(self, input_ids):
        return super().forward(input_ids.to("cpu")).to(input_ids.device)


def load_text_thinker(base_model, cpu_offload_embedding=False):
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    removed = {}
    for name in ("audio_tower", "visual"):
        encoder = getattr(model, name, None)
        if encoder is not None:
            removed[name] = sum(parameter.numel() * parameter.element_size() for parameter in encoder.parameters())
            setattr(model, name, None)
    model._text_only_removed_encoder_bytes = removed
    model.requires_grad_(False)
    model.config.use_cache = False
    if cpu_offload_embedding:
        model.set_input_embeddings(FrozenCPUEmbedding(model.get_input_embeddings()))
    return model
