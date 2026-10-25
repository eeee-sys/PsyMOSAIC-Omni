"""Load the Omni text Thinker without unused audio/vision encoder tensors."""
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
    """Exact frozen token lookup on CPU; transfer only selected embedding vectors."""
    def __init__(self, original):
        super().__init__(original.num_embeddings, original.embedding_dim,
                         padding_idx=original.padding_idx, max_norm=original.max_norm,
                         norm_type=original.norm_type, scale_grad_by_freq=original.scale_grad_by_freq,
                         sparse=original.sparse, _weight=original.weight.detach().cpu(), _freeze=True)

    def _apply(self, fn, recurse=True):
        # Trainer's model.to(cuda) keeps this frozen table on CPU.
        return self

    def forward(self, input_ids):
        return super().forward(input_ids.to("cpu")).to(input_ids.device)


def load_text_thinker(base_model, cpu_offload_embedding=False):
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    # These encoders are never called for transcript-only SFT or generation.
    # Adapter loading still uses the unchanged original base checkpoint.
    removed = {}
    for name in ("audio_tower", "visual"):
        encoder = getattr(model, name, None)
        if encoder is not None:
            removed[name] = sum(p.numel()*p.element_size() for p in encoder.parameters())
            setattr(model, name, None)
    model._text_only_removed_encoder_bytes = removed
    if cpu_offload_embedding:
        model.set_input_embeddings(FrozenCPUEmbedding(model.get_input_embeddings()))
    return model
