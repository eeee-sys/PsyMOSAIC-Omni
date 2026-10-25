"""Qwen2.5-Omni text Thinker runtime shared by GRPO and inference.

Evidence is supplied as text derived from the existing multimodal features;
raw encoders are unnecessary. The pretrained decoder weights are unchanged.
"""
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


from contextlib import contextmanager
from functools import wraps
import re

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoTokenizer, Qwen2_5OmniThinkerForConditionalGeneration


LORA_TARGETS = (r"model\.layers\.\d+\."
                r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))")


@contextmanager
def generation_last_token_projection(model):
    """Project only the final hidden token inside native autoregressive generation.

    GenerationMixin consumes logits[:, -1, :] at every step. Omni 4.57.3 does
    not implement logits_to_keep and otherwise projects every prefill position.
    A scoped pre-hook preserves the Linear module, weights, checkpoint keys,
    native decoder/RoPE and KV cache. Full ordinary forward and 2-D completion
    projections used by GRPO remain unchanged.
    """
    def last_hidden_only(module, inputs):
        if not torch.is_grad_enabled() and len(inputs) == 1 and inputs[0].ndim == 3:
            return (inputs[0][:, -1:, :],)
        return None

    handle = model.lm_head.register_forward_pre_hook(last_hidden_only)
    try:
        yield
    finally:
        handle.remove()


def enable_generation_last_token_projection(model):
    """Apply the scoped generation optimization once; no parameter replacement."""
    if getattr(model, "_generation_last_token_projection_enabled", False):
        return model
    original_generate = model.generate

    @wraps(original_generate)
    def generate(*args, **kwargs):
        with generation_last_token_projection(model):
            return original_generate(*args, **kwargs)

    model.generate = generate
    model._generation_last_token_projection_enabled = True
    return model


class FrozenCPUEmbedding(torch.nn.Embedding):
    """Exact optional CPU lookup for frozen token embeddings."""
    def __init__(self, original):
        super().__init__(original.num_embeddings, original.embedding_dim,
                         padding_idx=original.padding_idx,
                         _weight=original.weight.detach().cpu(), _freeze=True)

    def _apply(self, fn, recurse=True):
        return self

    def forward(self, input_ids):
        return super().forward(input_ids.to("cpu")).to(input_ids.device)


def load_text_thinker(base_model, cpu_offload_embedding=False):
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True)
    removed = {}
    for name in ("audio_tower", "visual"):
        encoder = getattr(model, name, None)
        if encoder is not None:
            removed[name] = sum(p.numel() * p.element_size() for p in encoder.parameters())
            setattr(model, name, None)
    model._text_only_removed_encoder_bytes = removed
    if cpu_offload_embedding:
        model.set_input_embeddings(FrozenCPUEmbedding(model.get_input_embeddings()))
    enable_generation_last_token_projection(model)
    return model


def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id != 151645:
        raise ValueError(f"Unexpected Qwen ChatML EOS: {tokenizer.eos_token_id}")
    return tokenizer


def add_decoder_lora(model, config):
    targets = [(name, module) for name, module in model.named_modules()
               if re.fullmatch(LORA_TARGETS, name)]
    if len(targets) != 196:
        raise ValueError(f"Expected 196 decoder projections; got {len(targets)}")
    expected = config.get("lora_rank", 32) * sum(
        module.in_features + module.out_features for _, module in targets)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=config.get("lora_rank", 32),
        lora_alpha=config.get("lora_alpha", 64),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=LORA_TARGETS, bias="none"))
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != expected or any(".model.layers." not in n or ".lora_" not in n for n in trainable):
        raise ValueError("LoRA trainable tensors differ from the decoder-only specification")
    return model, {"trainable_parameters": count,
                   "trainable_tensors": len(trainable),
                   "decoder_lora_modules": len(targets),
                   "lora_target_regex": LORA_TARGETS,
                   "trainable_parameter_names": trainable}


def load_named_adapters(base_model, adapters, device="cuda", cpu_offload_embedding=False):
    """Load one frozen base and separately selectable Generator/Diagnosis adapters."""
    model = load_text_thinker(base_model, cpu_offload_embedding)
    for index, (name, path) in enumerate(adapters.items()):
        if index == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name, is_trainable=False)
        else:
            model.load_adapter(path, adapter_name=name, is_trainable=False)
    model.to(device).eval()
    return model, load_tokenizer(base_model)


def generate_text(model, tokenizer, messages, max_new_tokens=384, adapter_name=None,
                  do_sample=False, context_limit=32768, **generation_kwargs):
    """Generate a complete chat response with explicit EOS and no prompt truncation."""
    return generate_text_batch(model, tokenizer, [messages], max_new_tokens,
                               adapter_name, do_sample, context_limit,
                               **generation_kwargs)[0]


def generate_text_batch(model, tokenizer, messages_batch, max_new_tokens=384,
                        adapter_name=None, do_sample=False, context_limit=32768,
                        **generation_kwargs):
    """Batched counterpart; padding is removed independently after each EOS."""
    if adapter_name is not None:
        model.set_adapter(adapter_name)
    encoded = tokenizer.apply_chat_template(
        messages_batch, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, padding=True, truncation=False)
    prompt_length = encoded["input_ids"].shape[1]
    if prompt_length + max_new_tokens > context_limit:
        raise ValueError(f"Full prompt ({prompt_length}) plus completion exceeds context_limit={context_limit}")
    device = next(p.device for p in model.parameters() if p.device.type != "cpu")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            result = model.generate(
                **encoded, max_new_tokens=max_new_tokens, do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                use_cache=True, **generation_kwargs)
    finally:
        model.train(was_training)
    rows = []
    for index in range(result.shape[0]):
        completion_ids = result[index, prompt_length:]
        eos_positions = (completion_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        ended = bool(eos_positions.numel())
        if ended:
            completion_ids = completion_ids[:eos_positions[0].item() + 1]
        rows.append({"text": tokenizer.decode(completion_ids, skip_special_tokens=True),
                     "generated_tokens": int(completion_ids.numel()),
                     "ended_with_eos": ended,
                     "prompt_tokens": int(encoded["attention_mask"][index].sum().item())})
    return rows
