"""Persistent, local-only text inference worker for psychotherapy evaluation.

The wire protocol is one JSON object per line. Model diagnostics go to stderr;
stdout is reserved for the readiness event and request responses. Clinical text
is never logged. Context removal requires an explicit request policy and is
reported by original message index in every response.
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


import argparse
from contextlib import contextmanager, redirect_stdout
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


CBT_NATIVE_INSTRUCTION = (
    "你是一名经验丰富的心理咨询师，专长于认知行为疗法, "
    "以心理咨询师的身份回答以下问题。"
    # This Chinese-trained baseline ignored the English common-prompt language
    # constraint in the initial smoke. Keep the author's role/template and state
    # the evaluation language explicitly in the same language as its native role.
    "\n请全程只用英语回答患者，不要使用中文。"
)


class RequestError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


def validate_messages(value):
    if not isinstance(value, list) or not value:
        raise RequestError("invalid_messages", "A nonempty messages list is required")
    result = []
    for index, message in enumerate(value):
        if (not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)):
            raise RequestError("invalid_messages", "Invalid role or content", {"message_index": index})
        result.append({"role": message["role"], "content": message["content"]})
    if result[-1]["role"] != "user":
        raise RequestError("invalid_messages", "The final message must be a user message")
    return result


def serialize_cbt(messages):
    """Extend the author's exact Human/Assistant format to a dialogue history."""
    systems = [m["content"] for m in messages if m["role"] == "system"]
    prefix = CBT_NATIVE_INSTRUCTION
    if systems:
        prefix += "\n" + "\n\n".join(systems)
    lines = [prefix]
    for message in messages:
        if message["role"] != "system":
            label = "Human" if message["role"] == "user" else "Assistant"
            lines.append(label + ": " + message["content"])
    return "\n".join(lines) + "\nAssistant: "


def serialize_messages(tokenizer, backend, messages):
    if backend == "cbt":
        return serialize_cbt(messages)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_messages(tokenizer, backend, messages):
    # ChatML already contains its special tokens. CBT uses the model author's
    # normal tokenizer call (whose configured add_bos_token is false).
    text = serialize_messages(tokenizer, backend, messages)
    options = {"return_tensors": "pt", "truncation": False}
    if backend in ("ours", "omni_base"):
        options["add_special_tokens"] = False
    encoded = tokenizer(text, **options)
    return encoded, int(encoded["input_ids"].shape[-1])


def fit_context(tokenizer, backend, messages, max_new_tokens, context_limit,
                history_policy="error", protected_message_indices=None):
    if history_policy not in ("error", "drop_oldest_pairs"):
        raise RequestError("invalid_history_policy", "Unknown history policy")
    indices = list(range(len(messages)))
    protected = set(protected_message_indices or [])
    if any(not isinstance(i, int) or i < 0 or i >= len(messages) for i in protected):
        raise RequestError("invalid_protected_indices", "Protected indices must reference existing messages")
    protected.update(i for i, m in enumerate(messages) if m["role"] == "system")
    protected.add(len(messages) - 1)
    encoded, input_tokens = tokenize_messages(tokenizer, backend, messages)
    original_input_tokens = input_tokens
    while input_tokens + max_new_tokens > context_limit:
        if history_policy == "error":
            raise RequestError("context_overflow", "Full prompt and completion exceed the context limit",
                               {"input_tokens": input_tokens, "max_new_tokens": max_new_tokens,
                                "context_limit": context_limit})
        drop_positions = None
        for position in range(len(indices) - 1):
            i, j = indices[position:position + 2]
            if (i not in protected and j not in protected
                    and messages[i]["role"] == "user" and messages[j]["role"] == "assistant"):
                drop_positions = (position, position + 1)
                break
        if drop_positions is None:
            raise RequestError("protected_context_overflow", "Protected context and completion exceed the limit",
                               {"input_tokens": input_tokens, "max_new_tokens": max_new_tokens,
                                "context_limit": context_limit,
                                "dropped_message_indices": sorted(set(range(len(messages))) - set(indices))})
        indices = [i for position, i in enumerate(indices) if position not in drop_positions]
        encoded, input_tokens = tokenize_messages(tokenizer, backend, [messages[i] for i in indices])
    report = {"context_limit": context_limit, "original_input_tokens": original_input_tokens,
              "input_tokens": input_tokens, "history_policy": history_policy,
              "dropped_message_indices": sorted(set(range(len(messages))) - set(indices)),
              "kept_message_indices": indices, "truncated": len(indices) != len(messages)}
    return encoded, report


@contextmanager
def last_token_projection(model):
    """Native generation consumes only the final token's vocabulary logits."""
    def last_hidden_only(module, inputs):
        if len(inputs) == 1 and inputs[0].ndim == 3:
            return (inputs[0][:, -1:, :],)
        return None
    handle = model.get_output_embeddings().register_forward_pre_hook(last_hidden_only)
    try:
        yield
    finally:
        handle.remove()


def load_qwen(args, torch):
    """Use the same original Thinker; attach a LoRA only for the SFT condition."""
    from transformers import AutoTokenizer
    runtime_path = Path(args.source_project) / "training" / "model.py"
    if not runtime_path.is_file():
        raise RuntimeError("The audited SFT model runtime is missing")
    spec = importlib.util.spec_from_file_location("psychotherapy_sft_model_runtime", runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    base = runtime.load_text_thinker(args.base_model, cpu_offload_embedding=False)
    base.config.use_cache = True
    if args.backend == "ours":
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
    elif args.backend == "omni_base":
        model = base
    else:
        raise ValueError("Unsupported Qwen backend")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model.requires_grad_(False)
    model = model.to(args.device).eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, min(args.context_limit or 16384, 16384)


def load_cbt(args, torch):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    # The checkpoint's AutoModelForCausalLM map references an upstream remote
    # repository. AutoModel selects the included identical causal-LM class.
    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True,
                                      torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    model = model.to(args.device).eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = int(config.pad_token_id if config.pad_token_id is not None else tokenizer.eos_token_id)
    model.config.use_cache = True
    context_limit = int(config.max_position_embeddings)
    return model, tokenizer, min(args.context_limit or context_limit, context_limit)


def validate_generation_request(request, context_limit):
    if not isinstance(request, dict):
        raise RequestError("invalid_request", "The request must be a JSON object")
    max_new_tokens = request.get("max_new_tokens", 256)
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens < context_limit:
        raise RequestError("invalid_max_new_tokens", "Invalid maximum completion length")
    temperature, top_p = float(request.get("temperature", 0.7)), float(request.get("top_p", 0.9))
    if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise RequestError("invalid_sampling", "Temperature and top_p must be finite and in range")
    return max_new_tokens, temperature, top_p


def generate(model, tokenizer, torch, args, context_limit, request):
    max_new_tokens, temperature, top_p = validate_generation_request(request, context_limit)
    messages = validate_messages(request.get("messages"))
    encoded, context = fit_context(tokenizer, args.backend, messages, max_new_tokens, context_limit,
                                  request.get("history_policy", "error"),
                                  request.get("protected_message_indices"))
    if request.get("op", "generate") == "count_tokens":
        return {"input_tokens": context["input_tokens"], "max_new_tokens": max_new_tokens,
                "context": context}
    if request.get("op", "generate") != "generate":
        raise RequestError("invalid_operation", "Unknown worker operation")
    seed = int(request.get("seed", 42))
    torch.manual_seed(seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    options = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
               "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
               "use_cache": True}
    if temperature > 0:
        options.update(temperature=temperature, top_p=top_p)
    started = time.monotonic()
    base = model.get_base_model() if args.backend == "ours" else model
    with torch.inference_mode(), last_token_projection(base):
        outputs = model.generate(**encoded, **options)
    completion = outputs[0, context["input_tokens"]:]
    completion_ids = completion.tolist()
    eos_id = tokenizer.eos_token_id
    eos_set = set(eos_id if isinstance(eos_id, list) else [eos_id])
    ended_with_eos = any(i in eos_set for i in completion_ids)
    text = tokenizer.decode(completion, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    finish_reason = "stop" if ended_with_eos else "length"
    if args.backend == "cbt":
        boundaries = [text.find(marker) for marker in ("\nHuman:", "\nAssistant:") if marker in text]
        if boundaries:
            text = text[:min(boundaries)]
            finish_reason = "role_boundary"
    return {"text": text.strip(), "finish_reason": finish_reason,
            "input_tokens": context["input_tokens"], "output_tokens": len(completion_ids),
            "ended_with_eos": ended_with_eos, "context": context,
            "elapsed_seconds": round(time.monotonic() - started, 3), "seed": seed}


def send(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("ours", "cbt", "omni_base"))
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--context-limit", type=int)
    args = parser.parse_args()
    if args.backend == "ours" and not args.adapter:
        parser.error("--adapter is required for the SFT model")
    if args.backend != "ours" and args.adapter:
        parser.error("--adapter is only supported for the SFT model")
    try:
        with redirect_stdout(sys.stderr):
            import torch
            from transformers import __version__ as transformers_version
            torch.set_num_threads(4)
            model, tokenizer, context_limit = (load_cbt(args, torch) if args.backend == "cbt"
                                               else load_qwen(args, torch))
        send({"event": "ready", "backend": args.backend, "context_limit": context_limit,
              "device": args.device, "transformers_version": transformers_version,
              "torch_version": torch.__version__, "base_model": args.base_model,
              "adapter": args.adapter if args.backend == "ours" else None,
              "has_lora_adapter": args.backend == "ours", "clinical_text_logging": False})
    except Exception as exc:
        print("Worker initialization failed: " + type(exc).__name__, file=sys.stderr, flush=True)
        raise
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id") if isinstance(request, dict) else None
            if isinstance(request, dict) and request.get("op") == "shutdown":
                send({"id": request_id, "event": "shutdown"})
                break
            with redirect_stdout(sys.stderr):
                result = generate(model, tokenizer, torch, args, context_limit, request)
            send({"id": request_id, **result})
        except RequestError as exc:
            send({"id": request_id, "error": {"code": exc.code, "message": str(exc), **exc.details}})
        except Exception as exc:
            # Exceptions can echo input contents. Report the class only.
            print("Worker request failed: " + type(exc).__name__, file=sys.stderr, flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            send({"id": request_id, "error": {"code": "generation_failed", "exception": type(exc).__name__}})


if __name__ == "__main__":
    main()
