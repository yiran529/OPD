from __future__ import annotations

import importlib
from typing import List, Sequence, Tuple

import torch


_SUPPORTED_MODEL_CLASSES = {
    "conditional_generation": "Qwen3_5ForConditionalGeneration",
    "causal_lm": "Qwen3_5ForCausalLM",
}

_DEFAULT_QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _import_transformers():
    try:
        return importlib.import_module("transformers")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "transformers is not importable. Install a recent transformers build with Qwen3.5 support."
        ) from exc


def _resolve_qwen_model_class(model_class: str):
    if model_class not in _SUPPORTED_MODEL_CLASSES:
        raise ValueError(
            f"Unsupported model_class: {model_class}. "
            f"Expected one of {sorted(_SUPPORTED_MODEL_CLASSES.keys())}."
        )

    transformers = _import_transformers()
    class_name = _SUPPORTED_MODEL_CLASSES[model_class]
    if not hasattr(transformers, class_name):
        raise RuntimeError(
            f"Installed transformers does not expose {class_name}. "
            "Qwen3.5 requires a newer transformers version."
        )
    return getattr(transformers, class_name)


def _build_tokenizer(
    model_name: str,
    tokenizer_name: str | None,
    trust_remote_code: bool,
):
    transformers = _import_transformers()
    tokenizer_id = tokenizer_name or model_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer must expose apply_chat_template for QwenOPSD formatting")
    return tokenizer


def _assert_qwen_model(model: torch.nn.Module, model_class: str) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("Loaded model is missing config")

    model_type = getattr(config, "model_type", None)
    if model_type != "qwen3_5":
        raise RuntimeError(f"Expected model_type=qwen3_5, got {model_type!r}")

    expected_class_name = _SUPPORTED_MODEL_CLASSES[model_class]
    actual_class_name = model.__class__.__name__
    if actual_class_name != expected_class_name:
        raise RuntimeError(
            f"Loaded model class mismatch: expected={expected_class_name}, got={actual_class_name}"
        )


def _linear_module_names(model: torch.nn.Module) -> List[str]:
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.append(module_name)
    return names


def _resolve_lora_target_modules(
    model: torch.nn.Module,
    requested_modules: Sequence[str],
) -> List[str]:
    linear_names = _linear_module_names(model)
    if not linear_names:
        raise RuntimeError("LoRA requested, but the model does not expose any torch.nn.Linear modules")

    requested = list(dict.fromkeys(requested_modules or _DEFAULT_QWEN_LORA_TARGETS))
    missing: list[str] = []
    for target in requested:
        has_match = any(
            module_name == target or module_name.endswith(f".{target}")
            for module_name in linear_names
        )
        if not has_match:
            missing.append(target)

    if missing:
        available_leaf_names = sorted({name.rsplit(".", 1)[-1] for name in linear_names})
        raise RuntimeError(
            "Some lora_target_modules did not match any Linear module. "
            f"missing={missing} available_leaf_names={available_leaf_names}"
        )
    return requested


def _assert_lora_trainable_params(model: torch.nn.Module) -> None:
    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_names:
        raise RuntimeError("LoRA wrapping produced zero trainable parameters")

    unexpected = [name for name in trainable_names if "lora_" not in name]
    if unexpected:
        preview = unexpected[:20]
        raise RuntimeError(
            "LoRA mode expected only LoRA adapter params to be trainable, "
            f"but found non-LoRA trainable params={preview} total_unexpected={len(unexpected)}"
        )


def _maybe_wrap_with_lora(
    model: torch.nn.Module,
    finetune_mode: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
) -> torch.nn.Module:
    if finetune_mode != "lora":
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("peft is not importable. Install `peft` to use finetune_mode=lora.") from exc

    target_modules = _resolve_lora_target_modules(
        model=model,
        requested_modules=lora_target_modules,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    _assert_lora_trainable_params(model)
    return model


@torch.no_grad()
def _run_startup_sanity(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> None:
    input_ids = torch.tensor(
        [[tokenizer.eos_token_id, tokenizer.eos_token_id]],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Startup sanity check failed: logits missing")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Startup sanity check failed: logits contain non-finite values")

    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("Startup sanity check failed: past_key_values missing with use_cache=True")


def build_model_and_tokenizer(
    model_name: str,
    tokenizer_name: str | None,
    model_class: str,
    trust_remote_code: bool,
    dtype: str,
    device: torch.device,
    finetune_mode: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str],
) -> Tuple[torch.nn.Module, object]:
    tokenizer = _build_tokenizer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        trust_remote_code=trust_remote_code,
    )
    model_dtype = resolve_dtype(dtype)
    qwen_model_class = _resolve_qwen_model_class(model_class)
    model = qwen_model_class.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
    )
    _assert_qwen_model(model=model, model_class=model_class)
    model = _maybe_wrap_with_lora(
        model=model,
        finetune_mode=finetune_mode,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    model.to(device)
    _run_startup_sanity(model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer


def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

