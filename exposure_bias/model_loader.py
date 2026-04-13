from __future__ import annotations

from typing import List, Tuple

import torch
from transformers import AutoTokenizer

from exposure_bias.train.config import ExposureBiasTrainConfig
from opd.model_loader import (
    _assert_clean_weight_loading,
    _assert_expected_model_impl,
    _assert_lora_trainable_params,
    _ensure_flash_linear_attention_importable,
    _resolve_expected_model_class,
    _run_startup_sanity,
    resolve_dtype,
)


def _resolve_last_n_block_target_modules(
    model: torch.nn.Module,
    last_n_blocks: int,
) -> List[str]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    assert layers is not None, "model must expose model.layers for last-n-block LoRA"
    assert isinstance(layers, (list, torch.nn.ModuleList)), "model.layers must be a module list"
    num_layers = len(layers)
    assert num_layers > 0, "model.layers must be non-empty"
    assert last_n_blocks <= num_layers, (
        f"lora_last_n_blocks exceeds layer count: last_n_blocks={last_n_blocks} num_layers={num_layers}"
    )

    start_idx = num_layers - last_n_blocks
    target_modules: list[str] = []
    prefix_roots = [f"model.layers.{layer_idx}." for layer_idx in range(start_idx, num_layers)]
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(module_name.startswith(prefix) for prefix in prefix_roots):
            target_modules.append(module_name)

    assert target_modules, f"no Linear modules found in the last {last_n_blocks} blocks"
    return target_modules


def _maybe_wrap_with_lora(
    model: torch.nn.Module,
    cfg: ExposureBiasTrainConfig,
) -> torch.nn.Module:
    if cfg.finetune_mode != "lora":
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PEFT is not importable. Install `peft` to use finetune_mode=lora.") from exc

    target_modules = _resolve_last_n_block_target_modules(
        model=model,
        last_n_blocks=cfg.lora_last_n_blocks,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    _assert_lora_trainable_params(model)
    return model


def build_model_and_tokenizer(
    cfg: ExposureBiasTrainConfig,
    device: torch.device,
) -> Tuple[torch.nn.Module, object]:
    _ensure_flash_linear_attention_importable()

    model_dtype = resolve_dtype(cfg.dtype)
    tokenizer_id = cfg.tokenizer_name or cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = _resolve_expected_model_class(cfg=cfg)
    model, loading_info = model_class.from_pretrained(
        cfg.model_name,
        torch_dtype=model_dtype,
        output_loading_info=True,
    )
    architectures = getattr(model.config, "architectures", None) or []
    if cfg.expected_architecture and cfg.expected_architecture not in architectures:
        raise ValueError(
            f"Expected architecture {cfg.expected_architecture} not found in config.architectures={architectures}"
        )

    _assert_expected_model_impl(model=model, expected_architecture=cfg.expected_architecture)
    _assert_clean_weight_loading(loading_info=loading_info)

    model = _maybe_wrap_with_lora(model=model, cfg=cfg)
    model.to(device)
    _run_startup_sanity(model=model, pad_token_id=tokenizer.pad_token_id, device=device)
    return model, tokenizer
