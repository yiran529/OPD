from __future__ import annotations

from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opd.config import TrainConfig


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _ensure_flash_linear_attention_importable() -> None:
    try:
        import fla  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "flash-linear-attention is not importable. Install fla-org/flash-linear-attention first."
        ) from exc


def build_model_and_tokenizer(cfg: TrainConfig, device: torch.device) -> Tuple[torch.nn.Module, object]:
    _ensure_flash_linear_attention_importable()

    model_dtype = resolve_dtype(cfg.dtype)
    model_id = cfg.model_name
    tokenizer_id = cfg.tokenizer_name or model_id

    model_config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=cfg.trust_remote_code,
    )

    architectures = getattr(model_config, "architectures", None) or []
    if cfg.expected_architecture and cfg.expected_architecture not in architectures:
        raise ValueError(
            f"Expected architecture {cfg.expected_architecture} not found in config.architectures={architectures}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=model_dtype,
    )
    model.to(device)

    return model, tokenizer
