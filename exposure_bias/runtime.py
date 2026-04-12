from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from eval.checkpoint_loader import load_model_checkpoint
from exposure_bias.config import ExposureBiasEvalConfig, resolve_train_config
from opd.config import TrainConfig
from opd.model_loader import build_model_and_tokenizer


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_model_max_length(model_config: Any, fallback: int) -> int:
    candidate_keys = (
        "max_position_embeddings",
        "n_positions",
        "n_ctx",
        "seq_len",
    )
    for key in candidate_keys:
        value = getattr(model_config, key, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return int(value)
    return int(fallback)


@dataclass
class RuntimeBundle:
    model: torch.nn.Module
    tokenizer: object
    train_cfg: TrainConfig
    device: torch.device
    loaded_step: int
    model_max_length: int


def build_runtime(cfg: ExposureBiasEvalConfig) -> RuntimeBundle:
    train_cfg = resolve_train_config(cfg)
    device = resolve_device()

    model, tokenizer = build_model_and_tokenizer(cfg=train_cfg, device=device)

    loaded_step = -1
    if cfg.checkpoint_path:
        loaded_step = load_model_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            model=model,
            device=device,
        )

    model.eval()
    fallback_max_len = cfg.prefix_len + cfg.rollout_len
    model_max_length = _resolve_model_max_length(
        model_config=getattr(model, "config", None),
        fallback=fallback_max_len,
    )
    return RuntimeBundle(
        model=model,
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        device=device,
        loaded_step=loaded_step,
        model_max_length=model_max_length,
    )
