from __future__ import annotations

from typing import Tuple

from opd.config import TrainConfig
from opd.model_loader import build_model_and_tokenizer as build_fla_model_and_tokenizer
import torch


def build_model_and_tokenizer(
    model_impl: str,
    train_cfg: TrainConfig,
    device: torch.device,
) -> Tuple[torch.nn.Module, object]:
    if model_impl != "fla":
        raise ValueError(f"Unsupported model_impl: {model_impl}")
    return build_fla_model_and_tokenizer(cfg=train_cfg, device=device)
