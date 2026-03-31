from __future__ import annotations

from typing import Optional, Tuple

import torch

from eval.checkpoint_loader import load_model_checkpoint
from eval.config import EvalConfig
from opd.config import TrainConfig, load_config
from opd.model_loader import build_model_and_tokenizer


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_eval_model_and_tokenizer(
    eval_cfg: EvalConfig,
) -> Tuple[torch.nn.Module, object, TrainConfig, int, torch.device]:
    return build_model_and_tokenizer_from_paths(
        train_config_path=eval_cfg.train_config_path,
        checkpoint_path=eval_cfg.checkpoint_path,
    )


def build_model_and_tokenizer_from_paths(
    train_config_path: str,
    checkpoint_path: Optional[str],
) -> Tuple[torch.nn.Module, object, TrainConfig, int, torch.device]:
    train_cfg = load_config(train_config_path)
    device = resolve_device()

    model, tokenizer = build_model_and_tokenizer(cfg=train_cfg, device=device)
    loaded_step = -1
    if checkpoint_path:
        loaded_step = load_model_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
        )

    model.eval()
    return model, tokenizer, train_cfg, loaded_step, device
