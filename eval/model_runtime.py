from __future__ import annotations

from typing import Tuple

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
    train_cfg = load_config(eval_cfg.train_config_path)
    device = resolve_device()

    model, tokenizer = build_model_and_tokenizer(cfg=train_cfg, device=device)
    loaded_step = load_model_checkpoint(
        checkpoint_path=eval_cfg.checkpoint_path,
        model=model,
        device=device,
    )

    model.eval()
    return model, tokenizer, train_cfg, loaded_step, device
