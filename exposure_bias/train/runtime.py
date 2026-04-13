from __future__ import annotations

from dataclasses import dataclass

import torch

from exposure_bias.model_loader import build_model_and_tokenizer
from exposure_bias.train.checkpoint import load_exposure_bias_train_checkpoint
from exposure_bias.train.config import ExposureBiasTrainConfig


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainRuntimeBundle:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    loaded_step: int


def build_train_runtime(cfg: ExposureBiasTrainConfig) -> TrainRuntimeBundle:
    device = resolve_device()
    model, tokenizer = build_model_and_tokenizer(cfg=cfg, device=device)

    loaded_step = -1
    if cfg.init_checkpoint_path:
        loaded_step = load_exposure_bias_train_checkpoint(
            checkpoint_path=cfg.init_checkpoint_path,
            model=model,
            device=device,
        )

    model.train()
    return TrainRuntimeBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        loaded_step=loaded_step,
    )
