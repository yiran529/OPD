from __future__ import annotations

from dataclasses import dataclass

import torch

from eval.checkpoint_loader import load_model_checkpoint
from memory_pollution.config import MemoryPollutionEvalConfig
from memory_pollution.model_loader import build_model_and_tokenizer
from opd.config import TrainConfig, load_config as load_train_config


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class RuntimeBundle:
    model: torch.nn.Module
    tokenizer: object
    train_cfg: TrainConfig
    device: torch.device
    loaded_step: int
    supports_state_drift: bool
    state_key: str


def build_runtime(cfg: MemoryPollutionEvalConfig) -> RuntimeBundle:
    train_cfg = load_train_config(cfg.train_config_path)
    device = resolve_device()

    model, tokenizer = build_model_and_tokenizer(
        model_impl=cfg.model_impl,
        train_cfg=train_cfg,
        device=device,
    )

    loaded_step = -1
    if cfg.checkpoint_path:
        loaded_step = load_model_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            model=model,
            device=device,
        )

    model.eval()
    state_key = cfg.state_key or train_cfg.state_key
    return RuntimeBundle(
        model=model,
        tokenizer=tokenizer,
        train_cfg=train_cfg,
        device=device,
        loaded_step=loaded_step,
        supports_state_drift=bool(cfg.collect_state_drift),
        state_key=state_key,
    )
