from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from eval.checkpoint_loader import load_model_checkpoint
from exposure_bias.eval.config import ExposureBiasEvalConfig, resolve_train_config
from exposure_bias.model_loader import build_model_and_tokenizer as build_train_model_and_tokenizer
from exposure_bias.train.checkpoint import load_exposure_bias_train_checkpoint
from exposure_bias.train.config import ExposureBiasTrainConfig
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


def _load_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert isinstance(state, dict), "checkpoint must deserialize to a dict"
    return state


def _resolve_lora_eval_train_cfg(
    cfg: ExposureBiasEvalConfig,
    checkpoint_state: dict[str, Any],
) -> ExposureBiasTrainConfig:
    raw_cfg = checkpoint_state.get("config")
    assert isinstance(raw_cfg, dict), "exposure_bias training checkpoint must contain config"

    train_cfg = ExposureBiasTrainConfig(**raw_cfg)
    if cfg.model_name is not None:
        assert cfg.model_name == train_cfg.model_name, (
            f"eval config model_name does not match LoRA checkpoint: "
            f"eval={cfg.model_name} checkpoint={train_cfg.model_name}"
        )
    if cfg.model_name is not None:
        train_cfg.model_name = cfg.model_name
    if cfg.tokenizer_name is not None and train_cfg.tokenizer_name is not None:
        assert cfg.tokenizer_name == train_cfg.tokenizer_name, (
            f"eval config tokenizer_name does not match LoRA checkpoint: "
            f"eval={cfg.tokenizer_name} checkpoint={train_cfg.tokenizer_name}"
        )
    if cfg.tokenizer_name is not None:
        train_cfg.tokenizer_name = cfg.tokenizer_name
    if cfg.expected_architecture is not None:
        assert cfg.expected_architecture == train_cfg.expected_architecture, (
            f"eval config expected_architecture does not match LoRA checkpoint: "
            f"eval={cfg.expected_architecture} checkpoint={train_cfg.expected_architecture}"
        )
    if cfg.expected_architecture is not None:
        train_cfg.expected_architecture = cfg.expected_architecture
    train_cfg.trust_remote_code = cfg.trust_remote_code
    train_cfg.dtype = cfg.dtype
    return train_cfg


def build_runtime(cfg: ExposureBiasEvalConfig) -> RuntimeBundle:
    train_cfg = resolve_train_config(cfg)
    device = resolve_device()

    loaded_step = -1
    checkpoint_state = None
    if cfg.checkpoint_path:
        checkpoint_state = _load_checkpoint_metadata(cfg.checkpoint_path)

    checkpoint_cfg = checkpoint_state.get("config") if isinstance(checkpoint_state, dict) else None
    checkpoint_finetune_mode = checkpoint_cfg.get("finetune_mode") if isinstance(checkpoint_cfg, dict) else None

    if checkpoint_finetune_mode == "lora":
        lora_train_cfg = _resolve_lora_eval_train_cfg(cfg=cfg, checkpoint_state=checkpoint_state)
        model, tokenizer = build_train_model_and_tokenizer(cfg=lora_train_cfg, device=device)
        loaded_step = load_exposure_bias_train_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            model=model,
            device=device,
        )
        train_cfg.model_name = lora_train_cfg.model_name
        train_cfg.tokenizer_name = lora_train_cfg.tokenizer_name
        train_cfg.expected_architecture = lora_train_cfg.expected_architecture
        train_cfg.trust_remote_code = lora_train_cfg.trust_remote_code
        train_cfg.dtype = lora_train_cfg.dtype
    else:
        model, tokenizer = build_model_and_tokenizer(cfg=train_cfg, device=device)
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
