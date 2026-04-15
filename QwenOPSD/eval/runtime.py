from __future__ import annotations

from dataclasses import dataclass

import torch

from QwenOPSD.checkpoint import load_model_checkpoint
from QwenOPSD.model_loader import build_model_and_tokenizer, freeze_model
from QwenOPSD.eval.config import QwenOPSDEvalConfig


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class EvalRuntimeBundle:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    loaded_step: int


def build_eval_runtime(cfg: QwenOPSDEvalConfig) -> EvalRuntimeBundle:
    device = resolve_device()
    model, tokenizer = build_model_and_tokenizer(
        model_name=cfg.model_name,
        tokenizer_name=cfg.tokenizer_name,
        model_class=cfg.model_class,
        trust_remote_code=cfg.trust_remote_code,
        dtype=cfg.dtype,
        device=device,
        finetune_mode=cfg.finetune_mode,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
    )
    loaded_step = -1
    if cfg.checkpoint_path:
        loaded_step = load_model_checkpoint(
            checkpoint_path=cfg.checkpoint_path,
            model=model,
            device=device,
        )
    freeze_model(model)
    return EvalRuntimeBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        loaded_step=loaded_step,
    )

