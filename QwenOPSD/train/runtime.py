from __future__ import annotations

from dataclasses import dataclass

import torch

from QwenOPSD.model_loader import build_model_and_tokenizer, freeze_model
from QwenOPSD.train.config import QwenOPSDTrainConfig


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainRuntimeBundle:
    student_model: torch.nn.Module
    teacher_model: torch.nn.Module
    tokenizer: object
    device: torch.device


def build_train_runtime(cfg: QwenOPSDTrainConfig) -> TrainRuntimeBundle:
    device = resolve_device()
    student_model, tokenizer = build_model_and_tokenizer(
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
    teacher_model, _ = build_model_and_tokenizer(
        model_name=cfg.effective_teacher_model_name,
        tokenizer_name=cfg.tokenizer_name,
        model_class=cfg.model_class,
        trust_remote_code=cfg.trust_remote_code,
        dtype=cfg.dtype,
        device=device,
        finetune_mode="full",
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_target_modules=cfg.lora_target_modules,
    )
    freeze_model(teacher_model)
    student_model.train()
    return TrainRuntimeBundle(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        device=device,
    )

