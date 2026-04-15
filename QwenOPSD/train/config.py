from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class QwenOPSDTrainConfig:
    run_name: str = "qwen3-opsd"
    output_dir: str = "outputs"
    seed: int = 42

    model_name: str = "Qwen/Qwen3.5-0.8B"
    teacher_model_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    model_class: str = "conditional_generation"
    trust_remote_code: bool = False
    dtype: str = "bf16"
    finetune_mode: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    resume_path: Optional[str] = None

    dataset_name: str = "open-r1/OpenThoughts-114k-math"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    local_dataset_path: Optional[str] = None
    filter_correct_only: bool = True
    max_train_samples: int = 0
    shuffle: bool = True
    num_workers: int = 0

    enable_thinking: bool = True
    max_prompt_tokens: int = 1024
    max_solution_tokens: int = 1024
    min_solution_tokens: int = 16

    alpha: float = 1.0
    rollout_len: int = 8
    corrupt_span_choices: list[int] = field(default_factory=lambda: [2])
    corrupt_start_min_ratio: float = 0.25
    corrupt_start_max_ratio: float = 0.5
    detach_rollout_cache: bool = True

    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    log_interval: int = 20
    save_every_n_steps: int = 500
    keep_last_k_checkpoints: int = 2
    wandb_enabled: bool = False
    wandb_project: str = "QwenOPSD"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: str = "online"

    @property
    def effective_teacher_model_name(self) -> str:
        return self.teacher_model_name or self.model_name

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_INT_FIELDS = {
    "seed",
    "lora_r",
    "lora_alpha",
    "max_train_samples",
    "num_workers",
    "max_prompt_tokens",
    "max_solution_tokens",
    "min_solution_tokens",
    "rollout_len",
    "micro_batch_size",
    "grad_accum_steps",
    "num_epochs",
    "warmup_steps",
    "log_interval",
    "save_every_n_steps",
    "keep_last_k_checkpoints",
}

_FLOAT_FIELDS = {
    "lora_dropout",
    "alpha",
    "corrupt_start_min_ratio",
    "corrupt_start_max_ratio",
    "learning_rate",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "max_grad_norm",
}


def _coerce_scalar_fields(raw: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(raw)
    for key in _INT_FIELDS:
        if key in coerced and isinstance(coerced[key], str):
            coerced[key] = int(coerced[key])
    for key in _FLOAT_FIELDS:
        if key in coerced and isinstance(coerced[key], str):
            coerced[key] = float(coerced[key])
    return coerced


def _validate_config_values(cfg: QwenOPSDTrainConfig) -> None:
    if cfg.model_class not in {"conditional_generation", "causal_lm"}:
        raise ValueError(f"Unsupported model_class: {cfg.model_class}")
    if cfg.dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")
    if cfg.finetune_mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported finetune_mode: {cfg.finetune_mode}")
    if cfg.finetune_mode == "lora":
        if cfg.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if cfg.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= cfg.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not cfg.lora_target_modules:
            raise ValueError("lora_target_modules must be non-empty when finetune_mode=lora")
    if cfg.resume_path and not Path(cfg.resume_path).exists():
        raise FileNotFoundError(f"resume_path not found: {cfg.resume_path}")
    if cfg.local_dataset_path and not Path(cfg.local_dataset_path).exists():
        raise FileNotFoundError(f"local_dataset_path not found: {cfg.local_dataset_path}")
    if not 0.0 <= cfg.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if cfg.rollout_len <= 0:
        raise ValueError("rollout_len must be positive")
    if not cfg.corrupt_span_choices:
        raise ValueError("corrupt_span_choices must be non-empty")
    if any((not isinstance(span_len, int) or span_len <= 0) for span_len in cfg.corrupt_span_choices):
        raise ValueError("corrupt_span_choices must contain positive integers")
    if not 0.0 <= cfg.corrupt_start_min_ratio <= cfg.corrupt_start_max_ratio <= 1.0:
        raise ValueError("corrupt_start_min_ratio/max_ratio must satisfy 0 <= min <= max <= 1")
    if cfg.max_prompt_tokens <= 0:
        raise ValueError("max_prompt_tokens must be positive")
    if cfg.max_solution_tokens <= 0:
        raise ValueError("max_solution_tokens must be positive")
    if cfg.min_solution_tokens <= 0:
        raise ValueError("min_solution_tokens must be positive")
    if cfg.min_solution_tokens < max(cfg.corrupt_span_choices) + cfg.rollout_len:
        raise ValueError(
            "min_solution_tokens must be at least max(corrupt_span_choices) + rollout_len"
        )
    if cfg.micro_batch_size <= 0 or cfg.grad_accum_steps <= 0:
        raise ValueError("micro_batch_size and grad_accum_steps must be positive")
    if cfg.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if cfg.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if cfg.save_every_n_steps <= 0:
        raise ValueError("save_every_n_steps must be positive")
    if cfg.keep_last_k_checkpoints < 0:
        raise ValueError("keep_last_k_checkpoints must be >= 0")
    if cfg.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported wandb_mode: {cfg.wandb_mode}")
    if cfg.wandb_enabled and not cfg.wandb_project:
        raise ValueError("wandb_project must be non-empty when wandb_enabled=true")
    if any((not isinstance(tag, str) or not tag) for tag in cfg.wandb_tags):
        raise ValueError("wandb_tags must be a list of non-empty strings")


def load_config(path: str) -> QwenOPSDTrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping")
    raw = _coerce_scalar_fields(raw)

    known_fields = {field.name for field in fields(QwenOPSDTrainConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = QwenOPSDTrainConfig(**raw)
    _validate_config_values(cfg)
    return cfg
