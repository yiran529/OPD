from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainConfig:
    run_name: str = "gdn340m-opd"
    output_dir: str = "outputs"
    seed: int = 42

    model_name: str = "m-a-p/340M-20B-GatedDeltaNet-pure-baseline"
    tokenizer_name: Optional[str] = None
    expected_architecture: str = "GatedDeltaNetForCausalLM"
    trust_remote_code: bool = True
    dtype: str = "bf16"
    finetune_mode: str = "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-100BT"
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    shuffle_buffer_size: int = 10000
    sanity_num_samples: int = 0
    sanity_disable_shuffle: bool = False

    objective: str = "opd_kl"
    context_len: int = 1024
    prefix_len: int = 128
    continuation_len: int = 128

    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    max_steps: int = 10000
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    lambda_state: float = 0.1
    state_key: str = "recurrent_state"
    state_time_stride: int = 1
    state_align_loss: str = "gram_mse"

    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    prefix_corrupt_topk_ratio: float = 0.25
    prefix_corrupt_topk_max: int = 64
    ema_teacher_enabled: bool = False
    ema_decay: float = 0.999
    ema_start_step: int = 0

    log_interval: int = 20
    save_interval: int = 500
    keep_last_k_checkpoints: int = 2
    wandb_enabled: bool = False
    wandb_project: str = "opd"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_mode: str = "online"

    resume_path: Optional[str] = None

    @property
    def sequence_length(self) -> int:
        if self.objective == "opd_kl":
            return self.context_len + self.prefix_len
        return self.context_len + self.prefix_len + self.continuation_len

    @property
    def sequence_plus_one(self) -> int:
        return self.sequence_length + 1

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_config_values(cfg: TrainConfig) -> None:
    if cfg.objective not in {"baseline_ce", "opd_kl"}:
        raise ValueError(f"Unsupported objective: {cfg.objective}")
    if cfg.dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")
    if cfg.finetune_mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported finetune_mode: {cfg.finetune_mode}")
    if cfg.context_len <= 0 or cfg.prefix_len <= 0 or cfg.continuation_len <= 0:
        raise ValueError("context_len/prefix_len/continuation_len must be positive")
    if cfg.micro_batch_size <= 0 or cfg.grad_accum_steps <= 0:
        raise ValueError("micro_batch_size and grad_accum_steps must be positive")
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if cfg.rollout_temperature < 0.0:
        raise ValueError("rollout_temperature must be >= 0")
    if not 0.0 < cfg.rollout_top_p <= 1.0:
        raise ValueError("rollout_top_p must be in (0, 1]")
    if not 0.0 < cfg.prefix_corrupt_topk_ratio <= 1.0:
        raise ValueError("prefix_corrupt_topk_ratio must be in (0, 1]")
    if cfg.prefix_corrupt_topk_max <= 0:
        raise ValueError("prefix_corrupt_topk_max must be positive")
    if int(cfg.prefix_corrupt_topk_ratio * cfg.prefix_len) <= 0:
        raise ValueError(
            "prefix_corrupt_topk_ratio * prefix_len must be >= 1 "
            f"(ratio={cfg.prefix_corrupt_topk_ratio}, prefix_len={cfg.prefix_len})"
        )
    if not 0.0 <= cfg.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if cfg.ema_start_step < 0:
        raise ValueError("ema_start_step must be >= 0")
    if not cfg.state_key:
        raise ValueError("state_key must be a non-empty string")
    if cfg.state_time_stride <= 0:
        raise ValueError("state_time_stride must be positive")
    if cfg.state_align_loss not in {"gram_mse", "cos_norm"}:
        raise ValueError(f"Unsupported state_align_loss: {cfg.state_align_loss}")
    if cfg.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"Unsupported wandb_mode: {cfg.wandb_mode}")
    if cfg.wandb_enabled and not cfg.wandb_project:
        raise ValueError("wandb_project must be non-empty when wandb_enabled=true")
    if any((not isinstance(tag, str) or not tag) for tag in cfg.wandb_tags):
        raise ValueError("wandb_tags must be a list of non-empty strings")
    if cfg.lora_r <= 0:
        raise ValueError(f"lora_r must be positive, got {cfg.lora_r}")
    if cfg.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {cfg.lora_alpha}")
    if not 0.0 <= cfg.lora_dropout < 1.0:
        raise ValueError(f"lora_dropout must be in [0, 1), got {cfg.lora_dropout}")
    if not isinstance(cfg.lora_target_modules, list):
        raise ValueError("lora_target_modules must be a list")
    if any((not isinstance(name, str) or not name) for name in cfg.lora_target_modules):
        raise ValueError("lora_target_modules must be a list of non-empty strings")
    if cfg.sanity_num_samples < 0:
        raise ValueError("sanity_num_samples must be >= 0")


def load_config(path: str) -> TrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping")

    known_fields = {field.name for field in fields(TrainConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = TrainConfig(**raw)
    _validate_config_values(cfg)
    return cfg
