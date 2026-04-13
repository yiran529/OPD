from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ExposureBiasTrainConfig:
    run_name: str = "hf-dataset-lora"
    output_dir: str = "outputs"
    seed: int = 42

    model_name: str = "m-a-p/340M-20B-GatedDeltaNet-pure-baseline"
    tokenizer_name: Optional[str] = None
    expected_architecture: str = "GatedDeltaNetForCausalLM"
    trust_remote_code: bool = True
    dtype: str = "bf16"

    init_checkpoint_path: Optional[str] = None

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: Optional[str] = "sample-10BT"
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    local_dataset_path: Optional[str] = None
    sequence_length: int = 512
    shuffle: bool = True

    finetune_mode: str = "lora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_last_n_blocks: int = 4

    micro_batch_size: int = 1
    grad_accum_steps: int = 8
    num_epochs: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    log_interval: int = 20
    save_every_n_epochs: int = 1
    keep_last_k_checkpoints: int = 2

    @property
    def sequence_plus_one(self) -> int:
        return self.sequence_length + 1

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_config_values(cfg: ExposureBiasTrainConfig) -> None:
    if cfg.dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")
    if cfg.finetune_mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported finetune_mode: {cfg.finetune_mode}")
    if not cfg.dataset_name and not cfg.local_dataset_path:
        raise ValueError("Either dataset_name or local_dataset_path must be set")
    if not cfg.dataset_split:
        raise ValueError("dataset_split must be non-empty")
    if not cfg.dataset_text_field:
        raise ValueError("dataset_text_field must be non-empty")
    if cfg.sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if cfg.micro_batch_size <= 0 or cfg.grad_accum_steps <= 0:
        raise ValueError("micro_batch_size and grad_accum_steps must be positive")
    if cfg.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if cfg.lora_r <= 0:
        raise ValueError("lora_r must be positive")
    if cfg.lora_alpha <= 0:
        raise ValueError("lora_alpha must be positive")
    if not 0.0 <= cfg.lora_dropout < 1.0:
        raise ValueError("lora_dropout must be in [0, 1)")
    if cfg.lora_last_n_blocks < 0:
        raise ValueError("lora_last_n_blocks must be >= 0")
    if cfg.finetune_mode == "lora" and cfg.lora_last_n_blocks <= 0:
        raise ValueError("lora_last_n_blocks must be positive when finetune_mode=lora")
    if cfg.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if cfg.save_every_n_epochs <= 0:
        raise ValueError("save_every_n_epochs must be positive")
    if cfg.keep_last_k_checkpoints < 0:
        raise ValueError("keep_last_k_checkpoints must be >= 0")

    if cfg.local_dataset_path and not Path(cfg.local_dataset_path).exists():
        raise FileNotFoundError(f"local_dataset_path not found: {cfg.local_dataset_path}")
    if cfg.init_checkpoint_path and not Path(cfg.init_checkpoint_path).exists():
        raise FileNotFoundError(f"init_checkpoint_path not found: {cfg.init_checkpoint_path}")


def load_config(path: str) -> ExposureBiasTrainConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping")

    known_fields = {field.name for field in fields(ExposureBiasTrainConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = ExposureBiasTrainConfig(**raw)
    _validate_config_values(cfg)
    return cfg
