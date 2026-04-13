from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from opd.config import TrainConfig, load_config as load_train_config


@dataclass
class ExposureBiasEvalConfig:
    task: str = "hf_dataset"
    train_config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    model_impl: str = "fla"
    model_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    expected_architecture: Optional[str] = None
    trust_remote_code: bool = True
    dtype: str = "bf16"

    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: Optional[str] = "sample-10BT"
    dataset_split: str = "train"
    dataset_text_field: str = "text"
    local_dataset_path: Optional[str] = None

    prefix_len: int = 128
    rollout_len: int = 128
    max_samples: int = 0
    batch_size: int = 8
    rollout_policy: str = "greedy"

    output_dir: str = "outputs"
    run_name: Optional[str] = None

    @property
    def is_local_dataset(self) -> bool:
        return self.local_dataset_path is not None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _has_inline_model_config(cfg: ExposureBiasEvalConfig) -> bool:
    return bool(cfg.model_name or cfg.expected_architecture or cfg.tokenizer_name)


def _validate_config_values(cfg: ExposureBiasEvalConfig) -> None:
    if cfg.task != "hf_dataset":
        raise ValueError(f"Unsupported task: {cfg.task}")
    if cfg.model_impl != "fla":
        raise ValueError(f"Unsupported model_impl: {cfg.model_impl}")
    if not cfg.train_config_path and not _has_inline_model_config(cfg):
        raise ValueError(
            "Specify either train_config_path or inline model config "
            "(at minimum model_name + expected_architecture)"
        )
    if _has_inline_model_config(cfg):
        if not cfg.model_name:
            raise ValueError("model_name must be set when using inline model config")
        if not cfg.expected_architecture:
            raise ValueError("expected_architecture must be set when using inline model config")
    if not cfg.dataset_name and not cfg.local_dataset_path:
        raise ValueError("Either dataset_name or local_dataset_path must be set")
    if not cfg.dataset_split:
        raise ValueError("dataset_split must be non-empty")
    if not cfg.dataset_text_field:
        raise ValueError("dataset_text_field must be non-empty")
    if cfg.prefix_len <= 0:
        raise ValueError("prefix_len must be positive")
    if cfg.rollout_len <= 0:
        raise ValueError("rollout_len must be positive")
    if cfg.max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.rollout_policy != "greedy":
        raise ValueError(f"Unsupported rollout_policy: {cfg.rollout_policy}")
    if cfg.local_dataset_path and not Path(cfg.local_dataset_path).exists():
        raise FileNotFoundError(f"local_dataset_path not found: {cfg.local_dataset_path}")


def resolve_train_config(cfg: ExposureBiasEvalConfig) -> TrainConfig:
    if cfg.train_config_path:
        train_cfg = load_train_config(cfg.train_config_path)
        if not _has_inline_model_config(cfg):
            return train_cfg
    else:
        train_cfg = TrainConfig()

    if cfg.model_name is not None:
        train_cfg.model_name = cfg.model_name
    if cfg.tokenizer_name is not None:
        train_cfg.tokenizer_name = cfg.tokenizer_name
    if cfg.expected_architecture is not None:
        train_cfg.expected_architecture = cfg.expected_architecture
    train_cfg.trust_remote_code = cfg.trust_remote_code
    train_cfg.dtype = cfg.dtype
    train_cfg.finetune_mode = "full"
    return train_cfg


def load_config(path: str) -> ExposureBiasEvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping")

    known_fields = {field.name for field in fields(ExposureBiasEvalConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = ExposureBiasEvalConfig(**raw)
    _validate_config_values(cfg)
    return cfg
