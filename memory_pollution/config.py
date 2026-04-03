from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from opd.config import TrainConfig, load_config as load_train_config


@dataclass
class MemoryPollutionEvalConfig:
    task: str = "arc"
    train_config_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    model_impl: str = "fla"
    model_name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    expected_architecture: Optional[str] = None
    trust_remote_code: bool = True
    dtype: str = "bf16"

    dataset_name: str = "allenai/ai2_arc"
    dataset_config: Optional[str] = "ARC-Challenge"
    dataset_split: str = "validation"
    local_dataset_path: Optional[str] = None

    max_samples: int = 0
    normalize_logprob_by_length: bool = True
    eval_batch_size: int = 4

    perturb_kind: str = "random_tokens"
    perturb_ratio: float = 0.1
    perturb_seed: int = 1234
    perturb_position: str = "random"
    perturb_min_tokens: int = 1

    collect_state_drift: bool = True
    state_key: Optional[str] = None

    output_dir: str = "outputs"
    run_name: Optional[str] = None

    @property
    def is_local_dataset(self) -> bool:
        return self.local_dataset_path is not None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _has_inline_model_config(cfg: MemoryPollutionEvalConfig) -> bool:
    return bool(cfg.model_name or cfg.expected_architecture or cfg.tokenizer_name)


def _validate_config_values(cfg: MemoryPollutionEvalConfig) -> None:
    if cfg.task not in {"arc", "lambada_openai"}:
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
    if not cfg.dataset_name:
        raise ValueError("dataset_name must be non-empty")
    if not cfg.dataset_split:
        raise ValueError("dataset_split must be non-empty")
    if cfg.task == "arc" and not cfg.dataset_config:
        raise ValueError("dataset_config must be non-empty for task=arc")
    if cfg.max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    if cfg.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if cfg.perturb_kind != "random_tokens":
        raise ValueError(f"Unsupported perturb_kind: {cfg.perturb_kind}")
    if cfg.perturb_position != "random":
        raise ValueError(f"Unsupported perturb_position: {cfg.perturb_position}")
    if not 0.0 <= cfg.perturb_ratio <= 1.0:
        raise ValueError(f"perturb_ratio must be in [0, 1], got {cfg.perturb_ratio}")
    if cfg.perturb_min_tokens < 0:
        raise ValueError("perturb_min_tokens must be >= 0")
    if cfg.state_key is not None and not cfg.state_key:
        raise ValueError("state_key must be null or a non-empty string")


def resolve_train_config(cfg: MemoryPollutionEvalConfig) -> TrainConfig:
    if cfg.train_config_path:
        train_cfg = load_train_config(cfg.train_config_path)
        if train_cfg.finetune_mode != "full":
            raise ValueError(
                "memory_pollution eval supports finetune_mode=full only; "
                f"got finetune_mode={train_cfg.finetune_mode} from train_config_path"
            )
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
    if cfg.state_key is not None:
        train_cfg.state_key = cfg.state_key
    return train_cfg


def load_config(path: str) -> MemoryPollutionEvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping")

    known_fields = {field.name for field in fields(MemoryPollutionEvalConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = MemoryPollutionEvalConfig(**raw)
    _validate_config_values(cfg)
    return cfg
