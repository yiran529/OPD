from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class MemoryPollutionEvalConfig:
    task: str = "arc"
    train_config_path: str = "configs/gdn_340m_opd.yaml"
    checkpoint_path: Optional[str] = None
    model_impl: str = "fla"

    dataset_name: str = "allenai/ai2_arc"
    dataset_config: str = "ARC-Challenge"
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


def _validate_config_values(cfg: MemoryPollutionEvalConfig) -> None:
    if cfg.task != "arc":
        raise ValueError(f"Unsupported task: {cfg.task}")
    if not cfg.train_config_path:
        raise ValueError("train_config_path must be non-empty")
    if cfg.model_impl != "fla":
        raise ValueError(f"Unsupported model_impl: {cfg.model_impl}")
    if not cfg.dataset_name:
        raise ValueError("dataset_name must be non-empty")
    if not cfg.dataset_config:
        raise ValueError("dataset_config must be non-empty")
    if not cfg.dataset_split:
        raise ValueError("dataset_split must be non-empty")
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
