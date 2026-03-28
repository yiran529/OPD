from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class EvalConfig:
    task: str = "arc_ai2"
    train_config_path: str = "configs/gdn_340m_opd.yaml"
    checkpoint_path: str = ""

    dataset_name: str = "allenai/ai2_arc"
    dataset_config: str = "ARC-Challenge"
    dataset_split: str = "validation"
    local_dataset_path: Optional[str] = None

    max_samples: int = 0
    normalize_logprob_by_length: bool = True

    output_dir: str = "outputs"
    run_name: Optional[str] = None

    @property
    def is_local_dataset(self) -> bool:
        return self.local_dataset_path is not None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_eval_config(cfg: EvalConfig) -> None:
    if cfg.task != "arc_ai2":
        raise ValueError(f"Unsupported task: {cfg.task}")
    if not cfg.train_config_path:
        raise ValueError("train_config_path must be non-empty")
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be non-empty")
    if not cfg.dataset_name:
        raise ValueError("dataset_name must be non-empty")
    if not cfg.dataset_config:
        raise ValueError("dataset_config must be non-empty")
    if not cfg.dataset_split:
        raise ValueError("dataset_split must be non-empty")
    if cfg.max_samples < 0:
        raise ValueError("max_samples must be >= 0")


def load_eval_config(path: str) -> EvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Eval config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Eval config root must be a mapping")

    known_fields = {field.name for field in fields(EvalConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown eval config keys: {unknown}")

    cfg = EvalConfig(**raw)
    _validate_eval_config(cfg)
    return cfg
