from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class InferConfig:
    train_config_path: str = "configs/gdn_340m_opd.yaml"
    checkpoint_path: str = ""

    input_path: str = ""
    input_format: str = "txt"
    max_samples: int = 0
    strip_lines: bool = True
    skip_empty_lines: bool = True

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    output_dir: str = "outputs"
    run_name: Optional[str] = None
    infer_name: str = "infer_text"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_infer_config(cfg: InferConfig) -> None:
    if not cfg.train_config_path:
        raise ValueError("train_config_path must be non-empty")
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be non-empty")
    if not cfg.input_path:
        raise ValueError("input_path must be non-empty")
    if cfg.input_format != "txt":
        raise ValueError(f"Unsupported input_format: {cfg.input_format}")
    if cfg.max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    if cfg.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if cfg.temperature < 0.0:
        raise ValueError("temperature must be >= 0")
    if not 0.0 < cfg.top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    if not cfg.infer_name:
        raise ValueError("infer_name must be non-empty")


def load_infer_config(path: str) -> InferConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Infer config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Infer config root must be a mapping")

    known_fields = {field.name for field in fields(InferConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown infer config keys: {unknown}")

    cfg = InferConfig(**raw)
    _validate_infer_config(cfg)
    return cfg
