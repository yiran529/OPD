from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class QwenOPSDEvalConfig:
    run_name: str = "qwen3-opsd"
    output_dir: str = "outputs"
    task: str = "corruption_robustness"

    model_name: str = "Qwen/Qwen3.5-0.8B"
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
    checkpoint_path: Optional[str] = None
    max_samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_config_values(cfg: QwenOPSDEvalConfig) -> None:
    if cfg.task not in {"corruption_robustness"}:
        raise ValueError(f"Unsupported eval task: {cfg.task}")
    if cfg.model_class not in {"conditional_generation", "causal_lm"}:
        raise ValueError(f"Unsupported model_class: {cfg.model_class}")
    if cfg.dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}")
    if cfg.finetune_mode not in {"full", "lora"}:
        raise ValueError(f"Unsupported finetune_mode: {cfg.finetune_mode}")
    if cfg.checkpoint_path and not Path(cfg.checkpoint_path).exists():
        raise FileNotFoundError(f"checkpoint_path not found: {cfg.checkpoint_path}")
    if cfg.max_samples < 0:
        raise ValueError("max_samples must be >= 0")


def load_config(path: str) -> QwenOPSDEvalConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping")

    known_fields = {field.name for field in fields(QwenOPSDEvalConfig)}
    unknown = sorted(set(raw.keys()) - known_fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = QwenOPSDEvalConfig(**raw)
    _validate_config_values(cfg)
    return cfg
