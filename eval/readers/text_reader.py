from __future__ import annotations

from pathlib import Path

from eval.infer_config import InferConfig


def read_inputs(cfg: InferConfig) -> list[str]:
    if cfg.input_format != "txt":
        raise ValueError(f"Unsupported input_format: {cfg.input_format}")
    return read_text_inputs(cfg)


def read_text_inputs(cfg: InferConfig) -> list[str]:
    path = Path(cfg.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input text file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    texts: list[str] = []

    for line in lines:
        item = line.strip() if cfg.strip_lines else line
        if cfg.skip_empty_lines and not item:
            continue
        texts.append(item)

        if cfg.max_samples > 0 and len(texts) >= cfg.max_samples:
            break

    if not texts:
        raise ValueError("No input texts after filtering; check input file and infer config")

    return texts
