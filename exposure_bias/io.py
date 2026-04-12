from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


def checkpoint_tag_from_path(checkpoint_path: str | None) -> str:
    if not checkpoint_path:
        return "pretrained"
    return Path(checkpoint_path).stem


def build_experiment_name(
    model_name: str,
    prefix_len: int,
    rollout_len: int,
) -> str:
    assert model_name, "model_name must be non-empty"
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip()).strip("._-")
    assert model_slug, f"failed to build model slug from model_name={model_name!r}"
    return f"{model_slug}_p{prefix_len}_r{rollout_len}"


def build_output_dir(
    output_dir: str,
    experiment_name: str,
    task: str,
    checkpoint_tag: str,
) -> Path:
    path = Path(output_dir) / experiment_name / "exposure_bias" / task / checkpoint_tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
