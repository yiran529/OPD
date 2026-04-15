from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


_DATASET_ALIAS_OVERRIDES = {
    "HuggingFaceFW/fineweb-edu": "fineweb",
    "lara-martin/Scifi_TV_Shows": "scifi",
    "WutYee/HarryPotter_books_1to7": "harrypotter",
    "gsm8k": "gsm8k",
    "openai/gsm8k": "gsm8k",
}


def checkpoint_tag_from_path(checkpoint_path: str | None) -> str:
    if not checkpoint_path:
        return "pretrained"
    return Path(checkpoint_path).stem


def dataset_tag_from_source(
    dataset_name: str | None,
    local_dataset_path: str | None,
) -> str:
    if dataset_name:
        if dataset_name in _DATASET_ALIAS_OVERRIDES:
            return _DATASET_ALIAS_OVERRIDES[dataset_name]
        dataset_source = dataset_name.rstrip("/").split("/")[-1]
    else:
        assert local_dataset_path, "either dataset_name or local_dataset_path must be set"
        dataset_source = Path(local_dataset_path).name

    dataset_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_source.strip()).strip("._-").lower()
    assert dataset_tag, f"failed to build dataset tag from source={dataset_source!r}"
    return dataset_tag


def build_experiment_name(
    dataset_tag: str,
    model_name: str,
    prefix_len: int | None = None,
    rollout_len: int | None = None,
) -> str:
    assert dataset_tag, "dataset_tag must be non-empty"
    assert model_name, "model_name must be non-empty"
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip()).strip("._-")
    assert model_slug, f"failed to build model slug from model_name={model_name!r}"
    if prefix_len is None or rollout_len is None:
        return f"{dataset_tag}_{model_slug}"
    return f"{dataset_tag}_{model_slug}_p{prefix_len}_r{rollout_len}"


def build_output_dir(
    output_dir: str,
    experiment_name: str,
    dataset_tag: str,
    checkpoint_tag: str,
) -> Path:
    path = Path(output_dir) / experiment_name / "exposure_bias" / dataset_tag / checkpoint_tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
