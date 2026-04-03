from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from memory_pollution.config import MemoryPollutionEvalConfig


def _to_split_dataset(dataset_obj: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if split not in dataset_obj:
        raise KeyError(f"Split {split} not found in local dataset; available={list(dataset_obj.keys())}")
    return dataset_obj[split]


def load_lambada_openai_dataset(cfg: MemoryPollutionEvalConfig) -> Dataset:
    if cfg.is_local_dataset:
        local_path = Path(cfg.local_dataset_path or "")
        if not local_path.exists():
            raise FileNotFoundError(f"local_dataset_path not found: {local_path}")
        dataset_obj = load_from_disk(str(local_path))
        return _to_split_dataset(dataset_obj=dataset_obj, split=cfg.dataset_split)

    if cfg.dataset_config:
        return load_dataset(
            cfg.dataset_name,
            cfg.dataset_config,
            split=cfg.dataset_split,
        )

    return load_dataset(
        cfg.dataset_name,
        split=cfg.dataset_split,
    )


def _normalize_lambada_openai_row(row: dict, row_idx: int) -> dict:
    text = row.get("text", "")
    assert isinstance(text, str) and text, "LAMBADA text must be non-empty str"
    assert " " in text, "LAMBADA text must contain at least one space"

    # Match lm-eval-harness task templating:
    #   doc_to_text: text.rsplit(" ", 1)[0]
    #   doc_to_target: " " + text.rsplit(" ", 1)[1]
    context_text, target_word = text.rsplit(" ", 1)
    assert context_text, "LAMBADA context must be non-empty"
    assert target_word, "LAMBADA target word must be non-empty"

    example_id = row.get("id", "")
    if not example_id:
        example_id = f"lambada_openai::{row_idx}"

    return {
        "id": example_id,
        "text": text,
        "context_text": context_text,
        "target_text": f" {target_word}",
    }


def iter_lambada_openai_examples(cfg: MemoryPollutionEvalConfig) -> Iterable[dict]:
    dataset = load_lambada_openai_dataset(cfg)
    count = 0
    for row_idx, row in enumerate(dataset):
        yield _normalize_lambada_openai_row(row, row_idx=row_idx)

        count += 1
        if cfg.max_samples > 0 and count >= cfg.max_samples:
            break
