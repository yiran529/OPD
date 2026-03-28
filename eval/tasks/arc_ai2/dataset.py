from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from eval.config import EvalConfig


def _to_split_dataset(dataset_obj: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if split not in dataset_obj:
        raise KeyError(f"Split {split} not found in local dataset; available={list(dataset_obj.keys())}")
    return dataset_obj[split]


def load_arc_dataset(cfg: EvalConfig) -> Dataset:
    if cfg.is_local_dataset:
        local_path = Path(cfg.local_dataset_path or "")
        if not local_path.exists():
            raise FileNotFoundError(f"local_dataset_path not found: {local_path}")
        dataset_obj = load_from_disk(str(local_path))
        return _to_split_dataset(dataset_obj=dataset_obj, split=cfg.dataset_split)

    return load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
    )


def iter_arc_examples(cfg: EvalConfig) -> Iterable[dict]:
    dataset = load_arc_dataset(cfg)
    count = 0
    for row in dataset:
        question = row.get("question", {})
        choices = question.get("choices", [])
        stem = question.get("stem", "")
        answer_key = row.get("answerKey", "")

        assert isinstance(stem, str) and stem, "ARC question.stem must be non-empty str"
        assert isinstance(choices, list) and choices, "ARC question.choices must be non-empty list"
        assert isinstance(answer_key, str) and answer_key, "ARC answerKey must be non-empty str"

        normalized_choices = []
        for choice in choices:
            label = choice.get("label", "")
            text = choice.get("text", "")
            assert isinstance(label, str) and label, "ARC choice.label must be non-empty str"
            assert isinstance(text, str) and text, "ARC choice.text must be non-empty str"
            normalized_choices.append({"label": label, "text": text})

        yield {
            "id": row.get("id", ""),
            "question": stem,
            "choices": normalized_choices,
            "answer_key": answer_key,
        }

        count += 1
        if cfg.max_samples > 0 and count >= cfg.max_samples:
            break
