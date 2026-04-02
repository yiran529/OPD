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


def load_arc_dataset(cfg: MemoryPollutionEvalConfig) -> Dataset:
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


def _normalize_arc_row(row: dict) -> dict:
    question_field = row.get("question", "")
    answer_key = row.get("answerKey", "")

    if isinstance(question_field, dict):
        stem = question_field.get("stem", "")
        raw_choices = question_field.get("choices", row.get("choices", []))
    else:
        stem = question_field
        raw_choices = row.get("choices", [])

    assert isinstance(stem, str) and stem, "ARC question must be non-empty str"
    assert isinstance(answer_key, str) and answer_key, "ARC answerKey must be non-empty str"

    normalized_choices: list[dict] = []
    if isinstance(raw_choices, list):
        for choice in raw_choices:
            assert isinstance(choice, dict), "ARC choice must be dict"
            label = choice.get("label", "")
            text = choice.get("text", "")
            assert isinstance(label, str) and label, "ARC choice.label must be non-empty str"
            assert isinstance(text, str) and text, "ARC choice.text must be non-empty str"
            normalized_choices.append({"label": label, "text": text})
    elif isinstance(raw_choices, dict):
        labels = raw_choices.get("label", [])
        texts = raw_choices.get("text", [])
        assert isinstance(labels, list) and isinstance(texts, list), (
            "ARC choices dict must contain list fields: label/text"
        )
        assert len(labels) == len(texts) and len(labels) > 0, (
            "ARC choices label/text length mismatch or empty"
        )
        for label, text in zip(labels, texts):
            assert isinstance(label, str) and label, "ARC choice.label must be non-empty str"
            assert isinstance(text, str) and text, "ARC choice.text must be non-empty str"
            normalized_choices.append({"label": label, "text": text})
    else:
        raise TypeError(f"Unsupported ARC choices type: {type(raw_choices)}")

    example_id = row.get("id", "")
    if not example_id:
        example_id = f"{stem[:32]}::{answer_key}"

    return {
        "id": example_id,
        "question": stem,
        "choices": normalized_choices,
        "answer_key": answer_key,
    }


def iter_arc_examples(cfg: MemoryPollutionEvalConfig) -> Iterable[dict]:
    dataset = load_arc_dataset(cfg)
    count = 0
    for row in dataset:
        yield _normalize_arc_row(row)

        count += 1
        if cfg.max_samples > 0 and count >= cfg.max_samples:
            break


def build_arc_prompt(question: str, choices: list[dict]) -> str:
    assert question, "question must be non-empty"
    assert choices, "choices must be non-empty"

    # Match lm-eval-harness ARC prompt formatting:
    #   doc_to_text: "Question: {{question}}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def build_choice_continuation(choice_text: str) -> str:
    assert choice_text, "choice_text must be non-empty"
    return f" {choice_text}"
