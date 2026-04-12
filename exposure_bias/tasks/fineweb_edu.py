from __future__ import annotations

from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk

from exposure_bias.config import ExposureBiasEvalConfig


def _to_split_dataset(dataset_obj: Dataset | DatasetDict, split: str) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if split not in dataset_obj:
        raise KeyError(f"Split {split} not found in local dataset; available={list(dataset_obj.keys())}")
    return dataset_obj[split]


def load_fineweb_edu_dataset(cfg: ExposureBiasEvalConfig) -> Dataset | IterableDataset:
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
            streaming=True,
        )

    return load_dataset(
        cfg.dataset_name,
        split=cfg.dataset_split,
        streaming=True,
    )


def iter_fineweb_edu_examples(
    cfg: ExposureBiasEvalConfig,
    tokenizer,
) -> Iterable[dict]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    assert eos_token_id is not None, "tokenizer must provide eos_token_id"

    dataset = load_fineweb_edu_dataset(cfg)
    seq_len = cfg.prefix_len + cfg.rollout_len
    token_buffer: list[int] = []
    sample_idx = 0

    for row in dataset:
        assert cfg.dataset_text_field in row, "missing dataset text field"
        text = row[cfg.dataset_text_field]
        assert isinstance(text, str), "dataset text must be str"
        if not text:
            continue

        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_ids.append(int(eos_token_id))
        token_buffer.extend(token_ids)

        while len(token_buffer) >= seq_len:
            chunk = token_buffer[:seq_len]
            yield {
                "id": f"fineweb_edu::{sample_idx}",
                "token_ids": chunk,
            }
            sample_idx += 1
            if cfg.max_samples > 0 and sample_idx >= cfg.max_samples:
                return
            token_buffer = token_buffer[seq_len:]
