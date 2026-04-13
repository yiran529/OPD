from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset


def _to_split_dataset(
    dataset_obj: Dataset | DatasetDict,
    split: str,
) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if split not in dataset_obj:
        raise KeyError(f"Split {split} not found in local dataset; available={list(dataset_obj.keys())}")
    return dataset_obj[split]


def load_hf_text_dataset(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    local_dataset_path: str | None,
    streaming: bool,
):
    if local_dataset_path:
        local_path = Path(local_dataset_path)
        if not local_path.exists():
            raise FileNotFoundError(f"local_dataset_path not found: {local_path}")

        # Support either a saved Arrow dataset (`load_from_disk`) or a local
        # dataset loading script snapshot such as a HF cache checkout.
        if local_path.is_dir():
            try:
                dataset_obj = load_from_disk(str(local_path))
            except Exception:
                dataset_obj = None
            if dataset_obj is not None:
                return _to_split_dataset(dataset_obj=dataset_obj, split=dataset_split)

        if dataset_config:
            return load_dataset(
                str(local_path),
                dataset_config,
                split=dataset_split,
                streaming=streaming,
            )
        return load_dataset(
            str(local_path),
            split=dataset_split,
            streaming=streaming,
        )

    if dataset_config:
        return load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
            streaming=streaming,
        )
    return load_dataset(
        dataset_name,
        split=dataset_split,
        streaming=streaming,
    )


def _load_token_stream(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    dataset_text_field: str,
    local_dataset_path: str | None,
    tokenizer,
) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    assert eos_token_id is not None, "tokenizer must provide eos_token_id"

    dataset = load_hf_text_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        local_dataset_path=local_dataset_path,
        streaming=False,
    )

    token_stream: list[int] = []
    for row in dataset:
        assert dataset_text_field in row, f"missing dataset text field: {dataset_text_field}"
        text = row[dataset_text_field]
        assert isinstance(text, str), f"dataset field must be str: {dataset_text_field}"
        if not text:
            continue

        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_stream.extend(token_ids)
        token_stream.append(int(eos_token_id))

    assert token_stream, "token stream is empty after reading HF dataset"
    return token_stream


def _build_chunks(
    token_stream: list[int],
    chunk_len: int,
) -> list[list[int]]:
    assert chunk_len > 0, "chunk_len must be positive"
    chunks: list[list[int]] = []
    for start in range(0, len(token_stream) - chunk_len + 1, chunk_len):
        chunks.append(token_stream[start : start + chunk_len])
    assert chunks, (
        f"not enough tokens to build any chunk: token_count={len(token_stream)} chunk_len={chunk_len}"
    )
    return chunks


class HFTextChunkDataset(TorchDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config: str | None,
        dataset_split: str,
        dataset_text_field: str,
        local_dataset_path: str | None,
        tokenizer,
        chunk_len: int,
    ) -> None:
        super().__init__()
        token_stream = _load_token_stream(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            dataset_text_field=dataset_text_field,
            local_dataset_path=local_dataset_path,
            tokenizer=tokenizer,
        )
        self.chunks = _build_chunks(token_stream=token_stream, chunk_len=chunk_len)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.chunks[index], dtype=torch.long)


def build_train_dataloader(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    dataset_text_field: str,
    local_dataset_path: str | None,
    tokenizer,
    chunk_len: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = HFTextChunkDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        dataset_text_field=dataset_text_field,
        local_dataset_path=local_dataset_path,
        tokenizer=tokenizer,
        chunk_len=chunk_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def iter_hf_dataset_examples(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    dataset_text_field: str,
    local_dataset_path: str | None,
    tokenizer,
    seq_len: int,
    max_samples: int,
    sample_prefix: str,
) -> Iterable[dict]:
    token_stream = _load_token_stream(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        dataset_text_field=dataset_text_field,
        local_dataset_path=local_dataset_path,
        tokenizer=tokenizer,
    )
    chunks = _build_chunks(token_stream=token_stream, chunk_len=seq_len)
    for idx, chunk in enumerate(chunks):
        if max_samples > 0 and idx >= max_samples:
            break
        yield {
            "id": f"{sample_prefix}::{idx}",
            "token_ids": chunk,
        }
