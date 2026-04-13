from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset


_SCIFI_TV_FIELD_NAMES = (
    "story_num",
    "story_line",
    "event",
    "gen_event",
    "sent",
    "gen_sent",
    "entities",
)

_SCIFI_TV_SPLIT_FILENAMES = {
    "train": ("scifi-train.txt",),
    "validation": ("scifi-val.txt", "scifi-valid.txt", "scifi-validation.txt"),
    "test": ("scifi-test.txt",),
}


def _to_split_dataset(
    dataset_obj: Dataset | DatasetDict,
    split: str,
) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if split not in dataset_obj:
        raise KeyError(f"Split {split} not found in local dataset; available={list(dataset_obj.keys())}")
    return dataset_obj[split]


def _find_scifi_tv_zip(local_path: Path) -> Path | None:
    if local_path.is_file() and local_path.suffix == ".zip":
        return local_path
    if not local_path.is_dir():
        return None

    direct_zip = local_path / "scifiTVshows.zip"
    if direct_zip.exists():
        return direct_zip

    zip_candidates = sorted(local_path.glob("*.zip"))
    if len(zip_candidates) == 1:
        return zip_candidates[0]
    return None


def _load_scifi_tv_rows_from_zip(
    zip_path: Path,
    split: str,
) -> list[dict]:
    assert split in _SCIFI_TV_SPLIT_FILENAMES, f"unsupported Scifi_TV_Shows split: {split}"

    with zipfile.ZipFile(zip_path) as zf:
        names = {Path(name).name: name for name in zf.namelist() if not name.endswith("/")}
        target_member = None
        for filename in _SCIFI_TV_SPLIT_FILENAMES[split]:
            if filename in names:
                target_member = names[filename]
                break
        assert target_member is not None, (
            f"missing split file in {zip_path}: split={split} candidates={_SCIFI_TV_SPLIT_FILENAMES[split]}"
        )

        rows: list[dict] = []
        with zf.open(target_member) as handle:
            for line_idx, raw_line in enumerate(handle, start=1):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                parts = [part.strip() for part in line.split("|||")]
                assert len(parts) == len(_SCIFI_TV_FIELD_NAMES), (
                    f"unexpected Scifi_TV_Shows column count at line {line_idx}: "
                    f"expected={len(_SCIFI_TV_FIELD_NAMES)} actual={len(parts)}"
                )
                rows.append(dict(zip(_SCIFI_TV_FIELD_NAMES, parts)))

    assert rows, f"no rows loaded from {zip_path} split={split}"
    return rows


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

        scifi_tv_zip = _find_scifi_tv_zip(local_path)
        if scifi_tv_zip is not None:
            return _load_scifi_tv_rows_from_zip(zip_path=scifi_tv_zip, split=dataset_split)

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
    try:
        return load_dataset(
            dataset_name,
            split=dataset_split,
            streaming=streaming,
        )
    except RuntimeError as exc:
        if dataset_name != "lara-martin/Scifi_TV_Shows":
            raise
        raise RuntimeError(
            "Scifi_TV_Shows uses a deprecated dataset script in the current datasets package. "
            "Set local_dataset_path to a local snapshot containing scifiTVshows.zip."
        ) from exc


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
