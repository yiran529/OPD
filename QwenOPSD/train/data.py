from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from QwenOPSD.train.config import QwenOPSDTrainConfig
from QwenOPSD.train.formatting import build_prompt_token_ids, encode_solution_token_ids


@dataclass
class PreparedMathSample:
    sample_id: int
    problem_text: str
    solution_text: str
    prompt_ids: list[int]
    solution_ids: list[int]


@dataclass
class DatasetPreparationStats:
    num_rows_seen: int
    num_rows_kept: int
    num_rows_skipped_missing_fields: int
    num_rows_skipped_correct_filter: int
    num_rows_skipped_prompt_len: int
    num_rows_skipped_solution_len: int


class PreparedMathDataset(TorchDataset):
    def __init__(self, samples: list[PreparedMathSample], stats: DatasetPreparationStats) -> None:
        super().__init__()
        self.samples = samples
        self.stats = stats

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PreparedMathSample:
        return self.samples[index]


def _to_split_dataset(
    dataset_obj: Dataset | DatasetDict,
    split: str,
) -> Dataset:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    if split not in dataset_obj:
        raise KeyError(f"Split {split} not found in local dataset; available={list(dataset_obj.keys())}")
    return dataset_obj[split]


def _load_dataset_rows(cfg: QwenOPSDTrainConfig) -> Dataset:
    if cfg.local_dataset_path:
        local_path = Path(cfg.local_dataset_path)
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


def _prepare_samples(
    cfg: QwenOPSDTrainConfig,
    tokenizer,
) -> PreparedMathDataset:
    dataset = _load_dataset_rows(cfg=cfg)
    samples: list[PreparedMathSample] = []

    skipped_missing_fields = 0
    skipped_correct_filter = 0
    skipped_prompt_len = 0
    skipped_solution_len = 0

    for row_idx, row in enumerate(dataset):
        problem_text = row.get("problem")
        solution_text = row.get("solution")
        if not isinstance(problem_text, str) or not problem_text.strip():
            skipped_missing_fields += 1
            continue
        if not isinstance(solution_text, str) or not solution_text.strip():
            skipped_missing_fields += 1
            continue

        if cfg.filter_correct_only and "correct" in row and row["correct"] is not True:
            skipped_correct_filter += 1
            continue

        prompt_ids = build_prompt_token_ids(
            tokenizer=tokenizer,
            problem_text=problem_text,
            enable_thinking=cfg.enable_thinking,
        )
        if len(prompt_ids) > cfg.max_prompt_tokens:
            skipped_prompt_len += 1
            continue

        solution_ids = encode_solution_token_ids(
            tokenizer=tokenizer,
            solution_text=solution_text,
        )
        if len(solution_ids) < cfg.min_solution_tokens or len(solution_ids) > cfg.max_solution_tokens:
            skipped_solution_len += 1
            continue

        samples.append(
            PreparedMathSample(
                sample_id=row_idx,
                problem_text=problem_text,
                solution_text=solution_text,
                prompt_ids=prompt_ids,
                solution_ids=solution_ids,
            )
        )
        if cfg.max_train_samples > 0 and len(samples) >= cfg.max_train_samples:
            break

    stats = DatasetPreparationStats(
        num_rows_seen=row_idx + 1 if "row_idx" in locals() else 0,
        num_rows_kept=len(samples),
        num_rows_skipped_missing_fields=skipped_missing_fields,
        num_rows_skipped_correct_filter=skipped_correct_filter,
        num_rows_skipped_prompt_len=skipped_prompt_len,
        num_rows_skipped_solution_len=skipped_solution_len,
    )
    if not samples:
        raise RuntimeError(
            "No training samples remain after filtering. "
            f"stats={stats}"
        )
    return PreparedMathDataset(samples=samples, stats=stats)


def _collate_samples(samples: list[PreparedMathSample]) -> list[PreparedMathSample]:
    return samples


def build_train_dataloader(
    cfg: QwenOPSDTrainConfig,
    tokenizer,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DatasetPreparationStats, DistributedSampler | None]:
    dataset = _prepare_samples(cfg=cfg, tokenizer=tokenizer)
    sampler: DistributedSampler | None = None
    shuffle = cfg.shuffle
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=cfg.shuffle,
            drop_last=False,
        )
        shuffle = False
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=_collate_samples,
    )
    return dataloader, dataset.stats, sampler
