from __future__ import annotations

from typing import Iterator, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from opd.config import TrainConfig


class FineWebPackedDataset(IterableDataset):
    def __init__(self, cfg: TrainConfig, tokenizer, rank: int, world_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size

    def _build_stream(self):
        stream = load_dataset(
            self.cfg.dataset_name,
            self.cfg.dataset_config,
            split=self.cfg.dataset_split,
            streaming=True,
        )

        if self.cfg.shuffle_buffer_size > 0:
            stream = stream.shuffle(
                buffer_size=self.cfg.shuffle_buffer_size,
                seed=self.cfg.seed,
            )

        if self.world_size > 1:
            stream = stream.shard(num_shards=self.world_size, index=self.rank)

        worker_info = get_worker_info()
        if worker_info is not None:
            stream = stream.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        return stream

    def __iter__(self) -> Iterator[torch.Tensor]:
        eos_token_id = self.tokenizer.eos_token_id
        assert eos_token_id is not None, "tokenizer must provide eos_token_id"

        seq_plus_one = self.cfg.sequence_plus_one
        seq_len = self.cfg.sequence_length
        token_buffer: List[int] = []

        stream = self._build_stream()
        text_field = self.cfg.dataset_text_field

        for row in stream:
            assert text_field in row, "missing dataset text field"
            text = row[text_field]
            assert isinstance(text, str), "dataset text must be str"
            if not text:
                continue

            token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                continue
            token_ids.append(eos_token_id)
            token_buffer.extend(token_ids)

            while len(token_buffer) >= seq_plus_one:
                chunk = token_buffer[:seq_plus_one]
                yield torch.tensor(chunk, dtype=torch.long)
                token_buffer = token_buffer[seq_len:]


def _collate_chunks(chunks: List[torch.Tensor]) -> torch.Tensor:
    assert chunks, "empty chunk list"
    return torch.stack(chunks, dim=0)


def build_dataloader(
    cfg: TrainConfig,
    tokenizer,
    rank: int,
    world_size: int,
) -> DataLoader:
    dataset = FineWebPackedDataset(cfg=cfg, tokenizer=tokenizer, rank=rank, world_size=world_size)
    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_chunks,
    )
