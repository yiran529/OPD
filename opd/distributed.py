from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistEnv:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    is_distributed = world_size > 1
    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with WORLD_SIZE>1 requires CUDA")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return DistEnv(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_distributed=is_distributed,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_mean(value: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= world_size
    return out


def broadcast_object(obj: Optional[object], src: int = 0) -> object:
    package = [obj]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(package, src=src)
    return package[0]
