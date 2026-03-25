from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _checkpoint_name(step: int) -> str:
    return f"step_{step:08d}.pt"


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    rollout_model: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    config_dict,
    keep_last_k: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_model = _unwrap_model(model)

    state = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rollout_model": rollout_model.state_dict() if rollout_model is not None else None,
        "config": config_dict,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    checkpoint_path = checkpoint_dir / _checkpoint_name(step)
    torch.save(state, checkpoint_path)

    files = sorted(checkpoint_dir.glob("step_*.pt"))
    if keep_last_k > 0 and len(files) > keep_last_k:
        for old_file in files[: len(files) - keep_last_k]:
            old_file.unlink()

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    rollout_model: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> int:
    raw_model = _unwrap_model(model)
    state = torch.load(checkpoint_path, map_location="cpu")

    raw_model.load_state_dict(state["model"], strict=True)

    if rollout_model is not None and state.get("rollout_model") is not None:
        rollout_model.load_state_dict(state["rollout_model"], strict=True)

    optimizer.load_state_dict(state["optimizer"])
    for param_state in optimizer.state.values():
        for key, value in param_state.items():
            if isinstance(value, torch.Tensor):
                param_state[key] = value.to(device)

    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])

    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    if "rng_python" in state:
        random.setstate(state["rng_python"])
    if "rng_numpy" in state:
        np.random.set_state(state["rng_numpy"])
    if "rng_torch_cpu" in state:
        torch.set_rng_state(state["rng_torch_cpu"])
    if torch.cuda.is_available() and state.get("rng_torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["rng_torch_cuda"])

    raw_model.to(device)
    if rollout_model is not None:
        rollout_model.to(device)

    return int(state["step"])
