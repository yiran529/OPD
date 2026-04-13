from __future__ import annotations

from pathlib import Path

import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _checkpoint_name(step: int) -> str:
    return f"step_{step:08d}.pt"


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    config_dict: dict,
    keep_last_k: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_model = _unwrap_model(model)
    state = {
        "step": step,
        "model": raw_model.state_dict(),
        "config": config_dict,
    }
    checkpoint_path = checkpoint_dir / _checkpoint_name(step)
    torch.save(state, checkpoint_path)

    files = sorted(checkpoint_dir.glob("step_*.pt"))
    if keep_last_k > 0 and len(files) > keep_last_k:
        for old_file in files[: len(files) - keep_last_k]:
            old_file.unlink()
    return checkpoint_path


def load_exposure_bias_train_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> int:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    state = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
    if "model" not in state:
        raise KeyError("Checkpoint missing required key: model")

    raw_model = _unwrap_model(model)
    raw_model.load_state_dict(state["model"], strict=True)
    raw_model.to(device)
    return int(state.get("step", -1))
