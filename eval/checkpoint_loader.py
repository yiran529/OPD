from __future__ import annotations

from pathlib import Path

import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def load_model_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
) -> int:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    state = torch.load(str(checkpoint_file), map_location="cpu")
    if "model" not in state:
        raise KeyError("Checkpoint missing required key: model")

    raw_model = _unwrap_model(model)
    raw_model.load_state_dict(state["model"], strict=True)
    raw_model.to(device)

    return int(state.get("step", -1))
