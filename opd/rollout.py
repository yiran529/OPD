from __future__ import annotations

from typing import Tuple

import torch


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def sync_rollout_model(rollout_model: torch.nn.Module, source_model: torch.nn.Module) -> None:
    rollout_model.load_state_dict(source_model.state_dict(), strict=True)
    freeze_module(rollout_model)


@torch.no_grad()
def generate_rollout_tokens(
    rollout_model: torch.nn.Module,
    context_tokens: torch.Tensor,
    prefix_len: int,
    continuation_len: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if context_tokens.dim() != 2:
        raise ValueError(f"Expected [batch, context_len], got shape {tuple(context_tokens.shape)}")

    total_new = prefix_len + continuation_len
    if total_new <= 0:
        raise ValueError("prefix_len + continuation_len must be positive")

    if pad_token_id is None:
        raise ValueError("pad_token_id must be set for generation")

    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": total_new,
        "min_new_tokens": total_new,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        # Force fixed-length decoding; no early stop by eos.
        "eos_token_id": None,
    }

    if do_sample:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    else:
        generation_kwargs["do_sample"] = False

    generated = rollout_model.generate(context_tokens, **generation_kwargs)
    expected_len = context_tokens.size(1) + total_new
    if generated.size(1) != expected_len:
        raise RuntimeError(
            f"Rollout returned length {generated.size(1)}, expected {expected_len}."
        )

    produced = generated[:, context_tokens.size(1) :]
    hat_y = produced[:, :prefix_len]
    z_tokens = produced[:, prefix_len:]

    if hat_y.size(1) != prefix_len or z_tokens.size(1) != continuation_len:
        raise RuntimeError(
            f"Unexpected rollout split shapes hat_y={tuple(hat_y.shape)}, z={tuple(z_tokens.shape)}"
        )

    return hat_y, z_tokens
