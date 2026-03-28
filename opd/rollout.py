from __future__ import annotations

from typing import Tuple

import torch


def _build_generation_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
) -> dict:
    assert max_new_tokens > 0, "max_new_tokens must be positive"
    assert pad_token_id is not None, "pad_token_id is required"

    do_sample = temperature > 0.0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": max_new_tokens,
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
    return generation_kwargs


@torch.no_grad()
def generate_dual_rollout_tokens(
    model: torch.nn.Module,
    context_tokens: torch.Tensor,
    clean_prefix_tokens: torch.Tensor,
    prefix_len: int,
    continuation_len: int,
    temperature: float,
    top_p: float,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert context_tokens.dim() == 2, f"context_tokens shape mismatch: expected rank=2, got shape={tuple(context_tokens.shape)}"
    assert clean_prefix_tokens.dim() == 2, (
        f"clean_prefix_tokens shape mismatch: expected rank=2, got shape={tuple(clean_prefix_tokens.shape)}"
    )
    assert context_tokens.size(0) == clean_prefix_tokens.size(0), "batch size mismatch between context and clean_prefix"
    assert clean_prefix_tokens.size(1) == prefix_len, (
        f"clean_prefix_tokens length mismatch: expected={prefix_len} got={clean_prefix_tokens.size(1)}"
    )

    total_new = prefix_len + continuation_len
    assert total_new > 0, "prefix_len + continuation_len must be positive"
    assert continuation_len > 0, "continuation_len must be positive"

    # ---- corrupted path rollout: x -> hat_y + hat_z ----
    corrupted_kwargs = _build_generation_kwargs(
        max_new_tokens=total_new,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_token_id,
    )
    corrupted_attention_mask = torch.ones_like(context_tokens)
    corrupted_generated = model.generate(
        context_tokens,
        attention_mask=corrupted_attention_mask,
        **corrupted_kwargs,
    )
    expected_corrupted_len = context_tokens.size(1) + total_new
    assert corrupted_generated.size(1) == expected_corrupted_len, (
        f"corrupted rollout length mismatch: expected={expected_corrupted_len} got={corrupted_generated.size(1)}"
    )

    corrupted_produced = corrupted_generated[:, context_tokens.size(1) :]
    hat_y = corrupted_produced[:, :prefix_len]
    hat_z = corrupted_produced[:, prefix_len:]
    assert hat_y.size(1) == prefix_len and hat_z.size(1) == continuation_len, (
        "corrupted rollout split mismatch: "
        f"expected hat_y={prefix_len}, hat_z={continuation_len}; got hat_y={hat_y.size(1)}, hat_z={hat_z.size(1)}"
    )

    # ---- clean path rollout: x + y -> z ----
    clean_prompt = torch.cat([context_tokens, clean_prefix_tokens], dim=1)
    clean_kwargs = _build_generation_kwargs(
        max_new_tokens=continuation_len,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_token_id,
    )
    clean_attention_mask = torch.ones_like(clean_prompt)
    clean_generated = model.generate(
        clean_prompt,
        attention_mask=clean_attention_mask,
        **clean_kwargs,
    )
    expected_clean_len = clean_prompt.size(1) + continuation_len
    assert clean_generated.size(1) == expected_clean_len, (
        f"clean rollout length mismatch: expected={expected_clean_len} got={clean_generated.size(1)}"
    )
    clean_z = clean_generated[:, clean_prompt.size(1) :]
    assert clean_z.size(1) == continuation_len, (
        f"clean rollout split mismatch: expected z={continuation_len}, got z={clean_z.size(1)}"
    )

    return hat_y, hat_z, clean_z
