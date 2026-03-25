from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OpdLossBundle:
    total: torch.Tensor
    kl: torch.Tensor
    state: torch.Tensor
    ce_anchor: torch.Tensor


def _continuation_slice(context_len: int, prefix_len: int, continuation_len: int) -> slice:
    start = context_len + prefix_len - 1
    end = start + continuation_len
    if start < 0:
        raise ValueError("Continuation slice start is negative")
    if continuation_len <= 0:
        raise ValueError("continuation_len must be positive")
    return slice(start, end)


def _extract_z_logits(
    logits: torch.Tensor,
    context_len: int,
    prefix_len: int,
    continuation_len: int,
) -> torch.Tensor:
    z_slice = _continuation_slice(context_len, prefix_len, continuation_len)
    z_logits = logits[:, z_slice, :]
    if z_logits.size(1) != continuation_len:
        raise RuntimeError(
            f"Unexpected z-logits shape {tuple(z_logits.shape)}, expected continuation_len={continuation_len}"
        )
    return z_logits


def _extract_z_state(
    model_outputs,
    context_len: int,
    prefix_len: int,
    continuation_len: int,
) -> torch.Tensor:
    hidden_states = getattr(model_outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model outputs do not include hidden_states; set output_hidden_states=True")
    if len(hidden_states) == 0:
        raise RuntimeError("hidden_states is empty")

    last_hidden = hidden_states[-1]
    z_slice = _continuation_slice(context_len, prefix_len, continuation_len)
    z_state = last_hidden[:, z_slice, :]
    if z_state.size(1) != continuation_len:
        raise RuntimeError(
            f"Unexpected z-state shape {tuple(z_state.shape)}, expected continuation_len={continuation_len}"
        )
    return z_state


def compute_opd_kl_state_loss(
    corrupted_outputs,
    clean_outputs,
    z_tokens: torch.Tensor,
    context_len: int,
    prefix_len: int,
    lambda_state: float,
    ce_anchor_weight: float,
) -> OpdLossBundle:
    continuation_len = z_tokens.size(1)

    corr_logits = _extract_z_logits(
        corrupted_outputs.logits,
        context_len=context_len,
        prefix_len=prefix_len,
        continuation_len=continuation_len,
    )
    clean_logits = _extract_z_logits(
        clean_outputs.logits,
        context_len=context_len,
        prefix_len=prefix_len,
        continuation_len=continuation_len,
    ).detach()

    student_log_probs = F.log_softmax(corr_logits, dim=-1)
    teacher_probs = F.softmax(clean_logits, dim=-1)
    kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    kl_loss = kl_per_token.mean()

    corr_state = _extract_z_state(
        corrupted_outputs,
        context_len=context_len,
        prefix_len=prefix_len,
        continuation_len=continuation_len,
    )
    clean_state = _extract_z_state(
        clean_outputs,
        context_len=context_len,
        prefix_len=prefix_len,
        continuation_len=continuation_len,
    ).detach()
    state_loss = F.mse_loss(corr_state, clean_state)

    ce_anchor_loss = torch.zeros_like(kl_loss)
    if ce_anchor_weight > 0.0:
        ce_anchor_loss = F.cross_entropy(
            corr_logits.reshape(-1, corr_logits.size(-1)),
            z_tokens.reshape(-1),
        )

    total = kl_loss + lambda_state * state_loss + ce_anchor_weight * ce_anchor_loss
    return OpdLossBundle(total=total, kl=kl_loss, state=state_loss, ce_anchor=ce_anchor_loss)
