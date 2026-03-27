from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OpdLossBundle:
    total: torch.Tensor
    kl: torch.Tensor
    state: torch.Tensor


def time_weighted_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    time_step: int,
    total_steps: int,
) -> torch.Tensor:
    assert student_logits.shape == teacher_logits.shape, (
        f"kl logits shape mismatch: expected={tuple(teacher_logits.shape)} got={tuple(student_logits.shape)}"
    )
    assert 0 <= time_step < total_steps, "time_step out of range"

    time_weight = ((time_step + 1) / total_steps) ** 2
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    kl_term = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean()
    return time_weight * kl_term
