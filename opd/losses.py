from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OpdLossBundle:
    total: torch.Tensor
    kl: torch.Tensor
    state: torch.Tensor


def kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    assert student_logits.shape == teacher_logits.shape, (
        f"kl logits shape mismatch: expected={tuple(teacher_logits.shape)} got={tuple(student_logits.shape)}"
    )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean()
