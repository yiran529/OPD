from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DistillLossBundle:
    total: torch.Tensor
    forward_kl: torch.Tensor
    reverse_kl: torch.Tensor


def mixed_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    alpha: float,
) -> DistillLossBundle:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logits shape mismatch: student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    student_probs = student_log_probs.exp()

    teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()

    forward_kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
    reverse_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
    total = alpha * forward_kl + (1.0 - alpha) * reverse_kl
    return DistillLossBundle(
        total=total,
        forward_kl=forward_kl,
        reverse_kl=reverse_kl,
    )

