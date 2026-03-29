from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OpdLossBundle:
    total: torch.Tensor
    # Legacy key name kept for logging/config compatibility; now this carries JSD.
    kl: torch.Tensor
    state: torch.Tensor


def time_weighted_jsd_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    time_step: int,
    total_steps: int,
) -> torch.Tensor:
    assert student_logits.shape == teacher_logits.shape, (
        f"kl logits shape mismatch: expected={tuple(teacher_logits.shape)} got={tuple(student_logits.shape)}"
    )
    assert 0 <= time_step < total_steps, "time_step out of range"

    time_weight = ((time_step + 1) / total_steps)

    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_probs = student_log_probs.exp()

    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    teacher_probs = teacher_log_probs.exp()

    mix_probs = 0.5 * (student_probs + teacher_probs)
    mix_log_probs = torch.log(mix_probs.clamp_min(1e-12))

    kl_student_mix = (student_probs * (student_log_probs - mix_log_probs)).sum(dim=-1).mean()
    kl_teacher_mix = (teacher_probs * (teacher_log_probs - mix_log_probs)).sum(dim=-1).mean()
    jsd_term = 0.5 * (kl_student_mix + kl_teacher_mix)
    return time_weight * jsd_term
