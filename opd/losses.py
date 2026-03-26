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


def ce_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    assert logits.dim() == 2, f"logits shape mismatch: expected rank=2 [batch,vocab], got shape={tuple(logits.shape)}"
    assert targets.dim() == 1, f"targets shape mismatch: expected rank=1 [batch], got shape={tuple(targets.shape)}"
    assert logits.size(0) == targets.size(0), "batch mismatch"
    return F.cross_entropy(logits, targets)
