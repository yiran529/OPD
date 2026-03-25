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
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "KL logits shape mismatch: "
            f"student={tuple(student_logits.shape)} teacher={tuple(teacher_logits.shape)}"
        )
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1).mean()


def ce_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"Expected logits [batch, vocab], got {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"Expected targets [batch], got {tuple(targets.shape)}")
    if logits.size(0) != targets.size(0):
        raise ValueError(
            "Batch mismatch between logits and targets: "
            f"logits={logits.size(0)} targets={targets.size(0)}"
        )
    return F.cross_entropy(logits, targets)
