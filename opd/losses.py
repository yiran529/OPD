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
    # return time_weight * jsd_term
    return jsd_term


def gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() >= 2, f"state tensor rank must be >= 2 for gram alignment, got shape={tuple(tensor.shape)}"

    matrix = tensor.float()
    rows = matrix.size(-2)
    cols = matrix.size(-1)
    matrix = matrix.reshape(-1, rows, cols)
    return torch.matmul(matrix, matrix.transpose(-1, -2))


def gram_matrix_mse(
    student_tensor: torch.Tensor,
    teacher_tensor: torch.Tensor,
) -> torch.Tensor:
    gram_student = gram_matrix(student_tensor)
    gram_teacher = gram_matrix(teacher_tensor.detach())
    return F.mse_loss(gram_student, gram_teacher)


def cosine_norm_state_loss(
    student_tensor: torch.Tensor,
    teacher_tensor: torch.Tensor,
) -> torch.Tensor:
    student_state = student_tensor.float()
    teacher_state = teacher_tensor.detach().float()
    cos_loss = 1.0 - F.cosine_similarity(student_state, teacher_state, dim=-1).mean()
    norm_loss = ((student_state.norm(dim=-1) - teacher_state.norm(dim=-1)) ** 2).mean()
    return cos_loss + 0.1 * norm_loss


def state_tensor_alignment_loss(
    student_tensor: torch.Tensor,
    teacher_tensor: torch.Tensor,
    state_align_loss: str,
) -> torch.Tensor:
    if state_align_loss == "gram_mse":
        return gram_matrix_mse(
            student_tensor=student_tensor,
            teacher_tensor=teacher_tensor,
        )

    if state_align_loss == "cos_norm":
        return cosine_norm_state_loss(
            student_tensor=student_tensor,
            teacher_tensor=teacher_tensor,
        )

    raise ValueError(f"Unsupported state_align_loss: {state_align_loss}")
