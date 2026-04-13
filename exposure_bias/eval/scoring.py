from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def compute_teacher_forcing_ce_batch(
    model: torch.nn.Module,
    device: torch.device,
    input_ids: torch.Tensor,
    prefix_len: int,
    rollout_len: int,
) -> dict:
    assert input_ids.dim() == 2, f"input_ids must be rank-2, got shape={tuple(input_ids.shape)}"
    batch_size, seq_len = input_ids.shape
    assert batch_size > 0, "input_ids batch must be non-empty"
    assert seq_len == prefix_len + rollout_len, (
        f"sequence length mismatch: expected={prefix_len + rollout_len} got={seq_len}"
    )

    with _autocast_context(device):
        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )

    logits = getattr(outputs, "logits", None)
    assert logits is not None, "teacher forcing: logits missing"
    assert logits.shape[:2] == input_ids.shape, "teacher forcing: logits shape mismatch"

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]

    step_logits = shifted_logits[:, prefix_len - 1 : prefix_len + rollout_len - 1, :]
    step_targets = shifted_targets[:, prefix_len - 1 : prefix_len + rollout_len - 1]
    assert step_logits.shape[1] == rollout_len, "teacher forcing step count mismatch"
    assert step_targets.shape[1] == rollout_len, "teacher forcing target count mismatch"

    ce_per_step = F.cross_entropy(
        step_logits.float().reshape(-1, step_logits.size(-1)),
        step_targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, rollout_len)

    return {
        "ce_per_step": ce_per_step,
        "ce_mean": ce_per_step.mean(dim=1),
    }


@torch.no_grad()
def compute_rollout_ce_batch(
    model: torch.nn.Module,
    device: torch.device,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    rollout_policy: str,
) -> dict:
    assert rollout_policy == "greedy", f"unsupported rollout_policy: {rollout_policy}"
    assert prefix_ids.dim() == 2, f"prefix_ids must be rank-2, got shape={tuple(prefix_ids.shape)}"
    assert target_ids.dim() == 2, f"target_ids must be rank-2, got shape={tuple(target_ids.shape)}"
    assert prefix_ids.size(0) == target_ids.size(0), "batch size mismatch"
    assert prefix_ids.size(1) > 0, "prefix_ids must be non-empty"
    assert target_ids.size(1) > 0, "target_ids must be non-empty"

    batch_size = prefix_ids.size(0)
    rollout_len = target_ids.size(1)

    # ---- prefill cache on the shared ground-truth prefix ----
    with _autocast_context(device):
        outputs = model(
            input_ids=prefix_ids,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    logits = getattr(outputs, "logits", None)
    assert logits is not None, "rollout prefill: logits missing"
    assert logits.dim() == 3 and logits.size(1) == prefix_ids.size(1), (
        "rollout prefill logits shape mismatch: "
        f"expected [batch,{prefix_ids.size(1)},vocab], got {tuple(logits.shape)}"
    )
    past_key_values = getattr(outputs, "past_key_values", None)
    assert past_key_values is not None, "rollout prefill: past_key_values missing"
    assert len(past_key_values) > 0, "rollout prefill: past_key_values empty"

    next_logits = logits[:, -1, :]
    ce_steps: list[torch.Tensor] = []
    generated_steps: list[torch.Tensor] = []

    # ---- autoregressive rollout under model-generated history ----
    for step_idx in range(rollout_len):
        step_target = target_ids[:, step_idx]
        step_ce = F.cross_entropy(next_logits.float(), step_target, reduction="none")
        ce_steps.append(step_ce)

        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated_steps.append(next_token.squeeze(1))

        if step_idx + 1 == rollout_len:
            break

        with _autocast_context(device):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )

        step_logits = getattr(outputs, "logits", None)
        assert step_logits is not None, "rollout decode: logits missing"
        assert step_logits.dim() == 3 and step_logits.shape[:2] == (batch_size, 1), (
            f"rollout decode logits shape mismatch: got {tuple(step_logits.shape)}"
        )
        past_key_values = getattr(outputs, "past_key_values", None)
        assert past_key_values is not None, "rollout decode: past_key_values missing"
        assert len(past_key_values) > 0, "rollout decode: past_key_values empty"
        next_logits = step_logits[:, 0, :]

    ce_per_step = torch.stack(ce_steps, dim=1)
    generated_ids = torch.stack(generated_steps, dim=1)
    token_match_rate = generated_ids.eq(target_ids).float().mean(dim=1)

    return {
        "ce_per_step": ce_per_step,
        "ce_mean": ce_per_step.mean(dim=1),
        "generated_ids": generated_ids,
        "token_match_rate": token_match_rate,
    }
