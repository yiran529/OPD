from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, List

import torch
import torch.nn.functional as F

from opd.losses import OpdLossBundle, ce_from_logits, kl_from_logits


def _iter_state_tensors(state_obj) -> Iterable[torch.Tensor]:
    if isinstance(state_obj, torch.Tensor):
        yield state_obj
        return
    if isinstance(state_obj, (list, tuple)):
        for item in state_obj:
            yield from _iter_state_tensors(item)
        return
    if isinstance(state_obj, dict):
        for key in sorted(state_obj.keys()):
            yield from _iter_state_tensors(state_obj[key])
        return
    raise TypeError(f"Unsupported state object type: {type(state_obj)}")


def _detach_tree(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, list):
        return [_detach_tree(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_tree(item) for item in obj)
    if isinstance(obj, dict):
        return {key: _detach_tree(value) for key, value in obj.items()}
    raise TypeError(f"Unsupported detach object type: {type(obj)}")


def _extract_layer_state(layer_state: dict, state_key: str, layer_idx: int):
    if not isinstance(layer_state, dict):
        raise TypeError(
            "Expected cache layer state to be dict, "
            f"got type={type(layer_state)} at layer={layer_idx}"
        )
    if state_key not in layer_state:
        raise KeyError(
            f"Missing state_key='{state_key}' at layer={layer_idx}; "
            f"available_keys={sorted(layer_state.keys())}"
        )
    state_obj = layer_state[state_key]
    if state_obj is None:
        raise RuntimeError(
            f"Cache state is None for state_key='{state_key}' at layer={layer_idx}"
        )
    return state_obj


def _assert_valid_cache(past_key_values, where: str) -> None:
    if past_key_values is None:
        raise RuntimeError(f"{where}: past_key_values is None")
    if len(past_key_values) == 0:
        raise RuntimeError(f"{where}: past_key_values is empty")


def _prefill_cache(
    model: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    requires_grad: bool,
):
    if prefix_tokens.dim() != 2:
        raise ValueError(f"Expected [batch, prefix_len], got shape {tuple(prefix_tokens.shape)}")
    if prefix_tokens.size(1) <= 0:
        raise ValueError("prefix_tokens must have positive length")

    grad_ctx = nullcontext() if requires_grad else torch.no_grad()
    with grad_ctx:
        outputs = model(
            input_ids=prefix_tokens,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
    past_key_values = getattr(outputs, "past_key_values", None)
    _assert_valid_cache(past_key_values, where="prefill")
    return past_key_values


def _decode_one_token(
    model: torch.nn.Module,
    token: torch.Tensor,
    past_key_values,
    requires_grad: bool,
):
    if token.dim() != 2 or token.size(1) != 1:
        raise ValueError(f"Expected token shape [batch, 1], got {tuple(token.shape)}")

    grad_ctx = nullcontext() if requires_grad else torch.no_grad()
    with grad_ctx:
        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("decode_one_token: logits is missing")
    if logits.dim() != 3 or logits.size(1) != 1:
        raise RuntimeError(f"decode_one_token: expected logits [batch,1,vocab], got {tuple(logits.shape)}")

    next_past_key_values = getattr(outputs, "past_key_values", None)
    _assert_valid_cache(next_past_key_values, where="decode_one_token")
    return logits[:, 0, :], next_past_key_values


def _state_alignment_loss_from_caches(
    corrupted_cache,
    clean_cache,
    state_key: str,
    time_step: int,
    total_steps: int,
) -> torch.Tensor:
    if len(corrupted_cache) != len(clean_cache):
        raise RuntimeError(
            "Cache layer count mismatch: "
            f"corrupted={len(corrupted_cache)} clean={len(clean_cache)}"
        )
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if time_step < 0 or time_step >= total_steps:
        raise ValueError(f"time_step out of range: time_step={time_step}, total_steps={total_steps}")

    time_weight = ((time_step + 1) / total_steps) ** 2
    align_terms: List[torch.Tensor] = []

    for layer_idx in range(len(corrupted_cache)):
        corr_layer_state = _extract_layer_state(corrupted_cache[layer_idx], state_key, layer_idx)
        clean_layer_state = _extract_layer_state(clean_cache[layer_idx], state_key, layer_idx)

        corr_tensors = list(_iter_state_tensors(corr_layer_state))
        clean_tensors = list(_iter_state_tensors(clean_layer_state))

        if len(corr_tensors) != len(clean_tensors):
            raise RuntimeError(
                "State tensor arity mismatch at layer "
                f"{layer_idx}: corrupted={len(corr_tensors)} clean={len(clean_tensors)}"
            )
        if len(corr_tensors) == 0:
            raise RuntimeError(f"No state tensors found at layer={layer_idx} key={state_key}")

        for tensor_idx, (corr_tensor, clean_tensor) in enumerate(zip(corr_tensors, clean_tensors)):
            if corr_tensor.shape != clean_tensor.shape:
                raise RuntimeError(
                    "State tensor shape mismatch "
                    f"at layer={layer_idx} tensor={tensor_idx}: "
                    f"corrupted={tuple(corr_tensor.shape)} clean={tuple(clean_tensor.shape)}"
                )

            a = corr_tensor.float()
            b = clean_tensor.detach().float()
            cos_loss = 1.0 - F.cosine_similarity(a, b, dim=-1).mean()
            norm_loss = ((a.norm(dim=-1) - b.norm(dim=-1)) ** 2).mean()
            align_terms.append(
                cos_loss + 0.01 * norm_loss
            )

    if not align_terms:
        raise RuntimeError("No state alignment terms collected from cache states")
    return time_weight * torch.stack(align_terms).mean()


def compute_stepwise_opd_losses(
    model: torch.nn.Module,
    context_tokens: torch.Tensor,
    corrupted_prefix_tokens: torch.Tensor,
    clean_prefix_tokens: torch.Tensor,
    z_tokens: torch.Tensor,
    lambda_state: float,
    ce_anchor_weight: float,
    state_key: str,
    state_time_stride: int,
) -> OpdLossBundle:
    if context_tokens.dim() != 2:
        raise ValueError(f"Expected context_tokens [batch, context_len], got {tuple(context_tokens.shape)}")
    if corrupted_prefix_tokens.dim() != 2 or clean_prefix_tokens.dim() != 2:
        raise ValueError("Expected prefix tensors to be rank-2")
    if z_tokens.dim() != 2:
        raise ValueError(f"Expected z_tokens [batch, continuation_len], got {tuple(z_tokens.shape)}")
    if context_tokens.size(0) != z_tokens.size(0):
        raise ValueError("Batch size mismatch between context_tokens and z_tokens")
    if state_time_stride <= 0:
        raise ValueError(f"state_time_stride must be positive, got {state_time_stride}")

    continuation_len = z_tokens.size(1)
    if continuation_len <= 0:
        raise ValueError("z_tokens must have positive continuation length")

    corrupted_prefix = torch.cat([context_tokens, corrupted_prefix_tokens], dim=1)
    clean_prefix = torch.cat([context_tokens, clean_prefix_tokens], dim=1)

    model_was_training = model.training

    if model_was_training:
        model.train()
    # Hard-code prefix stop-grad: do not backprop through x + y_hat prefill.
    corrupted_cache = _prefill_cache(
        model=model,
        prefix_tokens=corrupted_prefix,
        requires_grad=False,
    )

    model.eval()
    clean_cache = _prefill_cache(
        model=model,
        prefix_tokens=clean_prefix,
        requires_grad=False,
    )

    if model_was_training:
        model.train()

    # Stream accumulators keep loss construction explicit and avoid retaining per-step lists.
    kl_sum: torch.Tensor | None = None
    state_sum: torch.Tensor | None = None
    ce_sum: torch.Tensor | None = None
    state_count = 0
    ce_count = 0

    for t in range(continuation_len):
        token_t = z_tokens[:, t : t + 1]

        if model_was_training:
            model.train()
        corr_logits_t, next_corrupted_cache = _decode_one_token(
            model=model,
            token=token_t,
            past_key_values=corrupted_cache,
            requires_grad=True,
        )

        model.eval()
        clean_logits_t, clean_cache = _decode_one_token(
            model=model,
            token=token_t,
            past_key_values=clean_cache,
            requires_grad=False,
        )

        if t % state_time_stride == 0:
            state_term = _state_alignment_loss_from_caches(
                corrupted_cache=next_corrupted_cache,
                clean_cache=clean_cache,
                state_key=state_key,
                time_step=t,
                total_steps=continuation_len,
            )
            state_sum = state_term if state_sum is None else state_sum + state_term
            state_count += 1

        kl_term = kl_from_logits(
            student_logits=corr_logits_t,
            teacher_logits=clean_logits_t,
        )
        kl_sum = kl_term if kl_sum is None else kl_sum + kl_term

        if ce_anchor_weight > 0.0:
            ce_term = ce_from_logits(
                logits=corr_logits_t,
                targets=token_t.squeeze(1),
            )
            ce_sum = ce_term if ce_sum is None else ce_sum + ce_term
            ce_count += 1

        # Truncate history graph: gradients at step t+1 do not flow into step t.
        corrupted_cache = _detach_tree(next_corrupted_cache)

    if model_was_training:
        model.train()
    else:
        model.eval()

    if kl_sum is None:
        raise RuntimeError("No KL terms were collected")
    kl_loss = kl_sum / continuation_len

    if state_sum is not None and state_count > 0:
        state_loss = state_sum / state_count
    else:
        state_loss = torch.zeros_like(kl_loss)

    if ce_sum is not None and ce_count > 0:
        ce_anchor_loss = ce_sum / ce_count
    else:
        ce_anchor_loss = torch.zeros_like(kl_loss)

    total = kl_loss + lambda_state * state_loss + ce_anchor_weight * ce_anchor_loss
    return OpdLossBundle(total=total, kl=kl_loss, state=state_loss, ce_anchor=ce_anchor_loss)
