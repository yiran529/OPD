from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, List

import torch
import torch.nn.functional as F

from opd.losses import OpdLossBundle, time_weighted_kl_from_logits


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
    raise TypeError("unsupported state object type")


def _detach_tree(obj):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    if isinstance(obj, list):
        return [_detach_tree(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_tree(item) for item in obj)
    if isinstance(obj, dict):
        return {key: _detach_tree(value) for key, value in obj.items()}
    # FLA officially uses fla.models.utils.Cache for past_key_values.
    from fla.models.utils import Cache as FLACache
    if isinstance(obj, FLACache):
        detached_legacy = _detach_tree(obj.to_legacy_cache())
        return FLACache.from_legacy_cache(detached_legacy)
    raise TypeError(f"unsupported detach object type: {type(obj)}")


def _extract_layer_state(layer_state: dict, state_key: str, layer_idx: int):
    assert isinstance(layer_state, dict), f"layer {layer_idx}: state must be dict"
    assert state_key in layer_state, f"layer {layer_idx}: missing state_key={state_key}"
    state_obj = layer_state[state_key]
    assert state_obj is not None, f"layer {layer_idx}: state is None"
    return state_obj


def _assert_valid_cache(past_key_values, where: str) -> None:
    assert past_key_values is not None, f"{where}: past_key_values is None"
    assert len(past_key_values) > 0, f"{where}: past_key_values is empty"


def _prefill_cache(
    model: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    requires_grad: bool,
):
    assert prefix_tokens.dim() == 2, f"prefix_tokens shape mismatch: expected rank=2, got shape={tuple(prefix_tokens.shape)}"
    assert prefix_tokens.size(1) > 0, "prefix_tokens must be non-empty"

    grad_ctx = nullcontext() if requires_grad else torch.no_grad()
    with grad_ctx:
        outputs = model(
            input_ids=prefix_tokens,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
    logits = getattr(outputs, "logits", None)
    assert logits is not None, "prefill: logits missing"
    assert logits.dim() == 3 and logits.size(1) == prefix_tokens.size(1), (
        "prefill logits shape mismatch: "
        f"expected [batch,{prefix_tokens.size(1)},vocab], got shape={tuple(logits.shape)}"
    )
    past_key_values = getattr(outputs, "past_key_values", None)
    _assert_valid_cache(past_key_values, where="prefill")
    return logits[:, -1, :], past_key_values


def _decode_one_token(
    model: torch.nn.Module,
    token: torch.Tensor,
    past_key_values,
    requires_grad: bool,
):
    assert token.dim() == 2 and token.size(1) == 1, f"token shape mismatch: expected [batch,1], got shape={tuple(token.shape)}"

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
    assert logits is not None, "decode_one_token: logits missing"
    assert logits.dim() == 3 and logits.size(1) == 1, (
        f"decode_one_token logits shape mismatch: expected [batch,1,vocab], got shape={tuple(logits.shape)}"
    )

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
    assert len(corrupted_cache) == len(clean_cache), "cache layer count mismatch"
    assert total_steps > 0, "total_steps must be positive"
    assert 0 <= time_step < total_steps, "time_step out of range"

    # State alignment at continuation step t:
    #   L_state(t) = w_t * mean_{layers,tensors} [ (1 - cos(a_t, b_t)) + 0.01 * (||a_t|| - ||b_t||)^2 ]
    # where a_t is corrupted-path state and b_t is clean-path state (stop-grad), and w_t = ((t+1)/T)^2 with t=time_step, T=total_steps.
    time_weight = ((time_step + 1) / total_steps) ** 2
    align_terms: List[torch.Tensor] = []

    for layer_idx in range(len(corrupted_cache)):
        corr_layer_state = _extract_layer_state(corrupted_cache[layer_idx], state_key, layer_idx)
        clean_layer_state = _extract_layer_state(clean_cache[layer_idx], state_key, layer_idx)

        corr_tensors = list(_iter_state_tensors(corr_layer_state))
        clean_tensors = list(_iter_state_tensors(clean_layer_state))

        assert len(corr_tensors) == len(clean_tensors), f"layer {layer_idx}: state tensor arity mismatch"
        assert len(corr_tensors) > 0, f"layer {layer_idx}: no state tensors"

        for tensor_idx, (corr_tensor, clean_tensor) in enumerate(zip(corr_tensors, clean_tensors)):
            assert corr_tensor.shape == clean_tensor.shape, (
                f"layer {layer_idx} tensor {tensor_idx} shape mismatch: expected={tuple(clean_tensor.shape)} got={tuple(corr_tensor.shape)}"
            )

            a = corr_tensor.float()
            b = clean_tensor.detach().float()
            cos_loss = 1.0 - F.cosine_similarity(a, b, dim=-1).mean()
            norm_loss = ((a.norm(dim=-1) - b.norm(dim=-1)) ** 2).mean()
            align_terms.append(
                cos_loss + 0.01 * norm_loss
            )

    assert align_terms, "no state alignment terms"
    return time_weight * torch.stack(align_terms).mean()


def compute_stepwise_opd_losses(
    model: torch.nn.Module,
    context_tokens: torch.Tensor,
    corrupted_prefix_tokens: torch.Tensor,
    clean_prefix_tokens: torch.Tensor,
    corrupted_z_tokens: torch.Tensor,
    clean_z_tokens: torch.Tensor,
    lambda_state: float,
    state_key: str,
    state_time_stride: int,
) -> OpdLossBundle:
    assert context_tokens.dim() == 2, f"context_tokens shape mismatch: expected rank=2, got shape={tuple(context_tokens.shape)}"
    assert corrupted_prefix_tokens.dim() == 2, (
        f"corrupted_prefix_tokens shape mismatch: expected rank=2, got shape={tuple(corrupted_prefix_tokens.shape)}"
    )
    assert clean_prefix_tokens.dim() == 2, (
        f"clean_prefix_tokens shape mismatch: expected rank=2, got shape={tuple(clean_prefix_tokens.shape)}"
    )
    assert corrupted_z_tokens.dim() == 2, (
        f"corrupted_z_tokens shape mismatch: expected rank=2, got shape={tuple(corrupted_z_tokens.shape)}"
    )
    assert clean_z_tokens.dim() == 2, (
        f"clean_z_tokens shape mismatch: expected rank=2, got shape={tuple(clean_z_tokens.shape)}"
    )
    assert context_tokens.size(0) == corrupted_z_tokens.size(0), "batch size mismatch for corrupted_z_tokens"
    assert context_tokens.size(0) == clean_z_tokens.size(0), "batch size mismatch for clean_z_tokens"
    assert corrupted_z_tokens.shape == clean_z_tokens.shape, (
        f"continuation shape mismatch: corrupted={tuple(corrupted_z_tokens.shape)} clean={tuple(clean_z_tokens.shape)}"
    )
    assert state_time_stride > 0, "state_time_stride must be positive"

    continuation_len = corrupted_z_tokens.size(1)
    assert continuation_len > 0, "continuation tokens must be non-empty"

    corrupted_prefix = torch.cat([context_tokens, corrupted_prefix_tokens], dim=1)
    clean_prefix = torch.cat([context_tokens, clean_prefix_tokens], dim=1)

    model_was_training = model.training

    if model_was_training:
        model.train()
    # Hard-code prefix stop-grad: do not backprop through x + y_hat prefill.
    corr_prev_logits, corrupted_cache = _prefill_cache(
        model=model,
        prefix_tokens=corrupted_prefix,
        requires_grad=False,
    )

    model.eval()
    clean_prev_logits, clean_cache = _prefill_cache(
        model=model,
        prefix_tokens=clean_prefix,
        requires_grad=False,
    )

    if model_was_training:
        model.train()

    # Stream accumulators keep loss construction explicit and avoid retaining per-step lists.
    kl_sum: torch.Tensor | None = None
    state_sum: torch.Tensor | None = None
    state_count = 0

    for t in range(continuation_len):
        # KL at continuation step t (supervise current token z_t):
        #   L_KL(t) = KL( p_theta(. | x, y_hat, hat_z_{<t}) || sg(p_theta(. | x, y, z_{<t})) )
        kl_term = time_weighted_kl_from_logits(
            student_logits=corr_prev_logits,
            teacher_logits=clean_prev_logits,
            time_step=t,
            total_steps=continuation_len,
        )
        kl_sum = kl_term if kl_sum is None else kl_sum + kl_term

        corr_token_t = corrupted_z_tokens[:, t : t + 1]
        clean_token_t = clean_z_tokens[:, t : t + 1]

        if model_was_training:
            model.train()
        corr_next_logits, next_corrupted_cache = _decode_one_token(
            model=model,
            token=corr_token_t,
            past_key_values=corrupted_cache,
            requires_grad=True,
        )

        model.eval()
        clean_next_logits, clean_cache = _decode_one_token(
            model=model,
            token=clean_token_t,
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

        corr_prev_logits = corr_next_logits
        clean_prev_logits = clean_next_logits

        # Truncate history graph: gradients at step t+1 do not flow into step t.
        corrupted_cache = _detach_tree(next_corrupted_cache)

    if model_was_training:
        model.train()
    else:
        model.eval()

    assert kl_sum is not None, "no kl terms"
    kl_loss = kl_sum / continuation_len

    if state_sum is not None and state_count > 0:
        state_loss = state_sum / state_count
    else:
        state_loss = torch.zeros_like(kl_loss)

    total = kl_loss + lambda_state * state_loss
    return OpdLossBundle(total=total, kl=kl_loss, state=state_loss)
