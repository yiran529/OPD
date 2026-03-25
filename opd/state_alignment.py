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


def _state_mse_from_caches(
    corrupted_cache,
    clean_cache,
    state_key: str,
) -> torch.Tensor:
    if len(corrupted_cache) != len(clean_cache):
        raise RuntimeError(
            "Cache layer count mismatch: "
            f"corrupted={len(corrupted_cache)} clean={len(clean_cache)}"
        )

    mse_terms: List[torch.Tensor] = []

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
            mse_terms.append(
                F.mse_loss(
                    corr_tensor.float(),
                    clean_tensor.detach().float(),
                    reduction="mean",
                )
            )

    if not mse_terms:
        raise RuntimeError("No MSE terms collected from cache states")
    return torch.stack(mse_terms).mean()


def compute_stepwise_opd_losses(
    model: torch.nn.Module,
    context_tokens: torch.Tensor,
    corrupted_prefix_tokens: torch.Tensor,
    clean_prefix_tokens: torch.Tensor,
    z_tokens: torch.Tensor,
    lambda_state: float,
    ce_anchor_weight: float,
    state_key: str,
    grad_through_prefix: bool,
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
    corrupted_cache = _prefill_cache(
        model=model,
        prefix_tokens=corrupted_prefix,
        requires_grad=grad_through_prefix,
    )

    model.eval()
    clean_cache = _prefill_cache(
        model=model,
        prefix_tokens=clean_prefix,
        requires_grad=False,
    )

    if model_was_training:
        model.train()

    kl_terms: List[torch.Tensor] = []
    state_terms: List[torch.Tensor] = []
    ce_terms: List[torch.Tensor] = []

    for t in range(continuation_len):
        token_t = z_tokens[:, t : t + 1]

        if model_was_training:
            model.train()
        corr_logits_t, corrupted_cache = _decode_one_token(
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
            state_terms.append(
                _state_mse_from_caches(
                    corrupted_cache=corrupted_cache,
                    clean_cache=clean_cache,
                    state_key=state_key,
                )
            )

        kl_terms.append(
            kl_from_logits(
                student_logits=corr_logits_t,
                teacher_logits=clean_logits_t,
            )
        )

        if ce_anchor_weight > 0.0:
            ce_terms.append(
                ce_from_logits(
                    logits=corr_logits_t,
                    targets=token_t.squeeze(1),
                )
            )

    if model_was_training:
        model.train()
    else:
        model.eval()

    kl_loss = torch.stack(kl_terms).mean()
    state_loss = torch.stack(state_terms).mean() if state_terms else torch.zeros_like(kl_loss)
    ce_anchor_loss = torch.stack(ce_terms).mean() if ce_terms else torch.zeros_like(kl_loss)

    total = kl_loss + lambda_state * state_loss + ce_anchor_weight * ce_anchor_loss
    return OpdLossBundle(total=total, kl=kl_loss, state=state_loss, ce_anchor=ce_anchor_loss)
