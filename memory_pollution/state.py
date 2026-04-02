from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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
    assert isinstance(layer_state, dict), f"layer {layer_idx}: state must be dict"
    assert state_key in layer_state, f"layer {layer_idx}: missing state_key={state_key}"
    state_obj = layer_state[state_key]
    assert state_obj is not None, f"layer {layer_idx}: state is None"
    return state_obj


@torch.no_grad()
def capture_prompt_cache(
    model: torch.nn.Module,
    device: torch.device,
    prompt_token_ids: list[int],
):
    assert prompt_token_ids, "prompt_token_ids must be non-empty"

    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with _autocast_context(device):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,
        )

    past_key_values = getattr(outputs, "past_key_values", None)
    assert past_key_values is not None, "past_key_values is None"
    assert len(past_key_values) > 0, "past_key_values is empty"
    return past_key_values


def compute_state_drift(
    clean_cache,
    perturbed_cache,
    state_key: str,
) -> dict:
    assert len(clean_cache) == len(perturbed_cache), "cache layer count mismatch"

    total_num = None
    total_den = None
    per_layer_drift: list[dict] = []

    for layer_idx in range(len(clean_cache)):
        clean_state = _extract_layer_state(clean_cache[layer_idx], state_key=state_key, layer_idx=layer_idx)
        perturbed_state = _extract_layer_state(
            perturbed_cache[layer_idx],
            state_key=state_key,
            layer_idx=layer_idx,
        )

        clean_tensors = list(_iter_state_tensors(clean_state))
        perturbed_tensors = list(_iter_state_tensors(perturbed_state))
        assert len(clean_tensors) == len(perturbed_tensors), f"layer {layer_idx}: state tensor arity mismatch"
        assert clean_tensors, f"layer {layer_idx}: no state tensors"

        layer_num = None
        layer_den = None
        for tensor_idx, (clean_tensor, perturbed_tensor) in enumerate(zip(clean_tensors, perturbed_tensors)):
            assert clean_tensor.shape == perturbed_tensor.shape, (
                f"layer {layer_idx} tensor {tensor_idx} shape mismatch: "
                f"clean={tuple(clean_tensor.shape)} perturbed={tuple(perturbed_tensor.shape)}"
            )
            diff_sq = (perturbed_tensor.float() - clean_tensor.float()).pow(2).sum()
            clean_sq = clean_tensor.float().pow(2).sum()

            layer_num = diff_sq if layer_num is None else layer_num + diff_sq
            layer_den = clean_sq if layer_den is None else layer_den + clean_sq

        assert layer_num is not None and layer_den is not None, "layer drift accumulators must be initialized"
        layer_drift = float((layer_num / layer_den.clamp_min(1e-12)).item())
        per_layer_drift.append({"layer_idx": layer_idx, "drift": layer_drift})

        total_num = layer_num if total_num is None else total_num + layer_num
        total_den = layer_den if total_den is None else total_den + layer_den

    assert total_num is not None and total_den is not None, "overall drift accumulators must be initialized"
    return {
        "overall_drift": float((total_num / total_den.clamp_min(1e-12)).item()),
        "per_layer_drift": per_layer_drift,
    }

