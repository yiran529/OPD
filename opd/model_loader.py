from __future__ import annotations

from typing import Tuple

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from opd.config import TrainConfig


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _ensure_flash_linear_attention_importable() -> None:
    try:
        import fla  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "flash-linear-attention is not importable. Install fla-org/flash-linear-attention first."
        ) from exc


def _assert_expected_model_impl(model: torch.nn.Module, expected_architecture: str) -> None:
    actual_name = model.__class__.__name__
    actual_module = model.__class__.__module__
    if expected_architecture and actual_name != expected_architecture:
        raise RuntimeError(
            f"Loaded model class name mismatch: expected={expected_architecture}, got={actual_name}"
        )
    # Keep this generic for all FLA models (not only GatedDeltaNet).
    if not (actual_module.startswith("fla.") or actual_module.startswith("transformers_modules.")):
        raise RuntimeError(
            "Loaded model does not look like an FLA/remote-code model implementation: "
            f"module={actual_module}, class={actual_name}"
        )


def _assert_clean_weight_loading(loading_info: dict) -> None:
    unexpected = loading_info.get("unexpected_keys", []) or []
    missing = loading_info.get("missing_keys", []) or []
    mismatched = loading_info.get("mismatched_keys", []) or []
    error_msgs = loading_info.get("error_msgs", []) or []

    if unexpected or missing or mismatched or error_msgs:
        raise RuntimeError(
            "Model weights did not load cleanly from pretrained checkpoint. "
            f"missing_keys={missing} unexpected_keys={unexpected} "
            f"mismatched_keys={mismatched} error_msgs={error_msgs}"
        )


@torch.no_grad()
def _run_startup_sanity(
    model: torch.nn.Module,
    pad_token_id: int,
    device: torch.device,
) -> None:
    input_ids = torch.full((1, 4), pad_token_id, dtype=torch.long, device=device)
    outputs = model(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Startup sanity check failed: model outputs do not include logits")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Startup sanity check failed: logits contain non-finite values")

    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("Startup sanity check failed: past_key_values is None with use_cache=True")
    if len(past_key_values) == 0:
        raise RuntimeError("Startup sanity check failed: past_key_values is empty")

    for layer_idx in range(len(past_key_values)):
        layer_state = past_key_values[layer_idx]
        if not isinstance(layer_state, dict):
            raise RuntimeError(
                "Startup sanity check failed: expected cache layer state dict, "
                f"got type={type(layer_state)} at layer={layer_idx}"
            )
        state_keys = ("recurrent_state", "attn_state", "conv_state", "ffn_state")
        if not any(key in layer_state for key in state_keys):
            raise RuntimeError(
                "Startup sanity check failed: cache layer state has no known FLA cache keys "
                f"at layer={layer_idx}"
            )
        if all(layer_state.get(key, None) is None for key in state_keys):
            raise RuntimeError(
                "Startup sanity check failed: all known cache states are None "
                f"at layer={layer_idx}"
            )


def build_model_and_tokenizer(cfg: TrainConfig, device: torch.device) -> Tuple[torch.nn.Module, object]:
    _ensure_flash_linear_attention_importable()
    import fla

    model_dtype = resolve_dtype(cfg.dtype)
    model_id = cfg.model_name
    tokenizer_id = cfg.tokenizer_name or model_id

    model_config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=cfg.trust_remote_code,
    )

    architectures = getattr(model_config, "architectures", None) or []
    if cfg.expected_architecture and cfg.expected_architecture not in architectures:
        raise ValueError(
            f"Expected architecture {cfg.expected_architecture} not found in config.architectures={architectures}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=model_dtype,
        output_loading_info=True,
    )
    model.to(device)

    _assert_expected_model_impl(model=model, expected_architecture=cfg.expected_architecture)
    if not hasattr(fla, model.__class__.__name__):
        raise RuntimeError(
            "Loaded model class is not exported by fla package: "
            f"class={model.__class__.__name__} module={model.__class__.__module__}"
        )
    _assert_clean_weight_loading(loading_info=loading_info)

    print(
        "Model load summary: "
        f"fla_version={getattr(fla, '__version__', 'unknown')} "
        f"transformers_version={transformers.__version__} "
        f"model_class={model.__class__.__module__}.{model.__class__.__name__} "
        f"model_type={getattr(model.config, 'model_type', None)} "
        f"architectures={getattr(model.config, 'architectures', None)}",
        flush=True,
    )

    _run_startup_sanity(
        model=model,
        pad_token_id=tokenizer.pad_token_id,
        device=device,
    )

    return model, tokenizer
