from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from opd.config import load_config
from opd.model_loader import build_model_and_tokenizer

try:
    from lm_eval.api.model import TemplateLM
except Exception as exc:  # pragma: no cover - optional dependency
    TemplateLM = object  # type: ignore[assignment]
    _LM_EVAL_IMPORT_ERROR = exc
else:
    _LM_EVAL_IMPORT_ERROR = None


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_checkpoint_state(
    checkpoint_path: str,
    model: torch.nn.Module,
    device: torch.device,
    use_ema: bool,
) -> int:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    state = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
    state_key = "ema_model" if use_ema else "model"
    model_state = state.get(state_key, None)
    if model_state is None:
        raise KeyError(f"Checkpoint missing required key: {state_key}")

    raw_model = _unwrap_model(model)
    raw_model.load_state_dict(model_state, strict=True)
    raw_model.to(device)
    return int(state.get("step", -1))


class OPDLMEvalModel(TemplateLM):
    def __init__(
        self,
        train_config_path: str,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        use_ema: bool = False,
    ) -> None:
        if _LM_EVAL_IMPORT_ERROR is not None:
            raise RuntimeError(
                "lm_eval is not importable. Install lm-evaluation-harness first."
            ) from _LM_EVAL_IMPORT_ERROR

        super().__init__()
        assert batch_size > 0, "batch_size must be positive"
        if use_ema and not checkpoint_path:
            raise ValueError("use_ema requires a non-empty checkpoint_path")

        self.train_config_path = train_config_path
        self.checkpoint_path = checkpoint_path
        self.batch_size_per_forward = batch_size
        self.use_ema = use_ema

        self.train_cfg = load_config(train_config_path)
        self._device = _resolve_device(device)
        self.model, self.tokenizer = build_model_and_tokenizer(
            cfg=self.train_cfg,
            device=self._device,
        )
        self.loaded_step = -1
        if checkpoint_path:
            self.loaded_step = _load_checkpoint_state(
                checkpoint_path=checkpoint_path,
                model=self.model,
                device=self._device,
                use_ema=use_ema,
            )

        self.model.eval()
        self.config = getattr(self.model, "config", None)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id for lm_eval bridge")
        self._eot_token_id = int(eos_token_id)

        self._max_length = int(
            self.train_cfg.context_len + self.train_cfg.prefix_len + self.train_cfg.continuation_len
        )

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def tokenizer_name(self) -> str:
        tokenizer_name = self.train_cfg.tokenizer_name or self.train_cfg.model_name
        return str(tokenizer_name)

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        encoded = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        return list(encoded["input_ids"])

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=False)

    def _score_token_ids(
        self,
        context_enc: list[int],
        continuation_enc: list[int],
    ) -> tuple[float, bool]:
        results = self._score_batch_token_ids(
            batch_context_enc=[context_enc],
            batch_continuation_enc=[continuation_enc],
        )
        assert len(results) == 1, "single-item score must return exactly one result"
        return results[0]

    def _score_batch_token_ids(
        self,
        batch_context_enc: list[list[int]],
        batch_continuation_enc: list[list[int]],
    ) -> list[tuple[float, bool]]:
        assert batch_context_enc, "batch_context_enc must be non-empty"
        assert len(batch_context_enc) == len(batch_continuation_enc), (
            "batch context/continuation length mismatch"
        )

        batch_size = len(batch_context_enc)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self._eot_token_id

        full_batch: list[list[int]] = []
        context_lens: list[int] = []
        full_lens: list[int] = []
        continuation_lens: list[int] = []

        for context_enc, continuation_enc in zip(batch_context_enc, batch_continuation_enc):
            assert context_enc, "context_enc must be non-empty"
            assert continuation_enc, "continuation_enc must be non-empty"

            full_ids = context_enc + continuation_enc
            assert len(full_ids) <= self.max_length, (
                f"Input length exceeds max_length: got={len(full_ids)} max={self.max_length}"
            )
            full_batch.append(full_ids)
            context_lens.append(len(context_enc))
            full_lens.append(len(full_ids))
            continuation_lens.append(len(continuation_enc))

        max_len = max(full_lens)
        assert max_len >= 2, "full sequence length must be at least 2 tokens"

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=int(pad_token_id),
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self._device)
        target_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool, device=self._device)

        for idx, (full_ids, context_len, full_len, continuation_len) in enumerate(
            zip(full_batch, context_lens, full_lens, continuation_lens)
        ):
            pad_len = max_len - full_len
            full_tensor = torch.tensor(full_ids, dtype=torch.long, device=self._device)
            input_ids[idx, pad_len:] = full_tensor
            attention_mask[idx, pad_len:] = 1

            start = pad_len + context_len - 1
            end = pad_len + full_len - 1
            assert start >= 0, "invalid continuation start index"
            assert end > start, "continuation target range must be non-empty"
            target_mask[idx, start:end] = True

            expected_targets = continuation_len
            actual_targets = int(target_mask[idx].sum().item())
            assert actual_targets == expected_targets, (
                f"continuation target count mismatch: expected={expected_targets} got={actual_targets}"
            )

        with torch.no_grad():
            with _autocast_context(self._device):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )

        logits = outputs.logits
        assert logits.shape[:2] == input_ids.shape, "logits shape mismatch"

        shifted_logits = logits[:, :-1, :]
        shifted_targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(shifted_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)

        greedy_tokens = shifted_logits.argmax(dim=-1)
        greedy_matches = greedy_tokens.eq(shifted_targets)

        results: list[tuple[float, bool]] = []
        for idx in range(batch_size):
            mask = target_mask[idx]
            assert mask.any(), "no target positions found for continuation"
            selected = token_log_probs[idx, mask]
            selected_greedy = greedy_matches[idx, mask]
            results.append((float(selected.sum().item()), bool(selected_greedy.all().item())))

        return results

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        **kwargs,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for start in range(0, len(requests), self.batch_size_per_forward):
            chunk = requests[start : start + self.batch_size_per_forward]
            batch_context_enc: list[list[int]] = []
            batch_continuation_enc: list[list[int]] = []
            for _, context_enc, continuation_enc in chunk:
                batch_context_enc.append(context_enc)
                batch_continuation_enc.append(continuation_enc)
            results.extend(
                self._score_batch_token_ids(
                    batch_context_enc=batch_context_enc,
                    batch_continuation_enc=batch_continuation_enc,
                )
            )
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("loglikelihood_rolling is not implemented for OPDLMEvalModel")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("generate_until is not implemented for OPDLMEvalModel")

    def get_model_info(self) -> dict[str, Any]:
        return {
            "train_config_path": self.train_config_path,
            "checkpoint_path": self.checkpoint_path,
            "loaded_step": self.loaded_step,
            "use_ema": self.use_ema,
            "tokenizer_name": self.tokenizer_name,
        }
