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


def _resolve_model_max_length(model_config: Any, fallback: int) -> int:
    candidate_keys = (
        "max_position_embeddings",
        "n_positions",
        "n_ctx",
        "seq_len",
    )
    for key in candidate_keys:
        value = getattr(model_config, key, None)
        if isinstance(value, int) and 0 < value < 1_000_000:
            return int(value)
    return int(fallback)


def _normalize_until(until: Any) -> tuple[str, ...]:
    if until is None:
        return tuple()
    if isinstance(until, str):
        return (until,) if until else tuple()
    if isinstance(until, (list, tuple)):
        stops: list[str] = []
        for stop in until:
            if not isinstance(stop, str):
                raise TypeError(f"until must contain only str values, got type={type(stop)}")
            if stop:
                stops.append(stop)
        return tuple(stops)
    raise TypeError(f"until must be str/list/tuple/None, got type={type(until)}")


def _trim_by_until(text: str, until: tuple[str, ...]) -> str:
    if not until:
        return text
    cut_at: Optional[int] = None
    for stop in until:
        idx = text.find(stop)
        if idx < 0:
            continue
        if cut_at is None or idx < cut_at:
            cut_at = idx
    if cut_at is None:
        return text
    return text[:cut_at]


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

        fallback_max_len = int(
            self.train_cfg.context_len + self.train_cfg.prefix_len + self.train_cfg.continuation_len
        )
        self._max_length = _resolve_model_max_length(
            model_config=self.config,
            fallback=fallback_max_len,
        )
        self._max_gen_toks = 256

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

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

    def _parse_generate_request(self, request: Any, index: int) -> dict[str, Any]:
        args = getattr(request, "args", None)
        if args is None:
            if isinstance(request, (tuple, list)):
                args = request
            else:
                raise TypeError(f"Unsupported generate request type={type(request)}")

        if not isinstance(args, (tuple, list)):
            raise TypeError(f"request args must be tuple/list, got type={type(args)}")
        if len(args) < 1 or len(args) > 3:
            raise ValueError(f"generate request expects 1..3 args, got n={len(args)}")

        context = args[0]
        if not isinstance(context, str) or not context:
            raise ValueError("generate request context must be a non-empty str")

        raw_kwargs: dict[str, Any] = {}
        if len(args) == 2:
            second = args[1]
            if isinstance(second, dict):
                raw_kwargs = second or {}
            else:
                raw_kwargs = {"until": second}
        elif len(args) == 3:
            second = args[1]
            third = args[2] or {}
            if not isinstance(third, dict):
                raise TypeError(f"third generate arg must be dict, got type={type(third)}")
            raw_kwargs = dict(third)
            raw_kwargs.setdefault("until", second)
        if not isinstance(raw_kwargs, dict):
            raise TypeError(f"gen kwargs must be dict, got type={type(raw_kwargs)}")

        supported_keys = {
            "until",
            "stop",
            "max_gen_toks",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
            "repetition_penalty",
        }
        unknown_keys = sorted(set(raw_kwargs.keys()) - supported_keys)
        if unknown_keys:
            raise ValueError(f"Unsupported generation kwargs: {unknown_keys}")

        if "max_gen_toks" in raw_kwargs and "max_new_tokens" in raw_kwargs:
            raise ValueError("Specify only one of max_gen_toks or max_new_tokens")
        max_gen_toks = raw_kwargs.get("max_gen_toks", raw_kwargs.get("max_new_tokens", self.max_gen_toks))
        if not isinstance(max_gen_toks, int) or max_gen_toks <= 0:
            raise ValueError(f"max_gen_toks must be positive int, got={max_gen_toks}")

        stop_value = raw_kwargs.get("until", raw_kwargs.get("stop", None))
        until = _normalize_until(stop_value)

        temperature = float(raw_kwargs.get("temperature", 0.0))
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got={temperature}")

        top_p = float(raw_kwargs.get("top_p", 1.0))
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got={top_p}")

        top_k = int(raw_kwargs.get("top_k", 0))
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got={top_k}")

        do_sample = bool(raw_kwargs.get("do_sample", temperature > 0.0))
        repetition_penalty = float(raw_kwargs.get("repetition_penalty", 1.0))
        if repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got={repetition_penalty}")

        group_key = (
            until,
            max_gen_toks,
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
        )

        return {
            "index": index,
            "context": context,
            "until": until,
            "max_gen_toks": max_gen_toks,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "group_key": group_key,
        }

    def _build_generate_kwargs(self, item: dict[str, Any]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(item["max_gen_toks"]),
            "use_cache": True,
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "eos_token_id": int(self.tokenizer.eos_token_id),
            "do_sample": bool(item["do_sample"]),
            "repetition_penalty": float(item["repetition_penalty"]),
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(item["temperature"])
            kwargs["top_p"] = float(item["top_p"])
            if int(item["top_k"]) > 0:
                kwargs["top_k"] = int(item["top_k"])
        return kwargs

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        parsed_requests = [
            self._parse_generate_request(request=request, index=index)
            for index, request in enumerate(requests)
        ]
        outputs = [""] * len(parsed_requests)
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for item in parsed_requests:
            key = item["group_key"]
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)

        for group_items in grouped.values():
            for start in range(0, len(group_items), self.batch_size_per_forward):
                batch_items = group_items[start : start + self.batch_size_per_forward]
                contexts = [item["context"] for item in batch_items]
                encoded = self.tokenizer(
                    contexts,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                )

                input_ids = encoded["input_ids"].to(self._device)
                attention_mask = encoded["attention_mask"].to(self._device)
                prompt_lengths = attention_mask.sum(dim=-1)
                max_prompt_len = int(prompt_lengths.max().item())
                max_gen_toks = int(batch_items[0]["max_gen_toks"])
                if max_prompt_len + max_gen_toks > self.max_length:
                    raise ValueError(
                        "Prompt + generation exceeds max length: "
                        f"prompt_max={max_prompt_len} max_gen_toks={max_gen_toks} max_length={self.max_length}"
                    )

                generation_kwargs = self._build_generate_kwargs(batch_items[0])
                with torch.no_grad():
                    with _autocast_context(self._device):
                        generated = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **generation_kwargs,
                        )

                prompt_width = input_ids.size(1)
                new_tokens_batch = generated[:, prompt_width:]
                for row_idx, item in enumerate(batch_items):
                    text = self.tokenizer.decode(
                        new_tokens_batch[row_idx].tolist(),
                        skip_special_tokens=True,
                    )
                    text = _trim_by_until(text=text, until=item["until"])
                    outputs[item["index"]] = text

        assert all(isinstance(text, str) for text in outputs), "generate_until outputs must be str"
        return outputs

    def get_model_info(self) -> dict[str, Any]:
        return {
            "train_config_path": self.train_config_path,
            "checkpoint_path": self.checkpoint_path,
            "loaded_step": self.loaded_step,
            "use_ema": self.use_ema,
            "tokenizer_name": self.tokenizer_name,
        }
