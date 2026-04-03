from __future__ import annotations

from memory_pollution.config import MemoryPollutionEvalConfig
from memory_pollution.metrics import compute_lambada_openai_metrics
from memory_pollution.perturb import apply_random_token_insertion, build_perturb_token_preview
from memory_pollution.runtime import RuntimeBundle
from memory_pollution.scoring import score_continuation_details_from_text
from memory_pollution.state import capture_prompt_cache, compute_state_drift
from memory_pollution.tasks.lambada_openai import iter_lambada_openai_examples


def run_lambada_openai_eval(
    cfg: MemoryPollutionEvalConfig,
    runtime: RuntimeBundle,
) -> tuple[list[dict], dict]:
    predictions: list[dict] = []
    tokenizer = runtime.tokenizer
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        assert eos_token_id is not None, "tokenizer must define pad_token_id or eos_token_id"
        pad_token_id = int(eos_token_id)

    for example in iter_lambada_openai_examples(cfg):
        prompt_text = example["context_text"]
        target_text = example["target_text"]

        clean_prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        assert clean_prompt_ids, "clean prompt tokenization must be non-empty"

        perturbed_prompt_ids, perturb_meta = apply_random_token_insertion(
            tokenizer=tokenizer,
            prompt_token_ids=clean_prompt_ids,
            perturb_ratio=cfg.perturb_ratio,
            perturb_seed=cfg.perturb_seed,
            example_id=example["id"],
            perturb_min_tokens=cfg.perturb_min_tokens,
        )
        clean_prompt_text = tokenizer.decode(clean_prompt_ids, skip_special_tokens=False)
        perturbed_prompt_text = tokenizer.decode(perturbed_prompt_ids, skip_special_tokens=False)
        perturb_token_preview = build_perturb_token_preview(
            tokenizer=tokenizer,
            clean_prompt_token_ids=clean_prompt_ids,
            perturb_meta=perturb_meta,
        )

        clean_score = score_continuation_details_from_text(
            model=runtime.model,
            tokenizer=tokenizer,
            device=runtime.device,
            pad_token_id=int(pad_token_id),
            prompt_text=prompt_text,
            continuation_text=target_text,
            normalize_by_length=cfg.normalize_logprob_by_length,
        )
        perturb_score = score_continuation_details_from_text(
            model=runtime.model,
            tokenizer=tokenizer,
            device=runtime.device,
            pad_token_id=int(pad_token_id),
            prompt_text=perturbed_prompt_text,
            continuation_text=target_text,
            normalize_by_length=cfg.normalize_logprob_by_length,
        )

        state_drift = None
        per_layer_state_drift = None
        if runtime.supports_state_drift:
            clean_cache = capture_prompt_cache(
                model=runtime.model,
                device=runtime.device,
                prompt_token_ids=clean_prompt_ids,
            )
            perturbed_cache = capture_prompt_cache(
                model=runtime.model,
                device=runtime.device,
                prompt_token_ids=perturbed_prompt_ids,
            )
            drift = compute_state_drift(
                clean_cache=clean_cache,
                perturbed_cache=perturbed_cache,
                state_key=runtime.state_key,
            )
            state_drift = drift["overall_drift"]
            per_layer_state_drift = drift["per_layer_drift"]

        predictions.append(
            {
                "id": example["id"],
                "text": example["text"],
                "context_text": example["context_text"],
                "target_text": target_text,
                "clean_logprob": clean_score["logprob"],
                "clean_normalized_logprob": clean_score["normalized_logprob"],
                "clean_is_exact": clean_score["is_greedy_exact"],
                "clean_num_target_tokens": clean_score["num_tokens"],
                "perturb_logprob": perturb_score["logprob"],
                "perturb_normalized_logprob": perturb_score["normalized_logprob"],
                "perturb_is_exact": perturb_score["is_greedy_exact"],
                "perturb_num_target_tokens": perturb_score["num_tokens"],
                "logprob_drop": clean_score["logprob"] - perturb_score["logprob"],
                "normalized_logprob_drop": (
                    clean_score["normalized_logprob"] - perturb_score["normalized_logprob"]
                ),
                "state_drift": state_drift,
                "per_layer_state_drift": per_layer_state_drift,
                "clean_prompt_token_len": len(clean_prompt_ids),
                "perturbed_prompt_token_len": len(perturbed_prompt_ids),
                "clean_prompt_text": clean_prompt_text,
                "perturbed_prompt_text": perturbed_prompt_text,
                "perturb_token_preview": perturb_token_preview,
                "perturb": perturb_meta,
            }
        )

    metrics = compute_lambada_openai_metrics(predictions, cfg=cfg)
    return predictions, metrics
