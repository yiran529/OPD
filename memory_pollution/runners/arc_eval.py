from __future__ import annotations

from memory_pollution.config import MemoryPollutionEvalConfig
from memory_pollution.perturb import build_perturb_token_preview
from memory_pollution.metrics import compute_arc_metrics
from memory_pollution.perturb import apply_random_token_insertion
from memory_pollution.runtime import RuntimeBundle
from memory_pollution.scoring import score_continuation_batch_from_text, summarize_choice_scores
from memory_pollution.state import capture_prompt_cache, compute_state_drift
from memory_pollution.tasks.arc import (
    build_arc_prompt,
    build_choice_continuation,
    iter_arc_examples,
)


def _score_choice_batch(
    model,
    device,
    pad_token_id: int,
    prompt_text: str,
    choices: list[dict],
    tokenizer,
    normalize_by_length: bool,
    eval_batch_size: int,
) -> list[dict]:
    assert choices, "choices must be non-empty"
    choice_scores: list[dict] = []

    continuation_payloads: list[tuple[str, str]] = []
    for choice in choices:
        continuation_text = build_choice_continuation(choice["text"])
        continuation_payloads.append((choice["label"], continuation_text))

    for start in range(0, len(continuation_payloads), eval_batch_size):
        chunk = continuation_payloads[start : start + eval_batch_size]
        labels = [label for label, _ in chunk]
        batch_continuation_text = [continuation_text for _, continuation_text in chunk]
        batch_prompt_text = [prompt_text for _ in chunk]
        batch_scores = score_continuation_batch_from_text(
            model=model,
            tokenizer=tokenizer,
            device=device,
            pad_token_id=pad_token_id,
            batch_prompt_text=batch_prompt_text,
            batch_continuation_text=batch_continuation_text,
            normalize_by_length=normalize_by_length,
        )
        assert len(batch_scores) == len(labels), "batch score count mismatch"
        for label, score in zip(labels, batch_scores):
            choice_scores.append({"label": label, "score": score})

    return choice_scores


def run_arc_eval(
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

    for example in iter_arc_examples(cfg):
        prompt_text = build_arc_prompt(
            question=example["question"],
            choices=example["choices"],
        )
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

        clean_choice_scores = _score_choice_batch(
            model=runtime.model,
            device=runtime.device,
            pad_token_id=int(pad_token_id),
            prompt_text=prompt_text,
            choices=example["choices"],
            tokenizer=tokenizer,
            normalize_by_length=cfg.normalize_logprob_by_length,
            eval_batch_size=cfg.eval_batch_size,
        )
        perturb_choice_scores = _score_choice_batch(
            model=runtime.model,
            device=runtime.device,
            pad_token_id=int(pad_token_id),
            prompt_text=perturbed_prompt_text,
            choices=example["choices"],
            tokenizer=tokenizer,
            normalize_by_length=cfg.normalize_logprob_by_length,
            eval_batch_size=cfg.eval_batch_size,
        )

        clean_summary = summarize_choice_scores(
            choice_scores=clean_choice_scores,
            gold_label=example["answer_key"],
        )
        perturb_summary = summarize_choice_scores(
            choice_scores=perturb_choice_scores,
            gold_label=example["answer_key"],
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
                "question": example["question"],
                "gold_label": example["answer_key"],
                "clean_pred_label": clean_summary["pred_label"],
                "clean_is_correct": clean_summary["is_correct"],
                "clean_gold_score": clean_summary["gold_score"],
                "clean_best_non_gold_score": clean_summary["best_non_gold_score"],
                "clean_margin": clean_summary["margin"],
                "clean_choice_scores": clean_choice_scores,
                "perturb_pred_label": perturb_summary["pred_label"],
                "perturb_is_correct": perturb_summary["is_correct"],
                "perturb_gold_score": perturb_summary["gold_score"],
                "perturb_best_non_gold_score": perturb_summary["best_non_gold_score"],
                "perturb_margin": perturb_summary["margin"],
                "perturb_choice_scores": perturb_choice_scores,
                "margin_drop": clean_summary["margin"] - perturb_summary["margin"],
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

    metrics = compute_arc_metrics(predictions, cfg=cfg)
    return predictions, metrics
