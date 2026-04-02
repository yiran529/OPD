from __future__ import annotations

from memory_pollution.config import MemoryPollutionEvalConfig
from memory_pollution.metrics import compute_arc_metrics
from memory_pollution.perturb import apply_random_token_insertion
from memory_pollution.runtime import RuntimeBundle
from memory_pollution.scoring import score_continuation_from_ids, summarize_choice_scores
from memory_pollution.state import capture_prompt_cache, compute_state_drift
from memory_pollution.tasks.arc import (
    build_arc_prompt,
    build_choice_continuation,
    iter_arc_examples,
)


def run_arc_eval(
    cfg: MemoryPollutionEvalConfig,
    runtime: RuntimeBundle,
) -> tuple[list[dict], dict]:
    predictions: list[dict] = []
    tokenizer = runtime.tokenizer

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

        clean_choice_scores: list[dict] = []
        perturb_choice_scores: list[dict] = []
        for choice in example["choices"]:
            continuation_text = build_choice_continuation(choice["text"])
            continuation_ids = tokenizer(continuation_text, add_special_tokens=False)["input_ids"]
            assert continuation_ids, "choice continuation tokenization must be non-empty"

            clean_choice_scores.append(
                {
                    "label": choice["label"],
                    "score": score_continuation_from_ids(
                        model=runtime.model,
                        device=runtime.device,
                        prompt_token_ids=clean_prompt_ids,
                        continuation_token_ids=continuation_ids,
                        normalize_by_length=cfg.normalize_logprob_by_length,
                    ),
                }
            )
            perturb_choice_scores.append(
                {
                    "label": choice["label"],
                    "score": score_continuation_from_ids(
                        model=runtime.model,
                        device=runtime.device,
                        prompt_token_ids=perturbed_prompt_ids,
                        continuation_token_ids=continuation_ids,
                        normalize_by_length=cfg.normalize_logprob_by_length,
                    ),
                }
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
                "perturb": perturb_meta,
            }
        )

    metrics = compute_arc_metrics(predictions)
    metrics["dataset_name"] = cfg.dataset_name
    metrics["dataset_config"] = cfg.dataset_config
    metrics["dataset_split"] = cfg.dataset_split
    metrics["normalize_logprob_by_length"] = cfg.normalize_logprob_by_length
    metrics["perturb_kind"] = cfg.perturb_kind
    metrics["perturb_ratio"] = cfg.perturb_ratio
    metrics["perturb_seed"] = cfg.perturb_seed
    return predictions, metrics

