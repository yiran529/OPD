from __future__ import annotations

import math

from memory_pollution.config import MemoryPollutionEvalConfig


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _append_common_eval_metadata(
    metrics: dict,
    cfg: MemoryPollutionEvalConfig,
) -> dict:
    metrics["dataset_name"] = cfg.dataset_name
    metrics["dataset_config"] = cfg.dataset_config
    metrics["dataset_split"] = cfg.dataset_split
    metrics["normalize_logprob_by_length"] = cfg.normalize_logprob_by_length
    metrics["eval_batch_size"] = cfg.eval_batch_size
    metrics["perturb_kind"] = cfg.perturb_kind
    metrics["perturb_ratio"] = cfg.perturb_ratio
    metrics["perturb_seed"] = cfg.perturb_seed
    return metrics


def compute_arc_metrics(
    predictions: list[dict],
    cfg: MemoryPollutionEvalConfig,
) -> dict:
    if not predictions:
        raise ValueError("No predictions were produced")

    total = len(predictions)
    clean_correct = sum(1 for row in predictions if row["clean_is_correct"])
    perturb_correct = sum(1 for row in predictions if row["perturb_is_correct"])

    clean_correct_subset = [row for row in predictions if row["clean_is_correct"]]
    clean_correct_subset_perturb = sum(1 for row in clean_correct_subset if row["perturb_is_correct"])

    margin_drops = [float(row["margin_drop"]) for row in predictions]
    state_drifts = [
        float(row["state_drift"])
        for row in predictions
        if row.get("state_drift", None) is not None
    ]

    clean_accuracy = clean_correct / total
    perturb_accuracy = perturb_correct / total
    subset_accuracy = None
    if clean_correct_subset:
        subset_accuracy = clean_correct_subset_perturb / len(clean_correct_subset)

    metrics = {
        "num_examples": total,
        "num_clean_correct": clean_correct,
        "num_perturb_correct": perturb_correct,
        "clean_accuracy": clean_accuracy,
        "perturb_accuracy": perturb_accuracy,
        "accuracy_drop": clean_accuracy - perturb_accuracy,
        "num_clean_correct_subset": len(clean_correct_subset),
        "clean_correct_subset_perturb_accuracy": subset_accuracy,
        "mean_margin_drop": _mean(margin_drops),
        "mean_state_drift": _mean(state_drifts),
        "num_state_drift_examples": len(state_drifts),
    }
    return _append_common_eval_metadata(metrics, cfg=cfg)


def compute_lambada_openai_metrics(
    predictions: list[dict],
    cfg: MemoryPollutionEvalConfig,
) -> dict:
    if not predictions:
        raise ValueError("No predictions were produced")

    total = len(predictions)
    clean_exact = sum(1 for row in predictions if row["clean_is_exact"])
    perturb_exact = sum(1 for row in predictions if row["perturb_is_exact"])

    clean_logprobs = [float(row["clean_logprob"]) for row in predictions]
    perturb_logprobs = [float(row["perturb_logprob"]) for row in predictions]
    logprob_drops = [float(row["logprob_drop"]) for row in predictions]
    state_drifts = [
        float(row["state_drift"])
        for row in predictions
        if row.get("state_drift", None) is not None
    ]

    clean_acc = clean_exact / total
    perturb_acc = perturb_exact / total
    metrics = {
        "num_examples": total,
        "num_clean_exact": clean_exact,
        "num_perturb_exact": perturb_exact,
        "clean_acc": clean_acc,
        "perturb_acc": perturb_acc,
        "acc_drop": clean_acc - perturb_acc,
        # LAMBADA evaluates a single held-out final word per example, so
        # the task perplexity reduces to exp(-mean total logprob).
        "clean_perplexity": math.exp(-sum(clean_logprobs) / total),
        "perturb_perplexity": math.exp(-sum(perturb_logprobs) / total),
        "mean_logprob_drop": _mean(logprob_drops),
        "mean_state_drift": _mean(state_drifts),
        "num_state_drift_examples": len(state_drifts),
    }
    return _append_common_eval_metadata(metrics, cfg=cfg)
