from __future__ import annotations

from exposure_bias.config import ExposureBiasEvalConfig


def _mean(values: list[float]) -> float:
    assert values, "values must be non-empty"
    return sum(values) / len(values)


def compute_exposure_bias_metrics(
    predictions: list[dict],
    cfg: ExposureBiasEvalConfig,
) -> dict:
    if not predictions:
        raise ValueError("No predictions were produced")

    ce_tf = [float(row["ce_tf"]) for row in predictions]
    ce_rollout = [float(row["ce_rollout"]) for row in predictions]
    gaps = [float(row["exposure_bias_gap"]) for row in predictions]
    match_rates = [float(row["rollout_token_match_rate"]) for row in predictions]

    return {
        "num_examples": len(predictions),
        "mean_ce_tf": _mean(ce_tf),
        "mean_ce_rollout": _mean(ce_rollout),
        "mean_exposure_bias_gap": _mean(gaps),
        "mean_rollout_token_match_rate": _mean(match_rates),
        "dataset_name": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "dataset_split": cfg.dataset_split,
        "dataset_text_field": cfg.dataset_text_field,
        "prefix_len": cfg.prefix_len,
        "rollout_len": cfg.rollout_len,
        "batch_size": cfg.batch_size,
        "rollout_policy": cfg.rollout_policy,
    }
