from __future__ import annotations

from exposure_bias.eval.config import ExposureBiasEvalConfig


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


def compute_gsm8k_thought_reveal_metrics(
    predictions: list[dict],
    cfg: ExposureBiasEvalConfig,
) -> dict:
    if not predictions:
        raise ValueError("No predictions were produced")

    acc_by_ratio: dict[str, float] = {}
    for ratio in cfg.reveal_ratios:
        ratio_key = str(ratio)
        values = [1.0 if row["results_by_ratio"][ratio_key]["is_correct"] else 0.0 for row in predictions]
        acc_by_ratio[ratio_key] = _mean(values)

    base_acc = acc_by_ratio[str(0.0)]
    gap_by_ratio = {
        str(ratio): acc_by_ratio[str(ratio)] - base_acc
        for ratio in cfg.reveal_ratios
        if ratio != 0.0
    }

    named_gaps = {}
    for ratio_key, gap_value in gap_by_ratio.items():
        ratio_suffix = ratio_key.replace("0.", "")
        named_gaps[f"Gap_{ratio_suffix}"] = gap_value

    return {
        "num_examples": len(predictions),
        "acc_by_ratio": acc_by_ratio,
        "gap_by_ratio": gap_by_ratio,
        **named_gaps,
        "dataset_name": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "dataset_split": cfg.dataset_split,
        "reveal_ratios": cfg.reveal_ratios,
        "batch_size": cfg.batch_size,
        "max_new_tokens": cfg.max_new_tokens,
        "rollout_policy": cfg.rollout_policy,
    }
