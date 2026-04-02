from __future__ import annotations


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def compute_arc_metrics(predictions: list[dict]) -> dict:
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

    return {
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

