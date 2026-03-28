from __future__ import annotations


def compute_arc_metrics(predictions: list[dict]) -> dict:
    total = len(predictions)
    if total == 0:
        raise ValueError("No predictions were produced")

    correct = sum(1 for row in predictions if row["is_correct"])
    accuracy = correct / total

    return {
        "num_examples": total,
        "num_correct": correct,
        "accuracy": accuracy,
    }
