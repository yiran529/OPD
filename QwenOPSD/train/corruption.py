from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CorruptionResult:
    corrupted_prefix_ids: list[int]
    rollout_start: int
    span_len: int
    donor_positions: list[int]


def build_corrupted_prefix(
    solution_ids: Sequence[int],
    rollout_len: int,
    span_choices: Sequence[int],
    start_min_ratio: float,
    start_max_ratio: float,
) -> CorruptionResult:
    solution_length = len(solution_ids)
    assert solution_length > 0, "solution_ids must be non-empty"
    assert rollout_len > 0, "rollout_len must be positive"
    assert span_choices, "span_choices must be non-empty"

    # ---- sample span inside the early-to-middle region ----
    span_len = int(random.choice(list(span_choices)))
    assert span_len > 0, "sampled span_len must be positive"
    assert solution_length >= span_len + rollout_len, (
        "solution is too short for corruption + rollout: "
        f"solution_length={solution_length} span_len={span_len} rollout_len={rollout_len}"
    )

    valid_latest_start = solution_length - span_len - rollout_len
    desired_min_start = int(solution_length * start_min_ratio)
    desired_max_start = int(solution_length * start_max_ratio)
    start_min = min(desired_min_start, valid_latest_start)
    start_max = min(desired_max_start, valid_latest_start)
    if start_max < start_min:
        raise RuntimeError(
            "No valid corruption start satisfies the configured ratios. "
            f"solution_length={solution_length} span_len={span_len} rollout_len={rollout_len} "
            f"start_min_ratio={start_min_ratio} start_max_ratio={start_max_ratio}"
        )

    span_start = random.randint(start_min, start_max)
    span_end = span_start + span_len

    # ---- replace the span with tokens drawn from other positions in the same sample ----
    donor_positions = [idx for idx in range(solution_length) if idx < span_start or idx >= span_end]
    if len(donor_positions) < span_len:
        raise RuntimeError(
            "Not enough donor positions outside the corrupted span. "
            f"solution_length={solution_length} span_len={span_len}"
        )

    donor_sample = random.sample(donor_positions, k=span_len)
    corrupted_solution = list(solution_ids)
    for offset, donor_idx in enumerate(donor_sample):
        corrupted_solution[span_start + offset] = int(solution_ids[donor_idx])

    return CorruptionResult(
        corrupted_prefix_ids=corrupted_solution[:span_end],
        rollout_start=span_end,
        span_len=span_len,
        donor_positions=donor_sample,
    )
