from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CorruptedSpan:
    start: int
    length: int
    clean_tokens: list[int]
    corrupted_tokens: list[int]
    donor_positions: list[int]


@dataclass
class CorruptionResult:
    student_prefix_ids: list[int]
    teacher_prefix_ids: list[int]
    rollout_start: int
    span_len: int
    spans: list[CorruptedSpan]


def _sample_non_overlapping_starts(
    solution_length: int,
    span_len: int,
    rollout_len: int,
    num_spans: int,
    start_min_ratio: float,
    start_max_ratio: float,
) -> list[int]:
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

    candidate_starts = list(range(start_min, start_max + 1))
    random.shuffle(candidate_starts)

    selected_starts: list[int] = []
    for start in candidate_starts:
        overlaps_existing = any(
            not (start + span_len <= existing_start or existing_start + span_len <= start)
            for existing_start in selected_starts
        )
        if overlaps_existing:
            continue
        selected_starts.append(start)
        if len(selected_starts) == num_spans:
            break

    if len(selected_starts) != num_spans:
        raise RuntimeError(
            "Failed to sample the requested number of non-overlapping corruption spans. "
            f"solution_length={solution_length} span_len={span_len} rollout_len={rollout_len} "
            f"num_spans={num_spans}"
        )

    return sorted(selected_starts)


def _patch_clean_spans(
    corrupted_prefix_ids: list[int],
    spans: Sequence[CorruptedSpan],
) -> list[int]:
    teacher_prefix_ids = list(corrupted_prefix_ids)
    for span in spans:
        span_end = span.start + span.length
        if span_end > len(teacher_prefix_ids):
            raise RuntimeError(
                "Teacher patch span exceeds prefix length. "
                f"span_end={span_end} prefix_len={len(teacher_prefix_ids)}"
            )
        teacher_prefix_ids[span.start:span_end] = span.clean_tokens
    return teacher_prefix_ids


def build_corrupted_prefix(
    solution_ids: Sequence[int],
    rollout_len: int,
    num_spans: int,
    span_choices: Sequence[int],
    start_min_ratio: float,
    start_max_ratio: float,
) -> CorruptionResult:
    solution_length = len(solution_ids)
    assert solution_length > 0, "solution_ids must be non-empty"
    assert rollout_len > 0, "rollout_len must be positive"
    assert num_spans > 0, "num_spans must be positive"
    assert span_choices, "span_choices must be non-empty"

    # ---- sample B non-overlapping spans with a shared span length m ----
    span_len = int(random.choice(list(span_choices)))
    assert span_len > 0, "sampled span_len must be positive"
    assert solution_length >= num_spans * span_len + rollout_len, (
        "solution is too short for corruption + rollout: "
        f"solution_length={solution_length} span_len={span_len} "
        f"num_spans={num_spans} rollout_len={rollout_len}"
    )

    span_starts = _sample_non_overlapping_starts(
        solution_length=solution_length,
        span_len=span_len,
        rollout_len=rollout_len,
        num_spans=num_spans,
        start_min_ratio=start_min_ratio,
        start_max_ratio=start_max_ratio,
    )

    corrupted_positions = {
        position
        for span_start in span_starts
        for position in range(span_start, span_start + span_len)
    }
    donor_positions = [idx for idx in range(solution_length) if idx not in corrupted_positions]
    if not donor_positions:
        raise RuntimeError(
            "No donor positions remain outside the corrupted spans. "
            f"solution_length={solution_length} span_len={span_len} num_spans={num_spans}"
        )

    # ---- replace each corrupted token independently using random donor tokens ----
    corrupted_solution = list(solution_ids)
    spans: list[CorruptedSpan] = []
    for span_start in span_starts:
        clean_tokens = [int(token) for token in solution_ids[span_start : span_start + span_len]]
        sampled_donor_positions = [int(random.choice(donor_positions)) for _ in range(span_len)]
        corrupted_tokens = [int(solution_ids[donor_idx]) for donor_idx in sampled_donor_positions]
        corrupted_solution[span_start : span_start + span_len] = corrupted_tokens
        spans.append(
            CorruptedSpan(
                start=span_start,
                length=span_len,
                clean_tokens=clean_tokens,
                corrupted_tokens=corrupted_tokens,
                donor_positions=sampled_donor_positions,
            )
        )

    rollout_start = max(span.start + span.length for span in spans)
    student_prefix_ids = corrupted_solution[:rollout_start]
    teacher_prefix_ids = _patch_clean_spans(
        corrupted_prefix_ids=student_prefix_ids,
        spans=spans,
    )

    return CorruptionResult(
        student_prefix_ids=student_prefix_ids,
        teacher_prefix_ids=teacher_prefix_ids,
        rollout_start=rollout_start,
        span_len=span_len,
        spans=spans,
    )
