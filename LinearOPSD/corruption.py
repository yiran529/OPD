import random
import string
import warnings

import torch


STYLE_TOKEN_BLACKLIST = {
    "let",
    "so",
    "thus",
    "therefore",
    "hence",
    "now",
    "then",
}


def pad_token_sequences(sequences, pad_token_id):
    lengths = [len(ids) for ids in sequences]
    max_len = max(lengths) if lengths else 0

    padded = []
    attention_masks = []
    for ids in sequences:
        pad_len = max_len - len(ids)
        padded.append(ids + [pad_token_id] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "max_len": max_len,
    }


def sample_rollout_start_offset(base_offset, offset_jitter):
    assert base_offset >= 0, "rollout_start_offset must be non-negative"
    assert offset_jitter >= 0, "rollout_start_offset_jitter must be non-negative"

    min_delta = -min(base_offset, offset_jitter)
    max_delta = offset_jitter
    sampled_delta = random.randint(min_delta, max_delta)
    sampled_offset = base_offset + sampled_delta
    assert sampled_offset >= 0, "sampled rollout_start_offset must be non-negative"
    return sampled_offset, sampled_delta


def _decode_single_token(tokenizer, token_id):
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def is_style_token(tokenizer, token_id):
    if token_id in set(tokenizer.all_special_ids):
        return True

    token_text = _decode_single_token(tokenizer, token_id)
    stripped = token_text.strip()
    if not stripped:
        return True

    if all(ch in string.punctuation for ch in stripped):
        return True

    if stripped.lower() in STYLE_TOKEN_BLACKLIST:
        return True

    return False


def build_teacher_user_message(problem, solution, student_trace_text):
    return (
        f"Problem: {problem}\n\n"
        f"Here is a reference solution to this problem:\n"
        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        "\n\nAfter reading the reference solution above, make sure you truly understand "
        "the reasoning behind each step — do not copy or paraphrase it. Use it to identify "
        "and correct the corrupted parts in the student's reasoning trace below while "
        "continuing the same solution trajectory.\n\n"
        f"Here is the student's current reasoning trace:\n{student_trace_text}"
    )


def select_corruption_positions(
    entropies,
    solution_ids,
    tokenizer,
    num_corrupt_points,
    start_min_ratio,
    start_max_ratio,
):
    solution_length = len(solution_ids)
    if solution_length == 0:
        warnings.warn("solution_length is zero; falling back to clean prefix")
        return []

    start_min = max(0, min(solution_length - 1, int(solution_length * start_min_ratio)))
    start_max = max(0, min(solution_length - 1, int(solution_length * start_max_ratio)))
    assert start_max >= start_min, (
        "No valid corruption start satisfies the configured ratios. "
        f"solution_length={solution_length} start_min_ratio={start_min_ratio} start_max_ratio={start_max_ratio}"
    )

    candidates = []
    for pos in range(start_min, start_max + 1):
        token_id = int(solution_ids[pos])
        if is_style_token(tokenizer, token_id):
            continue
        candidates.append((float(entropies[pos].item()), pos))

    if not candidates:
        warnings.warn("No non-style corruption candidates found; falling back to clean prefix")
        return []

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [pos for _, pos in candidates[:num_corrupt_points]]
    if len(selected) < num_corrupt_points:
        warnings.warn(
            f"Requested {num_corrupt_points} corruption points but found only {len(selected)} valid candidates"
        )
    return sorted(selected)


def select_wrong_replacements(solution_logits, solution_ids, tokenizer, positions, top_k=64):
    replacements = {}
    vocab_size = int(solution_logits.shape[-1])
    top_k = min(top_k, vocab_size)
    special_ids = set(tokenizer.all_special_ids)

    for pos in positions:
        gold_id = int(solution_ids[pos])
        token_logits = solution_logits[pos]
        _, top_indices = torch.topk(token_logits, k=top_k, dim=-1)

        replacement_id = None
        for candidate_id in top_indices.tolist():
            if candidate_id == gold_id:
                continue
            if candidate_id in special_ids:
                continue
            replacement_id = int(candidate_id)
            break

        if replacement_id is None:
            warnings.warn(f"No suitable replacement token found at position {pos}; skipping corruption point")
            continue

        replacements[pos] = replacement_id

    return replacements


def build_teacher_visible_trace_text(tokenizer, corrupted_prefix_ids, corruption_positions, corrupt_marker_text):
    if not corruption_positions:
        return tokenizer.decode(corrupted_prefix_ids, skip_special_tokens=False)

    parts = []
    cursor = 0
    for pos in corruption_positions:
        if cursor < pos:
            parts.append(tokenizer.decode(corrupted_prefix_ids[cursor:pos], skip_special_tokens=False))
        parts.append(f" {corrupt_marker_text}")
        parts.append(tokenizer.decode([corrupted_prefix_ids[pos]], skip_special_tokens=False))
        cursor = pos + 1

    if cursor < len(corrupted_prefix_ids):
        parts.append(tokenizer.decode(corrupted_prefix_ids[cursor:], skip_special_tokens=False))

    return "".join(parts)


def build_online_corruption(
    tokenizer,
    problem,
    solution,
    problem_prompt_ids,
    solution_ids,
    solution_logits,
    num_corrupt_points,
    rollout_start_offset,
    rollout_start_offset_jitter,
    corrupt_start_min_ratio,
    corrupt_start_max_ratio,
    corrupt_marker_text,
):
    entropies = torch.distributions.Categorical(logits=solution_logits).entropy()
    sampled_offset, sampled_offset_delta = sample_rollout_start_offset(
        rollout_start_offset,
        rollout_start_offset_jitter,
    )

    candidate_positions = select_corruption_positions(
        entropies=entropies,
        solution_ids=solution_ids,
        tokenizer=tokenizer,
        num_corrupt_points=num_corrupt_points,
        start_min_ratio=corrupt_start_min_ratio,
        start_max_ratio=corrupt_start_max_ratio,
    )
    replacements = select_wrong_replacements(
        solution_logits=solution_logits,
        solution_ids=solution_ids,
        tokenizer=tokenizer,
        positions=candidate_positions,
    )
    corruption_positions = sorted(replacements.keys())

    corrupted_solution_ids = list(solution_ids)
    for pos, replacement_id in replacements.items():
        corrupted_solution_ids[pos] = replacement_id

    solution_length = len(solution_ids)
    if corruption_positions:
        rollout_start = min(solution_length, corruption_positions[-1] + 1 + sampled_offset)
    else:
        fallback_position = min(solution_length - 1, max(0, int(solution_length * corrupt_start_max_ratio)))
        rollout_start = min(solution_length, fallback_position + 1 + sampled_offset)

    corrupted_prefix_ids = corrupted_solution_ids[:rollout_start]
    teacher_trace_text = build_teacher_visible_trace_text(
        tokenizer=tokenizer,
        corrupted_prefix_ids=corrupted_prefix_ids,
        corruption_positions=[pos for pos in corruption_positions if pos < rollout_start],
        corrupt_marker_text=corrupt_marker_text,
    )

    return {
        "student_prompt_ids": list(problem_prompt_ids) + corrupted_prefix_ids,
        "teacher_user_message": build_teacher_user_message(
            problem=problem,
            solution=solution,
            student_trace_text=teacher_trace_text,
        ),
        "teacher_trace_text": teacher_trace_text,
        "corrupted_prefix_ids": corrupted_prefix_ids,
        "corruption_positions": corruption_positions,
        "replacement_token_ids": [replacements[pos] for pos in corruption_positions],
        "rollout_start": rollout_start,
        "rollout_start_offset": sampled_offset,
        "rollout_start_offset_delta": sampled_offset_delta,
        "entropies": entropies.detach().cpu(),
    }
