import random

import torch


def _set_use_cache_attr(obj, value):
    had_attr = hasattr(obj, "use_cache")
    original_value = getattr(obj, "use_cache", None)
    if had_attr:
        setattr(obj, "use_cache", value)
    return had_attr, original_value


def _restore_use_cache_attr(obj, had_attr, original_value):
    if had_attr:
        setattr(obj, "use_cache", original_value)


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


def _decode_ids(tokenizer, token_ids):
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def sample_gold_prefix_length(solution_length, gold_prefix_ratio_min, gold_prefix_ratio_max):
    assert solution_length > 0, "solution_length must be positive"
    assert 0.0 <= gold_prefix_ratio_min <= gold_prefix_ratio_max <= 1.0, (
        "gold_prefix ratios must satisfy 0 <= min <= max <= 1"
    )

    max_prefix_length = solution_length - 1
    min_prefix_length = min(max_prefix_length, max(0, int(solution_length * gold_prefix_ratio_min)))
    max_prefix_length = min(max_prefix_length, max(0, int(solution_length * gold_prefix_ratio_max)))
    assert max_prefix_length >= min_prefix_length, (
        "No valid gold prefix length satisfies the configured ratios. "
        f"solution_length={solution_length} "
        f"gold_prefix_ratio_min={gold_prefix_ratio_min} "
        f"gold_prefix_ratio_max={gold_prefix_ratio_max}"
    )
    return random.randint(min_prefix_length, max_prefix_length)


def build_teacher_trace_prefix_ids(gold_prefix_ids, careless_token_ids):
    return list(gold_prefix_ids) + list(careless_token_ids)


def build_teacher_trace_prefix_text(tokenizer, teacher_trace_prefix_ids):
    return _decode_ids(tokenizer, teacher_trace_prefix_ids)


def build_teacher_user_message(
    problem,
    solution,
):
    return (
        f"Problem:\n{problem}\n\n"
        "A mathematically correct solution is shown below.\n"
        f"=== Solution Begin ===\n{solution}\n=== Solution End ===\n\n"
        "Continue the assistant answer directly from the exact partial answer that follows. "
        "Use the solution above only to keep the next steps mathematically correct and natural. "
        "Write only the continuation."
    )


def _generate_careless_tokens(
    model,
    prompt_ids,
    tokenizer,
    careless_rollout_len,
    careless_temperature,
    careless_top_p,
    careless_top_k,
    device,
):
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(prompt_tensor)

    had_use_cache, original_use_cache = _set_use_cache_attr(model.config, True)
    try:
        generated = model.generate(
            input_ids=prompt_tensor,
            attention_mask=attention_mask,
            max_new_tokens=careless_rollout_len,
            min_new_tokens=careless_rollout_len,
            do_sample=True,
            temperature=careless_temperature,
            top_p=careless_top_p,
            top_k=careless_top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=getattr(model.generation_config, "eos_token_id", None),
            return_dict_in_generate=True,
            use_cache=True,
        )
    finally:
        _restore_use_cache_attr(model.config, had_use_cache, original_use_cache)

    generated_ids = generated.sequences[0, len(prompt_ids) :].tolist()
    assert generated_ids, "careless rollout generated zero tokens"
    return [int(token_id) for token_id in generated_ids]


def build_online_careless_prefix(
    model,
    tokenizer,
    problem,
    solution,
    problem_prompt_ids,
    solution_ids,
    gold_prefix_ratio_min,
    gold_prefix_ratio_max,
    clean_ratio,
    mild_ratio,
    mild_careless_rollout_len,
    careless_rollout_len,
    careless_temperature,
    careless_top_p,
    careless_top_k,
    careless_resample_trials,
    careless_marker_text,
    recovery_marker_text,
    device,
):
    solution_length = len(solution_ids)
    assert solution_length > 0, "solution_ids must be non-empty"
    assert 0.0 <= clean_ratio <= 1.0, "clean_ratio must be in [0, 1]"
    assert 0.0 <= mild_ratio <= 1.0, "mild_ratio must be in [0, 1]"
    assert clean_ratio + mild_ratio <= 1.0, "clean_ratio + mild_ratio must be <= 1"
    assert mild_careless_rollout_len > 0, "mild_careless_rollout_len must be positive"
    assert careless_rollout_len > 0, "careless_rollout_len must be positive"
    assert careless_resample_trials >= 0, "careless_resample_trials must be non-negative"

    gold_prefix_length = sample_gold_prefix_length(
        solution_length=solution_length,
        gold_prefix_ratio_min=gold_prefix_ratio_min,
        gold_prefix_ratio_max=gold_prefix_ratio_max,
    )
    gold_prefix_ids = list(solution_ids[:gold_prefix_length])
    prompt_ids = list(problem_prompt_ids) + gold_prefix_ids

    mode_sample = random.random()
    if mode_sample < clean_ratio:
        mixture_mode = "clean"
        active_careless_rollout_len = 0
    elif mode_sample < clean_ratio + mild_ratio:
        mixture_mode = "mild"
        active_careless_rollout_len = mild_careless_rollout_len
    else:
        mixture_mode = "hard"
        active_careless_rollout_len = careless_rollout_len

    gold_target_ids = list(solution_ids[gold_prefix_length : gold_prefix_length + active_careless_rollout_len])
    gold_recovery_target_ids = list(
        solution_ids[
            gold_prefix_length
            + active_careless_rollout_len : gold_prefix_length
            + active_careless_rollout_len
            + 256
        ]
    )
    if active_careless_rollout_len > 0:
        assert gold_target_ids, "gold prefix consumed the full solution; expected at least one target token"

    careless_token_ids = []
    careless_deviated = False
    resample_count = 0

    if active_careless_rollout_len > 0:
        for attempt_idx in range(careless_resample_trials + 1):
            candidate_ids = _generate_careless_tokens(
                model=model,
                prompt_ids=prompt_ids,
                tokenizer=tokenizer,
                careless_rollout_len=active_careless_rollout_len,
                careless_temperature=careless_temperature,
                careless_top_p=careless_top_p,
                careless_top_k=careless_top_k,
                device=device,
            )
            compare_len = min(len(candidate_ids), len(gold_target_ids))
            exact_prefix_match = compare_len > 0 and candidate_ids[:compare_len] == gold_target_ids[:compare_len]
            candidate_deviated = (not exact_prefix_match) or (len(candidate_ids) > len(gold_target_ids))

            careless_token_ids = candidate_ids
            careless_deviated = candidate_deviated
            if candidate_deviated:
                resample_count = attempt_idx
                break
            resample_count = attempt_idx + 1

    has_sampled_tail = active_careless_rollout_len > 0
    skip_kd = has_sampled_tail and not careless_deviated

    current_trace_ids = build_teacher_trace_prefix_ids(
        gold_prefix_ids=gold_prefix_ids,
        careless_token_ids=careless_token_ids,
    )
    current_trace = build_teacher_trace_prefix_text(
        tokenizer=tokenizer,
        teacher_trace_prefix_ids=current_trace_ids,
    )

    return {
        "student_prompt_ids": list(problem_prompt_ids) + current_trace_ids,
        "teacher_user_message": build_teacher_user_message(
            problem=problem,
            solution=solution,
        ),
        "teacher_trace_prefix_ids": current_trace_ids,
        "teacher_trace_prefix_text": current_trace,
        "current_trace_text": current_trace,
        "gold_prefix_ids": gold_prefix_ids,
        "careless_token_ids": careless_token_ids,
        "gold_target_ids": gold_target_ids,
        "gold_recovery_target_ids": gold_recovery_target_ids,
        "gold_prefix_length": gold_prefix_length,
        "active_careless_rollout_len": active_careless_rollout_len,
        "mixture_mode": mixture_mode,
        "careless_deviated": careless_deviated,
        "careless_resample_count": resample_count,
        "skip_kd": skip_kd,
    }
