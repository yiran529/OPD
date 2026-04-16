import random

import torch


def _build_problem_prompt_ids(tokenizer, problem):
    user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    messages = [{"role": "user", "content": user_message}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert isinstance(prompt_text, str) and prompt_text.strip(), "chat template returned an empty prompt"
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    assert isinstance(prompt_ids, list) and prompt_ids, "chat template text encoded to an empty prompt"
    return [int(token_id) for token_id in prompt_ids]


def _encode_solution_ids(tokenizer, solution):
    solution_ids = tokenizer(solution.strip(), add_special_tokens=False)["input_ids"]
    assert solution_ids, "solution encoded to zero tokens"
    return [int(token_id) for token_id in solution_ids]


def _sample_non_overlapping_starts(
    solution_length,
    span_len,
    rollout_len,
    num_spans,
    start_min_ratio,
    start_max_ratio,
):
    latest_start = solution_length - span_len - rollout_len
    assert latest_start >= 0, (
        "solution is too short for the requested corruption + rollout: "
        f"solution_length={solution_length} span_len={span_len} rollout_len={rollout_len}"
    )

    start_min = min(int(solution_length * start_min_ratio), latest_start)
    start_max = min(int(solution_length * start_max_ratio), latest_start)
    assert start_max >= start_min, (
        "No valid corruption start satisfies the configured ratios. "
        f"solution_length={solution_length} span_len={span_len} rollout_len={rollout_len} "
        f"start_min_ratio={start_min_ratio} start_max_ratio={start_max_ratio}"
    )

    candidate_starts = list(range(start_min, start_max + 1))
    random.shuffle(candidate_starts)

    selected = []
    for start in candidate_starts:
        overlaps = any(
            not (start + span_len <= existing or existing + span_len <= start) for existing in selected
        )
        if overlaps:
            continue
        selected.append(start)
        if len(selected) == num_spans:
            break

    assert len(selected) == num_spans, (
        "Failed to sample the requested number of non-overlapping corruption spans. "
        f"solution_length={solution_length} span_len={span_len} num_spans={num_spans}"
    )
    return sorted(selected)


def _build_linear_opsd_prefixes(
    solution_ids,
    rollout_len,
    num_spans,
    span_choices,
    start_min_ratio,
    start_max_ratio,
):
    solution_length = len(solution_ids)
    span_len = int(random.choice(span_choices))
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
        position for span_start in span_starts for position in range(span_start, span_start + span_len)
    }
    donor_positions = [idx for idx in range(solution_length) if idx not in corrupted_positions]
    assert donor_positions, "No donor positions remain outside the corrupted spans"

    corrupted_solution = list(solution_ids)
    clean_spans = []
    for span_start in span_starts:
        clean_tokens = list(solution_ids[span_start : span_start + span_len])
        corrupted_tokens = [int(solution_ids[random.choice(donor_positions)]) for _ in range(span_len)]
        corrupted_solution[span_start : span_start + span_len] = corrupted_tokens
        clean_spans.append((span_start, clean_tokens))

    rollout_start = max(span_start + span_len for span_start in span_starts)
    student_prefix_ids = corrupted_solution[:rollout_start]
    teacher_prefix_ids = list(student_prefix_ids)
    for span_start, clean_tokens in clean_spans:
        teacher_prefix_ids[span_start : span_start + span_len] = clean_tokens

    return {
        "student_prefix_ids": student_prefix_ids,
        "teacher_prefix_ids": teacher_prefix_ids,
        "rollout_start": rollout_start,
        "num_spans": num_spans,
        "span_len": span_len,
        "solution_length": solution_length,
    }


def _pad_sequences(sequences, pad_token_id):
    lengths = [len(ids) for ids in sequences]
    max_len = max(lengths)

    padded = []
    attention_masks = []
    for ids in sequences:
        pad_len = max_len - len(ids)
        padded_ids = ids + [pad_token_id] * pad_len
        attention_mask = [1] * len(ids) + [0] * pad_len
        padded.append(padded_ids)
        attention_masks.append(attention_mask)

    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "max_len": max_len,
    }


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    `opsd` keeps the original privileged teacher prompt.
    `linear_opsd` builds token-level corrupted and patched prefixes on top of the same
    problem prompt so the trainer can keep using the same rollout plumbing.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        conditioning_mode="opsd",
        rollout_len=128,
        num_corrupt_spans=1,
        corrupt_span_choices=None,
        corrupt_start_min_ratio=0.0,
        corrupt_start_max_ratio=0.5,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.conditioning_mode = conditioning_mode
        self.rollout_len = rollout_len
        self.num_corrupt_spans = num_corrupt_spans
        self.corrupt_span_choices = corrupt_span_choices or [2]
        self.corrupt_start_min_ratio = corrupt_start_min_ratio
        self.corrupt_start_max_ratio = corrupt_start_max_ratio

        assert self.conditioning_mode in {"opsd", "linear_opsd"}, (
            f"Unsupported conditioning_mode={self.conditioning_mode}"
        )
        if self.conditioning_mode == "linear_opsd":
            assert not self.reason_first, "reason_first is incompatible with conditioning_mode=linear_opsd"

        # Prompt for reasoning about the solution before teaching
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )

        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Conditioning mode: {self.conditioning_mode}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")

    def __call__(self, features):
        if self.conditioning_mode == "linear_opsd":
            return self._collate_linear_opsd(features)
        return self._collate_opsd(features)

    def _collate_linear_opsd(self, features):
        student_prompt_ids = []
        teacher_prompt_ids = []
        rollout_starts = []
        num_spans = []
        span_lens = []
        solution_lengths = []

        for feature in features:
            problem = feature["problem"]
            solution = feature["solution"]

            prompt_ids = _build_problem_prompt_ids(self.tokenizer, problem)
            solution_ids = _encode_solution_ids(self.tokenizer, solution)
            corruption = _build_linear_opsd_prefixes(
                solution_ids=solution_ids,
                rollout_len=self.rollout_len,
                num_spans=self.num_corrupt_spans,
                span_choices=self.corrupt_span_choices,
                start_min_ratio=self.corrupt_start_min_ratio,
                start_max_ratio=self.corrupt_start_max_ratio,
            )

            student_ids = prompt_ids + corruption["student_prefix_ids"]
            teacher_ids = prompt_ids + corruption["teacher_prefix_ids"]
            assert len(student_ids) <= self.max_length, (
                "linear_opsd student prompt exceeds max_length. "
                f"prompt_len={len(student_ids)} max_length={self.max_length}"
            )
            assert len(teacher_ids) <= self.max_length, (
                "linear_opsd teacher prompt exceeds max_length. "
                f"prompt_len={len(teacher_ids)} max_length={self.max_length}"
            )

            student_prompt_ids.append(student_ids)
            teacher_prompt_ids.append(teacher_ids)
            rollout_starts.append(corruption["rollout_start"])
            num_spans.append(corruption["num_spans"])
            span_lens.append(corruption["span_len"])
            solution_lengths.append(corruption["solution_length"])

        student_padded = _pad_sequences(student_prompt_ids, self.tokenizer.pad_token_id)
        teacher_padded = _pad_sequences(teacher_prompt_ids, self.tokenizer.pad_token_id)

        return {
            "student_prompts": student_padded["input_ids"],
            "student_prompt_attention_mask": student_padded["attention_mask"],
            "student_prompt_length": student_padded["max_len"],
            "student_prompt_lengths_per_example": student_padded["lengths"],
            "teacher_prompts": teacher_padded["input_ids"],
            "teacher_prompt_attention_mask": teacher_padded["attention_mask"],
            "teacher_prompt_length": teacher_padded["max_len"],
            "teacher_prompt_lengths_per_example": teacher_padded["lengths"],
            "rollout_start": torch.tensor(rollout_starts, dtype=torch.long),
            "num_spans": torch.tensor(num_spans, dtype=torch.long),
            "span_len": torch.tensor(span_lens, dtype=torch.long),
            "solution_length": torch.tensor(solution_lengths, dtype=torch.long),
        }

    def _collate_opsd(self, features):
        batch_size = len(features)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []

        for feature in features:
            problem = feature["problem"]
            solution = feature["solution"]

            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            student_messages = [{"role": "user", "content": student_user_message}]
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)
                teacher_prompts.append("")
            else:
                teacher_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                    f"{self.transition_prompt}\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{}}."
                )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]
                teacher_prompt = self.tokenizer.apply_chat_template(
                    teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
                teacher_prompts.append(teacher_prompt)

        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)

        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            transition_text = (
                f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            )
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
