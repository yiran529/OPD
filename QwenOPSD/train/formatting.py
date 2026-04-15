from __future__ import annotations


def build_prompt_token_ids(
    tokenizer,
    problem_text: str,
    enable_thinking: bool,
) -> list[int]:
    problem_text = problem_text.strip()
    assert problem_text, "problem_text must be non-empty"

    messages = [{"role": "user", "content": problem_text}]
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        raise RuntimeError(
            "tokenizer.apply_chat_template failed. "
            "This likely means the installed transformers/Qwen template path is too old."
        ) from exc

    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise RuntimeError("apply_chat_template returned an empty or unsupported prompt encoding")
    return [int(token_id) for token_id in prompt_ids]


def encode_solution_token_ids(
    tokenizer,
    solution_text: str,
) -> list[int]:
    solution_text = solution_text.strip()
    assert solution_text, "solution_text must be non-empty"
    token_ids = tokenizer(solution_text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise RuntimeError("solution_text encoded to zero tokens")
    return [int(token_id) for token_id in token_ids]

