from __future__ import annotations


def build_arc_prompt(question: str, choices: list[dict]) -> str:
    assert question, "question must be non-empty"
    assert choices, "choices must be non-empty"

    lines = [f"Question: {question}", "Choices:"]
    for choice in choices:
        label = choice["label"]
        text = choice["text"]
        lines.append(f"{label}. {text}")
    lines.append("Answer:")
    return "\n".join(lines)


def build_choice_suffix(choice_text: str) -> str:
    assert choice_text, "choice_text must be non-empty"
    return f" {choice_text}"
