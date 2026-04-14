from __future__ import annotations

import math
import re
from typing import Iterable

from exposure_bias.eval.config import ExposureBiasEvalConfig
from exposure_bias.text_data import load_hf_text_dataset
from exposure_bias.train.tasks.gsm8k import split_gsm8k_answer


_STEP_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def split_rationale_into_steps(rationale_text: str) -> list[str]:
    raw_lines = [line.strip() for line in rationale_text.splitlines() if line.strip()]
    if len(raw_lines) >= 2:
        return raw_lines

    single_text = rationale_text.strip()
    assert single_text, "gsm8k rationale must be non-empty"
    sentence_steps = [part.strip() for part in _STEP_SPLIT_RE.split(single_text) if part.strip()]
    if sentence_steps:
        return sentence_steps
    return [single_text]


def reveal_step_count(
    num_steps: int,
    ratio: float,
) -> int:
    assert num_steps > 0, "num_steps must be positive"
    assert 0.0 <= ratio <= 1.0, "ratio must be in [0,1]"
    if ratio == 0.0:
        return 0
    reveal_count = int(math.floor(num_steps * ratio))
    if reveal_count == 0:
        reveal_count = 1
    return min(reveal_count, num_steps)


def build_gsm8k_reveal_prompt(
    question: str,
    revealed_steps: list[str],
) -> str:
    question = question.strip()
    assert question, "gsm8k question must be non-empty"

    prompt = f"Question: {question}\n\nThoughts:\n"
    if revealed_steps:
        prompt += "\n".join(step.strip() for step in revealed_steps if step.strip())
        prompt += "\n"
    return prompt


def normalize_gsm8k_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text.replace(",", "")
    matches = _NUMBER_RE.findall(text)
    if matches:
        return matches[-1]
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_final_answer_from_completion(completion_text: str) -> str:
    completion_text = completion_text.strip()
    if not completion_text:
        return ""
    if "Final Answer:" in completion_text:
        completion_text = completion_text.rsplit("Final Answer:", 1)[1]
    return normalize_gsm8k_answer(completion_text)


def iter_gsm8k_examples(
    cfg: ExposureBiasEvalConfig,
) -> Iterable[dict]:
    dataset = load_hf_text_dataset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        dataset_split=cfg.dataset_split,
        local_dataset_path=cfg.local_dataset_path,
        streaming=False,
    )
    for idx, row in enumerate(dataset):
        if cfg.max_samples > 0 and idx >= cfg.max_samples:
            break
        assert "question" in row, "gsm8k row missing question"
        assert "answer" in row, "gsm8k row missing answer"
        question = row["question"]
        answer_text = row["answer"]
        assert isinstance(question, str), "gsm8k question must be str"
        assert isinstance(answer_text, str), "gsm8k answer must be str"

        rationale_text, final_answer = split_gsm8k_answer(answer_text)
        steps = split_rationale_into_steps(rationale_text)

        yield {
            "id": f"gsm8k::{idx}",
            "question": question,
            "rationale_text": rationale_text,
            "final_answer": final_answer,
            "normalized_final_answer": normalize_gsm8k_answer(final_answer),
            "steps": steps,
        }
