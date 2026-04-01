from __future__ import annotations

import torch

from eval.config import EvalConfig
from eval.tasks.arc_ai2.dataset import iter_arc_examples
from eval.tasks.arc_ai2.metrics import compute_arc_metrics
from eval.tasks.arc_ai2.prompt import build_arc_prompt, build_choice_suffix
from eval.tasks.arc_ai2.scorer import score_choice_logprob


def run_arc_ai2_eval(
    cfg: EvalConfig,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> tuple[list[dict], dict]:
    predictions: list[dict] = []

    for example in iter_arc_examples(cfg):
        prompt = build_arc_prompt(
            question=example["question"],
            choices=example["choices"],
        )

        choice_scores: list[dict] = []
        for choice in example["choices"]:
            label = choice["label"]
            suffix = build_choice_suffix(choice_text=choice["text"])
            score = score_choice_logprob(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt_text=prompt,
                choice_suffix=suffix,
                normalize_by_length=cfg.normalize_logprob_by_length,
            )
            choice_scores.append({"label": label, "score": score})

        assert choice_scores, "choice_scores must be non-empty"
        best = max(choice_scores, key=lambda item: item["score"])
        pred_label = best["label"]
        gold_label = example["answer_key"]
        is_correct = pred_label == gold_label

        predictions.append(
            {
                "id": example["id"],
                "question": example["question"],
                "pred_label": pred_label,
                "gold_label": gold_label,
                "is_correct": is_correct,
                "choice_scores": choice_scores,
            }
        )

    metrics = compute_arc_metrics(predictions)
    metrics["dataset_name"] = cfg.dataset_name
    metrics["dataset_config"] = cfg.dataset_config
    metrics["dataset_split"] = cfg.dataset_split
    metrics["normalize_logprob_by_length"] = cfg.normalize_logprob_by_length
    return predictions, metrics
