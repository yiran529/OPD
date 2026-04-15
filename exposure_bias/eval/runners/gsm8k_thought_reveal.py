from __future__ import annotations

from exposure_bias.eval.config import ExposureBiasEvalConfig
from exposure_bias.eval.metrics import compute_gsm8k_thought_reveal_metrics
from exposure_bias.eval.runtime import RuntimeBundle
from exposure_bias.eval.scoring import generate_greedy_batch
from exposure_bias.eval.tasks.gsm8k import (
    build_gsm8k_reveal_prompt,
    extract_final_answer_from_completion,
    iter_gsm8k_examples,
    reveal_step_count,
)


def _assert_prompt_lengths_fit(
    prompts: list[str],
    runtime: RuntimeBundle,
    cfg: ExposureBiasEvalConfig,
) -> None:
    prompt_ids = runtime.tokenizer(
        prompts,
        padding=False,
        add_special_tokens=False,
    )["input_ids"]
    for prompt_idx, token_ids in enumerate(prompt_ids):
        prompt_len = len(token_ids)
        total_len = prompt_len + cfg.max_new_tokens
        assert total_len <= runtime.model_max_length, (
            f"gsm8k prompt exceeds model max length budget: "
            f"prompt_idx={prompt_idx} prompt_len={prompt_len} "
            f"max_new_tokens={cfg.max_new_tokens} total_len={total_len} "
            f"model_max_length={runtime.model_max_length}"
        )


def _iter_example_batches(
    cfg: ExposureBiasEvalConfig,
):
    batch: list[dict] = []
    for example in iter_gsm8k_examples(cfg=cfg):
        batch.append(example)
        if len(batch) == cfg.batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_gsm8k_thought_reveal_eval(
    cfg: ExposureBiasEvalConfig,
    runtime: RuntimeBundle,
) -> tuple[list[dict], dict]:
    predictions_by_id: dict[str, dict] = {}

    for example_batch in _iter_example_batches(cfg=cfg):
        for example in example_batch:
            predictions_by_id[example["id"]] = {
                "id": example["id"],
                "question": example["question"],
                "gold_final_answer": example["final_answer"],
                "gold_final_answer_normalized": example["normalized_final_answer"],
                "num_thought_steps": len(example["steps"]),
                "results_by_ratio": {},
            }

        for ratio in cfg.reveal_ratios:
            prompts: list[str] = []
            reveal_counts: list[int] = []
            for example in example_batch:
                reveal_count = reveal_step_count(num_steps=len(example["steps"]), ratio=ratio)
                revealed_steps = example["steps"][:reveal_count]
                prompts.append(
                    build_gsm8k_reveal_prompt(
                        question=example["question"],
                        revealed_steps=revealed_steps,
                    )
                )
                reveal_counts.append(reveal_count)

            _assert_prompt_lengths_fit(prompts=prompts, runtime=runtime, cfg=cfg)
            generation = generate_greedy_batch(
                model=runtime.model,
                tokenizer=runtime.tokenizer,
                device=runtime.device,
                prompts=prompts,
                max_new_tokens=cfg.max_new_tokens,
            )

            for batch_idx, example in enumerate(example_batch):
                completion_ids = generation["continuation_ids"][batch_idx]
                completion_text = runtime.tokenizer.decode(completion_ids, skip_special_tokens=False)
                pred_answer = extract_final_answer_from_completion(completion_text)
                is_correct = pred_answer == example["normalized_final_answer"]
                predictions_by_id[example["id"]]["results_by_ratio"][str(ratio)] = {
                    "ratio": ratio,
                    "reveal_count": reveal_counts[batch_idx],
                    "revealed_steps": example["steps"][: reveal_counts[batch_idx]],
                    "predicted_answer_normalized": pred_answer,
                    "is_correct": is_correct,
                    "generated_text": completion_text,
                }

    predictions = list(predictions_by_id.values())
    metrics = compute_gsm8k_thought_reveal_metrics(predictions=predictions, cfg=cfg)
    return predictions, metrics
