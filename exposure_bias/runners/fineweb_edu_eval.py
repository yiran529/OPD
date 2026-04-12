from __future__ import annotations

import torch

from exposure_bias.config import ExposureBiasEvalConfig
from exposure_bias.metrics import compute_exposure_bias_metrics
from exposure_bias.runtime import RuntimeBundle
from exposure_bias.scoring import compute_rollout_ce_batch, compute_teacher_forcing_ce_batch
from exposure_bias.tasks.fineweb_edu import iter_fineweb_edu_examples


def _iter_example_batches(
    cfg: ExposureBiasEvalConfig,
    tokenizer,
):
    batch: list[dict] = []
    for example in iter_fineweb_edu_examples(cfg=cfg, tokenizer=tokenizer):
        batch.append(example)
        if len(batch) == cfg.batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_fineweb_edu_eval(
    cfg: ExposureBiasEvalConfig,
    runtime: RuntimeBundle,
) -> tuple[list[dict], dict]:
    seq_len = cfg.prefix_len + cfg.rollout_len
    assert seq_len <= runtime.model_max_length, (
        f"requested sequence length exceeds model max length: "
        f"seq_len={seq_len} model_max_length={runtime.model_max_length}"
    )

    predictions: list[dict] = []
    tokenizer = runtime.tokenizer

    for example_batch in _iter_example_batches(cfg=cfg, tokenizer=tokenizer):
        batch_ids = torch.tensor(
            [example["token_ids"] for example in example_batch],
            dtype=torch.long,
            device=runtime.device,
        )
        assert batch_ids.shape[1] == seq_len, "batch token length mismatch"

        prefix_ids = batch_ids[:, : cfg.prefix_len]
        target_ids = batch_ids[:, cfg.prefix_len :]

        tf_scores = compute_teacher_forcing_ce_batch(
            model=runtime.model,
            device=runtime.device,
            input_ids=batch_ids,
            prefix_len=cfg.prefix_len,
            rollout_len=cfg.rollout_len,
        )
        rollout_scores = compute_rollout_ce_batch(
            model=runtime.model,
            device=runtime.device,
            prefix_ids=prefix_ids,
            target_ids=target_ids,
            rollout_policy=cfg.rollout_policy,
        )

        # ---- materialize per-sample outputs ----
        for batch_idx, example in enumerate(example_batch):
            generated_ids = rollout_scores["generated_ids"][batch_idx].tolist()
            target_list = target_ids[batch_idx].tolist()
            prefix_list = prefix_ids[batch_idx].tolist()
            predictions.append(
                {
                    "id": example["id"],
                    "prefix_len": cfg.prefix_len,
                    "rollout_len": cfg.rollout_len,
                    "ce_tf": float(tf_scores["ce_mean"][batch_idx].item()),
                    "ce_rollout": float(rollout_scores["ce_mean"][batch_idx].item()),
                    "exposure_bias_gap": float(
                        rollout_scores["ce_mean"][batch_idx].item() - tf_scores["ce_mean"][batch_idx].item()
                    ),
                    "rollout_token_match_rate": float(rollout_scores["token_match_rate"][batch_idx].item()),
                    "teacher_forcing_ce_per_step": tf_scores["ce_per_step"][batch_idx].tolist(),
                    "rollout_ce_per_step": rollout_scores["ce_per_step"][batch_idx].tolist(),
                    "prefix_token_ids": prefix_list,
                    "ground_truth_token_ids": target_list,
                    "generated_token_ids": generated_ids,
                    "prefix_text_preview": tokenizer.decode(prefix_list, skip_special_tokens=False),
                    "ground_truth_text_preview": tokenizer.decode(target_list, skip_special_tokens=False),
                    "generated_text_preview": tokenizer.decode(generated_ids, skip_special_tokens=False),
                }
            )

    metrics = compute_exposure_bias_metrics(predictions, cfg=cfg)
    return predictions, metrics
