from __future__ import annotations

from typing import Iterable

from exposure_bias.eval.config import ExposureBiasEvalConfig
from exposure_bias.text_data import iter_hf_dataset_examples


def iter_hf_text_examples(
    cfg: ExposureBiasEvalConfig,
    tokenizer,
) -> Iterable[dict]:
    seq_len = cfg.prefix_len + cfg.rollout_len
    return iter_hf_dataset_examples(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        dataset_split=cfg.dataset_split,
        dataset_text_field=cfg.dataset_text_field,
        local_dataset_path=cfg.local_dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        max_samples=cfg.max_samples,
        sample_prefix="hf_dataset",
    )
