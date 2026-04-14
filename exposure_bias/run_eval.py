from __future__ import annotations

import argparse

from exposure_bias.eval.config import load_config
from exposure_bias.io import (
    build_experiment_name,
    build_output_dir,
    checkpoint_tag_from_path,
    dataset_tag_from_source,
    write_json,
    write_jsonl,
)
from exposure_bias.eval.runners.hf_dataset import run_hf_dataset_eval
from exposure_bias.eval.runners.gsm8k_thought_reveal import run_gsm8k_thought_reveal_eval
from exposure_bias.eval.runtime import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exposure-bias evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to exposure bias eval YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_runtime(cfg)
    dataset_tag = dataset_tag_from_source(
        dataset_name=cfg.dataset_name,
        local_dataset_path=cfg.local_dataset_path,
    )

    if cfg.task == "hf_dataset":
        predictions, metrics = run_hf_dataset_eval(cfg=cfg, runtime=runtime)
    elif cfg.task == "gsm8k_thought_reveal":
        predictions, metrics = run_gsm8k_thought_reveal_eval(cfg=cfg, runtime=runtime)
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")
    experiment_name = cfg.run_name or build_experiment_name(
        dataset_tag=dataset_tag,
        model_name=runtime.train_cfg.model_name,
        prefix_len=cfg.prefix_len,
        rollout_len=cfg.rollout_len,
    )
    output_dir = build_output_dir(
        output_dir=cfg.output_dir,
        experiment_name=experiment_name,
        dataset_tag=dataset_tag,
        checkpoint_tag=checkpoint_tag_from_path(cfg.checkpoint_path),
    )

    metrics["loaded_step"] = runtime.loaded_step
    metrics["model_impl"] = cfg.model_impl

    write_json(output_dir / "config.json", cfg.as_dict())
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)

    if cfg.task == "hf_dataset":
        print(
            f"Exposure bias eval done: dataset={dataset_tag} split={cfg.dataset_split} "
            f"mean_ce_tf={metrics['mean_ce_tf']:.4f} "
            f"mean_ce_rollout={metrics['mean_ce_rollout']:.4f} "
            f"mean_gap={metrics['mean_exposure_bias_gap']:.4f} "
            f"output_dir={output_dir}",
            flush=True,
        )
    else:
        print(
            f"GSM8K thought-reveal eval done: dataset={dataset_tag} split={cfg.dataset_split} "
            f"acc0={metrics['acc_by_ratio']['0.0']:.4f} "
            f"acc25={metrics['acc_by_ratio'].get('0.25', float('nan')):.4f} "
            f"acc50={metrics['acc_by_ratio'].get('0.5', float('nan')):.4f} "
            f"acc75={metrics['acc_by_ratio'].get('0.75', float('nan')):.4f} "
            f"output_dir={output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
