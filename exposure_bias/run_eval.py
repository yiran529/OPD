from __future__ import annotations

import argparse

from exposure_bias.config import load_config
from exposure_bias.io import (
    build_experiment_name,
    build_output_dir,
    checkpoint_tag_from_path,
    write_json,
    write_jsonl,
)
from exposure_bias.runners.fineweb_edu_eval import run_fineweb_edu_eval
from exposure_bias.runtime import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exposure-bias evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to exposure bias eval YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_runtime(cfg)

    if cfg.task != "fineweb_edu":
        raise ValueError(f"Unsupported task: {cfg.task}")

    predictions, metrics = run_fineweb_edu_eval(cfg=cfg, runtime=runtime)
    experiment_name = cfg.run_name or build_experiment_name(
        model_name=runtime.train_cfg.model_name,
        prefix_len=cfg.prefix_len,
        rollout_len=cfg.rollout_len,
    )
    output_dir = build_output_dir(
        output_dir=cfg.output_dir,
        experiment_name=experiment_name,
        task=cfg.task,
        checkpoint_tag=checkpoint_tag_from_path(cfg.checkpoint_path),
    )

    metrics["loaded_step"] = runtime.loaded_step
    metrics["model_impl"] = cfg.model_impl

    write_json(output_dir / "config.json", cfg.as_dict())
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)

    print(
        f"Exposure bias eval done: task={cfg.task} split={cfg.dataset_split} "
        f"mean_ce_tf={metrics['mean_ce_tf']:.4f} "
        f"mean_ce_rollout={metrics['mean_ce_rollout']:.4f} "
        f"mean_gap={metrics['mean_exposure_bias_gap']:.4f} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
