from __future__ import annotations

import argparse

from memory_pollution.config import load_config
from memory_pollution.io import build_output_dir, checkpoint_tag_from_path, write_json, write_jsonl
from memory_pollution.runners.arc_eval import run_arc_eval
from memory_pollution.runtime import build_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory pollution evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to memory pollution eval YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_runtime(cfg)

    if cfg.task != "arc":
        raise ValueError(f"Unsupported task: {cfg.task}")

    predictions, metrics = run_arc_eval(cfg=cfg, runtime=runtime)
    run_name = cfg.run_name or runtime.train_cfg.run_name
    assert run_name, "run_name must be set either in config or train config"
    output_dir = build_output_dir(
        output_dir=cfg.output_dir,
        run_name=run_name,
        task=cfg.task,
        checkpoint_tag=checkpoint_tag_from_path(cfg.checkpoint_path),
    )

    metrics["loaded_step"] = runtime.loaded_step
    metrics["model_impl"] = cfg.model_impl
    metrics["state_key"] = runtime.state_key if runtime.supports_state_drift else None

    write_json(output_dir / "config.json", cfg.as_dict())
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)

    print(
        f"Memory pollution eval done: task={cfg.task} config={cfg.dataset_config} split={cfg.dataset_split} "
        f"clean_accuracy={metrics['clean_accuracy']:.4f} perturb_accuracy={metrics['perturb_accuracy']:.4f} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

