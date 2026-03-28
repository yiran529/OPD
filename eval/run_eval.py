from __future__ import annotations

import argparse

from eval.config import load_eval_config
from eval.io import (
    build_eval_output_dir,
    checkpoint_tag_from_path,
    write_json,
    write_jsonl,
)
from eval.model_runtime import build_eval_model_and_tokenizer
from eval.tasks.arc_ai2.runner import run_arc_ai2_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to eval YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_cfg = load_eval_config(args.config)

    if eval_cfg.task != "arc_ai2":
        raise ValueError(f"Unsupported task: {eval_cfg.task}")

    model, tokenizer, train_cfg, loaded_step, device = build_eval_model_and_tokenizer(eval_cfg)
    predictions, metrics = run_arc_ai2_eval(
        cfg=eval_cfg,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    run_name = eval_cfg.run_name or train_cfg.run_name
    checkpoint_tag = checkpoint_tag_from_path(eval_cfg.checkpoint_path)
    output_dir = build_eval_output_dir(
        output_dir=eval_cfg.output_dir,
        run_name=run_name,
        task=eval_cfg.task,
        checkpoint_tag=checkpoint_tag,
    )

    metrics["loaded_step"] = loaded_step
    write_json(output_dir / "metrics.json", metrics)
    write_jsonl(output_dir / "predictions.jsonl", predictions)

    print(
        f"Eval done: task={eval_cfg.task} config={eval_cfg.dataset_config} split={eval_cfg.dataset_split} "
        f"accuracy={metrics['accuracy']:.4f} num_examples={metrics['num_examples']} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
