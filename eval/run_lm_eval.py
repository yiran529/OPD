from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eval.io import build_eval_output_dir, checkpoint_tag_from_path, write_json


def _parse_limit(value: str) -> int | float:
    try:
        return int(value)
    except ValueError:
        return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lm-eval-harness tasks with OPD model loader")
    parser.add_argument("--train-config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional .pt checkpoint path")
    parser.add_argument("--use-ema", action="store_true", help="Load ema_model from checkpoint")
    parser.add_argument("--tasks", type=str, default="arc_easy", help="Comma-separated lm-eval task names")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot example count")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-forward batch size (currently only 1)")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0")
    parser.add_argument("--limit", type=_parse_limit, default=None, help="Optional lm-eval limit")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Optional eval run name override")
    parser.add_argument("--no-log-samples", action="store_true", help="Disable per-sample logging")
    return parser.parse_args()


def _import_lm_eval_api():
    try:
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.tasks import TaskManager
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "lm_eval is not importable. Install lm-evaluation-harness first."
        ) from exc
    return simple_evaluate, TaskManager


def _build_output_dir(
    output_dir: str,
    run_name: str,
    checkpoint_path: str | None,
) -> Path:
    return build_eval_output_dir(
        output_dir=output_dir,
        run_name=run_name,
        task="lm_eval",
        checkpoint_tag=checkpoint_tag_from_path(checkpoint_path),
    )


def _print_summary(results: dict[str, Any]) -> None:
    task_results = results.get("results", {})
    for task_name, metrics in task_results.items():
        acc = metrics.get("acc,none", None)
        acc_norm = metrics.get("acc_norm,none", None)
        print(
            f"lm-eval done: task={task_name} acc={acc} acc_norm={acc_norm}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    simple_evaluate, TaskManager = _import_lm_eval_api()
    from eval.lm_eval_model import OPDLMEvalModel
    from opd.config import load_config

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    assert task_names, "At least one lm-eval task must be specified"

    train_cfg = load_config(args.train_config)
    run_name = args.run_name or train_cfg.run_name
    assert run_name, "run_name must be set either in args or train config"
    output_dir = _build_output_dir(
        output_dir=args.output_dir,
        run_name=run_name,
        checkpoint_path=args.checkpoint,
    )

    model = OPDLMEvalModel(
        train_config_path=args.train_config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        use_ema=args.use_ema,
    )

    results = simple_evaluate(
        model=model,
        tasks=task_names,
        task_manager=TaskManager(),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=not args.no_log_samples,
    )
    assert results is not None, "lm-eval returned no results"

    write_json(output_dir / "results.json", results)
    print(f"Results saved to: {output_dir / 'results.json'}", flush=True)
    _print_summary(results=results)


if __name__ == "__main__":
    main()
