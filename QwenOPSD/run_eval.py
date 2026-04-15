from __future__ import annotations

import argparse

from QwenOPSD.eval.config import load_config
from QwenOPSD.eval.runner import run_evaluation
from QwenOPSD.eval.runtime import build_eval_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QwenOPSD evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to QwenOPSD eval YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_eval_runtime(cfg)
    run_evaluation(
        cfg=cfg,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        device=runtime.device,
        loaded_step=runtime.loaded_step,
    )


if __name__ == "__main__":
    main()

