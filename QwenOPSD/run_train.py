from __future__ import annotations

import argparse

from QwenOPSD.train.config import load_config
from QwenOPSD.train.loop import run_training
from QwenOPSD.train.runtime import build_train_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QwenOPSD training")
    parser.add_argument("--config", type=str, required=True, help="Path to QwenOPSD train YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_train_runtime(cfg)
    run_training(
        cfg=cfg,
        student_model=runtime.student_model,
        teacher_model=runtime.teacher_model,
        tokenizer=runtime.tokenizer,
        device=runtime.device,
    )


if __name__ == "__main__":
    main()

