from __future__ import annotations

import argparse

from QwenOPSD.distributed import cleanup_distributed, init_distributed
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
    dist_env = init_distributed()
    try:
        runtime = build_train_runtime(cfg=cfg, device=dist_env.device)
        run_training(
            cfg=cfg,
            dist_env=dist_env,
            student_model=runtime.student_model,
            teacher_model=runtime.teacher_model,
            tokenizer=runtime.tokenizer,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
