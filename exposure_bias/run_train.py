from __future__ import annotations

import argparse

from exposure_bias.train.config import load_config
from exposure_bias.train.loop import run_training
from exposure_bias.train.runtime import build_train_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local-text LoRA finetuning for exposure-bias experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to exposure-bias train YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = build_train_runtime(cfg)

    run_training(
        cfg=cfg,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        device=runtime.device,
    )


if __name__ == "__main__":
    main()
