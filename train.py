from __future__ import annotations

import argparse

from opd.config import load_config
from opd.distributed import cleanup_distributed, init_distributed
from opd.model_loader import build_model_and_tokenizer
from opd.train_loop import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GatedDeltaNet with explicit OPD loop")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dist_env = init_distributed()
    try:
        model, tokenizer = build_model_and_tokenizer(cfg=cfg, device=dist_env.device)
        run_training(cfg=cfg, dist_env=dist_env, model=model, tokenizer=tokenizer)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
