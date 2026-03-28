from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AI2 ARC and save to disk")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/ai2_arc",
        help="HF dataset name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="ARC-Challenge",
        help="Dataset config (ARC-Challenge or ARC-Easy)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for datasets.save_to_disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, args.config)
    dataset.save_to_disk(str(output_dir))
    print(
        f"Saved dataset: name={args.dataset_name} config={args.config} path={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
