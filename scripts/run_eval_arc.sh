#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/eval/arc_ai2.yaml}

python3 eval/run_eval.py --config "$CONFIG_PATH"
