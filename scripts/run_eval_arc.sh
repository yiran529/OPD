#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/eval/arc_ai2.yaml}

python3 -m eval.run_eval --config "$CONFIG_PATH"
