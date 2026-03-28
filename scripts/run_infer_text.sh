#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/eval/infer_text.yaml}

python3 -m eval.run_infer --config "$CONFIG_PATH"
