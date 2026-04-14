# #!/usr/bin/env bash
# set -euo pipefail

# CONFIG_PATH=${1:-configs/eval/arc_ai2.yaml}

# python3 -m eval.run_eval.py --config "$CONFIG_PATH"
python3 -m eval.run_eval --config configs/eval/arc_ai2.yaml