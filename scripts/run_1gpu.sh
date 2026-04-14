#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/gdn_340m_opd.yaml}

python3 train.py --config "$CONFIG_PATH"
