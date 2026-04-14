from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: str) -> dict:
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(f'metrics file not found: {metrics_path}')
    data = json.loads(metrics_path.read_text())
    if 'acc_by_ratio' not in data or 'gap_by_ratio' not in data:
        raise ValueError(f'not a gsm8k thought-reveal metrics file: {metrics_path}')
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare GSM8K thought-reveal gaps between two models')
    parser.add_argument('--lhs', required=True, help='Path to first metrics.json')
    parser.add_argument('--rhs', required=True, help='Path to second metrics.json')
    parser.add_argument('--lhs-name', default='lhs')
    parser.add_argument('--rhs-name', default='rhs')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lhs = load_metrics(args.lhs)
    rhs = load_metrics(args.rhs)

    for ratio_key in ('0.25', '0.5', '0.75'):
        lhs_gap = float(lhs['gap_by_ratio'][ratio_key])
        rhs_gap = float(rhs['gap_by_ratio'][ratio_key])
        label = ratio_key[2:] if ratio_key.startswith('0.') else ratio_key.replace('.', '_')
        print(f'delta_Gap_{label}: {lhs_gap - rhs_gap:.6f}')

    print(f'{args.lhs_name}_acc_by_ratio={lhs["acc_by_ratio"]}')
    print(f'{args.rhs_name}_acc_by_ratio={rhs["acc_by_ratio"]}')
    print(f'{args.lhs_name}_gap_by_ratio={lhs["gap_by_ratio"]}')
    print(f'{args.rhs_name}_gap_by_ratio={rhs["gap_by_ratio"]}')


if __name__ == '__main__':
    main()
