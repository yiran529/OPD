from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def checkpoint_tag_from_path(checkpoint_path: str | None) -> str:
    if not checkpoint_path:
        return "pretrained"
    return Path(checkpoint_path).stem


def build_eval_output_dir(
    output_dir: str,
    run_name: str,
    task: str,
    checkpoint_tag: str,
) -> Path:
    eval_dir = Path(output_dir) / run_name / "eval" / task / checkpoint_tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

