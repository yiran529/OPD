from __future__ import annotations

from pathlib import Path

import yaml

from QwenOPSD.eval.config import QwenOPSDEvalConfig
from QwenOPSD.io import build_eval_output_dir, checkpoint_tag_from_path, write_json


def run_evaluation(
    cfg: QwenOPSDEvalConfig,
    model,
    tokenizer,
    device,
    loaded_step: int,
) -> None:
    del model
    del tokenizer
    del device

    checkpoint_tag = checkpoint_tag_from_path(cfg.checkpoint_path)
    output_dir = build_eval_output_dir(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        task=cfg.task,
        checkpoint_tag=checkpoint_tag,
    )
    (Path(output_dir) / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg.as_dict(), sort_keys=False),
        encoding="utf-8",
    )
    write_json(
        Path(output_dir) / "placeholder_metrics.json",
        {
            "implemented": False,
            "task": cfg.task,
            "checkpoint_tag": checkpoint_tag,
            "loaded_step": loaded_step,
            "message": "QwenOPSD eval scaffold exists, but evaluation logic is not implemented yet.",
        },
    )
    print(
        "QwenOPSD eval placeholder complete: "
        f"task={cfg.task} checkpoint_tag={checkpoint_tag} output_dir={output_dir}",
        flush=True,
    )

