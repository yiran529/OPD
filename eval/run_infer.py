from __future__ import annotations

import argparse

import torch

from eval.infer_config import load_infer_config
from eval.io import build_eval_output_dir, checkpoint_tag_from_path, write_json, write_jsonl
from eval.model_runtime import build_model_and_tokenizer_from_paths
from eval.readers.text_reader import read_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text inference")
    parser.add_argument("--config", type=str, required=True, help="Path to infer YAML config")
    return parser.parse_args()


def _build_generation_kwargs(cfg, tokenizer) -> dict:
    do_sample = cfg.temperature > 0.0
    kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["do_sample"] = True
        kwargs["temperature"] = cfg.temperature
        kwargs["top_p"] = cfg.top_p
    else:
        kwargs["do_sample"] = False
    return kwargs


@torch.no_grad()
def _generate_one(model, tokenizer, device: torch.device, prompt: str, generation_kwargs: dict) -> str:
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    if input_ids.numel() == 0:
        raise ValueError("Encountered empty prompt after tokenization")

    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs,
    )
    new_tokens = generated[:, input_ids.size(1) :]
    return tokenizer.decode(new_tokens[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    infer_cfg = load_infer_config(args.config)

    model, tokenizer, train_cfg, loaded_step, device = build_model_and_tokenizer_from_paths(
        train_config_path=infer_cfg.train_config_path,
        checkpoint_path=infer_cfg.checkpoint_path,
    )

    texts = read_inputs(infer_cfg)
    generation_kwargs = _build_generation_kwargs(infer_cfg, tokenizer)

    rows: list[dict] = []
    for index, prompt in enumerate(texts):
        output_text = _generate_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            generation_kwargs=generation_kwargs,
        )
        rows.append(
            {
                "index": index,
                "input_text": prompt,
                "output_text": output_text,
            }
        )

    run_name = infer_cfg.run_name or train_cfg.run_name
    checkpoint_tag = checkpoint_tag_from_path(infer_cfg.checkpoint_path)
    output_dir = build_eval_output_dir(
        output_dir=infer_cfg.output_dir,
        run_name=run_name,
        task=infer_cfg.infer_name,
        checkpoint_tag=checkpoint_tag,
    )

    summary = {
        "loaded_step": loaded_step,
        "num_inputs": len(rows),
        "max_new_tokens": infer_cfg.max_new_tokens,
        "temperature": infer_cfg.temperature,
        "top_p": infer_cfg.top_p,
        "input_format": infer_cfg.input_format,
        "input_path": infer_cfg.input_path,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "predictions.jsonl", rows)

    print(
        f"Infer done: num_inputs={len(rows)} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
