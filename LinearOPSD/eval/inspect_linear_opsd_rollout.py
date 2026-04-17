import argparse
import importlib
import json
import random
import sys
import warnings
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from corruption import build_online_corruption, is_style_token
from data_collator import _build_problem_prompt_ids, _encode_solution_ids


def _get_vllm_cache_dtype(llm) -> str:
    engine = llm.llm_engine

    cache_config = getattr(engine, "cache_config", None)
    if cache_config is None:
        vllm_config = getattr(engine, "vllm_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)

    cache_dtype = getattr(cache_config, "cache_dtype", None)
    return str(cache_dtype) if cache_dtype is not None else "unavailable"


def load_vllm_model(
    base_model_path: str,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = None,
    enable_thinking: bool = True,
):
    print(f"Loading model with vLLM from: {base_model_path}")

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768
        mode = "thinking" if enable_thinking else "non-thinking"
        print(f"Auto-setting max_model_len to {max_model_len} for {mode} mode")

    llm_config = {
        "model": base_model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "distributed_executor_backend": "mp",
        "enforce_eager": True,
    }

    if lora_adapter_path is not None:
        print(f"LoRA adapter path provided: {lora_adapter_path}")

        adapter_path = Path(lora_adapter_path) / "adapter_model.safetensors"
        if not adapter_path.exists():
            adapter_path = Path(lora_adapter_path) / "adapter_model.bin"

        if adapter_path.exists():
            print("LoRA weights found. Enabling LoRA support...")
            llm_config["enable_lora"] = True
            llm_config["max_lora_rank"] = 64
            llm_config["max_loras"] = 1
            llm_config["max_cpu_loras"] = 1
        else:
            print(f"Warning: No LoRA weights found at {lora_adapter_path}")
            print("Continuing with base model only...")

    llm = LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("\n" + "=" * 70)
    print("MODEL DTYPE INFORMATION")
    print("=" * 70)
    print(f"vLLM Model Config dtype: {llm.llm_engine.model_config.dtype}")
    print(f"vLLM Model quantization: {llm.llm_engine.model_config.quantization}")
    print(f"KV cache dtype: {_get_vllm_cache_dtype(llm)}")
    print("=" * 70 + "\n")

    print("vLLM model loaded successfully!")
    return llm, tokenizer


def load_hf_model_for_corruption(
    base_model_path: str,
    tokenizer,
    checkpoint_dir: str = None,
    device: str = "cpu",
):
    print(f"Loading HF model for corruption scoring from: {base_model_path} on {device}")
    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    architectures = list(getattr(config, "architectures", None) or [])
    print(f"HF corruption model config.architectures = {architectures or ['<missing>']}")

    transformers_module = importlib.import_module("transformers")
    model_class = None
    selected_arch = None
    for arch_name in architectures:
        candidate = getattr(transformers_module, arch_name, None)
        if candidate is not None:
            model_class = candidate
            selected_arch = arch_name
            break

    if model_class is None:
        print(
            "Warning: could not resolve a concrete model class from config.architectures; "
            "falling back to AutoModelForCausalLM"
        )

        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            promoted = {}
            if hasattr(text_config, "to_dict"):
                promoted.update(text_config.to_dict())
            for attr_name in dir(text_config):
                if attr_name.startswith("_") or attr_name in promoted:
                    continue
                try:
                    value = getattr(text_config, attr_name)
                except Exception:
                    continue
                if callable(value):
                    continue
                promoted[attr_name] = value
            for attr_name, value in promoted.items():
                if not hasattr(config, attr_name):
                    setattr(config, attr_name, value)

        if not hasattr(config, "vocab_size"):
            config.vocab_size = len(tokenizer)
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = tokenizer.pad_token_id
        if not hasattr(config, "bos_token_id"):
            config.bos_token_id = tokenizer.bos_token_id
        if not hasattr(config, "eos_token_id"):
            config.eos_token_id = tokenizer.eos_token_id

        model, loading_info = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=config,
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )
        selected_arch = "AutoModelForCausalLM"
    else:
        model, loading_info = model_class.from_pretrained(
            base_model_path,
            config=config,
            trust_remote_code=True,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            output_loading_info=True,
        )

    print(f"HF corruption model class = {selected_arch}")
    missing = list(loading_info.get("missing_keys", []) or [])
    unexpected = list(loading_info.get("unexpected_keys", []) or [])
    mismatched = list(loading_info.get("mismatched_keys", []) or [])
    errors = list(loading_info.get("error_msgs", []) or [])
    if missing or unexpected or mismatched or errors:
        preview = []
        if missing:
            preview.append(f"missing_keys={missing[:8]}")
        if unexpected:
            preview.append(f"unexpected_keys={unexpected[:8]}")
        if mismatched:
            preview.append(f"mismatched_keys={mismatched[:8]}")
        if errors:
            preview.append(f"error_msgs={errors[:4]}")
        raise RuntimeError(
            "HF corruption-scoring model did not load cleanly. "
            "This usually means the selected architecture class does not match the checkpoint structure. "
            + " | ".join(preview)
        )

    if checkpoint_dir is not None:
        adapter_safetensors = Path(checkpoint_dir) / "adapter_model.safetensors"
        adapter_bin = Path(checkpoint_dir) / "adapter_model.bin"
        if adapter_safetensors.exists() or adapter_bin.exists():
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, checkpoint_dir)
                print(f"Loaded LoRA adapter into HF scoring model from: {checkpoint_dir}")
            except Exception as exc:
                warnings.warn(
                    f"Failed to load LoRA adapter into HF scoring model ({exc}); falling back to base model only"
                )

    model.to(device)
    model.eval()
    return model


def _build_lora_request(checkpoint_dir):
    if checkpoint_dir is None:
        return None

    try:
        from vllm.lora.request import LoRARequest
    except ImportError:
        print("Warning: Could not import LoRARequest. Running without LoRA.")
        return None

    adapter_safetensors = Path(checkpoint_dir) / "adapter_model.safetensors"
    adapter_bin = Path(checkpoint_dir) / "adapter_model.bin"
    if not adapter_safetensors.exists() and not adapter_bin.exists():
        print(f"Warning: No LoRA adapter weights found at {checkpoint_dir}. Running with base model only.")
        return None

    return LoRARequest("inspection_lora", 1, checkpoint_dir)


def _decode_ids(tokenizer, token_ids):
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _format_corruption_details(tokenizer, solution_ids, corruption):
    details = []
    entropies = corruption["entropies"]
    replacement_token_ids = corruption["replacement_token_ids"]
    for pos, replacement_id in zip(corruption["corruption_positions"], replacement_token_ids):
        gold_id = int(solution_ids[pos])
        details.append(
            {
                "position": int(pos),
                "entropy": float(entropies[pos].item()),
                "gold_token_id": gold_id,
                "gold_token_text": _decode_ids(tokenizer, [gold_id]),
                "replacement_token_id": int(replacement_id),
                "replacement_token_text": _decode_ids(tokenizer, [replacement_id]),
                "is_style_token_gold": bool(is_style_token(tokenizer, gold_id)),
            }
        )
    return details


def _prepare_examples(dataset, tokenizer, hf_model, device, args):
    examples = []
    upper_bound = min(len(dataset), args.start_index + args.num_examples)

    for index in range(args.start_index, upper_bound):
        feature = dataset[index]
        problem = feature["problem"]
        solution = feature["solution"]

        problem_prompt_ids = _build_problem_prompt_ids(tokenizer, problem)
        solution_ids = _encode_solution_ids(tokenizer, solution)
        clean_input_ids = torch.tensor([problem_prompt_ids + solution_ids], dtype=torch.long, device=device)
        clean_attention_mask = torch.ones_like(clean_input_ids)

        with torch.no_grad():
            outputs = hf_model(
                input_ids=clean_input_ids,
                attention_mask=clean_attention_mask,
            )
        solution_logits = outputs.logits[0, len(problem_prompt_ids) - 1 : len(problem_prompt_ids) + len(solution_ids) - 1, :]

        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            corruption = build_online_corruption(
                tokenizer=tokenizer,
                problem=problem,
                solution=solution,
                problem_prompt_ids=problem_prompt_ids,
                solution_ids=solution_ids,
                solution_logits=solution_logits,
                num_corrupt_points=args.num_corrupt_points,
                rollout_start_offset=args.rollout_start_offset,
                rollout_start_offset_jitter=args.rollout_start_offset_jitter,
                corrupt_start_min_ratio=args.corrupt_start_min_ratio,
                corrupt_start_max_ratio=args.corrupt_start_max_ratio,
                corrupt_marker_text=args.corrupt_marker_text,
            )

        teacher_messages = [{"role": "user", "content": corruption["teacher_user_message"]}]
        teacher_prompt_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        example = {
            "dataset_index": index,
            "problem": problem,
            "solution_text": _decode_ids(tokenizer, solution_ids),
            "student_prefix_text": _decode_ids(tokenizer, corruption["corrupted_prefix_ids"]),
            "teacher_trace_text": corruption["teacher_trace_text"],
            "rollout_start": int(corruption["rollout_start"]),
            "rollout_start_offset": int(corruption["rollout_start_offset"]),
            "rollout_start_offset_delta": int(corruption["rollout_start_offset_delta"]),
            "solution_length": len(solution_ids),
            "num_corrupt_points_requested": int(args.num_corrupt_points),
            "num_corrupt_points_actual": len(corruption["corruption_positions"]),
            "corruption_points": _format_corruption_details(tokenizer, solution_ids, corruption),
            "warnings": [str(record.message) for record in warning_records],
            "student_prompt_text": _decode_ids(tokenizer, corruption["student_prompt_ids"]),
        }
        examples.append(example)

    assert examples, "No examples selected for inspection"
    return examples


def _build_sampling_params(args):
    if args.rollout_decoding == "greedy":
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            max_tokens=args.max_new_tokens,
            presence_penalty=0.0,
            n=1,
        )

    top_k = args.top_k if args.top_k and args.top_k > 0 else -1
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
        min_p=args.min_p,
        max_tokens=args.max_new_tokens,
        presence_penalty=args.presence_penalty,
        n=1,
    )


def _append_block(report_lines, title, content):
    report_lines.append("")
    report_lines.append("#" * 24 + f" {title} " + "#" * 24)
    report_lines.append(content if content else "(empty)")


def _format_corruption_block(example):
    lines = [
        f"rollout_start: {example['rollout_start']}",
        f"rollout_start_offset: {example['rollout_start_offset']}",
        f"rollout_start_offset_delta: {example['rollout_start_offset_delta']}",
        f"num_corrupt_points: {example['num_corrupt_points_actual']} / requested {example['num_corrupt_points_requested']}",
    ]
    if example["corruption_points"]:
        lines.append("")
        for idx, point in enumerate(example["corruption_points"], start=1):
            lines.append(
                f"[{idx}] pos={point['position']} entropy={point['entropy']:.4f} "
                f"gold={point['gold_token_text']!r} -> repl={point['replacement_token_text']!r}"
            )
    else:
        lines.append("")
        lines.append("No corruption points selected.")

    if example["warnings"]:
        lines.append("")
        lines.append("warnings:")
        for warning_text in example["warnings"]:
            lines.append(f"- {warning_text}")
    return "\n".join(lines)


def _write_outputs(examples, output_jsonl):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_txt = output_jsonl.with_suffix(".txt")

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    report_lines = []
    for example in examples:
        report_lines.append("")
        report_lines.append("=" * 120)
        report_lines.append(f"EXAMPLE {example['dataset_index']}")
        report_lines.append("=" * 120)
        _append_block(report_lines, "Problem", example["problem"])
        _append_block(report_lines, "Gold Solution", example["solution_text"])
        _append_block(report_lines, "Corruption", _format_corruption_block(example))
        _append_block(report_lines, "Student Prefix", example["student_prefix_text"])
        _append_block(report_lines, "Teacher Trace", example["teacher_trace_text"])
        _append_block(report_lines, "Rollout", example.get("rollout_text", ""))

    output_txt.write_text("\n".join(report_lines), encoding="utf-8")
    return output_txt


def main():
    parser = argparse.ArgumentParser(
        description="Inspect trainer-time entropy-based LinearOPSD corruption and student rollouts on math reasoning data."
    )
    parser.add_argument("--base_model", type=str, required=True, help="Base model path for vLLM inference.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Optional LoRA checkpoint directory. If absent, inspect the base model only.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="open-r1/OpenThoughts-114k-math",
        help="Dataset name or local path containing problem/solution fields.",
    )
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to inspect.")
    parser.add_argument("--num_examples", type=int, default=8, help="Number of examples to inspect.")
    parser.add_argument("--start_index", type=int, default=0, help="Starting dataset index.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for corruption sampling.")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen thinking-mode context length.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--corruption_device", type=str, default="cpu", help="Device for HF corruption scoring model.")
    parser.add_argument("--rollout_decoding", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--num_corrupt_points", type=int, default=1)
    parser.add_argument("--corrupt_marker_text", type=str, default="<corrupt>")
    parser.add_argument("--rollout_start_offset", type=int, default=2)
    parser.add_argument("--rollout_start_offset_jitter", type=int, default=10)
    parser.add_argument("--corrupt_start_min_ratio", type=float, default=0.0)
    parser.add_argument("--corrupt_start_max_ratio", type=float, default=0.5)
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="inspection_outputs/linear_opsd_rollout_inspect.jsonl",
        help="Output JSONL path. A sidecar .txt report will be written with the same stem.",
    )
    args = parser.parse_args()

    assert args.num_examples > 0, "num_examples must be positive"
    assert args.num_corrupt_points > 0, "num_corrupt_points must be positive"
    assert args.corrupt_marker_text.strip(), "corrupt_marker_text must be non-empty"
    assert args.rollout_start_offset >= 0, "rollout_start_offset must be non-negative"
    assert args.rollout_start_offset_jitter >= 0, "rollout_start_offset_jitter must be non-negative"
    assert 0.0 <= args.corrupt_start_min_ratio <= args.corrupt_start_max_ratio <= 1.0, (
        "corrupt_start ratios must satisfy 0 <= min <= max <= 1"
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )
    lora_request = _build_lora_request(args.checkpoint_dir)
    hf_model = load_hf_model_for_corruption(
        args.base_model,
        tokenizer=tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        device=args.corruption_device,
    )

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    examples = _prepare_examples(dataset, tokenizer, hf_model, args.corruption_device, args)

    sampling_params = _build_sampling_params(args)
    prompts = [example["student_prompt_text"] for example in examples]
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for example, output in zip(examples, outputs):
        generated = output.outputs[0]
        example["rollout_text"] = generated.text
        example["rollout_token_ids"] = [int(token_id) for token_id in generated.token_ids]
        example["decoding_mode"] = args.rollout_decoding
        example["checkpoint_dir"] = args.checkpoint_dir
        example["base_model"] = args.base_model

    output_jsonl = Path(args.output_jsonl)
    output_txt = _write_outputs(examples, output_jsonl)

    print("\n" + "=" * 80)
    print("LINEAROPSD ROLLOUT INSPECTION COMPLETE")
    print("=" * 80)
    print(f"examples: {len(examples)}")
    print(f"jsonl: {output_jsonl}")
    print(f"report: {output_txt}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
