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

from corruption import build_online_careless_prefix
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


def load_hf_model_for_prefix_build(
    base_model_path: str,
    tokenizer,
    checkpoint_dir: str = None,
    device: str = "cpu",
):
    print(f"Loading HF model for careless-prefix generation from: {base_model_path} on {device}")
    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    architectures = list(getattr(config, "architectures", None) or [])
    model_type = getattr(config, "model_type", None)
    print(f"HF model config.architectures = {architectures or ['<missing>']}")

    transformers_module = importlib.import_module("transformers")
    model_class = None
    selected_arch = None

    if model_type == "qwen3_5":
        expected_arch = architectures[0] if architectures else None
        assert expected_arch is not None, "Qwen3.5 checkpoint is missing config.architectures"
        model_class = getattr(transformers_module, expected_arch, None)
        assert model_class is not None, f"transformers is missing {expected_arch}"
        selected_arch = expected_arch
        print(
            "Pure-text task detected. Loading the checkpoint's published architecture "
            f"{expected_arch} to avoid key-mapping drift."
        )
    else:
        for arch_name in architectures:
            candidate = getattr(transformers_module, arch_name, None)
            if candidate is not None:
                model_class = candidate
                selected_arch = arch_name
                break

    if model_class is None:
        print(
            "Warning: could not resolve a concrete pure-text model class from config; "
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

    print(f"HF model class = {selected_arch}")
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
            "HF careless-prefix model did not load cleanly. "
            + " | ".join(preview)
        )

    if checkpoint_dir is not None:
        adapter_safetensors = Path(checkpoint_dir) / "adapter_model.safetensors"
        adapter_bin = Path(checkpoint_dir) / "adapter_model.bin"
        if adapter_safetensors.exists() or adapter_bin.exists():
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, checkpoint_dir)
                print(f"Loaded LoRA adapter into HF model from: {checkpoint_dir}")
            except Exception as exc:
                warnings.warn(
                    f"Failed to load LoRA adapter into HF model ({exc}); falling back to base model only"
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


def _prepare_examples(dataset, tokenizer, hf_model, device, args):
    examples = []
    upper_bound = min(len(dataset), args.start_index + args.num_examples)

    for index in range(args.start_index, upper_bound):
        feature = dataset[index]
        problem = feature["problem"]
        solution = feature["solution"]

        problem_prompt_ids = _build_problem_prompt_ids(tokenizer, problem)
        solution_ids = _encode_solution_ids(tokenizer, solution)

        rollout = build_online_careless_prefix(
            model=hf_model,
            tokenizer=tokenizer,
            problem=problem,
            solution=solution,
            problem_prompt_ids=problem_prompt_ids,
            solution_ids=solution_ids,
            gold_prefix_ratio_min=args.gold_prefix_ratio_min,
            gold_prefix_ratio_max=args.gold_prefix_ratio_max,
            careless_rollout_len=args.careless_rollout_len,
            careless_temperature=args.careless_temperature,
            careless_top_p=args.careless_top_p,
            careless_top_k=args.careless_top_k,
            careless_resample_trials=args.careless_resample_trials,
            careless_marker_text=args.careless_marker_text,
            recovery_marker_text=args.recovery_marker_text,
            device=device,
        )

        teacher_messages = [{"role": "user", "content": rollout["teacher_user_message"]}]
        teacher_prompt_prefix_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        example = {
            "dataset_index": index,
            "problem": problem,
            "problem_prompt_text": _decode_ids(tokenizer, problem_prompt_ids),
            "student_seen_prefix_text": _decode_ids(tokenizer, rollout["student_prompt_ids"]),
            "teacher_seen_prefix_text": teacher_prompt_prefix_text + rollout["teacher_trace_prefix_text"],
            "gold_prefix_length": int(rollout["gold_prefix_length"]),
            "careless_prefix_length": len(rollout["careless_token_ids"]),
            "careless_deviated": bool(rollout["careless_deviated"]),
            "careless_resample_count": int(rollout["careless_resample_count"]),
            "skip_kd": bool(rollout["skip_kd"]),
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
            max_tokens=args.recovery_rollout_len,
            presence_penalty=0.0,
            n=1,
        )

    raise AssertionError(f"Unsupported rollout_decoding={args.rollout_decoding}")


def _append_block(report_lines, title, content):
    report_lines.append("")
    report_lines.append("#" * 24 + f" {title} " + "#" * 24)
    report_lines.append(content if content else "(empty)")


def _generate_with_optional_lora(llm, prompts, sampling_params, lora_request):
    if lora_request is not None:
        return llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    return llm.generate(prompts, sampling_params, use_tqdm=True)


def _run_inspection_rollouts(llm, examples, sampling_params, lora_request):
    rollout_specs = [
        ("problem", "problem_prompt_text"),
        ("student", "student_seen_prefix_text"),
        ("teacher", "teacher_seen_prefix_text"),
    ]

    prompts = []
    prompt_specs = []
    for example_idx, example in enumerate(examples):
        for rollout_name, prompt_key in rollout_specs:
            prompts.append(example[prompt_key])
            prompt_specs.append((example_idx, rollout_name, prompt_key))

    outputs = _generate_with_optional_lora(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    for output, (example_idx, rollout_name, prompt_key) in zip(outputs, prompt_specs):
        generated = output.outputs[0]
        example = examples[example_idx]
        prefix_text = example[prompt_key]
        example[f"{rollout_name}_rollout_text"] = generated.text
        example[f"{rollout_name}_rollout_token_ids"] = [int(token_id) for token_id in generated.token_ids]
        example[f"{rollout_name}_full_text"] = prefix_text + generated.text


def _write_outputs(examples, output_jsonl):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_txt = output_jsonl.with_suffix(".txt")

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for example in examples:
            compact_example = {
                "problem_full_text": example.get("problem_full_text", ""),
                "student_full_text": example.get("student_full_text", ""),
                "teacher_full_text": example.get("teacher_full_text", ""),
                "problem_rollout_text": example.get("problem_rollout_text", ""),
                "student_rollout_text": example.get("student_rollout_text", ""),
                "teacher_rollout_text": example.get("teacher_rollout_text", ""),
            }
            handle.write(json.dumps(compact_example, ensure_ascii=False) + "\n")

    report_lines = []
    for example in examples:
        report_lines.append("")
        report_lines.append("=" * 120)
        report_lines.append(f"EXAMPLE {example['dataset_index']}")
        report_lines.append("=" * 120)
        report_lines.append(
            "gold_prefix_length="
            f"{example['gold_prefix_length']} "
            "careless_prefix_length="
            f"{example['careless_prefix_length']} "
            "careless_deviated="
            f"{example['careless_deviated']} "
            "skip_kd="
            f"{example['skip_kd']}"
        )
        _append_block(report_lines, "Problem Prefix", example.get("problem_prompt_text", ""))
        _append_block(report_lines, "Problem Rollout", example.get("problem_rollout_text", ""))
        _append_block(report_lines, "Problem Full", example.get("problem_full_text", ""))
        _append_block(report_lines, "Student Prefix", example.get("student_seen_prefix_text", ""))
        _append_block(report_lines, "Student Prefix Rollout", example.get("student_rollout_text", ""))
        _append_block(report_lines, "Student Full", example.get("student_full_text", ""))
        _append_block(report_lines, "Teacher Prefix", example.get("teacher_seen_prefix_text", ""))
        _append_block(report_lines, "Teacher Prefix Rollout", example.get("teacher_rollout_text", ""))
        _append_block(report_lines, "Teacher Full", example.get("teacher_full_text", ""))

    output_txt.write_text("\n".join(report_lines), encoding="utf-8")
    return output_txt


def main():
    parser = argparse.ArgumentParser(
        description="Inspect trainer-time LinearOPSD gold-prefix/careless-prefix/recovery rollouts on math reasoning data."
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
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for careless sampling.")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen thinking-mode context length.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--hf_device", type=str, default="cpu", help="Device for the HF careless-prefix model.")
    parser.add_argument("--gold_prefix_ratio_min", type=float, default=0.3)
    parser.add_argument("--gold_prefix_ratio_max", type=float, default=0.7)
    parser.add_argument("--careless_rollout_len", type=int, default=8)
    parser.add_argument("--careless_temperature", type=float, default=1.3)
    parser.add_argument("--careless_top_p", type=float, default=0.95)
    parser.add_argument("--careless_top_k", type=int, default=50)
    parser.add_argument("--careless_resample_trials", type=int, default=3)
    parser.add_argument("--rollout_decoding", choices=["greedy"], default="greedy")
    parser.add_argument("--recovery_rollout_len", type=int, default=8)
    parser.add_argument("--careless_marker_text", type=str, default="<careless>")
    parser.add_argument("--recovery_marker_text", type=str, default="<recovery>")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="inspection_outputs/linear_opsd_rollout_inspect.jsonl",
        help="Output JSONL path. A sidecar .txt report will be written with the same stem.",
    )
    args = parser.parse_args()

    assert args.num_examples > 0, "num_examples must be positive"
    assert 0.0 <= args.gold_prefix_ratio_min <= args.gold_prefix_ratio_max <= 1.0, (
        "gold_prefix ratios must satisfy 0 <= min <= max <= 1"
    )
    assert args.careless_rollout_len > 0, "careless_rollout_len must be positive"
    assert args.careless_temperature > 0.0, "careless_temperature must be positive"
    assert 0.0 < args.careless_top_p <= 1.0, "careless_top_p must be in (0, 1]"
    assert args.careless_top_k >= 0, "careless_top_k must be non-negative"
    assert args.careless_resample_trials >= 0, "careless_resample_trials must be non-negative"
    assert args.recovery_rollout_len > 0, "recovery_rollout_len must be positive"
    assert args.careless_marker_text.strip(), "careless_marker_text must be non-empty"
    assert args.recovery_marker_text.strip(), "recovery_marker_text must be non-empty"

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
    hf_model = load_hf_model_for_prefix_build(
        args.base_model,
        tokenizer=tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        device=args.hf_device,
    )

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    examples = _prepare_examples(dataset, tokenizer, hf_model, args.hf_device, args)

    sampling_params = _build_sampling_params(args)
    _run_inspection_rollouts(
        llm=llm,
        examples=examples,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    for example in examples:
        example["recovery_rollout_text"] = example["student_rollout_text"]
        example["recovery_rollout_token_ids"] = example["student_rollout_token_ids"]
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
