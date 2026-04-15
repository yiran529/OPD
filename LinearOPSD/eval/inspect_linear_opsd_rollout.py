import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from vllm import SamplingParams

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_collator import _build_linear_opsd_prefixes, _build_problem_prompt_ids, _encode_solution_ids
from evaluate_math import load_vllm_model


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


def _extract_corrupted_spans(tokenizer, student_prefix_ids, teacher_prefix_ids):
    assert len(student_prefix_ids) == len(teacher_prefix_ids), "student/teacher prefix lengths must match"

    spans = []
    span_start = None
    for idx, (student_token, teacher_token) in enumerate(zip(student_prefix_ids, teacher_prefix_ids)):
        if student_token != teacher_token and span_start is None:
            span_start = idx
        elif student_token == teacher_token and span_start is not None:
            span_end = idx
            spans.append(
                {
                    "start": span_start,
                    "end": span_end,
                    "length": span_end - span_start,
                    "corrupted_text": _decode_ids(tokenizer, student_prefix_ids[span_start:span_end]),
                    "clean_text": _decode_ids(tokenizer, teacher_prefix_ids[span_start:span_end]),
                }
            )
            span_start = None

    if span_start is not None:
        span_end = len(student_prefix_ids)
        spans.append(
            {
                "start": span_start,
                "end": span_end,
                "length": span_end - span_start,
                "corrupted_text": _decode_ids(tokenizer, student_prefix_ids[span_start:span_end]),
                "clean_text": _decode_ids(tokenizer, teacher_prefix_ids[span_start:span_end]),
            }
        )

    return spans


def _prepare_examples(dataset, tokenizer, args):
    examples = []
    upper_bound = min(len(dataset), args.start_index + args.num_examples)
    for index in range(args.start_index, upper_bound):
        feature = dataset[index]
        problem = feature["problem"]
        solution = feature["solution"]

        prompt_ids = _build_problem_prompt_ids(tokenizer, problem)
        solution_ids = _encode_solution_ids(tokenizer, solution)
        corruption = _build_linear_opsd_prefixes(
            solution_ids=solution_ids,
            rollout_len=args.max_new_tokens,
            num_spans=args.num_corrupt_spans,
            span_choices=args.corrupt_span_choices,
            start_min_ratio=args.corrupt_start_min_ratio,
            start_max_ratio=args.corrupt_start_max_ratio,
        )

        student_prefix_ids = corruption["student_prefix_ids"]
        teacher_prefix_ids = corruption["teacher_prefix_ids"]
        student_prompt_ids = prompt_ids + student_prefix_ids
        teacher_prompt_ids = prompt_ids + teacher_prefix_ids

        example = {
            "dataset_index": index,
            "problem": problem,
            "solution": solution,
            "prompt_ids": prompt_ids,
            "student_prefix_ids": student_prefix_ids,
            "teacher_prefix_ids": teacher_prefix_ids,
            "student_prompt_ids": student_prompt_ids,
            "teacher_prompt_ids": teacher_prompt_ids,
            "rollout_start": corruption["rollout_start"],
            "num_spans": corruption["num_spans"],
            "span_len": corruption["span_len"],
            "solution_length": corruption["solution_length"],
            "corrupted_spans": _extract_corrupted_spans(tokenizer, student_prefix_ids, teacher_prefix_ids),
            "problem_prompt_text": _decode_ids(tokenizer, prompt_ids),
            "student_prefix_text": _decode_ids(tokenizer, student_prefix_ids),
            "teacher_prefix_text": _decode_ids(tokenizer, teacher_prefix_ids),
            "clean_prefix_text": _decode_ids(tokenizer, solution_ids[: corruption["rollout_start"]]),
            "student_prompt_text": _decode_ids(tokenizer, student_prompt_ids),
            "teacher_prompt_text": _decode_ids(tokenizer, teacher_prompt_ids),
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


def _write_outputs(examples, output_jsonl):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_txt = output_jsonl.with_suffix(".txt")

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    report_lines = []
    for example in examples:
        report_lines.append("=" * 100)
        report_lines.append(f"dataset_index: {example['dataset_index']}")
        report_lines.append(f"rollout_start: {example['rollout_start']}")
        report_lines.append(f"num_spans: {example['num_spans']}")
        report_lines.append(f"span_len: {example['span_len']}")
        report_lines.append(f"solution_length: {example['solution_length']}")
        report_lines.append("problem:")
        report_lines.append(example["problem"])
        report_lines.append("clean_prefix_text:")
        report_lines.append(example["clean_prefix_text"])
        report_lines.append("student_prefix_text:")
        report_lines.append(example["student_prefix_text"])
        report_lines.append("teacher_prefix_text:")
        report_lines.append(example["teacher_prefix_text"])
        report_lines.append("corrupted_spans:")
        for span in example["corrupted_spans"]:
            report_lines.append(
                f"  start={span['start']} length={span['length']} "
                f"corrupted={span['corrupted_text']!r} clean={span['clean_text']!r}"
            )
        report_lines.append("rollout_text:")
        report_lines.append(example["rollout_text"])

    output_txt.write_text("\n".join(report_lines), encoding="utf-8")
    return output_txt


def main():
    parser = argparse.ArgumentParser(
        description="Inspect LinearOPSD corrupted prefixes and student rollouts on math reasoning data."
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
    parser.add_argument("--rollout_decoding", choices=["sample", "greedy"], default="sample")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--num_corrupt_spans", type=int, default=1)
    parser.add_argument(
        "--corrupt_span_choices",
        type=str,
        default="2",
        help="Comma-separated candidate span lengths, e.g. `2` or `2,4`.",
    )
    parser.add_argument("--corrupt_start_min_ratio", type=float, default=0.0)
    parser.add_argument("--corrupt_start_max_ratio", type=float, default=0.5)
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="inspection_outputs/linear_opsd_rollout_inspect.jsonl",
        help="Output JSONL path. A sidecar .txt report will be written with the same stem.",
    )
    args = parser.parse_args()

    args.corrupt_span_choices = [int(value.strip()) for value in args.corrupt_span_choices.split(",") if value.strip()]
    assert args.corrupt_span_choices, "corrupt_span_choices must contain at least one positive integer"
    assert all(value > 0 for value in args.corrupt_span_choices), "corrupt_span_choices values must be positive"
    assert args.num_examples > 0, "num_examples must be positive"
    assert args.num_corrupt_spans > 0, "num_corrupt_spans must be positive"
    assert 0.0 <= args.corrupt_start_min_ratio <= args.corrupt_start_max_ratio <= 1.0, (
        "corrupt_start ratios must satisfy 0 <= min <= max <= 1"
    )

    random.seed(args.seed)

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )
    lora_request = _build_lora_request(args.checkpoint_dir)

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    examples = _prepare_examples(dataset, tokenizer, args)

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
        example["student_prompt_plus_rollout_text"] = example["student_prompt_text"] + generated.text
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
