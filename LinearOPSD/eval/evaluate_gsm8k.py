import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_FINAL_MARKER_RE = re.compile(r"####\s*([^\n\r]+)")
_THE_ANSWER_IS_RE = re.compile(r"the answer is\s*:?\s*([^\n\r]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def normalize_gsm8k_answer(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".。")
    return text


def extract_gsm8k_ground_truth(answer_text: str) -> str:
    marker = "####"
    assert marker in answer_text, "gsm8k answer must contain '####' final-answer marker"
    final_text = answer_text.rsplit(marker, 1)[1]
    number_matches = _NUMBER_RE.findall(final_text)
    if number_matches:
        return normalize_gsm8k_answer(number_matches[-1])
    return normalize_gsm8k_answer(final_text)


def extract_predicted_answer(text: str) -> tuple[str | None, str]:
    """
    Return (answer, extraction_mode).

    GSM8K standard answers use '#### <number>'. Qwen3.5 is prompted to emit the
    same marker. Fallbacks keep the script useful for models that answer in the
    common 'The answer is ...' style or only leave a final number.
    """
    marker_matches = _FINAL_MARKER_RE.findall(text)
    if marker_matches:
        candidate_numbers = _NUMBER_RE.findall(marker_matches[-1])
        if candidate_numbers:
            return normalize_gsm8k_answer(candidate_numbers[-1]), "gsm8k_marker"
        return normalize_gsm8k_answer(marker_matches[-1]), "gsm8k_marker"

    answer_matches = _THE_ANSWER_IS_RE.findall(text)
    if answer_matches:
        candidate_numbers = _NUMBER_RE.findall(answer_matches[-1])
        if candidate_numbers:
            return normalize_gsm8k_answer(candidate_numbers[-1]), "the_answer_is"
        return normalize_gsm8k_answer(answer_matches[-1]), "the_answer_is"

    number_matches = _NUMBER_RE.findall(text)
    if number_matches:
        return normalize_gsm8k_answer(number_matches[-1]), "last_number"

    return None, "none"


def grade_gsm8k_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    return normalize_gsm8k_answer(predicted) == normalize_gsm8k_answer(ground_truth)


def build_qwen_gsm8k_message(question: str) -> list[dict[str, str]]:
    question = question.strip()
    assert question, "gsm8k question must be non-empty"
    user_message = (
        "Solve the following grade-school math problem.\n\n"
        f"Question:\n{question}\n\n"
        "Reason step by step. Put the final answer on the last line in exactly this format:\n"
        "#### <number>"
    )
    return [{"role": "user", "content": user_message}]


def build_raw_gsm8k_prompt(question: str) -> str:
    question = question.strip()
    assert question, "gsm8k question must be non-empty"
    return (
        "Solve the following grade-school math problem.\n\n"
        f"Question:\n{question}\n\n"
        "Reason step by step. Put the final answer on the last line in exactly this format:\n"
        "#### <number>\n\n"
        "Solution:\n"
    )


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
    lora_adapter_path: str | None = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    enable_thinking: bool = True,
):
    print(f"Loading model with vLLM from: {base_model_path}")

    if max_model_len is None:
        max_model_len = 40960 if enable_thinking else 32768
        print(
            f"Auto-setting max_model_len to {max_model_len} for "
            f"{'thinking' if enable_thinking else 'non-thinking'} mode"
        )

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
            lora_adapter_path = None

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


def evaluate_gsm8k(
    llm,
    tokenizer,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = -1,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    num_samples: int | None = None,
    output_file: str | None = None,
    lora_request=None,
    base_model_name: str | None = None,
    enable_thinking: bool = True,
    val_n: int = 1,
    dataset_config: str = "main",
    dataset_split: str = "test",
    prompt_style: str = "chat",
):
    print(f"\n{'=' * 70}")
    print("EVALUATION CONFIGURATION")
    print(f"{'=' * 70}")
    print("Dataset: GSM8K")
    print(f"Dataset config: {dataset_config}")
    print(f"Dataset split: {dataset_split}")
    print(f"Prompt style: {prompt_style}")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Temperature: {temperature}")
    print(f"Top-P: {top_p}")
    print(f"Top-K: {top_k}")
    print(f"Min-P: {min_p}")
    print(f"Presence Penalty: {presence_penalty}")
    print(f"Max New Tokens: {max_new_tokens}")
    print(f"Val-N (solutions per problem): {val_n}")
    print(f"{'=' * 70}\n")

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", dataset_config, split=dataset_split)
    print(f"Loaded openai/gsm8k ({dataset_config}/{dataset_split}) with {len(dataset)} problems")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        n=val_n,
    )

    all_prompts = []
    all_questions = []
    all_gt_answers = []

    for example in dataset:
        question = example["question"]
        answer_text = example["answer"]
        assert isinstance(question, str), "gsm8k question must be str"
        assert isinstance(answer_text, str), "gsm8k answer must be str"

        gt_answer = extract_gsm8k_ground_truth(answer_text)

        if prompt_style == "chat":
            messages = build_qwen_gsm8k_message(question)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        elif prompt_style == "raw":
            prompt = build_raw_gsm8k_prompt(question)
        else:
            raise ValueError(f"Unknown prompt_style: {prompt_style}")

        all_prompts.append(prompt)
        all_questions.append(question)
        all_gt_answers.append(gt_answer)

    print(f"\nRunning vLLM batch inference on {len(all_prompts)} GSM8K problems...")

    print("\n" + "=" * 70)
    print("GENERATION DTYPE CHECK")
    print("=" * 70)
    print(f"Model dtype: {llm.llm_engine.model_config.dtype}")
    print(f"Quantization: {llm.llm_engine.model_config.quantization}")
    print(f"KV cache dtype: {_get_vllm_cache_dtype(llm)}")
    print(f"Using LoRA: {lora_request is not None}")
    if lora_request is not None:
        if lora_request.lora_path is None:
            raise ValueError("LoRA request exists but lora_path is None")
        print(f"LoRA path: {lora_request.lora_path}")
    print("=" * 70 + "\n")

    if lora_request is not None:
        outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

    results = []
    total = 0
    extracted_count = 0
    formatted_count = 0
    pass_at_n = 0
    total_correct_per_problem = 0

    print("\nProcessing results...")
    for idx, (output, question, gt_answer) in enumerate(zip(outputs, all_questions, all_gt_answers)):
        generations = []
        predicted_answers = []
        normalized_predictions = []
        extraction_modes = []
        is_correct_list = []
        is_formatted_list = []
        is_extracted_list = []

        for candidate in output.outputs:
            generated_text = candidate.text
            predicted_answer, extraction_mode = extract_predicted_answer(generated_text)
            normalized_pred = normalize_gsm8k_answer(predicted_answer)
            is_extracted = bool(normalized_pred)
            is_formatted = extraction_mode == "gsm8k_marker"
            is_correct = grade_gsm8k_answer(normalized_pred if is_extracted else None, gt_answer)

            generations.append(generated_text)
            predicted_answers.append(predicted_answer if predicted_answer is not None else "[No answer found]")
            normalized_predictions.append(normalized_pred)
            extraction_modes.append(extraction_mode)
            is_correct_list.append(is_correct)
            is_formatted_list.append(is_formatted)
            is_extracted_list.append(is_extracted)

        num_correct = sum(is_correct_list)
        num_formatted = sum(is_formatted_list)
        num_extracted = sum(is_extracted_list)
        has_correct = any(is_correct_list)

        majority_vote_correct = False
        majority_vote_answer = None
        extracted_predictions = [pred for pred in normalized_predictions if pred]
        if extracted_predictions:
            majority_vote_answer = Counter(extracted_predictions).most_common(1)[0][0]
            majority_vote_correct = grade_gsm8k_answer(majority_vote_answer, gt_answer)

        if has_correct:
            pass_at_n += 1
        total_correct_per_problem += num_correct
        formatted_count += num_formatted
        extracted_count += num_extracted
        total += val_n

        result = {
            "problem_id": idx,
            "question": question,
            "ground_truth": gt_answer,
            "val_n": val_n,
            "generations": [
                {
                    "predicted_answer": pred,
                    "normalized_predicted_answer": norm_pred,
                    "extraction_mode": mode,
                    "full_generation": gen,
                    "correct": corr,
                    "formatted": fmt,
                    "extracted": ext,
                }
                for pred, norm_pred, mode, gen, corr, fmt, ext in zip(
                    predicted_answers,
                    normalized_predictions,
                    extraction_modes,
                    generations,
                    is_correct_list,
                    is_formatted_list,
                    is_extracted_list,
                )
            ],
            "num_correct": num_correct,
            "pass_at_n": has_correct,
            "majority_vote_answer": majority_vote_answer,
            "majority_vote_correct": majority_vote_correct,
            "predicted_answer": predicted_answers[0],
            "normalized_predicted_answer": normalized_predictions[0],
            "extraction_mode": extraction_modes[0],
            "full_generation": generations[0],
            "correct": is_correct_list[0],
            "formatted": is_formatted_list[0],
            "extracted": is_extracted_list[0],
        }
        results.append(result)

        format_rate = formatted_count / total * 100
        extract_rate = extracted_count / total * 100
        current_pass_at_n = pass_at_n / (idx + 1) * 100
        current_avg_at_n = total_correct_per_problem / total * 100

        status = "OK" if has_correct else "--"
        print(
            f"{status} [{idx + 1}/{len(dataset)}] "
            f"Pass@{val_n}: {current_pass_at_n:.1f}% | "
            f"Avg@{val_n}: {current_avg_at_n:.1f}% | "
            f"####: {format_rate:.1f}% | Extract: {extract_rate:.1f}%"
        )

        if (idx + 1) % 10 == 0:
            print(f"\n{'=' * 70}")
            print(f"Progress: {idx + 1}/{len(dataset)}")
            print(f"Pass@{val_n}: {current_pass_at_n:.2f}%")
            print(f"Average@{val_n}: {current_avg_at_n:.2f}%")
            print(f"Format Rate (####): {format_rate:.2f}%")
            print(f"Extract Rate: {extract_rate:.2f}%")
            print(f"Solutions correct: {num_correct}/{val_n}")
            print(f"Majority vote: {'OK' if majority_vote_correct else '--'}")
            print(f"Ground truth: {gt_answer}")
            print(f"{'=' * 70}\n")

    num_problems = len(dataset)
    format_rate = formatted_count / total * 100
    extract_rate = extracted_count / total * 100
    pass_at_n_pct = pass_at_n / num_problems * 100
    average_at_n_pct = total_correct_per_problem / total * 100
    majority_vote_correct_count = sum(1 for r in results if r["majority_vote_correct"])
    majority_vote_at_n_pct = majority_vote_correct_count / num_problems * 100

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print("Dataset: GSM8K")
    print(f"Thinking Mode: {'ENABLED' if enable_thinking else 'DISABLED'}")
    print(f"Prompt style: {prompt_style}")
    print(f"Total problems: {num_problems}")
    print(f"Solutions per problem: {val_n}")
    print(f"Total solutions: {total}")
    print("\nMetrics:")
    print(f"  Pass@{val_n}: {pass_at_n_pct:.2f}% ({pass_at_n}/{num_problems})")
    print(f"  Average@{val_n}: {average_at_n_pct:.2f}% ({total_correct_per_problem}/{total})")
    print(
        f"  Majority Vote@{val_n}: "
        f"{majority_vote_at_n_pct:.2f}% ({majority_vote_correct_count}/{num_problems})"
    )
    print("\nFormatting:")
    print(f"  Formatted (####) answers: {formatted_count}/{total}")
    print(f"  Format rate: {format_rate:.2f}%")
    print(f"  Extracted answers: {extracted_count}/{total}")
    print(f"  Extract rate: {extract_rate:.2f}%")
    print("=" * 70)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "base_model": base_model_name,
            "dataset": "gsm8k",
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "prompt_style": prompt_style,
            "enable_thinking": enable_thinking,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "max_new_tokens": max_new_tokens,
            "val_n": val_n,
            "num_problems": num_problems,
            "total_solutions": total,
            "pass_at_n": pass_at_n,
            "pass_at_n_pct": pass_at_n_pct,
            "average_at_n": total_correct_per_problem,
            "average_at_n_pct": average_at_n_pct,
            "majority_vote_at_n": majority_vote_correct_count,
            "majority_vote_at_n_pct": majority_vote_at_n_pct,
            "formatted_count": formatted_count,
            "format_rate": format_rate,
            "extracted_count": extracted_count,
            "extract_rate": extract_rate,
            "results": results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_file}")

    return average_at_n_pct, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5-style models on GSM8K with vLLM")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-0.8B", help="Path to base model")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to checkpoint directory with LoRA adapters. If not provided, uses base model only.",
    )
    parser.add_argument("--dataset_config", type=str, default="main", help="GSM8K dataset config")
    parser.add_argument("--dataset_split", type=str, default="test", help="GSM8K dataset split")
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="chat",
        choices=["chat", "raw"],
        help="Use Qwen chat template or a raw completion prompt",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum tokens to generate")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=True,
        help="Enable Qwen3 thinking mode (default: True)",
    )
    parser.add_argument(
        "--no_thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable Qwen3 thinking mode",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling parameter")
    parser.add_argument("--min_p", type=float, default=0.0, help="Minimum probability threshold")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save detailed results JSON")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of tensor-parallel GPUs")
    parser.add_argument("--max_model_len", type=int, default=None, help="Maximum model context length")
    parser.add_argument("--val_n", type=int, default=1, help="Number of solutions to sample per problem")

    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        checkpoint_path = Path(args.checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"\n{'=' * 70}")
            print("ERROR: Checkpoint directory does not exist")
            print(f"{'=' * 70}")
            print(f"Provided checkpoint directory: {args.checkpoint_dir}")
            print("Please provide a valid checkpoint directory or omit --checkpoint_dir.")
            print(f"{'=' * 70}\n")
            exit(1)

    if args.top_p is None:
        args.top_p = 0.95 if args.enable_thinking else 0.8
        print(
            f"Auto-setting top_p to {args.top_p} for "
            f"{'thinking' if args.enable_thinking else 'non-thinking'} mode"
        )

    if args.enable_thinking and args.temperature == 0.0:
        print("\n" + "!" * 70)
        print("WARNING: Using greedy decoding (temperature=0.0) in thinking mode.")
        print("Qwen3 commonly recommends sampling for thinking mode.")
        print("!" * 70 + "\n")

    if args.output_file is None:
        parts = ["eval_results", "gsm8k", Path(args.base_model).name]
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            parts += [checkpoint_path.parent.name, checkpoint_path.name]
        parts += [
            args.prompt_style,
            "thinking" if args.enable_thinking else "nonthinking",
            f"temp{args.temperature}",
            f"valn{args.val_n}",
        ]
        args.output_file = str(Path("eval_results") / ("_".join(parts) + ".json"))

    print(f"Results will be saved to: {args.output_file}")

    print("\n" + "=" * 70)
    print("QWEN3.5 GSM8K EVALUATION")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint_dir or 'None (base model only)'}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Thinking Mode: {'ENABLED' if args.enable_thinking else 'DISABLED'}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Min-p: {args.min_p}")
    print(f"Presence penalty: {args.presence_penalty}")
    print(f"Num samples: {args.num_samples or 'All'}")
    print(f"Val-N: {args.val_n}")
    print(f"Output file: {args.output_file}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("=" * 70 + "\n")

    llm, tokenizer = load_vllm_model(
        args.base_model,
        args.checkpoint_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    lora_request = None
    if args.checkpoint_dir is not None:
        try:
            from vllm.lora.request import LoRARequest

            adapter_safetensors = Path(args.checkpoint_dir) / "adapter_model.safetensors"
            adapter_bin = Path(args.checkpoint_dir) / "adapter_model.bin"

            if adapter_safetensors.exists() or adapter_bin.exists():
                lora_request = LoRARequest("checkpoint_lora", 1, args.checkpoint_dir)
                print(f"Successfully created LoRA request for: {args.checkpoint_dir}")
            else:
                print(f"Warning: No LoRA adapter weights found at {args.checkpoint_dir}")
                print("Expected 'adapter_model.safetensors' or 'adapter_model.bin'")
                print("Continuing with base model only...")
        except ImportError:
            print("Warning: Could not import LoRARequest. Running without LoRA.")
        except Exception as e:
            print(f"Warning: Could not create LoRA request: {e}")
            print("Continuing without LoRA.")

    average_at_n_pct, _ = evaluate_gsm8k(
        llm,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        num_samples=args.num_samples,
        output_file=args.output_file,
        lora_request=lora_request,
        base_model_name=args.base_model,
        enable_thinking=args.enable_thinking,
        val_n=args.val_n,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        prompt_style=args.prompt_style,
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Final Average@{args.val_n}: {average_at_n_pct:.2f}%")
    print(f"Results saved to: {args.output_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
