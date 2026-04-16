import os
import wandb

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


QWEN35_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]


def _patch_qwen35_lora_targets(model_args):
    if not getattr(model_args, "use_peft", False):
        return

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if getattr(config, "model_type", None) != "qwen3_5":
        return

    current_targets = getattr(model_args, "lora_target_modules", None)
    if current_targets is None:
        current_targets = []
    elif isinstance(current_targets, str):
        current_targets = [item.strip() for item in current_targets.split(",") if item.strip()]
    else:
        current_targets = [str(item).strip() for item in current_targets if str(item).strip()]

    if not current_targets:
        model_args.lora_target_modules = list(QWEN35_LORA_TARGETS)
        print(
            "Qwen3.5 detected. Setting LoRA target modules to: "
            + ", ".join(model_args.lora_target_modules)
        )
        return

    missing_targets = [target for target in QWEN35_LORA_TARGETS if target not in current_targets]
    if missing_targets:
        model_args.lora_target_modules = list(current_targets) + missing_targets
        print(
            "Qwen3.5 detected. Appended missing LoRA target modules: "
            + ", ".join(missing_targets)
        )


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    conditioning_mode: str = field(
        default="opsd",
        metadata={
            "help": "Prompt-conditioning path. `opsd` keeps the original privileged-prompt setup; "
            "`linear_opsd` uses trainer-time entropy-based corruption with privileged teacher traces."
        },
    )
    loss_mode: str = field(
        default="jsd",
        metadata={
            "help": "Distillation loss path. Supported values: `jsd`, `mixed_kl`. "
            "`use_tinker_loss=True` remains as a separate legacy switch."
        },
    )
    rollout_decoding: str = field(
        default="sample",
        metadata={"help": "On-policy rollout decoding mode: `sample` or `greedy`."},
    )
    linear_opsd_alpha: float = field(
        default=1.0,
        metadata={"help": "Forward/reverse KL mixing coefficient used by `loss_mode=mixed_kl`."},
    )
    num_corrupt_points: int = field(
        default=1,
        metadata={"help": "Number of entropy-selected corruption points for `conditioning_mode=linear_opsd`."},
    )
    corrupt_marker_text: str = field(
        default="<corrupt>",
        metadata={"help": "Inline marker inserted before each corrupted token in the teacher-visible trace."},
    )
    corrupt_start_min_ratio: float = field(
        default=0.0,
        metadata={"help": "Minimum corruption start ratio within the solution for `linear_opsd`."},
    )
    corrupt_start_max_ratio: float = field(
        default=0.5,
        metadata={"help": "Maximum corruption start ratio within the solution for `linear_opsd`."},
    )
    rollout_start_offset: int = field(
        default=2,
        metadata={"help": "Number of clean tokens kept after the last corruption point before rollout starts."},
    )
    rollout_start_offset_jitter: int = field(
        default=10,
        metadata={"help": "Maximum absolute random jitter added to rollout_start_offset for each sample."},
    )
    dataset_name: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd",
        metadata={"help": "Dataset name/path passed to `datasets.load_dataset`."},
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split used for training."},
    )

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    _patch_qwen35_lora_targets(model_args)

    assert script_args.conditioning_mode in {"opsd", "linear_opsd"}, (
        f"Unsupported conditioning_mode={script_args.conditioning_mode}"
    )
    assert script_args.loss_mode in {"jsd", "mixed_kl"}, f"Unsupported loss_mode={script_args.loss_mode}"
    assert script_args.rollout_decoding in {"sample", "greedy"}, (
        f"Unsupported rollout_decoding={script_args.rollout_decoding}"
    )
    assert 0.0 <= script_args.linear_opsd_alpha <= 1.0, (
        f"linear_opsd_alpha must be in [0, 1], got {script_args.linear_opsd_alpha}"
    )
    assert script_args.num_corrupt_points > 0, "num_corrupt_points must be positive"
    assert script_args.rollout_start_offset >= 0, "rollout_start_offset must be non-negative"
    assert script_args.rollout_start_offset_jitter >= 0, "rollout_start_offset_jitter must be non-negative"
    assert 0.0 <= script_args.corrupt_start_min_ratio <= script_args.corrupt_start_max_ratio <= 1.0, (
        "corrupt_start ratios must satisfy 0 <= min <= max <= 1"
    )
    assert script_args.corrupt_marker_text.strip(), "corrupt_marker_text must be non-empty"
    if script_args.use_tinker_loss:
        assert script_args.loss_mode == "jsd", (
            "use_tinker_loss is a legacy alternative loss path and cannot be combined with loss_mode!=jsd"
        )
    if script_args.conditioning_mode == "linear_opsd":
        assert not script_args.reason_first, "reason_first is incompatible with conditioning_mode=linear_opsd"

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]
        run_prefix = "linear_opsd" if script_args.conditioning_mode == "linear_opsd" else "opsd"

        # Create concise run name
        full_wandb_run_config = (
            f"{run_prefix}_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "conditioning_mode": script_args.conditioning_mode,
                "loss_mode": "tinker" if script_args.use_tinker_loss else script_args.loss_mode,
                "rollout_decoding": script_args.rollout_decoding,
                "linear_opsd_alpha": script_args.linear_opsd_alpha,
                "num_corrupt_points": script_args.num_corrupt_points,
                "corrupt_marker_text": script_args.corrupt_marker_text,
                "rollout_start_offset": script_args.rollout_start_offset,
                "rollout_start_offset_jitter": script_args.rollout_start_offset_jitter,
                "corrupt_start_min_ratio": script_args.corrupt_start_min_ratio,
                "corrupt_start_max_ratio": script_args.corrupt_start_max_ratio,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    dataset = load_dataset(script_args.dataset_name)
    train_dataset = dataset[script_args.dataset_split]

    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        conditioning_mode=script_args.conditioning_mode,
        loss_mode=script_args.loss_mode,
        rollout_decoding=script_args.rollout_decoding,
        linear_opsd_alpha=script_args.linear_opsd_alpha,
        num_corrupt_points=script_args.num_corrupt_points,
        corrupt_marker_text=script_args.corrupt_marker_text,
        rollout_start_offset=script_args.rollout_start_offset,
        rollout_start_offset_jitter=script_args.rollout_start_offset_jitter,
        corrupt_start_min_ratio=script_args.corrupt_start_min_ratio,
        corrupt_start_max_ratio=script_args.corrupt_start_max_ratio,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=script_args.rollout_decoding == "sample",
            temperature=training_args.temperature if script_args.rollout_decoding == "sample" else 1.0,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)
