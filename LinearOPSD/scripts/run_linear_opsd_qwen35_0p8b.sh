#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# AI/ideas/6.md default setting:
# - Qwen/Qwen3.5-0.8B
# - open-r1/OpenThoughts-114k-math
# - conditioning_mode=linear_opsd
# - loss_mode=mixed_kl
# - alpha=1.0
# - gold prefix ratio in [0.3, 0.7]
# - careless prefix length = 8
# - recovery rollout length = 8

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --dataset_name open-r1/OpenThoughts-114k-math \
    --dataset_split train \
    --output_dir outputs/linear_opsd \
    --run_config qwen35_0p8b_linear_opsd_a1_gp03to07_c8_r8 \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --max_length 20000 \
    # --max_completion_length 8 \
    --save_steps 10 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --beta 0 \
    # --lmbda 1 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --presence_penalty 0.0 \
    --conditioning_mode linear_opsd \
    --loss_mode mixed_kl \
    --linear_opsd_alpha 1.0 \
    --gold_prefix_ratio_min 0.3 \
    --gold_prefix_ratio_max 0.7 \
    --careless_rollout_len 32 \
    --careless_temperature 1.5 \
    --careless_top_p 0.95 \
    --careless_top_k 50 \
    --careless_resample_trials 2 \
    --recovery_rollout_len 256 \
    --normal_decoding greedy \
    --careless_marker_text "<careless>" \
    --recovery_marker_text "<recovery>" \
    --gradient_checkpointing \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj \
    --wandb_project LinearOPSD
