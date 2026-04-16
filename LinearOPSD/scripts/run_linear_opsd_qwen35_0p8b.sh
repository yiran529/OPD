#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# AI/ideas/5.md default setting:
# - Qwen/Qwen3.5-0.8B
# - open-r1/OpenThoughts-114k-math
# - conditioning_mode=linear_opsd
# - loss_mode=mixed_kl
# - alpha=1.0
# - B=1, m=2, offset=2±10, K=8
# - sampling rollout

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 4 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3.5-0.8B \
    --dataset_name open-r1/OpenThoughts-114k-math \
    --dataset_split train \
    --output_dir outputs/linear_opsd \
    --run_config qwen35_0p8b_linear_opsd_a1_b1_m2_o2pm10_k8_sample \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --max_length 20000 \
    --max_completion_length 8 \
    --save_steps 100 \
    --logging_steps 5 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --beta 0 \
    --lmbda 1 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --presence_penalty 0.0 \
    --conditioning_mode linear_opsd \
    --loss_mode mixed_kl \
    --rollout_decoding sample \
    --linear_opsd_alpha 1.0 \
    --num_corrupt_spans 1 \
    --rollout_start_offset 2 \
    --rollout_start_offset_jitter 10 \
    --corrupt_span_choices 2 \
    --corrupt_start_min_ratio 0.0 \
    --corrupt_start_max_ratio 0.5 \
    --gradient_checkpointing \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --wandb_project LinearOPSD
