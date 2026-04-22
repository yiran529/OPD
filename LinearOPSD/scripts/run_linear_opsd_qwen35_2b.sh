#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# AI/ideas/6.md default setting:
# - Qwen/Qwen3.5-0.8B
# - open-r1/OpenThoughts-114k-math
# - conditioning_mode=linear_opsd
# - loss_mode=jsd
# - beta=0
# - gold prefix ratio in [0.3, 0.7]
# - clean/mild/hard prefix mixture
# - hard sampled-tail length = 8
# - recovery rollout length = 16

# --num_train_epochs 3 \
# --max_completion_length 8 \
# --lmbda 1 \
# --use_vllm \
# --vllm_mode colocate \
# --vllm_gpu_memory_utilization 0.6 \
# --vllm_tensor_parallel_size 1 \

accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path Qwen/Qwen3.5-2B \
    --dataset_name open-r1/OpenThoughts-114k-math \
    --dataset_split train \
    --output_dir outputs/linear_opsd \
    --run_config qwen35_2b_linear_opsd_r512_refineprompt2_temp1 \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --max_steps 300 \
    --max_length 20000 \
    --save_strategy steps \
    --save_steps 50 \
    --logging_steps 1 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --beta 0 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --presence_penalty 0.0 \
    --conditioning_mode linear_opsd \
    --loss_mode jsd \
    --gold_prefix_ratio_min 0.3 \
    --gold_prefix_ratio_max 0.7 \
    --linear_opsd_clean_ratio 0.25 \
    --linear_opsd_mild_ratio 0.5 \
    --linear_opsd_mild_careless_rollout_len 32 \
    --careless_rollout_len 64 \
    --careless_temperature 1.3 \
    --careless_top_p 0.95 \
    --careless_top_k 50 \
    --careless_resample_trials 2 \
    --recovery_rollout_len 512 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 256 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --rollout_decoding sample \
    --jsd_token_clip 0 \
    --loss_detail_log_steps 20 \
    --loss_detail_top_events 50 \
    --loss_detail_top_vocab 8 \
    --loss_detail_context_tokens 64 \
    --loss_detail_position_buckets 32 \
    --loss_detail_write_jsonl true \
    --wandb_project LinearOPSD \
    --report_to wandb \
    --fixed_teacher
