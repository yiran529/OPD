#!/bin/bash

BASE_MODEL="Qwen/Qwen3.5-0.8B"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Evaluate base Qwen3.5 model on GSM8K with the chat template and GSM8K #### answers.
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON_BIN" evaluate_gsm8k.py \
    --base_model "$BASE_MODEL" \
    --tensor_parallel_size 4 \
    --prompt_style chat \
    --enable_thinking \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_new_tokens 2048 \
    --val_n 1
wait

# After training, uncomment and set CHECKPOINT_DIR to evaluate a LoRA checkpoint.
# CHECKPOINT_DIR="/path/to/checkpoint"
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON_BIN" evaluate_gsm8k.py \
#     --base_model "$BASE_MODEL" \
#     --checkpoint_dir "$CHECKPOINT_DIR" \
#     --tensor_parallel_size 4 \
#     --prompt_style chat \
#     --enable_thinking \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --max_new_tokens 2048 \
#     --val_n 1
# wait
