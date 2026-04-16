#!/bin/bash

BASE_MODEL="/data0/shared/Qwen3-1.7B"
OUTPUT_DIR="inspection_outputs"

mkdir -p "$OUTPUT_DIR"

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python inspect_linear_opsd_rollout.py \
    --base_model "$BASE_MODEL" \
    --dataset_name "open-r1/OpenThoughts-114k-math" \
    --dataset_split "train" \
    --num_examples 8 \
    --start_index 0 \
    --seed 1234 \
    --enable_thinking \
    --tensor_parallel_size 4 \
    --rollout_decoding sample \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 0 \
    --max_new_tokens 8 \
    --num_corrupt_points 1 \
    --corrupt_marker_text "<corrupt>" \
    --rollout_start_offset 2 \
    --rollout_start_offset_jitter 10 \
    --corrupt_start_min_ratio 0.0 \
    --corrupt_start_max_ratio 0.5 \
    --output_jsonl "$OUTPUT_DIR/linear_opsd_rollout_inspect_base.jsonl"
wait

# Example with a trained LoRA checkpoint:
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python inspect_linear_opsd_rollout.py \
#     --base_model "$BASE_MODEL" \
#     --checkpoint_dir /path/to/checkpoint \
#     --dataset_name "open-r1/OpenThoughts-114k-math" \
#     --dataset_split "train" \
#     --num_examples 8 \
#     --start_index 0 \
#     --seed 1234 \
#     --enable_thinking \
#     --tensor_parallel_size 4 \
#     --rollout_decoding sample \
#     --temperature 1.0 \
#     --top_p 1.0 \
#     --top_k 0 \
#     --max_new_tokens 8 \
#     --num_corrupt_points 1 \
#     --corrupt_marker_text "<corrupt>" \
#     --rollout_start_offset 2 \
#     --rollout_start_offset_jitter 10 \
#     --corrupt_start_min_ratio 0.0 \
#     --corrupt_start_max_ratio 0.5 \
#     --output_jsonl "$OUTPUT_DIR/linear_opsd_rollout_inspect_ckpt.jsonl"
# wait
