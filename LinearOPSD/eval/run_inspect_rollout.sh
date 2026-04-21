#!/bin/bash

BASE_MODEL="Qwen/Qwen3.5-0.8B"
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
    --gold_prefix_ratio_min 0.3 \
    --gold_prefix_ratio_max 0.7 \
    --linear_opsd_clean_ratio 0.25 \
    --linear_opsd_mild_ratio 0.50 \
    --linear_opsd_mild_careless_rollout_len 4 \
    --careless_rollout_len 8 \
    --careless_temperature 1.3 \
    --careless_top_p 0.95 \
    --careless_top_k 50 \
    --careless_resample_trials 3 \
    --rollout_decoding greedy \
    --recovery_rollout_len 16 \
    --output_jsonl "$OUTPUT_DIR/linear_opsd_three_way_rollout_inspect_base.jsonl"
wait

# Example with a trained LoRA checkpoint. The rollout length is shared by
# problem-prefix, student-prefix, and teacher-prefix inspection rollouts.
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
#     --gold_prefix_ratio_min 0.3 \
#     --gold_prefix_ratio_max 0.7 \
#     --linear_opsd_clean_ratio 0.25 \
#     --linear_opsd_mild_ratio 0.50 \
#     --linear_opsd_mild_careless_rollout_len 4 \
#     --careless_rollout_len 8 \
#     --careless_temperature 1.3 \
#     --careless_top_p 0.95 \
#     --careless_top_k 50 \
#     --careless_resample_trials 3 \
#     --rollout_decoding greedy \
#     --recovery_rollout_len 16 \
#     --output_jsonl "$OUTPUT_DIR/linear_opsd_three_way_rollout_inspect_ckpt.jsonl"
# wait
