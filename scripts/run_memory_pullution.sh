CUDA_VISIBLE_DEVICES=0 python3 -m memory_pollution.run_eval --config configs/memory_pollution/arc_transformer340m_random_tokens.yaml

CUDA_VISIBLE_DEVICES=7 python3 -m memory_pollution.run_eval --config configs/memory_pollution/arc_transformer340m_random_tokens.yaml

CUDA_VISIBLE_DEVICES=5 python3 -m memory_pollution.run_eval --config configs/memory_pollution/arc_transformer1p3b_random_tokens.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m memory_pollution.run_eval --config configs/memory_pollution/lambada_openai_gdn340m_random_tokens.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m memory_pollution.run_eval --config configs/memory_pollution/lambada_openai_gdn1p3b_random_tokens.yaml

CUDA_VISIBLE_DEVICES=4 python3 -m memory_pollution.run_eval --config configs/memory_pollution/lambada_openai_transformer1p3b_random_tokens.yaml
