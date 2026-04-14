CUDA_VISIBLE_DEVICES=0 python3 -m exposure_bias.run_eval --config configs/exposure_bias/fineweb_edu_gdn1p3b.yaml

CUDA_VISIBLE_DEVICES=1 python3 -m exposure_bias.run_eval --config configs/exposure_bias/fineweb_edu_transformer1p3b.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m exposure_bias.run_eval --config configs/exposure_bias/scifi_tv_gdn1p3b.yaml
CUDA_VISIBLE_DEVICES=1 python3 -m exposure_bias.run_eval --config configs/exposure_bias/scifi_tv_transformer1p3b.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m exposure_bias.run_eval --config configs/exposure_bias/harrypotter_gdn1p3b.yaml
CUDA_VISIBLE_DEVICES=1 python3 -m exposure_bias.run_eval --config configs/exposure_bias/harrypotter_transformer1p3b.yaml
