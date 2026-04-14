CUDA_VISIBLE_DEVICES=0 python3 -m exposure_bias.run_train --config configs/exposure_bias_train/scifi_tv_gdn1p3b_lora_last4.yaml
CUDA_VISIBLE_DEVICES=1 python3 -m exposure_bias.run_train --config configs/exposure_bias_train/scifi_tv_transformer1p3b_lora_last4.yaml
CUDA_VISIBLE_DEVICES=0 python3 -m exposure_bias.run_train --config configs/exposure_bias_train/harrypotter_gdn1p3b_lora_last4.yaml
CUDA_VISIBLE_DEVICES=1 python3 -m exposure_bias.run_train --config configs/exposure_bias_train/harrypotter_transformer1p3b_lora_last4.yaml
CUDA_VISIBLE_DEVICES=0 python3 -m exposure_bias.run_train --config configs/exposure_bias_train/gsm8k_gdn1p3b_lora_last4.yaml
CUDA_VISIBLE_DEVICES=1 python3 -m exposure_bias.run_train --config configs/exposure_bias_train/gsm8k_transformer1p3b_lora_last4.yaml
