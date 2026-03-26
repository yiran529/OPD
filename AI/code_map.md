# Code Map

## Training Entrypoint
- `train.py`: parse config, init distributed env, load model/tokenizer, run training loop.
- `README.md`: quick install/run commands for single GPU and torchrun.

## Configs
- `configs/gdn_340m_opd.yaml`: default config for `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` on FineWeb-Edu (now with `finetune_mode: lora`).

## Core Package (`opd/`)
- `opd/config.py`: `TrainConfig` dataclass, yaml loading, fail-fast config validation.
- `opd/distributed.py`: distributed init/cleanup, barrier, cross-rank metric reduce.
- `opd/model_loader.py`: checks `flash-linear-attention` importability, loads tokenizer/model from HF with architecture assertion, and conditionally wraps model with PEFT LoRA (with fail-fast target matching and trainable-param checks).
- `opd/fineweb_data.py`: FineWeb-Edu streaming dataset, rank sharding, token packing into fixed-length chunks.
- `opd/rollout.py`: rollout model sync and fixed-length generation of `hat_y` and `z`.
- `opd/losses.py`: shared loss primitives (`OpdLossBundle`, KL from logits).
- `opd/state_alignment.py`: stepwise OPD loss on FLA cache states (`recurrent_state` etc.), computing KL + state alignment loss in one serial continuation pass.
- `opd/checkpoint.py`: checkpoint save/load with optimizer/scheduler/scaler/RNG states.
- `opd/train_loop.py`: explicit training loop for `baseline_ce` and `opd_kl` objectives; optimizer/grad-clip operate only on trainable params (full or LoRA).

## Scripts
- `scripts/run_1gpu.sh`: single-GPU convenience runner.

## Dependencies
- `requirements.txt`: torch/transformers/datasets/pyyaml/accelerate/flash-linear-attention/peft.
