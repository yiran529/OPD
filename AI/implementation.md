# Implementation Status (2026-03-25)

## 大致说明
当前实现的想法是：自己实现training loop（仿照一些常见框架，比如https://github.com/jiaweizzhao/GaLore/blob/master/torchrun_main.py），调用flash-linear-attention库（https://github.com/fla-org/flash-linear-attention）的模型实现，然后加载https://huggingface.co/m-a-p 中的预训练权重，在fineweb edu 100B上进行微调。
我暂时只先想用https://huggingface.co/m-a-p/340M-20B-GatedDeltaNet-pure-baseline/tree/main的pure GatedDeltaNet做初步实验。

## Scope of this iteration
- Build a minimal explicit training scaffold for OPD experiments on `m-a-p/340M-20B-GatedDeltaNet-pure-baseline`.
- Keep training logic local and inspectable, with fail-fast checks.

## Implemented components
- Entrypoint + orchestration:
  - `train.py` loads config, initializes distributed env, loads model/tokenizer, and enters training loop.
- Config and validation:
  - `opd/config.py` defines `TrainConfig` and strict YAML key/value checks.
  - `configs/gdn_340m_opd.yaml` is the default OPD experiment config.
- Model loading:
  - `opd/model_loader.py` asserts `flash-linear-attention` importability and expected architecture before training.
- Data:
  - `opd/fineweb_data.py` streams FineWeb-Edu, shards by rank, tokenizes, and packs fixed-length chunks.
- Rollout:
  - `opd/rollout.py` builds entropy-ranked Top-K corrupted prefix `y_tilde` from clean prefix `y`.
- Losses:
  - `opd/losses.py` contains OPD loss bundle, time-weighted JSD primitive, and state-tensor alignment math (`gram_mse` / `cos_norm`).
  - `opd/state_alignment.py` performs stepwise continuation decoding with two caches (corrupted/student and clean/teacher), online-samples student continuation tokens from corrupted logits, and aggregates JSD + state alignment in one serial pass.
- Training loop:
  - `opd/train_loop.py` supports:
    - `baseline_ce` (plain CE finetune),
    - `opd_kl` (entropy-corruption JSD + state alignment, with optional EMA teacher),
    - gradient accumulation, AMP autocast, grad clipping, logging, and checkpoint save.
- Checkpointing:
  - `opd/checkpoint.py` saves/loads model, optional EMA teacher model, optimizer, scheduler, scaler, and RNG states.

## Objective mapping to idea
- `x` = context segment from packed sequence.
- `y` = clean prefix segment from ground-truth sequence.
- `y_tilde` = entropy-ranked Top-K local corruption of `y` where selected positions are replaced by model argmax predictions under clean teacher forcing.
- `hat_z` = student continuation sampled online from corrupted-branch logits during stepwise decoding.
- `teacher` = EMA(student) when enabled; otherwise teacher defaults to current model stop-grad path.
- Two forward paths share the same continuation history:
  - corrupted path cache initialized by `x + y_tilde`, then decoded with `hat_z`.
  - clean path cache initialized by `x + y`, then decoded with the same `hat_z` on teacher weights (stop-grad).
- Continuation is processed token-by-token; each step jointly computes:
  - JSD between corrupted vs clean logits at the current step.
  - state alignment between corrupted vs clean memory cache states for current step; the loss form is selected by config (`gram_mse` or `cos_norm`).
- Loss:
  - `L = lambda_kl * L_jsd + lambda_state * L_state`.

## Current limitations (intentional for first pass)
- PPO/PG objective is not implemented yet (only KL path).
- Runtime dependency installation is external (`requirements.txt`), not bootstrap-automated in code.

## Validation done
- Python syntax check passed: `python3 -m compileall train.py opd`.
- CLI help run is currently blocked by missing local dependency (`pyyaml`) until environment install.

## Immediate next steps
- Run short `baseline_ce` sanity job to verify end-to-end loading/data/backward/logging.
- Run short `opd_kl` sanity job and confirm finite `loss_kl`/`loss_state`.
- Benchmark `state_time_stride` trade-offs for speed/memory vs optimization fidelity.
