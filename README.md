# OPD Training Scaffold

Minimal explicit training loop for OPD experiments on linear attention models.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run (single GPU)

```bash
python3 train.py --config configs/gdn_340m_opd.yaml
```

or

```bash
./scripts/run_1gpu.sh configs/gdn_340m_opd.yaml
```

## Run (multi GPU)

```bash
torchrun --nproc_per_node=8 train.py --config configs/gdn_340m_opd.yaml
```

## Objective switch

Set `objective` in config:
- `baseline_ce`: plain next-token CE finetune
- `opd_kl`: rollout KL + state alignment

## Finetune mode

Set `finetune_mode` in config:
- `full`: full-parameter finetune
- `lora`: LoRA adapter finetune via PEFT

LoRA config keys:
- `lora_r`, `lora_alpha`, `lora_dropout`
- `lora_target_modules`:
  - non-empty list: explicit module-name suffixes to target
  - empty list: auto-target all distinct `torch.nn.Linear` leaf names except `lm_head` (fail-fast if no Linear modules)

For `opd_kl`, state alignment is cache-based (memory state from `past_key_values`) with:
- `state_key` (default `recurrent_state`)
- `state_time_stride` (compute state MSE every N continuation steps)
- `opd_grad_through_prefix` (whether corrupted-prefix prefill keeps gradients)
