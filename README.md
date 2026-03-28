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
- `state_time_stride` (compute state alignment loss every N continuation steps)

## ARC eval (AI2 ARC)

Default eval config:

```bash
configs/eval/arc_ai2.yaml
```

Run eval:

```bash
python3 -m eval.run_eval --config configs/eval/arc_ai2.yaml
```

or

```bash
./scripts/run_eval_arc.sh configs/eval/arc_ai2.yaml
```

Download ARC to local disk (optional, for offline eval):

```bash
python3 scripts/download_arc_ai2.py --config ARC-Challenge --output_dir data/arc/ai2_arc/ARC-Challenge
```

Then set in eval config:
- `local_dataset_path: data/arc/ai2_arc/ARC-Challenge`
- keep `dataset_split` as `train|validation|test`

## Text inference

Default infer config:

```bash
configs/eval/infer_text.yaml
```

Run infer:

```bash
python3 -m eval.run_infer --config configs/eval/infer_text.yaml
```

or

```bash
./scripts/run_infer_text.sh configs/eval/infer_text.yaml
```

Input format currently supports `txt` (one line per sample after optional strip/filter).
