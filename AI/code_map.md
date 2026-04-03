# Code Map

## Training Entrypoint
- `train.py`: parse config, init distributed env, load model/tokenizer, run training loop.
- `README.md`: quick install/run commands for single GPU and torchrun.

## Configs
- `configs/gdn_340m_opd.yaml`: default config for `m-a-p/340M-20B-GatedDeltaNet-pure-baseline` on FineWeb-Edu (now with `finetune_mode: lora`).
- `configs/eval/arc_ai2.yaml`: ARC(AI2) downstream evaluation config (task/checkpoint/dataset/output).
- `configs/eval/infer_text.yaml`: text inference config (checkpoint/input/generation/output).

## Core Package (`opd/`)
- `opd/config.py`: `TrainConfig` dataclass, yaml loading, fail-fast config validation.
- `opd/distributed.py`: distributed init/cleanup, barrier, cross-rank metric reduce.
- `opd/model_loader.py`: checks `flash-linear-attention` importability, loads tokenizer/model from HF with architecture assertion, and conditionally wraps model with PEFT LoRA (with fail-fast target matching and trainable-param checks).
- `opd/fineweb_data.py`: FineWeb-Edu streaming dataset, rank sharding, token packing into fixed-length chunks.
- `opd/rollout.py`: entropy-ranked Top-K prefix corruption from clean prefix (`y -> y_tilde`).
- `opd/losses.py`: shared loss primitives (`OpdLossBundle`, time-weighted JSD from logits, state-tensor alignment math for `gram_mse` / `cos_norm`).
- `opd/state_alignment.py`: stepwise OPD objective runner on FLA cache states (`recurrent_state` etc.), handling cache prefill/decode, online student continuation sampling, and aggregation of JSD + state alignment; clean/teacher branch can run on EMA teacher weights.
- `opd/checkpoint.py`: checkpoint save/load with model/EMA-model/optimizer/scheduler/scaler/RNG states.
- `opd/train_loop.py`: explicit training loop for `baseline_ce` and `opd_kl` objectives; `opd_kl` uses entropy-corrupted prefix + online continuation decoding, with optional EMA teacher (`ema_teacher_enabled`) and per-step EMA update.

## Scripts
- `scripts/run_1gpu.sh`: single-GPU convenience runner.
- `scripts/run_ngpu.sh`: multi-GPU training runner.
- `scripts/run_eval_arc.sh`: ARC(AI2) eval convenience runner.
- `scripts/run_infer_text.sh`: text inference convenience runner.
- `scripts/download_arc_ai2.py`: download/save AI2 ARC (`ARC-Easy`/`ARC-Challenge`) to local disk for offline eval.

## Eval Package (`eval/`)
- `eval/run_eval.py`: unified eval entrypoint (currently routes `task=arc_ai2`).
- `eval/run_infer.py`: text inference entrypoint (currently routes `input_format=txt`).
- `eval/run_lm_eval.py`: bridge entrypoint that runs standard `lm-eval-harness` tasks using the repo's native FLA model/checkpoint loader instead of HF auto-model loading.
- `eval/config.py`: `EvalConfig` dataclass, eval yaml loading, fail-fast validation.
- `eval/infer_config.py`: `InferConfig` dataclass, infer yaml loading, fail-fast validation.
- `eval/lm_eval_model.py`: thin `lm-eval` model wrapper that reuses `opd/model_loader.py` and local `.pt` checkpoint loading, implementing the minimal `loglikelihood` path needed by multiple-choice tasks like `arc_easy`.
- `eval/model_runtime.py`: builds model/tokenizer from train config and loads model checkpoint for eval.
- `eval/checkpoint_loader.py`: checkpoint-only model state restore for eval (`strict=True`).
- `eval/io.py`: eval output dir layout + json/jsonl writers.
- `eval/readers/text_reader.py`: inference input reader entry (currently `txt` only).
- `eval/tasks/arc_ai2/dataset.py`: AI2 ARC dataset loader (HF online or local `load_from_disk`) and normalized sample iterator.
- `eval/tasks/arc_ai2/prompt.py`: ARC question-to-prompt formatting and answer suffix construction.
- `eval/tasks/arc_ai2/scorer.py`: per-choice conditional logprob scoring from LM logits.
- `eval/tasks/arc_ai2/runner.py`: ARC eval loop over samples and choices.
- `eval/tasks/arc_ai2/metrics.py`: basic accuracy metrics aggregation.

## Memory Pollution Package (`memory_pollution/`)
- `memory_pollution/run_eval.py`: standalone entrypoint for memory-pollution experiments, separate from the training/eval stack under `opd/` and `eval/`.
- `memory_pollution/config.py`: experiment config dataclass + strict YAML validation; supports either `train_config_path` reuse or fully self-contained inline model-loader fields in `configs/memory_pollution/*.yaml`, but keeps eval path full-finetune only (no LoRA support).
- `memory_pollution/runtime.py`: runtime builder that loads the train config, resolves device, builds the requested model backend, and optionally restores a checkpoint.
- `memory_pollution/model_loader.py`: backend switch between native FLA loading (`model_impl=fla`, reusing `opd/model_loader.py`) and standard HF causal LM loading (`model_impl=hf_auto`).
- `memory_pollution/perturb.py`: deterministic random-token insertion on prompt token ids.
- `memory_pollution/scoring.py`: lm-eval-like multiple-choice continuation scoring over `log P(choice_text | prompt)`.
- `memory_pollution/state.py`: prompt cache capture plus normalized L2 state-drift computation from FLA cache states.
- `memory_pollution/metrics.py`: clean/perturb accuracy, accuracy drop, margin drop, and state-drift aggregation.
- `memory_pollution/tasks/arc.py`: ARC dataset loading, row normalization, prompt formatting, and answer continuation text construction.
- `memory_pollution/runners/arc_eval.py`: paired clean-vs-perturbed ARC evaluation loop with optional FLA state-drift extraction.

## Memory Pollution Configs
- `configs/memory_pollution/arc_gdn340m_random_tokens.yaml`: example ARC memory-pollution eval config using the FLA loader and random-token insertion.

## Dependencies
- `requirements.txt`: torch/transformers/datasets/pyyaml/accelerate/flash-linear-attention/peft.
