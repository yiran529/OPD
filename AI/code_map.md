# Code Map

## Training Entrypoint
- `train.py`: parse config, init distributed env, load model/tokenizer, run training loop.
- `README.md`: quick install/run commands for single GPU and torchrun.
- `QwenOPSD/run_train.py`: standalone Qwen3.5 OPSD-style KD training entrypoint.
- `QwenOPSD/run_eval.py`: standalone Qwen3.5 eval entrypoint; current implementation is a placeholder scaffold.

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
- `memory_pollution/config.py`: experiment config dataclass + strict YAML validation; supports `task=arc|lambada_openai`, either `train_config_path` reuse or fully self-contained inline model-loader fields in `configs/memory_pollution/*.yaml`, and keeps eval path full-finetune only (no LoRA support).
- `memory_pollution/runtime.py`: runtime builder that loads the train config, resolves device, builds the requested model backend, and optionally restores a checkpoint.
- `memory_pollution/model_loader.py`: thin wrapper over native FLA loading (`model_impl=fla`, reusing `opd/model_loader.py`).
- `memory_pollution/perturb.py`: deterministic random-token insertion on prompt token ids.
- `memory_pollution/scoring.py`: shared joint-tokenization scorer for continuation logprob / greedy-exact evaluation, reused by ARC and `lambada_openai`.
- `memory_pollution/state.py`: prompt cache capture plus normalized L2 state-drift computation from FLA cache states.
- `memory_pollution/metrics.py`: task-level metric aggregation for ARC and `lambada_openai`, plus shared experiment metadata attachment.
- `memory_pollution/tasks/arc.py`: ARC dataset loading, row normalization, prompt formatting, and answer continuation text construction.
- `memory_pollution/tasks/lambada_openai.py`: lm-eval-aligned LAMBADA OpenAI dataset loading and `context/target` splitting via final-space split.
- `memory_pollution/runners/arc_eval.py`: paired clean-vs-perturbed ARC evaluation loop with optional FLA state-drift extraction.
- `memory_pollution/runners/lambada_openai_eval.py`: paired clean-vs-perturbed `lambada_openai` evaluation loop with exact-match/logprob outputs and optional FLA state-drift extraction.

## Memory Pollution Configs
- `configs/memory_pollution/arc_gdn340m_random_tokens.yaml`: example ARC memory-pollution eval config using the FLA loader and random-token insertion.
- `configs/memory_pollution/lambada_openai_gdn340m_random_tokens.yaml`: example `lambada_openai` memory-pollution eval config for GatedDeltaNet.
- `configs/memory_pollution/lambada_openai_transformer340m_random_tokens.yaml`: example `lambada_openai` memory-pollution eval config for Transformer.

## Exposure Bias Package (`exposure_bias/`)
- `exposure_bias/run_eval.py`: thin wrapper entrypoint that dispatches to `exposure_bias/eval/`.
- `exposure_bias/run_train.py`: thin wrapper entrypoint that dispatches to `exposure_bias/train/`.
- `exposure_bias/io.py`: shared output dir construction plus `json/jsonl` writers; eval output names are derived from dataset aliases such as `fineweb`, `scifi`, and `harrypotter` instead of the generic `hf_dataset` task name.
- `exposure_bias/model_loader.py`: thin FLA loader wrapper that applies LoRA only to Linear modules in the last `N` blocks.
- `exposure_bias/text_data.py`: HF dataset loading (remote repo, local snapshot script, or `load_from_disk` dataset), token streaming, fixed-length chunking, train dataloader, and eval example iteration.
- `exposure_bias/train/config.py`: independent training config dataclass for HF-dataset LoRA finetuning; supports generic packed-text training (`task=hf_dataset`) and GSM8K SFT formatting (`task=gsm8k_sft`).
- `exposure_bias/train/runtime.py`: loads model/tokenizer and optional init checkpoint for standalone exposure-bias finetuning.
- `exposure_bias/train/checkpoint.py`: lightweight model checkpoint saver for exposure-bias finetuning.
- `exposure_bias/train/loop.py`: minimal next-token CE training loop (single-process, no OPD logic).
- `exposure_bias/train/tasks/gsm8k.py`: GSM8K train-split formatting into `Question / Thoughts / Final Answer` SFT text and packed-token dataloader construction.
- `exposure_bias/eval/config.py`: independent eval config dataclass + YAML validation for HF-text CE-rollout eval and GSM8K thought-prefix reveal eval; `prefix_len`/`rollout_len` only matter for `task=hf_dataset` and are ignored for GSM8K reveal eval.
- `exposure_bias/eval/runtime.py`: loads model/tokenizer/checkpoint through the existing FLA loader and records model max length.
- `exposure_bias/eval/metrics.py`: aggregates `CE_TF`, `CE_rollout`, exposure-bias gap, and rollout token match rate.
- `exposure_bias/eval/scoring.py`: batched teacher-forcing CE, batched autoregressive rollout CE, and greedy generation helpers.
- `exposure_bias/eval/tasks/hf_dataset.py`: HF-text eval samples from a dataset repo or local snapshot using the configured text field.
- `exposure_bias/eval/tasks/gsm8k.py`: GSM8K rationale/final-answer parsing, thought-step splitting, reveal prompt building, and final-answer normalization.
- `exposure_bias/eval/runners/hf_dataset.py`: batched exposure-bias eval loop for HF text datasets.
- `exposure_bias/eval/runners/gsm8k_thought_reveal.py`: GSM8K reveal-ratio eval loop that greedily completes hidden thought suffixes and final answers from full untruncated prompts, with fail-fast prompt-length checks against model context length.

## Exposure Bias Configs
- `configs/exposure_bias/fineweb_edu_gdn340m.yaml`: example FineWeb-Edu exposure-bias eval config for GatedDeltaNet.
- `configs/exposure_bias/fineweb_edu_transformer340m.yaml`: example FineWeb-Edu exposure-bias eval config for Transformer.
- `configs/exposure_bias/scifi_tv_gdn340m.yaml`: example exposure-bias eval config for `lara-martin/Scifi_TV_Shows`.
- `configs/exposure_bias/harrypotter_gdn340m.yaml`: example exposure-bias eval config for `WutYee/HarryPotter_books_1to7`.
- `configs/exposure_bias/harrypotter_transformer340m.yaml`: example Harry Potter exposure-bias eval config for Transformer.
- `configs/exposure_bias/gsm8k_gdn1p3b.yaml`: GSM8K thought-prefix reveal eval config for 1.3B GatedDeltaNet.
- `configs/exposure_bias/gsm8k_transformer1p3b.yaml`: GSM8K thought-prefix reveal eval config for 1.3B Transformer.
- `configs/exposure_bias_train/scifi_tv_gdn340m_lora_last4.yaml`: example standalone HF-dataset LoRA finetune config for `Scifi_TV_Shows`.

## QwenOPSD Package (`QwenOPSD/`)
- `QwenOPSD/distributed.py`: self-contained distributed helpers for QwenOPSD training (`DistEnv`, init/cleanup, barrier, reduce_mean) so the package does not depend on `opd/`.
- `QwenOPSD/model_loader.py`: explicit Qwen3.5 loader with fail-fast class checks (`Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM`), tokenizer/chat-template setup, optional LoRA wrapping, and startup sanity forward.
- `QwenOPSD/checkpoint.py`: save/load helpers for QwenOPSD training checkpoints and model-only restore for eval.
- `QwenOPSD/io.py`: eval output dir layout plus shared json/jsonl writing helpers.
- `QwenOPSD/train/config.py`: strict YAML config dataclass for QwenOPSD training.
- `QwenOPSD/train/formatting.py`: Qwen chat-template prompt building from `problem` and raw `solution` tokenization.
- `QwenOPSD/train/data.py`: OpenThoughts math dataset loading, `problem/solution` filtering, tokenization, and sample-list dataloader construction; now supports `DistributedSampler` for DDP.
- `QwenOPSD/train/corruption.py`: multi-span corruption over `solution` tokens; samples `B` non-overlapping spans, builds student-vs-teacher prefixes (teacher gets clean patched spans), and returns rollout-start metadata.
- `QwenOPSD/train/losses.py`: forward-KL / reverse-KL / mixed-KL loss from full-vocabulary logits.
- `QwenOPSD/train/runtime.py`: builds student/teacher/tokenizer/device bundle for QwenOPSD training.
- `QwenOPSD/train/loop.py`: explicit sample-wise KD training loop with two stages: no-grad greedy student rollout first, then full student/teacher forwards on the shared rollout tokens for mixed-KL supervision; supports single-node DDP student training plus rank-0 wandb logging/checkpointing.
- `QwenOPSD/eval/config.py`: strict YAML config dataclass for QwenOPSD eval.
- `QwenOPSD/eval/runtime.py`: model/tokenizer/checkpoint builder for eval.
- `QwenOPSD/eval/runner.py`: placeholder eval runner that creates output layout and writes a placeholder metrics file.

## QwenOPSD Configs
- `configs/QwenOPSD/train/qwen3_0p8b_openthoughts.yaml`: default Qwen3.5/OpenThoughts training config for mixed-KL rollout distillation.
- `configs/QwenOPSD/eval/qwen3_0p8b_placeholder.yaml`: placeholder eval config for QwenOPSD.
- `configs/exposure_bias_train/scifi_tv_gdn1p3b_lora_last4.yaml`: example 1.3B standalone HF-dataset LoRA finetune config for `Scifi_TV_Shows`.
- `configs/exposure_bias_train/harrypotter_gdn340m_lora_last4.yaml`: example standalone HF-dataset LoRA finetune config for `WutYee/HarryPotter_books_1to7`.
- `configs/exposure_bias_train/harrypotter_gdn1p3b_lora_last4.yaml`: example 1.3B Harry Potter LoRA finetune config.
- `configs/exposure_bias_train/gsm8k_gdn1p3b_lora_last4.yaml`: GSM8K SFT LoRA config for 1.3B GatedDeltaNet.
- `configs/exposure_bias_train/gsm8k_transformer1p3b_lora_last4.yaml`: GSM8K SFT LoRA config for 1.3B Transformer.
- `scripts/compare_gsm8k_thought_reveal.py`: compares two GSM8K reveal-eval `metrics.json` files and reports `delta_Gap_25/50/75`.

## LinearOPSD
- `LinearOPSD/opsd_train.py`: upstream OPSD training entry, now extended with `conditioning_mode`, `loss_mode`, rollout decoding, corruption controls, and configurable dataset selection so the same script can run original OPSD or `linear_opsd`.
- `LinearOPSD/data_collator.py`: keeps original privileged-prompt collation for `conditioning_mode=opsd`; adds token-level `linear_opsd` collation that builds `problem + corrupted solution prefix` for student and `problem + patched prefix` for teacher.
- `LinearOPSD/opsd_trainer.py`: upstream trainer shell with original JSD/tinker paths intact; now also supports `conditioning_mode=linear_opsd`, `loss_mode=mixed_kl`, and greedy-vs-sampling rollout control while keeping the existing generation / EMA / vLLM infrastructure.

## Dependencies
- `requirements.txt`: torch/transformers/datasets/pyyaml/accelerate/flash-linear-attention/peft.
