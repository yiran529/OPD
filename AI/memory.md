# Durable Memory
* note: 务必按照时间顺序写！！！

## 2026-03-25-11:00 : Initial OPD training scaffold
- The training pipeline is intentionally explicit and local: `train.py` + `opd/train_loop.py` implement the full loop without framework abstractions.
- Two objectives are supported with config switch:
  - `baseline_ce`: plain next-token CE finetuning.
  - `opd_kl`: rollout-based KL distillation with state alignment.
- In the first OPD implementation, state alignment uses `hidden_states[-1]` on continuation positions as the state representation.
  - This is a deliberate proxy until a stable public recurrent-state interface is exposed by the upstream model.
  - The code fails fast if `hidden_states` are not returned.
- `theta_old` is implemented as a separate frozen rollout model, synchronized every `rollout_sync_steps` optimizer steps.
- FineWeb-Edu data path uses streaming + token packing into fixed-length chunks of `context_len + prefix_len + continuation_len + 1`.
- Checkpoints persist model, rollout model, optimizer, scheduler, scaler, and RNG states for reproducible resume.


## 2026-03-25-15:00 : Problem/TODO
- [x] state应该是用Linear attention中的memory state，而不是last hidden states
- [x] 检查：是否使用了FLA库（https://github.com/fla-org/flash-linear-attention），是否正确使用了FLA库中的gated deltanet，权重加载是否正确
- [x] 是否真的会用到FLA中的算子？
- [x] 应该用LoRA 
- [] training script是不是仿照框架(比如 https://github.com/jiaweizzhao/GaLore/blob/master/torchrun_main.py)写的
- [] 应该用混合AR训练吗
- [x] theta_old是什么
- [] max_length/chunking
- [x] 应该切断rollout的tokens之间的梯度吗
- [x] 应该赋予接近corrupted tokens的loss低一些的权重？state MSE是不是太死了？
- [] KL方向


## 2026-03-25-16:35 : FLA/GatedDeltaNet loading hardening
- `opd/model_loader.py` now performs strict startup checks for FLA + GatedDeltaNet loading.
- Model loading uses `output_loading_info=True` and fails fast if any of: `missing_keys`, `unexpected_keys`, `mismatched_keys`, `error_msgs` are non-empty.
- Runtime now asserts loaded model implementation is actually from `fla.models.gated_deltanet` and class name matches expected architecture.
- Added startup sanity forward with `use_cache=True`; it fails fast if logits are non-finite, `past_key_values` is missing/empty, or per-layer cache lacks non-None `recurrent_state`.
- Startup logs now include `fla` version, `transformers` version, actual model class/module, and config model metadata to improve reproducibility/debugging.
## 2026-03-25-16:40 : Generalize strict checks to all FLA models
- Validation is no longer GatedDeltaNet-only; it now supports all FLA model families while keeping fail-fast behavior.
- Model implementation check now allows FLA/remote-code modules but requires class name to be exported by `fla`.
- Cache sanity check is generalized from `recurrent_state`-only to FLA cache state union: `recurrent_state|attn_state|conv_state|ffn_state` (at least one non-None).


## 2026-03-25-17:20 : Stepwise memory-state OPD loss
- `opd_kl` no longer uses `hidden_states[-1]` for state alignment.
- State alignment now reads memory state directly from FLA cache (`past_key_values`) with `state_key` (default `recurrent_state`).
- Continuation supervision is now a single serial loop over `z`: each step computes both
  - KL(logits_corrupted || logits_clean_stopgrad), and
  - MSE(memory_state_corrupted, memory_state_clean_stopgrad).
- Added config knobs:
  - `state_key` (default `recurrent_state`)
  - `state_time_stride` (default `1`)
- Implementation is fail-fast for cache/state structure mismatch (missing key, `None` state, layer/state shape mismatch).

## 2026-03-25-18:05 : Check on FLA accelerated operators usage
- Verified from upstream source (`4225ff950afdda0125d318567379ea17bcdbb3be`) that GatedDeltaNet layer calls `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` from `fla.ops.gated_delta_rule`.
- Verified from upstream source that training mode for GatedDeltaNet enforces `chunk` mode, i.e. training should use `chunk_gated_delta_rule` path.
- This check fetched upstream files via pipe-only commands (`curl | sed/rg`) and did not write temporary source files to disk.


## 2026-03-25-18:20 : Why AutoModel still resolves to FLA model
- Even though model construction uses `AutoModelForCausalLM`, loading is constrained to FLA/remote implementation by:
  - `trust_remote_code=True` with expected architecture check (`config.architectures`),
  - post-load class/module assertions (`fla.*` or `transformers_modules.*`),
  - export check that loaded class exists in `fla`,
  - startup cache sanity requiring FLA-style cache state keys.
- Therefore this path fails fast if model resolution drifts away from FLA implementation.


## 2026-03-25-19:05 : LoRA finetune mode (PEFT) for FLA models
- Added config switch `finetune_mode` with two explicit paths:
  - `full`: full-parameter finetune.
  - `lora`: PEFT LoRA adapter finetune.
- LoRA is applied as a thin wrapper in `opd/model_loader.py` after FLA preload checks, without modifying third-party FLA code.
- LoRA target selection is fail-fast:
  - if `lora_target_modules` is provided, every entry must match at least one `torch.nn.Linear` module suffix;
  - if empty, targets auto-resolve to all distinct Linear leaf names in the loaded model.
- LoRA mode enforces trainable-parameter discipline:
  - startup asserts trainable params are LoRA adapter params only;
  - optimizer and grad clipping operate only on `requires_grad=True` params.

## 2026-03-25-19:20 : Exclude output head from auto LoRA targets
- When `lora_target_modules` is empty (auto mode), LoRA target inference now excludes `lm_head` by default.
- If users want LoRA on output head explicitly, they can still provide it via non-empty `lora_target_modules`.

## 2026-03-25-19:35 : Comparison with GaLore torchrun_main.py training flow
- Checked against `https://github.com/jiaweizzhao/GaLore/blob/master/torchrun_main.py`.
- Conclusion: our script follows the same high-level training skeleton (distributed init, dataloader, optimizer/scheduler, grad accumulation, clipping, logging, checkpoint), but it is not a direct copy.
- Key difference is intentional OPD logic: frozen `theta_old` rollout model, periodic rollout sync, and clean-vs-corrupted KL/state losses.
- Therefore, similarity to GaLore improves engineering confidence, but correctness still depends on OPD-specific rollout/loss validation.

## 2026-03-26-10:10 : Hard-cut gradient flow in stepwise OPD state alignment
- `compute_stepwise_opd_losses` now hard-codes stop-grad through corrupted-prefix prefill (`x + y_hat`) to prevent gradient flow into prefix history.
- During continuation decoding, corrupted cache is detached every step before reuse, so gradients from `z_t` no longer backprop through `z_{<t}`.
- The goal is to avoid recurrent graph chaining and reduce memory blow-up risk while keeping per-step KL/state supervision.

## 2026-03-26-10:30 : Replace state MSE with time-weighted cosine+norm objective
- State alignment term in `opd/state_alignment.py` no longer uses MSE.
- Per-step state loss is now:
  - `w=((t+1)/T)^2`
  - `cos_loss=1-cosine_similarity(a,b)`
  - `norm_loss=(||a||-||b||)^2`
  - `loss_state_t = w * (cos_loss + 0.01 * norm_loss)`, averaged across all state tensors in all layers.
- `b` (clean path state) remains stop-grad (`detach`) and only corrupted-path state receives gradients.

## 2026-03-26-11:00 : Remove CE anchor branch from OPD objective
- Deleted `ce_anchor` from `OpdLossBundle` and removed `ce_from_logits` usage from `opd/state_alignment.py`.
- `compute_stepwise_opd_losses` now returns only `total/kl/state`, with `total = kl + lambda_state * state`.
- Removed `ce_anchor_weight` from config and default YAML, and removed CE metrics/logging from `opd/train_loop.py`.
- This aligns implementation with the original KL + state design in `AI/ideas/1.md`.

## 2026-03-26-11:20 : Remove stale `opd_grad_through_prefix` knob
- `opd_grad_through_prefix` was stale after gradient-through-prefix was hard-disabled in `opd/state_alignment.py`.
- Removed this knob from `TrainConfig`, default YAML, and docs to avoid configuration illusion.

## 2026-03-26-14:55 : Load FLA checkpoints via exported FLA model class (avoid AutoConfig model_type trap)
- `opd/model_loader.py` no longer uses `AutoConfig.from_pretrained` / `AutoModelForCausalLM.from_pretrained` for startup.
- Loader now resolves `expected_architecture` directly from `fla` exports and calls that class's `from_pretrained`.
- This avoids failures when checkpoint `model_type` (e.g. `gated_deltanet`) is not registered in local `transformers` `CONFIG_MAPPING`.
- Architecture consistency check is still fail-fast via `model.config.architectures` + runtime class/module assertions.

## 2026-03-26-15:05 : FLA 模型定义与加载路径（简要）
- 训练入口是 `main()` 调用 `build_model_and_tokenizer(cfg=cfg, device=dist_env.device)`。
- FLA 模型定义来源于已安装 `flash-linear-attention` 包；加载时先执行 `_ensure_flash_linear_attention_importable()`，再 `import fla`。
- 模型类由配置项 `expected_architecture` 决定，解析语句是 `model_class = getattr(fla_module, cfg.expected_architecture)`。
- 权重加载语句是 `model, loading_info = model_class.from_pretrained(model_id, torch_dtype=model_dtype, output_loading_info=True)`。
- tokenizer 加载语句是 `AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=cfg.trust_remote_code, use_fast=True)`，其中 `tokenizer_id = cfg.tokenizer_name or model_id`。
- 加载后保持 fail-fast：`_assert_expected_model_impl(...)`、`_assert_clean_weight_loading(...)`，并执行 `_run_startup_sanity(...)` 检查 logits 与 FLA cache state。

## 2026-03-26-15:20 : Robust FLA class resolution + transformers<5 pin
- Some environments expose `import fla` but do not export model classes (e.g., `GatedDeltaNetForCausalLM`) at top-level, so resolving by `hasattr(fla, expected_architecture)` is too strict.
- `opd/model_loader.py` now resolves expected class by trying multiple canonical modules:
  - `fla`
  - `fla.models`
  - `fla.models.gated_deltanet`
  - `fla.models.gated_deltanet.modeling_gated_deltanet`
- The loader no longer requires loaded class to be re-exported at `fla` top-level; runtime class/module + weight-loading assertions remain fail-fast.
- Pinned `transformers` to `<5` in `requirements.txt` to avoid FLA compatibility drift with 5.x.

## 2026-03-26-15:35 : Allow known checkpoint-only `attn.D` keys while keeping strict load checks
- For `m-a-p/340M-20B-GatedDeltaNet-pure-baseline`, loading can report `unexpected_keys` like `model.layers.<i>.attn.D` across all layers.
- `opd/model_loader.py` keeps strict fail-fast behavior for `missing_keys`, `mismatched_keys`, `error_msgs`, and any other unexpected keys.
- Only `unexpected_keys` matching `model.layers.<i>.attn.D` are whitelisted and logged as ignored checkpoint-only keys.

## 2026-03-26-16:05 : State detach now follows FLA official cache definition only
- From FLA upstream `gated_deltanet` implementation, `past_key_values` is `fla.models.utils.Cache` (legacy tuple/list inputs are converted into this type in forward).
- `opd/state_alignment.py` `_detach_tree` now supports official FLA `Cache` via `to_legacy_cache` -> recursive tensor detach -> `Cache.from_legacy_cache`.
- Removed broader generic container fallback to keep behavior explicit and fail-fast for non-FLA cache types.

## 2026-03-26-16:15 : FLA legacy cache detach must allow `None` leaves
- `Cache.to_legacy_cache()` may include `None` entries for inactive state slots.
- `_detach_tree` now treats `None` as a valid leaf (`None -> None`) before tensor/container recursion.

## 2026-03-26-16:55 : Objective-aware packed length for dataloader
- `opd_kl` no longer packs unused GT continuation tokens in streaming chunks.
- `TrainConfig.sequence_length/sequence_plus_one` are now objective-aware directly (`opd_kl`: `context+prefix`; `baseline_ce`: `context+prefix+continuation`).
- `opd/fineweb_data.py` packs by `sequence_*`; `opd/train_loop.py` uses OPD-specific split (`context + clean_prefix`) and objective-aware token throughput accounting.

## 2026-03-26-17:20 : Add small-sample sanity switches for overfit checks
- Added `sanity_num_samples` (default `0`) and `sanity_disable_shuffle` (default `false`) in `TrainConfig`.
- `opd/fineweb_data.py` now supports collecting a fixed number of packed samples and cycling them indefinitely, plus skipping shuffle for deterministic small-data sanity runs.
- Defaults keep original training behavior unchanged.

## 2026-03-27-12:00 : Problem/TODO
- [] 显存占用大，且计算慢
- [] 只需要state对齐子空间，或者SVD之类的？gram矩阵？
- [x] 别的前缀构成方式
- [] 可以放松t较小时KL loss的权重

## 2026-03-27-20:20 : Remove theta_old and switch to dual-rollout OPD
- `opd_kl` no longer keeps a frozen rollout copy (`theta_old`) or periodic rollout sync.
- Rollout sampling now uses the current model in inference mode along two natural trajectories:
  - corrupted path from `x`: produce `hat_y` then `hat_z`,
  - clean path from `x + y`: produce `z`.
- `compute_stepwise_opd_losses` now consumes two continuation token streams (`hat_z`, `z`) instead of a shared continuation.
- Checkpoint payload removed `rollout_model`; only model/optimizer/scheduler/scaler/RNG are persisted.
- Removed stale config knob `rollout_sync_steps` to avoid configuration illusion.

## 2026-03-28-00:20 : Time-weighting schedule softened + decoder-only generate masking fix
- Time weighting in both KL and state alignment was changed from quadratic to linear:
  - from `w_t=((t+1)/T)^2`
  - to `w_t=((t+1)/T)`
- For decoder-only rollout stability/warning mitigation:
  - tokenizer now sets `padding_side="left"` in `opd/model_loader.py`.
  - `model.generate(...)` in `opd/rollout.py` now passes explicit all-ones `attention_mask` for both corrupted and clean prompts.
- This makes generation no longer depend on implicit padding inference when `pad_token_id` may equal `eos_token_id`.

## 2026-03-28-09:30 : Eval layout flattening + task-local organization
- Eval shared utilities are flattened directly under `eval/` (no `eval/common/`) to keep code explicit and small.
- Task-specific logic lives under `eval/tasks/<task_name>/` (first task: `arc_ai2`) to isolate dataset/prompt/scoring differences.
- Eval outputs are standardized at `outputs/<run_name>/eval/<task>/<checkpoint_tag>/`.

## 2026-03-28-10:05 : PyTorch 2.6 eval checkpoint loading compatibility
- In `eval/checkpoint_loader.py`, `torch.load` now explicitly sets `weights_only=False`.
- Reason: PyTorch 2.6 changed `torch.load` default to `weights_only=True`, which breaks loading our full training checkpoints containing non-tensor RNG states.

## 2026-03-28-10:30 : Inference placed under eval with reusable runtime
- Text inference is implemented under `eval/` (not a separate top-level package) to reuse existing model/checkpoint/output helpers.
- Shared logic remains flat under `eval/`; format-specific input handling is isolated in `eval/readers/*`.
- Current supported `input_format` is `txt`; future formats (e.g., parquet) should be added by registering new readers only.

## 2026-03-28-15:00
- ARC跑的效果较差
- 用文本推理，发现是被hack了

## 2026-03-29-11:20 : Switch OPD objective to idea-3 style prefix corruption + shared-rollout JSD
- `opd_kl` objective key is kept unchanged for config compatibility, but training semantics changed:
  - student prefix is no longer rollout-generated;
  - corrupted prefix is built from ground-truth prefix by entropy-ranked Top-K token replacement (`y -> y_tilde`);
  - teacher path no longer performs an independent rollout.
- Student continuation is a single rollout `hat_z` from `x + y_tilde`; both student and teacher branches decode under shared history `hat_z_{<t}`.
- Main distribution loss changed from time-weighted KL to time-weighted JSD in `opd/losses.py`.
- Prefix corruption controls are explicit and fail-fast validated in config:
  - `prefix_corrupt_topk_ratio` (default `0.25`)
  - `prefix_corrupt_topk_max` (default `64`)
- State alignment stays linear time-weighted with `cos + 0.1 * norm` as current default experiment choice.

## 2026-03-29-15:30 : Merge student rollout into stepwise loss decode loop
- Removed the separate pre-rollout pass that generated full `student_z_tokens` before loss computation.
- `compute_stepwise_opd_losses` now samples student tokens online from corrupted-branch logits at each step (`rollout_temperature` / `rollout_top_p`) and immediately decodes both branches with that token.
- This keeps the same shared-history training semantics while reducing duplicated decode work (one continuation pass instead of rollout pass + loss pass).
- Deleted unused rollout helper `generate_student_rollout_tokens` from `opd/rollout.py`.

## 2026-03-29-17:20 : Optional EMA teacher for clean branch supervision
- Added optional EMA teacher path in training (`ema_teacher_enabled`, `ema_decay`, `ema_start_step`).
- When enabled, clean/teacher branch logits and cache states in `compute_stepwise_opd_losses` are computed from EMA model weights instead of current student weights.
- EMA teacher is initialized as a deepcopy of student and updated each optimizer step with decay, only for trainable student parameters; buffers are copied from student.
- Checkpoint payload now includes `ema_model` when EMA teacher is enabled; resume fails fast if EMA is enabled but checkpoint lacks `ema_model`.
- Default experiment config `configs/gdn_340m_opd.yaml` enables EMA teacher with `ema_decay=0.999`.


## 2026-03-31-00:00 : Replace state alignment with Gram-matrix MSE
- `opd/state_alignment.py` no longer uses `cos + 0.1 * norm` for state alignment.
- For each state tensor, the code now reshapes it into matrix slices over the last two dimensions, computes `Gram(x) = x x^T`, and applies `MSE(Gram(a), Gram(b))`.
- Clean-path state remains stop-grad; continuation-level state loss still uses the existing linear time weight `w_t=((t+1)/T)`.

## 2026-03-31-00:10 : Make state alignment loss selectable by config
- Added `state_align_loss` config with two supported values:
  - `gram_mse`: `MSE(Gram(a), Gram(b))`
  - `cos_norm`: legacy `1 - cos(a,b) + 0.1 * (||a|| - ||b||)^2`
- `compute_stepwise_opd_losses` now routes state alignment through this config while keeping the rest of the OPD training semantics unchanged.

## 2026-03-31-00:20 : Move pure loss math into `opd/losses.py`
- `opd/losses.py` now owns the pure state-alignment tensor losses (`gram_mse`, `cos_norm`) in addition to JSD.
- `opd/state_alignment.py` keeps only cache/state traversal, stepwise decode, and per-step aggregation logic.
- This keeps the boundary explicit: tensor-to-tensor loss math in `losses.py`, rollout/cache execution in `state_alignment.py`.

## 2026-03-31-00:30 : Add explicit `lambda_kl` weight for OPD total loss
- Added `lambda_kl` config (default `1.0`) so total OPD loss is now `lambda_kl * loss_kl + lambda_state * loss_state`.
- Logged `loss_kl` remains the raw unweighted JSD term; only `loss_total` reflects the weighting.


## 2026-03-31-01:00 : Eval/Infer can run from pretrained weights without finetune checkpoint
- `eval` / `infer` config `checkpoint_path` is now optional.
- When `checkpoint_path` is empty/null, runtime skips `load_model_checkpoint(...)` and evaluates/infers directly from `train_config.model_name` pretrained weights.
- Output directory checkpoint tag falls back to `pretrained` when no checkpoint path is provided.

## 2026-03-31-01:10 : ARC eval now scores choice text instead of choice label
- `eval/tasks/arc_ai2` no longer compares `P(" A" | prompt)` / `P(" B" | prompt)` style answer-label probabilities.
- The scorer now compares the conditional logprob of each choice's actual text continuation after `Answer:`.
- Prediction output still reports `pred_label` / `gold_label`; only the scoring continuation changed.

## 2026-04-02-10:30 : Separate memory-pollution experiment package
- Added a new top-level package `memory_pollution/` so the memory-pollution benchmark code stays isolated from the OPD training stack under `opd/` and the older downstream eval helpers under `eval/`.
- The first implementation supports ARC multiple-choice evaluation with deterministic random-token insertion perturbations and optional FLA cache state-drift measurement.
- Model loading is explicit by backend:
  - `model_impl=fla` reuses the strict FLA loader from `opd/model_loader.py`,
  - transformer / hybrid / linear checkpoints should all use the FLA backend, with the concrete model class selected by `expected_architecture`.
- Multiple-choice scoring in this package follows the lm-eval-style continuation objective `log P(choice_text | prompt)`.
- For ARC specifically, the prompt is aligned to lm-eval's official task format:
  - `prompt = "Question: <question>\nAnswer:"`
  - each answer option text is scored as the continuation.
- State drift uses normalized L2 distance on the requested FLA cache `state_key`; different FLA model families may require different keys (for example `recurrent_state` vs `attn_state`), and unsupported keys should fail fast.

## 2026-04-02-11:10 : Memory-pollution eval uses FLA backend for transformer too
- `memory_pollution/` no longer keeps a separate HF-auto loader path.
- Transformer baselines should use the FLA transformer implementation (`fla.models.transformer.TransformerForCausalLM`) so transformer / hybrid / linear all load through the same strict FLA loader and cache interface.
- `opd/model_loader.py` class resolution was widened beyond GatedDeltaNet to also probe FLA transformer, GLA, RetNet, HGRN/HGRN2, DeltaNet, and GatedDeltaNet module paths.

## 2026-03-31-02:00 : Add local `lm-eval-harness` bridge for FLA models
- Added `eval/run_lm_eval.py` and `eval/lm_eval_model.py` to run standard `lm-eval-harness` tasks without modifying `lm-eval` source code.
- The bridge does **not** use `lm-eval`'s default HF auto-model loader, because the local FLA `gated_deltanet` checkpoints are not directly loadable via `AutoConfig` / `AutoModel`.
- Instead, it reuses the repo's own `build_model_and_tokenizer(...)` path and the local `.pt` checkpoint format, with optional `ema_model` loading for eval.
- The first implementation is intentionally narrow:
  - supports only `loglikelihood`-based tasks;
  - `generate_until` and `loglikelihood_rolling` are not implemented.

## 2026-04-01-00:20 : Enable `batch_size>1` for lm-eval loglikelihood bridge
- `eval/lm_eval_model.py` now supports batched loglikelihood scoring (padding + attention_mask + per-sample continuation masks) for `batch_size>1`.
- The bridge still intentionally remains loglikelihood-only for now: `generate_until` and `loglikelihood_rolling` are still not implemented.

## 2026-04-01-00:45 : Add batched `generate_until` support to lm-eval bridge
- `eval/lm_eval_model.py` now implements `generate_until(...)` with multi-batch generation.
- Requests are normalized from lm-eval `Instance.args`, grouped by generation kwargs (`until`, `max_gen_toks`, sampling settings), then executed in padded batches via `model.generate`.
- Outputs preserve original request order and apply per-request stop-string truncation (`until`/`stop`) after decoding.
- `loglikelihood_rolling` is still not implemented.

## 2026-04-03-11:20 : Memory-pollution eval batches choice-scoring forward only
- `memory_pollution` now supports `eval_batch_size` for multiple-choice scoring.
- The current batching scope is intentionally narrow:
  - clean-choice scoring is batched within each example,
  - perturbed-choice scoring is batched within each example,
  - state-drift cache capture remains single-example.
- This keeps the implementation small while addressing the main eval bottleneck: repeated forward passes over answer choices.

## 2026-04-03-11:40 : Memory-pollution configs can be self-contained
- `configs/memory_pollution/*.yaml` no longer need to point to a separate train config file.
- `memory_pollution/config.py` now supports two loading modes:
  - `train_config_path` for reusing an existing `TrainConfig`,
  - inline model-loader fields (`model_name`, `expected_architecture`, `dtype`, etc.) for self-contained eval configs.
- When both are present, `train_config_path` provides the base config and inline model-loader fields override it.

## 2026-04-03-11:55 : Memory-pollution eval drops LoRA support
- `memory_pollution` is now explicitly eval-only and supports `finetune_mode=full` only.
- Self-contained memory-pollution configs no longer expose LoRA knobs.
- If `train_config_path` points to a LoRA training config, runtime now fails fast instead of silently inheriting adapter settings.

## 2026-04-03-12:30 : `memory_pollution` adds strict `lambada_openai` support
- `memory_pollution` now supports `task=lambada_openai` in addition to ARC.
- The LAMBADA task is aligned to lm-eval-harness `lambada_openai` semantics:
  - `context_text = text.rsplit(" ", 1)[0]`
  - `target_text = " " + text.rsplit(" ", 1)[1]`
  - metrics are based on gold continuation logprob and greedy exact match.
- `memory_pollution/scoring.py` is now the shared continuation scorer for both ARC and `lambada_openai`; task-specific aggregation lives in `memory_pollution/metrics.py`.

## 2026-04-12-12:00 : Standalone `exposure_bias` eval package
- Exposure-bias evaluation lives in a new top-level `exposure_bias/` package with its own config/entrypoint, instead of extending `eval/config.py`.
- The eval path now supports `task=hf_dataset`, using fixed-length chunks of `prefix_len + rollout_len` tokens from a tokenized HF dataset text field.
- The experiment computes two batched metrics per sample:
  - `CE_TF`: teacher-forcing CE on the held-out rollout segment,
  - `CE_rollout`: CE against the same ground-truth tokens while the model rolls out greedily under its own generated history.
- `exposure_bias_gap = CE_rollout - CE_TF` is the main aggregate metric; rollout is batched across samples with a shared prefix length and rollout length.

## 2026-04-13-12:30 : Standalone `exposure_bias` local-text finetuning path
- Domain finetuning for exposure-bias experiments now lives under `exposure_bias/` instead of reusing `train.py` / `opd/train_loop.py`.
- First version is intentionally minimal:
  - HF dataset input only,
  - next-token CE finetuning only,
  - single-process only,
  - optional model-only checkpoint initialization,
  - LoRA targeted only to Linear modules in the last `N` backbone blocks.
- HF dataset loading supports both remote dataset repos and local cached dataset snapshots/scripts via `load_dataset(path_or_name, ...)`, so datasets like `lara-martin/Scifi_TV_Shows` can be used without exporting to `.txt`.
- Within `exposure_bias/`, train-specific modules now live under `exposure_bias/train/` and eval-specific modules under `exposure_bias/eval/`; only shared helpers remain at the package top level.

## 2026-04-13-13:10 : Exposure-bias eval output names use dataset aliases
- `exposure_bias/io.py` now derives eval output names from dataset aliases instead of the generic `hf_dataset` task name.
- Current built-in aliases are:
  - `HuggingFaceFW/fineweb-edu -> fineweb`
  - `lara-martin/Scifi_TV_Shows -> scifi`
  - `WutYee/HarryPotter_books_1to7 -> harrypotter`
- The auto-generated experiment name format is now `{dataset_alias}_{model_slug}_p{prefix_len}_r{rollout_len}`.

## 2026-04-14-10:30 : GSM8K support in `exposure_bias`
- `exposure_bias/train` now supports `task=gsm8k_sft`, which formats each GSM8K example as:
  - `Question: ...`
  - `Thoughts:`
  - `<gold rationale>`
  - `Final Answer: <gold answer>`
  and then packs those formatted examples into fixed-length token chunks for CE finetuning.
- `exposure_bias/eval` now supports `task=gsm8k_thought_reveal`.
- GSM8K thought-reveal eval uses reveal ratios over heuristic rationale steps:
  - reveal ratios are config-driven and must include `0.0`,
  - rationale steps are split first by non-empty lines, then by sentence boundaries if needed,
  - prompt keeps the revealed gold thought prefix and asks the model to greedily continue the remaining thoughts and final answer.
- GSM8K metrics are aggregated as:
  - `Acc(r)` for each reveal ratio,
  - `Gap_r = Acc(r) - Acc(0.0)` for `r > 0`,
  and model-vs-model gap deltas are compared via `scripts/compare_gsm8k_thought_reveal.py`.
- For `task=gsm8k_thought_reveal`, `prefix_len` and `rollout_len` are legacy config fields with no experiment semantics:
  - evaluation uses the full question plus revealed gold thought prefix without token truncation,
  - output naming no longer includes `p{prefix_len}_r{rollout_len}` for GSM8K,
  - runtime fails fast if `prompt_len + max_new_tokens` exceeds the model context length.

## 2026-04-15-18:30 : Add standalone QwenOPSD experiment scaffold
- Added a separate top-level `QwenOPSD/` package instead of extending `opd/`, because idea-5 uses a different training objective and data shape:
  - teacher fixed at the initial Qwen checkpoint,
  - student trained on corrupted `solution` prefixes,
  - mixed forward/reverse KL only,
  - `problem/solution` pair data instead of packed raw text.
- `QwenOPSD` training uses Qwen chat-template formatting for the `problem` prompt (`apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)`), then treats `solution` as the assistant reasoning continuation to corrupt/distill.
- The first implementation keeps the training loop explicit and sample-wise:
  - teacher logits are collected from one clean teacher-forcing forward on the gold solution prefix,
  - student rollout is greedy from the corrupted prefix,
  - loss is averaged over rollout positions only,
  - eval exists only as a placeholder scaffold for now.

## 2026-04-15-19:10 : QwenOPSD now carries its own DDP and wandb integration
- `QwenOPSD/` does not import `opd.distributed`; it now has its own `QwenOPSD/distributed.py` with the same minimal primitives (`DistEnv`, init/cleanup, barrier, reduce_mean).
- In distributed training, only the student model is wrapped with DDP; the frozen teacher stays as one local copy per rank.
- `QwenOPSD/train/data.py` now uses `DistributedSampler` when `WORLD_SIZE>1`, while keeping the same sample-wise loss semantics.
- `wandb` initialization/logging is rank-0 only, and checkpoint save is guarded by `barrier()` + main-rank write to avoid multi-rank clobbering.
