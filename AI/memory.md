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

## 2026-04-18-00:00 : LinearOPSD generation logging trims padding and batched decode uses left-padded prompts
- `LinearOPSD/opsd_trainer.py` no longer decodes padded `student_prompts` directly into `generations_step_*.json`; prompt/completion logs are now trimmed by per-example lengths so `<|endoftext|>` padding tails do not dominate saved samples.
- For non-vLLM generation, batched decoder-only rollout now left-pads prompts only for the `model.generate(...)` call, then maps completions back onto the existing right-padded training layout.
- Reason: the trainer stores prompts in right-padded form for explicit label masking, but feeding those right-padded prompts directly into decoder-only `generate(...)` makes eos/pad tokens act like the effective sequence tail and can corrupt rollout quality.

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

## 2026-04-15-19:35 : QwenOPSD prompt-template source and training data flow summary
- `QwenOPSD/train/formatting.py` does not define its own chat template; it calls `tokenizer.apply_chat_template(...)`, so the source of truth is the model repo's tokenizer template for `Qwen/Qwen3.5-0.8B`.
- For this checkpoint, the template is embedded in `tokenizer_config.json` (`chat_template`) and also mirrored as `chat_template.jinja` in the Hugging Face model repo.
- Current implementation choice:
  - `messages=[{"role":"user","content": problem}]`
  - `add_generation_prompt=True`
  - `enable_thinking=True`
  - This means the rendered prompt ends at the assistant thinking prefix (`<|im_start|>assistant\n<think>\n`), and raw `solution` tokens are appended as the reasoning continuation target.
- Current QwenOPSD training data flow is:
  - load `problem/solution` rows from OpenThoughts math,
  - filter invalid / overlong / optionally `correct != true` rows,
  - tokenize `problem` via the Qwen chat template into `prompt_ids`,
  - tokenize `solution` directly into `solution_ids`,
  - corrupt a short span inside `solution_ids`,
  - teacher runs one clean forward on `prompt + clean solution prefix`,
  - student prefills on `prompt + corrupted solution prefix` and greedily rolls out,
  - loss is mixed forward/reverse KL averaged over rollout positions only.

## 2026-04-15-20:05 : QwenOPSD switched to shared-rollout teacher + multi-span corruption
- Revised QwenOPSD to match the updated idea-5 semantics:
  - student and teacher now share the same student-generated rollout history,
  - teacher no longer teacher-forces on the clean gold continuation,
  - teacher differs from student only by seeing clean patches on the corrupted spans inside the prefix.
- `QwenOPSD/train/corruption.py` now supports configurable `num_corrupt_spans` (`B`):
  - sample `B` non-overlapping spans with one shared span length `m`,
  - replace corrupted tokens by independently sampled donor tokens from other positions in the same solution,
  - build `student_prefix_ids`, `teacher_prefix_ids`, and `rollout_start = max_i(s_i + m)`.
- `QwenOPSD/train/loop.py` now runs student/teacher caches in lockstep:
  - prefill student on the corrupted prefix,
  - prefill teacher on the patched prefix,
  - compute mixed KL on current logits,
  - advance both paths with the same greedy student token.

## 2026-04-15-20:30 : QwenOPSD training switched to two-stage rollout-then-forward
- QwenOPSD no longer computes KD loss inside the same stepwise decode loop used to generate rollout tokens.
- New training structure:
  - first run a no-grad greedy student rollout on the corrupted prefix,
  - then run one full student forward on `prompt + student_prefix + rollout_tokens`,
  - and one full teacher forward on `prompt + teacher_prefix + rollout_tokens`,
  - then slice rollout-position logits and compute mixed KL on the fixed shared rollout.
- Rollout still uses the local HF/Transformers model only; no vLLM integration has been added yet.
- During rollout generation, the raw student model is temporarily switched to `eval()` so greedy tokens are not polluted by dropout, and then restored to its previous training mode.


## 2026-04-15-17:40 : LinearOPSD stays in upstream OPSD files and keeps OPSD defaults
- The `LinearOPSD/` adaptation is implemented by extending the existing upstream files (`opsd_train.py`, `data_collator.py`, `opsd_trainer.py`) instead of forking a second trainer path. This keeps DDP/accelerate/wandb/vLLM/EMA behavior on the original codepath.
- Default behavior intentionally remains OPSD-like:
  - `conditioning_mode="opsd"`
  - `loss_mode="jsd"`
  - `rollout_decoding="sample"`
  - dynamic teacher remains the default (neither `fixed_teacher` nor EMA is forced on).
- The new `conditioning_mode="linear_opsd"` path changes only the conditioning semantics:
  - collator builds student prompts from `problem + corrupted solution prefix`
  - collator builds teacher prompts from `problem + patched solution prefix`
  - trainer still reuses the same on-policy generation and shared-rollout plumbing, and adds `loss_mode="mixed_kl"` for the LinearOPSD objective.

## 2026-04-16-10:15 : LinearOPSD eval stays flat and adds a rollout-inspection script
- `LinearOPSD/eval/` remains a flat directory for now; we did not introduce `inspect/` or `benchmarks/` subfolders because the current eval surface is still small and the lowest-risk choice is to add new scripts alongside the existing ones.
- Added a dedicated inspection path separate from benchmark eval:
  - `eval/inspect_linear_opsd_rollout.py` reconstructs the training-time `linear_opsd` corrupted/patched prefixes from `problem/solution` data and runs rollout from the corrupted student prompt.
  - `eval/run_inspect_rollout.sh` is the matching launcher.
- The inspection script reuses the same corruption/prompt helpers from `LinearOPSD/data_collator.py` so eval-time inspection stays aligned with the training-time prefix construction.


## 2026-04-16-11:30 : LinearOPSD rollout start now uses an explicit post-corruption offset
- `LinearOPSD/data_collator.py` no longer starts rollout immediately after the last corrupted span.
- New knobs: `rollout_start_offset` (current default `2`) and `rollout_start_offset_jitter` (current default `10`).
- Effective rollout gap is now sampled per example as `base_offset + delta`, where `delta` is an integer jitter bounded by the configured max and clipped on the negative side so the final offset stays non-negative.
- The offset is now threaded through both training (`opsd_train.py` / `opsd_trainer.py`) and inspection (`eval/inspect_linear_opsd_rollout.py`) so the reported prefix layout matches training-time behavior.

## 2026-04-16-12:10 : Qwen3.5 LinearOPSD needs hybrid-aware LoRA targets and class-name fail-fast
- Qwen3.5 LoRA targets in `LinearOPSD` must cover both legacy attention/MLP names and hybrid linear-attention projection names:
  - `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
  - `in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj`
- `LinearOPSD/opsd_train.py` now auto-appends missing Qwen3.5 LoRA targets when `model_type == qwen3_5`.
- `LinearOPSD/opsd_trainer.py` now fail-fasts if the training model class name and the colocated vLLM model class name do not match, because that mismatch is unsafe for LoRA merge/sync.

## 2026-04-17-10:20 : LinearOPSD corruption moved to trainer-time point corruption submodule
- `LinearOPSD` no longer uses collator-time random span corruption for `conditioning_mode=linear_opsd`.
- New file `LinearOPSD/corruption.py` owns the corruption-specific logic:
  - high-entropy point selection on the clean solution trajectory,
  - heuristic style-token filtering,
  - top-1 non-gold replacement-token selection,
  - inline `<corrupt>` marker injection for the teacher-visible student trace,
  - teacher user-message construction,
  - generic token padding helpers.
- `LinearOPSD/data_collator.py` now leaves `linear_opsd` corruption untouched and only returns static tokenized materials (`problem` prompt ids and `solution` ids).
- `LinearOPSD/opsd_trainer.py` now performs an extra no-grad clean forward inside `training_step`, builds the online corruption from current model logits, then reuses the existing shared-rollout generation + distillation path.
- Old training-time span-corruption fields were removed from the `linear_opsd` CLI surface and replaced by point-corruption fields:
  - removed: `num_corrupt_spans`, `corrupt_span_choices`
  - added: `num_corrupt_points`, `corrupt_marker_text`
- Current inspection scripts are not yet aligned with the new trainer-time corruption path and need a dedicated rewrite instead of reusing the removed collator helper path.

## 2026-04-17-10:45 : LinearOPSD inspection now mirrors trainer-time corruption
- `LinearOPSD/eval/inspect_linear_opsd_rollout.py` no longer depends on the removed collator-time `_build_linear_opsd_prefixes` helper.
- The inspection path now mirrors training semantics more closely:
  - use a local HF causal LM forward pass to score the clean `problem + solution` trajectory,
  - call `LinearOPSD/corruption.py` to build entropy-based point corruption and the teacher-visible `<corrupt>` trace,
  - use vLLM only for rollout from the resulting student prompt.
- This keeps corruption selection and prompt construction aligned with the trainer, while still using vLLM for fast continuation inspection.

## 2026-04-17-14:30 : LinearOPSD switched from entropy point corruption to natural careless-prefix recovery
- `conditioning_mode=linear_opsd` no longer uses entropy-ranked point corruption, replacement-token search, or `<corrupt>` markers.
- The training path is now:
  - sample a `gold prefix` from the clean solution,
  - let the current student generate a short sampled `careless prefix`,
  - resample if that careless segment exactly matches the gold suffix,
  - then switch to normal decoding and distill only the fixed-length recovery rollout.
- Teacher-visible student traces now use explicit stage markers:
  - `<careless>` before the sampled polluted prefix,
  - `<recovery>` before the recovery rollout segment.
- `LinearOPSD/corruption.py` now owns gold-prefix sampling, careless-prefix generation, and teacher trace/message construction for both training and inspection.
- `LinearOPSD/opsd_train.py` / `opsd_trainer.py` expose the new `linear_opsd` knobs:
  - `gold_prefix_ratio_min/max`
  - `careless_rollout_len`, `careless_temperature`, `careless_top_p`, `careless_top_k`
  - `careless_resample_trials`
  - `recovery_rollout_len`
  - `normal_decoding`
  - `careless_marker_text`, `recovery_marker_text`

## 2026-04-17-15:10 : LinearOPSD removed mixed_kl and standardizes on JSD
- `LinearOPSD` no longer supports `loss_mode="mixed_kl"` or the `linear_opsd_alpha` knob.
- Distillation in `LinearOPSD` now uses only:
  - `loss_mode="jsd"`
  - `beta` to select forward-KL (`beta=0`), reverse-KL (`beta=1`), or generalized JSD (`0 < beta < 1`)
- The Qwen3.5 linear-OPSD launcher was updated to use `loss_mode=jsd` directly.

## 2026-04-18-00:10 : LinearOPSD keeps training layout unchanged while fixing batched generation input
- The `LinearOPSD/opsd_trainer.py` fix is intentionally an adapter-layer change, not a training-logic rewrite:
  - `model.generate(...)` now receives a temporary left-padded batch built from `student_prompt_lengths_per_example`,
  - but the post-generation tensors are immediately reconstructed back into the original right-padded training layout before label masking and teacher/student concatenation.
- This preserves the existing downstream assumptions:
  - `generation_ids = generated_ids[:, student_prompt_len:]` still extracts the shared rollout segment,
  - prompt masking still uses the original right-padded prompt block,
  - teacher inputs still append the same rollout tokens without changing the broader OPSD data flow.
- Generation dumps were also expanded to save `teacher_prompt`, `gold_prefix_text`, `careless_prefix_text`, `careless_deviated`, and `skip_kd`, so rollout inspection now exposes the actual `linear_opsd` prefix construction instead of only prompt/completion text.

## 2026-04-18-00:20 : LinearOPSD recovery rollout now reuses the shared OPSD generation config
- `LinearOPSD` no longer hard-codes `linear_opsd` recovery rollout to `temperature=1.0`, `top_p=1.0`, `top_k=0`, `do_sample=False` inside `OPSDTrainer`.
- Recovery rollout now uses the same shared generation path as `opsd`:
  - `args.max_completion_length` remains the single source of rollout length,
  - `rollout_decoding`, `temperature`, `top_p`, and `top_k` remain the single source of sampling behavior.
- For backward compatibility, `opsd_train.py` still accepts `normal_decoding` for `linear_opsd`, but immediately copies it into `rollout_decoding` before trainer construction and treats it as a deprecated alias.
- The eval/logging completion callback now follows the same shared rollout config instead of forcing `linear_opsd` samples back to greedy.

## 2026-04-18-00:30 : Remove `normal_decoding` and keep a single rollout-decoding interface
- `LinearOPSD` no longer exposes `normal_decoding` anywhere in train/eval/inspect paths; `rollout_decoding` is now the only decoding-mode knob.
- This removes duplicated configuration for the same recovery-rollout behavior and keeps `linear_opsd` aligned with the shared `opsd` generation interface.
- Updated surfaces:
  - `LinearOPSD/opsd_train.py` no longer defines/logs/passes `normal_decoding`,
  - `LinearOPSD/opsd_trainer.py` no longer stores or validates it,
  - `LinearOPSD/eval/inspect_linear_opsd_rollout.py` and related shell scripts now use `rollout_decoding`.

## 2026-04-18-00:40 : Rebuild train-time full sequences so completion follows the true prompt tail
- Fixing generation-time left padding alone was not sufficient: the train-time student/teacher forwards were still using `[right-padded prompt block][completion]`, which made the first completion token be predicted from a pad-position hidden state for shorter samples.
- `LinearOPSD/opsd_trainer.py` now rebuilds both student and teacher full sequences into:
  - `[left pad inside fixed prompt block][valid prompt][completion]`
  - while keeping the same fixed prompt-block width used by downstream loss slicing.
- This preserves the existing `student_prompt_length` / `teacher_prompt_length` based loss code, but ensures completion supervision is conditioned on each sample's actual last prompt token instead of a right-padding gap.

## 2026-04-20-00:00 : LinearOPSD protocol cleanup for GSM8K degradation
- Do not rely on the old `jsd_token_clip=0.05` setting for current LinearOPSD runs; the launchers now pass `--jsd_token_clip 0` so the existing full-vocab JSD path is not clipped by the known unsafe component-level clip.
- `linear_opsd` now has separate chat-template switches for student and teacher prompts:
  - `linear_opsd_student_enable_thinking`
  - `linear_opsd_teacher_enable_thinking`
  Both default to false so non-thinking sanity runs can avoid mixing teacher thinking-context style into student continuations.
- The teacher-visible sampled-tail trace no longer uses XML-like `<careless>` / `<recovery>` boundaries by default. The sampled tail is marked only for the teacher with `[recent sampled tail begins here]`; there is no recovery marker before the KD segment.
- Teacher prompts were made more continuation-oriented: the teacher gets the reference solution as private context and a lightweight note that the marked recent tail may contain a local inconsistency, then is asked to continue directly with the next math step.
- `linear_opsd` now samples prefix mode per example:
  - clean: no sampled tail, KD from a gold prefix;
  - mild: short sampled tail from `linear_opsd_mild_careless_rollout_len`;
  - hard: sampled tail from `careless_rollout_len`.
  This is intended to prevent every training example from looking like an explicit correction scenario.

## 2026-04-20-20:47 : LinearOPSD GSM8K outputs after protocol cleanup
- After retraining Qwen3.5-2B LinearOPSD with clipping disabled and clean/mild/hard mixture, GSM8K no longer collapses to near-zero accuracy, but remains clearly below the base model in both thinking and non-thinking eval.
- The observed failure modes differ by eval style:
  - thinking eval: many base-correct/trained-wrong examples contain the correct arithmetic result somewhere, but the model over-generates meta reasoning about request constraints, format interpretation, `wait/check` steps, and often ends with placeholder-like `#### <number>` / `<calculated_value>` instead of substituting the computed answer;
  - non-thinking eval: more examples miss the `####` marker, fall back to last-number extraction, choose an intermediate quantity as the final answer, or occasionally enter long repeated-digit loops.
- Training generations show teacher-prompt style leaking into student behavior. Student completions can mention "reference solution", "logical gap", or correction-style analysis even when the student eval prompt has no reference solution.
- The train/eval tasks differ in their visible context. In LinearOPSD training, student inputs are `problem + gold prefix + optional sampled tail`; teacher inputs are `problem + full reference solution + continuation instruction + marked partial solution`. In GSM8K eval, the model receives only the clean problem prompt and must produce a complete solution plus final `####` answer.
- The 2B run corresponding to `generations_step_195.json` used mild tail length 16, hard tail length 24, recovery length 128, and marker `[sampled tail]`. The saved generation sample contains clean/mild/hard examples, with most sampled examples using a non-empty sampled tail.

## 2026-04-20-21:07 : Detailed LinearOPSD GSM8K output observations
- Compared files:
  - base thinking: `LinearOPSD/outputs/eval_results_gsm8k_Qwen3.5-2B_chat_thinking_temp1.0_valn1.json`
  - trained thinking: `LinearOPSD/outputs/eval_results_gsm8k_Qwen3.5-2B_qwen35_2b_linear_opsd_checkpoint-200_chat_thinking_temp1.0_valn1.json`
  - base non-thinking: `LinearOPSD/outputs/eval_results_gsm8k_Qwen3.5-2B_chat_nonthinking_temp1.0_valn1.json`
  - trained non-thinking: `LinearOPSD/outputs/eval_results_gsm8k_Qwen3.5-2B_qwen35_2b_linear_opsd_checkpoint-200_chat_nonthinking_temp1.0_valn1.json`
- Overall GSM8K metrics in these files:
  - base thinking: accuracy 51.48%, format rate 83.32%, extraction rate 99.77%;
  - trained thinking: accuracy 33.21%, format rate 96.74%, extraction rate 99.92%;
  - base non-thinking: accuracy 74.30%, format rate 91.81%, extraction rate 100.00%;
  - trained non-thinking: accuracy 50.49%, format rate 60.20%, extraction rate 99.92%.
- Thinking-mode pairwise comparison:
  - 343 examples are base-correct/trained-wrong;
  - 102 examples are base-wrong/trained-correct;
  - among the 343 regressions, 98.83% still use the `####` marker, but 83.67% have a placeholder-like extracted answer containing `<...>`;
  - among the 343 regressions, 99.42% contain `wait`, 98.25% contain `check`, 98.25% contain `Analyze the Request`, 99.42% contain request/format meta text, and 48.98% contain correction/reference-style text;
  - among the 343 regressions, the ground-truth answer appears somewhere in 87.46% of full generations, and a correct `#### <ground_truth>` line appears somewhere in 49.85% of full generations;
  - median generation length for trained thinking regressions is 1191 words; mean length is 1181.0 words.
- Non-thinking pairwise comparison:
  - 396 examples are base-correct/trained-wrong;
  - 82 examples are base-wrong/trained-correct;
  - among the 396 regressions, 44.19% use the `####` marker and 50.00% use `last_number` extraction;
  - among the 396 regressions, 31.82% contain `wait`, 16.41% contain `check`, 20.71% contain format-related text, 28.79% contain correction/reference-style text, and 9.85% contain request/format meta text;
  - among the 396 regressions, the ground-truth answer appears somewhere in 33.84% of full generations, and no correct `#### <ground_truth>` line was observed by the simple scan;
  - 2.53% of the 396 non-thinking regressions contain a long repeated-digit span;
  - median generation length for trained non-thinking regressions is 233.5 words; mean length is 482.9 words.
- Typical thinking-mode visible pattern:
  - the model often spends many tokens on prompt analysis, output-format interpretation, and repeated checks;
  - some outputs compute the correct intermediate or final number in the body, then continue generating and end with `#### <number>`, `#### <calculated_value>`, or another placeholder-like extracted answer.
- Typical non-thinking visible pattern:
  - some outputs stop without the required `####` answer line, so grading falls back to the last number in the text;
  - some outputs select an intermediate quantity as the final answer;
  - some outputs enter repeated-number continuations such as long runs of the same digit sequence.
- `generations_step_195.json` training-generation observations:
  - 20 saved samples: 10 mild, 6 hard, 4 clean;
  - active sampled-tail lengths: 16 for mild, 24 for hard, 0 for clean;
  - `skip_kd` is false for 19 samples and true for 1 sample;
  - `careless_deviated` is true for 15 samples and false for 5 samples;
  - median completion length is 58 words; median teacher prompt length is 649.5 words;
  - 3 of 20 saved completions contain visible teacher/reference-style wording such as "reference solution", "logical gap", or correction-like analysis.

## 2026-04-20-21:14 : LinearOPSD GSM8K output length observations
- Overall output lengths in the GSM8K eval files:
  - base thinking: median 1100 words, mean 1003.6 words, p90 1268 words, max 1504 words;
  - trained thinking: median 1184 words, mean 1160.3 words, p90 1300 words, max 1732 words;
  - base non-thinking: median 195 words, mean 295.1 words, p90 679 words, max 1394 words;
  - trained non-thinking: median 199 words, mean 398.4 words, p90 1154 words, max 1961 words.
- Base thinking length split:
  - correct examples: median 883 words, mean 850.8 words, p90 1177 words;
  - wrong examples: median 1174 words, mean 1165.9 words, p90 1303 words.
- Trained thinking length split:
  - correct examples: median 1156 words, mean 1109.7 words, p90 1265 words;
  - wrong examples: median 1199 words, mean 1185.4 words, p90 1308 words.
- Base non-thinking length split:
  - correct examples: median 179 words, mean 227.2 words, p90 353 words;
  - wrong examples: median 283 words, mean 491.3 words, p90 1201 words.
- Trained non-thinking length split:
  - correct examples: median 166 words, mean 248.6 words, p90 541 words;
  - wrong examples: median 302 words, mean 551.2 words, p90 1232 words.
- Same-problem length comparison for thinking eval:
  - base-correct/trained-wrong examples: base median 992 words vs trained median 1191 words; median trained-minus-base difference is +168 words and mean difference is +265.4 words;
  - both-correct examples: base median 749.5 words vs trained median 1144 words; median trained-minus-base difference is +335 words and mean difference is +316.0 words;
  - base-wrong/trained-correct examples: base median 1150.5 words vs trained median 1167.5 words; median difference is +19.5 words;
  - both-wrong examples: base median 1177 words vs trained median 1203.5 words; median difference is +22 words.
- Same-problem length comparison for non-thinking eval:
  - base-correct/trained-wrong examples: base median 197.5 words vs trained median 233.5 words; median trained-minus-base difference is +25 words and mean difference is +217.5 words;
  - both-correct examples: base median 166.5 words vs trained median 160.5 words; median difference is -2.5 words;
  - base-wrong/trained-correct examples: base median 242 words vs trained median 212 words; median difference is -6.5 words;
  - both-wrong examples: base median 299 words vs trained median 538 words; median difference is +22 words and mean difference is +133.3 words.

## 2026-04-21 : LinearOPSD teacher prompt uses assistant-prefix continuation again
- `LinearOPSD/corruption.py` changed the `linear_opsd` teacher prompt so `current_trace` is no longer embedded inside the user message as `Current work`.
- The teacher user message now contains `Problem`, `Known correct work`, and a continuation instruction that asks the teacher to use the known work only as private context, continue naturally from the existing assistant work, and preserve mathematical correctness.
- The actual `current_trace` is returned as `teacher_trace_prefix_text`, so trainer and inspection prompts are shaped as user instruction followed by an assistant prefix containing `gold prefix + optional sampled-tail marker + optional sampled tail`.

## 2026-04-21 : LinearOPSD high-loss token diagnostics
- Added default-off detailed JSD diagnostics for `LinearOPSD/opsd_trainer.py` through a separate `LinearOPSD/loss_detail_logging.py` module.
- The diagnostics read detached unreduced JSD/logits only after the normal JSD loss is computed; they do not change rollout construction, the scalar training loss, or backward behavior.
- When enabled, the logger records position-loss quantiles/histograms, rollout-position buckets, top high-loss token events, student/teacher top vocab choices, JSD top contributors, console summaries, W&B tables, and local JSONL files under `output_dir/loss_detail/`.

## 2026-04-21 : LinearOPSD loss-detail context fields
- Loss-detail logging now records from `global_step=0` when `loss_detail_log_steps > 0`, so early prompt/recovery alignment issues are visible.
- High-loss events now include `student_prompt_tail`, `teacher_prompt_tail`, `gold_prefix_tail`, `careless_tail_text`, `gold_next_text`, and `teacher_actual_token_prob`.
- `LinearOPSD/corruption.py` stores `gold_recovery_target_ids` in linear-OPSD metadata so `gold_next_text` reflects the gold continuation after the current recovery start, including clean samples where there is no careless tail.

## 2026-04-21 : LinearOPSD exact token-level teacher trace
- Loss-detail analysis showed high first-token JSD was often caused by teacher trace text being decoded, joined with marker text/spaces, then retokenized, while the student prompt used raw token ids.
- In `loss_events_step_0/20.jsonl`, high-loss events were concentrated at the recovery boundary: 22/50 and 26/50 top events were at `rollout_pos=0`; W&B position buckets also showed step-20 bucket 0 (`pos 0-31`) had much higher average loss than later buckets.
- Clean `rollout_pos=0` events often had student tokens matching `gold_next_text` while teacher assigned low probability, indicating teacher/student prompt-tail mismatch rather than bad student continuation.
- The old inline sampled-tail marker frequently appeared inside math expressions and could collide with LaTeX `\[`; top teacher/JSD tokens sometimes became meta words such as `sample`, `continue`, `restore`, `correct`, `Known`, or `Correction`.
- `LinearOPSD/corruption.py` now returns `teacher_trace_prefix_ids = gold_prefix_ids + careless_token_ids`; no inline sampled-tail or recovery marker is inserted into the assistant trace.
- `LinearOPSD/opsd_trainer.py` now appends those trace ids directly after the teacher chat-template prefix and asserts the teacher trace ids exactly match the student prompt tail.
- `careless_marker_text` and `recovery_marker_text` remain only as deprecated compatibility fields; launch scripts no longer pass sampled-tail marker text.
