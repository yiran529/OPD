# Durable Memory

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