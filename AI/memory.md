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
- [] 应该用CE吗？
- [] theta_old是什么
- [] max_length/chunking
- [] 应该切断rollout的tokens之间的梯度吗
- [] 应该赋予接近corrupted tokens的loss低一些的权重？


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
  - `opd_grad_through_prefix` (default `true`)
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
