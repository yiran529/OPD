from __future__ import annotations

import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup

from QwenOPSD.checkpoint import load_training_checkpoint, save_training_checkpoint
from QwenOPSD.distributed import DistEnv, barrier, reduce_mean
from QwenOPSD.train.config import QwenOPSDTrainConfig
from QwenOPSD.train.corruption import build_corrupted_prefix
from QwenOPSD.train.data import PreparedMathSample, build_train_dataloader
from QwenOPSD.train.losses import DistillLossBundle, mixed_kl_from_logits


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _seed_all(seed: int, rank: int) -> None:
    merged_seed = seed + rank
    random.seed(merged_seed)
    np.random.seed(merged_seed)
    torch.manual_seed(merged_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(merged_seed)


def _autocast_context(cfg: QwenOPSDTrainConfig, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if cfg.dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if cfg.dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _build_optimizer(
    model: torch.nn.Module,
    cfg: QwenOPSDTrainConfig,
) -> tuple[torch.optim.Optimizer, list[torch.nn.Parameter]]:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    assert trainable_params, "no trainable parameters"
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    return optimizer, trainable_params


def _tensorize(token_ids: list[int], device: torch.device) -> torch.Tensor:
    if not token_ids:
        raise ValueError("token_ids must be non-empty")
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def _prefill_one_path(
    model: torch.nn.Module,
    prompt_ids: list[int],
    prefix_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, Any]:
    input_ids = _tensorize(prompt_ids + prefix_ids, device=device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Path prefill did not return logits")
    past_key_values = getattr(outputs, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("Path prefill did not return past_key_values")
    return logits[:, -1, :], past_key_values


def _decode_one_token(model: torch.nn.Module, next_token: torch.Tensor, past_key_values) -> tuple[torch.Tensor, Any]:
    outputs = model(
        input_ids=next_token,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Path decode did not return logits")
    next_cache = getattr(outputs, "past_key_values", None)
    if next_cache is None:
        raise RuntimeError("Path decode did not return past_key_values")
    return logits[:, -1, :], next_cache


@torch.no_grad()
def _generate_student_rollout_tokens(
    rollout_model: torch.nn.Module,
    prompt_ids: list[int],
    student_prefix_ids: list[int],
    rollout_len: int,
    device: torch.device,
) -> list[int]:
    assert rollout_len > 0, "rollout_len must be positive"

    was_training = rollout_model.training
    rollout_model.eval()
    try:
        # ---- prefill rollout cache on the corrupted prefix ----
        current_logits, cache = _prefill_one_path(
            model=rollout_model,
            prompt_ids=prompt_ids,
            prefix_ids=student_prefix_ids,
            device=device,
        )

        # ---- greedily generate a fixed rollout segment ----
        rollout_tokens: list[int] = []
        for step_idx in range(rollout_len):
            next_token = current_logits.argmax(dim=-1, keepdim=True)
            rollout_tokens.append(int(next_token.item()))
            if step_idx + 1 == rollout_len:
                break
            current_logits, cache = _decode_one_token(
                model=rollout_model,
                next_token=next_token,
                past_key_values=cache,
            )
        return rollout_tokens
    finally:
        if was_training:
            rollout_model.train()


def _forward_rollout_logits(
    model: torch.nn.Module,
    prompt_ids: list[int],
    prefix_ids: list[int],
    rollout_tokens: list[int],
    device: torch.device,
) -> torch.Tensor:
    assert rollout_tokens, "rollout_tokens must be non-empty"

    # ---- forward on the fixed rollout sequence ----
    full_input_ids = _tensorize(prompt_ids + prefix_ids + rollout_tokens, device=device)
    attention_mask = torch.ones_like(full_input_ids)
    outputs = model(
        input_ids=full_input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Forward path did not return logits")

    # ---- slice logits that predict the rollout tokens ----
    prefix_total_len = len(prompt_ids) + len(prefix_ids)
    rollout_len = len(rollout_tokens)
    gather_start = prefix_total_len - 1
    gather_end = gather_start + rollout_len
    if gather_start < 0:
        raise RuntimeError(
            f"Invalid gather_start={gather_start}; prompt+prefix must contain at least one token"
        )
    rollout_logits = logits[:, gather_start:gather_end, :]
    if rollout_logits.size(1) != rollout_len:
        raise RuntimeError(
            "Forward-sliced rollout logits have the wrong length: "
            f"expected={rollout_len} got={rollout_logits.size(1)}"
        )
    return rollout_logits


def _compute_sample_loss(
    student_model: torch.nn.Module,
    student_rollout_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    sample: PreparedMathSample,
    cfg: QwenOPSDTrainConfig,
    device: torch.device,
) -> tuple[DistillLossBundle, dict[str, int]]:
    # ---- corrupt the gold reasoning prefix with B spans ----
    corruption = build_corrupted_prefix(
        solution_ids=sample.solution_ids,
        rollout_len=cfg.rollout_len,
        num_spans=cfg.num_corrupt_spans,
        span_choices=cfg.corrupt_span_choices,
        start_min_ratio=cfg.corrupt_start_min_ratio,
        start_max_ratio=cfg.corrupt_start_max_ratio,
    )

    # ---- first stage: rollout under the student policy ----
    rollout_tokens = _generate_student_rollout_tokens(
        rollout_model=student_rollout_model,
        prompt_ids=sample.prompt_ids,
        student_prefix_ids=corruption.student_prefix_ids,
        rollout_len=cfg.rollout_len,
        device=device,
    )

    # ---- second stage: fixed-sequence forward on the shared rollout ----
    student_rollout_logits = _forward_rollout_logits(
        model=student_model,
        prompt_ids=sample.prompt_ids,
        prefix_ids=corruption.student_prefix_ids,
        rollout_tokens=rollout_tokens,
        device=device,
    )
    with torch.no_grad():
        teacher_rollout_logits = _forward_rollout_logits(
            model=teacher_model,
            prompt_ids=sample.prompt_ids,
            prefix_ids=corruption.teacher_prefix_ids,
            rollout_tokens=rollout_tokens,
            device=device,
        )

    loss_bundle = mixed_kl_from_logits(
        student_logits=student_rollout_logits,
        teacher_logits=teacher_rollout_logits,
        alpha=cfg.alpha,
    )
    return (
        loss_bundle,
        {
            "prompt_len": len(sample.prompt_ids),
            "solution_len": len(sample.solution_ids),
            "rollout_start": corruption.rollout_start,
            "span_len": corruption.span_len,
            "num_spans": len(corruption.spans),
        },
    )


def _optimizer_steps_per_epoch(
    num_micro_batches: int,
    grad_accum_steps: int,
) -> int:
    assert num_micro_batches > 0, "num_micro_batches must be positive"
    assert grad_accum_steps > 0, "grad_accum_steps must be positive"
    return (num_micro_batches + grad_accum_steps - 1) // grad_accum_steps


def run_training(
    cfg: QwenOPSDTrainConfig,
    dist_env: DistEnv,
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    tokenizer,
) -> None:
    _seed_all(cfg.seed, dist_env.rank)

    run_dir = Path(cfg.output_dir) / cfg.run_name
    checkpoint_dir = run_dir / "checkpoints"
    if dist_env.is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(cfg.as_dict(), sort_keys=False),
            encoding="utf-8",
        )
    barrier()

    dataloader, dataset_stats, sampler = build_train_dataloader(
        cfg=cfg,
        tokenizer=tokenizer,
        rank=dist_env.rank,
        world_size=dist_env.world_size,
    )
    if dist_env.is_main:
        print(
            "dataset preparation: "
            f"seen={dataset_stats.num_rows_seen} "
            f"kept={dataset_stats.num_rows_kept} "
            f"skip_missing={dataset_stats.num_rows_skipped_missing_fields} "
            f"skip_correct={dataset_stats.num_rows_skipped_correct_filter} "
            f"skip_prompt_len={dataset_stats.num_rows_skipped_prompt_len} "
            f"skip_solution_len={dataset_stats.num_rows_skipped_solution_len}",
            flush=True,
        )

    num_micro_batches = len(dataloader)
    steps_per_epoch = _optimizer_steps_per_epoch(
        num_micro_batches=num_micro_batches,
        grad_accum_steps=cfg.grad_accum_steps,
    )
    total_optimizer_steps = cfg.num_epochs * steps_per_epoch

    if dist_env.is_main:
        print(
            f"num_micro_batches={num_micro_batches} "
            f"steps_per_epoch={steps_per_epoch} "
            f"num_epochs={cfg.num_epochs} "
            f"total_optimizer_steps={total_optimizer_steps}",
            flush=True,
        )

    student_model.to(dist_env.device)
    teacher_model.to(dist_env.device)
    if dist_env.is_distributed:
        ddp_find_unused_parameters = (
            cfg.finetune_mode == "full" and cfg.model_class == "conditional_generation"
        )
        student_model = DDP(
            student_model,
            device_ids=[dist_env.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=ddp_find_unused_parameters,
        )

    raw_student_model = _unwrap_model(student_model)

    optimizer, trainable_params = _build_optimizer(model=student_model, cfg=cfg)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = None
    if dist_env.device.type == "cuda" and cfg.dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    total_params = 0
    trainable_param_count = 0
    if dist_env.is_main:
        total_params = sum(param.numel() for param in student_model.parameters())
        trainable_param_count = sum(param.numel() for param in trainable_params)
        print(
            "trainable parameter summary: "
            f"finetune_mode={cfg.finetune_mode} "
            f"trainable_params={trainable_param_count} "
            f"total_params={total_params} "
            f"trainable_ratio={100.0 * trainable_param_count / max(total_params, 1):.4f}%",
            flush=True,
        )

    wandb_log = lambda *_args, **_kwargs: None
    wandb_finish = lambda: None
    if dist_env.is_main and cfg.wandb_enabled and cfg.wandb_mode != "disabled":
        import wandb

        wandb_run: Any = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name or cfg.run_name,
            tags=cfg.wandb_tags or None,
            mode=cfg.wandb_mode,
            config=cfg.as_dict(),
            dir=str(run_dir),
        )
        wandb_run.summary["world_size"] = dist_env.world_size
        wandb_run.summary["model_class"] = cfg.model_class
        wandb_run.summary["alpha"] = cfg.alpha
        wandb_run.summary["rollout_len"] = cfg.rollout_len
        wandb_run.summary["num_corrupt_spans"] = cfg.num_corrupt_spans
        wandb_run.summary["total_params"] = total_params
        wandb_run.summary["trainable_params"] = trainable_param_count
        wandb_log = wandb_run.log
        wandb_finish = wandb_run.finish

    global_step = 0
    if cfg.resume_path:
        global_step = load_training_checkpoint(
            checkpoint_path=cfg.resume_path,
            model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=dist_env.device,
        )
        if dist_env.is_main:
            print(f"resumed training from step={global_step} path={cfg.resume_path}", flush=True)

    student_model.train()
    teacher_model.eval()

    accum_steps = 0
    step_time = time.time()
    running_total = torch.zeros([], device=dist_env.device)
    running_forward = torch.zeros([], device=dist_env.device)
    running_reverse = torch.zeros([], device=dist_env.device)
    running_span = 0
    running_num_spans = 0
    running_rollout_start = 0
    running_prompt = 0
    running_solution = 0
    running_samples = 0

    for epoch_idx in range(1, cfg.num_epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch_idx)
        optimizer.zero_grad(set_to_none=True)

        for micro_step, batch_samples in enumerate(dataloader, start=1):
            if global_step >= total_optimizer_steps:
                break

            group_start = ((micro_step - 1) // cfg.grad_accum_steps) * cfg.grad_accum_steps + 1
            accum_target = min(cfg.grad_accum_steps, num_micro_batches - group_start + 1)

            batch_total: torch.Tensor | None = None
            batch_forward: torch.Tensor | None = None
            batch_reverse: torch.Tensor | None = None
            valid_samples = 0
            batch_span_sum = 0
            batch_num_spans_sum = 0
            batch_rollout_start_sum = 0
            batch_prompt_sum = 0
            batch_solution_sum = 0

            with _autocast_context(cfg, dist_env.device):
                for sample in batch_samples:
                    sample_bundle, sample_meta = _compute_sample_loss(
                        student_model=student_model,
                        student_rollout_model=raw_student_model,
                        teacher_model=teacher_model,
                        sample=sample,
                        cfg=cfg,
                        device=dist_env.device,
                    )
                    batch_total = sample_bundle.total if batch_total is None else batch_total + sample_bundle.total
                    batch_forward = (
                        sample_bundle.forward_kl
                        if batch_forward is None
                        else batch_forward + sample_bundle.forward_kl
                    )
                    batch_reverse = (
                        sample_bundle.reverse_kl
                        if batch_reverse is None
                        else batch_reverse + sample_bundle.reverse_kl
                    )
                    valid_samples += 1
                    batch_span_sum += sample_meta["span_len"]
                    batch_num_spans_sum += sample_meta["num_spans"]
                    batch_rollout_start_sum += sample_meta["rollout_start"]
                    batch_prompt_sum += sample_meta["prompt_len"]
                    batch_solution_sum += sample_meta["solution_len"]

            if valid_samples == 0:
                continue

            batch_total = batch_total / valid_samples
            batch_forward = batch_forward / valid_samples
            batch_reverse = batch_reverse / valid_samples

            if not torch.isfinite(batch_total):
                raise FloatingPointError(
                    f"Non-finite batch_total encountered at epoch={epoch_idx} micro_step={micro_step}"
                )

            if scaler is not None:
                scaler.scale(batch_total / accum_target).backward()
            else:
                (batch_total / accum_target).backward()

            running_total += batch_total.detach()
            running_forward += batch_forward.detach()
            running_reverse += batch_reverse.detach()
            running_span += batch_span_sum
            running_num_spans += batch_num_spans_sum
            running_rollout_start += batch_rollout_start_sum
            running_prompt += batch_prompt_sum
            running_solution += batch_solution_sum
            running_samples += valid_samples
            accum_steps += 1

            should_step = accum_steps == cfg.grad_accum_steps or micro_step == num_micro_batches
            if not should_step:
                continue

            global_step += 1
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if global_step % cfg.log_interval == 0 or global_step == 1:
                elapsed = time.time() - step_time
                mean_total = reduce_mean(running_total / accum_steps, dist_env.world_size)
                mean_forward = reduce_mean(running_forward / accum_steps, dist_env.world_size)
                mean_reverse = reduce_mean(running_reverse / accum_steps, dist_env.world_size)
                mean_span = reduce_mean(
                    torch.tensor(running_span / max(running_samples, 1), device=dist_env.device, dtype=torch.float32),
                    dist_env.world_size,
                )
                mean_num_spans = reduce_mean(
                    torch.tensor(running_num_spans / max(running_samples, 1), device=dist_env.device, dtype=torch.float32),
                    dist_env.world_size,
                )
                mean_rollout_start = reduce_mean(
                    torch.tensor(
                        running_rollout_start / max(running_samples, 1),
                        device=dist_env.device,
                        dtype=torch.float32,
                    ),
                    dist_env.world_size,
                )
                mean_prompt = reduce_mean(
                    torch.tensor(running_prompt / max(running_samples, 1), device=dist_env.device, dtype=torch.float32),
                    dist_env.world_size,
                )
                mean_solution = reduce_mean(
                    torch.tensor(running_solution / max(running_samples, 1), device=dist_env.device, dtype=torch.float32),
                    dist_env.world_size,
                )
                grad_norm_tensor = reduce_mean(grad_norm.detach().float(), dist_env.world_size)
                current_lr = scheduler.get_last_lr()[0]

                if dist_env.is_main:
                    print(
                        f"epoch={epoch_idx}/{cfg.num_epochs} "
                        f"step={global_step}/{total_optimizer_steps} "
                        f"loss_kd={mean_total.item():.6f} "
                        f"loss_fwd_kl={mean_forward.item():.6f} "
                        f"loss_rev_kl={mean_reverse.item():.6f} "
                        f"alpha={cfg.alpha:.3f} "
                        f"avg_num_spans={mean_num_spans.item():.2f} "
                        f"avg_span_len={mean_span.item():.2f} "
                        f"avg_rollout_start={mean_rollout_start.item():.2f} "
                        f"avg_prompt_len={mean_prompt.item():.2f} "
                        f"avg_solution_len={mean_solution.item():.2f} "
                        f"lr={current_lr:.6e} "
                        f"grad_norm={grad_norm_tensor.item():.4f} "
                        f"dt={elapsed:.2f}s",
                        flush=True,
                    )
                    wandb_log(
                        {
                            "loss_kd": mean_total.item(),
                            "loss_fwd_kl": mean_forward.item(),
                            "loss_rev_kl": mean_reverse.item(),
                            "avg_num_spans": mean_num_spans.item(),
                            "avg_span_len": mean_span.item(),
                            "avg_rollout_start": mean_rollout_start.item(),
                            "avg_prompt_len": mean_prompt.item(),
                            "avg_solution_len": mean_solution.item(),
                            "grad_norm": grad_norm_tensor.item(),
                            "lr": current_lr,
                        },
                        step=global_step,
                    )
                step_time = time.time()

            if global_step % cfg.save_every_n_steps == 0 or global_step == total_optimizer_steps:
                barrier()
                if dist_env.is_main:
                    checkpoint_path = save_training_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=global_step,
                        model=student_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        config_dict=cfg.as_dict(),
                        keep_last_k=cfg.keep_last_k_checkpoints,
                    )
                    print(f"saved checkpoint: {checkpoint_path}", flush=True)
                barrier()

            accum_steps = 0
            running_total = torch.zeros([], device=dist_env.device)
            running_forward = torch.zeros([], device=dist_env.device)
            running_reverse = torch.zeros([], device=dist_env.device)
            running_span = 0
            running_num_spans = 0
            running_rollout_start = 0
            running_prompt = 0
            running_solution = 0
            running_samples = 0

    raw_student_model.eval()
    wandb_finish()
