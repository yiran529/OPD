from __future__ import annotations

import copy
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup

from opd.checkpoint import load_checkpoint, save_checkpoint
from opd.config import TrainConfig
from opd.distributed import DistEnv, barrier, reduce_mean
from opd.fineweb_data import build_dataloader
from opd.losses import OpdLossBundle
from opd.rollout import build_entropy_corrupted_prefix
from opd.state_alignment import compute_stepwise_opd_losses


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _seed_all(seed: int, rank: int) -> None:
    merged_seed = seed + rank
    random.seed(merged_seed)
    np.random.seed(merged_seed)
    torch.manual_seed(merged_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(merged_seed)


def _autocast_context(cfg: TrainConfig, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if cfg.dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if cfg.dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _split_opd_batch_segments(batch_tokens: torch.Tensor, cfg: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    assert batch_tokens.dim() == 2, f"batch_tokens shape mismatch: expected rank=2 [batch,seq+1], got shape={tuple(batch_tokens.shape)}"
    assert batch_tokens.size(1) == cfg.sequence_plus_one, (
        f"sequence length mismatch: expected={cfg.sequence_plus_one} got={batch_tokens.size(1)}"
    )

    context = batch_tokens[:, : cfg.context_len]
    clean_prefix = batch_tokens[:, cfg.context_len : cfg.context_len + cfg.prefix_len]
    return context, clean_prefix


def _build_optimizer(
    model: torch.nn.Module,
    cfg: TrainConfig,
) -> Tuple[torch.optim.Optimizer, List[torch.nn.Parameter]]:
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


def _compute_baseline_ce(model: torch.nn.Module, batch_tokens: torch.Tensor):
    input_ids = batch_tokens[:, :-1]
    labels = batch_tokens[:, 1:]
    return model(input_ids=input_ids, labels=labels, use_cache=False).loss


def _compute_opd_loss(
    model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    rollout_source_model: torch.nn.Module,
    batch_tokens: torch.Tensor,
    cfg: TrainConfig,
) -> OpdLossBundle:
    context, clean_prefix = _split_opd_batch_segments(batch_tokens, cfg)

    rollout_was_training = rollout_source_model.training
    rollout_source_model.eval()
    with torch.inference_mode():
        corrupted_prefix = build_entropy_corrupted_prefix(
            model=rollout_source_model,
            context_tokens=context,
            clean_prefix_tokens=clean_prefix,
            topk_ratio=cfg.prefix_corrupt_topk_ratio,
            topk_max=cfg.prefix_corrupt_topk_max,
        )
    if rollout_was_training:
        rollout_source_model.train()

    return compute_stepwise_opd_losses(
        model=model,
        teacher_model=teacher_model,
        context_tokens=context,
        corrupted_prefix_tokens=corrupted_prefix,
        clean_prefix_tokens=clean_prefix,
        continuation_len=cfg.continuation_len,
        rollout_temperature=cfg.rollout_temperature,
        rollout_top_p=cfg.rollout_top_p,
        lambda_kl=cfg.lambda_kl,
        lambda_state=cfg.lambda_state,
        state_key=cfg.state_key,
        state_time_stride=cfg.state_time_stride,
        state_align_loss=cfg.state_align_loss,
    )


def _build_ema_teacher(
    student_model: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
) -> torch.nn.Module | None:
    if not (cfg.objective == "opd_kl" and cfg.ema_teacher_enabled):
        return None

    ema_model = copy.deepcopy(student_model)
    ema_model.to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def _update_ema_teacher(
    ema_model: torch.nn.Module,
    student_model: torch.nn.Module,
    decay: float,
) -> None:
    assert 0.0 <= decay < 1.0, f"ema decay out of range: {decay}"
    with torch.no_grad():
        for ema_param, student_param in zip(ema_model.parameters(), student_model.parameters()):
            if not student_param.requires_grad:
                continue
            ema_param.lerp_(student_param, 1.0 - decay)

        for ema_buffer, student_buffer in zip(ema_model.buffers(), student_model.buffers()):
            ema_buffer.copy_(student_buffer)


def run_training(
    cfg: TrainConfig,
    dist_env: DistEnv,
    model: torch.nn.Module,
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

    model.to(dist_env.device)
    if dist_env.is_distributed:
        ddp_find_unused_parameters = cfg.objective == "opd_kl"
        model = DDP(
            model,
            device_ids=[dist_env.local_rank],
            broadcast_buffers=False,
            find_unused_parameters=ddp_find_unused_parameters,
        )

    raw_model = _unwrap_model(model)
    ema_model = _build_ema_teacher(student_model=raw_model, cfg=cfg, device=dist_env.device)
    if dist_env.is_main and ema_model is not None:
        print(
            "EMA teacher enabled: "
            f"decay={cfg.ema_decay} start_step={cfg.ema_start_step}",
            flush=True,
        )

    optimizer, trainable_params = _build_optimizer(model, cfg)
    total_params = 0
    trainable_param_count = 0
    if dist_env.is_main:
        total_params = sum(param.numel() for param in model.parameters())
        trainable_param_count = sum(param.numel() for param in trainable_params)
        print(
            "Trainable parameter summary: "
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
        wandb_run.summary["objective"] = cfg.objective
        wandb_run.summary["total_params"] = total_params
        wandb_run.summary["trainable_params"] = trainable_param_count
        wandb_log = wandb_run.log
        wandb_finish = wandb_run.finish

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    scaler = None
    if dist_env.device.type == "cuda" and cfg.dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    dataloader = build_dataloader(
        cfg=cfg,
        tokenizer=tokenizer,
        rank=dist_env.rank,
        world_size=dist_env.world_size,
    )
    data_iter = iter(dataloader)

    global_step = 0
    if cfg.resume_path:
        global_step = load_checkpoint(
            checkpoint_path=cfg.resume_path,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=dist_env.device,
        )
        if dist_env.is_main:
            print(f"Resumed from step={global_step} path={cfg.resume_path}", flush=True)

    model.train()

    step_time = time.time()
    while global_step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)

        local_metrics = {
            "loss_total": torch.zeros([], device=dist_env.device),
            "loss_kl": torch.zeros([], device=dist_env.device),
            "loss_state": torch.zeros([], device=dist_env.device),
        }

        for _ in range(cfg.grad_accum_steps):
            try:
                batch_tokens = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_tokens = next(data_iter)

            batch_tokens = batch_tokens.to(dist_env.device, non_blocking=True)

            with _autocast_context(cfg, dist_env.device):
                if cfg.objective == "baseline_ce":
                    loss_total = _compute_baseline_ce(model, batch_tokens)
                    loss_kl = torch.zeros_like(loss_total)
                    loss_state = torch.zeros_like(loss_total)
                else:
                    opd_loss = _compute_opd_loss(
                        model=model,
                        teacher_model=ema_model if ema_model is not None else raw_model,
                        rollout_source_model=raw_model,
                        batch_tokens=batch_tokens,
                        cfg=cfg,
                    )
                    loss_total = opd_loss.total
                    loss_kl = opd_loss.kl
                    loss_state = opd_loss.state

            if not torch.isfinite(loss_total):
                raise FloatingPointError(f"Non-finite loss encountered at step {global_step}: {loss_total.item()}")

            loss_backward = loss_total / cfg.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss_backward).backward()
            else:
                loss_backward.backward()

            local_metrics["loss_total"] += loss_total.detach()
            local_metrics["loss_kl"] += loss_kl.detach()
            local_metrics["loss_state"] += loss_state.detach()

        if scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            optimizer.step()

        scheduler.step()
        global_step += 1
        if ema_model is not None and global_step >= cfg.ema_start_step:
            _update_ema_teacher(
                ema_model=ema_model,
                student_model=raw_model,
                decay=cfg.ema_decay,
            )

        should_log = global_step == 1 or global_step % cfg.log_interval == 0
        if should_log:
            elapsed = max(time.time() - step_time, 1e-6)
            step_time = time.time()

            mean_metrics: Dict[str, torch.Tensor] = {}
            for key, value in local_metrics.items():
                mean_value = value / cfg.grad_accum_steps
                mean_value = reduce_mean(mean_value, dist_env.world_size)
                mean_metrics[key] = mean_value

            grad_norm_tensor = grad_norm.detach().float()
            grad_norm_tensor = reduce_mean(grad_norm_tensor, dist_env.world_size)

            tokens_per_step = (
                cfg.sequence_length
                * cfg.micro_batch_size
                * cfg.grad_accum_steps
                * dist_env.world_size
            )
            tokens_per_sec = tokens_per_step / elapsed
            current_lr = scheduler.get_last_lr()[0]

            if dist_env.is_main:
                print(
                    (
                        f"step={global_step} "
                        f"loss={mean_metrics['loss_total'].item():.6f} "
                        f"kl={mean_metrics['loss_kl'].item():.6f} "
                        f"state={mean_metrics['loss_state'].item():.6f} "
                        f"grad_norm={grad_norm_tensor.item():.4f} "
                        f"lr={current_lr:.3e} "
                        f"tok/s={tokens_per_sec:.1f}"
                    ),
                    flush=True,
                )
                wandb_log(
                    {
                        "loss_total": mean_metrics["loss_total"].item(),
                        "loss_kl": mean_metrics["loss_kl"].item(),
                        "loss_state": mean_metrics["loss_state"].item(),
                        "grad_norm": grad_norm_tensor.item(),
                        "lr": current_lr,
                        "tokens_per_sec": tokens_per_sec,
                    },
                    step=global_step,
                )

        should_save = global_step % cfg.save_interval == 0 or global_step == cfg.max_steps
        if should_save:
            barrier()
            if dist_env.is_main:
                save_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=global_step,
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config_dict=cfg.as_dict(),
                    keep_last_k=cfg.keep_last_k_checkpoints,
                )
                print(f"Saved checkpoint: {save_path}", flush=True)
            barrier()
    wandb_finish()
