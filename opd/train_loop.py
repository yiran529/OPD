from __future__ import annotations

import copy
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

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
from opd.rollout import generate_rollout_tokens, sync_rollout_model
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


def _split_batch_segments(batch_tokens: torch.Tensor, cfg: TrainConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert batch_tokens.dim() == 2, f"batch_tokens shape mismatch: expected rank=2 [batch,seq+1], got shape={tuple(batch_tokens.shape)}"
    assert batch_tokens.size(1) == cfg.sequence_plus_one, (
        f"sequence length mismatch: expected={cfg.sequence_plus_one} got={batch_tokens.size(1)}"
    )

    context = batch_tokens[:, : cfg.context_len]
    clean_prefix = batch_tokens[:, cfg.context_len : cfg.context_len + cfg.prefix_len]
    continuation = batch_tokens[
        :,
        cfg.context_len + cfg.prefix_len : cfg.context_len + cfg.prefix_len + cfg.continuation_len,
    ]
    return context, clean_prefix, continuation


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
    rollout_model: torch.nn.Module,
    batch_tokens: torch.Tensor,
    cfg: TrainConfig,
    pad_token_id: int,
) -> OpdLossBundle:
    context, clean_prefix, _ = _split_batch_segments(batch_tokens, cfg)

    with torch.no_grad():
        corrupted_prefix, z_tokens = generate_rollout_tokens(
            rollout_model=rollout_model,
            context_tokens=context,
            prefix_len=cfg.prefix_len,
            continuation_len=cfg.continuation_len,
            temperature=cfg.rollout_temperature,
            top_p=cfg.rollout_top_p,
            pad_token_id=pad_token_id,
        )

    return compute_stepwise_opd_losses(
        model=model,
        context_tokens=context,
        corrupted_prefix_tokens=corrupted_prefix,
        clean_prefix_tokens=clean_prefix,
        z_tokens=z_tokens,
        lambda_state=cfg.lambda_state,
        state_key=cfg.state_key,
        state_time_stride=cfg.state_time_stride,
    )


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
        model = DDP(model, device_ids=[dist_env.local_rank], broadcast_buffers=False)

    raw_model = _unwrap_model(model)
    rollout_model = None
    if cfg.objective == "opd_kl":
        rollout_model = copy.deepcopy(raw_model)
        rollout_model.to(dist_env.device)
        sync_rollout_model(rollout_model, raw_model)

    optimizer, trainable_params = _build_optimizer(model, cfg)
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
            rollout_model=rollout_model,
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
                    assert rollout_model is not None, "rollout_model is required for opd_kl"
                    opd_loss = _compute_opd_loss(
                        model=model,
                        rollout_model=rollout_model,
                        batch_tokens=batch_tokens,
                        cfg=cfg,
                        pad_token_id=tokenizer.pad_token_id,
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

        if rollout_model is not None and global_step % cfg.rollout_sync_steps == 0:
            sync_rollout_model(rollout_model, raw_model)

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

            if dist_env.is_main:
                print(
                    (
                        f"step={global_step} "
                        f"loss={mean_metrics['loss_total'].item():.6f} "
                        f"kl={mean_metrics['loss_kl'].item():.6f} "
                        f"state={mean_metrics['loss_state'].item():.6f} "
                        f"grad_norm={grad_norm_tensor.item():.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} "
                        f"tok/s={tokens_per_sec:.1f}"
                    ),
                    flush=True,
                )

        should_save = global_step % cfg.save_interval == 0 or global_step == cfg.max_steps
        if should_save:
            barrier()
            if dist_env.is_main:
                save_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=global_step,
                    model=model,
                    rollout_model=rollout_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config_dict=cfg.as_dict(),
                    keep_last_k=cfg.keep_last_k_checkpoints,
                )
                print(f"Saved checkpoint: {save_path}", flush=True)
            barrier()
