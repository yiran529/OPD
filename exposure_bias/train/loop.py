from __future__ import annotations

import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import get_cosine_schedule_with_warmup

from exposure_bias.train.checkpoint import save_checkpoint
from exposure_bias.text_data import build_train_dataloader
from exposure_bias.train.config import ExposureBiasTrainConfig


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast_context(cfg: ExposureBiasTrainConfig, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if cfg.dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if cfg.dtype == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _build_optimizer(
    model: torch.nn.Module,
    cfg: ExposureBiasTrainConfig,
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


def _compute_baseline_ce(model: torch.nn.Module, batch_tokens: torch.Tensor) -> torch.Tensor:
    input_ids = batch_tokens[:, :-1]
    labels = batch_tokens[:, 1:]
    return model(input_ids=input_ids, labels=labels, use_cache=False).loss


def _optimizer_steps_per_epoch(
    num_micro_batches: int,
    grad_accum_steps: int,
) -> int:
    assert num_micro_batches > 0, "num_micro_batches must be positive"
    assert grad_accum_steps > 0, "grad_accum_steps must be positive"
    return (num_micro_batches + grad_accum_steps - 1) // grad_accum_steps


def run_training(
    cfg: ExposureBiasTrainConfig,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> None:
    _seed_all(cfg.seed)

    run_dir = Path(cfg.output_dir) / cfg.run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg.as_dict(), sort_keys=False),
        encoding="utf-8",
    )

    dataloader = build_train_dataloader(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        dataset_split=cfg.dataset_split,
        dataset_text_field=cfg.dataset_text_field,
        local_dataset_path=cfg.local_dataset_path,
        tokenizer=tokenizer,
        chunk_len=cfg.sequence_plus_one,
        batch_size=cfg.micro_batch_size,
        shuffle=cfg.shuffle,
    )
    num_micro_batches = len(dataloader)
    steps_per_epoch = _optimizer_steps_per_epoch(
        num_micro_batches=num_micro_batches,
        grad_accum_steps=cfg.grad_accum_steps,
    )
    total_training_steps = cfg.num_epochs * steps_per_epoch

    print(
        f"num_micro_batches={num_micro_batches} "
        f"steps_per_epoch={steps_per_epoch} "
        f"num_epochs={cfg.num_epochs} "
        f"total_optimizer_steps={total_training_steps}",
        flush=True,
    )

    optimizer, trainable_params = _build_optimizer(model=model, cfg=cfg)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_training_steps,
    )

    scaler = None
    if device.type == "cuda" and cfg.dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    model.train()
    step_time = time.time()
    global_step = 0
    for epoch_idx in range(1, cfg.num_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros([], device=device)
        accum_steps = 0

        for micro_step, batch_tokens in enumerate(dataloader, start=1):
            group_start = ((micro_step - 1) // cfg.grad_accum_steps) * cfg.grad_accum_steps + 1
            accum_target = min(cfg.grad_accum_steps, num_micro_batches - group_start + 1)
            batch_tokens = batch_tokens.to(device, non_blocking=True)
            with _autocast_context(cfg, device):
                loss = _compute_baseline_ce(model, batch_tokens)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered at epoch {epoch_idx} micro_step {micro_step}: {loss.item()}"
                )

            if scaler is not None:
                scaler.scale(loss / accum_target).backward()
            else:
                (loss / accum_target).backward()
            loss_total += loss.detach()
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

            if global_step % cfg.log_interval == 0 or global_step == 1:
                elapsed = time.time() - step_time
                print(
                    f"epoch={epoch_idx}/{cfg.num_epochs} "
                    f"step={global_step}/{total_training_steps} "
                    f"loss_ce={loss_total.item() / accum_steps:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.6e} "
                    f"grad_norm={float(grad_norm):.4f} "
                    f"dt={elapsed:.2f}s",
                    flush=True,
                )
                step_time = time.time()

            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros([], device=device)
            accum_steps = 0

        if epoch_idx % cfg.save_every_n_epochs == 0 or epoch_idx == cfg.num_epochs:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=global_step,
                model=model,
                config_dict=cfg.as_dict(),
                keep_last_k=cfg.keep_last_k_checkpoints,
            )
            print(f"saved checkpoint at epoch {epoch_idx}: {checkpoint_path}", flush=True)

    raw_model = _unwrap_model(model)
    raw_model.eval()
