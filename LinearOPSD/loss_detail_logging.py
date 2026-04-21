import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate.utils import gather_object


class LossDetailLogger:
    """Windowed high-loss token diagnostics for JSD distillation."""

    def __init__(
        self,
        *,
        tokenizer,
        output_dir,
        accelerator,
        log_steps=0,
        top_events=50,
        top_vocab=5,
        context_tokens=64,
        position_buckets=32,
        write_jsonl=True,
        temperature=1.0,
    ):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.accelerator = accelerator
        self.log_steps = int(log_steps)
        self.top_events = int(top_events)
        self.top_vocab = int(top_vocab)
        self.context_tokens = int(context_tokens)
        self.position_buckets = int(position_buckets)
        self.write_jsonl = bool(write_jsonl)
        self.temperature = float(temperature)

        assert self.log_steps >= 0, "loss_detail_log_steps must be non-negative"
        assert self.top_events > 0, "loss_detail_top_events must be positive"
        assert self.top_vocab > 0, "loss_detail_top_vocab must be positive"
        assert self.context_tokens >= 0, "loss_detail_context_tokens must be non-negative"
        assert self.position_buckets > 0, "loss_detail_position_buckets must be positive"
        assert self.temperature > 0.0, "loss detail temperature must be positive"

        self._events = []
        self._position_loss_chunks = []
        self._bucket_sum = torch.zeros(self.position_buckets, dtype=torch.float64)
        self._bucket_count = torch.zeros(self.position_buckets, dtype=torch.float64)
        self._last_seq_len = None

    @property
    def enabled(self):
        return self.log_steps > 0

    def should_record(self, global_step):
        return self.enabled and global_step % self.log_steps == 0

    def has_pending(self):
        return bool(self._events) or bool(self._position_loss_chunks) or bool(self._bucket_count.sum().item())

    def record(
        self,
        *,
        global_step,
        jsd_by_vocab,
        top_k_indices,
        labels,
        sampled_token_ids,
        student_logits,
        teacher_logits,
        inputs,
    ):
        if not self.should_record(global_step):
            return

        with torch.no_grad():
            mask = labels != -100
            if not mask.any():
                return

            position_loss = jsd_by_vocab.sum(dim=-1).float()
            valid_position_loss = position_loss[mask]
            self._position_loss_chunks.append(valid_position_loss.detach().cpu())

            self._record_position_buckets(position_loss=position_loss, mask=mask)
            self._record_top_events(
                global_step=global_step,
                position_loss=position_loss,
                mask=mask,
                jsd_by_vocab=jsd_by_vocab,
                top_k_indices=top_k_indices,
                labels=labels,
                sampled_token_ids=sampled_token_ids,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                inputs=inputs,
            )

    def flush(self, *, logs, global_step, main_print, wandb_module=None, report_to_wandb=False):
        if not self.enabled:
            return
        local_pending = 1 if self.has_pending() else 0
        if dist.is_available() and dist.is_initialized():
            pending_tensor = torch.tensor(local_pending, dtype=torch.int64, device=self.accelerator.device)
            dist.all_reduce(pending_tensor, op=dist.ReduceOp.MAX)
            if int(pending_tensor.item()) == 0:
                return
        elif not local_pending:
            return

        local_losses = self._sample_losses_for_gather()
        gathered_losses = gather_object(local_losses)
        all_losses = []
        for item in gathered_losses:
            if isinstance(item, list):
                all_losses.extend(item)
            else:
                all_losses.append(item)

        gathered_events = gather_object(self._events)
        all_events = []
        for item in gathered_events:
            if isinstance(item, list):
                all_events.extend(item)
            else:
                all_events.append(item)
        all_events.sort(key=lambda event: event["loss"], reverse=True)
        all_events = all_events[: self.top_events]
        event_steps = sorted({event["step"] for event in all_events})
        output_step = event_steps[0] if len(event_steps) == 1 else global_step

        bucket_sum = self._bucket_sum.clone()
        bucket_count = self._bucket_count.clone()
        if dist.is_available() and dist.is_initialized():
            bucket_sum_device = bucket_sum.to(self.accelerator.device)
            bucket_count_device = bucket_count.to(self.accelerator.device)
            dist.all_reduce(bucket_sum_device, op=dist.ReduceOp.SUM)
            dist.all_reduce(bucket_count_device, op=dist.ReduceOp.SUM)
            bucket_sum = bucket_sum_device.cpu()
            bucket_count = bucket_count_device.cpu()

        if all_losses:
            loss_tensor = torch.tensor(all_losses, dtype=torch.float32)
            quantiles = torch.quantile(
                loss_tensor,
                torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=torch.float32),
            )
            logs["loss_detail/p50"] = round(float(quantiles[0].item()), 6)
            logs["loss_detail/p90"] = round(float(quantiles[1].item()), 6)
            logs["loss_detail/p95"] = round(float(quantiles[2].item()), 6)
            logs["loss_detail/p99"] = round(float(quantiles[3].item()), 6)
            logs["loss_detail/max"] = round(float(loss_tensor.max().item()), 6)
            logs["loss_detail/num_tokens_sampled"] = len(all_losses)

        for bucket_idx in range(self.position_buckets):
            count = float(bucket_count[bucket_idx].item())
            if count > 0:
                avg = float(bucket_sum[bucket_idx].item() / count)
                logs[f"loss_detail/bucket_{bucket_idx:02d}_avg"] = round(avg, 6)

        if self.accelerator.is_main_process:
            self._print_summary(main_print=main_print, global_step=output_step, events=all_events, logs=logs)
            if self.write_jsonl and all_events:
                self._write_jsonl(global_step=output_step, events=all_events)
            if report_to_wandb and wandb_module is not None and getattr(wandb_module, "run", None) is not None:
                self._log_wandb_tables(
                    wandb_module=wandb_module,
                    global_step=output_step,
                    all_losses=all_losses,
                    events=all_events,
                    bucket_sum=bucket_sum,
                    bucket_count=bucket_count,
                )

        self.reset()

    def reset(self):
        self._events.clear()
        self._position_loss_chunks.clear()
        self._bucket_sum.zero_()
        self._bucket_count.zero_()
        self._last_seq_len = None

    def _record_position_buckets(self, *, position_loss, mask):
        seq_len = position_loss.shape[1]
        self._last_seq_len = seq_len
        bucket_ids = torch.div(
            torch.arange(seq_len, device=position_loss.device) * self.position_buckets,
            seq_len,
            rounding_mode="floor",
        ).clamp(max=self.position_buckets - 1)
        bucket_ids = bucket_ids.unsqueeze(0).expand_as(position_loss)
        valid_bucket_ids = bucket_ids[mask].long()
        valid_losses = position_loss[mask].double()

        bucket_sum = torch.bincount(
            valid_bucket_ids,
            weights=valid_losses,
            minlength=self.position_buckets,
        ).cpu()
        bucket_count = torch.bincount(valid_bucket_ids, minlength=self.position_buckets).double().cpu()
        self._bucket_sum += bucket_sum
        self._bucket_count += bucket_count

    def _record_top_events(
        self,
        *,
        global_step,
        position_loss,
        mask,
        jsd_by_vocab,
        top_k_indices,
        labels,
        sampled_token_ids,
        student_logits,
        teacher_logits,
        inputs,
    ):
        valid_count = int(mask.sum().item())
        if valid_count == 0:
            return

        k = min(self.top_events, valid_count)
        flat_loss = position_loss.masked_fill(~mask, -1.0).reshape(-1)
        values, flat_indices = torch.topk(flat_loss, k=k)

        seq_len = position_loss.shape[1]
        batch_indices = torch.div(flat_indices, seq_len, rounding_mode="floor")
        token_positions = flat_indices % seq_len

        student_top = self._top_prob_tokens(
            student_logits[batch_indices, token_positions], self.top_vocab
        )
        teacher_top = self._top_prob_tokens(
            teacher_logits[batch_indices, token_positions], self.top_vocab
        )
        teacher_actual_probs = self._actual_token_probs(
            logits=teacher_logits[batch_indices, token_positions],
            token_ids=sampled_token_ids[batch_indices, token_positions],
        )
        jsd_top = self._top_jsd_tokens(
            jsd_by_vocab=jsd_by_vocab,
            top_k_indices=top_k_indices,
            batch_indices=batch_indices,
            token_positions=token_positions,
        )

        metadata_list = inputs.get("linear_opsd_metadata")
        rank = int(getattr(self.accelerator, "process_index", 0))

        for event_idx in range(k):
            loss_value = float(values[event_idx].item())
            if loss_value < 0:
                continue

            batch_idx = int(batch_indices[event_idx].item())
            token_pos = int(token_positions[event_idx].item())
            token_id = int(sampled_token_ids[batch_idx, token_pos].item())
            metadata = metadata_list[batch_idx] if metadata_list is not None else {}

            event = {
                "step": int(global_step),
                "rank": rank,
                "sample_idx": batch_idx,
                "rollout_pos": token_pos,
                "loss": loss_value,
                "token_id": token_id,
                "token_text": self._decode_token(token_id),
                "prefix_text": self._decode_context(
                    sampled_token_ids[batch_idx], labels[batch_idx], 0, token_pos
                ),
                "suffix_text": self._decode_context(
                    sampled_token_ids[batch_idx], labels[batch_idx], token_pos + 1, sampled_token_ids.shape[1]
                ),
                "student_prompt_tail": self._decode_prompt_tail(
                    inputs=inputs,
                    key="student_prompts",
                    lengths_key="student_prompt_lengths_per_example",
                    batch_idx=batch_idx,
                ),
                "teacher_prompt_tail": self._decode_prompt_tail(
                    inputs=inputs,
                    key="teacher_prompts",
                    lengths_key="teacher_prompt_lengths_per_example",
                    batch_idx=batch_idx,
                ),
                "gold_prefix_tail": self._decode_id_tail(metadata.get("gold_prefix_ids", [])),
                "careless_tail_text": self._decode_id_tail(metadata.get("careless_token_ids", [])),
                "gold_next_text": self._decode_id_head(
                    metadata.get("gold_recovery_target_ids", metadata.get("gold_target_ids", []))
                ),
                "teacher_actual_token_prob": teacher_actual_probs[event_idx],
                "student_top_tokens": student_top[event_idx],
                "teacher_top_tokens": teacher_top[event_idx],
                "jsd_top_tokens": jsd_top[event_idx],
                "mixture_mode": metadata.get("mixture_mode"),
                "active_careless_rollout_len": metadata.get("active_careless_rollout_len"),
                "careless_deviated": metadata.get("careless_deviated"),
                "skip_kd": metadata.get("skip_kd"),
            }
            self._events.append(event)

        self._events.sort(key=lambda event: event["loss"], reverse=True)
        self._events = self._events[: self.top_events]

    def _actual_token_probs(self, *, logits, token_ids):
        probs = F.softmax(logits.float() / self.temperature, dim=-1)
        values = torch.gather(probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
        return [float(value.item()) for value in values.detach().cpu()]

    def _top_prob_tokens(self, logits, k):
        k = min(k, logits.shape[-1])
        probs = F.softmax(logits.float() / self.temperature, dim=-1)
        values, indices = torch.topk(probs, k=k, dim=-1)
        output = []
        for row_values, row_indices in zip(values.detach().cpu(), indices.detach().cpu()):
            output.append(
                [
                    {
                        "token_id": int(token_id.item()),
                        "token": self._decode_token(int(token_id.item())),
                        "prob": float(prob.item()),
                    }
                    for prob, token_id in zip(row_values, row_indices)
                ]
            )
        return output

    def _top_jsd_tokens(self, *, jsd_by_vocab, top_k_indices, batch_indices, token_positions):
        selected_jsd = jsd_by_vocab[batch_indices, token_positions].float()
        k = min(self.top_vocab, selected_jsd.shape[-1])
        values, local_indices = torch.topk(selected_jsd, k=k, dim=-1)

        if top_k_indices is None:
            token_indices = local_indices
        else:
            selected_top_k_indices = top_k_indices[batch_indices, token_positions]
            token_indices = torch.gather(selected_top_k_indices, dim=-1, index=local_indices)

        output = []
        for row_values, row_indices in zip(values.detach().cpu(), token_indices.detach().cpu()):
            output.append(
                [
                    {
                        "token_id": int(token_id.item()),
                        "token": self._decode_token(int(token_id.item())),
                        "contrib": float(value.item()),
                    }
                    for value, token_id in zip(row_values, row_indices)
                ]
            )
        return output

    def _sample_losses_for_gather(self):
        if not self._position_loss_chunks:
            return []
        losses = torch.cat(self._position_loss_chunks)
        max_items = 4096
        if losses.numel() > max_items:
            stride = max(1, losses.numel() // max_items)
            losses = losses[::stride][:max_items]
        return [float(value) for value in losses.tolist()]

    def _decode_token(self, token_id):
        return self.tokenizer.decode([int(token_id)], skip_special_tokens=False)

    def _decode_context(self, token_ids, labels, start, end):
        if self.context_tokens == 0:
            return ""
        if end <= start:
            return ""

        token_pos = end if start == 0 else start - 1
        if start == 0:
            left = max(start, token_pos - self.context_tokens)
            right = end
        else:
            left = start
            right = min(end, start + self.context_tokens)

        segment_ids = token_ids[left:right]
        segment_labels = labels[left:right]
        valid_ids = segment_ids[segment_labels != -100].detach().cpu().tolist()
        if not valid_ids:
            return ""
        return self.tokenizer.decode(valid_ids, skip_special_tokens=False)

    def _decode_prompt_tail(self, *, inputs, key, lengths_key, batch_idx):
        if self.context_tokens == 0 or key not in inputs:
            return ""

        token_tensor = inputs[key][batch_idx]
        lengths_tensor = inputs.get(lengths_key)
        if lengths_tensor is None:
            actual_length = int((token_tensor != self.tokenizer.pad_token_id).sum().item())
        else:
            actual_length = int(lengths_tensor[batch_idx].item())

        if actual_length <= 0:
            return ""
        start = max(0, actual_length - self.context_tokens)
        token_ids = token_tensor[start:actual_length].detach().cpu().tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _decode_id_tail(self, token_ids):
        if self.context_tokens == 0 or not token_ids:
            return ""
        return self.tokenizer.decode(list(token_ids[-self.context_tokens :]), skip_special_tokens=False)

    def _decode_id_head(self, token_ids):
        if self.context_tokens == 0 or not token_ids:
            return ""
        return self.tokenizer.decode(list(token_ids[: self.context_tokens]), skip_special_tokens=False)

    def _print_summary(self, *, main_print, global_step, events, logs):
        main_print(f"\n{'='*80}")
        main_print(f"LOSS DETAIL SUMMARY step={global_step}")
        if "loss_detail/p50" in logs:
            main_print(
                "position_loss "
                f"p50={logs['loss_detail/p50']:.6f} "
                f"p90={logs['loss_detail/p90']:.6f} "
                f"p95={logs['loss_detail/p95']:.6f} "
                f"p99={logs['loss_detail/p99']:.6f} "
                f"max={logs['loss_detail/max']:.6f}"
            )
        main_print("Top high-loss token events:")
        for rank, event in enumerate(events[: min(10, len(events))], start=1):
            main_print(
                f"{rank:02d}. loss={event['loss']:.6f} "
                f"rank={event['rank']} sample={event['sample_idx']} pos={event['rollout_pos']} "
                f"token_id={event['token_id']} token={event['token_text']!r} mode={event['mixture_mode']}"
            )
            main_print(f"    prefix_tail={event['prefix_text']!r}")
            main_print(f"    student_prompt_tail={event['student_prompt_tail']!r}")
            main_print(f"    teacher_prompt_tail={event['teacher_prompt_tail']!r}")
            main_print(f"    gold_next={event['gold_next_text']!r}")
            main_print(f"    suffix_head={event['suffix_text']!r}")
            main_print(f"    teacher_actual_prob={event['teacher_actual_token_prob']:.6f}")
            main_print(f"    student_top={self._format_token_probs(event['student_top_tokens'], 'prob')}")
            main_print(f"    teacher_top={self._format_token_probs(event['teacher_top_tokens'], 'prob')}")
            main_print(f"    jsd_top={self._format_token_probs(event['jsd_top_tokens'], 'contrib')}")
        main_print(f"{'='*80}\n")

    def _write_jsonl(self, *, global_step, events):
        output_dir = self.output_dir / "loss_detail"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"loss_events_step_{global_step}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _log_wandb_tables(self, *, wandb_module, global_step, all_losses, events, bucket_sum, bucket_count):
        payload = {}
        if all_losses:
            payload["loss_detail/position_loss_hist"] = wandb_module.Histogram(all_losses)
        if events:
            payload["loss_detail/high_loss_events"] = wandb_module.Table(
                columns=[
                    "step",
                    "rank",
                    "sample_idx",
                    "rollout_pos",
                    "loss",
                    "token_id",
                    "token_text",
                    "mixture_mode",
                    "prefix_text",
                    "suffix_text",
                    "student_prompt_tail",
                    "teacher_prompt_tail",
                    "gold_prefix_tail",
                    "careless_tail_text",
                    "gold_next_text",
                    "teacher_actual_token_prob",
                    "student_top_tokens",
                    "teacher_top_tokens",
                    "jsd_top_tokens",
                ],
                data=[
                    [
                        event["step"],
                        event["rank"],
                        event["sample_idx"],
                        event["rollout_pos"],
                        event["loss"],
                        event["token_id"],
                        event["token_text"],
                        event["mixture_mode"],
                        event["prefix_text"],
                        event["suffix_text"],
                        event["student_prompt_tail"],
                        event["teacher_prompt_tail"],
                        event["gold_prefix_tail"],
                        event["careless_tail_text"],
                        event["gold_next_text"],
                        event["teacher_actual_token_prob"],
                        self._format_token_probs(event["student_top_tokens"], "prob"),
                        self._format_token_probs(event["teacher_top_tokens"], "prob"),
                        self._format_token_probs(event["jsd_top_tokens"], "contrib"),
                    ]
                    for event in events
                ],
            )

        bucket_rows = []
        seq_len = self._last_seq_len or self.position_buckets
        for bucket_idx in range(self.position_buckets):
            count = float(bucket_count[bucket_idx].item())
            if count <= 0:
                continue
            start_pos = int(bucket_idx * seq_len / self.position_buckets)
            end_pos = int((bucket_idx + 1) * seq_len / self.position_buckets) - 1
            avg = float(bucket_sum[bucket_idx].item() / count)
            bucket_rows.append([global_step, bucket_idx, start_pos, end_pos, avg, count])

        if bucket_rows:
            payload["loss_detail/position_buckets"] = wandb_module.Table(
                columns=["step", "bucket", "start_pos", "end_pos", "avg_loss", "count"],
                data=bucket_rows,
            )

        if payload:
            wandb_module.log(payload)

    @staticmethod
    def _format_token_probs(items, value_key):
        return "; ".join(
            f"{item['token']!r} {item[value_key]:.4f}" for item in items
        )
