from __future__ import annotations

from contextlib import nullcontext

import torch


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def score_continuation_batch_from_text(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    pad_token_id: int,
    batch_prompt_text: list[str],
    batch_continuation_text: list[str],
    normalize_by_length: bool,
) -> list[float]:
    assert batch_prompt_text, "batch_prompt_text must be non-empty"
    assert len(batch_prompt_text) == len(batch_continuation_text), (
        "batch prompt/continuation length mismatch"
    )

    # For multiple-choice scoring we evaluate log P(choice_text | prompt).
    # In ARC, `prompt` is the task prompt ("Question: ...\nAnswer:"),
    # while each answer option is scored as the continuation.
    batch_size = len(batch_prompt_text)
    full_batch: list[list[int]] = []
    prompt_lens: list[int] = []
    full_lens: list[int] = []
    continuation_lens: list[int] = []

    for prompt_text, continuation_text in zip(
        batch_prompt_text,
        batch_continuation_text,
    ):
        assert prompt_text, "prompt_text must be non-empty"
        assert continuation_text, "continuation_text must be non-empty"

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt_text + continuation_text, add_special_tokens=False)["input_ids"]
        continuation_len = len(full_ids) - len(prompt_ids)
        assert prompt_ids, "prompt tokenization must be non-empty"
        assert continuation_len > 0, "continuation must add at least one token"

        full_batch.append(full_ids)
        prompt_lens.append(len(prompt_ids))
        full_lens.append(len(full_ids))
        continuation_lens.append(continuation_len)

    max_len = max(full_lens)
    assert max_len >= 2, "full sequence length must be at least 2 tokens"

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    target_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool, device=device)

    for idx, (full_ids, prompt_len, full_len, continuation_len) in enumerate(
        zip(full_batch, prompt_lens, full_lens, continuation_lens)
    ):
        pad_len = max_len - full_len
        full_tensor = torch.tensor(full_ids, dtype=torch.long, device=device)
        input_ids[idx, pad_len:] = full_tensor
        attention_mask[idx, pad_len:] = 1

        start = pad_len + prompt_len - 1
        end = pad_len + full_len - 1
        assert start >= 0, "invalid continuation start index"
        assert end > start, "continuation target range must be non-empty"
        target_mask[idx, start:end] = True

        actual_targets = int(target_mask[idx].sum().item())
        assert actual_targets == continuation_len, (
            f"continuation target count mismatch: expected={continuation_len} got={actual_targets}"
        )

    with _autocast_context(device):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits
    assert logits.shape[:2] == input_ids.shape, "logits shape mismatch"

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]

    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)

    scores: list[float] = []
    for idx, continuation_len in enumerate(continuation_lens):
        mask = target_mask[idx]
        assert mask.any(), "no target positions found for continuation"
        selected = token_log_probs[idx, mask]
        assert selected.numel() == continuation_len, (
            f"continuation token count mismatch: expected={continuation_len} got={selected.numel()}"
        )
        total_logprob = selected.sum()

        if normalize_by_length:
            total_logprob = total_logprob / selected.numel()

        scores.append(float(total_logprob.item()))

    return scores


@torch.no_grad()
def score_continuation_from_text(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    pad_token_id: int,
    prompt_text: str,
    continuation_text: str,
    normalize_by_length: bool,
) -> float:
    scores = score_continuation_batch_from_text(
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=pad_token_id,
        batch_prompt_text=[prompt_text],
        batch_continuation_text=[continuation_text],
        normalize_by_length=normalize_by_length,
    )
    assert len(scores) == 1, "single-item score must return exactly one result"
    return scores[0]


def summarize_choice_scores(choice_scores: list[dict], gold_label: str) -> dict:
    assert choice_scores, "choice_scores must be non-empty"
    assert gold_label, "gold_label must be non-empty"

    by_label = {entry["label"]: entry for entry in choice_scores}
    if gold_label not in by_label:
        raise KeyError(f"gold_label not found in choice_scores: {gold_label}")

    pred_entry = max(choice_scores, key=lambda entry: entry["score"])
    wrong_scores = [entry["score"] for entry in choice_scores if entry["label"] != gold_label]
    assert wrong_scores, "multiple-choice scoring requires at least one non-gold option"

    gold_score = float(by_label[gold_label]["score"])
    best_non_gold_score = float(max(wrong_scores))
    return {
        "pred_label": pred_entry["label"],
        "is_correct": pred_entry["label"] == gold_label,
        "gold_score": gold_score,
        "best_non_gold_score": best_non_gold_score,
        "margin": gold_score - best_non_gold_score,
    }
