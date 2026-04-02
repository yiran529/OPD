from __future__ import annotations

from contextlib import nullcontext

import torch


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def score_continuation_from_ids(
    model: torch.nn.Module,
    device: torch.device,
    prompt_token_ids: list[int],
    continuation_token_ids: list[int],
    normalize_by_length: bool,
) -> float:
    assert prompt_token_ids, "prompt_token_ids must be non-empty"
    assert continuation_token_ids, "continuation_token_ids must be non-empty"

    # For multiple-choice scoring we evaluate log P(choice_text | prompt).
    # In ARC, `prompt` is the task prompt ("Question: ...\nAnswer:"),
    # while each answer option is scored as the continuation.
    full_ids = prompt_token_ids + continuation_token_ids
    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)

    with _autocast_context(device):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    logits = outputs.logits
    assert logits.size(1) == len(full_ids), "logits sequence length mismatch"

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    target_start = len(prompt_token_ids) - 1
    assert target_start >= 0, "prompt must contribute at least one next-token prediction"

    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)

    selected = token_log_probs[0, target_start:]
    assert selected.numel() == len(continuation_token_ids), "continuation token count mismatch"
    total_logprob = selected.sum()

    if normalize_by_length:
        total_logprob = total_logprob / selected.numel()

    return float(total_logprob.item())


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
