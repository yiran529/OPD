from __future__ import annotations

from contextlib import nullcontext

import torch


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def score_choice_logprob(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    prompt_text: str,
    choice_suffix: str,
    normalize_by_length: bool,
) -> float:
    assert prompt_text, "prompt_text must be non-empty"
    assert choice_suffix, "choice_suffix must be non-empty"

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + choice_suffix, add_special_tokens=False)["input_ids"]

    prompt_len = len(prompt_ids)
    full_len = len(full_ids)
    assert prompt_len > 0, "prompt token length must be > 0"
    assert full_len > prompt_len, "choice suffix must add at least one token"

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
    assert logits.size(1) == full_len, "logits sequence length mismatch"

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]

    position_ids = torch.arange(1, full_len, device=device)
    target_mask = position_ids >= prompt_len
    assert target_mask.any(), "no target positions found for choice suffix"

    log_probs = torch.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shifted_targets.unsqueeze(-1)).squeeze(-1)

    selected = token_log_probs[0, target_mask]
    total_logprob = selected.sum()

    if normalize_by_length:
        total_logprob = total_logprob / selected.numel()

    return float(total_logprob.item())
