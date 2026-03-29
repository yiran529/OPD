from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_entropy_corrupted_prefix(
    model: torch.nn.Module,
    context_tokens: torch.Tensor,
    clean_prefix_tokens: torch.Tensor,
    topk_ratio: float,
    topk_max: int,
) -> torch.Tensor:
    assert context_tokens.dim() == 2, f"context_tokens shape mismatch: expected rank=2, got shape={tuple(context_tokens.shape)}"
    assert clean_prefix_tokens.dim() == 2, (
        f"clean_prefix_tokens shape mismatch: expected rank=2, got shape={tuple(clean_prefix_tokens.shape)}"
    )
    assert context_tokens.size(0) == clean_prefix_tokens.size(0), "batch size mismatch between context and clean_prefix"
    assert context_tokens.size(1) > 0, "context_tokens must be non-empty"
    assert clean_prefix_tokens.size(1) > 0, "clean_prefix_tokens must be non-empty"
    assert 0.0 < topk_ratio <= 1.0, f"topk_ratio must be in (0,1], got {topk_ratio}"
    assert topk_max > 0, f"topk_max must be positive, got {topk_max}"

    prefix_len = clean_prefix_tokens.size(1)
    ratio_k = int(topk_ratio * prefix_len)
    topk = min(topk_max, ratio_k)
    assert topk > 0, (
        "corruption top-k is zero; increase topk_ratio or topk_max "
        f"(prefix_len={prefix_len}, ratio={topk_ratio}, max={topk_max})"
    )

    # ---- run teacher-forcing on clean prefix to score entropy per prefix position ----
    clean_prompt = torch.cat([context_tokens, clean_prefix_tokens], dim=1)
    clean_attention_mask = torch.ones_like(clean_prompt)
    outputs = model(
        input_ids=clean_prompt,
        attention_mask=clean_attention_mask,
        use_cache=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = getattr(outputs, "logits", None)
    assert logits is not None, "entropy corruption: logits missing"
    assert logits.dim() == 3 and logits.size(1) == clean_prompt.size(1), (
        "entropy corruption logits shape mismatch: "
        f"expected [batch,{clean_prompt.size(1)},vocab], got shape={tuple(logits.shape)}"
    )

    # For prefix token y_i, use p(. | x, y_{<i}) at index context_len + i - 1.
    prefix_start = context_tokens.size(1) - 1
    prefix_end = prefix_start + prefix_len
    prefix_logits = logits[:, prefix_start:prefix_end, :]
    assert prefix_logits.size(1) == prefix_len, (
        f"entropy corruption prefix logits length mismatch: expected={prefix_len} got={prefix_logits.size(1)}"
    )

    prefix_log_probs = F.log_softmax(prefix_logits.float(), dim=-1)
    prefix_probs = prefix_log_probs.exp()
    entropy = -(prefix_probs * prefix_log_probs).sum(dim=-1)
    predicted_tokens = prefix_logits.argmax(dim=-1)

    # ---- select top-k high-entropy positions and replace by model predictions ----
    topk_indices = torch.topk(entropy, k=topk, dim=1, largest=True, sorted=False).indices
    replace_mask = torch.zeros_like(clean_prefix_tokens, dtype=torch.bool)
    replace_mask.scatter_(dim=1, index=topk_indices, value=True)

    corrupted_prefix = clean_prefix_tokens.clone()
    corrupted_prefix[replace_mask] = predicted_tokens[replace_mask]
    return corrupted_prefix
