from __future__ import annotations

import hashlib
import math
import random


def _stable_seed(example_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{example_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _candidate_token_ids(tokenizer) -> list[int]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Tokenizer must define a positive vocab_size for random token perturbation")

    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    candidates = [token_id for token_id in range(vocab_size) if token_id not in special_ids]
    if not candidates:
        raise ValueError("Random token perturbation found no non-special token ids")
    return candidates


def apply_random_token_insertion(
    tokenizer,
    prompt_token_ids: list[int],
    perturb_ratio: float,
    perturb_seed: int,
    example_id: str,
    perturb_min_tokens: int,
) -> tuple[list[int], dict]:
    assert prompt_token_ids, "prompt_token_ids must be non-empty"
    assert 0.0 <= perturb_ratio <= 1.0, "perturb_ratio must be in [0, 1]"
    assert perturb_min_tokens >= 0, "perturb_min_tokens must be >= 0"

    num_insertions = int(math.floor(perturb_ratio * len(prompt_token_ids)))
    if perturb_ratio > 0.0:
        num_insertions = max(num_insertions, perturb_min_tokens)

    if num_insertions == 0:
        return list(prompt_token_ids), {"num_insertions": 0, "positions": [], "token_ids": []}

    rng = random.Random(_stable_seed(example_id=example_id, base_seed=perturb_seed))
    candidate_ids = _candidate_token_ids(tokenizer)

    scheduled_insertions = []
    for _ in range(num_insertions):
        position = rng.randrange(len(prompt_token_ids) + 1)
        token_id = rng.choice(candidate_ids)
        scheduled_insertions.append((position, token_id))

    scheduled_insertions.sort(key=lambda item: item[0])
    perturbed = list(prompt_token_ids)
    offset = 0
    for position, token_id in scheduled_insertions:
        perturbed.insert(position + offset, token_id)
        offset += 1

    return perturbed, {
        "num_insertions": num_insertions,
        "positions": [position for position, _ in scheduled_insertions],
        "token_ids": [token_id for _, token_id in scheduled_insertions],
    }

