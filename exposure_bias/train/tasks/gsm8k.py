from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset

from exposure_bias.text_data import load_hf_text_dataset


def split_gsm8k_answer(answer_text: str) -> tuple[str, str]:
    marker = "\n#### "
    assert marker in answer_text, "gsm8k answer must contain '\\n#### ' marker"
    rationale_text, final_answer = answer_text.rsplit(marker, 1)
    rationale_text = rationale_text.strip()
    final_answer = final_answer.strip()
    assert rationale_text, "gsm8k rationale must be non-empty"
    assert final_answer, "gsm8k final answer must be non-empty"
    return rationale_text, final_answer


def format_gsm8k_sft_text(
    question: str,
    rationale_text: str,
    final_answer: str,
) -> str:
    question = question.strip()
    rationale_text = rationale_text.strip()
    final_answer = final_answer.strip()
    assert question, "gsm8k question must be non-empty"
    assert rationale_text, "gsm8k rationale must be non-empty"
    assert final_answer, "gsm8k final answer must be non-empty"
    return (
        f"Question: {question}\n\n"
        f"Thoughts:\n{rationale_text}\n\n"
        f"Final Answer: {final_answer}"
    )


def iter_gsm8k_sft_texts(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    local_dataset_path: str | None,
) -> Iterable[str]:
    dataset = load_hf_text_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        local_dataset_path=local_dataset_path,
        streaming=False,
    )
    for row in dataset:
        assert "question" in row, "gsm8k row missing question"
        assert "answer" in row, "gsm8k row missing answer"
        question = row["question"]
        answer_text = row["answer"]
        assert isinstance(question, str), "gsm8k question must be str"
        assert isinstance(answer_text, str), "gsm8k answer must be str"
        rationale_text, final_answer = split_gsm8k_answer(answer_text)
        yield format_gsm8k_sft_text(
            question=question,
            rationale_text=rationale_text,
            final_answer=final_answer,
        )


def _build_token_stream(
    texts: Iterable[str],
    tokenizer,
) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    assert eos_token_id is not None, "tokenizer must provide eos_token_id"

    token_stream: list[int] = []
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        token_stream.extend(token_ids)
        token_stream.append(int(eos_token_id))

    assert token_stream, "gsm8k token stream is empty"
    return token_stream


def _build_chunks(
    token_stream: list[int],
    chunk_len: int,
) -> list[list[int]]:
    assert chunk_len > 0, "chunk_len must be positive"
    chunks: list[list[int]] = []
    for start in range(0, len(token_stream) - chunk_len + 1, chunk_len):
        chunks.append(token_stream[start : start + chunk_len])
    assert chunks, (
        f"not enough tokens to build any gsm8k training chunk: token_count={len(token_stream)} chunk_len={chunk_len}"
    )
    return chunks


class GSM8KChunkDataset(TorchDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config: str | None,
        dataset_split: str,
        local_dataset_path: str | None,
        tokenizer,
        chunk_len: int,
    ) -> None:
        super().__init__()
        texts = iter_gsm8k_sft_texts(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            local_dataset_path=local_dataset_path,
        )
        token_stream = _build_token_stream(texts=texts, tokenizer=tokenizer)
        self.chunks = _build_chunks(token_stream=token_stream, chunk_len=chunk_len)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.chunks[index], dtype=torch.long)


def build_gsm8k_train_dataloader(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    local_dataset_path: str | None,
    tokenizer,
    chunk_len: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = GSM8KChunkDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        local_dataset_path=local_dataset_path,
        tokenizer=tokenizer,
        chunk_len=chunk_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
