# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Lightweight tokenizer stand-in for tests that must not import transformers.

For the vendored Qwen2 weights and HuggingFace tokenizer, use
``tests.TINY_LLM_FIXTURE_PATH`` (``tests/assets/tiny_llm/``, built via
``build_tiny_llm_fixture.py``).
"""

from __future__ import annotations

import torch


class TinyTokenizer:
    """Encode/decode text to small token ids for lightweight LLM env tests."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        tokens = [((ord(ch) % 50) + 1) for ch in text][:16]
        return tokens or [1]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        chars = []
        for token in token_ids:
            token_id = int(token)
            if skip_special_tokens and token_id == self.pad_token_id:
                continue
            chars.append(chr(((token_id - 1) % 26) + 97))
        return "".join(chars)

    def __call__(
        self,
        texts,
        return_tensors: str = "pt",
        padding: bool = True,
        padding_side: str = "left",
        return_attention_mask: bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, return_attention_mask, add_special_tokens
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(text) for text in texts]
        max_len = max(len(item) for item in encoded) if padding else None
        padded_ids = []
        padded_masks = []
        for item in encoded:
            if max_len is None:
                padded_ids.append(item)
                padded_masks.append([1] * len(item))
                continue
            pad = max_len - len(item)
            if padding_side == "left":
                ids = [self.pad_token_id] * pad + item
                mask = [0] * pad + [1] * len(item)
            else:
                ids = item + [self.pad_token_id] * pad
                mask = [1] * len(item) + [0] * pad
            padded_ids.append(ids)
            padded_masks.append(mask)
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }
