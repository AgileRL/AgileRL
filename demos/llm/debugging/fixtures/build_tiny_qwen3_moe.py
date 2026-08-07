#!/usr/bin/env python3
# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Build a tiny randomly-init Qwen3-MoE + digit tokenizer for EP smokes.

Vocab matches ``TinyDigitTokenizer`` ids (digits 0–4, pad=5, eos=6) so
ConstantTargetEnv(target_digit=\"3\") is learnable in a short GRPO probe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

VOCAB_SIZE = 7
TARGET_TOKEN_ID = 3


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {str(i): i for i in range(5)}
    vocab["[PAD]"] = 5
    vocab["[EOS]"] = 6
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[PAD]"))
    tok.pre_tokenizer = Split(pattern="", behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[PAD]",
    )


def build_model(seed: int = 0) -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        max_position_embeddings=128,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_cache=False,
        head_dim=16,
        bos_token_id=5,
        eos_token_id=6,
        pad_token_id=5,
    )
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config)
    # Digit prior (~0.2 hit on token 3 alone) plus a mild target boost yields
    # ~0.5 ConstantTarget hit-rate at temperature 1.0 — enough positive reward
    # for a short GRPO probe (~200 sample-steps) to climb toward ~1.0.
    with torch.no_grad():
        model.model.embed_tokens.weight.normal_(0.0, 0.02)
        model.lm_head.weight.normal_(0.0, 0.02)
        model.lm_head.weight[0:5].add_(0.25)
        model.lm_head.weight[TARGET_TOKEN_ID].add_(0.1)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "tiny_qwen3_moe",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = build_model(seed=args.seed)
    tokenizer = build_tokenizer()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Wrote tiny Qwen3-MoE fixture to {args.out}")


if __name__ == "__main__":
    main()
