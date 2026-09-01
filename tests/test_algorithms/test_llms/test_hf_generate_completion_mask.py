# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The HuggingFace ``generate`` path must not train on prompt tokens.

``generate`` returns the prompt prefix *plus* the new tokens, so the completion
mask has to be built against the real prompt length. Passing ``None`` there marks
the whole sequence as generated and prompt positions enter the action mask and
the advantages — silently, and only on the non-vLLM path. These pin the contract
per algorithm so the vLLM and HF paths cannot drift apart again.
"""

from __future__ import annotations

import torch

from agilerl.algorithms import GRPO, LLMPPO, LLMREINFORCE
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.test_algorithms.test_llms.llm_helpers import create_module

_VOCAB = 64
_PAD = _VOCAB - 1
_PROMPT_LEN = 3


def _prompt() -> dict[str, torch.Tensor]:
    ids = torch.arange(1, _PROMPT_LEN + 1, dtype=torch.long).unsqueeze(0)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _cpu_algo(cls: type, *, value_head: bool = False, **kwargs: object) -> object:
    actor = create_module(input_size=6, max_tokens=4, vocab_size=_VOCAB, device="cpu")
    if value_head:
        # The value head sizes itself off the config; the dummy names it input_size.
        actor.config.hidden_size = actor.config.input_size
    defaults: dict[str, object] = {
        # LLMPPO needs a critic head on the actor; the other two do not.
        "actor_network": (
            AutoModelForCausalLMWithValueHead(actor) if value_head else actor
        ),
        "pad_token_id": _PAD,
        "pad_token": "<pad>",
        "batch_size": 1,
        "max_output_tokens": 4,
        "max_model_len": 12,
        "wrap": False,
        "gradient_checkpointing": False,
        "accelerator": None,
        "device": "cpu",
        "use_vllm": False,
    }
    defaults.update(kwargs)
    return cls(**defaults)


def _assert_prompt_positions_are_not_generated(algo: object) -> None:
    """No mask position inside the prompt may be ``True``."""
    result = algo.get_action([_prompt()], training=False)
    token_ids_list, completion_masks = result[0], result[1]
    assert token_ids_list, "the HF path returns one sequence per prompt"
    for token_ids, mask in zip(token_ids_list, completion_masks, strict=True):
        # generate() returns prompt + new tokens, so the sequence must have grown.
        assert token_ids.shape[-1] > _PROMPT_LEN
        # The mask drops the leading position, so prompt positions are [0, _PROMPT_LEN - 1).
        prompt_positions = mask[:, : _PROMPT_LEN - 1]
        assert not prompt_positions.any(), (
            "prompt tokens were marked as generated; build_completion_mask was "
            "given the wrong prompt_len"
        )
        # And the completion itself is not masked away wholesale.
        assert mask[:, _PROMPT_LEN - 1 :].any()


def test_grpo_hf_generate_masks_the_prompt() -> None:
    # group_size >= 2 is a GRPO constraint; evaluation mode does not fan out, so
    # one prompt still yields one sequence to check.
    _assert_prompt_positions_are_not_generated(_cpu_algo(GRPO, group_size=2))


def test_llmppo_hf_generate_masks_the_prompt() -> None:
    _assert_prompt_positions_are_not_generated(_cpu_algo(LLMPPO, value_head=True))


def test_llmreinforce_hf_generate_masks_the_prompt() -> None:
    _assert_prompt_positions_are_not_generated(_cpu_algo(LLMREINFORCE))
