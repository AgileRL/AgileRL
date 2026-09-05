# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""TypedDict guards and batch normalization for LLM prompts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard

import torch

from agilerl.typing import PreferencePrompts, RolloutPrompt, SFTPrompts


def normalize_prompt_batch(
    prompts: RolloutPrompt | list[RolloutPrompt],
) -> list[RolloutPrompt]:
    """Normalize rollout prompts into a list-of-dicts per sample.

    Supports both a list of per-sample dicts and a single stacked dict whose
    tensor/list values are batched on dimension 0.

    :param prompts: The prompts to normalize.
    :type prompts: RolloutPrompt | list[RolloutPrompt]
    :return: One prompt dict per sample.
    :rtype: list[RolloutPrompt]
    """
    if isinstance(prompts, list):
        return prompts

    input_ids = prompts["input_ids"]
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() == 1:
        return [prompts]

    batch_size = int(input_ids.shape[0])
    if batch_size == 0:
        return []

    # Inspect each key once and write it into every output dict in one pass.
    # Keys not declared on ``RolloutPrompt`` (caller-supplied metadata) are
    # copied through unchanged, which a key-by-key typed construction can't do.
    samples: list[dict[str, object]] = [{} for _ in range(batch_size)]
    for key, value in prompts.items():
        if (
            isinstance(value, torch.Tensor)
            and value.dim() > 0
            and value.shape[0] == batch_size
        ):
            chunks = value.unbind(0) if value.dim() == 1 else value.split(1, dim=0)
            for sample, chunk in zip(samples, chunks, strict=True):
                sample[key] = chunk
        elif isinstance(value, list) and len(value) == batch_size:
            for sample, item in zip(samples, value, strict=True):
                sample[key] = item
        else:
            for sample in samples:
                sample[key] = value
    # Open dicts preserve undeclared metadata; the closed TypedDict return can't
    # name those keys, and a TypeGuard pass would only add a Python loop.
    return samples  # ty: ignore[invalid-return-type]


def is_rollout_prompt(obs: Mapping[str, object]) -> TypeGuard[RolloutPrompt]:
    """Check whether a mapping is a tokenized rollout prompt.

    :param obs: A prompt mapping returned by a rollout env.
    :type obs: Mapping[str, object]
    :return: ``True`` when the mapping carries prompt tokens.
    :rtype: TypeGuard[RolloutPrompt]
    """
    return isinstance(obs.get("input_ids"), torch.Tensor)


def is_preference_prompts(batch: Mapping[str, object]) -> TypeGuard[PreferencePrompts]:
    """Check whether a collated batch carries the chosen/rejected pair DPO needs.

    :param batch: A batch collated by an ``objective="preference"`` ``DatasetEnv``.
    :type batch: Mapping[str, object]
    :return: ``True`` when the batch carries both preference encodings.
    :rtype: TypeGuard[PreferencePrompts]
    """
    return isinstance(batch.get("chosen_input_ids"), torch.Tensor) and isinstance(
        batch.get("rejected_input_ids"), torch.Tensor
    )


def is_sft_prompts(batch: Mapping[str, object]) -> TypeGuard[SFTPrompts]:
    """Check whether a collated batch carries the prompt/response pair SFT needs.

    :param batch: A batch collated by an ``objective="sft"`` ``DatasetEnv``.
    :type batch: Mapping[str, object]
    :return: ``True`` when the batch carries the teacher-forced encoding.
    :rtype: TypeGuard[SFTPrompts]
    """
    return isinstance(batch.get("input_ids"), torch.Tensor) and isinstance(
        batch.get("response"), list
    )
