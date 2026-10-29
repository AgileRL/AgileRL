# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Trainer-side estimator settings of a run."""

from __future__ import annotations

from collections.abc import Collection

from agilerl.arena.memory.specs import Algorithm, LoraTargetScope, PackedMoeDispatch

# Module names that mean an adapter touches attention projections only.
ATTENTION_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query", "key", "value"}
)


def lora_target_scope(
    lora_target_modules: Collection[str] | str | None,
) -> LoraTargetScope:
    """Layers a language-model adapter touches.

    :param lora_target_modules: Language-model module names, or ``all-linear``.
    :return: ``attention-only`` when every named module is an attention projection.
    """
    attention_only = (
        bool(lora_target_modules)
        and not isinstance(lora_target_modules, str)
        and set(lora_target_modules) <= ATTENTION_MODULES
    )
    return "attention-only" if attention_only else "all-linear"


def packed_moe_dispatch(
    lora_target_parameters: Collection[str] | None,
    expert_lora_materialized: bool,
) -> PackedMoeDispatch:
    """Packed-expert adapter path of the trainer.

    :param lora_target_parameters: Packed MoE expert parameter paths, or ``None``.
    :param expert_lora_materialized: A packed-expert adapter runs PEFT's
        delta-materializing forward.
    :return: ``contracted`` for split expert LoRA, else ``materialized``.
    """
    if lora_target_parameters and not expert_lora_materialized:
        return "contracted"
    return "materialized"


def sampling_support_width(records_sampling_supports: bool, top_k: int) -> int:
    """Widest vLLM sampling support rollout records per token.

    :param records_sampling_supports: Rollouts record vLLM sampling supports.
    :param top_k: Sampling ``top_k``.
    :return: Support width, or 0 when no supports are recorded.
    """
    # Support frames are padded to at least two ids.
    return max(top_k, 2) if records_sampling_supports else 0


def support_scored_loss(algorithm: Algorithm, liger_loss: bool) -> bool:
    """Whether the gradient pass scores supported tokens from their support rows.

    :param algorithm: Objective, as :data:`~agilerl.arena.memory.specs.Algorithm`.
    :param liger_loss: The update runs the fused Liger loss:
        :func:`~agilerl.arena.memory.resolve.liger.liger_loss_blocker` is ``None``.
    :return: ``True`` unless the loss scores the full vocabulary.
    """
    # PPO and REINFORCE's Liger loss scores the full vocabulary.
    return not (algorithm in {"ppo", "reinforce"} and liger_loss)
