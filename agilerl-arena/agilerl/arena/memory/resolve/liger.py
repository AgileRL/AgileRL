# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Whether the fused Liger loss can serve a run, and what Liger's layer kernels save."""

from __future__ import annotations

from agilerl.arena.memory.specs import Algorithm, ModelArch

# Group-relative objectives that share GRPO's Liger kernel.
GROUP_RELATIVE_ALGORITHMS = frozenset({"grpo", "gspo", "cispo"})
# Algorithms whose Liger loss applies the model's final-logit softcap.
SOFTCAP_LIGER_ALGORITHMS = frozenset({"ppo", "reinforce", "sft"})
# Algorithms on AgileRL's fused policy-loss kernel, which chunks one whole
# sequence at a time above token level.
WHOLE_SEQUENCE_LIGER_ALGORITHMS = frozenset({"ppo", "reinforce"})
ABOVE_TOKEN_LEVELS = frozenset({"turn", "trajectory"})


def liger_kernel_arch(arch: ModelArch, liger_layers: bool) -> ModelArch:
    """Training geometry under the trainer's RMSNorm and gated-MLP modules.

    :param arch: Geometry from the checkpoint config, sized with Liger's layer
        kernels on the model types Liger supports.
    :param liger_layers: A trainer module runs a Liger layer kernel, which needs
        liger-kernel installed and a supported model type.
    :return: ``arch``, or the stock HF modules' saved tensors when none does.
    """
    if liger_layers:
        return arch
    return arch.model_copy(update={"fp32_norm_inputs": True, "stock_gated_mlp": True})


def liger_importance_sampling_level(
    algorithm: Algorithm, importance_sampling_level: str | None
) -> str:
    """Importance-sampling level the Liger kernel pools at.

    :param algorithm: Objective, as :data:`~agilerl.arena.memory.specs.Algorithm`.
    :param importance_sampling_level: Configured level. ``None`` is token level.
    :return: ``trajectory`` for ``gspo``, else the configured level.
    """
    if algorithm == "gspo":
        return "trajectory"
    return importance_sampling_level or "token"


def liger_loss_blocker(
    algorithm: Algorithm,
    importance_sampling_level: str | None,
    records_sampling_supports: bool,
    kl_advantage_shaping: bool,
    final_logit_softcap: float | None,
    reweights_tokens: bool,
    advantage_granularity: str | None,
) -> str | None:
    """Why the fused Liger loss cannot run this objective, or ``None`` when it can.

    :param algorithm: Objective, as :data:`~agilerl.arena.memory.specs.Algorithm`.
    :param importance_sampling_level: ``token``, ``turn``, ``trajectory``, or
        ``auto``. ``None`` is token level. ``auto`` passes; evaluate again on
        each batch's resolved level. ``gspo`` always pools at trajectory level.
    :param records_sampling_supports: Rollouts record vLLM sampling supports,
        as :func:`~agilerl.arena.models.algorithms.rollout_llm.records_sampling_supports`.
        Per batch: the batch carries them.
    :param kl_advantage_shaping: GRPO folds the KL term into the advantage.
    :param final_logit_softcap: The model's final-logit softcap, or ``None``.
    :param reweights_tokens: The update weights tokens from vLLM sampling
        log-probs. PPO and REINFORCE: vLLM importance-sampling correction is
        set. GRPO: it or off-policy token or sequence masks are set. Per
        batch: the batch carries them.
    :param advantage_granularity: GRPO's advantage granularity, or ``None``.
        Per batch: ``turn`` when the batch carries per-turn advantages.
    :return: A short reason, or ``None`` when the fused loss is valid.
    """
    group_relative = algorithm in GROUP_RELATIVE_ALGORITHMS
    if final_logit_softcap is not None and algorithm not in SOFTCAP_LIGER_ALGORITHMS:
        return (
            f"the {algorithm} Liger kernel does not apply the model's final-logit "
            f"softcap ({final_logit_softcap})"
        )
    if group_relative and records_sampling_supports:
        return (
            "the GRPO Liger kernel normalizes over the full vocabulary, not "
            "vLLM's sampling support (top_k > 0 and temperature > 0)"
        )
    if group_relative and kl_advantage_shaping:
        return "KL advantage shaping has no fused Liger kernel"
    level = liger_importance_sampling_level(algorithm, importance_sampling_level)
    if level not in ABOVE_TOKEN_LEVELS:
        return None
    if algorithm == "cispo" or (algorithm == "grpo" and level == "turn"):
        return (
            f"the {algorithm} Liger kernel has no '{level}' importance-sampling level"
        )
    # Pooled ratios cannot carry a per-token weight.
    if reweights_tokens:
        return (
            f"the {algorithm} Liger kernel weights tokens by vLLM sampling "
            f"log-probs only at token-level importance sampling, not '{level}'"
        )
    if group_relative and advantage_granularity == "turn":
        return (
            f"the {algorithm} Liger kernel takes one advantage per sequence at "
            "trajectory level, not per-turn advantages"
        )
    return None


def liger_whole_sequence_loss(
    algorithm: Algorithm, importance_sampling_level: str | None
) -> bool:
    """Whether the fused Liger loss holds one whole sequence's logits per chunk.

    :param algorithm: Objective, as :data:`~agilerl.arena.memory.specs.Algorithm`.
    :param importance_sampling_level: Resolved level. ``None`` is token level.
    :return: ``True`` for DPO, whose preference loss chunks over chosen and
        rejected pairs; for PPO and REINFORCE above token level; and for the
        group-relative kernel at trajectory level, which chunks one sequence
        at a time.
    """
    if algorithm == "dpo":
        return True
    level = liger_importance_sampling_level(algorithm, importance_sampling_level)
    if algorithm in GROUP_RELATIVE_ALGORITHMS:
        return level == "trajectory"
    return algorithm in WHOLE_SEQUENCE_LIGER_ALGORITHMS and level in ABOVE_TOKEN_LEVELS


def liger_head_grad(
    algorithm: Algorithm, liger_loss: bool, whole_sequence_loss: bool
) -> bool:
    """Whether the fused Liger loss keeps a weight-dtype grad of the whole lm_head.

    :param algorithm: Objective, as :data:`~agilerl.arena.memory.specs.Algorithm`.
    :param liger_loss: The update runs the fused Liger loss.
    :param whole_sequence_loss: As :func:`liger_whole_sequence_loss`.
    :return: ``True`` on the policy and selective kernels at token tiles. SFT's
        cross-entropy skips the frozen head's grad, and a whole-sequence tile
        already includes it.
    """
    return liger_loss and not whole_sequence_loss and algorithm != "sft"


def liger_loss_layout(
    algorithm: Algorithm,
    importance_sampling_level: str | None,
    records_sampling_supports: bool,
    kl_advantage_shaping: bool,
    final_logit_softcap: float | None,
    reweights_tokens: bool,
    advantage_granularity: str | None,
) -> tuple[bool, bool]:
    """Whether the fused Liger loss serves the update's batches, and whether it holds whole sequences.

    :param algorithm: Objective, as :data:`~agilerl.arena.memory.specs.Algorithm`.
    :param importance_sampling_level: Level the batches resolve to. ``None`` is token level.
    :param records_sampling_supports: As :func:`liger_loss_blocker`.
    :param kl_advantage_shaping: As :func:`liger_loss_blocker`.
    :param final_logit_softcap: As :func:`liger_loss_blocker`.
    :param reweights_tokens: As :func:`liger_loss_blocker`.
    :param advantage_granularity: As :func:`liger_loss_blocker`.
    :return: ``(runs, whole_sequence)``: :func:`liger_loss_blocker` passes, and
        :func:`liger_whole_sequence_loss` when it does.
    """
    runs = (
        liger_loss_blocker(
            algorithm,
            importance_sampling_level,
            records_sampling_supports,
            kl_advantage_shaping,
            final_logit_softcap,
            reweights_tokens,
            advantage_granularity,
        )
        is None
    )
    return runs, runs and liger_whole_sequence_loss(
        algorithm, importance_sampling_level
    )
