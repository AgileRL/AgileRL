# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Batch shapes and the micro-batch units the training-memory search tries."""

from __future__ import annotations

from agilerl.arena.memory.specs import ModelArch, ModelSpec
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.hpo import MutationSpec, mutation_ceiling
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO

# Mean row length as a share of the row budget (0.64 countdown, 0.56 Sudoku),
# from Qwen3-4B LoRA on 8x A100-80.
ROW_FILL = 0.6
# Step between packed token budgets above the longest row.
TOKEN_BUDGET_STEP = 1024


def data_parallel_size(spec: LLMAlgorithmSpec, n_training_gpus: int) -> int:
    """Trainer world folded by tensor-parallel degree.

    :param spec: Validated LLM algorithm spec.
    :param n_training_gpus: Trainer ranks, ``training.training_gpus_per_agent``.
    :return: Data-parallel ranks.
    :raises ValueError: When ``fsdp.tp`` does not divide the trainer world.
    """
    tp = 1 if spec.fsdp is None else spec.fsdp.tp
    if n_training_gpus % tp != 0:
        msg = (
            f"world_size ({n_training_gpus}) must be divisible by tp ({tp}) "
            "for Tensor Parallel."
        )
        raise ValueError(msg)
    return n_training_gpus // tp


def rollout_group_size(spec: LLMAlgorithmSpec) -> int:
    """Completions per prompt. SFT and DPO train one row per sample.

    :param spec: Validated LLM algorithm spec.
    :return: ``group_size`` for a GRPO spec, otherwise 1.
    """
    if isinstance(spec, GRPOSpec):
        return spec.group_size
    return 1


def engine_group_size(spec: LLMAlgorithmSpec, mutation: MutationSpec | None) -> int:
    """Completions per prompt a rollout engine's ``max_num_seqs`` is sized for.

    :param spec: Validated LLM algorithm spec.
    :param mutation: The manifest's mutation section, or ``None``.
    :return: :func:`rollout_group_size`, raised to its HPO ceiling when
        ``group_size`` mutates.
    """
    return mutation_ceiling(mutation, "group_size", rollout_group_size(spec))


def resolve_mini_batch_size(configured: int | None, per_rank: int) -> int:
    """Per-rank trajectories one optimizer step covers.

    :param configured: Pinned mini-batch, or ``None`` when it is omitted.
    :param per_rank: Trajectories that rank holds.
    :return: The pinned mini-batch, otherwise ``per_rank``.
    """
    if configured is not None:
        return configured
    return per_rank


def omitted_micro_batch_size(spec: LLMAlgorithmSpec, n_training_gpus: int) -> int:
    """Trajectories in one forward when the micro-batch is omitted.

    :param spec: Validated LLM algorithm spec.
    :param n_training_gpus: Trainer ranks, ``training.training_gpus_per_agent``.
    :return: ``max(batch_size // data_parallel, 1) * group_size``.
    """
    dp = data_parallel_size(spec, n_training_gpus)
    return max(spec.batch_size // dp, 1) * rollout_group_size(spec)


def require_batch_split(mini: int, micro: int, per_rank: int) -> None:
    """Raise when the mini-batch and micro-batch do not divide the rank.

    :param mini: Trajectories one optimizer step covers.
    :param micro: Rows in one forward on one rank.
    :param per_rank: Trajectories that rank holds.
    :raises ValueError: When ``mini`` does not divide ``per_rank``, or
        ``micro`` does not divide ``mini``.
    """
    if mini < 1 or per_rank % mini != 0:
        msg = (
            f"batch_size_per_process ({per_rank}) must be "
            f"divisible by mini_batch_size ({mini}): a rank "
            "must split into whole optimizer steps."
        )
        raise ValueError(msg)
    if micro < 1 or mini % micro != 0:
        msg = (
            f"mini_batch_size ({mini}) must be divisible by "
            f"micro_batch_size_per_gpu ({micro}): "
            "gradient_accumulation_steps = mini_batch_size / "
            "micro_batch_size_per_gpu must be a whole number of backward "
            "passes."
        )
        raise ValueError(msg)


def catalog_model(spec: LLMAlgorithmSpec) -> ModelSpec | None:
    """Training model from the bundled catalog, or None when the id is absent.

    :param spec: Validated LLM algorithm spec.
    :return: The spec's catalog model.
    """
    hub_id = spec.pretrained_model_name_or_path
    info = SUPPORTED_MODEL_INFO.get(hub_id) if hub_id is not None else None
    if info is None or hub_id is None:
        return None
    counted = info.inspected["num_params"]
    n_params = counted if isinstance(counted, int) else None
    return ModelSpec(
        model_id=hub_id,
        arch=ModelArch.from_hf_config(info.config),
        n_params=n_params,
    )


def micro_batch_row_candidates(mini: int) -> list[int]:
    """Row micro-batches that split a mini-batch into whole backward passes.

    :param mini: Rows one optimizer step covers.
    :return: Positive divisors of ``mini``, smallest first.
    """
    found: list[int] = []
    root = 1
    while root * root <= mini:
        if mini % root == 0:
            found.append(root)
            paired = mini // root
            if paired != root:
                found.append(paired)
        root += 1
    return sorted(found)


def micro_batch_token_candidates(row_budget: int, mini: int) -> list[int]:
    """Packed token budgets, smallest first.

    :param row_budget: Tokens of the longest row; the smallest budget.
    :param mini: Rows one optimizer step covers.
    :return: The longest row, each multiple of :data:`TOKEN_BUDGET_STEP`
        above it, and the whole mini-batch of longest rows.
    """
    mini_tokens = mini * row_budget
    budgets = [
        row_budget,
        *range(
            (row_budget // TOKEN_BUDGET_STEP + 1) * TOKEN_BUDGET_STEP,
            mini_tokens,
            TOKEN_BUDGET_STEP,
        ),
    ]
    if mini_tokens > row_budget:
        budgets.append(mini_tokens)
    return budgets


def rows_within_budget(mini: int, budget: int, row_budget: int) -> int:
    """Largest row micro-batch that divides the mini-batch and fits a token budget.

    :param mini: Rows one optimizer step covers.
    :param budget: Tokens of one packed micro-batch.
    :param row_budget: Tokens of the longest row.
    :return: The row count, at least 1.
    """
    within = [
        rows for rows in micro_batch_row_candidates(mini) if rows * row_budget <= budget
    ]
    return within[-1] if within else 1


def per_rank_trajectories(
    spec: LLMAlgorithmSpec, n_training_gpus: int
) -> tuple[int, int]:
    """Data-parallel width and the trajectories one trainer rank holds.

    :param spec: Validated LLM algorithm spec.
    :param n_training_gpus: Trainer ranks, ``training.training_gpus_per_agent``.
    :return: ``(data_parallel_size, trajectories)``.
    :raises ValueError: When the batch does not split across those ranks.
    """
    dp = data_parallel_size(spec, n_training_gpus)
    if spec.batch_size % dp != 0:
        msg = (
            f"Batch size ({spec.batch_size}) must be divisible by the data-parallel "
            f"size ({dp})."
        )
        raise ValueError(msg)
    return dp, omitted_micro_batch_size(spec, n_training_gpus)


def resolve_micro_batch_size_per_gpu(
    spec: LLMAlgorithmSpec, n_training_gpus: int
) -> int:
    """Rows in one training forward on one rank, before any memory search.

    :param spec: Validated LLM algorithm spec. Not written.
    :param n_training_gpus: Trainer ranks, ``training.training_gpus_per_agent``.
    :return: The set micro-batch, else the resolved mini-batch.
    :raises ValueError: When a set mini-batch does not divide the rank.
    """
    configured = spec.micro_batch_size_per_gpu
    if configured is not None:
        return int(configured)
    _, per_rank = per_rank_trajectories(spec, n_training_gpus)
    mini = resolve_mini_batch_size(spec.mini_batch_size, per_rank)
    require_batch_split(mini, mini, per_rank)
    return mini
