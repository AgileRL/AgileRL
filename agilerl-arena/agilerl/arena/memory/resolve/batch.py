# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Choose an omitted ``micro_batch_size_per_gpu`` from the training estimate."""

from __future__ import annotations

from agilerl.arena.memory.estimator import estimate_training
from agilerl.arena.memory.specs import (
    DeviceSpec,
    ModelArch,
    ModelSpec,
    TrainingSettings,
)
from agilerl.arena.models.algorithms.base import LLMAlgorithmSpec
from agilerl.arena.models.algorithms.grpo import GRPOSpec
from agilerl.arena.models.hpo import MutationSpec, mutation_ceiling
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO


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


def _divisors_largest_first(n: int) -> list[int]:
    """Positive divisors of ``n``, largest first."""
    found: list[int] = []
    root = 1
    while root * root <= n:
        if n % root == 0:
            found.append(root)
            paired = n // root
            if paired != root:
                found.append(paired)
        root += 1
    found.sort(reverse=True)
    return found


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
    spec: LLMAlgorithmSpec,
    n_training_gpus: int,
    device: DeviceSpec | None = None,
    model: ModelSpec | None = None,
    settings: TrainingSettings | None = None,
) -> int:
    """Rows in one training forward on one rank.

    :param spec: Validated LLM algorithm spec. Not written.
    :param n_training_gpus: Trainer ranks, ``training.training_gpus_per_agent``.
    :param device: Training GPU. Without one, an omitted field is the
        resolved mini-batch.
    :param model: Geometry the search estimates when the spec's hub id is in
        the catalog. Omitted, that catalog model is used. An id outside the
        catalog returns the resolved mini-batch.
    :param settings: Training settings the search prices. Required when a
        device and a catalog id are both present.
    :return: The explicit micro-batch, a divisor of the mini-batch that fits,
        the resolved mini-batch when the search does not run, or 1 when every
        divisor is over the device.
    :raises ValueError: When a pinned mini-batch does not divide the rank.
    """
    configured = spec.micro_batch_size_per_gpu
    if configured is not None:
        return int(configured)
    _, per_rank = per_rank_trajectories(spec, n_training_gpus)
    mini = resolve_mini_batch_size(spec.mini_batch_size, per_rank)
    # The omitted forward is the mini-batch: the largest count that divides it.
    require_batch_split(mini, mini, per_rank)
    candidates = _divisors_largest_first(mini)
    catalogued = catalog_model(spec)
    if device is None or catalogued is None:
        return mini
    if settings is None:
        msg = (
            "resolve_micro_batch_size_per_gpu needs training settings "
            "when a device is set"
        )
        raise ValueError(msg)
    search_model = model if model is not None else catalogued
    for micro in candidates:
        trial = settings.model_copy(update={"micro_batch_size": micro})
        breakdown = estimate_training(search_model, device, trial, orchestrated=True)
        if breakdown.fits_with_buffer:
            return micro
    return 1
