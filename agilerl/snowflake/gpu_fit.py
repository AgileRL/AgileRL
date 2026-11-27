# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Whether a run's GPU request fits the account's GPU pools, and the refusals when it does not."""

from __future__ import annotations

import math
import re
from collections.abc import Callable

from agilerl.models.algorithms.opd import teacher_engine_gpus
from agilerl.models.algorithms.rollout_llm import RolloutLLMSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.snowflake.profile import GpuPool

# Arena's memory estimate reports a run that needs more GPUs than asked with this text.
SHORTFALL = re.compile(r"\d+ GPUs, the job needs (\d+)")


class GpuShortfallError(ValueError):
    """The estimator sized the run: it needs more GPUs, or larger nodes, than asked."""


def too_small_message(
    gpus: int, pools: list[tuple[str, GpuPool]], family: str | None
) -> str:
    """Refusal for families that can run ``gpus`` but whose GPU memory cannot hold the manifest.

    :param gpus: Requested GPU count.
    :param pools: Those families and their pools.
    :param family: Family the caller named, or ``None``.
    :return: The message, ending ``Nothing was started.``
    """
    named = sorted(pools)
    listed = "\n".join(f"{name} ({pool.gpu_memory_gib} GiB)" for name, pool in named)
    if family is not None:
        return f"{listed} cannot hold this manifest.\nNothing was started."
    return (
        f"No GPU family that can run {gpus} GPUs can hold this manifest.\n"
        f"Too small:\n{listed}\nNothing was started."
    )


def pool_gpus(pool: GpuPool) -> int:
    """GPUs on the pool's ``MAX_NODES``: ``max_gpus`` rounded up to whole nodes.

    :param pool: GPU pool from ``setup.json``.
    :return: The GPU count.
    """
    return math.ceil(pool.max_gpus / pool.gpus_per_node) * pool.gpus_per_node


def gpu_counts_message(gpus: int, pools: dict[str, GpuPool], family: str | None) -> str:
    """Refusal for a GPU count no pool can run, with the counts each one can.

    A pool runs a multiple of its GPUs per node, up to :func:`pool_gpus`.

    :param gpus: Requested GPU count.
    :param pools: Pools the run could use, keyed by family.
    :param family: Family the caller named, or ``None``.
    :return: The message, ending ``Nothing was started.``
    """
    names = sorted(pools) if family is None else [family]
    if family is None:
        first = f"No GPU pool can run {gpus} GPUs: a run takes whole nodes, up to the GPUs its pool holds."
    elif gpus % pools[family].gpus_per_node != 0:
        first = f"{family} has {pools[family].gpus_per_node} GPUs per node, so it cannot run {gpus} GPUs."
    else:
        first = f"The {family} pool holds {pool_gpus(pools[family])} GPUs, so it cannot run {gpus} GPUs."
    lines = [first]
    for name in names:
        per_node = pools[name].gpus_per_node
        held = pool_gpus(pools[name])
        listed = ", ".join(str(item) for item in range(per_node, held + 1, per_node))
        lines.append(
            f"Valid GPU counts on {name}: {listed} (multiples of {per_node}, "
            f"up to the {held} GPUs the pool holds)."
        )
    if any(gpus > pool_gpus(pools[name]) for name in names):
        lines.append("To run more GPUs, raise the pool's maximum GPUs first.")
    lines.append("Nothing was started.")
    return "\n".join(lines)


def one_node_gpus(manifest: TrainingManifest) -> int | None:
    """Fewest GPUs a reinforcement learning run needs on its one node.

    :param manifest: Validated training manifest.
    :return: One training GPU, the pinned engine width (1 when the run resolves
        it) and the teacher's GPUs; ``None`` for a run without rollout engines.
    """
    spec = manifest.algorithm
    if not isinstance(spec, RolloutLLMSpec):
        return None
    width = 1
    if spec.vllm_config is not None:
        width = spec.vllm_config.tensor_parallel_size or 1
    return 1 + width + teacher_engine_gpus(spec)


def one_node_refusal(
    manifest: TrainingManifest, family: str, pool: GpuPool, gpus: int, needed: int
) -> str:
    """Refusal for a reinforcement learning run that would not fit on one node.

    :param manifest: Validated training manifest.
    :param family: The pool's GPU family.
    :param pool: The family's pool.
    :param gpus: Requested GPU count.
    :param needed: GPUs the run needs on its one node.
    :return: The message, ending ``Nothing was started.``
    """
    per_node = pool.gpus_per_node
    if needed > per_node:
        detail = (
            f"{family} has {per_node} GPU(s) per node, and {manifest.algorithm.name} "
            f"needs at least {needed} on one node.\n"
            f"Use a GPU family with at least {needed} GPUs per node, and ask for at "
            "most its GPUs per node."
        )
    else:
        detail = (
            f"{family} has {per_node} GPUs per node, so {gpus} GPUs would span "
            f"{math.ceil(gpus / per_node)} nodes.\nAsk for {per_node} GPUs."
        )
    return (
        "On Snowflake, reinforcement learning runs on one GPU node so the trainer "
        f"can share weights with the rollout engines.\n{detail}\nNothing was started."
    )


def check_one_node(
    manifest: TrainingManifest, family: str, pool: GpuPool, gpus: int
) -> None:
    """Refuse a reinforcement learning run whose trainer and engines would not share one node.

    :param manifest: Validated training manifest.
    :param family: The pool's GPU family.
    :param pool: The family's pool.
    :param gpus: Requested GPU count.
    :raises GpuShortfallError: The node is too small, or ``gpus`` spans nodes.
    """
    needed = one_node_gpus(manifest)
    if needed is None or (needed <= pool.gpus_per_node and gpus <= pool.gpus_per_node):
        return
    raise GpuShortfallError(one_node_refusal(manifest, family, pool, gpus, needed))


def shortfall_needed(detail: str) -> int | None:
    """GPUs the job needs, when the estimator's error says it needs more than asked.

    :param detail: The estimator's error text.
    :return: The GPU count, or ``None`` for any other error.
    """
    match = SHORTFALL.search(detail)
    return None if match is None else int(match.group(1))


def shortfall_refusal(
    manifest: TrainingManifest, family: str, pool: GpuPool, needed: int
) -> str:
    """Refusal for a run the estimator says needs ``needed`` GPUs, with the counts to ask for.

    A reinforcement learning run gets :func:`one_node_refusal` instead.

    :param manifest: Validated training manifest.
    :param family: The pool's GPU family.
    :param pool: The family's pool.
    :param needed: GPUs the job lays out on.
    :return: The message, ending ``Nothing was started.``
    """
    if one_node_gpus(manifest) is not None:
        return one_node_refusal(manifest, family, pool, needed, needed)
    smallest = math.ceil(needed / pool.gpus_per_node) * pool.gpus_per_node
    lines = [
        (
            f"{manifest.algorithm.name} needs at least {needed} GPUs on {family}: "
            "training and rollout generation run on separate GPUs."
        ),
        f"Ask for {smallest} GPUs or more.",
    ]
    if smallest > pool_gpus(pool):
        lines.append(
            f"The {family} pool holds {pool_gpus(pool)} GPUs, so raise its maximum GPUs first."
        )
    lines.append("Nothing was started.")
    return "\n".join(lines)


def fitting_pools(
    pools: list[tuple[str, GpuPool]], fits: Callable[[str, GpuPool], bool]
) -> tuple[list[tuple[str, GpuPool]], str | None]:
    """Pools ``fits`` accepts, and the last shortfall refusal among the rest.

    :param pools: Families and pools to try.
    :param fits: Whether the manifest fits a family's pool. Raises
        :class:`GpuShortfallError` when the run needs more GPUs there.
    :return: The accepted pools, and a refusal to show when none is accepted.
    """
    answers = [(name, pool, fit_or_shortfall(fits, name, pool)) for name, pool in pools]
    holding = [(name, pool) for name, pool, answer in answers if answer is True]
    shortfalls = [answer for _, _, answer in answers if isinstance(answer, str)]
    return holding, shortfalls[-1] if shortfalls else None


def fit_or_shortfall(
    fits: Callable[[str, GpuPool], bool], name: str, pool: GpuPool
) -> bool | str:
    """``fits`` for one pool, or its shortfall refusal.

    :param fits: Whether the manifest fits a family's pool.
    :param name: The family.
    :param pool: Its pool.
    :return: What ``fits`` returned, or the :class:`GpuShortfallError` text.
    """
    try:
        return fits(name, pool)
    except GpuShortfallError as exc:
        return str(exc)
