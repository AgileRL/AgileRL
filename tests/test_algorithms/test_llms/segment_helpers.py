# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Segmented-episode batches shared by the LLM ``learn`` segment tests."""

from __future__ import annotations

import os
import socket
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from itertools import repeat
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from agilerl.algorithms.core import LLMAlgorithm
from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.distributed.fsdp import FSDPConfig
from agilerl.utils.segment_rows import SegmentRows

SEQ_LEN = 10
EPISODE_SEGMENTS = [
    EpisodeSegments(token_lengths=torch.tensor([4, 6])),
    None,
    EpisodeSegments(token_lengths=torch.tensor([5, 3])),
    None,
]
REAL_LENGTHS = [10, 8, 8, 10]
NUM_SEGMENT_ROWS = 6
# 24 action tokens over the 6 segment rows.
MEAN_SEGMENT_ROW_ACTION_TOKENS = 4.0


def episode_batch(pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Four ``T = 10`` episodes laid out per ``EPISODE_SEGMENTS``.

    Each segment's first token is prompt and the position predicting the next
    segment's first token is not an action.
    """
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, pad_token_id, (4, SEQ_LEN), generator=generator)
    mask = torch.zeros(4, SEQ_LEN - 1, dtype=torch.bool)
    for episode, segments in enumerate(EPISODE_SEGMENTS):
        lengths = (
            [REAL_LENGTHS[episode]]
            if segments is None
            else segments.token_lengths.tolist()
        )
        start = 0
        for length in lengths:
            mask[episode, start + 1 : start + length - 1] = True
            start += length
        ids[episode, start:] = pad_token_id
    return ids, mask


def segment_experiences(
    pad_token_id: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """``learn`` experiences of the ``EPISODE_SEGMENTS`` episodes."""
    ids, mask = episode_batch(pad_token_id)
    return list(ids.split(1)), list(mask.split(1)), torch.tensor([1.0, 0.0, 0.0, 1.0])


def episode_sampling_logps(pad_token_id: int) -> list[torch.Tensor]:
    """One flat vLLM sampling log-prob per action token of each episode."""
    _ids, mask = episode_batch(pad_token_id)
    generator = torch.Generator().manual_seed(5)
    return [
        -4.0 + 0.1 * torch.randn(int(row.sum()), generator=generator) for row in mask
    ]


def action_token_policy_loss(
    policy_hidden: torch.Tensor,
    _head_w: torch.Tensor,
    _head_b: torch.Tensor | None,
    _target_ids: torch.Tensor,
    mask: torch.Tensor,
    *_args: Any,
    **_kwargs: Any,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Liger fused policy loss stand-in: zero loss, ``kl`` = the micro-batch's action tokens."""
    zero = torch.tensor(0.0)
    return (policy_hidden * 0.0).sum(), (mask.sum().float(), zero, zero, zero)


def use_fake_liger_policy_loss(
    agent: LLMAlgorithm, monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    """Train ``agent`` on the Liger path with :func:`action_token_policy_loss` as its kernel."""
    monkeypatch.setattr(f"{module}.HAS_LIGER_KERNEL", True)
    monkeypatch.setattr(f"{module}.apply_fused_policy_loss", action_token_policy_loss)
    agent.use_liger_loss = True


def pad_to_eight_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run as one of two ranks whose other rank has 8 segment rows.

    The other rank's optimizer steps hold as many real micro-batches as this
    rank's.
    """
    for module in ("agilerl.algorithms.core.base", "agilerl.algorithms.grpo"):
        monkeypatch.setattr(f"{module}.get_world_size", lambda: 2)
        monkeypatch.setattr(
            f"{module}.allreduce_minmax_int",
            lambda value: (value, 8) if value == NUM_SEGMENT_ROWS else (value, value),
        )
    monkeypatch.setattr(
        "agilerl.algorithms.core.base.allreduce_sum_ints",
        lambda values: [2 * int(value) for value in values],
    )


def record_step_gradients(
    agent: LLMAlgorithm, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, torch.Tensor]]:
    """LoRA gradients the optimizer sees at each step of ``agent``."""
    recorded: list[dict[str, torch.Tensor]] = []
    optimizer_step = agent.optimizer.step

    def recording_step(*args: Any, **kwargs: Any) -> None:
        recorded.append(
            {
                name: param.grad.detach().clone()
                for name, param in agent.actor.named_parameters()
                if "lora_" in name and param.grad is not None
            }
        )
        optimizer_step(*args, **kwargs)

    monkeypatch.setattr(agent.optimizer, "step", recording_step)
    return recorded


def lora_weights(agent: LLMAlgorithm) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in agent.actor.named_parameters()
        if "lora_" in name
    }


def seed_lora_weights(agent: LLMAlgorithm) -> None:
    """Fill every LoRA weight from a fixed seed, however the agent was built."""
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, param in sorted(agent.actor.named_parameters()):
            if "lora_" in name:
                param.copy_(torch.randn(param.shape, generator=generator))


def rows_from_other_ranks(
    rows: SegmentRows,
    train_rows: np.ndarray,
    row_values: Sequence[tuple[torch.Tensor, float]],
    _pad_token_id: int,
    _shard_group_size: int | None,
) -> tuple[SegmentRows, list[torch.Tensor]]:
    """Training rows reversed under new episode numbers, as a cross-rank deal returns them."""
    order = train_rows[::-1].copy()
    dealt = replace(
        rows,
        token_ids=rows.token_ids[order],
        action_masks=rows.action_masks[order],
        row_episodes=rows.row_episodes[order] + 10,
        row_starts=rows.row_starts[order],
        row_ends=rows.row_ends[order],
        turn_ids=rows.turn_ids[order] if rows.turn_ids is not None else None,
    )
    return dealt, [value[order] for value, _ in row_values]


# Rank 0's group holds both segmented episodes (4 rows), rank 1's none (2 rows).
UNEVEN_ORDER = [0, 2, 1, 3]
RANK_EPISODES = (slice(0, 2), slice(2, 4))


def learn_uneven_episodes(
    agent: LLMAlgorithm, episodes: slice, sampling_logps: bool = False
) -> dict[str, float]:
    """One learn over ``episodes`` of the :data:`UNEVEN_ORDER` batch.

    :param sampling_logps: Whether the episodes carry vLLM sampling log-probs.
    """
    ids, mask, rewards = segment_experiences(agent.pad_token_id)
    order = UNEVEN_ORDER[episodes]
    episode_logps = episode_sampling_logps(agent.pad_token_id)
    return agent.learn(
        ([ids[i] for i in order], [mask[i] for i in order], rewards[order]),
        episode_segments=[EPISODE_SEGMENTS[i] for i in order],
        sampling_logps=[episode_logps[i] for i in order] if sampling_logps else None,
    )


def learn_rank_episodes(
    agent: LLMAlgorithm, rank: int, sampling_ranks: Sequence[int] = ()
) -> float:
    """Learn on ``rank``'s :data:`RANK_EPISODES`; report its padded training rows.

    :param sampling_ranks: Ranks whose episodes carry vLLM sampling log-probs.
    """
    metrics = learn_uneven_episodes(
        agent, RANK_EPISODES[rank], sampling_logps=rank in sampling_ranks
    )
    return metrics["train_rows_padded"]


def learn_with_nan_loss_on_rank_zero(
    agent: LLMAlgorithm,
    rank: int,
    scale_loss: Callable[[LLMAlgorithm, pytest.MonkeyPatch, Iterator[float]], None],
) -> tuple[str, bool]:
    """Learn on ``rank``'s :data:`RANK_EPISODES` with every rank 0 loss NaN.

    :param scale_loss: Makes each of ``agent``'s losses carry the next scale.
    :return: The error ``learn`` raised, and whether the weights stayed put.
    """
    scale_loss(agent, pytest.MonkeyPatch(), repeat(float("nan") if rank == 0 else 1.0))
    before = lora_weights(agent)
    with pytest.raises(ValueError, match="Loss is not finite") as error:
        learn_rank_episodes(agent, rank)
    after = lora_weights(agent)
    return str(error.value), all(
        torch.equal(after[name], weight) for name, weight in before.items()
    )


def balanced_learn_worker(
    rank: int,
    world_size: int,
    port: int,
    make_agent: Callable[..., LLMAlgorithm],
    learn: Callable[[LLMAlgorithm, int], Any],
    queue: Any,
) -> None:
    """One FSDP2 rank: report what ``learn`` returns for this rank, and the step grads."""
    try:
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
            }
        )
        # The default FSDP2 mesh follows the host accelerator (MPS on macOS);
        # these ranks train on CPU.
        torch._C._get_accelerator = lambda: torch.device("cpu")
        agent = make_agent(
            micro_batch_size_per_gpu=1,
            mini_batch_size=2,
            wrap=True,
            fsdp_config=FSDPConfig(optim_cpu_offload=False, param_dtype="float32"),
        )
        seed_lora_weights(agent)
        steps = record_step_gradients(agent, pytest.MonkeyPatch())
        report = learn(agent, rank)
        # NumPy arrays: a torch tensor in the queue shares memory that dies
        # with this process before the parent reads it.
        step_arrays = [
            {name: grad.numpy() for name, grad in step.items()} for step in steps
        ]
        queue.put((rank, "ok", (report, step_arrays)))
    except Exception:
        queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def spawn_balanced_learn(
    make_agent: Callable[..., LLMAlgorithm],
    learn: Callable[[LLMAlgorithm, int], Any] = learn_rank_episodes,
    world_size: int = 2,
) -> list[Any]:
    """Run :func:`balanced_learn_worker` on ``world_size`` gloo ranks; return each rank's report."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=balanced_learn_worker,
            args=(rank, world_size, port, make_agent, learn, queue),
        )
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results = sorted(queue.get(timeout=300) for _ in range(world_size))
    for proc in procs:
        proc.join(timeout=300)
    for rank, status, payload in results:
        assert status == "ok", f"rank {rank}: {payload}"
    return [payload for _, _, payload in results]


def rank_local_step_gradients(
    make_agent: Callable[..., LLMAlgorithm],
    monkeypatch: pytest.MonkeyPatch,
    sampling_ranks: Sequence[int] = (),
) -> list[dict[str, torch.Tensor]]:
    """Step gradient of each rank's :data:`RANK_EPISODES` learned alone on one process.

    :param sampling_ranks: Ranks whose episodes carry vLLM sampling log-probs.
    """
    gradients = []
    for rank, episodes in enumerate(RANK_EPISODES):
        agent = make_agent(micro_batch_size_per_gpu=1, mini_batch_size=2)
        seed_lora_weights(agent)
        steps = record_step_gradients(agent, monkeypatch)
        learn_uneven_episodes(agent, episodes, sampling_logps=rank in sampling_ranks)
        assert len(steps) == 1
        gradients.append(steps[0])
    return gradients
