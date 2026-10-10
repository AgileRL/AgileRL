# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`~agilerl.llm_envs.EnvResponse` and :class:`~agilerl.llm_envs.AsyncBatchCollector`."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch

from agilerl.components.llm_rollout_data import EpisodeSegments
from agilerl.llm_envs import (
    AsyncBatchCollector,
    EnvResponse,
    RolloutCollector,
    RolloutHarness,
)
from tests.helpers.rollout_doubles import FakeEnvClient

EpisodeTensors = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    EpisodeSegments | None,
]


def _episode_tensors() -> EpisodeTensors:
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(ids)
    turns = torch.zeros_like(ids)
    rewards = torch.tensor([1.0])
    return ids, mask, turns, rewards, None, None, None


class StubCollector:
    """In-process collector whose episode methods return known payloads."""

    num_envs = 1

    def __init__(self) -> None:
        self.io_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="env-io"
        )
        self.reset_calls: list[
            tuple[str, int | None, tuple[int | None, int | None] | None]
        ] = []
        self.eval_reset_calls: list[tuple[str, int | None]] = []
        self.step_calls: list[tuple[str, torch.Tensor]] = []
        self.finalize_calls: list[tuple[str, bool]] = []
        self.closed = False

    def reset_episode(
        self,
        episode_id: str,
        logical_slot: int | None = None,
        *,
        task: tuple[int | None, int | None] | None = None,
    ) -> tuple[dict[str, str], dict[str, int | None]]:
        self.reset_calls.append((episode_id, logical_slot, task))
        return (
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)},
            {"logical_slot": logical_slot},
        )

    def reset_eval_episode(
        self,
        episode_id: str,
        row_index: int | None,
    ) -> tuple[dict[str, str], dict[str, int | None]]:
        self.eval_reset_calls.append((episode_id, row_index))
        return (
            {"input_ids": torch.tensor([[3, 4]], dtype=torch.long)},
            {"row_index": row_index},
        )

    def step_episode(
        self,
        episode_id: str,
        token_ids: torch.Tensor,
        prompt_token_len: int | None = None,
    ) -> tuple[dict[str, str], float, bool, bool, dict[str, str]]:
        self.step_calls.append((episode_id, token_ids))
        return {"text": f"next-{episode_id}"}, 1.5, True, False, {"source": "step"}

    def finalize_episode(
        self,
        episode_id: str,
        *,
        missing_ok: bool = True,
    ) -> EpisodeTensors | None:
        self.finalize_calls.append((episode_id, missing_ok))
        if missing_ok:
            return None
        return _episode_tensors()

    def close(self) -> None:
        self.io_executor.shutdown(wait=False, cancel_futures=True)
        self.closed = True


class TestEnvResponse:
    def test_stores_episode_payload_fields(self) -> None:
        response = EnvResponse(
            episode_id="ep-1",
            observation={"input_ids": [1, 2]},
            reward=0.5,
            terminated=False,
            truncated=True,
            info={"seed": 7},
        )

        assert response.episode_id == "ep-1"
        assert response.observation == {"input_ids": [1, 2]}
        assert response.reward == 0.5
        assert response.terminated is False
        assert response.truncated is True
        assert response.info == {"seed": 7}


class TestEnvResponseDoneProperty:
    @pytest.mark.parametrize(
        ("terminated", "truncated", "done"),
        [(False, False, False), (True, False, True), (False, True, True)],
    )
    def test_done_is_terminated_or_truncated(
        self, terminated: bool, truncated: bool, done: bool
    ) -> None:
        response = EnvResponse(
            episode_id="ep-1",
            observation={},
            reward=0.0,
            terminated=terminated,
            truncated=truncated,
            info={},
        )

        assert response.done is done


class TestAsyncBatchCollectorReset:
    def test_reset_returns_env_response_from_stub_reset_episode(self) -> None:
        collector = AsyncBatchCollector(StubCollector())

        try:
            response = asyncio.run(collector.reset("ep-9", logical_slot_idx=3))
        finally:
            collector.close()

        assert isinstance(response, EnvResponse)
        assert response.episode_id == "ep-9"
        assert torch.equal(response.observation["input_ids"], torch.tensor([[1, 2]]))
        assert response.reward == 0.0
        assert response.done is False
        assert response.info == {"logical_slot": 3}

    def test_reset_forwards_task_to_reset_episode(self) -> None:
        inner = StubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            response = asyncio.run(collector.reset("ep-task", task=(42, 3)))
        finally:
            collector.close()

        assert response.episode_id == "ep-task"
        assert inner.reset_calls == [("ep-task", None, (42, 3))]


class TestAsyncBatchCollectorResetEval:
    def test_reset_eval_returns_the_held_out_reset(self) -> None:
        inner = StubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            response = asyncio.run(collector.reset_eval("eval-1", 3))
        finally:
            collector.close()

        assert inner.eval_reset_calls == [("eval-1", 3)]
        assert inner.reset_calls == []
        assert response.episode_id == "eval-1"
        assert torch.equal(response.observation["input_ids"], torch.tensor([[3, 4]]))
        assert response.done is False
        assert response.info == {"row_index": 3}


class TestAsyncBatchCollectorStep:
    def test_step_forwards_token_ids_and_returns_env_response(self) -> None:
        inner = StubCollector()
        collector = AsyncBatchCollector(inner)
        token_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)

        try:
            response = asyncio.run(collector.step("ep-2", token_ids=token_ids))
        finally:
            collector.close()

        assert isinstance(response, EnvResponse)
        assert response.episode_id == "ep-2"
        assert response.observation == {"text": "next-ep-2"}
        assert response.reward == 1.5
        assert response.terminated is True
        assert response.truncated is False
        assert response.info == {"source": "step"}
        assert len(inner.step_calls) == 1
        called_id, called_ids = inner.step_calls[0]
        assert called_id == "ep-2"
        assert torch.equal(called_ids, token_ids)

    def test_truncated_step_is_truncated_but_not_terminated(self) -> None:
        # Arrange
        inner = StubCollector()
        inner.step_episode = lambda *_args, **_kwargs: ({}, 0.0, False, True, {})
        collector = AsyncBatchCollector(inner)

        # Act
        try:
            response = asyncio.run(
                collector.step("ep-2", token_ids=torch.tensor([[1]]))
            )
        finally:
            collector.close()

        # Assert
        assert response.truncated is True
        assert response.terminated is False

    def test_step_rejects_completion_ids_keyword(self) -> None:
        collector = AsyncBatchCollector(StubCollector())
        token_ids = torch.tensor([[1]], dtype=torch.long)

        bad_kwargs = {"completion_ids": token_ids}
        try:
            with pytest.raises(TypeError, match="completion_ids"):
                asyncio.run(collector.step("ep-2", **bad_kwargs))
        finally:
            collector.close()


class TestAsyncBatchCollectorFinalizeEpisode:
    def test_finalize_episode_missing_ok_returns_none(self) -> None:
        inner = StubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            result = asyncio.run(collector.finalize_episode("ep-3"))
        finally:
            collector.close()

        assert result is None
        assert inner.finalize_calls == [("ep-3", True)]

    def test_finalize_episode_returns_tensors_when_present(self) -> None:
        inner = StubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            result = asyncio.run(
                collector.finalize_episode("ep-4", missing_ok=False),
            )
        finally:
            collector.close()

        assert result is not None
        expected = _episode_tensors()
        for got, want in zip(result, expected, strict=True):
            if want is None:
                assert got is None
            else:
                assert torch.equal(got, want)
        assert inner.finalize_calls == [("ep-4", False)]


class TestAsyncBatchCollectorClose:
    def test_close_returns_while_a_slot_call_is_blocked(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingCollector:
            num_envs = 1

            def __init__(self) -> None:
                self.io_executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="env-io"
                )

            def reset_episode(
                self,
                episode_id: str,
                logical_slot: int | None = None,
                *,
                task: tuple[int | None, int | None] | None = None,
            ) -> tuple[dict[str, str], dict[str, Any]]:
                started.set()
                release.wait(timeout=30)
                return {"text": f"hung-{episode_id}-{logical_slot}-{task}"}, {}

            def close(self) -> None:
                return None

        collector = AsyncBatchCollector(BlockingCollector())
        thread = threading.Thread(
            target=lambda: asyncio.run(collector.reset("ep-hung")),
            daemon=True,
        )
        thread.start()
        assert started.wait(timeout=2)

        started_s = time.monotonic()
        collector.close()
        elapsed_s = time.monotonic() - started_s
        release.set()
        thread.join(timeout=2)

        assert elapsed_s < 1.0


class _ChrTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"

    def __call__(self, texts, **kwargs):
        ids = [[ord(c) for c in texts[0]]]
        tokens = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(int(token)) for token in ids)


class _OverflowClient(FakeEnvClient):
    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        del seed, row_index
        self.reset_calls += 1
        self._episode_steps = 0
        return "x" * 80, {}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        msg = "turn-0 overflow must not generate"
        raise AssertionError(msg)


def _harness_factory(client: FakeEnvClient, **kwargs: Any) -> Any:
    def factory() -> RolloutHarness:
        return RolloutHarness(
            client,
            _ChrTokenizer(),
            apply_chat_template=False,
            **kwargs,
        )

    return factory


class TestAsyncBatchCollectorResetOverflow:
    def test_reset_marks_truncated_on_turn_zero_overflow(self) -> None:
        client = _OverflowClient()
        inner = RolloutCollector(
            env_factory=_harness_factory(client, max_turns=2, max_model_len=20),
            batch_size=1,
            group_size=1,
        )
        collector = AsyncBatchCollector(inner)

        try:
            response = asyncio.run(collector.reset("ep-ov"))
        finally:
            collector.close()

        assert response.truncated is True
        assert response.terminated is False
        assert response.observation == {}
        assert client.step_calls == 0


class _RowClient(FakeEnvClient):
    """Fake client over an 8-row task list recording each reset's ``(seed, row)``."""

    def __init__(self) -> None:
        super().__init__(dataset_size=8)
        self.resets: list[tuple[int | None, int | None]] = []

    def reset(
        self, seed: int | None = None, *, row_index: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        self.resets.append((seed, row_index))
        return super().reset(seed, row_index=row_index)


class TestAsyncBatchCollectorAssignGroupTask:
    def test_concurrent_groups_draw_distinct_rows_shared_by_members(self) -> None:
        # Arrange
        client = _RowClient()
        inner = RolloutCollector(
            env_factory=_harness_factory(client, max_turns=2),
            batch_size=8,
            group_size=2,
            base_seed=0,
        )
        collector = AsyncBatchCollector(inner)

        async def start_group(group_seed: int) -> None:
            task = await collector.assign_group_task(group_seed)
            await asyncio.gather(
                *(
                    collector.reset(f"g{group_seed}-m{member}", task=task)
                    for member in range(2)
                ),
            )

        async def start_overlapping_groups() -> int:
            await asyncio.gather(*(start_group(seed) for seed in range(8)))
            return collector.active_episode_count()

        # Act
        try:
            live = asyncio.run(start_overlapping_groups())
        finally:
            collector.close()

        # Assert
        group_tasks = set(client.resets)
        assert live == 16
        assert len(client.resets) == 16
        assert all(client.resets.count(task) == 2 for task in group_tasks)
        assert len({seed for seed, _row in group_tasks}) == 8
        assert sorted(row for _seed, row in group_tasks) == list(range(8))


class TestAsyncBatchCollectorRecordGroupOutcome:
    def test_outcomes_reach_the_collectors_row_weights(self) -> None:
        # Arrange
        inner = RolloutCollector(
            env_factory=_harness_factory(_RowClient(), max_turns=2),
            batch_size=1,
            group_size=1,
            base_seed=0,
            adaptive_task_sampling=True,
        )
        collector = AsyncBatchCollector(inner)

        # Act
        try:
            asyncio.run(collector.assign_group_task(0))
            collector.record_group_outcome(2, informative=True, success="mixed")
            collector.record_group_outcome(6, informative=False, success=None)
            stats = collector.task_row_stats()
        finally:
            collector.close()

        # Assert
        weights = {row.row: row.weight for row in stats}
        assert collector.adaptive_task_sampling is True
        assert weights[2] == pytest.approx(2 / 3)
        assert weights[6] == pytest.approx(1 / 3)
        assert weights[0] == 0.5
        assert {row.row: row.mixed for row in stats if row.mixed} == {2: 1}


class TestAsyncBatchCollectorLoadTaskSamplerState:
    def test_a_fresh_collector_restores_another_collectors_outcomes(self) -> None:
        # Arrange
        def make_collector() -> AsyncBatchCollector:
            return AsyncBatchCollector(
                RolloutCollector(
                    env_factory=_harness_factory(_RowClient(), max_turns=2),
                    batch_size=1,
                    group_size=1,
                    base_seed=0,
                    adaptive_task_sampling=True,
                )
            )

        saved = make_collector()
        restored = make_collector()
        try:
            asyncio.run(saved.assign_group_task(0))
            saved.record_group_outcome(2, informative=True, success="mixed")
            saved.record_group_outcome(6, informative=False, success="tied_failure")

            # Act
            restored.load_task_sampler_state(saved.task_sampler_state())

            # Assert
            assert restored.task_row_stats() == saved.task_row_stats()
            assert restored.task_sampler_state() == [
                {
                    "row": 2,
                    "informative": 1.0,
                    "observed": 1.0,
                    "tied_failure": 0,
                    "mixed": 1,
                    "tied_success": 0,
                },
                {
                    "row": 6,
                    "informative": 0.0,
                    "observed": 1.0,
                    "tied_failure": 1,
                    "mixed": 0,
                    "tied_success": 0,
                },
            ]
        finally:
            saved.close()
            restored.close()


class TestAsyncBatchCollectorMultiTurn:
    def test_two_non_terminal_steps_then_terminal_accumulate_tokens(self) -> None:
        client = FakeEnvClient(terminate_after=3)
        inner = RolloutCollector(
            env_factory=_harness_factory(client, max_turns=5),
            batch_size=1,
            group_size=1,
        )
        collector = AsyncBatchCollector(inner)
        gen = torch.tensor([[10, 11]], dtype=torch.long)

        async def _run() -> tuple[torch.Tensor, torch.Tensor]:
            reset = await collector.reset("ep-mt")
            assert reset.done is False

            step1 = await collector.step(
                "ep-mt",
                token_ids=torch.cat([reset.observation["input_ids"], gen], dim=1),
            )
            assert step1.done is False

            step2 = await collector.step(
                "ep-mt",
                token_ids=torch.cat([step1.observation["input_ids"], gen], dim=1),
            )
            assert step2.done is False

            step3 = await collector.step(
                "ep-mt",
                token_ids=torch.cat([step2.observation["input_ids"], gen], dim=1),
            )
            assert step3.done is True

            ids, mask, *_rest = await collector.get_episode_data("ep-mt")
            return ids, mask

        try:
            ids, mask = asyncio.run(_run())
        finally:
            collector.close()

        gen_len = int(gen.shape[-1])
        assert mask.shape[-1] == ids.shape[-1] - 1
        assert int(mask.sum()) == 3 * gen_len
        true_spans = _true_spans(mask[0])
        assert true_spans == [gen_len, gen_len, gen_len]


class TestAsyncBatchCollectorGetEpisodeData:
    def test_returns_the_segments_of_a_restarted_episode(self) -> None:
        # Arrange
        inner = RolloutCollector(
            env_factory=_harness_factory(
                FakeEnvClient(terminate_after=3), max_turns=5, segment_prompt_tokens=40
            ),
            batch_size=1,
            group_size=1,
        )
        collector = AsyncBatchCollector(inner)
        # The restart history keeps only the action after </think>.
        gen = torch.tensor([[ord(c) for c in "r" * 20 + "</think>a"]], dtype=torch.long)

        async def _run() -> EpisodeTensors:
            response = await collector.reset("ep-seg")
            while not response.done:
                response = await collector.step(
                    "ep-seg",
                    token_ids=torch.cat(
                        [response.observation["input_ids"], gen], dim=1
                    ),
                )
            return await collector.get_episode_data("ep-seg")

        # Act
        try:
            ids, mask, *_rest, segments = asyncio.run(_run())
        finally:
            collector.close()

        # Assert
        assert segments is not None
        # prompt(6) + gen(29); then the 32- and 37-char restart prompts + gen.
        assert segments.token_lengths.tolist() == [35, 61, 66]
        assert int(segments.token_lengths.sum()) == ids.shape[-1]
        assert _true_spans(mask[0]) == [29, 29, 29]


def _true_spans(mask: torch.Tensor) -> list[int]:
    """Lengths of contiguous True runs in a 1-D bool mask."""
    spans: list[int] = []
    run = 0
    for flag in mask.tolist():
        if flag:
            run += 1
            continue
        if run:
            spans.append(run)
            run = 0
    if run:
        spans.append(run)
    return spans


class _WindowStubCollector(StubCollector):
    """Stub recording the window-control calls the facade forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.group_seeds: list[int] = []
        self.geometries: list[tuple[int, int]] = []
        self.active_ids_limits: list[int] = []

    def set_group_seed(self, group_seed: int) -> None:
        self.group_seeds.append(group_seed)

    def update_rollout_geometry(
        self, *, rollout_batch_size: int, group_size: int
    ) -> None:
        self.geometries.append((rollout_batch_size, group_size))

    def active_episode_count(self) -> int:
        return 4

    def active_episode_ids(self, max_ids: int = 16) -> list[str]:
        self.active_ids_limits.append(max_ids)
        return ["ep-1", "ep-2"]


class TestAsyncBatchCollectorWindowControls:
    """Window control and diagnostics pass straight through to the collector.

    These are synchronous on the facade (no executor offload) because they touch
    collector state rather than doing env I/O.
    """

    def test_set_group_seed_is_forwarded(self) -> None:
        inner = _WindowStubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            collector.set_group_seed(11)
        finally:
            collector.close()

        assert inner.group_seeds == [11]

    def test_update_rollout_geometry_forwards_both_dimensions(self) -> None:
        inner = _WindowStubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            collector.update_rollout_geometry(rollout_batch_size=3, group_size=2)
        finally:
            collector.close()

        assert inner.geometries == [(3, 2)]

    def test_active_episode_count_is_forwarded(self) -> None:
        inner = _WindowStubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            assert collector.active_episode_count() == 4
        finally:
            collector.close()

    def test_active_episode_ids_forwards_the_cap(self) -> None:
        inner = _WindowStubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            assert collector.active_episode_ids(3) == ["ep-1", "ep-2"]
        finally:
            collector.close()

        assert inner.active_ids_limits == [3]


class TestAsyncBatchCollectorEpisodeImageCalls:
    def test_forwards_to_the_inner_collector(self) -> None:
        class _VisionStub(StubCollector):
            def episode_image_calls(self, episode_id: str):
                return [episode_id]

        inner = _VisionStub()
        collector = AsyncBatchCollector(inner)

        try:
            assert collector.episode_image_calls("ep-7") == ["ep-7"]
        finally:
            collector.close()
