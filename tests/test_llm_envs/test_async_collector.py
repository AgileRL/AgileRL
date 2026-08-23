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
]


def _episode_tensors() -> EpisodeTensors:
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    mask = torch.ones_like(ids)
    turns = torch.zeros_like(ids)
    rewards = torch.tensor([1.0])
    return ids, mask, turns, rewards, None


class StubCollector:
    """In-process collector whose episode methods return known payloads."""

    num_envs = 1

    def __init__(self) -> None:
        self.io_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="env-io"
        )
        self.reset_calls: list[tuple[str, int | None, int | None]] = []
        self.step_calls: list[tuple[str, torch.Tensor]] = []
        self.finalize_calls: list[tuple[str, bool]] = []
        self.closed = False

    def reset_episode(
        self,
        episode_id: str,
        logical_slot: int | None = None,
        *,
        seed: int | None = None,
    ) -> tuple[dict[str, str], dict[str, int | None]]:
        self.reset_calls.append((episode_id, logical_slot, seed))
        return (
            {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)},
            {"logical_slot": logical_slot},
        )

    def step_episode(
        self,
        episode_id: str,
        token_ids: torch.Tensor,
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
            done=True,
            info={"seed": 7},
        )

        assert response.episode_id == "ep-1"
        assert response.observation == {"input_ids": [1, 2]}
        assert response.reward == 0.5
        assert response.done is True
        assert response.info == {"seed": 7}


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

    def test_reset_forwards_seed_to_reset_episode(self) -> None:
        inner = StubCollector()
        collector = AsyncBatchCollector(inner)

        try:
            response = asyncio.run(
                collector.reset("ep-seed", logical_slot_idx=1, seed=42),
            )
        finally:
            collector.close()

        assert response.episode_id == "ep-seed"
        assert inner.reset_calls == [("ep-seed", 1, 42)]


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
        assert response.done is True
        assert response.info == {"source": "step"}
        assert len(inner.step_calls) == 1
        called_id, called_ids = inner.step_calls[0]
        assert called_id == "ep-2"
        assert torch.equal(called_ids, token_ids)

    def test_step_rejects_completion_ids_keyword(self) -> None:
        collector = AsyncBatchCollector(StubCollector())
        token_ids = torch.tensor([[1]], dtype=torch.long)

        try:
            with pytest.raises(TypeError, match="completion_ids"):
                asyncio.run(collector.step("ep-2", completion_ids=token_ids))
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
                seed: int | None = None,
            ) -> tuple[dict[str, str], dict[str, Any]]:
                started.set()
                release.wait(timeout=30)
                return {"text": f"hung-{episode_id}-{logical_slot}-{seed}"}, {}

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
    def test_reset_marks_done_on_turn_zero_overflow(self) -> None:
        client = _OverflowClient()
        inner = RolloutCollector(
            env_factory=_harness_factory(
                client, max_turns=2, max_model_len=20, max_output_tokens=4
            ),
            batch_size=1,
            group_size=1,
        )
        collector = AsyncBatchCollector(inner)

        try:
            response = asyncio.run(collector.reset("ep-ov"))
        finally:
            collector.close()

        assert response.done is True
        assert response.observation == {}
        assert client.step_calls == 0


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
