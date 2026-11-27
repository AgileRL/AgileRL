# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Scoring completions with the function bound to the app's ``reward_function`` reference."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from agilerl.snowflake import rewards
from agilerl.snowflake.rewards import (
    REWARD_QUERY,
    FunctionReward,
    RewardRow,
    answer_text,
    open_reward_session,
)
from tests.test_snowflake_setup import Frame, Row

ROWS = [
    RewardRow(prompt="2+2?", completion="4", answer="4"),
    RewardRow(prompt="3+3?", completion="7", answer="6"),
]


class Blip(OSError):
    """A dropped connection."""


class FunctionSession:
    """Snowpark session whose bound function scores each row with ``score``.

    :param fail: Errors the next calls raise, one per call, before any succeeds.
    :param gate: When set, every call waits for it before answering.
    """

    def __init__(
        self,
        score: object = None,
        rewards: list[object] | None = None,
        fail: list[Exception] | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.score = score
        self.rewards = rewards
        self.fail = list(fail or [])
        self.gate = gate
        self.calls: list[tuple[str, Sequence[object]]] = []
        self.started = threading.Event()

    def sql(self, query: str, params: Sequence[object]) -> Frame:
        self.calls.append((query, params))
        self.started.set()
        if self.gate is not None:
            self.gate.wait()
        if self.fail:
            raise self.fail.pop(0)
        rows = json.loads(str(params[0]))
        values = (
            self.rewards
            if self.rewards is not None
            else [self.score(**row) for row in rows]
        )
        return Frame([Row(REWARD=value) for value in values])


def exact(prompt: str, completion: str, answer: str | None) -> float:
    return 1.0 if completion == answer else 0.0


def function_reward(*sessions: FunctionSession) -> tuple[FunctionReward, list[float]]:
    """A FunctionReward that opens ``sessions`` in turn, and the delays it slept."""
    opened = iter(sessions)
    slept: list[float] = []
    score = FunctionReward(lambda: next(opened), (Blip,), slept.append)
    return score, slept


class TestFunctionReward:
    def test_scores_a_batch_in_one_query_in_row_order(self) -> None:
        session = FunctionSession(exact)
        score, _ = function_reward(session)

        assert score(ROWS) == [1.0, 0.0]
        [(query, params)] = session.calls
        assert query == REWARD_QUERY
        assert json.loads(str(params[0])) == [
            {"prompt": "2+2?", "completion": "4", "answer": "4"},
            {"prompt": "3+3?", "completion": "7", "answer": "6"},
        ]

    def test_the_query_calls_the_reference_and_keeps_completions_out_of_the_sql(
        self,
    ) -> None:
        session = FunctionSession(exact)
        score, _ = function_reward(session)

        score([RewardRow("p", "'); DROP TABLE x; --", None)])

        [(query, _)] = session.calls
        assert "reference('REWARD_FUNCTION')" in query
        assert "PARSE_JSON(?)" in query
        assert "DROP TABLE" not in query

    def test_batches_waiting_on_a_query_share_the_next_one(self) -> None:
        # Arrange
        gate = threading.Event()
        session = FunctionSession(exact, gate=gate)
        score, _ = function_reward(session)
        groups = [
            [RewardRow(f"q{i}", str(i), str(i)), RewardRow(f"q{i}", "x", str(i))]
            for i in range(8)
        ]
        results: dict[int, list[float]] = {}

        def run(index: int) -> None:
            results[index] = score(groups[index])

        first = threading.Thread(target=run, args=(0,))
        first.start()
        session.started.wait()
        rest = [threading.Thread(target=run, args=(i,)) for i in range(1, 8)]

        # Act
        for thread in rest:
            thread.start()
        while len(score._pending) < 7:  # every later group is queued
            time.sleep(0.01)
        gate.set()
        for thread in [first, *rest]:
            thread.join()

        # Assert
        assert [len(json.loads(str(params[0]))) for _, params in session.calls] == [
            2,
            14,
        ]
        assert results == {i: [1.0, 0.0] for i in range(8)}

    def test_a_dropped_connection_reopens_the_session_and_retries(self) -> None:
        broken = FunctionSession(exact, fail=[Blip("reset by peer")])
        fresh = FunctionSession(exact, fail=[Blip("reset by peer")])
        healthy = FunctionSession(exact)
        score, slept = function_reward(broken, fresh, healthy)

        assert score(ROWS) == [1.0, 0.0]
        assert slept == [2.0, 4.0]
        assert len(healthy.calls) == 1

    def test_gives_up_after_the_last_attempt(self) -> None:
        sessions = [FunctionSession(exact, fail=[Blip("down")]) for _ in range(4)]
        score, slept = function_reward(*sessions)

        with pytest.raises(Blip, match="down"):
            score(ROWS)

        assert slept == [2.0, 4.0, 8.0]

    def test_a_failing_function_is_not_retried(self) -> None:
        def fail(**_row: object) -> float:
            msg = "ZeroDivisionError in score_answer"
            raise ValueError(msg)

        score, slept = function_reward(FunctionSession(fail))

        with pytest.raises(ValueError, match="ZeroDivisionError in score_answer"):
            score(ROWS)

        assert slept == []

    def test_an_error_reaches_the_next_batch_too_and_scoring_carries_on(self) -> None:
        session = FunctionSession(rewards=[1.0])
        score, _ = function_reward(session)

        with pytest.raises(RuntimeError, match="returned 1 rewards for 2 completions"):
            score(ROWS)
        assert score(ROWS[:1]) == [1.0]

    def test_an_integer_reward_is_a_float(self) -> None:
        score, _ = function_reward(FunctionSession(rewards=[1, 0]))

        assert score(ROWS) == [1.0, 0.0]

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, "1.0"])
    def test_refuses_a_reward_that_is_not_a_finite_number(self, bad: object) -> None:
        score, _ = function_reward(FunctionSession(rewards=[1.0, bad]))

        with pytest.raises(
            RuntimeError, match=r"for completion 1\. It must return a finite FLOAT"
        ):
            score(ROWS)


class TestAnswerText:
    @pytest.mark.parametrize(
        ("answer", "text"),
        [
            (None, None),
            ("42", "42"),
            (42, "42"),
            (
                {"name": "search", "arguments": {"q": "x"}},
                '{"name": "search", "arguments": {"q": "x"}}',
            ),
        ],
    )
    def test_text_stays_and_anything_else_is_json(
        self, answer: object, text: str | None
    ) -> None:
        assert answer_text(answer) == text


class TestOpenRewardSession:
    def test_outside_a_snowflake_container_it_says_only_the_app_runs_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(rewards, "token_path", lambda: tmp_path / "token")

        with pytest.raises(ValueError, match="runs only in the AgileRL Snowflake app"):
            open_reward_session()

    def test_in_a_container_it_opens_the_container_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        token = tmp_path / "token"
        token.write_text("t", encoding="utf-8")
        opened: list[str | None] = []
        monkeypatch.setattr(rewards, "token_path", lambda: token)
        monkeypatch.setattr(
            rewards, "open_session", lambda name: opened.append(name) or "session"
        )

        assert open_reward_session() == "session"
        assert opened == [None]
