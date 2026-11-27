# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Rewards on Snowflake: the submit check, and the call to a bound reward function.

The Snowflake app calls a function the buyer owns through its
``reward_function`` reference. Snowflake runs that function in the buyer's
account; the app never reads its code.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from typing import Protocol

from agilerl.models.algorithms.opd import OPDSpec
from agilerl.models.env import LLMEnvSpec
from agilerl.models.manifest import TrainingManifest
from agilerl.snowflake.session import open_session, token_path
from agilerl.snowflake.source import SnowflakeFrame

REWARD_REFERENCE = "REWARD_FUNCTION"
# One bound JSON array per batch keeps long completions out of the SQL text,
# which Snowflake caps at 1 MB.
REWARD_QUERY = (
    f"SELECT reference('{REWARD_REFERENCE}')("
    "r.value:prompt::STRING, r.value:completion::STRING, r.value:answer::STRING"
    ")::FLOAT AS REWARD FROM TABLE(FLATTEN(INPUT => PARSE_JSON(?))) r ORDER BY r.index"
)


def check_reward_file(
    manifest: TrainingManifest, environment: LLMEnvSpec, has_reward_file: bool
) -> None:
    """Reject a reward file the manifest does not use, or a missing reward it needs.

    :param manifest: Parsed training manifest.
    :param environment: The manifest's LLM environment.
    :param has_reward_file: Whether a reward file was passed.
    :raises ValueError: A dataset-backed rollout has no reward file or
        ``environment.reward`` and is not OPD with no rubric, or a non-rollout
        manifest has one.
    """
    rollout = environment.env_type == "rollout"
    opd_without_rubric = (
        isinstance(manifest.algorithm, OPDSpec) and environment.rubric_file_path is None
    )
    if (
        rollout
        and environment.dataset_backed_rollout
        and not has_reward_file
        and environment.reward is None
        and not opd_without_rubric
    ):
        msg = (
            "A dataset-backed rollout needs environment.reward, "
            "unless it runs OPD and names no rubric."
        )
        raise ValueError(msg)
    if not rollout and has_reward_file:
        msg = "A reward file is only used by RL manifests."
        raise ValueError(msg)


@dataclass(frozen=True)
class RewardRow:
    """One completion the bound function scores.

    :param prompt: The row's question, or the episode's first prompt.
    :param completion: The final answer, the text after any reasoning.
    :param answer: The row's answer, as text, or ``None``.
    """

    prompt: str
    completion: str
    answer: str | None


class BindingSession(Protocol):
    """Snowpark session that runs a statement with bound parameters."""

    def sql(self, query: str, params: Sequence[object]) -> SnowflakeFrame:
        """Prepare ``query`` with ``?`` placeholders bound to ``params``."""
        ...


def answer_text(answer: object) -> str | None:
    """A row's answer as the function's ``answer`` argument: text as is, else JSON."""
    if answer is None or isinstance(answer, str):
        return answer
    return json.dumps(answer)


def open_reward_session() -> BindingSession:
    """Open the job container's Snowpark session.

    :return: The session.
    :raises ValueError: The process is not in a Snowflake container.
    """
    if not token_path().is_file():
        msg = (
            "reward type snowflake_function runs only in the AgileRL Snowflake "
            "app, which calls the function bound to its reward_function reference."
        )
        raise ValueError(msg)
    return open_session(None)


@dataclass
class PendingBatch:
    """Rows one caller waits on, and where its rewards or error land."""

    rows: Sequence[RewardRow]
    result: Future[list[float]] = field(default_factory=Future)


class FunctionReward:
    """Scores batches of completions with the function bound to ``reward_function``.

    One query is in flight per process. Batches that arrive while it runs are
    merged into the next query, because a Python function costs about 1.3 s a
    query whatever the row count and queues on the app's XSMALL warehouse.
    A transient connection error reopens the session and retries with backoff.

    :param open_reward_session: Opens the Snowpark session.
    :param retry_on: Errors worth another attempt. Any other error raises at once.
    :param sleep: Called between attempts.
    :param attempts: Tries per query before the error reaches the callers.
    :param retry_s: First backoff delay. Each retry doubles it.
    """

    def __init__(
        self,
        open_reward_session: Callable[[], BindingSession],
        retry_on: tuple[type[Exception], ...],
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 4,
        retry_s: float = 2.0,
    ) -> None:
        self._open = open_reward_session
        self._retry_on = retry_on
        self._sleep = sleep
        self._attempts = attempts
        self._retry_s = retry_s
        self._session: BindingSession | None = None
        self._lock = threading.Lock()
        self._pending: list[PendingBatch] = []
        self._draining = False

    def __call__(self, rows: Sequence[RewardRow]) -> list[float]:
        """Score each row with the bound function, sharing a query with other callers.

        :param rows: The completions to score.
        :return: One finite reward per row, in order.
        :raises RuntimeError: The function returned another number of rows, or a
            value that is not a finite number.
        """
        batch = PendingBatch(rows)
        with self._lock:
            self._pending.append(batch)
            drains = not self._draining
            self._draining = True
        if drains:
            self._drain()
        return batch.result.result()

    def _drain(self) -> None:
        """Score pending batches, one merged query at a time, until none are left."""
        while True:
            with self._lock:
                batches, self._pending = self._pending, []
                if not batches:
                    self._draining = False
                    return
            rows = [row for batch in batches for row in batch.rows]
            try:
                rewards = self._score(rows)
            except Exception as exc:  # every waiting caller raises it
                for batch in batches:
                    batch.result.set_exception(exc)
                continue
            start = 0
            for batch in batches:
                batch.result.set_result(rewards[start : start + len(batch.rows)])
                start += len(batch.rows)

    def _score(self, rows: Sequence[RewardRow]) -> list[float]:
        """One query over ``rows``, retried on transient errors, with its shape checked."""
        params = [json.dumps([asdict(row) for row in rows])]
        attempt = 1
        while True:
            if self._session is None:
                self._session = self._open()
            try:
                result = self._session.sql(REWARD_QUERY, params).collect()
                break
            except self._retry_on:
                if attempt == self._attempts:
                    raise
                self._session = None
                self._sleep(self._retry_s * 2 ** (attempt - 1))
                attempt += 1
        if len(result) != len(rows):
            msg = (
                f"The function bound to reward_function returned {len(result)} "
                f"rewards for {len(rows)} completions."
            )
            raise RuntimeError(msg)
        rewards = [row.as_dict()["REWARD"] for row in result]
        for index, reward in enumerate(rewards):
            if (
                isinstance(reward, bool)
                or not isinstance(reward, (int, float))
                or not math.isfinite(reward)
            ):
                msg = (
                    f"The function bound to reward_function returned {reward!r} "
                    f"for completion {index}. It must return a finite FLOAT for "
                    "every completion."
                )
                raise RuntimeError(msg)
        return [float(reward) for reward in rewards]
