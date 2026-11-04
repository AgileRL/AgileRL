# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The LLM client a rubric's judge calls: one per process and per :class:`JudgeSpec`.

Built on OpenEnv's ``LLMClient``. The provider SDK under it retries rate limits,
server errors and timed-out attempts with backoff; this client caps calls in
flight and bounds each attempt's time. A call still failing after that raises.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from openenv.core.llm_client import (
    AnthropicClient,
    LLMClient,
    OpenAIClient,
    create_llm_client,
)

from agilerl.arena.models.env import JudgeSpec

# Every rubric in the process shares one client per spec, so the concurrency cap
# and the connection pool are per process, not per rollout slot.
JUDGE_CLIENTS: dict[JudgeSpec, JudgeClient] = {}
JUDGE_CLIENTS_LOCK = threading.Lock()


class AnthropicMessagesClient(AnthropicClient):
    """OpenEnv's Anthropic client without ``temperature``, which Messages ``create`` rejects in anthropic>=1."""

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send one prompt and return the text blocks of the reply."""
        response = await self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return "".join(block.text for block in response.content if block.type == "text")


@dataclass
class JudgeStats:
    """Judge counters since the last :func:`drain_judge_stats`."""

    calls: int = 0
    parse_failures: int = 0
    latency_s: float = 0.0

    def add(self, other: JudgeStats) -> None:
        """Add another set of counters to these."""
        self.calls += other.calls
        self.parse_failures += other.parse_failures
        self.latency_s += other.latency_s


class JudgeClient(LLMClient):
    """An OpenEnv ``LLMClient`` that caps calls in flight and counts them.

    :param spec: The judge's provider, model, endpoint and limits.
    :param inner: The provider client the calls go to.
    """

    def __init__(self, spec: JudgeSpec, inner: LLMClient) -> None:
        super().__init__(inner.endpoint, inner.port)
        self._inner = inner
        self._semaphore = asyncio.Semaphore(spec.max_concurrency)
        self._stats_lock = threading.Lock()
        self._stats = JudgeStats()

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send one prompt and return the judge's text reply."""
        async with self._semaphore:
            start = time.monotonic()
            text = await self._inner.complete(prompt, **kwargs)
        with self._stats_lock:
            self._stats.calls += 1
            self._stats.latency_s += time.monotonic() - start
        return text

    def record_parse_failure(self) -> None:
        """Count a verdict the rubric could not read."""
        with self._stats_lock:
            self._stats.parse_failures += 1

    def drain_stats(self) -> JudgeStats:
        """Return the counters since the last drain and reset them."""
        with self._stats_lock:
            stats, self._stats = self._stats, JudgeStats()
        return stats


def build_judge_client(spec: JudgeSpec) -> JudgeClient:
    """Build a judge client from ``spec``, reading the API key from ``spec.api_key_env``.

    :raises ValueError: If the key's variable is unset.
    """
    api_key = None
    if spec.api_key_env is not None:
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            msg = f"The judge's API key variable {spec.api_key_env} is not set."
            raise ValueError(msg)
    inner: LLMClient
    if spec.provider == "openai" and spec.base_url is None and api_key is not None:
        # The factory picks the token parameter each hosted OpenAI model takes.
        inner = create_llm_client(
            spec.provider,
            spec.model,
            api_key,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    else:
        url = urlsplit(spec.base_url or "https://api.anthropic.com")
        client_cls = (
            OpenAIClient if spec.provider == "openai" else AnthropicMessagesClient
        )
        inner = client_cls(
            f"{url.scheme}://{url.hostname}",
            url.port or (443 if url.scheme == "https" else 80),
            spec.model,
            api_key=api_key,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
        )
    # A per-attempt timeout leaves the SDK free to retry a hung request.
    options: dict[str, Any] = {"timeout": spec.timeout_s}
    # OpenEnv builds host:port/v1; a URL with a path is the provider's whole API root.
    if spec.base_url is not None and urlsplit(spec.base_url).path not in ("", "/"):
        options["base_url"] = spec.base_url
    if isinstance(inner, OpenAIClient):
        inner._client = inner._client.with_options(**options)
    elif isinstance(inner, AnthropicClient):
        inner._client = inner._client.with_options(**options)
    return JudgeClient(spec, inner)


def shared_judge_client(spec: JudgeSpec) -> JudgeClient:
    """This process's judge client for ``spec``, built on first use."""
    with JUDGE_CLIENTS_LOCK:
        if spec not in JUDGE_CLIENTS:
            JUDGE_CLIENTS[spec] = build_judge_client(spec)
        return JUDGE_CLIENTS[spec]


def drain_judge_counts() -> JudgeStats | None:
    """Judge counters across this process's clients since the last drain.

    :return: The summed counters; ``None`` when the process has no judge.
    """
    with JUDGE_CLIENTS_LOCK:
        clients = list(JUDGE_CLIENTS.values())
    if not clients:
        return None
    total = JudgeStats()
    for client in clients:
        total.add(client.drain_stats())
    return total


def judge_metrics(stats: JudgeStats) -> dict[str, float]:
    """``judge_calls``, ``judge_parse_failure_rate`` and ``judge_mean_latency_s`` from counters."""
    calls = max(stats.calls, 1)
    return {
        "judge_calls": float(stats.calls),
        "judge_parse_failure_rate": stats.parse_failures / calls,
        "judge_mean_latency_s": stats.latency_s / calls,
    }


def drain_judge_stats() -> dict[str, float]:
    """Judge metrics across this process's clients since the last drain.

    The keys depend only on whether a judge exists, never on whether it was
    called, so every data-parallel rank reduces the same metrics.

    :return: :func:`judge_metrics` of the drained counters; empty when the
        process has no judge.
    """
    stats = drain_judge_counts()
    return {} if stats is None else judge_metrics(stats)
