# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The shared LLM judge client and the judge rubric."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from openenv.core.llm_client import LLMClient

from agilerl.arena.models.env import JudgeSpec
from agilerl.llm_envs import judge as judge_module
from agilerl.llm_envs.judge import (
    JudgeClient,
    JudgeStats,
    build_judge_client,
    drain_judge_counts,
    drain_judge_stats,
    judge_metrics,
    shared_judge_client,
)
from agilerl.llm_envs.rubrics import (
    LLMJudgeRubric,
    TurnAction,
    TurnObservation,
    render_transcript,
    verdict_score,
)

LOCAL = JudgeSpec(model="judge-model", base_url="http://judge:8000")


class ScriptedLLM(LLMClient):
    """Provider double: replies with ``reply`` after ``delay_s``, tracking calls in flight."""

    def __init__(self, reply: str = '{"score": 0.8}', delay_s: float = 0.0) -> None:
        super().__init__("http://fake", 1)
        self.reply = reply
        self.delay_s = delay_s
        self.prompts: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(self.delay_s)
        self.in_flight -= 1
        return self.reply


@pytest.fixture(autouse=True)
def no_shared_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge_module, "JUDGE_CLIENTS", {})


def last_turn(transcript: list[dict[str, str]], done: bool = True) -> TurnObservation:
    return TurnObservation(
        env_reward=0.0,
        env_ran=True,
        done=done,
        turn_index=0,
        prompt="Say hi.",
        transcript=transcript,
    )


TRANSCRIPT = [
    {"role": "user", "content": "Say hi."},
    {"role": "assistant", "content": "hi", "completion": "<think>greet</think>hi"},
    {"role": "user", "content": "Thanks."},
]


def fake_anthropic_sdk(create: Any) -> SimpleNamespace:
    """An ``anthropic`` module whose ``AsyncAnthropic().messages.create`` is ``create``."""

    class FakeAsyncAnthropic:
        def __init__(self, **_kwargs: Any) -> None:
            self.messages = SimpleNamespace(create=create)

        def with_options(self, **_kwargs: Any) -> FakeAsyncAnthropic:
            return self

    return SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic)


class TestBuildJudgeClient:
    def test_points_at_the_base_url(self) -> None:
        client = build_judge_client(LOCAL)

        assert client.base_url == "http://judge:8000"

    def test_reads_the_key_from_the_named_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JUDGE_KEY", "sk-test")
        spec = JudgeSpec(model="gpt-4.1-mini", api_key_env="JUDGE_KEY")

        client = build_judge_client(spec)

        assert client.base_url == "https://api.openai.com:443"

    def test_an_unset_key_variable_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("JUDGE_KEY", raising=False)
        spec = JudgeSpec(model="gpt-4.1-mini", api_key_env="JUDGE_KEY")

        with pytest.raises(ValueError, match="JUDGE_KEY is not set"):
            build_judge_client(spec)

    def test_hosted_anthropic_points_at_its_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_sdk(create=None))
        monkeypatch.setenv("JUDGE_KEY", "sk-ant-test")
        spec = JudgeSpec(
            provider="anthropic", model="claude-haiku-5-5", api_key_env="JUDGE_KEY"
        )

        client = build_judge_client(spec)

        assert client.base_url == "https://api.anthropic.com:443"


class TestAnthropicMessagesClientComplete:
    def test_sends_no_temperature_and_joins_the_text_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        sent: list[dict[str, Any]] = []

        async def create(**kwargs: Any) -> SimpleNamespace:
            sent.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", text="hmm"),
                    SimpleNamespace(type="text", text='{"score": 1}'),
                ]
            )

        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic_sdk(create))
        monkeypatch.setenv("JUDGE_KEY", "sk-ant-test")
        spec = JudgeSpec(
            provider="anthropic",
            model="claude-haiku-5-5",
            api_key_env="JUDGE_KEY",
            max_tokens=64,
        )
        client = build_judge_client(spec)

        # Act
        reply = asyncio.run(client.complete("grade this"))

        # Assert
        assert reply == '{"score": 1}'
        assert sent == [
            {
                "model": "claude-haiku-5-5",
                "messages": [{"role": "user", "content": "grade this"}],
                "max_tokens": 64,
            }
        ]


class TestJudgeClientComplete:
    def test_returns_the_reply_and_counts_the_call(self) -> None:
        client = JudgeClient(LOCAL, ScriptedLLM(reply="verdict"))

        reply = asyncio.run(client.complete("grade this"))

        assert reply == "verdict"
        assert client.drain_stats().calls == 1

    def test_caps_calls_in_flight(self) -> None:
        spec = LOCAL.model_copy(update={"max_concurrency": 2})
        inner = ScriptedLLM(delay_s=0.01)
        client = JudgeClient(spec, inner)

        async def many() -> None:
            await asyncio.gather(*(client.complete(str(i)) for i in range(6)))

        asyncio.run(many())

        assert inner.max_in_flight == 2

    @pytest.mark.parametrize(
        ("path", "request_path"),
        [
            ("", b"/v1/chat/completions"),
            ("/api/v1", b"/api/v1/chat/completions"),
        ],
    )
    def test_a_url_path_is_the_api_root(self, path: str, request_path: bytes) -> None:
        # Arrange
        seen: list[bytes] = []

        async def scenario() -> str:
            async def handle(
                reader: asyncio.StreamReader, writer: asyncio.StreamWriter
            ) -> None:
                head = await reader.readuntil(b"\r\n\r\n")
                seen.append(head.split(b" ")[1])
                body = (
                    b'{"choices": [{"index": 0, "message": '
                    b'{"role": "assistant", "content": "verdict"}}]}'
                )
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Connection: close\r\nContent-Length: %d\r\n\r\n%b"
                    % (len(body), body)
                )
                await writer.drain()
                writer.close()

            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            client = build_judge_client(
                JudgeSpec(model="m", base_url=f"http://127.0.0.1:{port}{path}")
            )
            try:
                return await client.complete("grade this")
            finally:
                server.close()

        # Act
        reply = asyncio.run(scenario())

        # Assert
        assert reply == "verdict"
        assert seen == [request_path]

    def test_a_hung_attempt_is_retried_with_the_real_sdk(self) -> None:
        # Arrange
        attempts = 0

        async def scenario() -> str:
            release = asyncio.Event()

            async def handle(
                reader: asyncio.StreamReader, writer: asyncio.StreamWriter
            ) -> None:
                nonlocal attempts
                attempts += 1
                attempt = attempts
                await reader.readuntil(b"\r\n\r\n")
                if attempt == 1:
                    await release.wait()
                else:
                    body = (
                        b'{"choices": [{"index": 0, "message": '
                        b'{"role": "assistant", "content": "verdict"}}]}'
                    )
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        b"Connection: close\r\nContent-Length: %d\r\n\r\n%b"
                        % (len(body), body)
                    )
                    await writer.drain()
                writer.close()

            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            client = build_judge_client(
                JudgeSpec(model="m", base_url=f"http://127.0.0.1:{port}", timeout_s=0.2)
            )
            try:
                return await client.complete("grade this")
            finally:
                release.set()
                server.close()

        # Act
        reply = asyncio.run(scenario())

        # Assert
        assert reply == "verdict"
        assert attempts == 2


class TestSharedJudgeClient:
    def test_one_client_per_spec(self) -> None:
        assert shared_judge_client(LOCAL) is shared_judge_client(LOCAL)


class TestDrainJudgeStats:
    def test_empty_without_a_judge(self) -> None:
        assert drain_judge_stats() == {}

    def test_reports_calls_and_parse_failures_then_resets(self) -> None:
        # Arrange
        client = JudgeClient(LOCAL, ScriptedLLM(reply="no verdict here"))
        judge_module.JUDGE_CLIENTS[LOCAL] = client
        rubric = LLMJudgeRubric(client, "Be polite.")
        action = TurnAction(message="hi", completion="hi")

        # Act
        asyncio.run(rubric(action, last_turn(TRANSCRIPT)))
        first = drain_judge_stats()
        second = drain_judge_stats()

        # Assert
        assert first["judge_calls"] == 1.0
        assert first["judge_parse_failure_rate"] == 1.0
        assert second == {
            "judge_calls": 0.0,
            "judge_parse_failure_rate": 0.0,
            "judge_mean_latency_s": 0.0,
        }


class TestDrainJudgeCounts:
    def test_none_without_a_judge(self) -> None:
        assert drain_judge_counts() is None

    def test_sums_every_client_then_resets(self) -> None:
        # Arrange
        other = LOCAL.model_copy(update={"model": "other-model"})
        for spec in (LOCAL, other):
            client = JudgeClient(spec, ScriptedLLM())
            judge_module.JUDGE_CLIENTS[spec] = client
            asyncio.run(client.complete("grade this"))
        judge_module.JUDGE_CLIENTS[other].record_parse_failure()

        # Act
        first = drain_judge_counts()
        second = drain_judge_counts()

        # Assert
        assert first is not None
        assert (first.calls, first.parse_failures) == (2, 1)
        assert second == JudgeStats()


class TestJudgeMetrics:
    def test_rates_divide_by_the_calls(self) -> None:
        stats = JudgeStats(calls=4, parse_failures=1, latency_s=6.0)

        assert judge_metrics(stats) == {
            "judge_calls": 4.0,
            "judge_parse_failure_rate": 0.25,
            "judge_mean_latency_s": 1.5,
        }

    def test_no_calls_reports_zero_rates(self) -> None:
        assert judge_metrics(JudgeStats()) == {
            "judge_calls": 0.0,
            "judge_parse_failure_rate": 0.0,
            "judge_mean_latency_s": 0.0,
        }


class TestLLMJudgeRubric:
    def test_scores_the_last_turn_from_the_verdict(self) -> None:
        inner = ScriptedLLM(reply='Sure. {"score": 0.8, "reason": "polite"}')
        rubric = LLMJudgeRubric(JudgeClient(LOCAL, inner), "Be polite.")

        score = asyncio.run(
            rubric(TurnAction(message="hi", completion="hi"), last_turn(TRANSCRIPT))
        )

        assert score == pytest.approx(0.8)
        assert "Be polite." in inner.prompts[0]
        assert "ASSISTANT: hi" in inner.prompts[0]
        assert "greet" not in inner.prompts[0]

    def test_include_reasoning_shows_the_whole_turn(self) -> None:
        inner = ScriptedLLM()
        rubric = LLMJudgeRubric(
            JudgeClient(LOCAL, inner), "Be polite.", include_reasoning=True
        )

        asyncio.run(
            rubric(TurnAction(message="hi", completion="hi"), last_turn(TRANSCRIPT))
        )

        assert "ASSISTANT: <think>greet</think>hi" in inner.prompts[0]

    def test_earlier_turns_score_zero_without_a_call(self) -> None:
        inner = ScriptedLLM()
        rubric = LLMJudgeRubric(JudgeClient(LOCAL, inner), "Be polite.")

        score = asyncio.run(
            rubric(
                TurnAction(message="hi", completion="hi"),
                last_turn(TRANSCRIPT, done=False),
            )
        )

        assert score == 0.0
        assert inner.prompts == []

    def test_every_turn_judges_unfinished_turns(self) -> None:
        inner = ScriptedLLM()
        rubric = LLMJudgeRubric(
            JudgeClient(LOCAL, inner), "Be polite.", every_turn=True
        )

        score = asyncio.run(
            rubric(
                TurnAction(message="hi", completion="hi"),
                last_turn(TRANSCRIPT, done=False),
            )
        )

        assert score == pytest.approx(0.8)

    def test_clamps_the_score_to_the_unit_range(self) -> None:
        rubric = LLMJudgeRubric(
            JudgeClient(LOCAL, ScriptedLLM(reply='{"score": 7}')), "Be polite."
        )

        score = asyncio.run(
            rubric(TurnAction(message="hi", completion="hi"), last_turn(TRANSCRIPT))
        )

        assert score == 1.0

    @pytest.mark.parametrize("reply", ['{"score": NaN}', '{"score": Infinity}'])
    def test_a_non_finite_verdict_scores_zero(self, reply: str) -> None:
        rubric = LLMJudgeRubric(
            JudgeClient(LOCAL, ScriptedLLM(reply=reply)), "Be polite."
        )

        score = asyncio.run(
            rubric(TurnAction(message="hi", completion="hi"), last_turn(TRANSCRIPT))
        )

        assert score == 0.0


class TestVerdictScore:
    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ('{"score": 0.25}', 0.25),
            ('Verdict:\n{"score": 1, "reason": "ok"}\nDone.', 1.0),
            ('{"score": 0.8, "reason": "ok"}\nSee {note}.', 0.8),
            ('{"score": 0.8} extra {"score": 0.1}', 0.8),
            ("no json", None),
            ('{"score": "high"}', None),
            ('{"score": true}', None),
            ("{broken", None),
            ('{"score": NaN}', None),
            ('{"score": Infinity}', None),
            ('{"score": -Infinity}', None),
        ],
    )
    def test_reads_a_numeric_score(self, reply: str, expected: float | None) -> None:
        assert verdict_score(reply) == expected


class TestRenderTranscript:
    def test_renders_role_blocks_without_reasoning(self) -> None:
        assert render_transcript(TRANSCRIPT, False) == (
            "USER: Say hi.\n\nASSISTANT: hi\n\nUSER: Thanks."
        )
