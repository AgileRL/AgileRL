# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import aiohttp
import pytest

from agilerl.llm_envs.call_record import CallOrigin, CallRecord, prompt_prefix_hashes
from agilerl.llm_envs.model_endpoint import (
    CallRecorder,
    CallSample,
    ModelCallEndpoint,
)

ORIGIN = CallOrigin(
    organization_id="org",
    deployment_id="train-run",
    instance_id="rank-0",
    user_id="trainer",
    base_model="tiny",
    chat_template_hash="t",
)
SAMPLE = CallSample(
    prompt_token_ids=[1, 2, 3],
    completion_token_ids=[4, 5],
    completion_logprobs=[-0.1, -0.2],
    model_version="v7",
    checkpoint_id="step-7",
    raw_completion="hello",
    sampling={"temperature": 1.0},
)
REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
REPLY = {"role": "assistant", "content": "hello"}


async def serve_one(
    endpoint: ModelCallEndpoint,
    request: dict[str, Any],
    headers: dict[str, str],
    sample: CallSample | None,
) -> tuple[int, dict[str, str], str]:
    """Post ``request``, answer it with :data:`REPLY` and ``sample``; return status, headers, body."""
    await endpoint.open()
    try:
        async with aiohttp.ClientSession(base_url=endpoint.base_url + "/") as session:
            post = asyncio.create_task(
                session.post("chat/completions", json=request, headers=headers)
            )
            held = asyncio.ensure_future(endpoint.next_call())
            await asyncio.wait({post, held}, return_when=asyncio.FIRST_COMPLETED)
            if held.done():
                endpoint.answer(held.result().call_id, REPLY, sample)
            else:
                held.cancel()
            response = await post
            return response.status, dict(response.headers), await response.text()
    finally:
        await endpoint.close()


class TestModelCallEndpointRecording:
    def test_records_each_answered_call_with_its_tokens(self) -> None:
        # Arrange
        records: list[CallRecord] = []
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, records.append))
        headers = {
            "X-AgileRL-Session-Id": "episode-3",
            "X-AgileRL-Metadata": json.dumps({"group": "g1"}),
        }

        # Act
        status, response_headers, text = asyncio.run(
            serve_one(endpoint, {**REQUEST, "user": "alice"}, headers, SAMPLE)
        )

        # Assert
        assert status == 200
        (record,) = records
        assert json.loads(text)["id"] == record.call_id
        assert response_headers["X-AgileRL-Call-Id"] == record.call_id
        assert record.client_session_id == "episode-3"
        assert record.end_user == "alice"
        assert record.metadata == {"group": "g1"}
        assert (record.deployment_id, record.model_version, record.checkpoint_id) == (
            "train-run",
            "v7",
            "step-7",
        )
        assert record.prompt_token_ids == [1, 2, 3]
        assert record.completion_token_ids == [4, 5]
        assert record.completion_logprobs == [-0.1, -0.2]
        assert (record.prompt_tokens, record.completion_tokens) == (3, 2)
        assert record.messages == REQUEST["messages"]
        assert record.response_message == REPLY
        assert record.finish_reason == "stop"
        assert record.prompt_prefix_hashes == prompt_prefix_hashes(REQUEST["messages"])
        assert (
            record.response_prefix_hash
            == prompt_prefix_hashes([*REQUEST["messages"], REPLY])[-1]
        )

    def test_streamed_call_carries_the_call_id(self) -> None:
        records: list[CallRecord] = []
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, records.append))

        _, headers, text = asyncio.run(
            serve_one(endpoint, {**REQUEST, "stream": True}, {}, SAMPLE)
        )

        first = json.loads(text.splitlines()[0][len("data: ") :])
        assert first["id"] == records[0].call_id == headers["X-AgileRL-Call-Id"]
        assert records[0].stream is True

    def test_capture_off_header_skips_the_record(self) -> None:
        records: list[CallRecord] = []
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, records.append))

        status, headers, text = asyncio.run(
            serve_one(endpoint, REQUEST, {"X-AgileRL-Capture": "off"}, SAMPLE)
        )

        assert (status, records, json.loads(text)["id"]) == (200, [], "agilerl")
        assert "X-AgileRL-Call-Id" not in headers

    def test_without_a_recorder_responses_are_unchanged(self) -> None:
        endpoint = ModelCallEndpoint()

        status, headers, text = asyncio.run(serve_one(endpoint, REQUEST, {}, None))

        assert (status, json.loads(text)["id"]) == (200, "agilerl")
        assert "X-AgileRL-Call-Id" not in headers

    def test_closing_before_opening_leaves_the_endpoint_usable(self) -> None:
        endpoint = ModelCallEndpoint()
        asyncio.run(endpoint.close())

        status, _, text = asyncio.run(serve_one(endpoint, REQUEST, {}, None))

        assert (status, json.loads(text)["choices"][0]["message"]) == (200, REPLY)

    def test_rejects_metadata_that_is_not_a_string_map(self) -> None:
        records: list[CallRecord] = []
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, records.append))

        status, _, text = asyncio.run(
            serve_one(
                endpoint, REQUEST, {"X-AgileRL-Metadata": json.dumps({"n": 1})}, SAMPLE
            )
        )

        assert status == 400
        assert "string keys and string values" in json.loads(text)["error"]["message"]
        assert records == []

    def test_rejects_metadata_that_is_not_json(self) -> None:
        records: list[CallRecord] = []
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, records.append))

        status, _, text = asyncio.run(
            serve_one(endpoint, REQUEST, {"X-AgileRL-Metadata": "{"}, SAMPLE)
        )

        assert status == 400
        assert "is not valid JSON" in json.loads(text)["error"]["message"]
        assert records == []

    def test_answering_a_recording_endpoint_needs_a_sample(self) -> None:
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, lambda _: None))

        with pytest.raises(ValueError, match="needs the CallSample"):
            endpoint.answer(0, REPLY)

    def test_missing_sample_cancels_the_held_request(self) -> None:
        # Arrange
        records: list[CallRecord] = []
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, records.append))

        async def run() -> tuple[int, str]:
            await endpoint.open()
            try:
                async with aiohttp.ClientSession(
                    base_url=endpoint.base_url + "/"
                ) as session:
                    post = asyncio.create_task(
                        session.post("chat/completions", json=REQUEST)
                    )
                    held = await endpoint.next_call()
                    with pytest.raises(ValueError, match="needs the CallSample"):
                        endpoint.answer(held.call_id, REPLY)
                    response = await post
                    return response.status, await response.text()
            finally:
                await endpoint.close()

        # Act
        status, text = asyncio.run(run())

        # Assert
        assert status == 503
        assert "cancelled" in json.loads(text)["error"]["message"]
        assert records == []

    def test_cancel_fails_the_held_request(self) -> None:
        endpoint = ModelCallEndpoint()

        async def run() -> tuple[int, str]:
            await endpoint.open()
            try:
                async with aiohttp.ClientSession(
                    base_url=endpoint.base_url + "/"
                ) as session:
                    post = asyncio.create_task(
                        session.post("chat/completions", json=REQUEST)
                    )
                    held = await endpoint.next_call()
                    endpoint.cancel(held.call_id)
                    response = await post
                    return response.status, await response.text()
            finally:
                await endpoint.close()

        status, text = asyncio.run(run())

        assert status == 503
        assert "cancelled" in json.loads(text)["error"]["message"]

    def test_reset_fails_unanswered_calls(self) -> None:
        endpoint = ModelCallEndpoint()

        async def run() -> int:
            await endpoint.open()
            try:
                async with aiohttp.ClientSession(
                    base_url=endpoint.base_url + "/"
                ) as session:
                    post = asyncio.create_task(
                        session.post("chat/completions", json=REQUEST)
                    )
                    await endpoint.next_call()
                    endpoint.reset()
                    response = await post
                    return response.status
            finally:
                await endpoint.close()

        assert asyncio.run(run()) == 503

    def test_streamed_tool_call_is_in_the_delta(self) -> None:
        endpoint = ModelCallEndpoint()
        reply = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t0",
                    "type": "function",
                    "function": {"name": "add", "arguments": "{}"},
                }
            ],
        }

        async def run() -> str:
            await endpoint.open()
            try:
                async with aiohttp.ClientSession(
                    base_url=endpoint.base_url + "/"
                ) as session:
                    post = asyncio.create_task(
                        session.post(
                            "chat/completions", json={**REQUEST, "stream": True}
                        )
                    )
                    held = await endpoint.next_call()
                    endpoint.answer(held.call_id, reply)
                    response = await post
                    return await response.text()
            finally:
                await endpoint.close()

        text = asyncio.run(run())
        first = json.loads(
            next(
                line.removeprefix("data: ")
                for line in text.splitlines()
                if line.startswith("data: {")
            )
        )
        assert first["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, **reply["tool_calls"][0]}
        ]
        finish = json.loads(
            [
                line.removeprefix("data: ")
                for line in text.splitlines()
                if line.startswith("data: {")
            ][1]
        )
        assert finish["choices"][0]["finish_reason"] == "tool_calls"

    def test_sample_finish_reason_is_sent_on_the_completion(self) -> None:
        # Arrange
        records: list[CallRecord] = []
        endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, records.append))
        sample = dataclasses.replace(SAMPLE, finish_reason="length")

        # Act
        status, _, text = asyncio.run(serve_one(endpoint, REQUEST, {}, sample))

        # Assert
        assert status == 200
        assert json.loads(text)["choices"][0]["finish_reason"] == "length"
        assert records[0].finish_reason == "length"

        stream_records: list[CallRecord] = []
        stream_endpoint = ModelCallEndpoint(CallRecorder(ORIGIN, stream_records.append))
        _, _, stream_text = asyncio.run(
            serve_one(stream_endpoint, {**REQUEST, "stream": True}, {}, sample)
        )
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in stream_text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert chunks[1]["choices"][0]["finish_reason"] == "length"
        assert stream_records[0].finish_reason == "length"
