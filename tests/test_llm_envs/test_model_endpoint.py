# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from agilerl.llm_envs.model_endpoint import ModelCallEndpoint

REQUEST = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
REPLY = {"role": "assistant", "content": "hello"}


async def serve_one(
    endpoint: ModelCallEndpoint, request: dict[str, Any]
) -> tuple[int, str]:
    """Post ``request`` and answer it with :data:`REPLY`; return status and body."""
    await endpoint.open()
    try:
        async with aiohttp.ClientSession(base_url=endpoint.base_url + "/") as session:
            post = asyncio.create_task(session.post("chat/completions", json=request))
            held = await endpoint.next_call()
            endpoint.answer(held.call_id, REPLY)
            response = await post
            return response.status, await response.text()
    finally:
        await endpoint.close()


class TestModelCallEndpoint:
    def test_answers_the_held_call(self) -> None:
        endpoint = ModelCallEndpoint()

        status, text = asyncio.run(serve_one(endpoint, REQUEST))

        assert (status, json.loads(text)["id"]) == (200, "agilerl")
        assert json.loads(text)["choices"][0]["message"] == REPLY

    def test_closing_before_opening_leaves_the_endpoint_usable(self) -> None:
        endpoint = ModelCallEndpoint()
        asyncio.run(endpoint.close())

        status, text = asyncio.run(serve_one(endpoint, REQUEST))

        assert (status, json.loads(text)["choices"][0]["message"]) == (200, REPLY)

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
