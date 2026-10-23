# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Hold an env-side loop's chat completions open until something answers them."""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, NamedTuple

from aiohttp import web

__all__ = [
    "AnsweredCall",
    "ModelCallEndpoint",
    "PendingCall",
]


class PendingCall(NamedTuple):
    """A chat completion the endpoint holds open until answered."""

    call_id: int
    body: dict[str, Any]


class AnsweredCall(NamedTuple):
    """A held completion and the reply supplied for it."""

    call_id: int
    request: dict[str, Any]
    reply: dict[str, Any]


class ModelCallEndpoint:
    """A loopback OpenAI-compatible endpoint whose completions something else answers.

    Serves ``POST /v1/chat/completions`` (plain and streamed) on an ephemeral
    loopback port. :meth:`next_call` waits for the next request;
    :meth:`answer` replies to it. Calls take sequential ids, and every
    answered call is appended to :attr:`calls` in answer order.
    """

    def __init__(self) -> None:
        """Create the endpoint; :meth:`open` starts serving."""
        self.calls: list[AnsweredCall] = []
        self._pending: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
        self._awaiting: dict[
            int, tuple[dict[str, Any], asyncio.Future[dict[str, Any]]]
        ] = {}
        self._next_call_id = 0
        self._runner: web.AppRunner | None = None

    async def open(self) -> str:
        """Serve completions; return the ``/v1`` base URL."""
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._complete)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        base_url = f"http://127.0.0.1:{sock.getsockname()[1]}/v1"
        await web.SockSite(self._runner, sock).start()
        return base_url

    async def close(self) -> None:
        """Fail unanswered calls and stop serving."""
        self.reset()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def next_call(self) -> PendingCall:
        """Wait for the next chat completion and hold it open."""
        call_id, body = await self._pending.get()
        return PendingCall(call_id, body)

    def answer(self, call_id: int, message: dict[str, Any]) -> None:
        """Reply ``message`` to the held call and record it.

        :param call_id: Id from :meth:`next_call`.
        :param message: The assistant message: ``content`` and/or ``tool_calls``.
        """
        body, reply = self._awaiting.pop(call_id)
        reply.set_result(message)
        self.calls.append(AnsweredCall(call_id, body, message))

    def cancel(self, call_id: int) -> None:
        """Fail the held call ``call_id`` without recording it."""
        _, reply = self._awaiting.pop(call_id)
        reply.cancel()

    def reset(self) -> None:
        """Fail every unanswered call and drop the queue."""
        while self._awaiting:
            _, (_, reply) = self._awaiting.popitem()
            reply.cancel()
        self._pending = asyncio.Queue()

    async def _complete(self, request: web.Request) -> web.StreamResponse:
        """Hold one chat completion open until answered."""
        body = await request.json()
        call_id = self._next_call_id
        self._next_call_id += 1
        reply: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._awaiting[call_id] = (body, reply)
        await self._pending.put((call_id, body))
        try:
            message = await reply
        finally:
            self._awaiting.pop(call_id, None)
        if not body.get("stream"):
            return web.json_response(_completion(body["model"], message))
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for chunk in _completion_chunks(body["model"], message):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


def _completion(model: str, message: dict[str, Any]) -> dict[str, Any]:
    """A chat completion answering with ``message``."""
    return {
        "id": "agilerl",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": _finish(message)}
        ],
    }


def _completion_chunks(model: str, message: dict[str, Any]) -> list[dict[str, Any]]:
    """Streamed chat completion chunks answering with ``message``, usage last."""

    def chunk(choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {
            "id": "agilerl",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": choices,
            **extra,
        }

    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        delta["tool_calls"] = [
            {"index": index, **call} for index, call in enumerate(tool_calls)
        ]
    return [
        chunk([{"index": 0, "delta": delta, "finish_reason": None}]),
        chunk([{"index": 0, "delta": {}, "finish_reason": _finish(message)}]),
        chunk(
            [],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    ]


def _finish(message: dict[str, Any]) -> str:
    """The finish reason for an assistant ``message``."""
    return "tool_calls" if message.get("tool_calls") else "stop"
