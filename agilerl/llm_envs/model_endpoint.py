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
    "ModelCallEndpoint",
    "PendingCall",
]


class PendingCall(NamedTuple):
    """A chat completion the endpoint holds open until answered."""

    call_id: int
    body: dict[str, Any]


class ModelCallCancelled(Exception):
    """A held completion was failed by :meth:`ModelCallEndpoint.cancel` or ``reset``."""


class ModelCallEndpoint:
    """A loopback OpenAI-compatible endpoint whose completions something else answers.

    Serves ``POST /v1/chat/completions`` (plain and streamed) on an ephemeral
    loopback port at :attr:`base_url`. :meth:`next_call` waits for the next
    request; :meth:`answer` replies to it. Calls take sequential ids.
    """

    def __init__(self) -> None:
        """Reserve the port; :meth:`open` starts serving."""
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self.base_url = f"http://127.0.0.1:{self._socket.getsockname()[1]}/v1"
        self._pending: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
        self._awaiting: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._next_call_id = 0
        self._runner: web.AppRunner | None = None

    async def open(self) -> None:
        """Serve completions at :attr:`base_url`."""
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._complete)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.SockSite(self._runner, self._socket).start()

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
        """Reply ``message`` to the held call.

        :param call_id: Id from :meth:`next_call`.
        :param message: The assistant message: ``content`` and/or ``tool_calls``.
        """
        self._awaiting.pop(call_id).set_result(message)

    def cancel(self, call_id: int) -> None:
        """Fail the held call ``call_id``."""
        self._awaiting.pop(call_id).set_exception(ModelCallCancelled())

    def reset(self) -> None:
        """Fail every unanswered call and drop the queue."""
        while self._awaiting:
            _, reply = self._awaiting.popitem()
            reply.set_exception(ModelCallCancelled())
        self._pending = asyncio.Queue()

    async def _complete(self, request: web.Request) -> web.StreamResponse:
        """Hold one chat completion open until answered; a cancelled call gets a 503."""
        pending = self._pending
        body = await request.json()
        call_id = self._next_call_id
        self._next_call_id += 1
        reply: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._awaiting[call_id] = reply
        await pending.put((call_id, body))
        try:
            message = await reply
        except ModelCallCancelled:
            return web.json_response(
                {"error": {"message": "model call cancelled"}}, status=503
            )
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
