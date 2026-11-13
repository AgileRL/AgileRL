# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Hold an env-side loop's chat completions open until something answers them."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, NamedTuple

from aiohttp import web

from agilerl.llm_envs.call_record import (
    CAPTURE_OFF,
    CallOrigin,
    CallRecord,
    FinishReason,
    new_call_id,
    prompt_prefix_hashes,
)

__all__ = [
    "CallRecorder",
    "CallSample",
    "ModelCallEndpoint",
    "PendingCall",
]

SESSION_HEADER = "X-AgileRL-Session-Id"
END_USER_HEADER = "X-AgileRL-End-User"
METADATA_HEADER = "X-AgileRL-Metadata"
CAPTURE_HEADER = "X-AgileRL-Capture"
CALL_ID_HEADER = "X-AgileRL-Call-Id"
MAX_METADATA_KEYS = 16


class PendingCall(NamedTuple):
    """A chat completion the endpoint holds open until answered."""

    call_id: int
    body: dict[str, Any]


@dataclass(frozen=True)
class CallSample:
    """What the policy sampled to answer a call, for its call record."""

    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    model_version: str
    checkpoint_id: str | None = None
    raw_completion: str | None = None
    sampling: dict[str, Any] = field(default_factory=dict)
    finish_reason: FinishReason | None = None


@dataclass(frozen=True)
class CallRecorder:
    """Where a recording endpoint sends one :class:`CallRecord` per answered call."""

    origin: CallOrigin
    write: Callable[[CallRecord], None]


class RequestFacts(NamedTuple):
    """A held call's request facts its record needs."""

    received_at: datetime
    started: float
    session_id: str | None
    end_user: str | None
    metadata: dict[str, str]


class ModelCallCancelled(Exception):
    """A held completion was failed by :meth:`ModelCallEndpoint.cancel` or ``reset``."""


class ModelCallEndpoint:
    """A loopback OpenAI-compatible endpoint whose completions something else answers.

    Serves ``POST /v1/chat/completions`` (plain and streamed) on an ephemeral
    loopback port at :attr:`base_url`. :meth:`next_call` waits for the next
    request; :meth:`answer` replies to it. Calls take sequential ids.

    With a ``recorder``, every answered call not sent with
    ``X-AgileRL-Capture: off`` is written as one :class:`CallRecord`, and the
    response carries its ``call_id`` as ``id`` and in ``X-AgileRL-Call-Id``.
    Cancelled calls are not recorded.

    :param recorder: Where call records go; ``None`` records nothing.
    """

    def __init__(self, recorder: CallRecorder | None = None) -> None:
        """Reserve the port; :meth:`open` starts serving."""
        self._recorder = recorder
        self._taken: dict[int, float] = {}
        self._socket = socket.socket()
        self._socket.bind(("127.0.0.1", 0))
        self.base_url = f"http://127.0.0.1:{self._socket.getsockname()[1]}/v1"
        self._pending: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
        self._awaiting: dict[
            int, asyncio.Future[tuple[dict[str, Any], CallSample | None]]
        ] = {}
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
        if self._recorder is not None:
            self._taken[call_id] = time.monotonic()
        return PendingCall(call_id, body)

    def answer(
        self, call_id: int, message: dict[str, Any], sample: CallSample | None = None
    ) -> None:
        """Reply ``message`` to the held call.

        :param call_id: Id from :meth:`next_call`.
        :param message: The assistant message: ``content`` and/or ``tool_calls``.
        :param sample: The tokens behind ``message``; required when recording.
        :raises ValueError: If the endpoint records and ``sample`` is ``None``.
            The held request is cancelled.
        """
        if self._recorder is not None and sample is None:
            pending = self._awaiting.pop(call_id, None)
            if pending is not None:
                pending.set_exception(ModelCallCancelled())
            msg = (
                "a recording ModelCallEndpoint needs the CallSample behind each answer"
            )
            raise ValueError(msg)
        self._awaiting.pop(call_id).set_result((message, sample))

    def cancel(self, call_id: int) -> None:
        """Fail the held call ``call_id``."""
        self._awaiting.pop(call_id).set_exception(ModelCallCancelled())

    def reset(self) -> None:
        """Fail every unanswered call and drop the queue."""
        while self._awaiting:
            _, reply = self._awaiting.popitem()
            reply.set_exception(ModelCallCancelled())
        self._pending = asyncio.Queue()
        self._taken.clear()

    async def _complete(self, request: web.Request) -> web.StreamResponse:
        """Hold one chat completion open until answered; a cancelled call gets a 503."""
        pending = self._pending
        body = await request.json()
        facts = None
        if (
            self._recorder is not None
            and request.headers.get(CAPTURE_HEADER) != CAPTURE_OFF
        ):
            metadata = _metadata(request.headers.get(METADATA_HEADER))
            if isinstance(metadata, str):
                return web.json_response({"error": {"message": metadata}}, status=400)
            facts = RequestFacts(
                received_at=datetime.now(timezone.utc),
                started=time.monotonic(),
                session_id=request.headers.get(SESSION_HEADER),
                end_user=request.headers.get(END_USER_HEADER, body.get("user")),
                metadata=metadata,
            )
        call_id = self._next_call_id
        self._next_call_id += 1
        reply: asyncio.Future[tuple[dict[str, Any], CallSample | None]] = (
            asyncio.get_running_loop().create_future()
        )
        self._awaiting[call_id] = reply
        await pending.put((call_id, body))
        try:
            message, sample = await reply
        except ModelCallCancelled:
            return web.json_response(
                {"error": {"message": "model call cancelled"}}, status=503
            )
        finally:
            self._awaiting.pop(call_id, None)
            taken = self._taken.pop(call_id, None)
        response_id = "agilerl"
        headers: dict[str, str] = {}
        if self._recorder is not None and facts is not None and sample is not None:
            record = _record(self._recorder.origin, body, message, sample, facts, taken)
            self._recorder.write(record)
            response_id = record.call_id
            headers[CALL_ID_HEADER] = record.call_id
        finish_reason = None if sample is None else sample.finish_reason
        if not body.get("stream"):
            return web.json_response(
                _completion(body["model"], message, response_id, finish_reason),
                headers=headers,
            )
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", **headers}
        )
        await response.prepare(request)
        for chunk in _completion_chunks(
            body["model"], message, response_id, finish_reason
        ):
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response


def _metadata(header: str | None) -> dict[str, str] | str:
    """The ``X-AgileRL-Metadata`` object, or why it is rejected."""
    if header is None:
        return {}
    try:
        value = json.loads(header)
    except json.JSONDecodeError as error:
        return f"{METADATA_HEADER} is not valid JSON: {error}"
    if (
        not isinstance(value, dict)
        or len(value) > MAX_METADATA_KEYS
        or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        )
    ):
        return (
            f"{METADATA_HEADER} must be a JSON object of at most "
            f"{MAX_METADATA_KEYS} string keys and string values."
        )
    return value


def _record(
    origin: CallOrigin,
    body: dict[str, Any],
    message: dict[str, Any],
    sample: CallSample,
    facts: RequestFacts,
    taken: float | None,
) -> CallRecord:
    """The call record of one answered call."""
    messages = body["messages"]
    hashes = prompt_prefix_hashes(messages)
    response_hash = prompt_prefix_hashes([message], hashes[-1] if hashes else None)
    call_id = new_call_id()
    total_ms = round((time.monotonic() - facts.started) * 1000)
    return CallRecord(
        call_id=call_id,
        request_id=call_id,
        organization_id=origin.organization_id,
        deployment_id=origin.deployment_id,
        instance_id=origin.instance_id,
        user_id=origin.user_id,
        end_user=facts.end_user,
        client_session_id=facts.session_id,
        metadata=facts.metadata,
        base_model=origin.base_model,
        checkpoint_id=sample.checkpoint_id,
        model_version=sample.model_version,
        chat_template_hash=origin.chat_template_hash,
        sampling=sample.sampling,
        received_at=facts.received_at,
        queue_ms=None if taken is None else round((taken - facts.started) * 1000),
        # The reply is written whole, so its first token arrives with the rest.
        first_token_ms=total_ms,
        total_ms=total_ms,
        prompt_tokens=len(sample.prompt_token_ids),
        completion_tokens=len(sample.completion_token_ids),
        finish_reason=sample.finish_reason or _finish(message),
        stream=bool(body.get("stream")),
        prompt_prefix_hashes=hashes,
        response_prefix_hash=response_hash[0],
        messages=messages,
        tools=body.get("tools"),
        tool_choice=body.get("tool_choice"),
        response_message=message,
        raw_completion=sample.raw_completion,
        prompt_token_ids=sample.prompt_token_ids,
        completion_token_ids=sample.completion_token_ids,
        completion_logprobs=sample.completion_logprobs,
    )


def _completion(
    model: str,
    message: dict[str, Any],
    response_id: str,
    finish_reason: FinishReason | None = None,
) -> dict[str, Any]:
    """A chat completion answering with ``message``."""
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or _finish(message),
            }
        ],
    }


def _completion_chunks(
    model: str,
    message: dict[str, Any],
    response_id: str,
    finish_reason: FinishReason | None = None,
) -> list[dict[str, Any]]:
    """Streamed chat completion chunks answering with ``message``, usage last."""

    def chunk(choices: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        return {
            "id": response_id,
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
        chunk(
            [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason or _finish(message),
                }
            ]
        ),
        chunk(
            [],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    ]


def _finish(message: dict[str, Any]) -> FinishReason:
    """The finish reason for an assistant ``message``."""
    return "tool_calls" if message.get("tool_calls") else "stop"
