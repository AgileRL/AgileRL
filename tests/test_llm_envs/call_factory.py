# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Build call records the way a recorder would, for the trace tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from agilerl.llm_envs.call_record import CallRecord, prompt_prefix_hashes

START = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)


def make_call(
    call_id: str,
    messages: list[dict[str, Any]],
    response: dict[str, Any] | None,
    minute: float = 0.0,
    client_session_id: str | None = None,
    user_id: str | None = "user-1",
    end_user: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    content: bool = True,
    organization_id: str | None = "org-1",
) -> CallRecord:
    """A call record for ``messages`` answered with ``response``, ``minute`` minutes after :data:`START`."""
    hashes = prompt_prefix_hashes(messages)
    response_hash = (
        None if response is None else prompt_prefix_hashes([response], hashes[-1])[0]
    )
    return CallRecord(
        call_id=call_id,
        request_id=call_id,
        organization_id=organization_id,
        deployment_id="dep-1",
        instance_id="pod-1",
        user_id=user_id,
        end_user=end_user,
        client_session_id=client_session_id,
        base_model="tiny",
        checkpoint_id=None,
        model_version="v1",
        chat_template_hash="t1",
        received_at=START + timedelta(minutes=minute),
        total_ms=5,
        finish_reason="stop" if response is not None else "error",
        stream=False,
        prompt_prefix_hashes=hashes,
        response_prefix_hash=response_hash,
        messages=messages if content else None,
        tools=tools if content else None,
        response_message=response if content else None,
    )


def user(text: str) -> dict[str, Any]:
    """A user message."""
    return {"role": "user", "content": text}


def assistant(text: str, **extra: Any) -> dict[str, Any]:
    """An assistant message."""
    return {"role": "assistant", "content": text, **extra}
