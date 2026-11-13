# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Stitch call records into sessions and segments.

Standard library only, so the trace query service and the inference pod
import it without the LLM extras.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta

from agilerl.llm_envs.call_record import CallRecord

__all__ = [
    "SESSION_IDLE_TIMEOUT",
    "Session",
    "extends",
    "stitch_sessions",
]

SESSION_IDLE_TIMEOUT = timedelta(minutes=30)


@dataclass(frozen=True)
class Session:
    """The calls of one agent run, as segments of calls whose prompts each extend the last.

    :param key: The ``client_session_id`` when the calls carry one, else the
        ``call_id`` of the session's first call.
    :param segments: Calls in order, split where a prompt does not extend the
        previous call.
    """

    key: str
    segments: list[list[CallRecord]] = field(default_factory=list)

    @property
    def tail(self) -> CallRecord:
        """The session's latest call."""
        return self.segments[-1][-1]


def extends(previous: CallRecord, call: CallRecord) -> bool:
    """Whether ``call``'s prompt continues ``previous``'s prompt, and its reply when recorded.

    :param previous: The earlier call.
    :param call: The later call.
    :return: ``True`` when ``call``'s messages start with ``previous``'s
        messages followed by ``previous``'s response, or, when that response
        has no hash, with ``previous``'s messages and at least one more.
    """
    known = len(previous.prompt_prefix_hashes)
    hashes = call.prompt_prefix_hashes
    if previous.response_prefix_hash is not None:
        return len(hashes) > known and hashes[known] == previous.response_prefix_hash
    return (
        known > 0
        and len(hashes) > known
        and hashes[known - 1] == previous.prompt_prefix_hashes[-1]
    )


def stitch_sessions(
    calls: Iterable[CallRecord], idle_timeout: timedelta = SESSION_IDLE_TIMEOUT
) -> list[Session]:
    """Group calls into sessions and segments.

    Calls sharing a ``client_session_id`` form one session. A call without one
    joins the open session, without a client id, of the same deployment, user
    and end user whose latest call it :func:`extends`, a ``None`` user or end
    user matching only ``None``; the most recent such
    session when several match. A session without a client id closes after
    ``idle_timeout`` with no call, and each of its calls is extended at most
    once, so a call that repeats an extended prompt starts its own session.
    A call with no prompt or response hash is not offered for extension.
    Inside a session, a call that does not extend the previous one starts a
    new segment.

    :param calls: Call records, in any order.
    :param idle_timeout: How long a session without a client id stays open.
    :return: Sessions ordered by their first call, calls ordered by
        ``received_at`` then ``call_id``.
    """
    ordered = sorted(calls, key=lambda call: (call.received_at, call.call_id))
    sessions: list[Session] = []
    by_client_id: dict[str, Session] = {}
    # Open sessions without a client id, by identity and their tail's hash.
    by_tail: dict[tuple[str, str | None, str | None, str], Session] = {}
    for call in ordered:
        if call.client_session_id is not None:
            session = by_client_id.get(call.client_session_id)
            if session is None:
                session = Session(call.client_session_id, [[call]])
                by_client_id[call.client_session_id] = session
                sessions.append(session)
            elif extends(session.tail, call):
                session.segments[-1].append(call)
            else:
                session.segments.append([call])
            continue
        identity = (call.deployment_id, call.user_id, call.end_user)
        session = _extended_session(call, identity, by_tail, idle_timeout)
        if session is None:
            session = Session(call.call_id, [[call]])
            sessions.append(session)
        else:
            del by_tail[(*identity, _tail_hash(session.tail))]
            session.segments[-1].append(call)
        if call.response_prefix_hash is not None or call.prompt_prefix_hashes:
            by_tail[(*identity, _tail_hash(call))] = session
    return sessions


def _extended_session(
    call: CallRecord,
    identity: tuple[str, str | None, str | None],
    by_tail: dict[tuple[str, str | None, str | None, str], Session],
    idle_timeout: timedelta,
) -> Session | None:
    """The open session ``call`` extends, preferring the longest shared prefix."""
    for tail_hash in reversed(call.prompt_prefix_hashes):
        session = by_tail.get((*identity, tail_hash))
        if (
            session is not None
            and call.received_at - session.tail.received_at <= idle_timeout
            and extends(session.tail, call)
        ):
            return session
    return None


def _tail_hash(call: CallRecord) -> str:
    """The hash a call extending ``call`` repeats: its response's, else its prompt's."""
    if call.response_prefix_hash is not None:
        return call.response_prefix_hash
    return call.prompt_prefix_hashes[-1]
