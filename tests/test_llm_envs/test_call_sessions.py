# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

from agilerl.llm_envs.call_sessions import Session, extends, stitch_sessions
from tests.test_llm_envs.call_factory import assistant, make_call, user

TOOL_RESULT = {"role": "tool", "tool_call_id": "t0", "content": "42"}


def layout(sessions: list[Session]) -> list[tuple[str, list[list[str]]]]:
    """Each session's key and its segments' call ids."""
    return [
        (
            session.key,
            [[call.call_id for call in segment] for segment in session.segments],
        )
        for session in sessions
    ]


class TestExtends:
    def test_a_prompt_repeating_the_reply_extends(self) -> None:
        first = make_call("a", [user("q")], assistant("r"))
        second = make_call("b", [user("q"), assistant("r"), TOOL_RESULT], None)

        assert extends(first, second)

    def test_a_prompt_with_another_reply_does_not_extend(self) -> None:
        first = make_call("a", [user("q")], assistant("r"))
        second = make_call("b", [user("q"), assistant("other"), user("x")], None)

        assert not extends(first, second)

    def test_without_a_response_hash_a_longer_shared_prompt_extends(self) -> None:
        first = dataclasses.replace(
            make_call("a", [user("q")], None), response_prefix_hash=None
        )

        assert extends(first, make_call("b", [user("q"), assistant("r")], None))
        assert not extends(first, make_call("c", [user("q")], None))


class TestStitchSessions:
    def test_prefix_matching_calls_without_an_id_form_one_session(self) -> None:
        # Arrange
        calls = [
            make_call("a", [user("q")], assistant("r1"), minute=0),
            make_call(
                "b",
                [user("q"), assistant("r1"), TOOL_RESULT],
                assistant("r2"),
                minute=1,
            ),
            make_call("z", [user("other")], assistant("o"), minute=0.5),
        ]

        # Act
        sessions = stitch_sessions(reversed(calls))

        # Assert
        assert layout(sessions) == [("a", [["a", "b"]]), ("z", [["z"]])]

    def test_client_session_id_groups_and_a_rewrite_starts_a_segment(self) -> None:
        # Arrange
        calls = [
            make_call("a", [user("q")], assistant("r1"), 0, "run-1"),
            make_call(
                "b",
                [user("q"), assistant("r1"), user("go")],
                assistant("r2"),
                1,
                "run-1",
            ),
            # The harness compacted its history.
            make_call(
                "c", [user("summary"), user("go on")], assistant("r3"), 2, "run-1"
            ),
            make_call(
                "d",
                [user("summary"), user("go on"), assistant("r3"), user("k")],
                assistant("r4"),
                3,
                "run-1",
            ),
        ]

        # Act
        sessions = stitch_sessions(calls)

        # Assert
        assert layout(sessions) == [("run-1", [["a", "b"], ["c", "d"]])]

    def test_parallel_rollouts_of_one_prompt_stay_apart(self) -> None:
        # Arrange: two rollouts open with the same prompt and reply differently.
        calls = [
            make_call("a1", [user("q")], assistant("x"), 0),
            make_call("b1", [user("q")], assistant("y"), 0.1),
            make_call(
                "b2", [user("q"), assistant("y"), user("more")], assistant("y2"), 0.2
            ),
            make_call(
                "a2", [user("q"), assistant("x"), user("more")], assistant("x2"), 0.3
            ),
        ]

        # Act
        sessions = stitch_sessions(calls)

        # Assert
        assert layout(sessions) == [("a1", [["a1", "a2"]]), ("b1", [["b1", "b2"]])]

    def test_other_user_or_end_user_does_not_join(self) -> None:
        calls = [
            make_call("a", [user("q")], assistant("r"), 0, end_user="u1"),
            make_call(
                "b", [user("q"), assistant("r"), user("n")], None, 1, end_user="u2"
            ),
            make_call(
                "c",
                [user("q"), assistant("r"), user("n")],
                None,
                1,
                user_id="user-2",
                end_user="u1",
            ),
        ]

        assert layout(stitch_sessions(calls)) == [
            ("a", [["a"]]),
            ("b", [["b"]]),
            ("c", [["c"]]),
        ]

    def test_calls_without_a_user_stitch_only_with_each_other(self) -> None:
        # Arrange: a deployment without an auth server records no user.
        follow_up = [user("q"), assistant("r"), user("n")]
        calls = [
            make_call("a", [user("q")], assistant("r"), 0, user_id=None),
            make_call("b", follow_up, assistant("s"), 1, user_id=None),
            make_call("c", follow_up, None, 2),
        ]

        # Act
        sessions = stitch_sessions(calls)

        # Assert
        assert layout(sessions) == [("a", [["a", "b"]]), ("c", [["c"]])]

    def test_an_idle_session_closes(self) -> None:
        calls = [
            make_call("a", [user("q")], assistant("r"), 0),
            make_call("b", [user("q"), assistant("r"), user("n")], None, 31),
        ]

        assert layout(stitch_sessions(calls)) == [("a", [["a"]]), ("b", [["b"]])]

    def test_a_call_extends_the_longest_matching_session(self) -> None:
        # Arrange
        calls = [
            make_call("a1", [user("q")], assistant("r"), 0),
            make_call("a2", [user("q"), assistant("r"), user("n")], assistant("s"), 1),
            make_call(
                "b",
                [user("q"), assistant("r"), user("n"), assistant("s"), user("m")],
                None,
                2,
            ),
        ]

        # Act / Assert
        assert layout(stitch_sessions(calls)) == [("a1", [["a1", "a2", "b"]])]

    def test_a_call_with_no_prefix_hash_is_not_offered_for_extension(self) -> None:
        # Arrange
        bare = dataclasses.replace(
            make_call("a", [], None),
            prompt_prefix_hashes=[],
            response_prefix_hash=None,
        )
        follow = make_call("b", [user("q")], assistant("r"), 1)

        # Act
        sessions = stitch_sessions([bare, follow])

        # Assert
        assert layout(sessions) == [("a", [["a"]]), ("b", [["b"]])]

    def test_a_retried_prompt_starts_its_own_session(self) -> None:
        calls = [
            make_call("a", [user("q")], assistant("r"), 0),
            make_call("b", [user("q"), assistant("r"), user("n")], assistant("s"), 1),
            make_call("c", [user("q"), assistant("r"), user("n")], assistant("s"), 2),
        ]

        assert layout(stitch_sessions(calls)) == [("a", [["a", "b"]]), ("c", [["c"]])]
