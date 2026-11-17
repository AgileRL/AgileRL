# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

pq = pytest.importorskip("pyarrow.parquet", reason="LLM dependencies not installed")
transformers = pytest.importorskip(
    "transformers", reason="LLM dependencies not installed"
)

from agilerl.arena.models.traces import validate_traces_file  # noqa: E402
from agilerl.arena.traces_format import (  # noqa: E402
    ExcludedCounts,
    ModelVersionCount,
    SnapshotFile,
    TimeRange,
    read_manifest,
)
from agilerl.llm_envs.call_record import ScoreRecord  # noqa: E402
from agilerl.llm_envs.call_sessions import stitch_sessions  # noqa: E402
from agilerl.llm_envs.call_trace import (  # noqa: E402
    CALL_RECORD_SCHEMA,
    SNAPSHOT_SCHEMA,
    calls_table,
    snapshot_segments,
    snapshot_table,
    write_snapshot,
)
from agilerl.llm_envs.dataset_env import encode_sft_row  # noqa: E402
from tests import TINY_LLM_FIXTURE_PATH  # noqa: E402
from tests.test_llm_envs.call_factory import (  # noqa: E402
    START,
    assistant,
    make_call,
    user,
)

TOOLS = [{"type": "function", "function": {"name": "add", "parameters": {}}}]
CALL_REPLY = assistant(
    "",
    reasoning_content="add them",
    tool_calls=[
        {
            "id": "t0",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a": 1}'},
        }
    ],
)
# The harness resends the reply without its reasoning.
RESENT_REPLY = {
    key: value for key, value in CALL_REPLY.items() if key != "reasoning_content"
}
TOOL_RESULT = {"role": "tool", "tool_call_id": "t0", "content": "2"}


def agent_run() -> list:
    """Two calls of a tool-using run, then a compacted restart, all in one client session."""
    first = [user("1+1?")]
    second = [*first, RESENT_REPLY, TOOL_RESULT]
    third = [user("summary: it is 2"), user("say it")]
    return [
        make_call("a", first, CALL_REPLY, 0, "run", tools=TOOLS),
        make_call("b", second, assistant("2"), 1, "run", tools=TOOLS),
        make_call("c", third, assistant("2."), 2, "run"),
    ]


def segment_ids(calls: list) -> list[list[str]]:
    """The call ids of each snapshot segment ``calls`` stitch into."""
    segments, _ = snapshot_segments(stitch_sessions(calls))
    return [[call.call_id for call in segment.calls] for segment in segments]


class TestSnapshotSegments:
    def test_appended_calls_share_a_segment_and_a_rewrite_starts_one(self) -> None:
        # Arrange
        sessions = stitch_sessions(agent_run())

        # Act
        segments, excluded = snapshot_segments(sessions)

        # Assert
        assert [
            (segment.session_key, [c.call_id for c in segment.calls])
            for segment in segments
        ] == [
            ("run", ["a", "b"]),
            ("run", ["c"]),
        ]
        assert excluded == ExcludedCounts(0, 0)

    def test_a_model_version_change_starts_a_segment(self) -> None:
        # Arrange
        first = [user("q")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run"),
            dataclasses.replace(
                make_call(
                    "b", [*first, assistant("r"), user("n")], assistant("s"), 1, "run"
                ),
                model_version="v2",
            ),
        ]

        # Act / Assert
        assert segment_ids(calls) == [["a"], ["b"]]

    def test_a_tools_change_starts_a_segment(self) -> None:
        first = [user("q")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run"),
            make_call(
                "b",
                [*first, assistant("r"), user("n")],
                assistant("s"),
                1,
                "run",
                tools=TOOLS,
            ),
        ]

        assert segment_ids(calls) == [["a"], ["b"]]

    def test_an_inserted_assistant_message_starts_a_segment(self) -> None:
        # Arrange: the harness adds an assistant message no call produced.
        first = [user("q")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run"),
            make_call(
                "b",
                [*first, assistant("r"), assistant("inserted"), user("n")],
                assistant("s"),
                1,
                "run",
            ),
        ]

        # Act / Assert
        assert segment_ids(calls) == [["a"], ["b"]]

    def test_a_call_after_an_excluded_one_must_append_to_the_last_kept_call(
        self,
    ) -> None:
        # Arrange: "b" failed; "c" extends "b"'s request, not "a"'s response.
        first = [user("q")]
        second = [*first, assistant("r"), user("n")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run"),
            make_call("b", second, None, 1, "run"),
            make_call("c", [*second, user("again")], assistant("s"), 2, "run"),
            make_call(
                "d",
                [*second, user("again"), assistant("s"), user("more")],
                assistant("t"),
                3,
                "run",
            ),
        ]

        # Act
        segments, excluded = snapshot_segments(stitch_sessions(calls))

        # Assert
        assert [[c.call_id for c in segment.calls] for segment in segments] == [
            ["a", "c", "d"]
        ]
        assert excluded == ExcludedCounts(0, 1)

    def test_a_call_that_drops_the_previous_reply_starts_a_segment(self) -> None:
        # Arrange: "c" resends "b"'s request without its reply, after "b" failed.
        first = [user("q")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run"),
            make_call("b", [*first, user("n")], None, 1, "run"),
            make_call("c", [*first, user("n"), user("m")], assistant("s"), 2, "run"),
        ]

        # Act / Assert
        assert segment_ids(calls) == [["a"], ["c"]]

    def test_a_dropped_reply_starts_a_segment_without_a_response_hash(
        self,
    ) -> None:
        # Arrange: without a response hash, stitching checks only the prompt prefix.
        first = [user("q")]
        calls = [
            dataclasses.replace(
                make_call("a", first, assistant("r"), 0, "run"),
                response_prefix_hash=None,
            ),
            make_call("b", [*first, user("n")], assistant("s"), 1, "run"),
        ]

        # Act
        rows = snapshot_table(stitch_sessions(calls))[0].to_pylist()

        # Assert
        assert segment_ids(calls) == [["a"], ["b"]]
        assert json.loads(rows[1]["messages"]) == [user("q"), user("n"), assistant("s")]

    def test_empty_and_missing_tools_share_a_segment(self) -> None:
        first = [user("q")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run", tools=[]),
            make_call(
                "b", [*first, assistant("r"), user("n")], assistant("s"), 1, "run"
            ),
        ]

        assert segment_ids(calls) == [["a", "b"]]


class TestSnapshotTable:
    def test_one_row_per_segment_with_per_turn_lists(self) -> None:
        # Arrange
        calls = agent_run()
        calls[0] = dataclasses.replace(
            calls[0],
            metadata={"group": "g"},
            completion_token_ids=[3, 4],
            completion_logprobs=[-0.25, -0.5],
        )
        sessions = stitch_sessions(calls)

        # Act
        table, excluded = snapshot_table(sessions)

        # Assert
        rows = table.to_pylist()
        assert table.schema == SNAPSHOT_SCHEMA
        assert [
            (r["session_id"], r["segment_index"], r["call_ids"], r["finish_reasons"])
            for r in rows
        ] == [
            ("run", 0, ["a", "b"], ["stop", "stop"]),
            ("run", 1, ["c"], ["stop"]),
        ]
        assert json.loads(rows[0]["messages"]) == [
            user("1+1?"),
            CALL_REPLY,
            TOOL_RESULT,
            assistant("2"),
        ]
        assert json.loads(rows[0]["tools"]) == TOOLS
        assert rows[0]["completion_token_ids"] == [[3, 4], None]
        assert rows[0]["completion_logprobs"] == [[-0.25, -0.5], None]
        assert json.loads(rows[0]["metadata"]) == {"group": "g"}
        assert (rows[0]["received_at"], rows[0]["model_version"]) == (START, "v1")
        assert (rows[0]["base_model"], rows[0]["checkpoint_id"]) == ("tiny", None)
        assert json.loads(rows[1]["messages"]) == [
            user("summary: it is 2"),
            user("say it"),
            assistant("2."),
        ]
        assert (rows[1]["tools"], rows[1]["metadata"]) == (None, None)
        assert (rows[1]["scores"], rows[1]["answer"]) == (None, None)
        assert excluded == ExcludedCounts(0, 0)

    def test_an_unrecorded_prompt_assistant_turn_has_no_call(self) -> None:
        # Arrange: "a" failed, so the harness's own reply "s" has no call.
        first = [user("q"), assistant("r"), user("n")]
        calls = [
            make_call("a", first, None, 0, "run"),
            make_call(
                "b", [*first, assistant("s"), user("m")], assistant("t"), 1, "run"
            ),
        ]

        # Act
        (row,) = snapshot_table(stitch_sessions(calls))[0].to_pylist()

        # Assert
        assert json.loads(row["messages"]) == [
            *first,
            assistant("s"),
            user("m"),
            assistant("t"),
        ]
        assert row["call_ids"] == [None, None, "b"]
        assert row["finish_reasons"] == [None, None, "stop"]
        assert row["completion_token_ids"] == [None, None, None]

    def test_assistant_reasoning_is_reasoning_content_and_arguments_json_text(
        self,
    ) -> None:
        # Arrange: the harness resends the reply with dict arguments and thinking.
        reply = {
            "role": "assistant",
            "content": None,
            "thinking": "add",
            "tool_calls": [
                {
                    "id": "t0",
                    "type": "function",
                    "function": {"name": "add", "arguments": {"a": 1}},
                }
            ],
        }
        calls = [
            make_call("a", [user("q")], reply, 0, "run", tools=TOOLS),
            make_call(
                "b",
                [user("q"), reply, TOOL_RESULT],
                assistant("2"),
                1,
                "run",
                tools=TOOLS,
            ),
        ]

        # Act
        (row,) = snapshot_table(stitch_sessions(calls))[0].to_pylist()

        # Assert
        assert json.loads(row["messages"])[1] == {
            "role": "assistant",
            "content": None,
            "reasoning_content": "add",
            "tool_calls": [
                {
                    "id": "t0",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1}'},
                }
            ],
        }

    def test_a_reply_with_conflicting_reasoning_is_counted_invalid(self) -> None:
        # Arrange
        reply = assistant("r", thinking="a", reasoning_content="b")
        calls = [
            make_call("a", [user("q")], reply, 0),
            make_call("b", [user("x")], assistant("y"), 1),
        ]

        # Act
        table, excluded = snapshot_table(stitch_sessions(calls))

        # Assert
        assert table.column("call_ids").to_pylist() == [["b"]]
        assert excluded == ExcludedCounts(invalid=1)

    def test_excludes_and_counts_calls_that_are_not_traces(self) -> None:
        # Arrange
        truncated = dataclasses.replace(
            make_call("long", [user("q")], assistant("cut")), finish_reason="length"
        )
        calls = [
            truncated,
            make_call("err", [user("x")], None, 1),
            dataclasses.replace(
                make_call("gone", [user("y")], None, 2), finish_reason="cancelled"
            ),
            make_call("meta", [user("z")], assistant("w"), 3, content=False),
        ]

        # Act
        table, excluded = snapshot_table(stitch_sessions(calls))

        # Assert
        assert table.column("call_ids").to_pylist() == [["long"]]
        assert table.column("finish_reasons").to_pylist() == [["length"]]
        assert excluded == ExcludedCounts(no_content=1, error_or_cancelled=2)

    def test_a_segment_that_breaks_the_spec_is_left_out_and_counted(self) -> None:
        # Arrange: "b" replies with reasoning only, which the spec rejects.
        first = [user("q")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run"),
            make_call(
                "b",
                [*first, assistant("r"), user("n")],
                assistant("", reasoning_content="thinking"),
                1,
                "run",
            ),
            make_call("z", [user("other")], assistant("o"), 2),
        ]

        # Act
        table, excluded = snapshot_table(stitch_sessions(calls))

        # Assert
        assert table.column("call_ids").to_pylist() == [["z"]]
        assert excluded == ExcludedCounts(invalid=2)

    def test_joins_call_scores_on_their_turn_and_session_scores_on_the_row(
        self,
    ) -> None:
        # Arrange
        sessions = stitch_sessions(agent_run())
        scores = [
            ScoreRecord("b", None, "judge-1:r1", 0.75, "judge"),
            ScoreRecord(None, "run", "outcome", 1.0, "harness", label="solved"),
            ScoreRecord(
                "c", None, "feedback", None, "user", label="rude", comment="too short"
            ),
        ]
        outcome = {
            "name": "outcome",
            "value": 1.0,
            "source": "harness",
            "label": "solved",
            "comment": None,
            "turn": None,
        }

        # Act
        rows = snapshot_table(sessions, scores)[0].to_pylist()

        # Assert
        assert [row["scores"] for row in rows] == [
            [
                {
                    "name": "judge-1:r1",
                    "value": 0.75,
                    "source": "judge",
                    "label": None,
                    "comment": None,
                    "turn": 1,
                },
                outcome,
            ],
            [
                {
                    "name": "feedback",
                    "value": None,
                    "source": "user",
                    "label": "rude",
                    "comment": "too short",
                    "turn": 0,
                },
                outcome,
            ],
        ]

    @pytest.mark.parametrize(("call_id", "session_key"), [("a", "run"), (None, None)])
    def test_rejects_a_score_not_on_exactly_one_call_or_session(
        self, call_id: str | None, session_key: str | None
    ) -> None:
        sessions = stitch_sessions(agent_run())
        scores = [ScoreRecord(call_id, session_key, "outcome", 1.0, "harness")]

        with pytest.raises(ValueError, match="exactly one of call_id and session_key"):
            snapshot_table(sessions, scores)

    def test_calls_table_has_the_call_record_schema(self) -> None:
        table = calls_table(agent_run())

        assert table.schema == CALL_RECORD_SCHEMA
        assert table.num_rows == 3

    def test_rows_encode_on_the_sft_path(self) -> None:
        # Arrange
        tokenizer = transformers.AutoTokenizer.from_pretrained(TINY_LLM_FIXTURE_PATH)
        calls = [
            make_call("a", [user("hi")], assistant("hello"), 0, "run"),
            make_call(
                "b",
                [user("hi"), assistant("hello"), user("again")],
                assistant("hello again"),
                1,
                "run",
            ),
        ]
        (row,) = snapshot_table(stitch_sessions(calls))[0].to_pylist()

        # Act
        ids, mask = encode_sft_row(tokenizer, row, "response", None)

        # Assert
        trained = tokenizer.decode([i for i, m in zip(ids, mask, strict=True) if m])
        assert trained.count("hello") == 2
        assert trained.count("again") == 1


class TestWriteSnapshot:
    def test_writes_segments_and_a_manifest_computed_from_them(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        calls = [
            *agent_run(),
            dataclasses.replace(
                make_call("z", [user("other")], assistant("o"), 5),
                model_version="v2",
                checkpoint_id="ck-2",
            ),
            make_call("err", [user("x")], None, 6),
        ]
        sessions = stitch_sessions(calls)
        scores = [ScoreRecord(None, "run", "outcome", 1.0, "harness")]

        # Act
        manifest = write_snapshot(
            tmp_path,
            sessions,
            "user-9",
            {"deployment": "dep-1"},
            scores,
            parent_snapshot_id="01PARENT",
            label_score="outcome",
        )

        # Assert
        segments_path = tmp_path / "segments.parquet"
        written = json.loads((tmp_path / "manifest.json").read_text())
        assert read_manifest(tmp_path / "manifest.json") == manifest
        assert (manifest.row_count, written["turn_count"]) == (3, 4)
        assert manifest.session_count == 2
        assert manifest.model_versions == [
            ModelVersionCount("v1", "tiny", None, 2),
            ModelVersionCount("v2", "tiny", "ck-2", 1),
        ]
        assert manifest.time_range == TimeRange(START, START + timedelta(minutes=5))
        assert manifest.excluded == ExcludedCounts(0, 1)
        assert (manifest.score_names, manifest.label_score) == (["outcome"], "outcome")
        assert (manifest.organization_id, manifest.deployment_ids) == (
            "org-1",
            ["dep-1"],
        )
        assert manifest.files == [
            SnapshotFile(
                "segments.parquet",
                3,
                hashlib.sha256(segments_path.read_bytes()).hexdigest(),
            )
        ]
        assert (manifest.created_by, manifest.filter) == (
            "user-9",
            {"deployment": "dep-1"},
        )
        assert manifest.parent_snapshot_id == "01PARENT"
        assert len(manifest.snapshot_id) == 26
        assert not (tmp_path / "calls.parquet").exists()

    def test_segments_round_trip_through_parquet_and_validate(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        sessions = stitch_sessions(agent_run())
        scores = [ScoreRecord("b", None, "judge-1:r1", 0.75, "judge")]
        expected = snapshot_table(sessions, scores)[0]

        # Act
        write_snapshot(tmp_path, sessions, "user-9", {}, scores)
        read = pq.read_table(tmp_path / "segments.parquet")

        # Assert
        assert read.schema == SNAPSHOT_SCHEMA
        assert read.to_pylist() == expected.to_pylist()
        assert validate_traces_file(tmp_path / "segments.parquet") == 2

    def test_an_empty_snapshot_has_no_rows_or_time_range(self, tmp_path: Path) -> None:
        manifest = write_snapshot(tmp_path, [], "user-9", {})

        assert (manifest.row_count, manifest.time_range) == (0, None)
        assert json.loads((tmp_path / "manifest.json").read_text())["turn_count"] == 0

    def test_time_range_spans_the_rows_received_at(self, tmp_path: Path) -> None:
        first = [user("q")]
        calls = [
            make_call("a", first, assistant("r"), 0, "run"),
            make_call(
                "b", [*first, assistant("r"), user("n")], assistant("s"), 3, "run"
            ),
        ]

        manifest = write_snapshot(tmp_path, stitch_sessions(calls), "user-9", {})

        assert manifest.time_range == TimeRange(START, START)

    def test_rejects_a_label_score_no_row_has(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="label_score 'outcome'"):
            write_snapshot(
                tmp_path,
                stitch_sessions(agent_run()),
                "user-9",
                {},
                label_score="outcome",
            )

        assert not (tmp_path / "segments.parquet").exists()

    def test_a_query_json_cannot_encode_writes_nothing(self, tmp_path: Path) -> None:
        root = tmp_path / "snap"

        with pytest.raises(TypeError, match="not JSON serializable"):
            write_snapshot(
                root, stitch_sessions(agent_run()), "user-9", {"mark": object()}
            )

        assert not root.exists()

    def test_rows_without_scores_name_none(self, tmp_path: Path) -> None:
        manifest = write_snapshot(tmp_path, stitch_sessions(agent_run()), "user-9", {})

        assert manifest.score_names == []
        assert json.loads((tmp_path / "manifest.json").read_text())["turn_count"] == 3

    def test_rejects_calls_of_more_than_one_organisation(self, tmp_path: Path) -> None:
        calls = [
            make_call("a", [user("q")], assistant("r"), 0, organization_id="org-1"),
            make_call("b", [user("x")], assistant("y"), 1, organization_id="org-2"),
        ]

        with pytest.raises(ValueError, match="one organisation's calls"):
            write_snapshot(tmp_path, stitch_sessions(calls), "user-9", {})
