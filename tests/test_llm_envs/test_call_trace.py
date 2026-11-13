# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow", reason="LLM dependencies not installed")
pq = pytest.importorskip("pyarrow.parquet", reason="LLM dependencies not installed")
transformers = pytest.importorskip(
    "transformers", reason="LLM dependencies not installed"
)

from agilerl.arena.models.traces import validate_traces_file  # noqa: E402
from agilerl.arena.traces_format import (  # noqa: E402
    TRACE_COLUMNS,
    ExcludedCounts,
    ModelVersionCount,
    SnapshotFile,
    TimeRange,
    read_manifest,
)
from agilerl.llm_envs.call_record import (  # noqa: E402
    FeedbackRecord,
    ScoreRecord,
    feedback_score,
)
from agilerl.llm_envs.call_sessions import stitch_sessions  # noqa: E402
from agilerl.llm_envs.call_trace import (  # noqa: E402
    CALL_RECORD_SCHEMA,
    FEEDBACK_SCHEMA,
    SNAPSHOT_SCHEMA,
    calls_table,
    read_snapshot,
    sft_rows,
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


class TestSnapshotTable:
    def test_writes_traces_columns_beside_the_recorder_columns(self) -> None:
        # Arrange
        sessions = stitch_sessions(agent_run())

        # Act
        table, excluded = snapshot_table(sessions)

        # Assert
        rows = table.to_pylist()
        assert table.schema == SNAPSHOT_SCHEMA
        assert set(TRACE_COLUMNS) <= set(table.column_names)
        assert {"messages", "response_message", "session_key"}.isdisjoint(
            table.column_names
        )
        assert [(r["call_id"], r["session_id"], r["segment_index"]) for r in rows] == [
            ("a", "run", 0),
            ("b", "run", 0),
            ("c", "run", 1),
        ]
        assert json.loads(rows[1]["prompt"]) == [
            user("1+1?"),
            RESENT_REPLY,
            TOOL_RESULT,
        ]
        assert json.loads(rows[1]["completion"]) == [assistant("2")]
        assert json.loads(rows[1]["tools"]) == TOOLS
        assert rows[1]["client_session_id"] == "run"
        assert rows[1]["finish_reason"] == "stop"
        assert (rows[1]["scores"], rows[1]["answer"]) == (None, None)
        assert excluded == ExcludedCounts(0, 0)

    def test_completion_reasoning_is_reasoning_content_and_arguments_json_text(
        self,
    ) -> None:
        # Arrange
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
        sessions = stitch_sessions([make_call("a", [user("q")], reply)])

        # Act
        (row,) = snapshot_table(sessions)[0].to_pylist()

        # Assert
        assert json.loads(row["completion"]) == [
            {
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
        ]

    def test_rejects_a_reply_with_conflicting_reasoning(self) -> None:
        reply = assistant("r", thinking="a", reasoning_content="b")
        sessions = stitch_sessions([make_call("a", [user("q")], reply)])

        with pytest.raises(ValueError, match="to different reasoning"):
            snapshot_table(sessions)

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
        assert table.column("call_id").to_pylist() == ["long"]
        assert table.column("finish_reason").to_pylist() == ["length"]
        assert excluded == ExcludedCounts(no_content=1, error_or_cancelled=2)

    def test_joins_call_and_session_scores(self) -> None:
        # Arrange
        sessions = stitch_sessions(agent_run())
        scores = [
            ScoreRecord("b", None, "judge-1:r1", 0.75, "judge"),
            ScoreRecord(None, "run", "outcome", 1.0, "harness", label="solved"),
            feedback_score(
                FeedbackRecord(
                    feedback_id="f1",
                    call_id="c",
                    session_key=None,
                    score=None,
                    label="rude",
                    comment="too short",
                    source="user",
                    user_id=None,
                    created_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
                )
            ),
        ]
        outcome = {
            "name": "outcome",
            "value": 1.0,
            "source": "harness",
            "label": "solved",
            "comment": None,
        }

        # Act
        rows = snapshot_table(sessions, scores)[0].to_pylist()

        # Assert
        assert [row["scores"] for row in rows] == [
            [outcome],
            [
                {
                    "name": "judge-1:r1",
                    "value": 0.75,
                    "source": "judge",
                    "label": None,
                    "comment": None,
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


class TestWriteSnapshot:
    def test_writes_calls_and_a_manifest_computed_from_the_rows(
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
        calls_path = tmp_path / "calls.parquet"
        assert read_manifest(tmp_path / "manifest.json") == manifest
        assert manifest.row_count == 4
        assert manifest.session_count == 2
        assert manifest.model_versions == [
            ModelVersionCount("v1", "tiny", None, 3),
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
                "calls.parquet",
                4,
                hashlib.sha256(calls_path.read_bytes()).hexdigest(),
            )
        ]
        assert (manifest.created_by, manifest.filter) == (
            "user-9",
            {"deployment": "dep-1"},
        )
        assert manifest.parent_snapshot_id == "01PARENT"
        assert len(manifest.snapshot_id) == 26
        assert [session.key for session in read_snapshot(calls_path)] == ["run", "z"]

    def test_round_trips_sessions_through_read_snapshot(self, tmp_path: Path) -> None:
        # Arrange
        calls = agent_run()
        calls[0] = dataclasses.replace(
            calls[0],
            metadata={"group": "g"},
            prompt_token_ids=[1, 2],
            completion_token_ids=[3],
            completion_logprobs=[-0.25],
        )
        sessions = stitch_sessions(calls)

        # Act
        write_snapshot(tmp_path, sessions, "user-9", {})
        read = read_snapshot(tmp_path / "calls.parquet")

        # Assert
        assert read == sessions
        assert read[0].segments[0][0].completion_logprobs == [-0.25]
        assert validate_traces_file(tmp_path / "calls.parquet") == 3

    def test_prompt_assistant_turns_validate_with_the_completion(
        self, tmp_path: Path
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
        follow = [user("q"), reply, TOOL_RESULT]
        calls = [
            make_call("a", [user("q")], reply, 0, "run", tools=TOOLS),
            make_call("b", follow, assistant("2"), 1, "run", tools=TOOLS),
        ]

        # Act
        write_snapshot(tmp_path, stitch_sessions(calls), "user-9", {})
        rows = pq.read_table(tmp_path / "calls.parquet").to_pylist()

        # Assert
        prompt = json.loads(rows[1]["prompt"])
        assert prompt[1]["reasoning_content"] == "add"
        assert "thinking" not in prompt[1]
        assert prompt[1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
        assert validate_traces_file(tmp_path / "calls.parquet") == 2

    def test_rejects_a_label_score_no_row_has(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="label_score 'outcome'"):
            write_snapshot(
                tmp_path,
                stitch_sessions(agent_run()),
                "user-9",
                {},
                label_score="outcome",
            )

    def test_rejects_calls_of_more_than_one_organisation(self, tmp_path: Path) -> None:
        calls = [
            make_call("a", [user("q")], assistant("r"), 0, organization_id="org-1"),
            make_call("b", [user("x")], assistant("y"), 1, organization_id="org-2"),
        ]

        with pytest.raises(ValueError, match="one organisation's calls"):
            write_snapshot(tmp_path, stitch_sessions(calls), "user-9", {})


class TestFeedbackSchema:
    def test_feedback_records_round_trip_through_parquet(self, tmp_path: Path) -> None:
        # Arrange
        feedback = FeedbackRecord(
            feedback_id="f1",
            call_id="c1",
            session_key=None,
            score=0.5,
            label="wrong_tool",
            comment=None,
            source="user",
            user_id=None,
            created_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
        )
        path = tmp_path / "feedback.parquet"

        # Act
        pq.write_table(
            pa.Table.from_pylist(
                [dataclasses.asdict(feedback)], schema=FEEDBACK_SCHEMA
            ),
            path,
        )

        # Assert
        assert FeedbackRecord(**pq.read_table(path).to_pylist()[0]) == feedback


class TestSftRows:
    def test_one_row_per_segment_with_recorded_replies(self) -> None:
        # Arrange
        sessions = stitch_sessions(agent_run())

        # Act
        rows = sft_rows(sessions)

        # Assert
        assert [(row["session_id"], row["segment_index"]) for row in rows] == [
            ("run", 0),
            ("run", 1),
        ]
        assert json.loads(rows[0]["messages"]) == [
            user("1+1?"),
            CALL_REPLY,
            TOOL_RESULT,
            assistant("2"),
        ]
        assert json.loads(rows[0]["tools"]) == TOOLS
        assert json.loads(rows[1]["messages"]) == [
            user("summary: it is 2"),
            user("say it"),
            assistant("2."),
        ]
        assert rows[1]["tools"] is None

    def test_drops_trailing_calls_without_a_response(self) -> None:
        calls = [
            make_call("a", [user("q")], assistant("r"), 0, "run"),
            make_call("b", [user("q"), assistant("r"), user("n")], None, 1, "run"),
        ]

        rows = sft_rows(stitch_sessions(calls))

        assert [json.loads(row["messages"]) for row in rows] == [
            [user("q"), assistant("r")]
        ]

    def test_keeps_the_client_reply_after_a_failed_call(self) -> None:
        # Arrange
        first = [user("q"), assistant("r"), user("n")]
        calls = [
            make_call("a", first, None, 0, "run"),
            make_call(
                "b", [*first, assistant("s"), user("m")], assistant("t"), 1, "run"
            ),
        ]

        # Act
        rows = sft_rows(stitch_sessions(calls))

        # Assert
        assert [json.loads(row["messages"]) for row in rows] == [
            [*first, assistant("s"), user("m"), assistant("t")]
        ]

    def test_skips_a_segment_with_no_response(self) -> None:
        calls = [make_call("a", [user("q")], None, 0, "run")]

        assert sft_rows(stitch_sessions(calls)) == []

    def test_rejects_calls_captured_without_content(self) -> None:
        calls = [make_call("a", [user("q")], assistant("r"), content=False)]

        with pytest.raises(ValueError, match="captured at the 'full' level"):
            sft_rows(stitch_sessions(calls))

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
        (row,) = sft_rows(stitch_sessions(calls))

        # Act
        ids, mask = encode_sft_row(tokenizer, row, "response", None)

        # Assert
        trained = tokenizer.decode([i for i, m in zip(ids, mask, strict=True) if m])
        assert trained.count("hello") == 2
        assert trained.count("again") == 1
