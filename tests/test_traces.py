# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The traces dataset spec: row models, the local file validator, the CLI upload and the manifest contract."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from agilerl.cli import main
from agilerl.client import JSONL_CONTENT_TYPE, ArenaClient
from agilerl.exceptions import ArenaValidationError
from agilerl.models import TrainingManifest
from agilerl.models.traces import (
    AssistantMessage,
    TraceScore,
    TracesSegmentRow,
    TracesValidationError,
    validate_traces_file,
    validate_traces_rows,
)
from agilerl.traces_format import SCORE_FIELDS, TRACE_COLUMNS

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_user",
        "description": "Find a user",
        "parameters": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
}


def lookup_call(call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup_user", "arguments": '{"email": "a@b.com"}'},
    }


def helpdesk_row() -> dict[str, Any]:
    """The spec's example: a tool call and its result, then the reference reply."""
    return {
        "messages": [
            {"role": "system", "content": "You are the IT helpdesk agent."},
            {"role": "user", "content": "I can't sign in, MFA keeps failing."},
            {"role": "assistant", "content": None, "tool_calls": [lookup_call()]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"locked": true}'},
            {
                "role": "assistant",
                "reasoning_content": "Account is locked.",
                "content": "Your account was locked. I've sent an unlock link.",
            },
        ],
        "tools": [LOOKUP_TOOL],
        "session_id": "s-42",
        "segment_index": 0,
        "model_version": "m-7f3a",
    }


def helpdesk_segment() -> dict[str, Any]:
    """The spec example as one segment: the tool-call turn and the reply turn, with reasoning."""
    row = helpdesk_row()
    return {
        "messages": row["messages"],
        "tools": row["tools"],
        "call_ids": ["01JC", "01JD"],
        "finish_reasons": ["tool_calls", "stop"],
        "scores": [
            {"name": "feedback", "value": 1.0, "source": "user"},
            {"name": "judge", "value": 0.5, "source": "j", "turn": 1},
        ],
        "session_id": "s-42",
        "segment_index": 0,
    }


class TestTracesSegmentRowMessages:
    def test_accepts_the_spec_example(self) -> None:
        row = TracesSegmentRow.model_validate(helpdesk_row())

        assert isinstance(row.messages[-1], AssistantMessage)
        assert row.messages[-1].reasoning_text == "Account is locked."
        assert row.tools is not None
        assert row.tools[0].function.name == "lookup_user"

    def test_decodes_json_text_columns(self) -> None:
        raw = helpdesk_row()
        for column in ("messages", "tools"):
            raw[column] = json.dumps(raw[column])

        row = TracesSegmentRow.model_validate(raw)

        assert len(row.messages) == 5

    @pytest.mark.parametrize("key", ["reasoning", "thinking"])
    def test_reads_reasoning_under_any_reader_key(self, key: str) -> None:
        message = AssistantMessage.model_validate(
            {"role": "assistant", key: "why", "content": "ok"}
        )

        assert message.reasoning_text == "why"

    def test_ignores_unknown_columns_and_message_fields(self) -> None:
        raw = {**helpdesk_row(), "trace_tag": "x"}
        raw["messages"][1]["audio"] = None

        row = TracesSegmentRow.model_validate(raw)

        assert len(row.messages) == 5

    def test_rejects_back_to_back_target_assistant_messages(self) -> None:
        raw = helpdesk_row()
        raw["messages"].append(raw["messages"][-1])

        with pytest.raises(ValidationError, match="back-to-back assistant"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_a_tool_result_no_call_asked_for(self) -> None:
        raw = helpdesk_row()
        raw["messages"][3]["tool_call_id"] = "c9"

        with pytest.raises(ValidationError, match="'c9', which no earlier"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_a_call_to_an_undeclared_tool(self) -> None:
        raw = helpdesk_row()
        raw["messages"][4] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "reset_mfa", "arguments": "{}"},
                }
            ],
        }

        with pytest.raises(ValidationError, match="'reset_mfa', which is not in tools"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_object_arguments(self) -> None:
        raw = helpdesk_row()
        raw["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"email": "x"}

        with pytest.raises(ValidationError, match="string"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_arguments_that_are_not_a_json_object(self) -> None:
        raw = helpdesk_row()
        raw["messages"][2]["tool_calls"][0]["function"]["arguments"] = "[1]"

        with pytest.raises(ValidationError, match="must encode a JSON object"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_arguments_that_are_not_json(self) -> None:
        raw = helpdesk_row()
        raw["messages"][2]["tool_calls"][0]["function"]["arguments"] = "{oops"

        with pytest.raises(ValidationError, match="arguments is not valid JSON"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_a_json_text_column_that_is_not_json(self) -> None:
        with pytest.raises(
            ValidationError, match="messages is a string but not valid JSON"
        ):
            TracesSegmentRow.model_validate({"messages": "[{oops"})

    def test_rejects_a_row_that_is_not_an_object(self) -> None:
        with pytest.raises(ValidationError):
            TracesSegmentRow.model_validate(["messages"])

    def test_accepts_text_parts_in_a_user_message(self) -> None:
        raw = helpdesk_row()
        raw["messages"][1]["content"] = [{"type": "text", "text": "I can't sign in."}]

        row = TracesSegmentRow.model_validate(raw)

        assert isinstance(row.messages[1].content, list)

    @pytest.mark.parametrize("index", [4, 3], ids=["assistant", "tool"])
    def test_rejects_content_parts_outside_system_and_user_messages(
        self, index: int
    ) -> None:
        raw = helpdesk_row()
        raw["messages"][index]["content"] = [{"type": "text", "text": "x"}]

        with pytest.raises(ValidationError, match="Input should be a valid string"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_an_empty_assistant_message(self) -> None:
        raw = helpdesk_row()
        raw["messages"][4] = {"role": "assistant", "content": ""}

        with pytest.raises(ValidationError, match="needs content or tool_calls"):
            TracesSegmentRow.model_validate(raw)

    def test_ignores_an_empty_reasoning_key_beside_another(self) -> None:
        message = AssistantMessage.model_validate(
            {
                "role": "assistant",
                "content": "ok",
                "reasoning_content": "a",
                "reasoning": "",
            }
        )

        assert message.reasoning_text == "a"

    def test_rejects_conflicting_reasoning_keys(self) -> None:
        with pytest.raises(ValidationError, match="different reasoning"):
            AssistantMessage.model_validate(
                {
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": "a",
                    "thinking": "b",
                }
            )


class TestTracesRowColumns:
    def test_covers_every_shared_trace_column(self) -> None:
        assert set(TRACE_COLUMNS) == set(TracesSegmentRow.model_fields)

    def test_score_entries_take_every_shared_field(self) -> None:
        assert set(SCORE_FIELDS) == set(TraceScore.model_fields)


class TestTracesSegmentRow:
    def test_accepts_a_conversation(self) -> None:
        raw = helpdesk_row()

        row = TracesSegmentRow.model_validate(
            {"messages": raw["messages"], "tools": raw["tools"]}
        )

        assert len(row.messages) == 5

    def test_accepts_a_conversation_without_tools(self) -> None:
        row = TracesSegmentRow.model_validate(
            {
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        )

        assert row.tools is None

    def test_rejects_a_conversation_starting_with_an_assistant_turn(self) -> None:
        with pytest.raises(ValidationError, match="starts with an assistant turn"):
            TracesSegmentRow.model_validate(
                {
                    "messages": [
                        {"role": "assistant", "content": "a"},
                        {"role": "user", "content": "u"},
                    ]
                }
            )

    def test_rejects_a_conversation_without_an_assistant_turn(self) -> None:
        with pytest.raises(ValidationError, match="no assistant turn"):
            TracesSegmentRow.model_validate(
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                    ]
                }
            )


class TestTracesSegmentRowTurns:
    def test_accepts_tool_calls_reasoning_and_per_turn_columns(self) -> None:
        row = TracesSegmentRow.model_validate(helpdesk_segment())

        assert row.call_ids == ["01JC", "01JD"]
        assert row.scores is not None
        assert row.scores[1].turn == 1
        assert isinstance(row.messages[-1], AssistantMessage)
        assert row.messages[-1].reasoning_text == "Account is locked."

    def test_decodes_json_text_messages(self) -> None:
        raw = helpdesk_segment()
        raw["messages"] = json.dumps(raw["messages"])

        row = TracesSegmentRow.model_validate(raw)

        assert len(row.messages) == 5

    def test_decodes_every_nested_column_from_json_text(self) -> None:
        raw = {
            **helpdesk_segment(),
            "call_ids": [None, "01JD"],
            "completion_token_ids": [None, [5]],
            "completion_logprobs": [None, [-0.5]],
            "metadata": {"tenant": "t1"},
        }
        text = {
            key: json.dumps(value) if isinstance(value, (list, dict)) else value
            for key, value in raw.items()
        }

        row = TracesSegmentRow.model_validate(text)

        assert row == TracesSegmentRow.model_validate(raw)

    def test_accepts_unanswered_calls_on_the_final_turn_and_trailing_messages(
        self,
    ) -> None:
        raw = helpdesk_segment()
        raw["messages"][-1] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [lookup_call("c2")],
        }
        raw["messages"].append({"role": "user", "content": "Thanks."})

        row = TracesSegmentRow.model_validate(raw)

        assert len(row.messages) == 6

    @pytest.mark.parametrize(
        "result",
        [
            {"role": "tool", "tool_call_id": "c2", "content": None},
            {"role": "tool", "tool_call_id": "c2"},
        ],
        ids=["null", "absent"],
    )
    def test_accepts_a_trailing_tool_result_without_content(
        self, result: dict[str, Any]
    ) -> None:
        # Arrange
        raw = helpdesk_segment()
        raw["messages"][-1] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [lookup_call("c2")],
        }
        raw["messages"].append(result)

        # Act
        row = TracesSegmentRow.model_validate(raw)

        # Assert
        assert row.messages[-1].content is None

    def test_rejects_a_tool_result_without_content_before_a_turn(self) -> None:
        raw = helpdesk_segment()
        raw["messages"][3]["content"] = None

        with pytest.raises(ValidationError, match=r"messages\[3\] is a tool result"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_a_call_unanswered_before_the_next_turn(self) -> None:
        raw = helpdesk_segment()
        raw["messages"][3] = {"role": "user", "content": "Any news?"}

        with pytest.raises(
            ValidationError, match=r"messages\[2\] calls \['c1'\], which no tool"
        ):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_back_to_back_assistant_messages(self) -> None:
        raw = helpdesk_segment()
        raw["messages"].insert(4, {"role": "assistant", "content": "One moment."})
        raw["messages"][3], raw["messages"][4] = raw["messages"][4], raw["messages"][3]
        raw["call_ids"] = None
        raw["finish_reasons"] = None
        raw["scores"] = None

        with pytest.raises(ValidationError, match="back-to-back assistant messages"):
            TracesSegmentRow.model_validate(raw)

    def test_accepts_a_context_assistant_right_after_another(self) -> None:
        raw = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "r"},
                {
                    "role": "assistant",
                    "content": "inserted by the harness",
                    "weight": 0,
                },
                {"role": "user", "content": "n"},
                {"role": "assistant", "content": "s"},
            ],
        }

        row = TracesSegmentRow.model_validate(raw)

        assert [
            message.is_target
            for message in row.messages
            if isinstance(message, AssistantMessage)
        ] == [True, False, True]

    def test_rejects_a_target_right_after_another_assistant(self) -> None:
        raw = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "r", "weight": 0},
                {"role": "assistant", "content": "s"},
            ],
        }

        with pytest.raises(ValidationError, match="Mark it context"):
            TracesSegmentRow.model_validate(raw)

    def test_accepts_completion_token_ids_and_logprobs(self) -> None:
        raw = {
            **helpdesk_segment(),
            "completion_token_ids": [None, [5, 6, 7]],
            "completion_logprobs": [None, [-0.1, -0.2, -0.3]],
        }

        row = TracesSegmentRow.model_validate(raw)

        assert row.completion_token_ids == [None, [5, 6, 7]]

    def test_rejects_tokens_on_a_context_turn(self) -> None:
        raw = {**helpdesk_segment(), "completion_token_ids": [[1], None]}
        raw["messages"][2]["weight"] = 0

        with pytest.raises(ValidationError, match="turn 0 is context"):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_token_ids_and_logprobs_of_different_lengths(self) -> None:
        raw = {
            **helpdesk_segment(),
            "completion_token_ids": [[1], [5, 6, 7]],
            "completion_logprobs": [[-0.1], [-0.2]],
        }

        with pytest.raises(ValidationError, match="turn 1 has 3 completion_token_ids"):
            TracesSegmentRow.model_validate(raw)

    @pytest.mark.parametrize(
        "column",
        ["call_ids", "finish_reasons", "completion_token_ids", "completion_logprobs"],
    )
    def test_rejects_a_per_turn_list_of_the_wrong_length(self, column: str) -> None:
        raw = {**helpdesk_segment(), column: [None]}

        with pytest.raises(
            ValidationError, match=f"{column} has 1 entries; messages has 2"
        ):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_a_segment_with_no_target_turn(self) -> None:
        raw = helpdesk_segment()
        for message in raw["messages"]:
            if message["role"] == "assistant":
                message["weight"] = 0

        with pytest.raises(
            ValidationError, match="every assistant message has weight 0"
        ):
            TracesSegmentRow.model_validate(raw)

    def test_null_call_ids_do_not_mark_context(self) -> None:
        row = TracesSegmentRow.model_validate(
            {**helpdesk_segment(), "call_ids": [None, None]}
        )

        assert row.call_ids == [None, None]
        assert all(
            message.is_target
            for message in row.messages
            if isinstance(message, AssistantMessage)
        )

    @pytest.mark.parametrize("weight", [2, -1, 0.5, 1.0, True, "1"])
    def test_rejects_a_weight_other_than_zero_or_one(self, weight: object) -> None:
        raw = helpdesk_segment()
        raw["messages"][2]["weight"] = weight

        with pytest.raises(ValidationError, match="weight must be 0 or 1"):
            TracesSegmentRow.model_validate(raw)

    @pytest.mark.parametrize("at", [0, 1, 3], ids=["system", "user", "tool"])
    def test_rejects_weight_on_a_non_assistant_message(self, at: int) -> None:
        raw = helpdesk_segment()
        raw["messages"][at]["weight"] = 1

        with pytest.raises(
            ValidationError, match="only assistant messages are weighted"
        ):
            TracesSegmentRow.model_validate(raw)

    def test_rejects_a_score_on_a_turn_past_the_last(self) -> None:
        raw = helpdesk_segment()
        raw["scores"] = [{"name": "judge", "value": 1.0, "turn": 2}]

        with pytest.raises(ValidationError, match="'judge' is on turn 2"):
            TracesSegmentRow.model_validate(raw)


class TestValidateTracesRows:
    def test_counts_valid_rows(self) -> None:
        assert validate_traces_rows([helpdesk_row(), helpdesk_row()]) == 2

    def test_numbers_bad_rows_from_one(self) -> None:
        bad = helpdesk_row()
        bad["messages"] = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "x"},
        ]

        with pytest.raises(TracesValidationError) as raised:
            validate_traces_rows([helpdesk_row(), bad, bad], source="t.jsonl")

        assert raised.value.total == 2
        assert raised.value.errors[0].startswith("row 2: ")
        assert "no assistant turn" in raised.value.errors[0]
        assert "t.jsonl is not a valid traces dataset (2 bad rows)" in str(raised.value)

    def test_rejects_no_rows(self) -> None:
        with pytest.raises(TracesValidationError, match="the file has no rows"):
            validate_traces_rows([])

    def test_counts_segment_rows(self) -> None:
        assert validate_traces_rows([helpdesk_segment(), helpdesk_segment()]) == 2

    def test_names_the_field_of_a_bad_segment(self) -> None:
        bad = helpdesk_segment()
        bad["messages"][3]["tool_call_id"] = "c9"

        with pytest.raises(TracesValidationError) as raised:
            validate_traces_rows([helpdesk_segment(), bad])

        assert raised.value.errors[0].startswith("row 2: row: Value error, ")
        assert "'c9'" in raised.value.errors[0]

    def test_names_a_later_row_without_messages(self) -> None:
        with pytest.raises(
            TracesValidationError, match="row 2: messages: Field required"
        ):
            validate_traces_rows(
                [helpdesk_segment(), {"prompt": "q", "completion": "a"}]
            )


class TestValidateTracesFile:
    def test_validates_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text(
            "\n".join(json.dumps(helpdesk_row()) for _ in range(3)) + "\n",
            encoding="utf-8",
        )

        assert validate_traces_file(path) == 3

    def test_names_a_line_that_is_not_json(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text(json.dumps(helpdesk_row()) + "\n{oops\n", encoding="utf-8")

        with pytest.raises(TracesValidationError, match="row 2: not valid JSON"):
            validate_traces_file(path)

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text(f"{json.dumps(helpdesk_row())}\n\n", encoding="utf-8")

        assert validate_traces_file(path) == 1

    def test_names_a_line_that_is_not_an_object(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.jsonl"
        path.write_text("[1, 2]\n", encoding="utf-8")

        with pytest.raises(TracesValidationError, match="row 1: not a JSON object"):
            validate_traces_file(path)

    def test_parquet_needs_pyarrow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        path = tmp_path / "traces.parquet"
        path.write_bytes(b"")
        monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)

        # Act / Assert
        with pytest.raises(ImportError, match="needs pyarrow"):
            validate_traces_file(path)

    def test_validates_parquet_with_arrow_null_fields(self, tmp_path: Path) -> None:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        path = tmp_path / "traces.parquet"
        pq.write_table(pa.Table.from_pylist([helpdesk_row(), helpdesk_row()]), path)

        assert validate_traces_file(path) == 2

    def test_validates_snapshot_parquet_with_json_text_and_arrow_scores(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        rows = []
        for call_id, scores in (
            (
                "a",
                [
                    {
                        "name": "feedback",
                        "value": 1.0,
                        "source": "user",
                        "label": "up",
                        "comment": None,
                    }
                ],
            ),
            (
                "b",
                [
                    {
                        "name": "feedback",
                        "value": None,
                        "source": "user",
                        "label": None,
                        "comment": "meh",
                    }
                ],
            ),
            ("c", None),
        ):
            row = helpdesk_row()
            for column in ("messages", "tools"):
                row[column] = json.dumps(row[column])
            rows.append(
                {
                    **row,
                    "call_ids": [None, call_id],
                    "scores": scores,
                    "finish_reasons": ["tool_calls", "stop"],
                    "answer": None,
                }
            )
        path = tmp_path / "calls.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)

        # Act
        count = validate_traces_file(path)

        # Assert
        assert count == 3
        assert pa.types.is_string(pq.read_schema(path).field("messages").type)

    def test_validates_segment_parquet_with_json_text_messages(
        self, tmp_path: Path
    ) -> None:
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        row = helpdesk_segment()
        row["messages"] = json.dumps(row["messages"])
        row["tools"] = json.dumps(row["tools"])
        path = tmp_path / "segments.parquet"
        pq.write_table(pa.Table.from_pylist([row, row]), path)

        assert validate_traces_file(path) == 2

    def test_accepts_duplicate_prompts(self) -> None:
        first, second = helpdesk_row(), helpdesk_row()
        second["messages"][4] = {"role": "assistant", "content": "Another reply."}

        assert validate_traces_rows([first, second]) == 2

    def test_rejects_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.csv"
        path.write_text("prompt,completion\n", encoding="utf-8")

        with pytest.raises(ValueError, match="JSONL or parquet"):
            validate_traces_file(path)


@pytest.fixture
def api_key_client() -> ArenaClient:
    with patch("agilerl.auth.KeycloakOpenID"):
        return ArenaClient(api_key="test-key")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class TestArenaClientCreateDatasetTraces:
    def test_uploads_a_valid_file_with_an_empty_mapping(
        self, api_key_client: ArenaClient, tmp_path: Path
    ) -> None:
        # Arrange
        path = write_jsonl(tmp_path / "traces.jsonl", [helpdesk_row()])
        api_key_client._request = MagicMock(return_value={"name": "t"})

        # Act
        api_key_client.create_dataset(name="t", category="traces", file=path)

        # Assert
        files = {
            name: part
            for name, part in api_key_client._request.call_args[1]["files"]
            if name != "file"
        }
        uploads = [
            part
            for name, part in api_key_client._request.call_args[1]["files"]
            if name == "file"
        ]
        assert files["category"][1] == "traces"
        assert json.loads(files["column_mapping"][1]) == {}
        assert uploads[0][0] == "traces.jsonl"
        assert uploads[0][2] == JSONL_CONTENT_TYPE

    def test_uploads_a_json_file_as_jsonl(
        self, api_key_client: ArenaClient, tmp_path: Path
    ) -> None:
        path = write_jsonl(tmp_path / "traces.json", [helpdesk_row()])
        api_key_client._request = MagicMock(return_value={"name": "t"})

        api_key_client.create_dataset(name="t", category="traces", file=path)

        uploads = [
            part
            for name, part in api_key_client._request.call_args[1]["files"]
            if name == "file"
        ]
        assert uploads[0][2] == JSONL_CONTENT_TYPE

    def test_accepts_an_empty_mapping_file(
        self, api_key_client: ArenaClient, tmp_path: Path
    ) -> None:
        path = write_jsonl(tmp_path / "traces.jsonl", [helpdesk_row()])
        api_key_client._request = MagicMock(return_value={"name": "t"})

        api_key_client.create_dataset(
            name="t", category="traces", column_mapping="{}\n", file=path
        )

        files = dict(api_key_client._request.call_args[1]["files"])
        assert json.loads(files["column_mapping"][1]) == {}

    def test_uploads_import_layouts_unchecked(
        self, api_key_client: ArenaClient, tmp_path: Path
    ) -> None:
        # Arrange
        rows = [
            {"prompt": "Hi there.", "completion": "Hello!"},
            {
                "conversations": [
                    {"from": "human", "value": "Hi there."},
                    {"from": "gpt", "value": "Hello!"},
                ]
            },
        ]
        path = write_jsonl(tmp_path / "traces.jsonl", rows)
        api_key_client._request = MagicMock(return_value={"name": "t"})

        # Act
        api_key_client.create_dataset(name="t", category="traces", file=path)

        # Assert
        uploads = [
            part
            for name, part in api_key_client._request.call_args[1]["files"]
            if name == "file"
        ]
        assert uploads[0][0] == "traces.jsonl"

    def test_rejects_a_file_that_is_not_jsonl_or_parquet(
        self, api_key_client: ArenaClient, tmp_path: Path
    ) -> None:
        path = tmp_path / "traces.csv"
        path.write_text("prompt,completion\n", encoding="utf-8")
        api_key_client._request = MagicMock()

        with pytest.raises(ArenaValidationError, match="JSONL or parquet"):
            api_key_client.create_dataset(name="t", category="traces", file=path)

        api_key_client._request.assert_not_called()

    def test_rejects_a_column_mapping(
        self, api_key_client: ArenaClient, tmp_path: Path
    ) -> None:
        path = write_jsonl(tmp_path / "traces.jsonl", [helpdesk_row()])

        with pytest.raises(ArenaValidationError, match="takes no column mapping"):
            api_key_client.create_dataset(
                name="t",
                category="traces",
                column_mapping={"prompt": "input"},
                file=path,
            )

    def test_rejects_bytes(self, api_key_client: ArenaClient) -> None:
        with pytest.raises(ArenaValidationError, match="not bytes"):
            api_key_client.create_dataset(name="t", category="traces", file=b"{}")

    def test_sends_a_hub_dataset_without_a_file(
        self, api_key_client: ArenaClient
    ) -> None:
        api_key_client._request = MagicMock(return_value={"name": "t"})

        api_key_client.create_dataset(
            name="t", category="traces", hf_dataset_name="org/traces"
        )

        files = dict(api_key_client._request.call_args[1]["files"])
        assert files["category"][1] == "traces"

    def test_other_categories_still_need_a_mapping(
        self, api_key_client: ArenaClient
    ) -> None:
        with pytest.raises(ArenaValidationError, match="needs a column mapping"):
            api_key_client.create_dataset(name="t", category="sft")


class TestDatasetsCreateCommandTraces:
    def test_creates_without_a_column_mapping(self, tmp_path: Path) -> None:
        # Arrange
        path = write_jsonl(tmp_path / "traces.jsonl", [helpdesk_row()])
        client = MagicMock()
        client.__enter__.return_value = client
        client.create_dataset.return_value = {"name": "t"}

        # Act
        with patch("agilerl.cli_datasets.arena_client", return_value=client):
            result = CliRunner().invoke(
                main,
                [
                    "datasets",
                    "create",
                    "t",
                    "--category",
                    "traces",
                    "--file",
                    str(path),
                ],
            )

        # Assert
        assert result.exit_code == 0, result.output
        assert client.create_dataset.call_args.kwargs["category"] == "traces"
        assert client.create_dataset.call_args.kwargs["column_mapping"] is None


OPD_TRACES = {
    "algorithm": {"name": "OPD", "batch_size": 2},
    "environment": {
        "env_type": "rollout",
        "dataset_format": "traces",
        "dataset": "traces.jsonl",
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_context_length": 512,
    },
    "training": {"max_steps": 100},
}

SFT_TRACES = {
    "algorithm": {"name": "SFT", "batch_size": 2},
    "environment": {
        "env_type": "dataset",
        "objective": "sft",
        "dataset_format": "traces",
        "dataset": "traces.jsonl",
    },
    "network": {
        "pretrained_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_context_length": 512,
    },
    "training": {"max_steps": 100},
}


def manifest(base: dict[str, Any], **environment: Any) -> dict[str, Any]:
    doc = copy.deepcopy(base)
    doc["environment"].update(environment)
    return doc


class TestTracesManifest:
    def test_opd_needs_no_rubric_or_prompt_template(self) -> None:
        env = TrainingManifest.model_validate(OPD_TRACES).environment

        assert env.dataset_format == "traces"
        assert env.rubric_file_path is None
        assert env.prompt_template is None
        assert env.max_turns == 1

    def test_opd_keeps_a_rubric(self) -> None:
        env = TrainingManifest.model_validate(
            manifest(OPD_TRACES, rubric_file_path="reward.py")
        ).environment

        assert env.rubric_name == "reward_fn"

    def test_sft_trains_the_completion_column(self) -> None:
        env = TrainingManifest.model_validate(SFT_TRACES).environment

        assert env.response_column == "completion"

    def test_grpo_without_a_rubric_is_rejected(self) -> None:
        doc = copy.deepcopy(OPD_TRACES)
        doc["algorithm"] = {"name": "GRPO", "batch_size": 2}

        with pytest.raises(ValidationError, match="only OPD"):
            TrainingManifest.model_validate(doc)

    def test_grpo_with_a_rubric_is_accepted(self) -> None:
        doc = manifest(OPD_TRACES, rubric_file_path="reward.py")
        doc["algorithm"] = {"name": "GRPO", "batch_size": 2}

        env = TrainingManifest.model_validate(doc).environment

        assert env.rubric_file_path == "reward.py"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("columns", {"prompt": "input"}),
            ("prompt_template", {"user_0": "{question}"}),
        ],
    )
    def test_rejects_a_column_mapping(self, field: str, value: Any) -> None:
        with pytest.raises(ValidationError, match=f"drop {field}"):
            TrainingManifest.model_validate(manifest(OPD_TRACES, **{field: value}))

    def test_rejects_multi_turn(self) -> None:
        with pytest.raises(ValidationError, match="max_turns must be 1"):
            TrainingManifest.model_validate(manifest(OPD_TRACES, max_turns=3))

    def test_rejects_preference_objective(self) -> None:
        doc = manifest(SFT_TRACES, objective="preference")
        doc["algorithm"] = {"name": "DPO", "batch_size": 2}

        with pytest.raises(ValidationError, match="objective 'preference'"):
            TrainingManifest.model_validate(doc)

    def test_rejects_an_objective_on_a_rollout(self) -> None:
        with pytest.raises(ValidationError, match="objective 'preference'"):
            TrainingManifest.model_validate(
                manifest(OPD_TRACES, objective="preference")
            )

    def test_rejects_another_response_column(self) -> None:
        with pytest.raises(ValidationError, match="drop response_column"):
            TrainingManifest.model_validate(
                manifest(SFT_TRACES, response_column="answer")
            )

    def test_rejects_a_format_without_a_dataset(self) -> None:
        doc = manifest(OPD_TRACES, dataset=None, entrypoint="pkg.mod:Env")

        with pytest.raises(ValidationError, match="name the dataset too"):
            TrainingManifest.model_validate(doc)

    def test_non_traces_rollout_still_needs_a_prompt_template(self) -> None:
        doc = manifest(OPD_TRACES, dataset_format=None)

        with pytest.raises(ValidationError, match="Prompt template is required"):
            TrainingManifest.model_validate(doc)
