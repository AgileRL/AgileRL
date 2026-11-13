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

from agilerl.arena.cli import main
from agilerl.arena.client import JSONL_CONTENT_TYPE, ArenaClient
from agilerl.arena.exceptions import ArenaValidationError
from agilerl.arena.models import TrainingManifest
from agilerl.arena.models.traces import (
    AssistantMessage,
    TraceScore,
    TracesMessagesRow,
    TracesRow,
    TracesValidationError,
    row_form,
    validate_traces_file,
    validate_traces_rows,
)
from agilerl.arena.traces_format import SCORE_FIELDS, TRACE_COLUMNS

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
        "prompt": [
            {"role": "system", "content": "You are the IT helpdesk agent."},
            {"role": "user", "content": "I can't sign in, MFA keeps failing."},
            {"role": "assistant", "content": None, "tool_calls": [lookup_call()]},
            {"role": "tool", "tool_call_id": "c1", "content": '{"locked": true}'},
        ],
        "completion": [
            {
                "role": "assistant",
                "reasoning_content": "Account is locked.",
                "content": "Your account was locked. I've sent an unlock link.",
            }
        ],
        "tools": [LOOKUP_TOOL],
        "call_id": "01JD",
        "session_id": "s-42",
        "segment_index": 0,
        "model_version": "m-7f3a",
    }


class TestTracesRow:
    def test_accepts_the_spec_example(self) -> None:
        row = TracesRow.model_validate(helpdesk_row())

        assert row.completion_message.reasoning_text == "Account is locked."
        assert row.tools is not None
        assert row.tools[0].function.name == "lookup_user"

    def test_decodes_json_text_columns(self) -> None:
        raw = helpdesk_row()
        for column in ("prompt", "completion", "tools"):
            raw[column] = json.dumps(raw[column])

        row = TracesRow.model_validate(raw)

        assert len(row.prompt) == 4

    @pytest.mark.parametrize("key", ["reasoning", "thinking"])
    def test_reads_reasoning_under_any_reader_key(self, key: str) -> None:
        message = AssistantMessage.model_validate(
            {"role": "assistant", key: "why", "content": "ok"}
        )

        assert message.reasoning_text == "why"

    def test_ignores_unknown_columns_and_message_fields(self) -> None:
        raw = {**helpdesk_row(), "trace_tag": "x"}
        raw["prompt"][1]["audio"] = None

        row = TracesRow.model_validate(raw)

        assert len(row.prompt) == 4

    def test_rejects_two_completion_messages(self) -> None:
        raw = helpdesk_row()
        raw["completion"] = raw["completion"] * 2

        with pytest.raises(ValidationError, match="at most 1 item"):
            TracesRow.model_validate(raw)

    def test_rejects_a_prompt_ending_with_an_assistant_turn(self) -> None:
        raw = helpdesk_row()
        raw["prompt"].append({"role": "assistant", "content": "prefill"})

        with pytest.raises(ValidationError, match="prompt ends with an assistant"):
            TracesRow.model_validate(raw)

    def test_rejects_a_tool_result_no_call_asked_for(self) -> None:
        raw = helpdesk_row()
        raw["prompt"][3]["tool_call_id"] = "c9"

        with pytest.raises(ValidationError, match="'c9', which no earlier"):
            TracesRow.model_validate(raw)

    def test_rejects_a_call_to_an_undeclared_tool(self) -> None:
        raw = helpdesk_row()
        raw["completion"] = [
            {
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
        ]

        with pytest.raises(ValidationError, match="'reset_mfa', which is not in tools"):
            TracesRow.model_validate(raw)

    def test_rejects_object_arguments(self) -> None:
        raw = helpdesk_row()
        raw["prompt"][2]["tool_calls"][0]["function"]["arguments"] = {"email": "x"}

        with pytest.raises(ValidationError, match="string"):
            TracesRow.model_validate(raw)

    def test_rejects_arguments_that_are_not_a_json_object(self) -> None:
        raw = helpdesk_row()
        raw["prompt"][2]["tool_calls"][0]["function"]["arguments"] = "[1]"

        with pytest.raises(ValidationError, match="must encode a JSON object"):
            TracesRow.model_validate(raw)

    def test_rejects_arguments_that_are_not_json(self) -> None:
        raw = helpdesk_row()
        raw["prompt"][2]["tool_calls"][0]["function"]["arguments"] = "{oops"

        with pytest.raises(ValidationError, match="arguments is not valid JSON"):
            TracesRow.model_validate(raw)

    def test_rejects_a_json_text_column_that_is_not_json(self) -> None:
        raw = {**helpdesk_row(), "prompt": "[{oops"}

        with pytest.raises(
            ValidationError, match="prompt is a string but not valid JSON"
        ):
            TracesRow.model_validate(raw)

    def test_rejects_a_row_that_is_not_an_object(self) -> None:
        with pytest.raises(ValidationError):
            TracesRow.model_validate(["prompt", "completion"])

    def test_rejects_an_empty_assistant_message(self) -> None:
        raw = helpdesk_row()
        raw["completion"] = [{"role": "assistant", "content": ""}]

        with pytest.raises(ValidationError, match="needs content or tool_calls"):
            TracesRow.model_validate(raw)

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
        assert set(TRACE_COLUMNS) <= set(TracesRow.model_fields)

    def test_score_entries_take_every_shared_field(self) -> None:
        assert set(SCORE_FIELDS) == set(TraceScore.model_fields)


class TestTracesMessagesRow:
    def test_accepts_a_conversation(self) -> None:
        raw = helpdesk_row()
        messages = [*raw["prompt"], *raw["completion"]]

        row = TracesMessagesRow.model_validate(
            {"messages": messages, "tools": raw["tools"]}
        )

        assert len(row.messages) == 5

    def test_accepts_a_conversation_without_tools(self) -> None:
        row = TracesMessagesRow.model_validate(
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
            TracesMessagesRow.model_validate(
                {
                    "messages": [
                        {"role": "assistant", "content": "a"},
                        {"role": "user", "content": "u"},
                    ]
                }
            )

    def test_rejects_a_conversation_without_an_assistant_turn(self) -> None:
        with pytest.raises(ValidationError, match="no assistant turn"):
            TracesMessagesRow.model_validate(
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "user", "content": "u"},
                    ]
                }
            )


class TestRowForm:
    def test_prompt_completion_columns_select_the_call_row(self) -> None:
        assert row_form(["prompt", "completion", "tools"]) is TracesRow

    def test_messages_column_selects_the_conversation_row(self) -> None:
        assert row_form(["messages", "tools"]) is TracesMessagesRow

    def test_rejects_other_columns(self) -> None:
        with pytest.raises(ValueError, match="needs 'prompt' and 'completion'"):
            row_form(["question", "answer"])


class TestValidateTracesRows:
    def test_counts_valid_rows(self) -> None:
        assert validate_traces_rows([helpdesk_row(), helpdesk_row()]) == 2

    def test_numbers_bad_rows_from_one(self) -> None:
        bad = helpdesk_row()
        bad["completion"] = [{"role": "user", "content": "x"}]

        with pytest.raises(TracesValidationError) as raised:
            validate_traces_rows([helpdesk_row(), bad, bad], source="t.jsonl")

        assert raised.value.total == 2
        assert raised.value.errors[0].startswith("row 2: completion.0")
        assert "t.jsonl is not a valid traces dataset (2 bad rows)" in str(raised.value)

    def test_rejects_no_rows(self) -> None:
        with pytest.raises(TracesValidationError, match="the file has no rows"):
            validate_traces_rows([])


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
            for column in ("prompt", "completion", "tools"):
                row[column] = json.dumps(row[column])
            rows.append(
                {
                    **row,
                    "call_id": call_id,
                    "scores": scores,
                    "finish_reason": "stop",
                    "answer": None,
                }
            )
        path = tmp_path / "calls.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)

        # Act
        count = validate_traces_file(path)

        # Assert
        assert count == 3
        assert pa.types.is_string(pq.read_schema(path).field("prompt").type)

    def test_accepts_duplicate_prompts(self) -> None:
        first, second = helpdesk_row(), helpdesk_row()
        second["call_id"] = "01JE"
        second["completion"] = [{"role": "assistant", "content": "Another reply."}]

        assert validate_traces_rows([first, second]) == 2

    def test_rejects_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "traces.csv"
        path.write_text("prompt,completion\n", encoding="utf-8")

        with pytest.raises(ValueError, match="JSONL or parquet"):
            validate_traces_file(path)


@pytest.fixture
def api_key_client() -> ArenaClient:
    with patch("agilerl.arena.auth.KeycloakOpenID"):
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

    def test_rejects_an_invalid_file_before_upload(
        self, api_key_client: ArenaClient, tmp_path: Path
    ) -> None:
        bad = helpdesk_row()
        del bad["completion"]
        path = write_jsonl(tmp_path / "traces.jsonl", [bad])
        api_key_client._request = MagicMock()

        with pytest.raises(
            ArenaValidationError, match="needs 'prompt' and 'completion'"
        ):
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
        with patch("agilerl.arena.cli_datasets.arena_client", return_value=client):
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
        [("columns", {"prompt": "input"}), ("prompt_template", {"user_0": "{q}"})],
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
