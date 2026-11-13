# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from agilerl.llm_envs.call_record import (
    FINISH_REASONS,
    SCHEMA_VERSION,
    directory_digest,
    from_row,
    model_version,
    new_call_id,
    prompt_prefix_hashes,
    to_row,
)
from tests.test_llm_envs.call_factory import assistant, make_call, user


class TestToRow:
    def test_round_trips_through_from_row(self) -> None:
        # Arrange
        record = dataclasses.replace(
            make_call(
                "c1",
                [user("hi")],
                assistant("yo"),
                tools=[{"type": "function", "function": {"name": "f"}}],
            ),
            metadata={"team": "a"},
            sampling={"temperature": 0.7, "stop": ["\n"]},
            prompt_token_ids=[1, 2],
            completion_token_ids=[3],
            completion_logprobs=[-0.5],
        )

        # Act
        row = to_row(record)

        # Assert
        assert json.loads(row["messages"]) == [user("hi")]
        assert json.loads(row["sampling"]) == {"temperature": 0.7, "stop": ["\n"]}
        assert json.loads(row["metadata"]) == {"team": "a"}
        assert row["schema_version"] == SCHEMA_VERSION
        assert from_row(row) == record

    def test_leaves_absent_content_null(self) -> None:
        record = make_call("c1", [user("hi")], assistant("yo"), content=False)

        row = to_row(record)

        assert (row["messages"], row["response_message"], row["tools"]) == (
            None,
            None,
            None,
        )


class TestFromRow:
    def test_reads_json_metadata_and_ignores_extra_columns(self) -> None:
        row = {
            **to_row(make_call("c1", [user("hi")], assistant("yo"))),
            "metadata": '{"team": "a"}',
            "session_key": "s",
        }

        record = from_row(row)

        assert record.metadata == {"team": "a"}

    def test_keeps_null_organization_and_user(self) -> None:
        record = make_call(
            "c1", [user("hi")], assistant("yo"), user_id=None, organization_id=None
        )

        read = from_row(to_row(record))

        assert (read.organization_id, read.user_id) == (None, None)

    def test_rejects_an_unknown_finish_reason(self) -> None:
        row = {**to_row(make_call("c1", [user("hi")], None)), "finish_reason": "abort"}

        with pytest.raises(ValueError, match="call c1 has finish_reason 'abort'"):
            from_row(row)

    def test_rejects_another_schema_version(self) -> None:
        row = {**to_row(make_call("c1", [user("hi")], None)), "schema_version": 2}

        with pytest.raises(ValueError, match="schema_version 2"):
            from_row(row)


class TestFinishReasons:
    def test_lists_every_value_a_record_may_hold(self) -> None:
        assert FINISH_REASONS == ("stop", "length", "tool_calls", "error", "cancelled")


class TestNewCallId:
    def test_is_26_crockford_characters_sorting_by_time(self) -> None:
        earlier, later = new_call_id(1_000), new_call_id(2_000)

        assert len(earlier) == 26
        assert set(earlier) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
        assert earlier < later

    def test_ids_in_the_same_millisecond_differ(self) -> None:
        assert new_call_id(5) != new_call_id(5)


class TestPromptPrefixHashes:
    def test_entries_match_exactly_while_messages_match(self) -> None:
        first = prompt_prefix_hashes([user("a"), assistant("b"), user("c")])
        second = prompt_prefix_hashes([user("a"), assistant("b"), user("d")])

        assert len(first) == 3
        assert first[:2] == second[:2]
        assert first[2] != second[2]

    def test_ignores_what_harnesses_drop_when_resending(self) -> None:
        # Arrange
        sent = [
            user("a"),
            assistant("b", reasoning_content="think", tool_calls=[]),
        ]
        resent = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "assistant", "content": "b", "refusal": None},
        ]

        # Act / Assert
        assert prompt_prefix_hashes(sent) == prompt_prefix_hashes(resent)

    def test_start_continues_a_chain(self) -> None:
        messages = [user("a"), assistant("b")]

        whole = prompt_prefix_hashes(messages)
        continued = prompt_prefix_hashes(
            messages[1:], prompt_prefix_hashes(messages[:1])[-1]
        )

        assert continued == whole[1:]


class TestModelVersion:
    def test_changes_with_each_part(self) -> None:
        base = model_version("m", "a", "tmpl")

        assert base == model_version("m", "a", "tmpl")
        assert (
            len(
                {
                    base,
                    model_version("m2", "a", "tmpl"),
                    model_version("m", "b", "tmpl"),
                    model_version("m", None, "tmpl"),
                    model_version("m", "a", "tmpl2"),
                }
            )
            == 5
        )


class TestDirectoryDigest:
    def test_follows_file_content_and_names(self, tmp_path: Path) -> None:
        # Arrange
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "adapter_model.safetensors").write_bytes(b"w1")
        (tmp_path / "b" / "adapter_model.safetensors").write_bytes(b"w1")
        before = directory_digest(tmp_path / "a")

        # Act
        (tmp_path / "a" / "adapter_model.safetensors").write_bytes(b"w2")

        # Assert
        assert before == directory_digest(tmp_path / "b")
        assert directory_digest(tmp_path / "a") != before
