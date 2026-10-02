# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl/training/llm/completion_logging.py."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from unittest.mock import patch

import pytest
import torch

from agilerl.training.llm.completion_logging import (
    CompletionLogger,
    CompletionLoggingConfig,
    CompletionRecord,
    CompletionWriter,
    ConsoleCompletionWriter,
    JsonlCompletionWriter,
    WandbCompletionWriter,
    build_completion_record,
    truncate_middle,
)


class LetterTokenizer:
    """Token ``i`` decodes to the ``i``-th lowercase letter; ``0`` is special."""

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(
            chr(ord("a") + i - 1) if i else ("" if skip_special_tokens else "<s>")
            for i in ids
        )


class ListWriter(CompletionWriter):
    def __init__(self) -> None:
        self.writes: list[list[CompletionRecord]] = []
        self.closed = False

    def write(self, records: Sequence[CompletionRecord]) -> None:
        self.writes.append(list(records))

    def close(self) -> None:
        self.closed = True


def _record(**overrides: object) -> CompletionRecord:
    values: dict = {
        "step": 3,
        "agent_index": 0,
        "group_index": 1,
        "sample_index": 2,
        "reward": 0.5,
        "num_turns": 1,
        "completion_tokens": 2,
        "prompt": "ab",
        "completion": "cd",
    }
    values.update(overrides)
    return CompletionRecord(**values)


def _batch(num_rows: int) -> tuple[list, list, list]:
    """``num_rows`` single-turn rows: prompt ``ab``, completion ``cd``, reward = row index."""
    token_ids = [torch.tensor([[1, 2, 3, 4]]) for _ in range(num_rows)]
    action_masks = [torch.tensor([[False, True, True]]) for _ in range(num_rows)]
    rewards = [torch.tensor([float(row)]) for row in range(num_rows)]
    return token_ids, action_masks, rewards


class TestCompletionLoggingConfig:
    def test_defaults_are_valid(self):
        config = CompletionLoggingConfig()

        assert config.interval == 10
        assert config.jsonl_path is None

    @pytest.mark.parametrize(
        "field", ["interval", "num_groups", "max_chars", "history_size"]
    )
    def test_rejects_non_positive_counts(self, field):
        with pytest.raises(ValueError, match=f"{field} must be >= 1, got 0"):
            CompletionLoggingConfig(**{field: 0})


class TestTruncateMiddle:
    def test_short_text_is_unchanged(self):
        assert truncate_middle("abcdef", 6) == "abcdef"

    def test_long_text_keeps_head_and_tail(self):
        assert truncate_middle("abcdefghij", 4) == "ab\n[... 6 chars truncated ...]\nij"

    def test_one_char_cap_keeps_the_last_char(self):
        assert truncate_middle("abc", 1) == "\n[... 2 chars truncated ...]\nc"


class TestBuildCompletionRecord:
    def test_single_turn_splits_prompt_from_completion(self):
        # Arrange
        token_ids = torch.tensor([[1, 2, 3, 4]])
        action_mask = torch.tensor([[False, True, True]])

        # Act
        record = build_completion_record(
            LetterTokenizer(),
            token_ids,
            action_mask,
            torch.tensor([1.0, 0.5]),
            step=7,
            agent_index=1,
            group_index=2,
            sample_index=3,
            max_chars=100,
        )

        # Assert
        assert record == CompletionRecord(
            step=7,
            agent_index=1,
            group_index=2,
            sample_index=3,
            reward=1.5,
            num_turns=1,
            completion_tokens=2,
            prompt="ab",
            completion="cd",
        )

    def test_multi_turn_completion_includes_feedback_and_counts_turns(self):
        # Tokens: prompt a | policy b c | feedback d | policy e
        token_ids = torch.tensor([1, 2, 3, 4, 5])
        action_mask = torch.tensor([True, True, False, True])

        record = build_completion_record(
            LetterTokenizer(),
            token_ids,
            action_mask,
            torch.tensor([0.0, 1.0]),
            step=0,
            agent_index=0,
            group_index=0,
            sample_index=0,
            max_chars=100,
        )

        assert record.prompt == "a"
        assert record.completion == "bcde"
        assert record.num_turns == 2
        assert record.completion_tokens == 3

    def test_episode_without_policy_tokens_is_all_prompt(self):
        record = build_completion_record(
            LetterTokenizer(),
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[False, False]]),
            torch.tensor([0.0]),
            step=0,
            agent_index=0,
            group_index=0,
            sample_index=0,
            max_chars=100,
        )

        assert record.prompt == "abc"
        assert record.completion == ""
        assert record.num_turns == 0

    def test_skips_special_tokens_and_truncates_long_fields(self):
        record = build_completion_record(
            LetterTokenizer(),
            torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]]),
            torch.tensor([[False, False, True, True, True, True, True, True]]),
            torch.tensor([0.0]),
            step=0,
            agent_index=0,
            group_index=0,
            sample_index=0,
            max_chars=4,
        )

        assert record.prompt == "ab"
        assert record.completion == "cd\n[... 2 chars truncated ...]\ngh"

    def test_rejects_a_tokenizer_that_decodes_to_non_text(self):
        class NonTextTokenizer:
            def decode(self, ids, skip_special_tokens=False):
                return [1, 2]

        with pytest.raises(TypeError, match=r"tokenizer\.decode must return a string"):
            build_completion_record(
                NonTextTokenizer(),
                torch.tensor([[1, 2, 3, 4]]),
                torch.tensor([[False, True, True]]),
                torch.tensor([0.0]),
                step=0,
                agent_index=0,
                group_index=0,
                sample_index=0,
                max_chars=100,
            )


class TestCompletionRecordRender:
    def test_render_has_header_prompt_and_completion(self):
        rendered = _record().render()

        assert rendered == (
            "step 3 | agent 0 | group 1 | sample 2 | reward 0.5 | turns 1 "
            "| completion tokens 2\n--- prompt ---\nab\n--- completion ---\ncd"
        )


class TestCompletionLoggerLog:
    def test_interval_step_writes_one_sampled_group(self):
        # Arrange
        writer = ListWriter()
        completion_logger = CompletionLogger(
            CompletionLoggingConfig(interval=2, num_groups=1), [writer]
        )
        token_ids, action_masks, rewards = _batch(6)

        # Act
        completion_logger.log(
            step=4,
            agent_index=1,
            tokenizer=LetterTokenizer(),
            token_ids=token_ids,
            action_masks=action_masks,
            rewards=rewards,
            group_size=2,
        )

        # Assert
        assert len(writer.writes) == 1
        (written,) = writer.writes
        group_index = written[0].group_index
        assert [r.group_index for r in written] == [group_index, group_index]
        assert [r.sample_index for r in written] == [0, 1]
        assert [r.reward for r in written] == [
            float(2 * group_index),
            float(2 * group_index + 1),
        ]
        assert {r.step for r in written} == {4}
        assert {r.agent_index for r in written} == {1}

    def test_off_interval_step_fills_history_without_writing(self):
        writer = ListWriter()
        completion_logger = CompletionLogger(
            CompletionLoggingConfig(interval=10, history_size=3), [writer]
        )
        token_ids, action_masks, rewards = _batch(4)

        completion_logger.log(
            step=1,
            agent_index=0,
            tokenizer=LetterTokenizer(),
            token_ids=token_ids,
            action_masks=action_masks,
            rewards=rewards,
            group_size=2,
        )

        assert writer.writes == []
        assert len(completion_logger.history) == 2

    def test_history_keeps_only_the_latest_records(self):
        completion_logger = CompletionLogger(
            CompletionLoggingConfig(interval=1, history_size=3), []
        )
        token_ids, action_masks, rewards = _batch(2)

        for step in range(3):
            completion_logger.log(
                step=step,
                agent_index=0,
                tokenizer=LetterTokenizer(),
                token_ids=token_ids,
                action_masks=action_masks,
                rewards=rewards,
                group_size=2,
            )

        assert [r.step for r in completion_logger.history] == [1, 2, 2]

    def test_num_groups_above_batch_samples_every_group(self):
        writer = ListWriter()
        completion_logger = CompletionLogger(
            CompletionLoggingConfig(interval=1, num_groups=5), [writer]
        )
        token_ids, action_masks, rewards = _batch(4)

        completion_logger.log(
            step=0,
            agent_index=0,
            tokenizer=LetterTokenizer(),
            token_ids=token_ids,
            action_masks=action_masks,
            rewards=rewards,
            group_size=2,
        )

        assert [r.group_index for r in writer.writes[0]] == [0, 0, 1, 1]

    def test_empty_batch_writes_nothing(self):
        writer = ListWriter()
        completion_logger = CompletionLogger(
            CompletionLoggingConfig(interval=1), [writer]
        )

        completion_logger.log(
            step=0,
            agent_index=0,
            tokenizer=LetterTokenizer(),
            token_ids=[],
            action_masks=[],
            rewards=[],
            group_size=2,
        )

        assert writer.writes == [[]]
        assert len(completion_logger.history) == 0


class TestCompletionLoggerWriteIfDue:
    def test_one_write_includes_every_agent_on_the_step(self):
        writer = ListWriter()
        completion_logger = CompletionLogger(
            CompletionLoggingConfig(interval=1, num_groups=1), [writer]
        )
        token_ids, action_masks, rewards = _batch(2)

        records: list[CompletionRecord] = []
        for agent_index in (0, 1):
            records.extend(
                completion_logger.sample_groups(
                    step=0,
                    agent_index=agent_index,
                    tokenizer=LetterTokenizer(),
                    token_ids=token_ids,
                    action_masks=action_masks,
                    rewards=rewards,
                    group_size=2,
                )
            )
        completion_logger.write_if_due(0, records)

        assert len(writer.writes) == 1
        assert [r.agent_index for r in writer.writes[0]] == [0, 0, 1, 1]

    def test_off_interval_step_does_not_write(self):
        writer = ListWriter()
        completion_logger = CompletionLogger(
            CompletionLoggingConfig(interval=10), [writer]
        )

        completion_logger.write_if_due(1, [_record()])

        assert writer.writes == []


class TestCompletionLoggerDumpRecent:
    def test_logs_history_at_error_level(self, caplog):
        completion_logger = CompletionLogger(CompletionLoggingConfig(), [])
        completion_logger.history.append(_record(completion="final answer"))

        with caplog.at_level(logging.ERROR):
            completion_logger.dump_recent()

        assert "Last 1 sampled completions before failure" in caplog.text
        assert "final answer" in caplog.text

    def test_empty_history_logs_nothing(self, caplog):
        completion_logger = CompletionLogger(CompletionLoggingConfig(), [])

        with caplog.at_level(logging.ERROR):
            completion_logger.dump_recent()

        assert caplog.text == ""


class TestCompletionLoggerClose:
    def test_closes_every_writer(self):
        writers = [ListWriter(), ListWriter()]
        completion_logger = CompletionLogger(CompletionLoggingConfig(), writers)

        completion_logger.close()

        assert all(writer.closed for writer in writers)


class TestJsonlCompletionWriter:
    def test_appends_one_json_line_per_record(self, tmp_path):
        path = tmp_path / "nested" / "completions.jsonl"
        writer = JsonlCompletionWriter(path)

        writer.write([_record(sample_index=0), _record(sample_index=1)])
        writer.write([_record(sample_index=2)])
        writer.close()

        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert [line["sample_index"] for line in lines] == [0, 1, 2]
        assert lines[0]["completion"] == "cd"


class TestConsoleCompletionWriter:
    def test_prints_only_the_first_record(self, capsys):
        ConsoleCompletionWriter().write(
            [_record(completion="first"), _record(completion="second")]
        )

        out = capsys.readouterr().out
        assert "first" in out
        assert "second" not in out


class TestWandbCompletionWriter:
    def test_logs_a_completions_table_without_committing(self):
        with patch("agilerl.training.llm.completion_logging.wandb.log") as mock_log:
            WandbCompletionWriter().write([_record()])

        (payload,), kwargs = mock_log.call_args
        table = payload["completions"]
        assert kwargs == {"commit": False}
        assert table.columns == [
            "step",
            "agent_index",
            "group_index",
            "sample_index",
            "reward",
            "num_turns",
            "completion_tokens",
            "prompt",
            "completion",
        ]
        assert table.data == [[3, 0, 1, 2, 0.5, 1, 2, "ab", "cd"]]
