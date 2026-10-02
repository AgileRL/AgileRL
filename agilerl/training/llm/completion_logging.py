# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Sampled prompt / completion / reward records from LLM rollout batches.

Records are decoded from the trainer's token tensors, so any loop holding
``token_ids``, ``action_masks`` and ``rewards`` can log them. Writers decide
where records go; a ring buffer keeps the latest ones for a failure dump.
"""

from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, astuple, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import wandb

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

__all__ = [
    "CompletionLogger",
    "CompletionLoggingConfig",
    "CompletionRecord",
    "CompletionWriter",
    "ConsoleCompletionWriter",
    "JsonlCompletionWriter",
    "WandbCompletionWriter",
    "build_completion_record",
    "truncate_middle",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletionLoggingConfig:
    """How often and how much rollout text to log.

    :param interval: Training steps between writes to the writers.
    :param num_groups: Prompt groups sampled per step.
    :param max_chars: Character cap per text field; longer text keeps its head and tail.
    :param history_size: Latest records kept for :meth:`CompletionLogger.dump_recent`.
    :param jsonl_path: File the training loop appends written records to, as JSON lines.
    :param seed: Seed for the group sampler.
    """

    interval: int = 10
    num_groups: int = 1
    max_chars: int = 2000
    history_size: int = 8
    jsonl_path: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject non-positive counts."""
        for name, value in self.__dict__.items():
            if name not in ("seed", "jsonl_path") and value < 1:
                msg = f"{name} must be >= 1, got {value}"
                raise ValueError(msg)


@dataclass(frozen=True)
class CompletionRecord:
    """One sampled trajectory: its prompt, completion and episode reward.

    ``completion`` runs from the first policy token to the end of the episode,
    so a multi-turn record includes the env's feedback turns.
    """

    step: int
    agent_index: int
    group_index: int
    sample_index: int
    reward: float
    num_turns: int
    completion_tokens: int
    prompt: str
    completion: str

    def render(self) -> str:
        """Readable header line, prompt and completion."""
        header = (
            f"step {self.step} | agent {self.agent_index} | group {self.group_index} "
            f"| sample {self.sample_index} | reward {self.reward:.4g} "
            f"| turns {self.num_turns} | completion tokens {self.completion_tokens}"
        )
        return (
            f"{header}\n--- prompt ---\n{self.prompt}\n"
            f"--- completion ---\n{self.completion}"
        )


def truncate_middle(text: str, max_chars: int) -> str:
    """Cap ``text`` at ``max_chars`` characters, keeping its head and tail."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    dropped = len(text) - max_chars
    return f"{text[:head]}\n[... {dropped} chars truncated ...]\n{text[-tail:]}"


def _decode_str(tokenizer: PreTrainedTokenizerBase, token_ids: list[int]) -> str:
    """Decode one flat id sequence to a string."""
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    if not isinstance(decoded, str):
        msg = "tokenizer.decode must return a string"
        raise TypeError(msg)
    return decoded


def build_completion_record(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: torch.Tensor,
    action_mask: torch.Tensor,
    rewards: torch.Tensor,
    *,
    step: int,
    agent_index: int,
    group_index: int,
    sample_index: int,
    max_chars: int,
) -> CompletionRecord:
    """Decode one trajectory into a :class:`CompletionRecord`.

    :param tokenizer: Tokenizer the episode was encoded with.
    :param token_ids: Episode tokens, ``(1, seq)`` or ``(seq,)``.
    :param action_mask: ``(1, seq - 1)`` or ``(seq - 1,)``; ``True`` at ``i``
        marks token ``i + 1`` as policy-generated.
    :param rewards: Per-turn rewards; the record carries their sum.
    :param step: Training step the trajectory was learned on.
    :param agent_index: Population member that generated it.
    :param group_index: Prompt group within the batch.
    :param sample_index: Position within its group.
    :param max_chars: Character cap per text field.
    :return: The decoded record.
    """
    ids = token_ids.reshape(-1).tolist()
    mask = action_mask.reshape(-1).bool().cpu()
    policy_positions = torch.nonzero(mask).flatten()
    first_policy_token = (
        int(policy_positions[0]) + 1 if policy_positions.numel() else len(ids)
    )
    turn_starts = mask.clone()
    turn_starts[1:] &= ~mask[:-1]
    prompt = _decode_str(tokenizer, ids[:first_policy_token])
    completion = _decode_str(tokenizer, ids[first_policy_token:])
    return CompletionRecord(
        step=step,
        agent_index=agent_index,
        group_index=group_index,
        sample_index=sample_index,
        reward=float(rewards.float().sum()),
        num_turns=int(turn_starts.sum()),
        completion_tokens=int(mask.sum()),
        prompt=truncate_middle(prompt, max_chars),
        completion=truncate_middle(completion, max_chars),
    )


class CompletionWriter(ABC):
    """Destination for written completion records."""

    @abstractmethod
    def write(self, records: Sequence[CompletionRecord]) -> None:
        """Persist one step's sampled records."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources the writer holds."""


class JsonlCompletionWriter(CompletionWriter):
    """Append records to a file as JSON lines."""

    def __init__(self, path: str | Path) -> None:
        """Open ``path`` for appending, creating parent directories.

        :param path: Output file.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, records: Sequence[CompletionRecord]) -> None:
        """Append one JSON line per record and flush."""
        for record in records:
            self._file.write(json.dumps(asdict(record)) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the file."""
        self._file.close()


class ConsoleCompletionWriter(CompletionWriter):
    """Print the first record of each write."""

    def write(self, records: Sequence[CompletionRecord]) -> None:
        """Print one rendered record."""
        if records:
            print(records[0].render())

    def close(self) -> None:
        """Nothing to release."""


class WandbCompletionWriter(CompletionWriter):
    """Log each write as a ``completions`` table on the active W&B run."""

    def write(self, records: Sequence[CompletionRecord]) -> None:
        """Attach a table of ``records`` to the run's next metrics commit."""
        table = wandb.Table(
            columns=[field.name for field in fields(CompletionRecord)],
            data=[list(astuple(record)) for record in records],
        )
        wandb.log({"completions": table}, commit=False)

    def close(self) -> None:
        """The metrics logger finishes the run."""


class CompletionLogger:
    """Sample rollout groups every step; write them to writers every ``interval`` steps."""

    def __init__(
        self,
        config: CompletionLoggingConfig,
        writers: Sequence[CompletionWriter],
    ) -> None:
        """Build a logger.

        :param config: Sampling and truncation settings.
        :param writers: Where written records go; empty keeps only the history.
        """
        self.config = config
        self.writers = list(writers)
        self.history: deque[CompletionRecord] = deque(maxlen=config.history_size)
        self._rng = random.Random(config.seed)

    def sample_groups(
        self,
        *,
        step: int,
        agent_index: int,
        tokenizer: PreTrainedTokenizerBase,
        token_ids: Sequence[torch.Tensor],
        action_masks: Sequence[torch.Tensor],
        rewards: Sequence[torch.Tensor],
        group_size: int,
    ) -> list[CompletionRecord]:
        """Decode sampled groups into :attr:`history` and return this call's records.

        Rows are group-contiguous: row ``g * group_size + s`` is sample ``s`` of group ``g``.

        :param step: Training step the batch is learned on.
        :param agent_index: Population member that generated the batch.
        :param tokenizer: Tokenizer the batch was encoded with.
        :param token_ids: One episode token tensor per row.
        :param action_masks: One policy-token mask per row.
        :param rewards: One per-turn reward tensor per row.
        :param group_size: Rows per prompt group.
        :return: The sampled records, in group then sample order.
        """
        num_groups = len(token_ids) // group_size
        sampled = self._rng.sample(
            range(num_groups), min(self.config.num_groups, num_groups)
        )
        records: list[CompletionRecord] = []
        for group_index in sorted(sampled):
            for sample_index in range(group_size):
                row = group_index * group_size + sample_index
                records.append(
                    build_completion_record(
                        tokenizer,
                        token_ids[row],
                        action_masks[row],
                        rewards[row],
                        step=step,
                        agent_index=agent_index,
                        group_index=group_index,
                        sample_index=sample_index,
                        max_chars=self.config.max_chars,
                    )
                )
        self.history.extend(records)
        return records

    def write_if_due(self, step: int, records: Sequence[CompletionRecord]) -> None:
        """Write ``records`` to every writer when ``step`` is an interval step.

        :param step: Training step the records were sampled on.
        :param records: Records to write; typically every agent sampled this step.
        """
        if step % self.config.interval != 0:
            return
        for writer in self.writers:
            writer.write(records)

    def log(
        self,
        *,
        step: int,
        agent_index: int,
        tokenizer: PreTrainedTokenizerBase,
        token_ids: Sequence[torch.Tensor],
        action_masks: Sequence[torch.Tensor],
        rewards: Sequence[torch.Tensor],
        group_size: int,
    ) -> None:
        """Sample this agent's groups and write them if ``step`` is on the interval.

        :param step: Training step the batch is learned on.
        :param agent_index: Population member that generated the batch.
        :param tokenizer: Tokenizer the batch was encoded with.
        :param token_ids: One episode token tensor per row.
        :param action_masks: One policy-token mask per row.
        :param rewards: One per-turn reward tensor per row.
        :param group_size: Rows per prompt group.
        """
        records = self.sample_groups(
            step=step,
            agent_index=agent_index,
            tokenizer=tokenizer,
            token_ids=token_ids,
            action_masks=action_masks,
            rewards=rewards,
            group_size=group_size,
        )
        self.write_if_due(step, records)

    def dump_recent(self) -> None:
        """Log :attr:`history` at error level, for a run that is failing."""
        if not self.history:
            return
        logger.error(
            "Last %d sampled completions before failure:\n\n%s",
            len(self.history),
            "\n\n".join(record.render() for record in self.history),
        )

    def close(self) -> None:
        """Close every writer."""
        for writer in self.writers:
            writer.close()
