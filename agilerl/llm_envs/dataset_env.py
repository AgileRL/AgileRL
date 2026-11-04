# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Dataset-backed LLM env for SFT and DPO.

Scores the model on the dataset's own completions in a single forward pass (no
generation): cross-entropy for SFT, chosen-vs-rejected preference for DPO. SFT
rows are ``(prompt, response)`` pairs or multi-turn ``messages`` with optional
``tools``; both train the chat-template transcript a rollout would build.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable, Collection
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from agilerl.llm_envs.chat_formats import (
    ChatFormat,
    ToolCallParseStatus,
    detect_chat_format,
    frame_repeats_terminator,
)
from agilerl.utils.chat_template import (
    feedback_frame_parts,
    inject_chat_template_kwargs,
    special_token_ids,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    CollateBuilder = Callable[
        ["DatasetEnv", "PreTrainedTokenizerBase"],
        Callable[[list[dict[str, Any]]], dict[str, Any]],
    ]

TURN_OPENER = {"role": "user", "content": "."}


def sft_required_columns(columns: Collection[str], response_column: str) -> set[str]:
    """Columns an SFT split needs: ``messages``, else a prompt and a response.

    :param columns: Columns the dataset has.
    :param response_column: Column holding the response of a ``prompt`` row.
    :return: The required column names.
    """
    return {"messages"} if "messages" in columns else {"prompt", response_column}


class DatasetEnv:
    """Dataset-backed LLM env for SFT and DPO (no generation).

    :ivar dataset_size: ``{"train": N, "test": M}`` row counts after filtering.
    :ivar num_epochs: Full passes completed over the train split.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        *,
        objective: Literal["preference", "sft"],
        response_column: str = "response",
        chat_template_kwargs: dict[str, Any] | None = None,
        data_batch_size_per_gpu: int = 8,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Build a dataset env for the selected ``objective``.

        One contiguous equal-size shard per rank, taken after context-length
        filtering so every rank agrees on the rows and epoch boundaries.

        :param train_dataset: Training split of prompt/label rows. SFT rows with a
            ``messages`` column are multi-turn conversations (optional ``tools``
            column) trained on their assistant turns.
        :param test_dataset: Held-out split used under :meth:`eval_mode`.
        :param tokenizer: Tokenizer used by the collate function.
        :param objective: ``"preference"`` (DPO) or ``"sft"``.
        :param response_column: Target column for ``(prompt, response)`` SFT rows.
        :param chat_template_kwargs: Extra kwargs for any chat-template render of
            the dataset's prompts (e.g. ``{"enable_thinking": False}``).
        :param data_batch_size_per_gpu: Batch size for the train/test dataloaders.
        :param max_context_length: Optional max prompt+completion token budget.
        :param min_completion_length: Minimum reserved completion token budget.
        :param seed: Seed for dataloader shuffling.
        :param rank: This process's shard index in ``[0, world_size)``.
        :param world_size: Number of data-parallel shards (``1`` = no sharding).
        """
        if world_size < 1:
            msg = f"world_size must be >= 1, got {world_size}."
            raise ValueError(msg)
        if not 0 <= rank < world_size:
            msg = f"rank must be in [0, {world_size}), got {rank}."
            raise ValueError(msg)
        if objective == "preference":
            required_columns = {"prompt", "chosen", "rejected"}
            collate_builder: CollateBuilder = preference_collate_builder
        elif objective == "sft":
            if tokenizer.pad_token_id is None:
                msg = (
                    "SFT pads its batches with tokenizer.pad_token_id, but this "
                    "tokenizer has no pad token. Set one before building the env."
                )
                raise ValueError(msg)
            required_columns = sft_required_columns(
                train_dataset.features, response_column
            )
            collate_builder = sft_collate_builder
        else:
            msg = f"Unknown dataset objective {objective!r}; expected 'preference' or 'sft'."
            raise ValueError(msg)
        self.objective = objective
        self.required_columns = set(required_columns)
        self.response_column = response_column
        self._collate_builder = collate_builder
        for label, dataset in (("Train", train_dataset), ("Test", test_dataset)):
            if len(dataset) == 0:
                msg = f"{label} dataset is empty; each split needs at least one row."
                raise ValueError(msg)
            if not self.required_columns.issubset(set(dataset.features.keys())):
                msg = (
                    f"{label} dataset must contain columns "
                    f"{sorted(self.required_columns)}."
                )
                raise ValueError(msg)

        self.name = train_dataset.info.dataset_name
        self.tokenizer = tokenizer
        self.chat_template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        self.data_batch_size_per_gpu = data_batch_size_per_gpu
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.min_completion_length = (
            0 if min_completion_length is None else min_completion_length
        )
        self.max_context_length = max_context_length
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        custom_collate_fn = self.create_collate_fn(tokenizer)
        train_dataset = self._filter_dataset_by_max_context_length(
            train_dataset,
            "train dataset",
        )
        test_dataset = self._filter_dataset_by_max_context_length(
            test_dataset,
            "test dataset",
        )
        train_dataset = self._shard_for_rank(train_dataset, "train dataset")
        test_dataset = self._shard_for_rank(test_dataset, "test dataset")
        self.train_dataloader = DataLoader(
            train_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=data_batch_size_per_gpu,
            shuffle=True,
            collate_fn=custom_collate_fn,
            generator=generator,
        )
        self.test_dataloader = DataLoader(
            test_dataset,  # ty: ignore[invalid-argument-type]
            batch_size=data_batch_size_per_gpu,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        self.dataset_size = {
            "train": len(train_dataset),
            "test": len(test_dataset),
        }
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = self.train_dataloader_iter
        self.evaluation_mode = False
        self.num_epochs = 0

    def create_collate_fn(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Build the row-collation callable for the current objective.

        :param tokenizer: Tokenizer used to encode prompts and labels.
        :return: Callable mapping rows to a tensor batch dict.
        """
        return self._collate_builder(self, tokenizer)

    def reset(self, reset_dataloaders: bool = False) -> dict[str, Any]:
        """Return the next batch (walks the dataset), optionally rewinding dataloaders first.

        :param reset_dataloaders: Whether to rewind train/test iterators first.
        :return: Objective-specific collated batch from the active split.
        """
        if reset_dataloaders:
            self._reset_dataloaders()
            self.num_epochs = 0
        return self._get_next_batch()

    def _get_next_batch(self) -> dict[str, Any]:
        """Read one batch and cycle dataloaders at split boundaries."""
        try:
            return next(self.dataloader)
        except StopIteration:
            if not self.evaluation_mode:
                self.num_epochs += 1
            self._reset_dataloaders(
                reset_train=not self.evaluation_mode,
                reset_test=self.evaluation_mode,
            )
            return next(self.dataloader)

    @contextmanager
    def eval_mode(self) -> Generator[None, None, None]:
        """Temporarily switch reads to the held-out split, restoring the prior mode.

        Saves/restores the entry mode so nested probes don't flip an outer eval context.
        """
        previous_mode = self.evaluation_mode
        self.dataloader = self.test_dataloader_iter
        self.evaluation_mode = True
        try:
            yield
        finally:
            self.evaluation_mode = previous_mode
            # Repoint at the live iterator; an inner reset may have rebuilt it.
            self.dataloader = (
                self.test_dataloader_iter
                if previous_mode
                else self.train_dataloader_iter
            )

    def __len__(self) -> int:
        """Return row count for the currently active split."""
        return self.dataset_size["test" if self.evaluation_mode else "train"]

    def close(self) -> None:
        # LocalTrainer.close() always calls close(); this env holds no file handles.
        pass

    def _reset_dataloaders(
        self, reset_train: bool = True, reset_test: bool = True
    ) -> None:
        """Rebuild train/test iterators and refresh the active iterator pointer."""
        if reset_train:
            self.train_dataloader_iter = iter(self.train_dataloader)
        if reset_test:
            self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = (
            self.test_dataloader_iter
            if self.evaluation_mode
            else self.train_dataloader_iter
        )

    def _shard_for_rank(self, dataset: Dataset, dataset_type: str) -> Dataset:
        """This rank's contiguous, equal-size shard of ``dataset`` (identity when unsharded).

        Shards are truncated to ``len // world_size`` rows so every rank walks the same
        number of batches per epoch; the remainder rows (< ``world_size``) are dropped.
        """
        if self.world_size == 1:
            return dataset
        shard_size = len(dataset) // self.world_size
        if shard_size == 0:
            msg = (
                f"rank {self.rank} of {self.world_size} gets an empty shard of the "
                f"{len(dataset)}-row {dataset_type}; reduce world_size."
            )
            raise ValueError(msg)
        start = self.rank * shard_size
        return dataset.select(range(start, start + shard_size))

    def _filter_dataset_by_max_context_length(
        self,
        dataset: Dataset,
        dataset_type: str | None = None,
    ) -> Dataset:
        """Drop rows where ``prompt_len > max_context_length - min_completion_length``; warns if any."""
        dataset_type = "dataset" if dataset_type is None else dataset_type
        if self.objective == "sft":
            return self._select_trainable_sft_rows(dataset, dataset_type)
        if (
            self.max_context_length is None
            or "prompt" not in dataset.features
            or not isinstance(dataset[0]["prompt"], str)
        ):
            return dataset
        max_prompt_tokens = self.max_context_length - self.min_completion_length
        filtered_dataset = dataset.filter(
            lambda x: len(self.tokenizer.encode(x["prompt"])) <= max_prompt_tokens,
        )
        if len(filtered_dataset) == 0:
            msg = f"No samples left in the {dataset_type} after filtering by the max context length constraint, use a larger max context length."
            raise ValueError(msg)
        if (dataset_difference := len(dataset) - len(filtered_dataset)) > 0:
            warnings.warn(
                f"{dataset_difference} samples were filtered out of the {dataset_type} due to the max context length constraint.",
                stacklevel=2,
            )
        return filtered_dataset

    def _select_trainable_sft_rows(
        self, dataset: Dataset, dataset_type: str
    ) -> Dataset:
        """Keep the rows that encode to a transcript whose first assistant turn fits; warns if any are dropped.

        :param dataset: Split to check.
        :param dataset_type: Name of the split, for messages.
        :return: The split without the dropped rows.
        :raises ValueError: If no row is left.
        """
        chat_format = detect_chat_format(self.tokenizer)
        kept: list[int] = []
        too_long = 0
        errors: list[str] = []
        for index, item in enumerate(dataset):
            # One malformed row must not stop a run; all of them failing raises below.
            try:
                ids, mask = encode_sft_row(
                    self.tokenizer, item, self.response_column, chat_format
                )
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                continue
            if self._fit_to_context(ids, mask) is None:
                too_long += 1
            else:
                kept.append(index)
        if not kept:
            msg = (
                f"No valid samples in the {dataset_type}. First error: {errors[0]}"
                if errors
                else f"No samples left in the {dataset_type} after filtering by the max context length constraint, use a larger max context length."
            )
            raise ValueError(msg)
        if errors:
            warnings.warn(
                f"{len(errors)} invalid samples were dropped from the {dataset_type}. First error: {errors[0]}",
                stacklevel=2,
            )
        if too_long:
            warnings.warn(
                f"{too_long} samples were filtered out of the {dataset_type} due to the max context length constraint.",
                stacklevel=2,
            )
        return dataset if len(kept) == len(dataset) else dataset.select(kept)

    def _fit_to_context(
        self, ids: list[int], mask: list[bool]
    ) -> tuple[list[int], list[bool]] | None:
        """Transcript cut to its last complete assistant turn within the context length.

        A cut inside a turn would train a partial tool call with no stop token.

        :param ids: Transcript token ids.
        :param mask: Whether each id is an assistant token.
        :return: The cut transcript, or ``None`` when its prompt leaves less than
            ``min_completion_length`` or its first assistant turn does not fit.
        """
        if self.max_context_length is None:
            return ids, mask
        prompt_length = mask.index(True)
        if prompt_length > self.max_context_length - self.min_completion_length:
            return None
        if len(ids) <= self.max_context_length:
            return ids, mask
        for end in range(self.max_context_length, prompt_length, -1):
            if mask[end - 1] and not mask[end]:
                return ids[:end], mask[:end]
        return None


def _prompt_lengths(
    tokenizer: PreTrainedTokenizerBase, prompts: list[str]
) -> list[int]:
    """Unpadded prompt token lengths, for masking the prompt span of a pair encoding."""
    enc = tokenizer(prompts, truncation=True, padding=False, add_special_tokens=True)
    return [len(ids) for ids in enc["input_ids"]]


def preference_collate_builder(
    env: DatasetEnv,
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    """Build a collate function for ``(prompt, chosen, rejected)`` DPO rows."""
    tokenizer = inject_chat_template_kwargs(tokenizer, env.chat_template_kwargs)

    def collate_fn(batch: list[dict[str, str]]) -> dict[str, Any]:
        """Tokenize preference triples into chosen/rejected paired tensors."""
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]

        prompt_lengths = _prompt_lengths(tokenizer, prompts)

        eos = tokenizer.eos_token or ""
        chosen_eos = [completion + eos for completion in chosen]
        rejected_eos = [completion + eos for completion in rejected]

        if env.max_context_length is not None:
            chosen_enc = tokenizer(
                prompts,
                chosen_eos,
                max_length=env.max_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_enc = tokenizer(
                prompts,
                rejected_eos,
                max_length=env.max_context_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            chosen_ids = tokenizer(prompts, chosen_eos, truncation=True, padding=False)
            rejected_ids = tokenizer(
                prompts, rejected_eos, truncation=True, padding=False
            )
            max_len = max(
                *(len(ids) for ids in chosen_ids["input_ids"]),
                *(len(ids) for ids in rejected_ids["input_ids"]),
            )
            chosen_enc = tokenizer.pad(
                chosen_ids,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            rejected_enc = tokenizer.pad(
                rejected_ids,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )

        return {
            "prompt": prompts,
            "prompt_lengths": prompt_lengths,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc["attention_mask"].long(),
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"].long(),
        }

    return collate_fn


def chat_messages(value: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize one row's OpenAI-style ``messages`` for a chat template.

    Accepts a JSON string or a list. Drops ``None`` fields and decodes string
    tool-call arguments, which HF chat templates expect as a dict. A list is
    Arrow data, which fills every struct key at every depth, so ``None`` is also
    dropped inside its struct arguments; nulls in JSON text are real and kept.

    :param value: The row's ``messages`` value.
    :return: Messages ready for ``apply_chat_template``.
    :raises ValueError: If the row or a call's arguments are not valid JSON, or
        a call's arguments are not an object.
    """
    from_arrow = not isinstance(value, str)
    raw = (
        _parsed_json(value, "The messages column") if isinstance(value, str) else value
    )
    messages = []
    for raw_message in raw:
        message = {key: item for key, item in raw_message.items() if item is not None}
        if "tool_calls" in message:
            message["tool_calls"] = [
                _decoded_tool_call(call, from_arrow) for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages


def sft_tools(item: dict[str, Any]) -> list[dict[str, Any]] | None:
    """One SFT row's tool schemas, without the ``None`` keys Arrow fills into struct schemas.

    :param item: Dataset row.
    :return: Tool schemas for the chat template, or ``None`` when the row has none.
    :raises ValueError: If a string ``tools`` value is not valid JSON.
    """
    tools = item.get("tools")
    if isinstance(tools, str):
        return _parsed_json(tools, "The tools column")
    return _without_none_keys(tools)


def encode_sft_row(
    tokenizer: PreTrainedTokenizerBase,
    item: dict[str, Any],
    response_column: str,
    chat_format: ChatFormat | None,
) -> tuple[list[int], list[bool]]:
    """Token ids and assistant mask of one SFT row, see :func:`encode_chat_transcript`.

    :param tokenizer: Tokenizer whose chat template renders the conversation.
    :param item: Dataset row.
    :param response_column: Column holding the response of a ``prompt`` row.
    :param chat_format: The model family's chat format.
    :return: Token ids and, per id, whether it is an assistant token.
    """
    return encode_chat_transcript(
        tokenizer, sft_messages(item, response_column), sft_tools(item), chat_format
    )


def sft_eval_sample(
    tokenizer: PreTrainedTokenizerBase, item: dict[str, Any]
) -> tuple[str, str]:
    """A ``messages`` row's text up to its first assistant turn, and that turn's text.

    :param tokenizer: Tokenizer whose chat template renders the conversation.
    :param item: Dataset row with a ``messages`` column.
    :return: Prompt text with special tokens, and the turn without them.
    """
    ids, mask = encode_chat_transcript(
        tokenizer,
        chat_messages(item["messages"]),
        sft_tools(item),
        detect_chat_format(tokenizer),
    )
    start = mask.index(True)
    end = next((pos for pos in range(start, len(mask)) if not mask[pos]), len(mask))
    prompt, turn = (
        tokenizer.decode(ids[:start]),
        tokenizer.decode(ids[start:end], skip_special_tokens=True),
    )
    if not isinstance(prompt, str) or not isinstance(turn, str):
        msg = "tokenizer.decode must return a string"
        raise TypeError(msg)
    return prompt, turn


def sft_messages(item: dict[str, Any], response_column: str) -> list[dict[str, Any]]:
    """One SFT row as chat messages: its ``messages``, else ``prompt`` then the response.

    A string prompt is one user message and a string response one assistant
    message; a list is taken as messages.

    :param item: Dataset row.
    :param response_column: Column holding the response of a ``prompt`` row.
    :return: Messages ready for :func:`encode_chat_transcript`.
    """
    if "messages" in item:
        return chat_messages(item["messages"])
    return [
        *_as_messages(item["prompt"], "user"),
        *_as_messages(item[response_column], "assistant"),
    ]


def _as_messages(value: str | list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    """A string as one ``role`` message, or a list of messages normalized."""
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return chat_messages(value)


def _parsed_json(text: str, label: str) -> Any:  # noqa: ANN401 -- any JSON value
    """Decode JSON ``text``, naming ``label`` in the error."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        msg = f"{label} is not valid JSON ({error}): {text[:200]!r}"
        raise ValueError(msg) from error


def _without_none_keys(value: Any) -> Any:  # noqa: ANN401 -- any nested Arrow value
    """``value`` without ``None`` dict entries at any depth; ``None`` list items are data."""
    if isinstance(value, dict):
        return {
            key: _without_none_keys(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_none_keys(item) for item in value]
    return value


def _decoded_tool_call(call: dict[str, Any], from_arrow: bool) -> dict[str, Any]:
    """Tool call without ``None`` fields, with its arguments decoded to an object."""
    call = {key: item for key, item in call.items() if item is not None}
    if "function" in call:
        return {**call, "function": _decoded_tool_call(call["function"], from_arrow)}
    if "arguments" not in call:
        return call
    arguments = call["arguments"]
    label = f"The arguments of tool call {call.get('name')!r}"
    if isinstance(arguments, str):
        arguments = _parsed_json(arguments, label)
    elif from_arrow:
        arguments = _without_none_keys(arguments)
    if not isinstance(arguments, dict):
        msg = f"{label} must be an object, got {type(arguments).__name__}."
        raise ValueError(msg)
    return {**call, "arguments": arguments}


def encode_chat_transcript(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
) -> tuple[list[int], list[bool]]:
    """Token ids of a multi-turn conversation laid out as a rollout transcript, with an assistant mask.

    Matches what :class:`~agilerl.llm_envs.harness.RolloutHarness` builds: the
    messages before the first assistant turn are rendered once with the
    generation prompt, each assistant turn is the text the template renders
    for it up to and including its end-of-turn token, and the messages between
    assistant turns use the chat format's tool-result blocks when the template
    renders results inside the assistant turn (Gemma 4, gpt-oss), else
    :func:`feedback_frame_parts`. Earlier turns are never
    re-rendered, so templates that strip past reasoning keep it here, as they
    do in a rollout. Messages after the last assistant turn are dropped.

    :param tokenizer: Tokenizer whose chat template renders the conversation.
    :param messages: Normalized messages, see :func:`chat_messages`.
    :param tools: Tool schemas passed to the template, or ``None``.
    :param chat_format: The model family's chat format, see
        :func:`~agilerl.llm_envs.chat_formats.detect_chat_format`.
    :return: Token ids and, per id, whether it is an assistant token.
    :raises ValueError: If the conversation has no prompt or no assistant
        turn, has back-to-back assistant turns, the template renders an
        assistant turn that does not extend the generation prompt, or the chat
        format does not parse a turn's rendered tool calls back to its calls.
    """
    render_kwargs: dict[str, Any] = {} if tools is None else {"tools": tools}
    assistant_indices = [
        idx for idx, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indices:
        msg = "An SFT conversation needs at least one assistant message."
        raise ValueError(msg)
    if assistant_indices[0] == 0:
        msg = "An SFT conversation needs a message before its first assistant turn."
        raise ValueError(msg)

    prompt = _render(tokenizer, messages[: assistant_indices[0]], True, render_kwargs)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    mask = [False] * len(ids)
    special_ids = special_token_ids(tokenizer)
    scaffold = _turn_scaffold(tokenizer, render_kwargs, special_ids)
    previous = None
    for idx in assistant_indices:
        if previous is not None:
            between = _named_tool_results(
                messages[previous], messages[previous + 1 : idx]
            )
            if not between:
                msg = f"Messages {previous} and {idx} are back-to-back assistant turns."
                raise ValueError(msg)
            feedback = _feedback_ids(tokenizer, between, render_kwargs, chat_format)
            if len(feedback) > 1 and frame_repeats_terminator(
                ids[-1], feedback[0], special_ids, chat_format
            ):
                feedback = feedback[1:]
            ids += feedback
            mask += [False] * len(feedback)
        turn = _assistant_turn_ids(
            tokenizer,
            _written_tool_calls(messages[idx], chat_format),
            render_kwargs,
            special_ids,
            scaffold,
        )
        _check_tool_calls_parse(messages[idx], turn, tools, chat_format)
        ids += turn
        mask += [True] * len(turn)
        previous = idx
    return ids, mask


def _render(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    render_kwargs: dict[str, Any],
) -> str:
    """Chat-template text of ``messages``."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **render_kwargs,
    )
    if not isinstance(rendered, str):
        msg = f"apply_chat_template returned {type(rendered).__name__}, expected str."
        raise TypeError(msg)
    return rendered


def _turn_scaffold(
    tokenizer: PreTrainedTokenizerBase,
    render_kwargs: dict[str, Any],
    special_ids: set[int],
) -> tuple[str, int]:
    """Generation prompt after :data:`TURN_OPENER`, and its text length through the last special token."""
    prompt = _render(tokenizer, [TURN_OPENER], True, render_kwargs)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    last_special = max(
        (pos for pos, token in enumerate(prompt_ids) if token in special_ids),
        default=-1,
    )
    return prompt, len(tokenizer.decode(prompt_ids[: last_special + 1]))


def _assistant_turn_ids(
    tokenizer: PreTrainedTokenizerBase,
    message: dict[str, Any],
    render_kwargs: dict[str, Any],
    special_ids: set[int],
    scaffold: tuple[str, int],
) -> list[int]:
    """Ids the model samples for one assistant turn, through its end-of-turn token.

    A re-render may drop text the generation prompt prefills after its last
    special token (a ``<think>`` opener), so the turn starts where the two renders part.
    """
    prompt, header = scaffold
    rendered = _render(tokenizer, [TURN_OPENER, message], False, render_kwargs)
    shared = len(os.path.commonprefix([prompt, rendered]))
    if shared < header:
        msg = (
            "The chat template renders an assistant turn that does not extend "
            "its generation prompt, so the turn's sampled tokens cannot be cut out."
        )
        raise ValueError(msg)
    turn = tokenizer.encode(rendered[shared:], add_special_tokens=False)
    last_special = max(
        (pos for pos, token in enumerate(turn) if token in special_ids), default=None
    )
    if last_special is None:
        return [*turn, tokenizer.eos_token_id]
    return turn[: last_special + 1]


def _written_tool_calls(
    message: dict[str, Any], chat_format: ChatFormat | None
) -> dict[str, Any]:
    """``message`` with its tool calls in its content when the chat format writes them."""
    calls = [call.get("function", call) for call in message.get("tool_calls", [])]
    if chat_format is None or not calls:
        return message
    if not all(call.get("name") for call in calls):
        msg = f"An assistant tool call has no name: {calls}."
        raise ValueError(msg)
    text = chat_format.format_tool_calls(
        [
            {"name": call["name"], "arguments": call.get("arguments", {})}
            for call in calls
        ]
    )
    if text is None:
        return message
    content = message.get("content") or ""
    if not isinstance(content, str):
        msg = f"An assistant tool-call turn needs string content, got {type(content).__name__}."
        raise ValueError(msg)
    written = {key: value for key, value in message.items() if key != "tool_calls"}
    written["content"] = content + text
    return written


def _check_tool_calls_parse(
    message: dict[str, Any],
    turn: list[int],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
) -> None:
    """Raise unless the chat format parses ``turn`` to ``message``'s tool calls.

    A rollout runs only the calls the chat format parses, so SFT must not train
    calls it would miss. Names are compared; Gemma's parser keeps scalar
    arguments as text for the executor to coerce.
    """
    expected = [
        call.get("function", call).get("name") for call in message.get("tool_calls", [])
    ]
    if chat_format is None:
        if expected:
            msg = (
                "No chat format parses tool calls for this tokenizer, so a rollout "
                f"could not run the calls {expected} this turn trains."
            )
            raise ValueError(msg)
        return
    _, calls = chat_format.extract_tool_calls(turn, tools)
    parsed = [call.name for call in calls]
    if parsed != expected or any(
        call.status is not ToolCallParseStatus.OK for call in calls
    ):
        statuses = [call.status.value for call in calls]
        msg = (
            f"{type(chat_format).__name__} parses the rendered turn as calls "
            f"{parsed} ({statuses}), but the message calls {expected}."
        )
        raise ValueError(msg)


def _named_tool_results(
    assistant: dict[str, Any], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """``messages`` with each unnamed tool result named after the call it answers.

    A result answers the call its ``tool_call_id`` names; one with neither an id
    nor a name answers the next call no other result answers. Ids are ignored
    when the assistant's calls have none.

    :param assistant: The assistant turn the results follow.
    :param messages: The messages between that turn and the next.
    :return: The messages with tool results named.
    :raises ValueError: If a ``tool_call_id`` matches none of the turn's calls.
    """
    calls = [
        (call.get("id"), call.get("function", call).get("name"))
        for call in assistant.get("tool_calls", [])
    ]
    name_by_id = {call_id: name for call_id, name in calls if call_id is not None}
    unanswered = list(calls)
    for message in messages:
        if message["role"] != "tool":
            continue
        call_id = message.get("tool_call_id") if name_by_id else None
        if call_id is not None and call_id not in name_by_id:
            msg = f"Tool result tool_call_id {call_id!r} is not one of the calls {list(name_by_id)}."
            raise ValueError(msg)
        answered = next(
            (
                call
                for call in unanswered
                if (
                    call[0] == call_id
                    if call_id is not None
                    else call[1] == message.get("name")
                )
            ),
            None,
        )
        if answered is not None:
            unanswered.remove(answered)
    next_names = iter(name for _, name in unanswered)
    named = []
    for message in messages:
        if message["role"] != "tool" or message.get("name"):
            named.append(message)
            continue
        call_id = message.get("tool_call_id") if name_by_id else None
        name = name_by_id[call_id] if call_id is not None else next(next_names, None)
        named.append({**message, "name": name})
    return named


def _feedback_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    render_kwargs: dict[str, Any],
    chat_format: ChatFormat | None,
) -> list[int]:
    """Ids of the non-assistant messages between two assistant turns."""
    for message in messages:
        if not isinstance(message.get("content"), str):
            msg = (
                f"A {message['role']} message between assistant turns needs string "
                f"content, got {type(message.get('content')).__name__}."
            )
            raise ValueError(msg)
    in_turn = None if chat_format is None else chat_format.format_tool_results(messages)
    if in_turn is not None:
        return tokenizer.encode(in_turn, add_special_tokens=False)
    roles = tuple(message["role"] for message in messages)
    parts = feedback_frame_parts(tokenizer, roles, render_kwargs)
    if parts is None:
        msg = f"The chat template cannot render a {roles} turn after an assistant turn."
        raise ValueError(msg)
    text = parts[0] + "".join(
        message["content"] + part
        for message, part in zip(messages, parts[1:], strict=True)
    )
    return tokenizer.encode(text, add_special_tokens=False)


def sft_collate_builder(
    env: DatasetEnv,
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate function for SFT rows, see :func:`sft_messages`."""
    tokenizer = inject_chat_template_kwargs(tokenizer, env.chat_template_kwargs)
    chat_format = detect_chat_format(tokenizer)

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Encode each conversation, cut to the context length and right-pad."""
        rows = []
        for item in batch:
            ids, mask = encode_sft_row(
                tokenizer, item, env.response_column, chat_format
            )
            fitted = env._fit_to_context(ids, mask)
            if fitted is None:
                msg = (
                    f"A {len(ids)}-token conversation has no complete assistant "
                    f"turn within max_context_length={env.max_context_length}."
                )
                raise ValueError(msg)
            rows.append(fitted)
        id_rows = [torch.tensor(ids, dtype=torch.long) for ids, _ in rows]
        return {
            "input_ids": pad_sequence(
                id_rows, batch_first=True, padding_value=tokenizer.pad_token_id
            ),
            "attention_mask": pad_sequence(
                [torch.ones_like(row) for row in id_rows], batch_first=True
            ),
            "loss_mask": pad_sequence(
                [torch.tensor(mask, dtype=torch.bool) for _, mask in rows],
                batch_first=True,
            ),
        }

    return collate_fn
