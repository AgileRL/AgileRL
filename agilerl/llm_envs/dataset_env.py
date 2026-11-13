# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Dataset-backed LLM env for SFT and DPO.

Scores the model on the dataset's own completions in a single forward pass (no
generation): cross-entropy for SFT, chosen-vs-rejected preference for DPO. SFT
rows are ``(prompt, response)`` pairs or multi-turn ``messages`` with optional
``tools``; both train the chat template's render of that conversation. DPO rows
are ``(prompt, chosen, rejected)`` strings, or messages trained the same way.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from agilerl.llm_envs.chat_formats import (
    ChatFormat,
    detect_chat_format,
)
from agilerl.llm_envs.chat_transcript import (
    _as_messages,
    _messages_for_render,
    _named_tool_results,
    _parsed_json,
    _render,
    _with_reasoning_key,
    _without_none_keys,
    chat_messages,
    encode_chat_transcript,
    template_reasoning_key,
)
from agilerl.utils.chat_template import (
    inject_chat_template_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    CollateBuilder = Callable[
        ["DatasetEnv", "PreTrainedTokenizerBase"],
        Callable[[list[dict[str, Any]]], dict[str, Any]],
    ]


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
    :ivar test_dataset: Held-out split after filtering and sharding.
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
            column) trained on their assistant turns. Preference rows whose
            ``prompt``, ``chosen`` or ``rejected`` are messages are trained on
            each side's assistant turns after the shared prompt.
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
        self.test_dataset = test_dataset
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
        if self.max_context_length is not None and is_chat_preference_row(dataset[0]):
            return self._select_fitting_preference_pairs(dataset, dataset_type)
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
        cut = 0
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
            fitted = self._fit_to_context(ids, mask)
            if fitted is None:
                too_long += 1
                continue
            kept.append(index)
            if len(fitted[0]) < len(ids):
                cut += 1
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
        if cut:
            warnings.warn(
                f"{cut} samples in the {dataset_type} were cut back to their last "
                f"complete assistant turn within max_context_length="
                f"{self.max_context_length}; their later turns are not trained.",
                stacklevel=2,
            )
        return dataset if len(kept) == len(dataset) else dataset.select(kept)

    def _select_fitting_preference_pairs(
        self, dataset: Dataset, dataset_type: str
    ) -> Dataset:
        """Keep the chat preference pairs that fit the context, shortening their shared prompt; warns with counts.

        :param dataset: Split to check.
        :param dataset_type: Name of the split, for messages.
        :return: The split without the pairs whose completions alone do not fit.
        :raises ValueError: If no pair is left.
        """
        tokenizer = inject_chat_template_kwargs(
            self.tokenizer, self.chat_template_kwargs
        )
        chat_format = detect_chat_format(tokenizer)
        kept: list[int] = []
        shortened = 0
        errors: list[str] = []
        for index, item in enumerate(dataset):
            # One malformed row must not stop a run; all of them failing raises below.
            try:
                fitted = fit_preference_pair(
                    tokenizer, item, chat_format, self.max_context_length
                )
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                continue
            if fitted is None:
                continue
            kept.append(index)
            shortened += fitted.shortened
        dropped = len(dataset) - len(kept) - len(errors)
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
        if shortened:
            warnings.warn(
                f"{shortened} preference pairs in the {dataset_type} had messages "
                f"dropped from the front of their shared prompt to fit "
                f"max_context_length={self.max_context_length}.",
                stacklevel=2,
            )
        if dropped:
            warnings.warn(
                f"{dropped} preference pairs were filtered out of the {dataset_type}: "
                f"their chosen or rejected completion alone does not fit "
                f"max_context_length={self.max_context_length}.",
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


def completion_loss_mask(
    prompt_lengths: list[int], attention_mask: torch.Tensor
) -> torch.Tensor:
    """True at the non-pad tokens after each row's prompt."""
    positions = torch.arange(attention_mask.shape[1]).unsqueeze(0)
    after_prompt = positions >= torch.tensor(prompt_lengths).unsqueeze(1)
    return after_prompt & attention_mask.bool()


def preference_collate_builder(
    env: DatasetEnv,
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Build a collate function for ``(prompt, chosen, rejected)`` DPO rows.

    String rows are tokenized as ``prompt`` then completion plus eos. Rows whose
    fields are messages go through the chat template, see :func:`fit_preference_pair`.
    """
    tokenizer = inject_chat_template_kwargs(tokenizer, env.chat_template_kwargs)
    chat_format = detect_chat_format(tokenizer)

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Tokenize preference triples into chosen/rejected paired tensors and loss masks."""
        if any(is_chat_preference_row(item) for item in batch):
            return _collate_chat_pairs(env, tokenizer, chat_format, batch)
        return _collate_string_pairs(env, tokenizer, batch)

    return collate_fn


def _collate_string_pairs(
    env: DatasetEnv,
    tokenizer: PreTrainedTokenizerBase,
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tokenize string preference rows as prompt then completion plus eos."""
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
        rejected_ids = tokenizer(prompts, rejected_eos, truncation=True, padding=False)
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

    chosen_attention_mask = chosen_enc["attention_mask"].long()
    rejected_attention_mask = rejected_enc["attention_mask"].long()
    return {
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_loss_mask": completion_loss_mask(prompt_lengths, chosen_attention_mask),
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_loss_mask": completion_loss_mask(
            prompt_lengths, rejected_attention_mask
        ),
    }


def _collate_chat_pairs(
    env: DatasetEnv,
    tokenizer: PreTrainedTokenizerBase,
    chat_format: ChatFormat | None,
    batch: list[dict[str, Any]],
) -> dict[str, Any]:
    """Encode chat preference rows and right-pad both sides to one width."""
    if tokenizer.pad_token_id is None:
        msg = (
            "Chat preference rows are padded with tokenizer.pad_token_id, but this "
            "tokenizer has no pad token. Set one before building the env."
        )
        raise ValueError(msg)
    sides: list[tuple[list[int], list[bool]]] = []
    for item in batch:
        fitted = fit_preference_pair(
            tokenizer, item, chat_format, env.max_context_length
        )
        if fitted is None:
            msg = (
                "A preference pair's chosen or rejected completion alone does not "
                f"fit max_context_length={env.max_context_length}."
            )
            raise ValueError(msg)
        sides += [fitted.chosen, fitted.rejected]
    width = max(len(ids) for ids, _ in sides)
    ids = torch.full((len(sides), width), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(sides), width), dtype=torch.long)
    loss = torch.zeros((len(sides), width), dtype=torch.bool)
    for row, (side_ids, side_loss) in enumerate(sides):
        ids[row, : len(side_ids)] = torch.tensor(side_ids, dtype=torch.long)
        attention[row, : len(side_ids)] = 1
        loss[row, : len(side_loss)] = torch.tensor(side_loss, dtype=torch.bool)
    return {
        "prompt": [item["prompt"] for item in batch],
        "chosen": [item["chosen"] for item in batch],
        "rejected": [item["rejected"] for item in batch],
        "chosen_input_ids": ids[0::2],
        "chosen_attention_mask": attention[0::2],
        "chosen_loss_mask": loss[0::2],
        "rejected_input_ids": ids[1::2],
        "rejected_attention_mask": attention[1::2],
        "rejected_loss_mask": loss[1::2],
    }


def is_chat_preference_row(item: dict[str, Any]) -> bool:
    """Whether a preference row holds messages rather than plain strings.

    :param item: Dataset row with ``prompt``, ``chosen`` and ``rejected``.
    :return: ``True`` when any of the three is not a string.
    """
    return any(
        not isinstance(item[key], str) for key in ("prompt", "chosen", "rejected")
    )


@dataclass(frozen=True)
class FittedPreferencePair:
    """Token ids and loss masks of one chat preference pair.

    :ivar chosen: ``prompt + chosen`` ids and, per id, whether it is trained.
    :ivar rejected: ``prompt + rejected`` ids and, per id, whether it is trained.
    :ivar shortened: Whether messages were dropped from the shared prompt to fit.
    """

    chosen: tuple[list[int], list[bool]]
    rejected: tuple[list[int], list[bool]]
    shortened: bool


def fit_preference_pair(
    tokenizer: PreTrainedTokenizerBase,
    item: dict[str, Any],
    chat_format: ChatFormat | None,
    max_context_length: int | None,
) -> FittedPreferencePair | None:
    """Encode one chat preference row, shortening its shared prompt to fit the context.

    A string field is one user (``prompt``) or assistant message. When a side
    does not fit, the messages chosen and rejected share up to their first
    differing assistant turn join the prompt, then whole prompt messages after
    the system messages and the first user message are dropped: the oldest tool
    results first, then the oldest turns. Completions are never cut.

    :param tokenizer: Tokenizer whose chat template renders the conversation.
    :param item: Row with ``prompt``, ``chosen``, ``rejected`` and optional ``tools``.
    :param chat_format: The model family's chat format.
    :param max_context_length: Token budget per side, or ``None`` for no limit.
    :return: The fitted pair, or ``None`` when the completions do not fit with
        the shortest prompt.
    """
    prompt = _with_named_tool_results(_as_messages(item["prompt"], "user"))
    chosen = _as_messages(item["chosen"], "assistant")
    rejected = _as_messages(item["rejected"], "assistant")
    tools = sft_tools(item)
    pair = encode_preference_pair(
        tokenizer, prompt, chosen, rejected, tools, chat_format
    )
    if max_context_length is None or _pair_fits(pair, max_context_length):
        return FittedPreferencePair(*pair, shortened=False)
    prompt, chosen, rejected = _shared_turns_in_prompt(prompt, chosen, rejected)
    prompts = _shortened_prompts(prompt)

    @cache
    def encoded(drops: int) -> tuple[tuple[list[int], list[bool]], ...]:
        """The pair encoded on the prompt after ``drops`` drops."""
        return encode_preference_pair(
            tokenizer, prompts[drops], chosen, rejected, tools, chat_format
        )

    shortest = len(prompts) - 1
    if not _pair_fits(encoded(shortest), max_context_length):
        return None
    # Each drop removes whole messages, so a side's length only falls with ``drops``.
    low, high = 0, shortest
    while low < high:
        mid = (low + high) // 2
        if _pair_fits(encoded(mid), max_context_length):
            high = mid
        else:
            low = mid + 1
    return FittedPreferencePair(*encoded(low), shortened=low > 0)


def _pair_fits(
    pair: tuple[tuple[list[int], list[bool]], ...], max_context_length: int
) -> bool:
    """Whether both sides of an encoded pair fit ``max_context_length``."""
    return all(len(ids) <= max_context_length for ids, _ in pair)


def encode_preference_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
) -> tuple[tuple[list[int], list[bool]], tuple[list[int], list[bool]]]:
    """Encode ``prompt + chosen`` and ``prompt + rejected``, training each side's assistant turns after the prompt.

    Assistant turns inside ``prompt`` are context. Tool results are never trained.

    :param tokenizer: Tokenizer whose chat template renders the conversation.
    :param prompt: Normalized shared messages.
    :param chosen: Normalized chosen messages, starting with an assistant turn.
    :param rejected: Normalized rejected messages, starting with an assistant turn.
    :param tools: Tool schemas passed to the template, or ``None``.
    :param chat_format: The model family's chat format.
    :return: Per side, token ids and whether each id is trained.
    :raises ValueError: If a completion does not start with an assistant turn,
        or the prompt's render is not a token prefix of a side.
    """
    prompt_ids = _prompt_render_ids(tokenizer, prompt, tools, chat_format)
    sides = []
    for label, completion in (("chosen", chosen), ("rejected", rejected)):
        if not completion or completion[0]["role"] != "assistant":
            msg = f"The {label} messages must start with an assistant turn."
            raise ValueError(msg)
        ids, mask = encode_chat_transcript(
            tokenizer, [*prompt, *completion], tools, chat_format
        )
        if ids[: len(prompt_ids)] != prompt_ids:
            msg = (
                f"The chat template's render of the prompt is not a token prefix of "
                f"prompt + {label}, so the shared context cannot be masked out: "
                f"{tokenizer.decode(prompt_ids)!r} vs {tokenizer.decode(ids)!r}."
            )
            raise ValueError(msg)
        sides.append((ids, [False] * len(prompt_ids) + mask[len(prompt_ids) :]))
    return sides[0], sides[1]


def _prompt_render_ids(
    tokenizer: PreTrainedTokenizerBase,
    prompt: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_format: ChatFormat | None,
) -> list[int]:
    """Ids of the chat template's render of ``prompt``, normalized as :func:`encode_chat_transcript` does."""
    reasoning_key = template_reasoning_key(tokenizer)
    messages = [_with_reasoning_key(message, reasoning_key) for message in prompt]
    render_kwargs: dict[str, Any] = {} if tools is None else {"tools": tools}
    text = _render(
        tokenizer, _messages_for_render(messages, chat_format), False, render_kwargs
    )
    return tokenizer.encode(text, add_special_tokens=False)


def _with_named_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``messages`` with each tool result named after the call it answers, see :func:`_named_tool_results`."""
    named = list(messages)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        end = next(
            (
                later
                for later in range(index + 1, len(messages))
                if messages[later]["role"] == "assistant"
            ),
            len(messages),
        )
        named[index + 1 : end] = _named_tool_results(message, named[index + 1 : end])
    return named


def _shared_turns_in_prompt(
    prompt: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Move the leading messages chosen and rejected share into the prompt.

    The move stops where both sides still start with an assistant turn, so each
    keeps a completion. Shared tokens cancel in the DPO margin.
    """
    split = 0
    for index, (left, right) in enumerate(zip(chosen, rejected, strict=False)):
        if index > 0 and left["role"] == right["role"] == "assistant":
            split = index
        if left != right:
            break
    shared = _with_named_tool_results([*prompt, *chosen[:split]])
    return shared, chosen[split:], rejected[split:]


def _shortened_prompts(prompt: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """``prompt``, then each prompt one more drop shorter, down to its protected head.

    The system messages and the first user message are kept. The oldest tool
    result is dropped first while that leaves no assistant turn without a
    following message; then the oldest turn: a user message, or an assistant
    message with the tool results after it.
    """
    first_user = next(
        (index for index, message in enumerate(prompt) if message["role"] == "user"),
        None,
    )
    if first_user is None:
        systems = next(
            (i for i, m in enumerate(prompt) if m["role"] != "system"), len(prompt)
        )
        head = max(1, systems)
    else:
        head = first_user + 1
    current = list(prompt)
    prompts = [current]
    index = head
    while index < len(current):
        if current[index]["role"] != "tool":
            index += 1
            continue
        candidate = [*current[:index], *current[index + 1 :]]
        if _assistant_turns_are_followed(candidate):
            current = candidate
            prompts.append(current)
        else:
            index += 1
    while len(current) > head:
        end = next(
            (
                later
                for later in range(head + 1, len(current))
                if current[later]["role"] in ("user", "assistant")
            ),
            len(current),
        )
        current = [*current[:head], *current[end:]]
        prompts.append(current)
    return prompts


def _assistant_turns_are_followed(prompt: list[dict[str, Any]]) -> bool:
    """Whether every assistant turn in ``prompt`` has a non-assistant message after it."""
    return all(
        message["role"] != "assistant"
        or (index + 1 < len(prompt) and prompt[index + 1]["role"] != "assistant")
        for index, message in enumerate(prompt)
    )


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
