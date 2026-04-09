import copy
import random
import shutil
import textwrap
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

import gymnasium as gym
import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from agilerl import HAS_LIGER_KERNEL, HAS_LLM_DEPENDENCIES
from agilerl.typing import PreferencePrompts, ReasoningPrompts, SFTPrompts

if HAS_LLM_DEPENDENCIES:
    try:
        import deepspeed
    except ImportError:
        deepspeed = None
    if HAS_LIGER_KERNEL:
        from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOFunction
        from liger_kernel.chunked_loss.fused_linear_preference import (
            LigerFusedLinearPreferenceBase,
        )

        class _LigerDPOWithAlpha(LigerFusedLinearPreferenceBase):
            """Thin wrapper that exposes ``alpha`` for NLL scaling.

            ``LigerFusedLinearDPOFunction`` passes ``compute_nll_loss`` as a bool
            but never forwards ``alpha`` to the base class (which defaults to 1.0).
            This subclass reuses the DPO preference loss and adds ``alpha`` so the
            fused kernel correctly scales the NLL component.
            """

            preference_loss_fn = staticmethod(
                LigerFusedLinearDPOFunction.preference_loss_fn
            )

            @classmethod
            def forward(
                cls,
                ctx,
                _input,
                weight,
                target,
                bias=None,
                ref_input=None,
                ref_weight=None,
                ref_bias=None,
                ignore_index=-100,
                beta=0.1,
                alpha=1.0,
                compute_nll_loss=True,
                compiled=True,
                use_ref_model=True,
                average_log_prob=False,
                chunk_size=1,
                loss_type="sigmoid",
            ):
                return LigerFusedLinearPreferenceBase.forward(
                    cls=cls,
                    ctx=ctx,
                    _input=_input,
                    weight=weight,
                    target=target,
                    bias=bias,
                    ignore_index=ignore_index,
                    alpha=alpha,
                    beta=beta,
                    compute_nll_loss=compute_nll_loss,
                    compiled=compiled,
                    use_ref_model=use_ref_model,
                    ref_input=ref_input,
                    ref_weight=ref_weight,
                    ref_bias=ref_bias,
                    average_log_prob=average_log_prob,
                    chunk_size=chunk_size,
                    loss_type=loss_type,
                )

            @staticmethod
            def backward(ctx, *grad_output):
                grads = LigerFusedLinearPreferenceBase.backward(ctx, grad_output)[:4]
                return (*grads, *(None,) * 12)
    else:
        _LigerDPOWithAlpha = None
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import BatchEncoding
else:
    deepspeed = None
    _LigerDPOWithAlpha = None
    AutoTokenizer = Any
    PreTrainedModel = Any
    BatchEncoding = Any
    Dataset = Any


def apply_chat_template(
    conversation_template: list[dict[str, str]],
    question: str,
    answer: str,
    tokenizer: AutoTokenizer,
) -> BatchEncoding:
    """Create and tokenize a chat template for a reasoning task.

    :param conversation_template: The conversation template to be tokenized.
    :type conversation_template: list[dict[str, str]]
    :param question: The question to be tokenized.
    :type question: str
    :param answer: The answer to be tokenized.
    :type answer: str
    :param tokenizer: The tokenizer to be used.
    :type tokenizer: AutoTokenizer
    :return: The tokenized prompt.
    :rtype: BatchEncoding
    """
    formatted_conversation = [
        {
            "role": msg["role"],
            "content": msg["content"].format(question=question, answer=answer),
        }
        for msg in conversation_template
    ]
    updated_prompt = tokenizer.apply_chat_template(
        formatted_conversation,
        tokenize=False,
        continue_final_message=True,
    )
    return tokenizer(
        [updated_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    )


class HuggingFaceGym(gym.Env, ABC):
    """Abstract base class for HuggingFace Gymnasium environments.

    :param train_dataset: Train dataset to be loaded from HuggingFace datasets.
    :type train_dataset: Dataset
    :param test_dataset: Test dataset to be loaded from HuggingFace datasets.
    :type test_dataset: Dataset
    :param tokenizer: Tokenizer to be used for encoding and decoding the promåpts.
    :type tokenizer: AutoTokenizer
    :param custom_collate_fn: Custom collate function to be used for creating the batch, defaults to None
    :type custom_collate_fn: Callable, optional
    :param conversation_template: A structured conversation that acts as a base pattern for each data point.
    :type conversation_template: list[dict[str, str]]
    :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param max_context_length: Maximum context length, defaults to None
    :type max_context_length: int | None, optional
    :param min_completion_length: Minimum completion length, defaults to None
    :type min_completion_length: int, optional
    :param accelerator: Accelerator to be used for training, defaults to None
    :type accelerator: Accelerator, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        conversation_template: list[dict[str, str]],
        data_batch_size_per_gpu: int = 8,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        accelerator: Accelerator | None = None,
        seed: int = 42,
    ) -> None:
        self.name = train_dataset.info.dataset_name
        self.tokenizer = tokenizer
        self.data_batch_size_per_gpu = data_batch_size_per_gpu
        self.accelerator = accelerator
        self.min_completion_length = (
            0 if min_completion_length is None else min_completion_length
        )
        self.max_context_length = max_context_length
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        self.conversation_template = conversation_template
        custom_collate_fn = self.create_collate_fn(tokenizer)
        dataloader_kwargs = {"collate_fn": custom_collate_fn}
        train_dataset = self._filter_dataset_by_max_context_length(
            train_dataset,
            "train dataset",
        )
        test_dataset = self._filter_dataset_by_max_context_length(
            test_dataset,
            "test dataset",
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=data_batch_size_per_gpu,
            shuffle=True,
            **dataloader_kwargs,
            generator=generator,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=data_batch_size_per_gpu,
            shuffle=False,
            **dataloader_kwargs,
            generator=generator,
        )
        self.dataset_size = {
            "train": len(train_dataset),
            "test": len(test_dataset),
        }
        if self.accelerator is not None:
            self.train_dataloader = self.accelerator.prepare(self.train_dataloader)
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = self.train_dataloader_iter
        self.reset_called = False
        self.evaluation_mode = False
        self.num_epochs = 0

    @abstractmethod
    def reset(
        self,
        reset_dataloaders: bool = False,
    ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
        """Reset the environment and get the next batch of tokenized prompts."""

    @abstractmethod
    def step(
        self,
        completions: torch.Tensor,
    ) -> tuple[list[ReasoningPrompts], torch.Tensor]:
        """Take a step in a HuggingFaceGym environment, calculate rewards from completions generated from previous prompt and provide new batch
        of prompts.
        """

    @contextmanager
    def eval_mode(self) -> Generator[None, None, None]:
        """Context manager to switch to evaluation mode."""
        self.dataloader = self.test_dataloader_iter
        self.evaluation_mode = True
        last_tokenized_prompts = None
        if hasattr(self, "last_tokenized_prompts"):
            last_tokenized_prompts = copy.deepcopy(self.last_tokenized_prompts)
        try:
            yield
        finally:
            self.dataloader = self.train_dataloader_iter
            self.evaluation_mode = False
            if last_tokenized_prompts is not None:
                self.last_tokenized_prompts = last_tokenized_prompts

    @abstractmethod
    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function that applies the chat template to the batch of questions and answers."""

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.evaluation_mode:
            return len(self.test_dataloader.dataset)
        return len(self.train_dataloader.dataset)

    def _reset_dataloaders(
        self, reset_train: bool = True, reset_test: bool = True
    ) -> None:
        """Reset the dataloaders to the beginning of the dataset.

        :param reset_train: Whether to reset the train dataloader, defaults to True
        :type reset_train: bool, optional
        :param reset_test: Whether to reset the test dataloader, defaults to True
        :type reset_test: bool, optional
        """
        if reset_train:
            self.train_dataloader_iter = iter(self.train_dataloader)
        if reset_test:
            self.test_dataloader_iter = iter(self.test_dataloader)
        self.dataloader = (
            self.test_dataloader_iter
            if self.evaluation_mode
            else self.train_dataloader_iter
        )

    def _filter_dataset_by_max_context_length(
        self,
        dataset: Dataset,
        dataset_type: str | None = None,
    ) -> Dataset:
        """Filter the dataset by the max context length.

        :param dataset: Dataset to be filtered.
        :type dataset: Dataset
        :return: Filtered train and test datasets.
        :rtype: tuple[Dataset, Dataset]
        """
        dataset_type = "dataset" if dataset_type is None else dataset_type
        filter_keyword = "prompt" if "prompt" in dataset.features else "question"
        if self.max_context_length is None or not isinstance(
            dataset[0][filter_keyword],
            str,
        ):
            return dataset
        filtered_dataset = dataset.filter(
            lambda x: (
                len(self.tokenizer.encode(x[filter_keyword]))
                <= self.max_context_length - self.min_completion_length
            ),
        )
        if len(filtered_dataset) == 0:
            msg = f"No samples left in the {dataset_type} after filtering by the max context length constraint, use a larger max context length."
            raise ValueError(
                msg,
            )
        if (dataset_difference := len(dataset) - len(filtered_dataset)) > 0:
            warnings.warn(
                f"{dataset_difference} samples were filtered out of the {dataset_type} due to the max context length constraint.",
                stacklevel=2,
            )
        return filtered_dataset


class ReasoningGym(HuggingFaceGym):
    """Class to convert HuggingFace datasets into Gymnasium style environment.

    :param train_dataset: Train dataset to be loaded from HuggingFace datasets.
    :type train_dataset: Dataset
    :param test_dataset: Test dataset to be loaded from HuggingFace datasets.
    :type test_dataset: Dataset
    :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
    :type tokenizer: AutoTokenizer
    :param reward_fn: Reward function for evaluating completions.
    :type reward_fn: Callable[..., float]
    :param conversation_template: A structured conversation that acts as a base pattern for each data point.
    :type conversation_template: list[dict[str, str]]
    :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param accelerator: Accelerator to be used for training, defaults to None
    :type accelerator: Accelerator, optional
    :param max_context_length: Maximum context length, defaults to None
    :type max_context_length: int | None, optional
    :param min_completion_length: Minimum completion length, defaults to 128
    :type min_completion_length: int, optional
    :param seed: Seed for the random number generator, defaults to 42
    :type seed: int, optional
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        reward_fn: Callable[[str, str, str], float],
        conversation_template: list[dict[str, str]],
        data_batch_size_per_gpu: int = 8,
        accelerator: Accelerator | None = None,
        return_raw_completions: bool = False,
        max_context_length: int | None = None,
        seed: int = 42,
    ) -> None:
        assert {"question", "answer"}.issubset(
            set(train_dataset.features.keys()),
        ), "Train dataset must contain 'question' and 'answer' features."
        assert {"question", "answer"}.issubset(
            set(test_dataset.features.keys()),
        ), "Train dataset must contain 'question' and 'answer' features."

        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            conversation_template=conversation_template,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=0,
            accelerator=accelerator,
            seed=seed,
        )
        self.reward_fn = reward_fn
        self.return_raw_completions = return_raw_completions

    def step(
        self,
        completions: torch.Tensor,
    ) -> tuple[list[ReasoningPrompts], torch.Tensor]:
        """Take a step in the ReasoningGym environment, calculate rewards from completions generated from previous prompt and provide new batch
        of prompts.

        :param completions: Completion IDs generated by the agent.
        :type completions: torch.Tensor
        :return: New tokenized prompts and an information dictionary.
        :rtype: tuple[list[BatchEncoding], torch.Tensor]
        """
        self.reset_called = False
        rewards = self._decode_and_evaluate(completions)
        new_tokenized_prompts = self._get_next_batch()
        self.last_tokenized_prompts = new_tokenized_prompts
        return new_tokenized_prompts, rewards

    def reset(
        self,
        reset_dataloaders: bool = False,
    ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
        """Reset the environment and get the next batch of tokenized prompts.

        :param reset_dataloaders: Whether to reset the dataloaders, defaults to False
        :type reset_dataloaders: bool, optional
        :return: New tokenized prompts
        :rtype: tuple[list[BatchEncoding], dict[str, Any]]
        """
        if reset_dataloaders:
            self._reset_dataloaders()
            warnings.warn(
                "env.reset() called with reset_dataloaders=True, this will reset the dataloaders to the beginning of the dataset, proceed with caution.",
                stacklevel=2,
            )
        if self.reset_called:
            warnings.warn(
                "env.reset() called more than once sequentially, it should typically follow with env.step().",
                stacklevel=2,
            )
        self.reset_called = True
        new_tokenized_prompts = self._get_next_batch()
        self.last_tokenized_prompts = new_tokenized_prompts
        return new_tokenized_prompts

    def _decode_and_evaluate(self, completions: list[torch.Tensor]) -> torch.Tensor:
        """Decode the completions and evaluate the rewards.

        :param completions: Completion IDs generated by the agent.
        :type completions: list[torch.Tensor]
        :return: Rewards for the completions.
        :rtype: torch.Tensor
        """
        # This is for a batch of completions (prompt_batch x group_size), List of tensors of length batch size, each tensor is a group of answers
        total_rewards = []
        for idx, (group_completion, answer, question) in enumerate(
            zip(completions, self.answers, self.questions, strict=False),
        ):
            completion_to_decode = group_completion[
                :,
                self.last_tokenized_prompts[idx]["input_ids"].shape[1] :,
            ]

            # Vectorize this in the future
            decoded_group_completion = self.tokenizer.batch_decode(
                completion_to_decode,
                skip_special_tokens=True,
            )
            rewards = [
                self.reward_fn(completion, answer, question)
                for completion in decoded_group_completion
            ]
            total_rewards.append(rewards)
        return torch.tensor(total_rewards)

    def _get_next_batch(self) -> list[ReasoningPrompts]:
        """Get the next batch of tokenized prompts."""
        try:
            batch = next(self.dataloader)
            self.questions = batch["question"]
            self.answers = batch["answer"]

            returned_prompts = [
                {
                    "input_ids": returned_prompt["input_ids"],
                    "attention_mask": returned_prompt["attention_mask"],
                    "text": (
                        self.tokenizer.batch_decode(
                            returned_prompt["input_ids"],
                            skip_special_tokens=False,  # Needs to be False here as we need to provide context about user roles to the model
                            clean_up_tokenization_spaces=False,
                        )[0]
                        if self.return_raw_completions
                        else None
                    ),
                }
                for returned_prompt in batch["tokenized_prompts"]
            ]
        except StopIteration:
            if not self.evaluation_mode:
                self.num_epochs += 1

            self._reset_dataloaders(
                reset_train=not self.evaluation_mode,
                reset_test=self.evaluation_mode,
            )
            return self._get_next_batch()
        return returned_prompts

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function that applies the chat template to the batch of questions and answers.

        :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
        :type tokenizer: AutoTokenizer
        :return: Collate function that applies the chat template to the batch of questions and answers.
        :rtype: Callable[[list[dict[str, Any]]], dict[str, Any]]
        """

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]

            # Apply chat template to all samples
            tokenized_prompts = [
                apply_chat_template(self.conversation_template, q, a, tokenizer)
                for q, a in zip(questions, answers, strict=False)
            ]

            return {
                "question": questions,
                "answer": answers,
                "tokenized_prompts": tokenized_prompts,  # Keep individual tokenized prompts
            }

        return collate_fn


class PreferenceGym(HuggingFaceGym):
    """Class to convert HuggingFace preference datasets into Gymnasium style environment.

    :param dataset_name: Dataset name to be loaded from HuggingFace datasets.
    :type dataset_name: str
    :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
    :type tokenizer: AutoTokenizer
    :param reward_fn: Reward function for evaluating completions.
    :type reward_fn: Callable[..., float]
    :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param custom_collate_fn: Custom collate function to be used for creating the batch, defaults to None
    :type custom_collate_fn: Callable, optional
    :param accelerator: Accelerator to be used for training, defaults to None
    :type accelerator: Accelerator, optional
    :param max_context_length: Maximum context length, defaults to None
    :type max_context_length: int | None, optional
    :param min_completion_length: Minimum completion length, defaults to 128
    :type min_completion_length: int, optional
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        data_batch_size_per_gpu: int = 8,
        accelerator: Accelerator | None = None,
        max_context_length: int | None = None,
        min_completion_length: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            conversation_template=None,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=min_completion_length,
            accelerator=accelerator,
            seed=seed,
        )
        assert {"prompt", "chosen", "rejected"}.issubset(
            set(train_dataset.features.keys()),
        ), "Train dataset must contain 'prompt', 'chosen', and 'rejected' features."
        assert {"prompt", "chosen", "rejected"}.issubset(
            set(test_dataset.features.keys()),
        ), "Train dataset must contain 'prompt', 'chosen', and 'rejected' features."

    def reset(self, reset_dataloaders: bool = False) -> PreferencePrompts:
        """Reset the environment and get the next batch of tokenized prompts."""
        if reset_dataloaders:
            self._reset_dataloaders()
            warnings.warn(
                "env.reset() called with reset_dataloaders=True, this will reset the dataloaders to the beginning of the dataset, proceed with caution.",
                stacklevel=2,
            )
        if self.reset_called:
            warnings.warn(
                "env.reset() called more than once sequentially, it should typically follow with env.step().",
                stacklevel=2,
            )
        self.reset_called = True
        return self._get_next_batch()

    def step(
        self,
        completions: torch.Tensor | None = None,
    ) -> PreferencePrompts:
        """Take a step in the PreferenceGym environment, calculate rewards from completions generated from previous prompt and provide new batch
        of prompts.

        :param completions: Completions from the model (unused in PreferenceGym; kept for signature compatibility with HuggingFaceGym).
        """
        self.reset_called = False
        return self._get_next_batch()

    def _get_next_batch(self) -> PreferencePrompts:
        try:
            batch = next(self.dataloader)
        except StopIteration:
            if not self.evaluation_mode:
                self.num_epochs += 1
            self._reset_dataloaders(
                reset_train=not self.evaluation_mode,
                reset_test=self.evaluation_mode,
            )
            return self._get_next_batch()
        return batch

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function that applies the chat template to the batch of questions and answers.

        :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
        :type tokenizer: AutoTokenizer
        :param max_context_length: Maximum context length, defaults to None
        :type max_context_length: int | None, optional
        :return: Collate function that applies the chat template to the batch of questions and answers.
        :rtype: Callable[[list[dict[str, Any]]], dict[str, Any]]
        """

        def collate_fn(batch: list[dict[str, str]]) -> dict[str, str]:
            prompts = [item["prompt"] for item in batch]
            chosen = [item["chosen"] for item in batch]
            rejected = [item["rejected"] for item in batch]

            # Tokenize prompts separately to get their lengths
            prompt_encodings = tokenizer(
                prompts,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

            # Tokenise chosen and rejected, padding both to the same length.
            # When max_context_length is set we pad directly to that fixed cap
            # (one __call__ each, no separate .pad() step).  When it is None we
            # need a first pass to find the joint max length before padding.
            if self.max_context_length is not None:
                chosen_enc = tokenizer(
                    prompts,
                    chosen,
                    max_length=self.max_context_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                rejected_enc = tokenizer(
                    prompts,
                    rejected,
                    max_length=self.max_context_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            else:
                # First pass: get lengths without padding to compute joint max
                chosen_ids = tokenizer(prompts, chosen, truncation=True, padding=False)
                rejected_ids = tokenizer(
                    prompts, rejected, truncation=True, padding=False
                )
                max_len = max(
                    *(len(ids) for ids in chosen_ids["input_ids"]),
                    *(len(ids) for ids in rejected_ids["input_ids"]),
                )
                # Second pass: pad both to the joint max in a single __call__
                chosen_enc = tokenizer(
                    prompts,
                    chosen,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt",
                )
                rejected_enc = tokenizer(
                    prompts,
                    rejected,
                    truncation=True,
                    padding="max_length",
                    max_length=max_len,
                    return_tensors="pt",
                )

            return {
                "prompt": prompts,
                "prompt_lengths": prompt_lengths,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_input_ids": chosen_enc["input_ids"],  # [batch_size, max_len]
                "chosen_attention_mask": chosen_enc[
                    "attention_mask"
                ].long(),  # [batch_size, max_len]
                "rejected_input_ids": rejected_enc[
                    "input_ids"
                ],  # [batch_size, max_len]
                "rejected_attention_mask": rejected_enc[
                    "attention_mask"
                ].long(),  # [batch_size, max_len]
            }

        return collate_fn


class SFTGym(HuggingFaceGym):
    """Gymnasium-style environment for supervised fine-tuning (SFT) datasets.

    Each batch yields tokenised ``(prompt + response)`` pairs that can be
    consumed directly by :class:`~agilerl.algorithms.sft.SFT`.  The loss is
    computed on the response tokens only; prompt and padding positions are
    masked out inside :meth:`~agilerl.algorithms.sft.SFT.learn`.

    The dataset must have a ``"prompt"`` column and a response column (name
    set by ``response_column``, defaults to ``"response"``).  No
    negative/rejected response column is required — SFT only needs the target
    good response.

    .. tip::
        If your dataset uses DPO-style column names (``"chosen"`` /
        ``"rejected"``), pass ``response_column="chosen"`` to use it for SFT
        warm-up before a subsequent DPO fine-tuning stage.

    :param train_dataset: HuggingFace ``Dataset`` split used during training.
    :type train_dataset: Dataset
    :param test_dataset: HuggingFace ``Dataset`` split used for evaluation.
    :type test_dataset: Dataset
    :param tokenizer: Tokenizer for the target model.
    :type tokenizer: AutoTokenizer
    :param data_batch_size_per_gpu: DataLoader batch size per GPU, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param response_column: Name of the dataset column containing the target
        response, defaults to ``"response"``
    :type response_column: str, optional
    :param accelerator: Accelerate handle for distributed training, defaults to
        None
    :type accelerator: Accelerator, optional
    :param max_context_length: Maximum tokenised sequence length.  Sequences
        longer than this are truncated, defaults to None
    :type max_context_length: int | None, optional
    :param seed: Random seed for dataset shuffling, defaults to 42
    :type seed: int, optional
    """

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: AutoTokenizer,
        data_batch_size_per_gpu: int = 8,
        response_column: str = "response",
        accelerator: Accelerator | None = None,
        max_context_length: int | None = None,
        seed: int = 42,
    ) -> None:
        self.response_column = response_column
        required = {"prompt", response_column}
        assert required.issubset(set(train_dataset.features.keys())), (
            f"Train dataset must contain {required} features."
        )
        assert required.issubset(set(test_dataset.features.keys())), (
            f"Test dataset must contain {required} features."
        )
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            conversation_template=None,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=None,
            accelerator=accelerator,
            seed=seed,
        )

    def reset(self, reset_dataloaders: bool = False) -> SFTPrompts:
        """Reset the environment and return the first batch of tokenised data."""
        if reset_dataloaders:
            self._reset_dataloaders()
            import warnings

            warnings.warn(
                "env.reset() called with reset_dataloaders=True; dataloaders are "
                "rewound to the beginning of the dataset.",
                stacklevel=2,
            )
        if self.reset_called:
            import warnings

            warnings.warn(
                "env.reset() called more than once sequentially; it should be "
                "followed by env.step().",
                stacklevel=2,
            )
        self.reset_called = True
        return self._get_next_batch()

    def step(
        self,
        completions: torch.Tensor | None = None,
    ) -> SFTPrompts:
        """Advance the data iterator and return the next batch.

        :param completions: Unused; kept for API compatibility with other Gym types.
        """
        self.reset_called = False
        return self._get_next_batch()

    def _get_next_batch(self) -> SFTPrompts:
        try:
            batch = next(self.dataloader)
        except StopIteration:
            if not self.evaluation_mode:
                self.num_epochs += 1
            self._reset_dataloaders(
                reset_train=not self.evaluation_mode,
                reset_test=self.evaluation_mode,
            )
            return self._get_next_batch()
        return batch

    def create_collate_fn(
        self,
        tokenizer: AutoTokenizer,
        max_context_length: int | None = None,
    ) -> Any:
        """Build a collate function that tokenises ``(prompt, response)`` pairs.

        :param tokenizer: Tokenizer to use.
        :type tokenizer: AutoTokenizer
        :param max_context_length: Truncation length, defaults to None
        :type max_context_length: int | None, optional
        :return: Collate function.
        :rtype: Callable
        """
        response_column = self.response_column

        def collate_fn(batch: list[dict[str, Any]]) -> SFTPrompts:
            prompts = [item["prompt"] for item in batch]
            responses = [item[response_column] for item in batch]

            # Tokenise prompts alone to measure prompt lengths (for loss masking)
            prompt_encodings = tokenizer(
                prompts,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            prompt_lengths = [len(ids) for ids in prompt_encodings["input_ids"]]

            # Tokenise prompt + response together in one __call__ to avoid the
            # "encode then pad" performance warning from fast tokenizers.
            # Use self.max_context_length (the instance attribute) rather than
            # the create_collate_fn parameter, which is never passed by the
            # base class and would silently default to None.
            pair_enc = tokenizer(
                prompts,
                responses,
                max_length=self.max_context_length,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            )

            return {
                "prompt": prompts,
                "prompt_lengths": prompt_lengths,
                "response": responses,
                "input_ids": pair_enc["input_ids"],  # [B, max_len]
                "attention_mask": pair_enc["attention_mask"].long(),  # [B, max_len]
            }

        return collate_fn


@contextmanager
def gather_if_zero3(
    zero_stage: int,
    params: list[torch.Tensor],
    modifier_rank: int | None = None,
) -> Generator[None, None, None]:
    """Conditional context manager for setting the zero stage for the model.

    :param zero_stage: The zero stage
    :type zero_stage: int
    :param params: The parameters to gather
    :type params: list[torch.Tensor]
    :param modifier_rank: The modifier rank
    :type modifier_rank: int | None
    """
    if zero_stage == 3:
        if deepspeed is None:
            msg = (
                "DeepSpeed is required for ZeRO stage 3 parameter gathering, but it "
                "is not installed."
            )
            raise ImportError(msg)
        with deepspeed.zero.GatheredParameters(
            params=params,
            modifier_rank=modifier_rank,
        ):
            yield
    else:
        yield


def get_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Get the state dict of the model for zero3.

    :param model: The model to get the state dict of.
    :type model: nn.Module
    :return: The state dict of the model.
    :rtype: dict[str, torch.Tensor]
    """
    with gather_if_zero3(3, list(model.parameters()), modifier_rank=0):
        return model.state_dict()


def create_model_from_name_or_path(
    model_name_or_path: str,
    model_config: dict[str, Any] | None = None,
) -> PreTrainedModel:
    """Create a model from a name or path.

    :param model_name_or_path: The name or path of the model to create.
    :type model_name_or_path: str
    :param model_config: The configuration of the model to create.
    :type model_config: dict[str, Any ] | None
    :return: The created model.
    :rtype: PreTrainedModel
    """
    if model_config is None:
        model_config = {
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_config,
    )


def sample_eval_prompts(
    env: Any, n: int = 5, seed: int = 0
) -> list[tuple[str, str | None, str | None]]:
    """Randomly sample *n* ``(prompt, chosen, rejected)`` triples from
    *env*'s held-out test dataset.

    Columns are resolved automatically per gym type:

    * :class:`SFTGym` — ``chosen`` is ``env.response_column``; ``rejected``
      is ``None`` (SFT has no negative example).
    * :class:`PreferenceGym` — ``chosen`` and ``rejected`` map to the
      dataset's ``"chosen"`` / ``"rejected"`` columns.
    * Any other gym — both are ``None``.

    :param env: AgileRL gym environment with a ``test_dataloader`` attribute.
    :param n: Number of samples to draw, defaults to 5.
    :type n: int, optional
    :param seed: Random seed for reproducible sampling, defaults to 0.
    :type seed: int, optional
    :return: List of ``(prompt, chosen, rejected)`` tuples; unused fields are
        ``None``.
    :rtype: list[tuple[str, str | None, str | None]]
    """
    dataset = env.test_dataloader.dataset
    indices = random.Random(seed).sample(range(len(dataset)), min(n, len(dataset)))

    chosen_col: str | None = None
    rejected_col: str | None = None
    if hasattr(env, "response_column"):  # SFTGym
        chosen_col = env.response_column
    elif "chosen" in dataset.features:  # PreferenceGym
        chosen_col = "chosen"
        rejected_col = "rejected"

    return [
        (
            dataset[i]["prompt"],
            dataset[i][chosen_col] if chosen_col else None,
            dataset[i][rejected_col] if rejected_col else None,
        )
        for i in indices
    ]


def compare_responses(
    agent: Any,
    tokenizer: Any,
    samples: list[tuple[str, str | None, str | None]],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    do_sample: bool = False,
    skip_special_tokens: bool = True,
    show_base_model: bool = True,
) -> None:
    """Run each prompt through the base model and the fine-tuned LoRA model,
    printing a formatted comparison to the terminal one sample at a time.

    After each sample the user is prompted to press **Enter** to continue or
    **q + Enter** to quit early.  Intended to be called at the end of a
    training script for a quick qualitative sanity-check.

    Works with any LoRA-adapted
    :class:`~agilerl.algorithms.core.base.LLMAlgorithm` (``SFT``, ``DPO``,
    …).  When the model has no LoRA adapter the base-model column is omitted
    and only the current model's output is shown.

    :param agent: Trained AgileRL LLM agent exposing ``agent.actor`` and
        ``agent.device``.
    :type agent: LLMAlgorithm
    :param tokenizer: HuggingFace tokenizer matching the model.
    :param samples: ``(prompt, chosen, rejected)`` triples as returned by
        :func:`sample_eval_prompts`.  ``None`` fields are silently skipped.
    :type samples: list[tuple[str, str | None, str | None]]
    :param max_new_tokens: Maximum tokens to generate per response, defaults
        to 200.
    :type max_new_tokens: int, optional
    :param temperature: Sampling temperature, defaults to 1.0.
    :type temperature: float, optional
    :param do_sample: Use sampling instead of greedy decoding, defaults to
        False.  Set ``True`` together with a ``temperature`` != 1.0 for
        stochastic outputs.
    :type do_sample: bool, optional
    :param skip_special_tokens: Strip special tokens when decoding, defaults
        to True.
    :type skip_special_tokens: bool, optional
    :param show_base_model: If ``False``, skip the base-model generation block
        (only the current model output is printed).  Useful when the adapter is
        merged or base vs. adapter outputs are identical.
    :type show_base_model: bool, optional
    """
    model = agent.actor
    device = agent.device
    width = min(shutil.get_terminal_size(fallback=(100, 40)).columns, 120)
    divider = "─" * width
    has_adapter = hasattr(model, "disable_adapter")

    def _generate(prompt_text: str, *, use_base: bool) -> str:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs: dict[str, Any] = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        model.eval()
        with torch.no_grad():
            if use_base and has_adapter:
                with model.disable_adapter():
                    output_ids = model.generate(**gen_kwargs)
            else:
                output_ids = model.generate(**gen_kwargs)
        new_tokens = output_ids[0][prompt_len:]
        return tokenizer.decode(
            new_tokens, skip_special_tokens=skip_special_tokens
        ).strip()

    def _wrap(text: str, indent: int = 2) -> str:
        prefix = " " * indent
        return textwrap.fill(
            text,
            width=width - indent,
            initial_indent=prefix,
            subsequent_indent=prefix,
        )

    total = len(samples)
    for i, (prompt, chosen, rejected) in enumerate(samples, 1):
        header = f"  SAMPLE {i} / {total}  "
        padding = max(0, width - len(header))
        left_pad = padding // 2
        right_pad = padding - left_pad
        print(f"\n{'═' * left_pad}{header}{'═' * right_pad}")

        print(f"\nPROMPT\n{divider}")
        print(_wrap(prompt))

        if chosen is not None:
            print(f"\nDATASET RESPONSE (CHOSEN)\n{divider}")
            print(_wrap(chosen))

        if rejected is not None:
            print(f"\nDATASET RESPONSE (REJECTED)\n{divider}")
            print(_wrap(rejected))

        if has_adapter and show_base_model:
            print(f"\nBASE MODEL\n{divider}")
            print(_wrap(_generate(prompt, use_base=True)))

        label = "FINE-TUNED MODEL" if has_adapter else "MODEL RESPONSE"
        print(f"\n{label}\n{divider}")
        print(_wrap(_generate(prompt, use_base=False)))

        if i < total:
            nav = "  [Enter] next sample   [q + Enter] quit  "
            nav_padding = max(0, width - len(nav))
            print(
                f"\n{'─' * (nav_padding // 2)}{nav}{'─' * (nav_padding - nav_padding // 2)}"
            )
            try:
                if input().strip().lower() == "q":
                    break
            except EOFError:
                break

    print(f"\n{'═' * width}\n")
