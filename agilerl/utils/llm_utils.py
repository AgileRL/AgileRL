from __future__ import annotations

import copy
import logging
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

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.typing import PreferencePrompts, ReasoningPrompts

logger = logging.getLogger(__name__)

if HAS_LLM_DEPENDENCIES:
    import deepspeed
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import BatchEncoding

    from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead

    AutoTokenizer = AutoTokenizer
else:
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
    """Create and tokenize a chat template for a reaosning task.

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


def max_prompt_tokens_for_sliding_window(
    max_model_len: int,
    max_output_tokens: int | None,
) -> int:
    """Upper bound on prompt tokens so at least one completion token can be generated.

    Matches the headroom logic used by vLLM colocate generation: reserve
    ``min(max_output_cap, max_model_len)`` with at least one token reserved for
    generation when room exists.

    :param max_model_len: Engine context length (prompt + completion ceiling).
    :type max_model_len: int
    :param max_output_tokens: Configured completion cap; if ``None``, uses
        ``max_model_len`` as the cap when computing headroom.
    :type max_output_tokens: int | None
    :return: Largest allowed prompt length under that headroom (may be 0).
    :rtype: int
    """
    cap = max_output_tokens if max_output_tokens is not None else max_model_len
    gen_reserve = max(1, min(cap, max_model_len))
    return max(0, max_model_len - gen_reserve)


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
            if hasattr(self, "last_tokenized_prompts"):
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

    def step(self) -> PreferencePrompts:
        """Take a step in the PreferenceGym environment, calculate rewards from completions generated from previous prompt and provide new batch
        of prompts.
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

            # Tokenize without padding
            chosen_enc = tokenizer(
                prompts,
                chosen,
                max_length=self.max_context_length,
                truncation=True,
                padding=False,
            )
            rejected_enc = tokenizer(
                prompts,
                rejected,
                max_length=self.max_context_length,
                truncation=True,
                padding=False,
            )

            # Compute the joint max length across both
            max_len = max(
                *(len(ids) for ids in chosen_enc["input_ids"]),
                *(len(ids) for ids in rejected_enc["input_ids"]),
            )

            max_len = (
                min(max_len, self.max_context_length)
                if self.max_context_length is not None
                else max_len
            )

            # Now pad both encodings to the same target length
            chosen_enc = tokenizer.pad(
                chosen_enc,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )
            rejected_enc = tokenizer.pad(
                rejected_enc,
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
    add_value_head: bool = False,
) -> PreTrainedModel:
    """Create a model from a name or path.

    :param model_name_or_path: The name or path of the model to create.
    :type model_name_or_path: str
    :param model_config: The configuration of the model to create.
    :type model_config: dict[str, Any ] | None
    :param use_value_head: Flag to indicate if a value head should be added to the model, defaults to False
    :type use_value_head: bool, optional
    :return: The created model.
    :rtype: PreTrainedModel
    """
    if model_config is None:
        model_config = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
    print("model_config", model_config)
    if add_value_head:
        return AutoModelForCausalLMWithValueHead.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **model_config,
        )
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_config,
    )


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: bool | None = None
) -> torch.Tensor:
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    return (values * mask).sum() / mask.sum()


def masked_var(
    values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True
) -> torch.Tensor:
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            msg = (
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
            raise ValueError(msg)
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(
    values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True
) -> torch.Tensor:
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def pool_by_turns(
    token_values: torch.Tensor,
    turn_ids: torch.Tensor,
    num_turns: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Aggregate per-token values into per-turn scalars.

    :param token_values: [batch, seq_len] per-token scalars.
    :param turn_ids: [batch, seq_len] turn index per token, -1 for non-action.
    :param num_turns: Total number of turns (max turn_id + 1).
    :param reduction: ``"mean"`` (default) for mean-pooling,
        ``"sum"`` for sum-pooling (e.g. to aggregate log-ratios).
    :return: [batch, num_turns] aggregated values per turn.
    """
    batch_size = token_values.shape[0]
    turn_values = torch.zeros(batch_size, num_turns, device=token_values.device)
    for t in range(num_turns):
        mask_t = (turn_ids == t).float()
        summed = (token_values * mask_t).sum(dim=1)
        if reduction == "mean":
            count = mask_t.sum(dim=1).clamp(min=1)
            turn_values[:, t] = summed / count
        else:
            turn_values[:, t] = summed
    return turn_values


def create_llm_accelerator(
    *,
    model_size_gb: float | None = None,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    gradient_clipping: float = 1.0,
    mixed_precision: str = "bf16",
    offload_optimizer: bool = False,
    zero_stage: int | None = None,
) -> Accelerator | None:
    """Create an :class:`Accelerator` with sensible DeepSpeed defaults.

    Detects the number of available GPUs and picks an appropriate
    configuration automatically:

    * **0 GPUs** — returns ``None`` (the ``accelerator=None`` code-path
      in :class:`~agilerl.algorithms.core.base.LLMAlgorithm` handles
      CPU-only training).
    * **1 GPU** — returns a plain ``Accelerator`` with no DeepSpeed
      (avoids ZeRO partitioning overhead on a single device).
    * **2+ GPUs** — returns an ``Accelerator`` with a ``DeepSpeedPlugin``
      whose ZeRO stage is chosen based on ``model_size_gb`` and the
      per-GPU memory available.

    The ``zero_stage`` argument overrides automatic selection when the
    caller knows exactly what they want.

    :param model_size_gb: Approximate model size in GB (parameters in
        fp16/bf16).  Used to pick the ZeRO stage on multi-GPU setups.
        When ``None``, defaults to ZeRO-1.
    :param micro_batch_size: Per-GPU micro-batch size written into the
        DeepSpeed config.
    :param gradient_accumulation_steps: Gradient accumulation steps
        written into the DeepSpeed config.
    :param gradient_clipping: Maximum gradient norm.
    :param mixed_precision: Mixed-precision mode for Accelerator
        (``'bf16'``, ``'fp16'``, or ``'no'``).
    :param offload_optimizer: If ``True``, offload optimizer states to
        CPU in the DeepSpeed config.
    :param zero_stage: Explicit ZeRO stage override.  When set, the
        automatic GPU-count / model-size heuristic is skipped.
    :return: A configured ``Accelerator``, or ``None`` when no GPU is
        available.

    Example::

        from agilerl.utils.llm_utils import create_llm_accelerator

        # Auto-detect: 1 GPU → no DeepSpeed, 2+ → ZeRO-1
        accelerator = create_llm_accelerator()

        # Hint for a large model on multi-GPU
        accelerator = create_llm_accelerator(model_size_gb=14.0)

        # Explicit override
        accelerator = create_llm_accelerator(zero_stage=2, offload_optimizer=True)
    """
    from accelerate.utils import DeepSpeedPlugin

    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        logger.info("No GPUs detected — returning None (CPU-only path).")
        return None

    if num_gpus == 1 and zero_stage is None:
        logger.info("Single GPU detected — using plain Accelerator (no DeepSpeed).")
        return Accelerator(mixed_precision=mixed_precision)

    stage = (
        zero_stage
        if zero_stage is not None
        else _auto_zero_stage(num_gpus, model_size_gb)
    )
    logger.info(
        "Creating Accelerator with DeepSpeed ZeRO-%d for %d GPU(s).", stage, num_gpus
    )

    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "bf16": {"enabled": mixed_precision == "bf16"},
        "fp16": {"enabled": mixed_precision == "fp16"},
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": stage >= 2,
            "contiguous_gradients": True,
            "reduce_bucket_size": int(2e8),
        },
    }

    if offload_optimizer:
        ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu"}

    if stage >= 3:
        ds_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = (
            True
        )

    plugin = DeepSpeedPlugin(hf_ds_config=ds_config)
    return Accelerator(deepspeed_plugin=plugin, mixed_precision=mixed_precision)


def _auto_zero_stage(num_gpus: int, model_size_gb: float | None) -> int:
    """Pick a ZeRO stage based on GPU count and model size.

    Heuristic:

    * If ``model_size_gb`` is unknown, default to ZeRO-1 (lightest
      multi-GPU overhead, partitions only optimizer states).
    * If the model fits comfortably in per-GPU memory (< 60% of VRAM),
      use ZeRO-1.
    * If the model is tight but fits (60-90% of VRAM), use ZeRO-2
      (also partitions gradients).
    * If the model exceeds per-GPU memory, use ZeRO-3 (also partitions
      parameters).
    """
    if model_size_gb is None:
        return 1

    try:
        per_gpu_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    except Exception:
        return 1

    ratio = model_size_gb / per_gpu_gb
    if ratio < 0.6:
        return 1
    if ratio < 0.9:
        return 2
    return 3


def move_params_to_gpu(unwrapped_model: torch.nn.Module, device: torch.device) -> None:
    """Move params to GPU.

    :param agent: Distributed agent
    :type agent: DistributedLLMAgent
    :return: None
    :rtype: None
    """
    unwrapped_model.to(device, non_blocking=True)
    torch.cuda.synchronize()


def move_params_to_cpu(unwrapped_model: torch.nn.Module) -> None:
    """Move params to CPU.

    :param agent: Distributed agent
    :type agent: DistributedLLMAgent
    :return: None
    :rtype: None
    """
    unwrapped_model.to("cpu", non_blocking=True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
