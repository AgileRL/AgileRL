import copy
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Generator

import deepspeed
import gymnasium as gym
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from agilerl.typing import PreferencePrompts, ReasoningPrompts


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
    :param apply_chat_template_fn: Function to apply the chat template to the batch of questions and answers, defaults to None
    :type apply_chat_template_fn: Callable, optional
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
        custom_collate_fn: (
            Callable[[list[dict[str, Any]]], dict[str, Any]] | None
        ) = None,
        apply_chat_template_fn: (
            Callable[[str, str, AutoTokenizer], BatchEncoding] | None
        ) = None,
        data_batch_size_per_gpu: int = 8,
        max_context_length: int | None = None,
        min_completion_length: int = None,
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
        if custom_collate_fn is None:
            collate_kwargs = {"tokenizer": tokenizer}
            if apply_chat_template_fn is not None:
                collate_kwargs["apply_chat_template_fn"] = apply_chat_template_fn
            custom_collate_fn = self.create_collate_fn(**collate_kwargs)
        dataloader_kwargs = {"collate_fn": custom_collate_fn}
        train_dataset = self._filter_dataset_by_max_context_length(
            train_dataset, "train dataset"
        )
        test_dataset = self._filter_dataset_by_max_context_length(
            test_dataset, "test dataset"
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
        self.observation_space = gym.spaces.Box(low=0, high=tokenizer.vocab_size - 1)
        self.action_space = gym.spaces.Box(
            low=0,
            high=tokenizer.vocab_size - 1,
        )
        self.evaluation_mode = False
        self.num_epochs = 0

    @abstractmethod
    def reset(
        self,
        reset_dataloaders: bool = False,
    ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
        """Reset the environment and get the next batch of tokenized prompts."""
        pass

    @abstractmethod
    def step(
        self,
        completions: torch.Tensor,
    ) -> tuple[list[ReasoningPrompts], torch.Tensor]:
        """Take a step in a HuggingFaceGym environment, calculate rewards from completions generated from previous prompt and provide new batch
        of prompts.
        """
        pass

    @contextmanager
    def eval_mode(self) -> Generator[None, None, None]:
        """Context manager to switch to evaluation mode."""
        self.dataloader = self.test_dataloader_iter
        self.evaluation_mode = True
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
        *args,
        **kwargs,
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """Create a collate function that applies the chat template to the batch of questions and answers."""
        pass

    def __len__(self):
        """Return the length of the dataset."""
        if self.evaluation_mode:
            return len(self.test_dataloader.dataset)
        return len(self.train_dataloader.dataset)

    def _reset_dataloaders(self, reset_train: bool = True, reset_test: bool = True):
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
        """
        Filter the dataset by the max context length.

        :param dataset: Dataset to be filtered.
        :type dataset: Dataset
        :return: Filtered train and test datasets.
        :rtype: tuple[Dataset, Dataset]
        """
        dataset_type = "dataset" if dataset_type is None else dataset_type
        if self.max_context_length is None:
            return dataset
        filter_keyword = "prompt" if "prompt" in dataset.features.keys() else "question"
        filtered_dataset = dataset.filter(
            lambda x: len(self.tokenizer.encode(x[filter_keyword]))
            <= self.max_context_length - self.min_completion_length
        )
        if len(filtered_dataset) == 0:
            raise ValueError(
                f"No samples left in the {dataset_type} after filtering by the max context length constraint, use a larger max context length."
            )
        if (dataset_difference := len(dataset) - len(filtered_dataset)) > 0:
            warnings.warn(
                f"{dataset_difference} samples were filtered out of the {dataset_type} due to the max context length constraint."
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
    :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
    :type data_batch_size_per_gpu: int, optional
    :param custom_collate_fn: Custom collate fxwunction to be used for creating the batch, defaults to None
    :type custom_collate_fn: Callable, optional
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
        apply_chat_template_fn: Callable[[str, str, AutoTokenizer], BatchEncoding],
        data_batch_size_per_gpu: int = 8,
        custom_collate_fn: Callable | None = None,
        accelerator: Accelerator | None = None,
        return_raw_completions: bool = False,
        max_context_length: int | None = None,
        seed: int = 42,
    ) -> None:
        assert {"question", "answer"}.issubset(
            set(train_dataset.features.keys())
        ), "Train dataset must contain 'question' and 'answer' features."
        assert {"question", "answer"}.issubset(
            set(test_dataset.features.keys())
        ), "Train dataset must contain 'question' and 'answer' features."

        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            custom_collate_fn=custom_collate_fn,
            apply_chat_template_fn=apply_chat_template_fn,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=0,
            accelerator=accelerator,
            seed=seed,
        )
        self.reward_fn = reward_fn
        self.return_raw_completions = return_raw_completions

    def step(
        self, completions: torch.Tensor
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
        self, reset_dataloaders: bool = False
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
                "env.reset() called with reset_dataloaders=True, this will reset the dataloaders to the beginning of the dataset, proceed with caution."
            )
        if self.reset_called:
            warnings.warn(
                "env.reset() called more than once sequentially, it should typically follow with env.step()."
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
            zip(completions, self.answers, self.questions)
        ):
            completion_to_decode = group_completion[
                :, self.last_tokenized_prompts[idx]["input_ids"].shape[1] :
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
        apply_chat_template_fn: Callable[[str, str, AutoTokenizer], BatchEncoding],
    ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
        """
        Create a collate function that applies the chat template to the batch of questions and answers.

        :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
        :type tokenizer: AutoTokenizer
        :param apply_chat_template_fn: Function to apply the chat template to the batch of questions and answers.
        :type apply_chat_template_fn: Callable[[str, str, AutoTokenizer], BatchEncoding]
        :return: Collate function that applies the chat template to the batch of questions and answers.
        :rtype: Callable[[list[dict[str, Any]]], dict[str, Any]]
        """

        def collate_fn(batch):

            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]

            # Apply chat template to all samples
            tokenized_prompts = [
                apply_chat_template_fn(q, a, tokenizer)
                for q, a in zip(questions, answers)
            ]

            return {
                "question": questions,
                "answer": answers,
                "tokenized_prompts": tokenized_prompts,  # Keep individual tokenized prompts
            }

        return collate_fn


# class UpdatedReasoningGym(HuggingFaceGym):
#     """Updated ReasoningGym with efficient batch collation and individual reward component reporting.

#     :param train_dataset: Train dataset to be loaded from HuggingFace datasets.
#     :type train_dataset: Dataset
#     :param test_dataset: Test dataset to be loaded from HuggingFace datasets.
#     :type test_dataset: Dataset
#     :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
#     :type tokenizer: AutoTokenizer
#     :param reward_fn: Reward function for evaluating completions. Should return either a float or dict with 'total' key.
#     :type reward_fn: Callable[[str, str, str], float | dict[str, float]]
#     :param data_batch_size_per_gpu: DataLoader batch size, defaults to 8
#     :type data_batch_size_per_gpu: int, optional
#     :param custom_collate_fn: Custom collate function to be used for creating the batch, defaults to None
#     :type custom_collate_fn: Callable, optional
#     :param accelerator: Accelerator to be used for training, defaults to None
#     :type accelerator: Accelerator, optional
#     :param max_context_length: Maximum context length, defaults to None
#     :type max_context_length: int | None, optional
#     :param min_completion_length: Minimum completion length, defaults to 0
#     :type min_completion_length: int, optional
#     :param seed: Seed for the random number generator, defaults to 42
#     :type seed: int, optional
#     :param return_raw_completions: Whether to return raw completion text, defaults to False
#     :type return_raw_completions: bool, optional
#     """

#     def __init__(
#         self,
#         train_dataset: Dataset,
#         test_dataset: Dataset,
#         tokenizer: AutoTokenizer,
#         reward_fn: Callable[[str, str, str], float | dict[str, float]],
#         apply_chat_template_fn: Callable[[str, str, AutoTokenizer], BatchEncoding],
#         data_batch_size_per_gpu: int = 8,
#         custom_collate_fn: Callable | None = None,
#         accelerator: Accelerator | None = None,
#         return_raw_completions: bool = False,
#         max_context_length: int | None = None,
#         seed: int = 42,
#     ) -> None:
#         assert {"question", "answer"}.issubset(
#             set(train_dataset.features.keys())
#         ), "Train dataset must contain 'question' and 'answer' features."
#         assert {"question", "answer"}.issubset(
#             set(test_dataset.features.keys())
#         ), "Test dataset must contain 'question' and 'answer' features."

#         super().__init__(
#             train_dataset=train_dataset,
#             test_dataset=test_dataset,
#             tokenizer=tokenizer,
#             custom_collate_fn=custom_collate_fn,
#             apply_chat_template_fn=apply_chat_template_fn,
#             data_batch_size_per_gpu=data_batch_size_per_gpu,
#             max_context_length=max_context_length,
#             min_completion_length=0,
#             accelerator=accelerator,
#             seed=seed,
#         )
#         self.reward_fn = reward_fn
#         self.return_raw_completions = return_raw_completions
#         self.apply_chat_template_fn = apply_chat_template_fn

#     def step(
#         self, completions: torch.Tensor
#     ) -> tuple[list[ReasoningPrompts], dict[str, torch.Tensor]]:
#         """Take a step in the UpdatedReasoningGym environment, calculate rewards from completions
#         and provide new batch of prompts.

#         :param completions: Completion IDs generated by the agent.
#         :type completions: torch.Tensor
#         :return: New tokenized prompts and a dict of reward tensors (including 'total' and individual components).
#         :rtype: tuple[list[ReasoningPrompts], dict[str, torch.Tensor]]
#         """
#         self.reset_called = False
#         rewards_dict = self._decode_and_evaluate(completions)
#         new_tokenized_prompts = self._get_next_batch()
#         self.last_tokenized_prompts = new_tokenized_prompts
#         return new_tokenized_prompts, rewards_dict

#     def reset(
#         self, reset_dataloaders: bool = False
#     ) -> tuple[list[ReasoningPrompts], dict[str, Any]]:
#         """Reset the environment and get the next batch of tokenized prompts.

#         :param reset_dataloaders: Whether to reset the dataloaders, defaults to False
#         :type reset_dataloaders: bool, optional
#         :return: New tokenized prompts
#         :rtype: tuple[list[ReasoningPrompts], dict[str, Any]]
#         """
#         if reset_dataloaders:
#             self._reset_dataloaders()
#             warnings.warn(
#                 "env.reset() called with reset_dataloaders=True, this will reset the dataloaders to the beginning of the dataset, proceed with caution."
#             )
#         if self.reset_called:
#             warnings.warn(
#                 "env.reset() called more than once sequentially, it should typically follow with env.step()."
#             )
#         self.reset_called = True
#         new_tokenized_prompts = self._get_next_batch()
#         self.last_tokenized_prompts = new_tokenized_prompts
#         return new_tokenized_prompts

#     def _decode_and_evaluate(
#         self, completions: list[torch.Tensor]
#     ) -> dict[str, torch.Tensor]:
#         """Decode the completions and evaluate the rewards with individual component tracking.

#         :param completions: Completion IDs generated by the agent.
#         :type completions: list[torch.Tensor]
#         :return: Dictionary containing 'total' rewards and individual reward components.
#         :rtype: dict[str, torch.Tensor]
#         """
#         # This is for a batch of completions (prompt_batch x group_size)
#         total_rewards = []
#         reward_components = {}  # Track individual reward components

#         for idx, (group_completion, answer, question) in enumerate(
#             zip(completions, self.answers, self.questions)
#         ):
#             completion_to_decode = group_completion[
#                 :, self.last_tokenized_prompts[idx]["input_ids"].shape[1] :
#             ]

#             # Batch decode all completions in the group at once
#             decoded_group_completion = self.tokenizer.batch_decode(
#                 completion_to_decode,
#                 skip_special_tokens=True,
#             )

#             # Evaluate each completion in the group
#             group_total_rewards = []
#             group_component_rewards = {}

#             for completion in decoded_group_completion:
#                 reward_result = self.reward_fn(completion, answer, question)

#                 # Handle both scalar and dict reward functions
#                 if isinstance(reward_result, dict):
#                     # Extract total reward
#                     total_reward = reward_result.get("total", 0.0)
#                     group_total_rewards.append(total_reward)

#                     # Collect individual components
#                     for key, value in reward_result.items():
#                         if key != "total":
#                             if key not in group_component_rewards:
#                                 group_component_rewards[key] = []
#                             group_component_rewards[key].append(value)
#                 else:
#                     # Backward compatibility: scalar reward
#                     group_total_rewards.append(float(reward_result))

#             total_rewards.append(group_total_rewards)

#             # Aggregate component rewards for this group
#             for key, values in group_component_rewards.items():
#                 if key not in reward_components:
#                     reward_components[key] = []
#                 reward_components[key].append(values)

#         # Convert to tensors
#         result = {"total": torch.tensor(total_rewards)}

#         # Add individual components if they exist
#         for key, values in reward_components.items():
#             result[key] = torch.tensor(values)

#         return result

#     def _get_next_batch(self) -> list[ReasoningPrompts]:
#         """Get the next batch of tokenized prompts."""
#         try:
#             batch = next(self.dataloader)
#             self.questions = batch["question"]
#             self.answers = batch["answer"]

#             returned_prompts = [
#                 {
#                     "input_ids": returned_prompt["input_ids"],
#                     "attention_mask": returned_prompt["attention_mask"],
#                     "text": (
#                         self.tokenizer.batch_decode(
#                             returned_prompt["input_ids"],
#                             skip_special_tokens=False,
#                             clean_up_tokenization_spaces=False,
#                         )[0]
#                         if self.return_raw_completions
#                         else None
#                     ),
#                 }
#                 for returned_prompt in batch["tokenized_prompts"]
#             ]
#         except StopIteration:
#             if not self.evaluation_mode:
#                 self.num_epochs += 1

#             self._reset_dataloaders(
#                 reset_train=not self.evaluation_mode,
#                 reset_test=self.evaluation_mode,
#             )
#             return self._get_next_batch()
#         return returned_prompts

#     def create_collate_fn(
#         self,
#         tokenizer: AutoTokenizer,
#         apply_chat_template_fn: Callable[[str, str, AutoTokenizer], BatchEncoding],
#     ) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
#         """
#         Create an efficient collate function that batches tokenization operations.

#         :param tokenizer: Tokenizer to be used for encoding and decoding the prompts.
#         :type tokenizer: AutoTokenizer
#         :param apply_chat_template_fn: Function to apply the chat template to questions and answers.
#         :type apply_chat_template_fn: Callable[[str, str, AutoTokenizer], BatchEncoding]
#         :return: Collate function that efficiently processes batches.
#         :rtype: Callable[[list[dict[str, Any]]], dict[str, Any]]
#         """

#         def collate_fn(batch):
#             questions = [item["question"] for item in batch]
#             answers = [item["answer"] for item in batch]

#             # Apply chat template to all samples
#             # Note: This still processes individually, but the result is more efficiently handled
#             tokenized_prompts = [
#                 apply_chat_template_fn(q, a, tokenizer)
#                 for q, a in zip(questions, answers)
#             ]

#             # Optional: If you want to create a fully batched version, uncomment below
#             # This would require modifying how tokenized_prompts are used downstream
#             #
#             # # Extract all input_ids and find max length in this batch
#             # all_input_ids = [tp["input_ids"][0] for tp in tokenized_prompts]
#             # max_length = max(len(ids) for ids in all_input_ids)
#             #
#             # # Pad to batch max length
#             # padded_batch = tokenizer.pad(
#             #     {"input_ids": all_input_ids},
#             #     padding="max_length",
#             #     max_length=max_length,
#             #     return_tensors="pt",
#             #     return_attention_mask=True
#             # )
#             #
#             # return {
#             #     "question": questions,
#             #     "answer": answers,
#             #     "input_ids": padded_batch["input_ids"],
#             #     "attention_mask": padded_batch["attention_mask"],
#             # }

#             return {
#                 "question": questions,
#                 "answer": answers,
#                 "tokenized_prompts": tokenized_prompts,
#             }

#         return collate_fn


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
    ):
        super().__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            custom_collate_fn=None,
            apply_chat_template_fn=None,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            min_completion_length=min_completion_length,
            accelerator=accelerator,
            seed=seed,
        )
        assert {"prompt", "chosen", "rejected"}.issubset(
            set(train_dataset.features.keys())
        ), "Train dataset must contain 'prompt', 'chosen', and 'rejected' features."
        assert {"prompt", "chosen", "rejected"}.issubset(
            set(test_dataset.features.keys())
        ), "Train dataset must contain 'prompt', 'chosen', and 'rejected' features."

    def reset(self, reset_dataloaders: bool = False) -> PreferencePrompts:
        """Reset the environment and get the next batch of tokenized prompts."""
        if reset_dataloaders:
            self._reset_dataloaders()
            warnings.warn(
                "env.reset() called with reset_dataloaders=True, this will reset the dataloaders to the beginning of the dataset, proceed with caution."
            )
        if self.reset_called:
            warnings.warn(
                "env.reset() called more than once sequentially, it should typically follow with env.step()."
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
        """
        Create a collate function that applies the chat template to the batch of questions and answers.

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
                prompts, truncation=True, padding=False, add_special_tokens=True
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
                max(len(ids) for ids in chosen_enc["input_ids"]),
                max(len(ids) for ids in rejected_enc["input_ids"]),
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


class DummyOptimizer:
    """
    Placeholder optimizer class to pass to the OptimizerWrapper when the optimizer is defined in the deepspeed config.
    """

    def __init__(self, params: list[torch.Tensor], lr: float, **kwargs) -> None:
        """
        Sentinel class to use for the optimizer when the optimizer is defined in the deepspeed config.

        :param params: Parameters to optimize.
        :type params: list[torch.Tensor]
        :param lr: Learning rate.
        :type lr: float
        """
        pass

    def step(self, closure=None):
        raise RuntimeError(
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )

    def zero_grad(self):
        raise RuntimeError(
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )

    def state_dict(self):
        raise RuntimeError(
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )

    def load_state_dict(self, state_dict):
        raise RuntimeError(
            "DummyOptimizer is a placeholder optimizer and should not be used."
            "Please ensure you are calling accelerator.prepare() on the optimizer."
        )


@contextmanager
def gather_if_zero3(
    zero_stage: int, params: list[torch.Tensor], modifier_rank: int | None = None
):
    """
    Conditional context manager for setting the zero stage for the model.

    :param zero_stage: The zero stage
    :type zero_stage: int
    :param params: The parameters to gather
    :type params: List[torch.Tensor]
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
    """
    Get the state dict of the model for zero3.

    :param model: The model to get the state dict of.
    :type model: nn.Module
    :return: The state dict of the model.
    :rtype: dict[str, torch.Tensor]
    """

    with gather_if_zero3(3, list(model.parameters()), modifier_rank=0):
        return model.state_dict()


def create_model_from_name_or_path(model_name_or_path: str, model_config: dict[str, Any ] | None = None) -> PreTrainedModel:
    """
    Create a model from a name or path.

    :param model_name_or_path: The name or path of the model to create.
    :type model_name_or_path: str
    :param model_config: The configuration of the model to create.
    :type model_config: dict[str, Any ] | None
    :return: The created model.
    :rtype: PreTrainedModel
    """
    if model_config is None:
        model_config = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        **model_config
    )
    return model