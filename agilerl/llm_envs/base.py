"""Base helpers and classes for LLM gym-style environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import BatchEncoding


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


class LLMEnv(ABC):
    """Base contract for teacher-forced dataset envs.

    The minimal ``reset`` / ``step`` / ``close`` surface the dataset trainer
    holds uniformly — :class:`~agilerl.llm_envs.dataset_env.DatasetEnv` is the
    concrete subtype. Generation envs
    (:class:`~agilerl.llm_envs.rollout_env.RolloutEnv`) deliberately do not
    subclass this: they speak the token-level rollout contract instead, so
    ``isinstance(env, LLMEnv)`` selects the dataset family only.
    ``evaluation_mode`` flags whether the env is serving its held-out split;
    :meth:`eval_mode` toggles it for the duration of a block and restores the
    prior value.

    :ivar evaluation_mode: Whether the env is currently serving the held-out split.
    :vartype evaluation_mode: bool
    """

    evaluation_mode: bool = False

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the environment and return its first observation."""

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> Any:
        """Advance the environment by one step.

        Generation envs consume an action and return the next observation;
        teacher-forced dataset envs advance via :meth:`reset` and implement this
        as a no-op.
        """

    @contextmanager
    def eval_mode(self) -> Generator[None, None, None]:
        """Serve the held-out split for the duration of the block, then restore."""
        previous = self.evaluation_mode
        self.evaluation_mode = True
        try:
            yield
        finally:
            self.evaluation_mode = previous

    def close(self) -> None:
        """Release any resources held by the environment.

        No-op by default; envs that hold resources (tool backends, dataloaders,
        sockets) override this.
        """
        return
