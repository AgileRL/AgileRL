# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared LLM algorithm test helpers that must not require vLLM.

SFT/DPO tests import these without pulling in ``test_grpo``'s vLLM
``importorskip``, which is unavailable on macOS/Windows.
"""

from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead


class DummyConfig(PretrainedConfig):
    def __init__(
        self,
        input_size=16,
        max_tokens=8,
        vocab_size=100,
        intermediate_size=128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size


class DummyForwardOutput:
    def __init__(self, logits):
        self.logits = logits


class DummyMLPPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = DummyConfig
    base_model_prefix = "dummy_mlp"
    supports_gradient_checkpointing = True

    def __init__(self, config: DummyConfig, device="cpu"):
        super().__init__(config)
        self.input_size = config.input_size
        self.max_tokens = config.max_tokens
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing_enabled = False
        self.datatype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        hidden_size = 32
        # Standard causal-LM shape (embed -> body -> lm_head) so the
        # (now unconditional) fused-linear-logprob path can identity-patch
        # ``lm_head`` and read the hidden state. ``linear_1`` stays the LoRA
        # target the fixtures expect.
        self.embed = nn.Embedding(
            self.vocab_size, hidden_size, device=device, dtype=self.datatype
        )
        self.linear_1 = nn.Linear(
            hidden_size,
            hidden_size,
            device=device,
            dtype=self.datatype,
        )
        self.lm_head = nn.Linear(
            hidden_size,
            self.vocab_size,
            bias=False,
            device=device,
            dtype=self.datatype,
        )

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @property
    def model(self):
        return self

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> DummyForwardOutput:
        # ``lm_head`` may be identity-patched by the fused-linear-logprob path,
        # in which case this returns the hidden state instead of logits.
        hidden = self.linear_1(self.embed(input_ids.long()))
        return DummyForwardOutput(logits=self.lm_head(hidden))

    def generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            msg = "`input_ids` must be provided for generation."
            raise ValueError(msg)
        input_shape = input_ids.shape
        group_size = input_shape[0]
        prompt_size = input_shape[1]
        # Simple generation: just return random tokens based on vocab size and desired length
        return torch.randint(
            0,
            self.vocab_size,
            (group_size, prompt_size + self.config.max_tokens),
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.gradient_checkpointing_enabled = True

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return


def create_module(input_size, max_tokens, vocab_size, device):
    return DummyMLPPreTrainedModel(
        config=DummyConfig(
            input_size=input_size,
            max_tokens=max_tokens,
            vocab_size=vocab_size,
        ),
        device=device,
    )


class DummyHiddenStatesModel(DummyMLPPreTrainedModel):
    """Dummy causal LM that returns hidden states for a value head."""

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden = self.linear_1(self.embed(input_ids.long()))
        return CausalLMOutputWithPast(
            logits=self.lm_head(hidden), hidden_states=(hidden,)
        )


def create_value_head_module(
    input_size: int, max_tokens: int, vocab_size: int, device: str
) -> AutoModelForCausalLMWithValueHead:
    """Dummy causal LM wrapped with a value head, as PPO trains it."""
    config = DummyConfig(
        input_size=input_size,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
        hidden_size=32,
    )
    return AutoModelForCausalLMWithValueHead(
        DummyHiddenStatesModel(config=config, device=device)
    )


def scale_losses(
    loss_fn: Callable[..., Any], scales: Iterator[float]
) -> Callable[..., Any]:
    """``loss_fn`` whose loss, the first output of a tuple, carries the next scale."""

    def scaled(*args: Any, **kwargs: Any) -> Any:
        out = loss_fn(*args, **kwargs)
        if isinstance(out, tuple):
            return (out[0] * next(scales), *out[1:])
        return out * next(scales)

    return scaled


def trainable_weights(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def optimizer_state(optimizer: OptimizerWrapper) -> list[torch.Tensor]:
    """Copy of every tensor of the optimizer's per-parameter state, in a fixed order."""
    state = optimizer.state_dict()["state"]
    return [
        value.detach().clone()
        for _, param_state in sorted(state.items())
        for _, value in sorted(param_state.items())
        if isinstance(value, torch.Tensor)
    ]


def record_outputs(fn: Callable[..., Any]) -> tuple[Callable[..., Any], list[Any]]:
    """``fn`` that also appends each output to the returned list."""
    outputs: list[Any] = []

    def recorded(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        outputs.append(out)
        return out

    return recorded, outputs


def record_calls(
    fn: Callable[..., Any],
) -> tuple[Callable[..., Any], list[tuple[tuple[Any, ...], dict[str, Any], Any]]]:
    """``fn`` that also appends each call's ``(args, kwargs, output)`` to the returned list."""
    calls: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []

    def recorded(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        calls.append((args, kwargs, out))
        return out

    return recorded, calls
