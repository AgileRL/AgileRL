# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared LLM algorithm test helpers that must not require vLLM.

SFT/DPO tests import these without pulling in ``test_grpo``'s vLLM
``importorskip``, which is unavailable on macOS/Windows.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

deepspeed_base_config = {
    "bf16": {
        "enabled": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    },
    "auto_cast": True,
    "gradient_clipping": 0.5,
    "gradient_accumulation_steps": 1,
}

deepspeed_config_stage_1 = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 1,
    },
}

deepspeed_config_stage_2 = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 2,
    },
}

deepspeed_config_stage_3 = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 3,
    },
}

deepspeed_config_stage_1_with_scheduler = deepspeed_base_config | {
    "zero_optimization": {
        "stage": 1,
    },
    "scheduler": {
        "params": {
            "warmup_max_lr": 0.001,
            "num_epochs": 10,
            "warmup_proportion": 0.05,
        },
    },
}


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

    def __init__(self, config: DummyConfig, device="cpu"):
        super().__init__(config)
        self.input_size = config.input_size
        self.max_tokens = config.max_tokens
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing_enabled = False
        self.datatype = (
            torch.bfloat16
            if deepspeed_base_config.get("bf16", {}).get("enabled", False)
            else (
                torch.float16
                if deepspeed_base_config.get("fp16", {}).get("enabled", False)
                else torch.float32
            )
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


def _patch_mps_learn_hooks(monkeypatch: pytest.MonkeyPatch, module: str) -> MagicMock:
    """Make ``learn`` think MPS is available and record ``torch.mps.empty_cache`` calls."""
    mock_empty = MagicMock()
    monkeypatch.setattr(f"{module}.torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr(f"{module}.torch.mps.empty_cache", mock_empty)
    return mock_empty
