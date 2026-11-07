# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch import nn
from transformers import PretrainedConfig
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from agilerl.distributed.fsdp import FSDPConfig
from agilerl.distributed.runtime import FSDPRuntime


def actor_with(config: PretrainedConfig, attn_implementation: str) -> nn.Module:
    config._attn_implementation = attn_implementation
    actor = nn.Linear(1, 1)
    actor.config = config
    return actor


def prepare_with_cp(actor: nn.Module, cp: int = 2, tp: int = 1) -> None:
    runtime = FSDPRuntime(
        FSDPConfig(cp=cp, tp=tp, reshard_after_forward=True, optim_cpu_offload=False)
    )
    runtime.prepare_actor(actor, "cpu", None, 1e-5, None, lambda _names: None)


class TestFSDPRuntimePrepareActor:
    def test_context_parallel_accepts_deepseek_v4_on_flex(self) -> None:
        actor = actor_with(DeepseekV4Config(), "flex_attention")

        # Validation passes; building the mesh then needs a process group.
        with pytest.raises(RuntimeError, match="initialised process group"):
            prepare_with_cp(actor)

    def test_context_parallel_refuses_a_backend_ulysses_does_not_patch(
        self,
    ) -> None:
        actor = actor_with(Qwen3Config(), "sdpa")

        with pytest.raises(
            ValueError, match="needs flash_attention_2 or flex_attention"
        ):
            prepare_with_cp(actor)

    def test_context_parallel_splits_tp_local_heads(self) -> None:
        config = Qwen3Config(num_attention_heads=4, num_key_value_heads=4)
        actor = actor_with(config, "flex_attention")

        with pytest.raises(
            ValueError, match=r"num_attention_heads \(2\) divisible by cp \(4\)"
        ):
            prepare_with_cp(actor, cp=4, tp=2)
