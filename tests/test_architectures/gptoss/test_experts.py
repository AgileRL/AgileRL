# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from agilerl.architectures.gptoss import experts as experts_mod
from agilerl.architectures.gptoss.experts import (
    expert_matmul_loop,
    gpt_oss_experts_local_forward,
    is_gpt_oss_experts_module,
)

NUM_EXPERTS = 2
HIDDEN = 4
INTERMEDIATE = 3


class GptOssLayout(nn.Module):
    """Minimal module with the GptOssExperts parameter shapes."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.zeros(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE)
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE)
        )
        self.down_proj = nn.Parameter(torch.zeros(NUM_EXPERTS, INTERMEDIATE, HIDDEN))
        self.down_proj_bias = nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN))
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states, router_indices, routing_weights):
        return hidden_states


def _break_layout(module: nn.Module, kind: str) -> None:
    if kind == "up":
        module.gate_up_proj = nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN))
    elif kind == "down":
        module.down_proj = nn.Parameter(torch.zeros(NUM_EXPERTS, HIDDEN))
    elif kind == "bias":
        module.gate_up_proj_bias = None
    elif kind == "alpha":
        module.alpha = "no"
    else:
        module.gate_up_proj = nn.Parameter(
            torch.zeros(NUM_EXPERTS, HIDDEN, 2 * INTERMEDIATE + 1)
        )


class TestIsGptOssExpertsModule:
    def test_accepts_matching_layout(self) -> None:
        assert is_gpt_oss_experts_module(GptOssLayout()) is True

    @pytest.mark.parametrize("kind", ["up", "down", "bias", "alpha", "odd"])
    def test_rejects_layout(self, kind: str) -> None:
        module = GptOssLayout()
        _break_layout(module, kind)

        assert is_gpt_oss_experts_module(module) is False


class TestExpertMatmulLoop:
    def test_reads_dtensor_local_shard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        weight = torch.randn(2, 4, 3)
        bias = torch.randn(2, 3)
        rows = torch.randn(4, 4)
        counts = [2, 2]

        class Shard:
            def __init__(self, local: torch.Tensor) -> None:
                self.local = local

            def to_local(self) -> torch.Tensor:
                return self.local

        monkeypatch.setattr(experts_mod, "DTensor", Shard)

        sharded = expert_matmul_loop(rows, Shard(weight), counts, Shard(bias))
        dense = expert_matmul_loop(rows, weight, counts, bias)

        assert torch.allclose(sharded, dense)


class TestGptOssExpertsLocalForward:
    def test_rejects_other_layouts(self) -> None:
        with pytest.raises(RuntimeError, match="GptOssExperts layout"):
            gpt_oss_experts_local_forward(
                nn.Linear(4, 4),
                torch.randn(2, 4),
                torch.zeros(2, 1, dtype=torch.long),
                torch.ones(2, 1),
            )
