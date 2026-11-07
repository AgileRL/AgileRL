# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DeepseekV4Config, Glm5NextTextConfig, MiMoV2FlashConfig
from transformers.activations import SiLUActivation
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextExperts
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashExperts,
)

from agilerl.lora.moe import layouts as moe_layouts

SWIGLU_LIMIT = 10.0


def torch_swiglu(projected):
    gate, up = projected.chunk(2, dim=-1)
    return F.silu(gate) * up


def clamped_swiglu(projected):
    gate, up = projected.chunk(2, dim=-1)
    return F.silu(gate.clamp(max=SWIGLU_LIMIT)) * up.clamp(
        min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT
    )


def deepseek_v4_experts():
    return DeepseekV4Experts(
        DeepseekV4Config(
            hidden_size=8,
            intermediate_size=4,
            num_local_experts=2,
            swiglu_limit=SWIGLU_LIMIT,
        )
    )


def glm5_next_experts():
    return Glm5NextTextExperts(
        Glm5NextTextConfig(
            hidden_size=8,
            moe_intermediate_size=4,
            num_local_experts=2,
            swiglu_limit=SWIGLU_LIMIT,
        )
    )


def mimo_v2_flash_experts():
    return MiMoV2FlashExperts(
        MiMoV2FlashConfig(hidden_size=8, moe_intermediate_size=4, num_local_experts=2)
    )


class PlainExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act_fn = F.silu


CLAMPED_FAMILIES = [
    pytest.param(deepseek_v4_experts, id="deepseek_v4"),
    pytest.param(glm5_next_experts, id="glm5_next"),
]


class TestIsSilu:
    @pytest.mark.parametrize(
        "act_fn",
        [
            pytest.param(F.silu, id="functional"),
            pytest.param(nn.SiLU(), id="torch_module"),
            pytest.param(SiLUActivation(), id="transformers_module"),
        ],
    )
    def test_recognises_silu(self, act_fn):
        # Act / Assert
        assert moe_layouts.is_silu(act_fn)

    @pytest.mark.parametrize(
        "act_fn",
        [pytest.param(F.gelu, id="gelu"), pytest.param(nn.ReLU(), id="relu")],
    )
    def test_rejects_other_activations(self, act_fn):
        # Act / Assert
        assert not moe_layouts.is_silu(act_fn)


class TestRoutedExpertsGate:
    @pytest.mark.parametrize("make_experts", CLAMPED_FAMILIES)
    def test_returns_the_family_clamped_gate(self, make_experts):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8) * 3 * SWIGLU_LIMIT

        # Act
        apply_gate = moe_layouts.routed_experts_gate(make_experts())

        # Assert
        assert isinstance(apply_gate, moe_layouts.ClampedSwiGLU)
        assert apply_gate.limit == SWIGLU_LIMIT
        assert torch.equal(apply_gate(projected), clamped_swiglu(projected))

    def test_non_silu_clamped_gate_is_not_a_clamped_swiglu(self):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8) * 3 * SWIGLU_LIMIT
        experts = DeepseekV4Experts(
            DeepseekV4Config(
                hidden_size=8,
                intermediate_size=4,
                num_local_experts=2,
                swiglu_limit=SWIGLU_LIMIT,
                hidden_act="gelu",
            )
        )

        # Act
        apply_gate = moe_layouts.routed_experts_gate(experts)

        # Assert
        assert apply_gate is not None
        assert not isinstance(apply_gate, moe_layouts.ClampedSwiGLU)
        assert torch.equal(apply_gate(projected), experts._apply_gate(projected))

    @pytest.mark.parametrize(
        "make_experts",
        [
            pytest.param(mimo_v2_flash_experts, id="hf_default_gate"),
            pytest.param(PlainExperts, id="no_apply_gate"),
        ],
    )
    def test_returns_none_for_act_fn_gate_times_up(self, make_experts):
        # Act / Assert
        assert moe_layouts.routed_experts_gate(make_experts()) is None


class TestExpertActivation:
    def test_gated_silu_on_kernels_matches_torch_and_saves_only_the_projection(
        self, reference_expert_kernels
    ):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8, requires_grad=True)
        reference = projected.detach().clone().requires_grad_(True)
        grad = torch.randn(6, 4)
        saved = []

        def record(tensor):
            saved.append(tensor)
            return tensor

        # Act
        with torch.autograd.graph.saved_tensors_hooks(record, lambda t: t):
            out = moe_layouts.expert_activation(
                projected, F.silu, True, reference_expert_kernels.kernels, None
            )
        out.backward(grad)
        torch_swiglu(reference).backward(grad)

        # Assert
        # fp32 elementwise maths in a different association; one rounding apart.
        torch.testing.assert_close(out, torch_swiglu(reference), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(projected.grad, reference.grad, rtol=1e-6, atol=1e-6)
        assert reference_expert_kernels.used == {"swiglu", "swiglu_backward"}
        assert len(saved) == 1
        assert saved[0] is projected

    @pytest.mark.parametrize("make_experts", CLAMPED_FAMILIES)
    def test_clamped_family_gate_on_kernels_matches_hf_and_saves_only_the_projection(
        self, reference_expert_kernels, make_experts
    ):
        # Arrange
        torch.manual_seed(0)
        # Most gate and up entries fall outside ``[-limit, limit]``.
        projected = (torch.randn(6, 8) * 3 * SWIGLU_LIMIT).requires_grad_(True)
        reference = projected.detach().clone().requires_grad_(True)
        grad = torch.randn(6, 4)
        apply_gate = moe_layouts.routed_experts_gate(make_experts())
        saved = []

        def record(tensor):
            saved.append(tensor)
            return tensor

        # Act
        with torch.autograd.graph.saved_tensors_hooks(record, lambda t: t):
            out = moe_layouts.expert_activation(
                projected, F.silu, True, reference_expert_kernels.kernels, apply_gate
            )
        out.backward(grad)
        expected = clamped_swiglu(reference)
        expected.backward(grad)

        # Assert
        gate, up = projected.detach().chunk(2, dim=-1)
        assert (gate > SWIGLU_LIMIT).any()
        assert (up.abs() > SWIGLU_LIMIT).any()
        # fp32 closed-form backward against autograd's; one rounding apart.
        torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(projected.grad, reference.grad, rtol=1e-6, atol=1e-6)
        assert reference_expert_kernels.used == {
            "clamped_swiglu",
            "clamped_swiglu_backward",
        }
        assert len(saved) == 1
        assert saved[0] is projected

    @pytest.mark.parametrize("make_experts", CLAMPED_FAMILIES)
    def test_clamped_family_gate_without_kernels_reruns_the_hf_gate(self, make_experts):
        # Arrange
        torch.manual_seed(0)
        projected = (torch.randn(6, 8) * 3 * SWIGLU_LIMIT).requires_grad_(True)
        reference = projected.detach().clone().requires_grad_(True)
        grad = torch.randn(6, 4)
        apply_gate = moe_layouts.routed_experts_gate(make_experts())
        saved = []

        def record(tensor):
            saved.append(tensor)
            return tensor

        # Act
        with torch.autograd.graph.saved_tensors_hooks(record, lambda t: t):
            out = moe_layouts.expert_activation(
                projected, F.silu, True, None, apply_gate
            )
        out.backward(grad)
        expected = clamped_swiglu(reference)
        expected.backward(grad)

        # Assert
        # Same torch ops on the same values, forward and recomputed backward.
        assert torch.equal(out, expected)
        assert torch.equal(projected.grad, reference.grad)
        assert len(saved) == 1
        assert saved[0] is projected

    def test_without_kernels_runs_torch_ops(self):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8)

        # Act
        out = moe_layouts.expert_activation(projected, F.silu, True, None, None)

        # Assert
        torch.testing.assert_close(out, torch_swiglu(projected))

    def test_other_gated_activation_stays_on_torch(self, reference_expert_kernels):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8)
        gate, up = projected.chunk(2, dim=-1)

        # Act
        out = moe_layouts.expert_activation(
            projected, F.gelu, True, reference_expert_kernels.kernels, None
        )

        # Assert
        torch.testing.assert_close(out, F.gelu(gate) * up)
        assert reference_expert_kernels.used == set()

    def test_ungated_activation_applies_to_the_whole_projection(
        self, reference_expert_kernels
    ):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8)

        # Act
        out = moe_layouts.expert_activation(
            projected, F.silu, False, reference_expert_kernels.kernels, None
        )

        # Assert
        torch.testing.assert_close(out, F.silu(projected))
        assert reference_expert_kernels.used == set()
