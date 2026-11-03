# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import SiLUActivation

from agilerl.lora.moe import layouts as moe_layouts


def torch_swiglu(projected):
    gate, up = projected.chunk(2, dim=-1)
    return F.silu(gate) * up


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
                projected, F.silu, True, reference_expert_kernels.kernels
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

    def test_without_kernels_runs_torch_ops(self):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8)

        # Act
        out = moe_layouts.expert_activation(projected, F.silu, True, None)

        # Assert
        torch.testing.assert_close(out, torch_swiglu(projected))

    def test_other_gated_activation_stays_on_torch(self, reference_expert_kernels):
        # Arrange
        torch.manual_seed(0)
        projected = torch.randn(6, 8)
        gate, up = projected.chunk(2, dim=-1)

        # Act
        out = moe_layouts.expert_activation(
            projected, F.gelu, True, reference_expert_kernels.kernels
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
            projected, F.silu, False, reference_expert_kernels.kernels
        )

        # Assert
        torch.testing.assert_close(out, F.silu(projected))
        assert reference_expert_kernels.used == set()
