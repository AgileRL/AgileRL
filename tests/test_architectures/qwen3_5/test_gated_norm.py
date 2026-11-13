# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""fla's fused gated RMSNorm as the Qwen GatedDeltaNet output norm."""

import importlib.util

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from agilerl.architectures.qwen3_5.layers import install_fused_gated_norm
from tests.test_architectures.qwen3_5.gdn_helpers import GATED_NORMS

HAS_FLA = importlib.util.find_spec("fla") is not None
# Qwen3.6-35B-A3B GDN value-head dim.
HEAD_V_DIM = 128
EPS = 1e-6


def _reference(
    x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """``rmsnorm(x) * weight * silu(gate)`` in fp32."""
    x, gate, weight = x.float(), gate.float(), weight.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return normed * weight * F.silu(gate)


def _inputs(
    rows: int, dtype: torch.dtype, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn(rows, HEAD_V_DIM, dtype=dtype, device=device)
    gate = torch.randn(rows, HEAD_V_DIM, dtype=dtype, device=device)
    weight = torch.empty(HEAD_V_DIM, dtype=dtype, device=device).uniform_(0.5, 1.5)
    return x, gate, weight


def _built_norm(norm_cls: type[nn.Module], weight: torch.Tensor) -> nn.Module:
    norm = norm_cls(HEAD_V_DIM, eps=EPS).to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm


@pytest.mark.usefixtures("pristine_qwen_gdn_classes")
class TestInstallFusedGatedNorm:
    @pytest.mark.skipif(HAS_FLA, reason="checks the install without fla")
    @pytest.mark.parametrize("norm_cls", GATED_NORMS)
    def test_keeps_hf_norm_without_fla(self, norm_cls: type[nn.Module]) -> None:
        # Arrange
        torch.manual_seed(0)
        x, gate, weight = _inputs(16, torch.float32, "cpu")
        norm = _built_norm(norm_cls, weight)

        # Act
        install_fused_gated_norm()
        out = norm(x, gate)

        # Assert
        torch.testing.assert_close(out, _reference(x, gate, weight))


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_FLA, reason="needs flash-linear-attention")
@pytest.mark.usefixtures("pristine_qwen_gdn_classes")
class TestFusedGatedRmsNorm:
    @pytest.mark.parametrize("norm_cls", GATED_NORMS)
    def test_matches_fp32_reference_and_grads(self, norm_cls: type[nn.Module]) -> None:
        # Arrange
        torch.manual_seed(0)
        # Tokens x value heads of a packed GDN row; more rows than SMs.
        x, gate, weight = _inputs(4096, torch.bfloat16, "cuda")
        norm = _built_norm(norm_cls, weight)
        install_fused_gated_norm()
        x_ref, gate_ref, weight_ref = (
            t.detach().float().requires_grad_() for t in (x, gate, weight)
        )
        x.requires_grad_()
        gate.requires_grad_()
        grad_out = torch.randn_like(x)

        # Act
        out = norm(x, gate)
        out.backward(grad_out)
        ref = _reference(x_ref, gate_ref, weight_ref)
        ref.backward(grad_out.float())

        # Assert: fp32 math, one bf16 rounding per output, so one bf16 ulp.
        tolerance = {"rtol": 2**-8, "atol": 1e-5}
        torch.testing.assert_close(out.float(), ref, **tolerance)
        torch.testing.assert_close(x.grad.float(), x_ref.grad, **tolerance)
        torch.testing.assert_close(gate.grad.float(), gate_ref.grad, **tolerance)
        torch.testing.assert_close(
            norm.weight.grad.float(), weight_ref.grad, **tolerance
        )
