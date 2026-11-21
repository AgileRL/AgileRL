# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for pinning Liger RoPE to its Triton kernel."""

from __future__ import annotations

import pytest
import torch

from agilerl import HAS_LIGER_KERNEL

if not HAS_LIGER_KERNEL:
    pytest.skip(
        "Liger RoPE tests require liger-kernel; skipping on this platform.",
        allow_module_level=True,
    )
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from agilerl.architectures.liger_rope import pin_triton_rope, triton_rotary_pos_emb


@pytest.fixture
def restored_qwen3_rope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the module-level RoPE swap a test makes."""
    monkeypatch.setattr(
        modeling_qwen3, "apply_rotary_pos_emb", modeling_qwen3.apply_rotary_pos_emb
    )


def _tiny_qwen3() -> modeling_qwen3.Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
    )
    return modeling_qwen3.Qwen3ForCausalLM(config)


def _reference_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotate_half = modeling_qwen3.rotate_half
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class TestPinTritonRope:
    def test_points_liger_rope_at_triton(self, restored_qwen3_rope) -> None:
        # Arrange
        modeling_qwen3.apply_rotary_pos_emb = liger_rotary_pos_emb

        # Act
        pin_triton_rope(_tiny_qwen3())

        # Assert
        assert modeling_qwen3.apply_rotary_pos_emb is triton_rotary_pos_emb

    def test_leaves_non_liger_rope(self, restored_qwen3_rope) -> None:
        # Arrange
        def custom_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            return q, k

        modeling_qwen3.apply_rotary_pos_emb = custom_rope

        # Act
        pin_triton_rope(_tiny_qwen3())

        # Assert
        assert modeling_qwen3.apply_rotary_pos_emb is custom_rope


@pytest.mark.gpu
class TestTritonRotaryPosEmb:
    @pytest.mark.parametrize("seq_len", [7, 300, 301])
    def test_matches_reference_rotation_and_grads(self, seq_len: int) -> None:
        # Arrange
        torch.manual_seed(0)
        head_dim = 128
        q = torch.randn(1, 4, seq_len, head_dim, device="cuda", requires_grad=True)
        k = torch.randn(1, 2, seq_len, head_dim, device="cuda", requires_grad=True)
        angles = torch.randn(1, seq_len, head_dim // 2, device="cuda")
        cos = torch.cat([angles, angles], dim=-1).cos()
        sin = torch.cat([angles, angles], dim=-1).sin()
        grad_q = torch.randn_like(q)
        grad_k = torch.randn_like(k)
        ref_q, ref_k = _reference_rope(q, k, cos, sin)
        ref_grads = torch.autograd.grad((ref_q, ref_k), (q, k), (grad_q, grad_k))

        # Act
        out_q, out_k = triton_rotary_pos_emb(q.clone(), k.clone(), cos, sin)
        grads = torch.autograd.grad((out_q, out_k), (q, k), (grad_q, grad_k))

        # Assert
        # fp32 rotation: the kernel and the reference differ only in rounding.
        tolerance = {"rtol": 1e-5, "atol": 1e-5}
        assert torch.allclose(out_q, ref_q, **tolerance)
        assert torch.allclose(out_k, ref_k, **tolerance)
        assert torch.allclose(grads[0], ref_grads[0], **tolerance)
        assert torch.allclose(grads[1], ref_grads[1], **tolerance)
