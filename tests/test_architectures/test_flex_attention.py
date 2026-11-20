# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import sys
import types

import pytest
import torch

from agilerl.architectures.flex_attention import (
    device_shared_memory,
    large_head_dim_kernel_options,
    patch_flex_attention_kernel_options,
)

H100_SHARED_MEMORY = 232448
A100_SHARED_MEMORY = 166912
# L4 opt-in shared memory per block (99 KiB).
L4_SHARED_MEMORY = 101376
SPARSE_BLOCK_SIZE = 128
BF16_BYTES = 2


def shared_memory_bytes(options: dict[str, int], head_dim: int) -> dict[str, int]:
    """Pipelined tiles plus held tiles per kernel, as the template loads them."""
    row = 2 * head_dim * BF16_BYTES
    return {
        "forward": options["fwd_num_stages"] * options["BLOCK_N"] * row
        + options["BLOCK_M"] * head_dim * BF16_BYTES,
        "dkdv": (options["bwd_num_stages"] * options["BLOCK_M1"] + options["BLOCK_N1"])
        * row,
        "dq": (options["bwd_num_stages"] * options["BLOCK_N2"] + options["BLOCK_M2"])
        * row,
    }


class TestLargeHeadDimKernelOptions:
    def test_h100_gets_wider_forward_and_deeper_backward_pipeline(self):
        options = large_head_dim_kernel_options(
            512, 512, BF16_BYTES, H100_SHARED_MEMORY
        )

        assert options == {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "fwd_num_stages": 2,
            "fwd_num_warps": 8,
            "BLOCK_M1": 16,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 16,
            "bwd_num_stages": 3,
            "bwd_num_warps": 8,
        }

    def test_a100_gets_narrower_forward_and_two_stage_backward(self):
        options = large_head_dim_kernel_options(
            512, 512, BF16_BYTES, A100_SHARED_MEMORY
        )

        assert options["BLOCK_N"] == 16
        assert options["bwd_num_stages"] == 2

    def test_l4_shrinks_the_query_tile_and_the_backward_pipeline(self):
        options = large_head_dim_kernel_options(512, 512, BF16_BYTES, L4_SHARED_MEMORY)

        assert options == {
            "BLOCK_M": 16,
            "BLOCK_N": 16,
            "fwd_num_stages": 2,
            "fwd_num_warps": 8,
            "BLOCK_M1": 16,
            "BLOCK_N1": 16,
            "BLOCK_M2": 16,
            "BLOCK_N2": 16,
            "bwd_num_stages": 1,
            "bwd_num_warps": 8,
        }

    @pytest.mark.parametrize(
        "shared_memory", [H100_SHARED_MEMORY, A100_SHARED_MEMORY, L4_SHARED_MEMORY]
    )
    def test_tiles_fit_shared_memory(self, shared_memory):
        options = large_head_dim_kernel_options(512, 512, BF16_BYTES, shared_memory)

        footprint = shared_memory_bytes(options, 512)

        assert max(footprint.values()) <= shared_memory

    @pytest.mark.parametrize(
        "shared_memory", [H100_SHARED_MEMORY, A100_SHARED_MEMORY, L4_SHARED_MEMORY]
    )
    def test_tiles_satisfy_the_template_divisibility_rules(self, shared_memory):
        options = large_head_dim_kernel_options(512, 512, BF16_BYTES, shared_memory)

        assert options["BLOCK_N1"] % options["BLOCK_M1"] == 0
        assert options["BLOCK_M2"] % options["BLOCK_N2"] == 0
        for key in (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_M1",
            "BLOCK_N1",
            "BLOCK_M2",
            "BLOCK_N2",
        ):
            assert SPARSE_BLOCK_SIZE % options[key] == 0

    def test_fp32_accumulators_stay_within_128_registers_per_thread(self):
        options = large_head_dim_kernel_options(
            512, 512, BF16_BYTES, H100_SHARED_MEMORY
        )

        forward = options["BLOCK_M"] * 512 // (32 * options["fwd_num_warps"])
        dkdv = options["BLOCK_N1"] * 2 * 512 // (32 * options["bwd_num_warps"])
        dq = options["BLOCK_M2"] * 512 // (32 * options["bwd_num_warps"])

        assert max(forward, dkdv, dq) <= 128


class TestGemma4FlexHeads:
    """Gemma 4 E4B full attention is head dim 512; sliding attention is 256."""

    @staticmethod
    def _install_fake_flex(monkeypatch):
        seen = []

        def fake_flex_attention_forward(
            module, query, key, value, attention_mask, **kwargs
        ):
            seen.append((query, key, value, kwargs))
            return "flex-out"

        registry = {"flex_attention": fake_flex_attention_forward}
        flex_mod = types.ModuleType("transformers.integrations.flex_attention")
        flex_mod.flex_attention_forward = fake_flex_attention_forward
        modeling_mod = types.ModuleType("transformers.modeling_utils")
        modeling_mod.ALL_ATTENTION_FUNCTIONS = registry
        monkeypatch.setitem(
            sys.modules, "transformers.integrations.flex_attention", flex_mod
        )
        monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_mod)
        return registry, seen

    @staticmethod
    def _transposed_heads(heads: int, head_dim: int, offset: int) -> torch.Tensor:
        """``(batch, seq, heads, head_dim)`` transposed the way Gemma4 attention does.

        ``offset`` shifts the storage so the view's base is not 16-byte aligned.
        """
        width = head_dim + offset
        base = torch.empty(1, 17, heads, width, dtype=torch.bfloat16)
        view = base.narrow(-1, offset, head_dim)
        return view.transpose(1, 2)

    def test_cpu_tensor_has_no_device_shared_memory(self):
        query = torch.zeros(1, 8, 17, 512, dtype=torch.bfloat16)

        assert device_shared_memory(query) is None

    @pytest.mark.parametrize(
        ("heads", "head_dim"),
        [(8, 512), (8, 256), (2, 512)],
    )
    def test_transposed_heads_are_packed(self, monkeypatch, heads, head_dim):
        registry, seen = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        patch_flex_attention_kernel_options()
        query = self._transposed_heads(heads, head_dim, 1)
        key = self._transposed_heads(2, head_dim, 1)
        value = self._transposed_heads(2, head_dim, 1)
        assert query.data_ptr() % 16 != 0
        assert not query.is_contiguous()

        registry["flex_attention"](None, query, key, value, None)

        packed_q, packed_k, packed_v, _kwargs = seen[-1]
        for packed in (packed_q, packed_k, packed_v):
            assert packed.is_contiguous()
            assert packed.dtype == torch.bfloat16
            assert packed.data_ptr() % 16 == 0
            assert packed.stride(-1) == 1
            assert packed.stride(-2) * packed.element_size() % 16 == 0
        assert packed_q.shape == (1, heads, 17, head_dim)
        assert packed_k.shape == (1, 2, 17, head_dim)

    def test_l4_full_attention_uses_tiles_that_fit(self, monkeypatch):
        registry, seen = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
        monkeypatch.setattr(
            "agilerl.architectures.flex_attention.device_shared_memory",
            lambda query: L4_SHARED_MEMORY,
        )
        patch_flex_attention_kernel_options()
        query = torch.zeros(1, 8, 128, 512, dtype=torch.bfloat16)
        key = torch.zeros(1, 2, 128, 512, dtype=torch.bfloat16)
        value = torch.zeros(1, 2, 128, 512, dtype=torch.bfloat16)

        registry["flex_attention"](None, query, key, value, None)

        options = seen[0][3]["kernel_options"]
        assert options == large_head_dim_kernel_options(
            512, 512, BF16_BYTES, L4_SHARED_MEMORY
        )
        assert max(shared_memory_bytes(options, 512).values()) <= L4_SHARED_MEMORY

    def test_hopper_full_attention_uses_the_wide_head_tiles(self, monkeypatch):
        registry, seen = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
        monkeypatch.setattr(
            "agilerl.architectures.flex_attention.device_shared_memory",
            lambda query: H100_SHARED_MEMORY,
        )
        patch_flex_attention_kernel_options()
        query = torch.zeros(1, 8, 128, 512, dtype=torch.bfloat16)

        registry["flex_attention"](None, query, "k", "v", None)

        assert seen[0][3]["kernel_options"] == large_head_dim_kernel_options(
            512, 512, BF16_BYTES, H100_SHARED_MEMORY
        )

    def test_explicit_options_win_on_a_wide_head(self, monkeypatch):
        registry, seen = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
        patch_flex_attention_kernel_options(options={"BLOCK_M": 64})
        query = torch.zeros(1, 8, 32, 512, dtype=torch.bfloat16)

        registry["flex_attention"](None, query, "k", "v", None)

        assert seen[0][3]["kernel_options"] == {"BLOCK_M": 64}

    def test_sliding_head_on_l4_keeps_the_256_wide_tiles(self, monkeypatch):
        registry, seen = self._install_fake_flex(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
        patch_flex_attention_kernel_options()
        query = torch.zeros(1, 8, 128, 256, dtype=torch.bfloat16)

        registry["flex_attention"](None, query, "k", "v", None)

        options = seen[0][3]["kernel_options"]
        assert options["BLOCK_M"] == 32
        assert options["BLOCK_N"] == 32
        assert options["BLOCK_N1"] == 16
        assert "fwd_num_warps" not in options
