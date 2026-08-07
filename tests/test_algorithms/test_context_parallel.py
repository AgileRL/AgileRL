"""Unit tests for context-parallel primitives and config validation (CP0)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from agilerl.utils.cp import (
    ParallelDims,
    assert_ulysses_head_divisibility,
    pad_seq_to_cp_multiple,
    shard_for_cp,
    shift_labels_for_cp,
    substitute_cp_attention,
    validate_cp_config,
)
from agilerl.utils.distributed import FSDPConfig
from agilerl.utils.ulysses_attn import (
    ULYSSES_PARAMS,
    _replicate_kv_heads,
    update_ulysses_params,
)


class TestParallelDimsMath:
    def test_dp_shard_times_cp_equals_world(self):
        dims = ParallelDims.from_world(world_size=8, cp=2)
        assert dims.dp_shard == 4
        assert dims.cp == 2
        assert dims.dp_shard * dims.cp == dims.world_size
        assert dims.dp_size == 4

    def test_cp1_is_noop_topology(self):
        dims = ParallelDims.from_world(world_size=4, cp=1)
        assert dims.dp_shard == 4
        assert dims.cp_enabled is False
        assert dims.build_mesh() is None
        assert dims.world_mesh is None

    def test_rejects_world_not_divisible_by_cp(self):
        with pytest.raises(ValueError, match="divisible by cp"):
            ParallelDims.from_world(world_size=4, cp=3)

    def test_rejects_mismatched_product(self):
        with pytest.raises(ValueError, match="Invalid parallel dims"):
            ParallelDims(dp_shard=3, cp=2, world_size=4)


class TestShardForCp:
    def test_round_trip_shapes(self):
        t = torch.arange(16, dtype=torch.float32).view(1, 16)
        shards = [shard_for_cp(t, cp_rank=r, cp_world_size=4) for r in range(4)]
        assert all(s.shape == (1, 4) for s in shards)
        assert torch.equal(torch.cat(shards, dim=1), t)

    def test_wrong_seq_len_raises(self):
        t = torch.zeros(1, 5)
        with pytest.raises(ValueError, match="divisible by cp size"):
            shard_for_cp(t, cp_rank=0, cp_world_size=2)

    def test_batch_gt_one_raises(self):
        t = torch.zeros(2, 8)
        with pytest.raises(ValueError, match="batch dimension 1"):
            shard_for_cp(t, cp_rank=0, cp_world_size=2)

    def test_shift_then_shard_keeps_boundary_token(self):
        ids = torch.arange(8, dtype=torch.long).view(1, 8)
        labels = shift_labels_for_cp(ids, pad_token_id=-100)
        # Position 3 (last of rank-0 shard) should predict token 4 (first of rank-1).
        assert labels[0, 3].item() == 4
        local0 = shard_for_cp(labels, cp_rank=0, cp_world_size=2)
        local1 = shard_for_cp(labels, cp_rank=1, cp_world_size=2)
        assert local0[0, -1].item() == 4
        assert local1[0, 0].item() == 5
        assert labels[0, -1].item() == -100


class TestShardForCpRingDivisor:
    def test_pad_to_2cp_for_ring_zigzag(self):
        t = torch.zeros(1, 5, dtype=torch.long)
        padded = pad_seq_to_cp_multiple(t, cp_world_size=2, ring_zigzag=True, pad_value=0)
        assert padded.shape[1] % 4 == 0
        assert padded.shape[1] == 8

    def test_ulysses_pad_uses_cp_only(self):
        t = torch.zeros(1, 5, dtype=torch.long)
        padded = pad_seq_to_cp_multiple(t, cp_world_size=2, ring_zigzag=False)
        assert padded.shape[1] == 6


class TestGqaKvReplicate:
    def test_replicate_layout(self):
        # H_kv=2, cp=4 → each head repeated twice → [S, 4, D]
        t = torch.arange(2 * 3, dtype=torch.float32).view(1, 2, 3)
        out = _replicate_kv_heads(t, cp_size=4)
        assert out.shape == (1, 4, 3)
        assert torch.equal(out[0, 0], t[0, 0])
        assert torch.equal(out[0, 1], t[0, 0])
        assert torch.equal(out[0, 2], t[0, 1])
        assert torch.equal(out[0, 3], t[0, 1])

    def test_replicate_grad_sums(self):
        t = torch.randn(2, 2, 4, requires_grad=True)
        out = _replicate_kv_heads(t, cp_size=4)
        out.sum().backward()
        # Each original head appears twice → grad == 2
        assert torch.allclose(t.grad, torch.full_like(t.grad, 2.0))

    def test_reject_when_neither_divides(self):
        with pytest.raises(ValueError, match="GQA"):
            assert_ulysses_head_divisibility(
                num_attention_heads=8,
                num_key_value_heads=3,
                cp=2,
            )

    def test_accept_kv_replicate_path(self):
        assert_ulysses_head_divisibility(
            num_attention_heads=32,
            num_key_value_heads=2,
            cp=8,
        )

    def test_reject_query_heads_not_divisible(self):
        with pytest.raises(ValueError, match="num_attention_heads"):
            assert_ulysses_head_divisibility(
                num_attention_heads=7,
                num_key_value_heads=7,
                cp=2,
            )


class TestCpConfigValidation:
    def test_cp_requires_fsdp(self):
        with pytest.raises(ValueError, match="requires fsdp_config"):
            validate_cp_config(
                cp=2,
                cp_style="ulysses",
                fsdp_config=None,
                world_size=2,
                check_flash_attn=False,
            )

    def test_world_size_must_divide(self):
        with pytest.raises(ValueError, match="divisible by cp"):
            validate_cp_config(
                cp=3,
                cp_style="ulysses",
                fsdp_config=FSDPConfig(),
                world_size=4,
                check_flash_attn=False,
            )

    def test_unknown_style(self):
        with pytest.raises(ValueError, match="cp_style"):
            validate_cp_config(
                cp=2,
                cp_style="zigzag",  # type: ignore[arg-type]
                fsdp_config=FSDPConfig(),
                world_size=2,
                check_flash_attn=False,
            )

    def test_liger_rejected(self):
        with pytest.raises(ValueError, match="Liger"):
            validate_cp_config(
                cp=2,
                cp_style="ulysses",
                fsdp_config=FSDPConfig(),
                world_size=2,
                use_liger_loss=True,
                check_flash_attn=False,
            )

    def test_flex_packing_rejected(self):
        with pytest.raises(ValueError, match="flex"):
            validate_cp_config(
                cp=2,
                cp_style="ulysses",
                fsdp_config=FSDPConfig(),
                world_size=2,
                packing_mode="flex",
                check_flash_attn=False,
            )

    def test_sdpa_rejected(self):
        with pytest.raises(ValueError, match="flash_attention_2"):
            validate_cp_config(
                cp=2,
                cp_style="ulysses",
                fsdp_config=FSDPConfig(),
                world_size=2,
                attn_implementation="sdpa",
                check_flash_attn=False,
            )

    def test_cp1_skips_checks(self):
        style = validate_cp_config(
            cp=1,
            cp_style="ring",
            fsdp_config=None,
            world_size=1,
            check_flash_attn=True,
        )
        assert style == "ulysses"

    def test_missing_flash_attn_raises(self):
        with (
            patch("agilerl.utils.cp.flash_attn_available", return_value=False),
            pytest.raises(ValueError, match="flash-attn"),
        ):
            validate_cp_config(
                cp=2,
                cp_style="ulysses",
                fsdp_config=FSDPConfig(),
                world_size=2,
                check_flash_attn=True,
            )

    def test_ring_without_dep_raises(self):
        with (
            patch("agilerl.utils.cp.flash_attn_available", return_value=True),
            patch("agilerl.utils.cp.ring_flash_attn_available", return_value=False),
            pytest.raises(ValueError, match="ring-flash-attn"),
        ):
            validate_cp_config(
                cp=2,
                cp_style="ring",
                fsdp_config=FSDPConfig(),
                world_size=2,
                check_flash_attn=True,
            )


class TestCpStyleSelection:
    def test_ulysses_selected(self):
        with patch("agilerl.utils.ulysses_attn.substitute_hf_ulysses_attn") as sub:
            group = MagicMock()
            substitute_cp_attention("ulysses", group)
            sub.assert_called_once_with(group)

    def test_ring_selected(self):
        import agilerl.utils.ring_attn_compat  # noqa: F401 — before ring_flash_attn

        with (
            patch("ring_flash_attn.substitute_hf_flash_attn") as sub,
            patch("ring_flash_attn.adapters.hf_adapter.use_ring_attn") as use_ring,
        ):
            group = MagicMock()
            substitute_cp_attention("ring", group)
            sub.assert_called_once_with(group, heads_k_stride=1)
            use_ring.assert_called_once_with(False)

    def test_unknown_style_raises(self):
        with pytest.raises(ValueError, match="Unknown cp_style"):
            substitute_cp_attention("nope", MagicMock())  # type: ignore[arg-type]


class TestUlyssesMonkeypatch:
    def test_all_attention_functions_replaced(self):
        import transformers.modeling_flash_attention_utils as mfu
        import transformers.modeling_utils as mu

        from agilerl.utils import ulysses_attn as ua

        original_registry = mu.ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")
        original_attr = mfu._flash_attention_forward
        fake_flash = MagicMock()

        try:
            with (
                patch.object(ua, "flash_attn_varlen_func", fake_flash, create=True),
                patch.dict(
                    "sys.modules",
                    {"flash_attn": MagicMock(flash_attn_varlen_func=fake_flash)},
                ),
                patch("torch.distributed.get_world_size", return_value=2),
            ):
                ua.substitute_hf_ulysses_attn(
                    MagicMock(), patch_all_attention_functions=True
                )
                assert mu.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is not original_registry
                assert callable(mu.ALL_ATTENTION_FUNCTIONS["flash_attention_2"])
                assert mfu._flash_attention_forward is not original_attr
        finally:
            if original_registry is not None:
                mu.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = original_registry
            mfu._flash_attention_forward = original_attr

    def test_attr_only_patch_leaves_registry(self):
        """Negative: module-attr patch alone must not touch ALL_ATTENTION_FUNCTIONS."""
        import transformers.modeling_flash_attention_utils as mfu
        import transformers.modeling_utils as mu

        from agilerl.utils import ulysses_attn as ua

        original_registry = mu.ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")
        original_attr = mfu._flash_attention_forward
        fake_flash = MagicMock()

        try:
            with (
                patch.dict(
                    "sys.modules",
                    {"flash_attn": MagicMock(flash_attn_varlen_func=fake_flash)},
                ),
                patch("torch.distributed.get_world_size", return_value=2),
            ):
                ua.substitute_hf_ulysses_attn(
                    MagicMock(), patch_all_attention_functions=False
                )
                assert (
                    mu.ALL_ATTENTION_FUNCTIONS.get("flash_attention_2")
                    is original_registry
                )
                assert mfu._flash_attention_forward is not original_attr
        finally:
            if original_registry is not None:
                mu.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = original_registry
            mfu._flash_attention_forward = original_attr

    def test_update_ulysses_params(self):
        cu = torch.tensor([0, 4], dtype=torch.int32)
        update_ulysses_params(cu, 4)
        assert ULYSSES_PARAMS["max_seqlen"] == 4
        assert torch.equal(ULYSSES_PARAMS["cu_seqlens"], cu)


class TestCpBatchSizing:
    def test_dp_size_excludes_cp(self):
        from agilerl.algorithms.core.base import LLMAlgorithm

        agent = SimpleNamespace(
            cp=2,
            parallel_dims=ParallelDims.from_world(4, cp=2),
            _requested_gradient_accumulation_steps=1,
        )
        agent._dp_world_size = lambda: ParallelDims.from_world(4, cp=2).dp_size
        LLMAlgorithm._configure_batch_size_per_process(
            agent, batch_size=8, micro_batch_size_per_gpu=None
        )
        assert agent.batch_size_per_process == 4


class TestFusedMultiAdapterDisabledUnderCp:
    def test_fused_forward_rejects_value_head_stacking(self):
        from agilerl.algorithms.core.base import LLMAlgorithm

        agent = SimpleNamespace(
            cp=2,
            use_value_head=True,
            pad_token_id=0,
        )
        agent._cp_enabled = lambda: True
        ids = torch.zeros(1, 8, dtype=torch.long)
        mask = torch.ones(1, 8, dtype=torch.long)
        with pytest.raises(ValueError, match="value-head"):
            LLMAlgorithm._fused_forward(
                agent, ids, batch_size=1, attention_mask=mask
            )
