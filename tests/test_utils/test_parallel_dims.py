# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Layout tests for hybrid FSDP / CP / EP mesh planning."""

from __future__ import annotations

import pytest

from agilerl.utils.parallel_dims import (
    compute_hybrid_layout,
    validate_hybrid_parallel_config,
)


class TestComputeHybridLayout:
    def test_flat(self):
        layout = compute_hybrid_layout(8, cp=1, ep=1)
        assert layout.dp_shard == 8
        assert layout.cp == 1
        assert layout.ep == 1

    def test_cp_only(self):
        layout = compute_hybrid_layout(8, cp=2, ep=1)
        assert layout.dp_shard == 4
        assert layout.cp == 2
        assert not layout.ep_enabled

    def test_ep_only(self):
        layout = compute_hybrid_layout(8, cp=1, ep=4)
        assert layout.dp_mod_ep == 2
        assert layout.dp_in_ep == 4
        assert layout.ep == 4

    def test_compose_two_gpu_degenerate(self):
        """2×GPU joint smoke: cp=2, ep=2 → dp_mod_ep=1, dp_in_ep=1, cp=2."""
        layout = compute_hybrid_layout(2, cp=2, ep=2)
        assert layout.dp_mod_ep == 1
        assert layout.dp_in_ep == 1
        assert layout.cp == 2
        assert layout.ep == 2
        assert layout.dp_shard == 1

    def test_compose_world8_cp2_ep4(self):
        layout = compute_hybrid_layout(8, cp=2, ep=4)
        assert layout.dp_mod_ep == 2
        assert layout.dp_in_ep == 2
        assert layout.cp == 2
        assert layout.ep == 4
        assert layout.dp_shard * layout.cp == layout.world_size
        assert (layout.dp_shard * layout.cp) % layout.ep == 0
        assert layout.ep % layout.cp == 0

    def test_ep_not_divisible_by_cp_raises(self):
        with pytest.raises(ValueError, match="divisible by cp"):
            compute_hybrid_layout(8, cp=4, ep=2)

    def test_world_not_divisible_by_ep_raises(self):
        with pytest.raises(ValueError, match="divisible by ep"):
            compute_hybrid_layout(6, cp=1, ep=4)

    def test_validate_alias(self):
        layout = validate_hybrid_parallel_config(world_size=2, cp=2, ep=2)
        assert layout.dp_in_ep == 1
