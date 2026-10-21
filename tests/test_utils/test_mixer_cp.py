# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU wrap contracts for Mamba/GDN context parallel."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from agilerl.architectures.runtime import ContextParallelPlan
from agilerl.distributed.context_parallel.mixer import (
    MixerInProjBypass,
    _call_local,
    apply_mixer_context_parallel,
    install_gdn_context_parallel,
    install_mamba_context_parallel,
)


class DummyMambaMixer(nn.Module):
    def __init__(self, num_heads: int = 4, n_groups: int = 2) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.head_dim = 8
        self.ssm_state_size = 4
        self.intermediate_size = num_heads * 8
        self.conv_dim = self.intermediate_size + 2 * n_groups * 4
        self.in_proj = nn.Identity()
        self.out_proj = nn.Identity()
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, 4, groups=self.conv_dim)
        self.dt_bias = nn.Parameter(torch.zeros(num_heads))
        self.A_log = nn.Parameter(torch.zeros(num_heads))
        self.D = nn.Parameter(torch.zeros(num_heads))
        self.norm = nn.LayerNorm(self.intermediate_size)

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        return self.out_proj(self.in_proj(hidden_states))


class DummyTpMambaMixer(DummyMambaMixer):
    def __init__(self, num_heads: int = 8, n_groups: int = 4) -> None:
        super().__init__(num_heads=num_heads, n_groups=n_groups)
        self.use_mem_eff_path = True
        self.scan_heads: int | None = None
        self.projected_last_dim: int | None = None

    def torch_forward(
        self, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ):
        self.scan_heads = int(self.num_heads)
        projected = self.in_proj(hidden_states)
        self.projected_last_dim = int(projected.shape[-1])
        return self.out_proj(projected)

    def cuda_kernels_forward(
        self, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ):
        raise AssertionError("cuda_kernels_forward should not run")

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        tp = int(getattr(self, "_tp_degree", 1) or 1)
        saved = (
            self.num_heads,
            self.n_groups,
            self.intermediate_size,
            self.conv_dim,
        )
        self.num_heads = self.num_heads // tp
        self.n_groups = self.n_groups // tp
        self.intermediate_size = self.num_heads * self.head_dim
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        try:
            return self.torch_forward(hidden_states, *args, **kwargs)
        finally:
            (
                self.num_heads,
                self.n_groups,
                self.intermediate_size,
                self.conv_dim,
            ) = saved


class DummyGatedDeltaNet(nn.Module):
    def __init__(self, num_k_heads: int = 4, n_groups: int = 2) -> None:
        super().__init__()
        self.num_k_heads = num_k_heads
        self.n_groups = n_groups
        self.proj = nn.Identity()

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        return hidden_states


def _cp_mesh(size: int = 2) -> MagicMock:
    mesh = MagicMock()
    mesh.size.return_value = size
    mesh.get_group.return_value = object()
    return mesh


class TestApplyMixerContextParallel:
    def test_skips_when_mesh_is_none_or_size_one(self) -> None:
        model = DummyMambaMixer()

        assert apply_mixer_context_parallel(model, None) == 0
        assert apply_mixer_context_parallel(model, _cp_mesh(1)) == 0
        assert not getattr(model, "_agilerl_mamba_cp", False)

    def test_runs_catalog_plans(self, monkeypatch) -> None:
        seen: list[nn.Module] = []

        def wrap(model: nn.Module, _mesh: object) -> int:
            seen.append(model)
            return 3

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mixer.family_context_parallel_plans",
            lambda: [ContextParallelPlan(wrap=wrap)],
        )

        model = nn.Linear(2, 2)

        assert apply_mixer_context_parallel(model, _cp_mesh()) == 3
        assert seen == [model]


class TestInstallMambaContextParallel:
    def test_wraps_and_is_idempotent(self) -> None:
        model = DummyMambaMixer()
        mesh = _cp_mesh()

        assert install_mamba_context_parallel(model, mesh) is True
        assert model._agilerl_mamba_cp is True
        assert install_mamba_context_parallel(model, mesh) is False

    def test_composes_with_tensor_parallel_without_replacing_tp_forward(
        self, monkeypatch
    ) -> None:
        model = DummyTpMambaMixer()
        object.__setattr__(model, "_agilerl_mamba_tp", True)
        object.__setattr__(model, "_tp_degree", 2)
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mixer.all_to_all_seq_to_hidden",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mixer.all_to_all_hidden_to_seq",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mixer.dist.get_rank",
            lambda _group: 0,
        )

        assert install_mamba_context_parallel(model, _cp_mesh()) is True
        assert model.forward.__func__ is DummyTpMambaMixer.forward
        assert model.use_mem_eff_path is False

        width = model.intermediate_size
        hidden = torch.zeros(2, 4, width)
        out = model(hidden)

        assert out.shape == (2, 4, width)
        assert model.scan_heads == 2
        assert model.projected_last_dim == width
        assert not isinstance(model.in_proj, MixerInProjBypass)

    def test_rejects_when_groups_do_not_divide_tp_times_cp(self) -> None:
        model = DummyMambaMixer(num_heads=4, n_groups=2)
        object.__setattr__(model, "_agilerl_mamba_tp", True)
        object.__setattr__(model, "_tp_degree", 2)

        with pytest.raises(ValueError, match="n_groups"):
            install_mamba_context_parallel(model, _cp_mesh())

    def test_rejects_heads_that_do_not_divide_cp(self) -> None:
        model = DummyMambaMixer(num_heads=3, n_groups=1)

        with pytest.raises(ValueError, match="mamba_num_heads"):
            install_mamba_context_parallel(model, _cp_mesh())

    def test_forward_swaps_projections_without_module_assign(self, monkeypatch):
        model = DummyMambaMixer()
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mixer.all_to_all_seq_to_hidden",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mixer.all_to_all_hidden_to_seq",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mixer.dist.get_rank",
            lambda _group: 0,
        )
        install_mamba_context_parallel(model, _cp_mesh())

        width = model.intermediate_size
        hidden = torch.zeros(2, 4, width)
        out = model(hidden)

        assert out.shape == (2, 4, width)
        assert isinstance(model.in_proj, MixerInProjBypass)
        model(hidden)
        assert isinstance(model.in_proj, MixerInProjBypass)


class TestInstallGdnContextParallel:
    def test_wraps_and_is_idempotent(self) -> None:
        model = DummyGatedDeltaNet()

        assert install_gdn_context_parallel(model, _cp_mesh()) is True
        assert model._agilerl_gdn_cp is True
        assert install_gdn_context_parallel(model, _cp_mesh()) is False

    def test_tensor_parallel_flag_still_gathers(self) -> None:
        model = DummyGatedDeltaNet()
        object.__setattr__(model, "_tp_degree", 2)

        assert install_gdn_context_parallel(model, _cp_mesh()) is True
        assert model._agilerl_gdn_cp is True


class _LoraInProj(nn.Module):
    """Base shard is local; LoRA B is still the full TP width."""

    def __init__(self, lora_out: int) -> None:
        super().__init__()
        self.base_layer = nn.Linear(4, 4, bias=False)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, lora_out, bias=False)
        object.__setattr__(self.lora_B, "tp_row_ranges", ((0, 4),))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


class TestCallLocal:
    def test_slices_lora_b_to_tp_row_ranges(self) -> None:
        torch.manual_seed(0)
        layer = _LoraInProj(lora_out=8)
        x = torch.randn(2, 3, 4)

        out = _call_local(layer, x)

        ranges = layer.lora_B.tp_row_ranges
        sliced_b = torch.cat([layer.lora_B.weight[start:stop] for start, stop in ranges])
        expected = torch.nn.functional.linear(x, layer.base_layer.weight) + (
            torch.nn.functional.linear(
                torch.nn.functional.linear(x, layer.lora_A.weight), sliced_b
            )
        )
        assert out.shape == (2, 3, 4)
        assert torch.allclose(out, expected)

    def test_leaves_already_local_lora_b(self) -> None:
        torch.manual_seed(1)
        layer = _LoraInProj(lora_out=4)
        x = torch.randn(2, 3, 4)

        out = _call_local(layer, x)

        assert out.shape == (2, 3, 4)
        assert torch.allclose(out, layer(x))


class TestCatalogIsolation:
    def test_tensor_parallel_module_does_not_import_context_parallel(self) -> None:
        from agilerl.architectures.nemotron_h import tensor_parallel as mamba_tp

        assert "context_parallel" not in Path(mamba_tp.__file__).read_text()

    def test_family_context_parallel_plans_still_return_both_families(self) -> None:
        from agilerl.architectures import family_context_parallel_plans
        from agilerl.architectures.nemotron_h.context_parallel import (
            NEMOTRON_H_CONTEXT_PARALLEL_PLAN,
        )
        from agilerl.architectures.qwen3_5.context_parallel import (
            QWEN3_5_CONTEXT_PARALLEL_PLAN,
        )

        assert family_context_parallel_plans() == [
            NEMOTRON_H_CONTEXT_PARALLEL_PLAN,
            QWEN3_5_CONTEXT_PARALLEL_PLAN,
        ]
