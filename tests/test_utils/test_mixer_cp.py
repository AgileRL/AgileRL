# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""CPU wrap contracts for Mamba/GDN context parallel."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)
from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
    GraniteMoeHybridMambaLayer,
    GraniteMoeHybridRMSNormGated,
)
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from agilerl.architectures.nemotron_h.tensor_parallel import (
    apply_mamba_tensor_parallel,
    mamba_row_orders,
    shard_mamba_mixer,
)
from agilerl.architectures.runtime import ContextParallelPlan
from agilerl.distributed import build_parallel_mesh
from agilerl.distributed.context_parallel.gdn import (
    _fla_build_cp_context,
    _fla_causal_conv1d,
    _fla_chunk_gated_delta_rule,
    _fla_chunk_kda,
    install_gdn_context_parallel,
    wraps_gdn_context_parallel,
)
from agilerl.distributed.context_parallel.mamba import (
    MambaDims,
    MixerGatedNormThenSeq,
    MixerInProjBypass,
    install_mamba_context_parallel,
    mamba_cp_group_index,
    mamba_cp_index,
    pack_mamba_projected_for_cp,
)
from agilerl.distributed.context_parallel.mixer import (
    apply_mixer_context_parallel,
    call_local,
    local_param_slots,
)
from agilerl.distributed.context_parallel.sequence import CP_SCAN_CHUNK
from agilerl.distributed.context_parallel.ulysses import (
    ULYSSES_PARAMS,
    PublishedUlyssesParams,
    clear_ulysses_params,
    update_ulysses_params,
)
from agilerl.distributed.tensor_parallel import sum_tp_partial_grads
from tests.parallel_helpers import assert_exact, publish_varlen_offsets
from tests.rank_launch import run_ranks


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
        self.cuda_scans = 0

    def torch_forward(
        self, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ):
        self.scan_heads = int(self.num_heads)
        self.scan_seq = int(hidden_states.shape[1])
        self.scan_mask = kwargs.get("attention_mask")
        if self.scan_mask is None and len(args) >= 2:
            self.scan_mask = args[1]
        projected = self.in_proj(hidden_states)
        self.projected_last_dim = int(projected.shape[-1])
        return self.out_proj(projected)

    def cuda_kernels_forward(
        self, hidden_states: torch.Tensor, *args: object, **kwargs: object
    ):
        self.cuda_scans += 1
        return DummyTpMambaMixer.torch_forward(self, hidden_states, *args, **kwargs)

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
            inner = getattr(self, "_mamba_inner_forward", None)
            if inner is None:
                return self.torch_forward(hidden_states, *args, **kwargs)
            return inner(hidden_states, *args, **kwargs)
        finally:
            (
                self.num_heads,
                self.n_groups,
                self.intermediate_size,
                self.conv_dim,
            ) = saved


class DummyCudaMambaMixer(DummyTpMambaMixer):
    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        return self.cuda_kernels_forward(hidden_states, *args, **kwargs)


def _identity_cp_collectives(monkeypatch) -> None:
    monkeypatch.setattr(
        "agilerl.distributed.context_parallel.mamba.all_to_all_seq_to_hidden",
        lambda data, _cp, _group: data,
    )
    monkeypatch.setattr(
        "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
        lambda data, _cp, _group: data,
    )
    monkeypatch.setattr(
        "agilerl.distributed.context_parallel.mamba.dist.get_rank",
        lambda _group: 0,
    )
    monkeypatch.setattr(
        "agilerl.distributed.context_parallel.mamba.pack_mamba_projected_for_cp",
        lambda projected, **_kwargs: projected,
    )


def _fake_conv(*args: object, **kwargs: object) -> object:
    return kwargs.get("x", args[0] if args else None)


def _fake_scan(*args: object, **kwargs: object) -> tuple[object, None]:
    return args[0], None


class DummyGatedDeltaNet(nn.Module):
    def __init__(self, num_k_heads: int = 4, n_groups: int = 2) -> None:
        super().__init__()
        self.num_k_heads = num_k_heads
        self.n_groups = n_groups
        self.conv_kernel_size = 4
        self.proj = nn.Identity()
        self.seen_kwargs: dict[str, object] = {}

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object):
        self.seen_kwargs = dict(kwargs)
        return hidden_states


def _importable_fla(monkeypatch) -> None:
    _fake_fla_module(
        monkeypatch, "fla.ops.cp", build_cp_context=lambda *_args, **_kwargs: "ctx"
    )
    _fake_fla_module(
        monkeypatch, "fla.ops.gated_delta_rule", chunk_gated_delta_rule=_fake_scan
    )


@pytest.fixture
def gdn_kernels(monkeypatch) -> types.ModuleType:
    """DummyGatedDeltaNet's modeling module with stock kernels, and FLA importable."""
    module = types.ModuleType("gdn_dummy_modeling")
    module.causal_conv1d_fn = _fake_conv
    module.torch_chunk_gated_delta_rule = _fake_scan
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(DummyGatedDeltaNet, "__module__", module.__name__)
    _importable_fla(monkeypatch)
    return module


def _mark_composed_tp(model: DummyTpMambaMixer, degree: int = 2) -> None:
    object.__setattr__(model, "_mamba_inner_forward", model.torch_forward)
    object.__setattr__(model, "_agilerl_mamba_tp", True)
    object.__setattr__(model, "_tp_degree", degree)


def _cp_mesh(size: int = 2) -> MagicMock:
    mesh = MagicMock()
    mesh.size.return_value = size
    mesh.get_group.return_value = object()
    mesh.get_local_rank.return_value = 0
    mesh.device_type = "cpu"
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
    @pytest.fixture(autouse=True)
    def _publish_training_cu_seqlens(self) -> Iterator[None]:
        publish_varlen_offsets(torch.tensor([0, 4, 8], dtype=torch.int32))
        yield
        clear_ulysses_params()

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
        _mark_composed_tp(model)
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_seq_to_hidden",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.dist.get_rank",
            lambda _group: 0,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.pack_mamba_projected_for_cp",
            lambda projected, **_kwargs: projected,
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

    def test_composed_inner_scan_uses_cuda_kernels_forward(self, monkeypatch) -> None:
        model = DummyTpMambaMixer()
        _mark_composed_tp(model)
        object.__setattr__(model, "_mamba_inner_forward", model.cuda_kernels_forward)
        _identity_cp_collectives(monkeypatch)
        install_mamba_context_parallel(model, _cp_mesh())
        width = model.intermediate_size
        hidden = torch.zeros(2, 4, width)

        out = model(hidden)

        assert out.shape == (2, 4, width)
        assert model.cuda_scans == 1
        assert model.scan_heads == 2
        assert not isinstance(model.in_proj, MixerInProjBypass)

    def test_composed_inner_scan_skips_the_fused_training_return(
        self, monkeypatch
    ) -> None:
        model = DummyTpMambaMixer()
        model.train()

        def hf_forward(
            hidden_states: torch.Tensor, *_args: object, **_kwargs: object
        ) -> torch.Tensor:
            model.saw_training = model.training
            if model.training:
                return hidden_states
            projected = model.in_proj(hidden_states)
            model.scan_heads = int(model.num_heads)
            model.projected_last_dim = int(projected.shape[-1])
            return model.out_proj(projected)

        object.__setattr__(model, "_mamba_inner_forward", hf_forward)
        object.__setattr__(model, "_agilerl_mamba_tp", True)
        object.__setattr__(model, "_tp_degree", 2)
        _identity_cp_collectives(monkeypatch)

        install_mamba_context_parallel(model, _cp_mesh())
        width = model.intermediate_size
        hidden = torch.randn(2, 4, width)

        out = model(hidden)

        assert out.shape == (2, 4, width)
        assert model.saw_training is False
        assert model.training is True
        assert model.scan_heads == 2

    def test_inner_scan_uses_the_mixer_forward_dispatch(self, monkeypatch) -> None:
        model = DummyCudaMambaMixer()
        _identity_cp_collectives(monkeypatch)
        install_mamba_context_parallel(model, _cp_mesh())
        width = model.intermediate_size
        hidden = torch.zeros(2, 4, width)

        out = model(hidden)

        assert out.shape == (2, 4, width)
        assert model.cuda_scans == 1
        assert model.scan_heads == 4

    def test_forward_skips_the_fused_training_return(self, monkeypatch) -> None:
        class FusedWhenTraining(DummyMambaMixer):
            def forward(
                self, hidden_states: torch.Tensor, *_args: object, **_kwargs: object
            ) -> torch.Tensor:
                self.saw_training = self.training
                if self.training:
                    return hidden_states
                return self.out_proj(self.in_proj(hidden_states))

        model = FusedWhenTraining()
        model.train()
        _identity_cp_collectives(monkeypatch)
        install_mamba_context_parallel(model, _cp_mesh())
        width = model.intermediate_size
        hidden = torch.randn(2, 4, width)

        out = model(hidden)

        assert out.shape == (2, 4, width)
        assert model.saw_training is False
        assert model.training is True

    def test_inner_scan_drops_a_sequence_shard_mask(self, monkeypatch) -> None:
        model = DummyTpMambaMixer()
        _mark_composed_tp(model)
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_seq_to_hidden",
            lambda data, cp, _group: data.repeat_interleave(cp, dim=1),
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            lambda data, cp, _group: data[:, : data.shape[1] // cp],
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.dist.get_rank",
            lambda _group: 0,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.pack_mamba_projected_for_cp",
            lambda projected, **_kwargs: projected,
        )
        install_mamba_context_parallel(model, _cp_mesh())
        width = model.intermediate_size
        hidden = torch.zeros(2, 4, width)
        mask = torch.ones(2, 4)

        out = model(hidden, None, mask)

        assert out.shape == (2, 4, width)
        assert model.scan_seq == 8
        assert model.scan_mask is None

    def test_rejects_tp_local_groups_and_cp_that_do_not_divide_one_another(
        self,
    ) -> None:
        model = DummyMambaMixer(num_heads=12, n_groups=6)
        object.__setattr__(model, "_agilerl_mamba_tp", True)
        object.__setattr__(model, "_tp_degree", 2)

        with pytest.raises(ValueError, match="n_groups"):
            install_mamba_context_parallel(model, _cp_mesh())

    def test_rejects_heads_that_do_not_divide_cp(self) -> None:
        model = DummyMambaMixer(num_heads=3, n_groups=1)

        with pytest.raises(ValueError, match="mamba_num_heads"):
            install_mamba_context_parallel(model, _cp_mesh())

    def test_wraps_fewer_groups_than_cp(self) -> None:
        model = DummyMambaMixer(num_heads=4, n_groups=1)

        assert install_mamba_context_parallel(model, _cp_mesh()) is True

    def test_rejects_groups_and_cp_that_do_not_divide_one_another(self) -> None:
        model = DummyMambaMixer(num_heads=6, n_groups=3)

        with pytest.raises(ValueError, match="n_groups"):
            install_mamba_context_parallel(model, _cp_mesh())

    def test_composed_norm_spanning_cp_ranks_runs_on_the_sequence_shard(
        self, monkeypatch
    ) -> None:
        # Arrange
        model = DummyTpMambaMixer()
        norm = model.norm
        out_proj = model.out_proj
        seen: dict[str, nn.Module] = {}

        def inner(hidden_states: torch.Tensor, *_args: object, **_kwargs: object):
            seen["norm"] = model.norm
            seen["out_proj"] = model.out_proj
            return model.torch_forward(hidden_states)

        _mark_composed_tp(model)
        object.__setattr__(model, "_mamba_inner_forward", inner)
        _identity_cp_collectives(monkeypatch)
        install_mamba_context_parallel(
            model, _cp_mesh(), norm_group_size=model.intermediate_size
        )

        # Act
        model(torch.zeros(2, 4, model.intermediate_size))

        # Assert
        assert isinstance(seen["norm"], MixerGatedNormThenSeq)
        assert seen["out_proj"] is out_proj
        assert model.norm is norm

    def test_scan_gets_a_shape_only_hidden_placeholder(self, monkeypatch) -> None:
        # Arrange
        model = DummyTpMambaMixer()
        seen: dict[str, torch.Tensor] = {}

        def inner(hidden_states: torch.Tensor, *_args: object, **_kwargs: object):
            seen["hidden"] = hidden_states
            return model.torch_forward(hidden_states)

        _mark_composed_tp(model)
        object.__setattr__(model, "_mamba_inner_forward", inner)
        _identity_cp_collectives(monkeypatch)
        install_mamba_context_parallel(model, _cp_mesh())
        hidden = torch.zeros(2, 4, model.intermediate_size, dtype=torch.bfloat16)

        # Act
        model(hidden)

        # Assert
        placeholder = seen["hidden"]
        assert placeholder.shape == hidden.shape
        assert placeholder.dtype == torch.bfloat16
        assert placeholder.untyped_storage().nbytes() == placeholder.element_size()

    def test_forward_restores_the_projections_it_swaps(self, monkeypatch):
        model = DummyMambaMixer()
        in_proj = model.in_proj
        out_proj = model.out_proj
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_seq_to_hidden",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.dist.get_rank",
            lambda _group: 0,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.pack_mamba_projected_for_cp",
            lambda projected, **_kwargs: projected,
        )
        install_mamba_context_parallel(model, _cp_mesh())

        width = model.intermediate_size
        hidden = torch.zeros(2, 4, width)
        out = model(hidden)
        again = model(hidden)

        assert out.shape == (2, 4, width)
        assert again.shape == (2, 4, width)
        assert model.in_proj is in_proj
        assert model.out_proj is out_proj

    def test_cp_forward_packs_in_proj_before_the_hidden_all_to_all(
        self, monkeypatch
    ) -> None:
        packed_widths: list[int] = []

        def pack(
            projected: torch.Tensor,
            **_kwargs: object,
        ) -> torch.Tensor:
            packed_widths.append(int(projected.shape[-1]))
            return projected

        model = DummyMambaMixer()
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_seq_to_hidden",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            lambda data, _cp, _group: data,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.dist.get_rank",
            lambda _group: 0,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.pack_mamba_projected_for_cp",
            pack,
        )
        install_mamba_context_parallel(model, _cp_mesh())
        width = model.intermediate_size

        model(torch.zeros(2, 4, width))

        assert packed_widths == [width]

    @pytest.mark.parametrize("num_heads", [2, 8])
    def test_cp_forward_hands_in_proj_rows_with_a_stride_of_8(
        self, monkeypatch, num_heads: int
    ) -> None:
        # Arrange
        seen: list[torch.Tensor] = []

        class RecordingMixer(DummyMambaMixer):
            def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
                projected = self.in_proj(hidden_states)
                seen.append(projected)
                return projected[..., : self.intermediate_size]

        model = RecordingMixer(num_heads=num_heads, n_groups=2)
        _identity_cp_collectives(monkeypatch)
        install_mamba_context_parallel(model, _cp_mesh())
        width = 2 * model.intermediate_size + 2 * 2 * 4 + num_heads
        hidden = torch.randn(2, 4, width)

        # Act
        model(hidden)

        # Assert
        assert torch.equal(seen[0], hidden)
        assert seen[0].stride(-2) % 8 == 0

    def test_unpublished_cu_seqlens_runs_the_stock_forward(self, monkeypatch) -> None:
        exchanges: list[tuple[int, ...]] = []

        def exchange(data: torch.Tensor, _cp: int, _group: object) -> torch.Tensor:
            exchanges.append(tuple(data.shape))
            return data

        model = DummyMambaMixer()
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_seq_to_hidden",
            exchange,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            exchange,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.dist.get_rank",
            lambda _group: 0,
        )
        install_mamba_context_parallel(model, _cp_mesh())
        clear_ulysses_params()
        hidden = torch.zeros(2, 4, model.intermediate_size)

        out = model(hidden)

        assert torch.equal(out, hidden)
        assert exchanges == []

    def test_composed_unpublished_cu_seqlens_runs_the_stock_inner(
        self, monkeypatch
    ) -> None:
        exchanges: list[tuple[int, ...]] = []

        def exchange(data: torch.Tensor, _cp: int, _group: object) -> torch.Tensor:
            exchanges.append(tuple(data.shape))
            return data

        model = DummyTpMambaMixer()
        _mark_composed_tp(model)
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_seq_to_hidden",
            exchange,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            exchange,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.dist.get_rank",
            lambda _group: 0,
        )
        install_mamba_context_parallel(model, _cp_mesh())
        clear_ulysses_params()
        hidden = torch.zeros(2, 4, model.intermediate_size)

        out = model(hidden)

        assert out.shape == hidden.shape
        assert model.scan_heads == 4
        assert exchanges == []


class TestPackMambaProjectedForCp:
    def test_chunk_after_pack_is_the_local_gate_xbc_dt(self) -> None:
        num_heads = 4
        head_dim = 2
        n_groups = 2
        ssm_state = 2
        cp = 2
        batch, seq = 2, 3
        d_inner = num_heads * head_dim
        group_state = n_groups * ssm_state
        conv_dim = d_inner + 2 * group_state
        width = d_inner + conv_dim + num_heads
        projected = torch.arange(batch * seq * width, dtype=torch.float32).reshape(
            batch, seq, width
        )

        packed = pack_mamba_projected_for_cp(
            projected,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            ssm_state_size=ssm_state,
            cp=cp,
            group_index=mamba_cp_group_index(num_heads, n_groups, cp, "cpu"),
        )

        gate, xbc, dt = projected.split([d_inner, conv_dim, num_heads], dim=-1)
        x_proj, b_proj, c_proj = xbc.split([d_inner, group_state, group_state], dim=-1)
        local_inner = d_inner // cp
        local_group = group_state // cp
        local_heads = num_heads // cp
        for rank in range(cp):
            expected = torch.cat(
                [
                    gate[..., rank * local_inner : (rank + 1) * local_inner],
                    x_proj[..., rank * local_inner : (rank + 1) * local_inner],
                    b_proj[..., rank * local_group : (rank + 1) * local_group],
                    c_proj[..., rank * local_group : (rank + 1) * local_group],
                    dt[..., rank * local_heads : (rank + 1) * local_heads],
                ],
                dim=-1,
            )

            assert torch.equal(packed.chunk(cp, dim=-1)[rank], expected)

    def test_fewer_groups_than_cp_give_each_rank_its_heads_group(self) -> None:
        # Arrange
        num_heads = 4
        head_dim = 2
        n_groups = 2
        ssm_state = 2
        cp = 4
        d_inner = num_heads * head_dim
        group_state = n_groups * ssm_state
        conv_dim = d_inner + 2 * group_state
        width = d_inner + conv_dim + num_heads
        projected = torch.arange(2 * 3 * width, dtype=torch.float32).reshape(
            2, 3, width
        )

        # Act
        packed = pack_mamba_projected_for_cp(
            projected,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            ssm_state_size=ssm_state,
            cp=cp,
            group_index=mamba_cp_group_index(num_heads, n_groups, cp, "cpu"),
        )

        # Assert
        gate, xbc, dt = projected.split([d_inner, conv_dim, num_heads], dim=-1)
        x_proj, b_proj, c_proj = xbc.split([d_inner, group_state, group_state], dim=-1)
        for rank in range(cp):
            # Two heads per group, one head per rank: ranks 0-1 hold group 0.
            group = rank // 2
            states = slice(group * ssm_state, (group + 1) * ssm_state)
            expected = torch.cat(
                [
                    gate[..., rank * head_dim : (rank + 1) * head_dim],
                    x_proj[..., rank * head_dim : (rank + 1) * head_dim],
                    b_proj[..., states],
                    c_proj[..., states],
                    dt[..., rank : rank + 1],
                ],
                dim=-1,
            )
            assert torch.equal(packed.chunk(cp, dim=-1)[rank], expected)

    def test_raw_last_dim_chunk_is_not_the_local_projection(self) -> None:
        num_heads = 4
        head_dim = 2
        n_groups = 2
        ssm_state = 2
        projected = torch.arange(1 * 1 * 28, dtype=torch.float32).reshape(1, 1, 28)

        packed_rank0 = pack_mamba_projected_for_cp(
            projected,
            num_heads=num_heads,
            head_dim=head_dim,
            n_groups=n_groups,
            ssm_state_size=ssm_state,
            cp=2,
            group_index=mamba_cp_group_index(num_heads, n_groups, 2, "cpu"),
        ).chunk(2, dim=-1)[0]

        assert not torch.equal(projected.chunk(2, dim=-1)[0], packed_rank0)

    def test_rejects_a_width_that_is_not_gate_xbc_dt(self) -> None:
        with pytest.raises(ValueError, match=r"gate\+xBC\+dt"):
            pack_mamba_projected_for_cp(
                torch.zeros(1, 2, 7),
                num_heads=4,
                head_dim=2,
                n_groups=2,
                ssm_state_size=2,
                cp=2,
                group_index=mamba_cp_group_index(4, 2, 2, "cpu"),
            )

    def test_cp_one_is_identity(self) -> None:
        data = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

        assert torch.equal(
            pack_mamba_projected_for_cp(
                data,
                num_heads=2,
                head_dim=1,
                n_groups=1,
                ssm_state_size=1,
                cp=1,
                group_index=mamba_cp_group_index(2, 1, 1, "cpu"),
            ),
            data,
        )


class TestWrapsGdnContextParallel:
    def test_finds_a_wrapped_mixer_in_the_model(self, gdn_kernels) -> None:
        mixer = DummyGatedDeltaNet()
        install_gdn_context_parallel(mixer, _cp_mesh())

        assert wraps_gdn_context_parallel(nn.Sequential(nn.Identity(), mixer))

    def test_unwrapped_model_has_none(self) -> None:
        assert not wraps_gdn_context_parallel(nn.Sequential(DummyGatedDeltaNet()))


class TestInstallGdnContextParallel:
    def test_wraps_and_is_idempotent(self, gdn_kernels) -> None:
        model = DummyGatedDeltaNet()

        assert install_gdn_context_parallel(model, _cp_mesh()) is True
        assert model._agilerl_gdn_cp is True
        assert install_gdn_context_parallel(model, _cp_mesh()) is False

    def test_tensor_parallel_flag_leaves_the_sequence_shard(self, gdn_kernels) -> None:
        model = DummyGatedDeltaNet()
        object.__setattr__(model, "_tp_degree", 2)

        assert install_gdn_context_parallel(model, _cp_mesh()) is True
        assert model._agilerl_gdn_cp is True
        hidden = torch.zeros(1, 2, 2)
        seq_idx = torch.zeros(1, 2, dtype=torch.int32)

        out = model(hidden, seq_idx=seq_idx)

        assert out.shape == hidden.shape
        assert model.seen_kwargs["seq_idx"] is seq_idx

    def test_missing_kernels_raise(self) -> None:
        model = nn.Linear(2, 2)

        with pytest.raises(RuntimeError, match="chunk_gated_delta_rule"):
            install_gdn_context_parallel(model, _cp_mesh())

    @pytest.mark.parametrize(
        ("bounds", "match"),
        [
            ([0, 8, 16, 24], "got 2 rows and 3 sequences"),
            ([0, 8, 20], "rows must share one length"),
        ],
        ids=["row_count", "row_length"],
    )
    def test_rows_must_match_the_published_sequences(
        self, gdn_kernels, bounds: list[int], match: str
    ) -> None:
        install_gdn_context_parallel(DummyGatedDeltaNet(), _cp_mesh())
        publish_varlen_offsets(torch.tensor(bounds, dtype=torch.int32))

        try:
            with pytest.raises(RuntimeError, match=match):
                gdn_kernels.torch_chunk_gated_delta_rule(torch.zeros(2, 4, 1, 2))
        finally:
            clear_ulysses_params()

    def test_row_final_states_are_concatenated(self, monkeypatch, gdn_kernels) -> None:
        # Arrange
        def scan(*args: object, **_kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            query = args[0]
            assert isinstance(query, torch.Tensor)
            return query, query.sum(dim=1)

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_chunk_gated_delta_rule",
            scan,
        )
        install_gdn_context_parallel(DummyGatedDeltaNet(), _cp_mesh())
        publish_varlen_offsets(torch.tensor([0, 8, 16], dtype=torch.int32))
        query = torch.arange(16, dtype=torch.float32).reshape(2, 4, 1, 2)

        # Act
        try:
            output, state = gdn_kernels.torch_chunk_gated_delta_rule(
                query, query, query
            )
        finally:
            clear_ulysses_params()

        # Assert
        assert torch.equal(output, query)
        assert torch.equal(state, query.sum(dim=1))

    @pytest.mark.parametrize(
        "failing", ["fla.ops.cp", "fla.ops.gated_delta_rule"], ids=["cp", "scan"]
    )
    def test_failing_fla_import_raises_and_leaves_the_kernels(
        self, monkeypatch, failing: str
    ) -> None:
        # Arrange
        config = Qwen3_5TextConfig(
            hidden_size=16,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
        )
        mixer = Qwen3_5GatedDeltaNet(config, 0)
        module = sys.modules[type(mixer).__module__]
        conv = module.causal_conv1d_fn
        scan = module.torch_chunk_gated_delta_rule
        _importable_fla(monkeypatch)
        real_import = importlib.import_module

        def import_module(name: str, package: str | None = None) -> types.ModuleType:
            if name == failing:
                msg = "undefined symbol: fla_kernel"
                raise ImportError(msg)
            return real_import(name, package)

        monkeypatch.setattr(importlib, "import_module", import_module)

        # Act
        with pytest.raises(
            RuntimeError,
            match=r"needs flash-linear-attention \(ImportError: undefined symbol",
        ):
            install_gdn_context_parallel(mixer, _cp_mesh())

        # Assert
        assert not getattr(mixer, "_agilerl_gdn_cp", False)
        assert module.causal_conv1d_fn is conv
        assert module.torch_chunk_gated_delta_rule is scan

    def test_kernels_receive_cp_context_and_keep_the_shard(
        self, monkeypatch, gdn_kernels
    ) -> None:
        model = DummyGatedDeltaNet()
        seen: dict[str, object] = {}

        def fla_conv(
            x: torch.Tensor,
            weight: torch.Tensor | None,
            bias: torch.Tensor | None,
            activation: str | None,
            cp_context: object,
        ) -> tuple[torch.Tensor, None]:
            seen["conv_x"] = x.shape
            seen["conv_context"] = cp_context
            return x, None

        def scan(*args: object, **kwargs: object) -> tuple[object, None]:
            seen["scan_context"] = kwargs.get("cp_context")
            seen["scan_q"] = args[0]
            return args[0], None

        built: list[list[int]] = []

        def build(cu_seqlens: torch.Tensor, _group: object, **_kwargs: object) -> str:
            built.append(cu_seqlens.tolist())
            return "ctx"

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_causal_conv1d",
            fla_conv,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_build_cp_context",
            build,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_chunk_gated_delta_rule",
            scan,
        )
        install_gdn_context_parallel(model, _cp_mesh())
        hidden = torch.zeros(1, 6, 4)
        seq_idx = torch.zeros(1, 4, dtype=torch.int32)
        publish_varlen_offsets(torch.tensor([0, 3, 8], dtype=torch.int32))
        try:
            out = model(hidden, seq_idx=seq_idx, cu_seq_lens_q=torch.tensor([0, 3, 8]))
            conv_out = gdn_kernels.causal_conv1d_fn(
                x=hidden,
                weight=torch.zeros(6, 4),
                bias=None,
                activation="silu",
                seq_idx=seq_idx,
            )
            query = torch.zeros(1, 4, 2, 2)
            scan_out = gdn_kernels.torch_chunk_gated_delta_rule(
                query, query, query, g=query
            )
        finally:
            clear_ulysses_params()

        assert out.shape == hidden.shape
        assert model.seen_kwargs["seq_idx"] is seq_idx
        assert conv_out.shape == hidden.shape
        assert seen["conv_x"] == (1, 4, 6)
        assert seen["conv_context"] == "ctx"
        assert seen["scan_context"] == "ctx"
        # Conv and scan share the micro-batch's context.
        assert built == [[0, 3, 8]]
        assert scan_out[0] is query

    def test_each_row_is_its_own_sequence(self, monkeypatch, gdn_kernels) -> None:
        built: list[list[int]] = []
        rows: list[tuple[int, ...]] = []

        def scan(*args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
            query = args[0]
            assert isinstance(query, torch.Tensor)
            rows.append(tuple(query.shape))
            assert kwargs["cp_context"] == "ctx"
            return query, None

        def build(
            cu_seqlens: torch.Tensor, _group: object, conv1d_kernel_size: int
        ) -> str:
            built.append(cu_seqlens.tolist())
            assert conv1d_kernel_size == 4
            return "ctx"

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_build_cp_context",
            build,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_chunk_gated_delta_rule",
            scan,
        )
        install_gdn_context_parallel(DummyGatedDeltaNet(), _cp_mesh())
        publish_varlen_offsets(torch.tensor([0, 8, 16], dtype=torch.int32))
        query = torch.zeros(2, 4, 1, 2)
        try:
            output, state = gdn_kernels.torch_chunk_gated_delta_rule(
                query, query, query
            )
        finally:
            clear_ulysses_params()

        assert rows == [(1, 4, 1, 2), (1, 4, 1, 2)]
        assert built == [[0, 8]]
        assert output.shape == query.shape
        assert state is None

    def test_scan_chunk_divides_the_rank_boundary(
        self, monkeypatch, gdn_kernels
    ) -> None:
        seen: list[int] = []

        def scan(*args: object, **kwargs: object) -> tuple[object, None]:
            chunk_size = kwargs["chunk_size"]
            assert isinstance(chunk_size, int)
            seen.append(chunk_size)
            return args[0], None

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_chunk_gated_delta_rule",
            scan,
        )
        install_gdn_context_parallel(DummyGatedDeltaNet(), _cp_mesh())
        cases = (
            ([0, 128], 64),
            ([0, 160], 16),
            ([0, 192], 32),
            ([0, 32, 128], 32),
            ([0, 50, 128], 64),
        )
        query = torch.zeros(1, 2, 1, 2)
        try:
            for bounds, _chunk in cases:
                publish_varlen_offsets(torch.tensor(bounds, dtype=torch.int32))
                gdn_kernels.torch_chunk_gated_delta_rule(query, query, query)
        finally:
            clear_ulysses_params()

        assert seen == [expected for _bounds, expected in cases]

    def test_unpublished_cu_seqlens_runs_the_stock_kernels(
        self, monkeypatch, gdn_kernels
    ) -> None:
        built: list[object] = []
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_build_cp_context",
            lambda *_args, **_kwargs: built.append("ctx") or "ctx",
        )
        install_gdn_context_parallel(DummyGatedDeltaNet(), _cp_mesh())
        clear_ulysses_params()
        hidden = torch.zeros(1, 4, 4)
        query = torch.zeros(1, 4, 1, 2)

        conv_out = gdn_kernels.causal_conv1d_fn(x=hidden)
        output = gdn_kernels.torch_chunk_gated_delta_rule(query)

        assert conv_out is hidden
        assert output[0] is query
        assert built == []

    def test_short_cu_seqlens_raises(self, gdn_kernels) -> None:
        install_gdn_context_parallel(DummyGatedDeltaNet(), _cp_mesh())
        publish_varlen_offsets(torch.tensor([0], dtype=torch.int32))

        try:
            with pytest.raises(RuntimeError, match="full-sequence cu_seqlens"):
                gdn_kernels.torch_chunk_gated_delta_rule(torch.zeros(1, 4, 1, 2))
        finally:
            clear_ulysses_params()

    def test_modeling_module_kernels_receive_cp_context(self, monkeypatch) -> None:
        import sys
        import types

        module = types.ModuleType("gdn_modeling_mod")

        def causal_conv1d_fn(*_args: object, **_kwargs: object) -> None:
            msg = "original conv"
            raise AssertionError(msg)

        def torch_chunk_gated_delta_rule(*_args: object, **_kwargs: object) -> None:
            msg = "original scan"
            raise AssertionError(msg)

        module.causal_conv1d_fn = causal_conv1d_fn
        module.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        sys.modules["gdn_modeling_mod"] = module

        class Mixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_kernel_size = 4

        Mixer.__module__ = "gdn_modeling_mod"
        seen: dict[str, object] = {}

        def fla_conv(
            x: torch.Tensor,
            weight: torch.Tensor | None,
            bias: torch.Tensor | None,
            activation: str | None,
            cp_context: object,
        ) -> tuple[torch.Tensor, None]:
            seen["conv_x"] = tuple(x.shape)
            seen["conv_context"] = cp_context
            return x, None

        def fla_scan(*args: object, **kwargs: object) -> tuple[object, None]:
            seen["scan_context"] = kwargs.get("cp_context")
            return args[0], None

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_causal_conv1d",
            fla_conv,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_chunk_gated_delta_rule",
            fla_scan,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_build_cp_context",
            lambda *_args, **_kwargs: "ctx",
        )
        _importable_fla(monkeypatch)
        model = Mixer()
        publish_varlen_offsets(torch.tensor([0, 8], dtype=torch.int32))
        try:
            assert install_gdn_context_parallel(model, _cp_mesh()) is True
            hidden = torch.zeros(1, 8, 4)
            out = module.causal_conv1d_fn(
                hidden, torch.zeros(8, 4), None, activation="silu"
            )
            query = torch.zeros(1, 4, 1, 2)
            module.torch_chunk_gated_delta_rule(
                query, query, query, g=query, beta=query
            )
        finally:
            clear_ulysses_params()
            del sys.modules["gdn_modeling_mod"]

        assert model._agilerl_gdn_cp is True
        assert tuple(out.shape) == (1, 8, 4)
        assert seen["conv_x"] == (1, 4, 8)
        assert seen["conv_context"] == "ctx"
        assert seen["scan_context"] == "ctx"

    def test_unpublished_modeling_module_kernels_run_stock(self, monkeypatch) -> None:
        module = types.ModuleType("gdn_modeling_mod_stock")
        seen: list[str] = []

        def causal_conv1d_fn(*_args: object, **_kwargs: object) -> str:
            seen.append("conv")
            return "stock-conv"

        def torch_chunk_gated_delta_rule(*_args: object, **_kwargs: object) -> str:
            seen.append("scan")
            return "stock-scan"

        module.causal_conv1d_fn = causal_conv1d_fn
        module.torch_chunk_gated_delta_rule = torch_chunk_gated_delta_rule
        sys.modules["gdn_modeling_mod_stock"] = module

        class Mixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_kernel_size = 4

        Mixer.__module__ = "gdn_modeling_mod_stock"
        _importable_fla(monkeypatch)
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_build_cp_context",
            lambda *_args, **_kwargs: seen.append("ctx") or "ctx",
        )
        model = Mixer()
        try:
            assert install_gdn_context_parallel(model, _cp_mesh()) is True
            clear_ulysses_params()
            conv_out = module.causal_conv1d_fn(torch.zeros(1, 4, 4))
            scan_out = module.torch_chunk_gated_delta_rule(torch.zeros(1, 2, 1, 2))
        finally:
            del sys.modules["gdn_modeling_mod_stock"]

        assert conv_out == "stock-conv"
        assert scan_out == "stock-scan"
        assert seen == ["conv", "scan"]

    def test_kimi_delta_attention_module_kernels_receive_cp_context(
        self, monkeypatch
    ) -> None:
        # Arrange
        module = types.ModuleType("kda_modeling_mod")

        def causal_conv1d_fn(*_args: object, **_kwargs: object) -> None:
            msg = "original conv"
            raise AssertionError(msg)

        def chunk_kimi_delta_attention(*_args: object, **_kwargs: object) -> None:
            msg = "original scan"
            raise AssertionError(msg)

        module.causal_conv1d_fn = causal_conv1d_fn
        module.chunk_kimi_delta_attention = chunk_kimi_delta_attention
        monkeypatch.setitem(sys.modules, "kda_modeling_mod", module)

        class Mixer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv_kernel_size = 4

        Mixer.__module__ = "kda_modeling_mod"
        seen: list[tuple[object, object]] = []

        def fla_kda(*args: object, **kwargs: object) -> tuple[object, None]:
            seen.append((kwargs["cp_context"], kwargs["chunk_size"]))
            return args[0], None

        def fla_gdn(*_args: object, **_kwargs: object) -> None:
            msg = "gated delta rule scan"
            raise AssertionError(msg)

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_chunk_kda", fla_kda
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_chunk_gated_delta_rule",
            fla_gdn,
        )
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_build_cp_context",
            lambda *_args, **_kwargs: "ctx",
        )
        _fake_fla_module(monkeypatch, "fla.ops.cp", build_cp_context=lambda *_a: "ctx")
        _fake_fla_module(monkeypatch, "fla.ops.kda", chunk_kda=fla_kda)
        # 160 tokens over two ranks leave 80-token pieces: KDA has no 16 chunk.
        cases = (([0, 128], 64), ([0, 192], 32), ([0, 160], 64))
        query = torch.zeros(1, 2, 1, 2)

        # Act
        assert install_gdn_context_parallel(Mixer(), _cp_mesh()) is True
        try:
            for bounds, _chunk in cases:
                publish_varlen_offsets(torch.tensor(bounds, dtype=torch.int32))
                output, _state = module.chunk_kimi_delta_attention(
                    query, query, query, g=query, beta=query
                )
                assert output is query
        finally:
            clear_ulysses_params()

        # Assert
        assert seen == [("ctx", chunk) for _bounds, chunk in cases]
        assert getattr(module.causal_conv1d_fn, "_agilerl_gdn_cp_kernel", False) is True


def _fake_fla_module(monkeypatch, name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)


class TestGdnCpContextCache:
    def test_builds_one_context_per_published_offsets(
        self, monkeypatch, gdn_kernels
    ) -> None:
        # Arrange
        built: list[list[int]] = []
        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.gdn._fla_build_cp_context",
            lambda cu_seqlens, *_args, **_kwargs: (
                built.append(cu_seqlens.tolist()) or "ctx"
            ),
        )
        install_gdn_context_parallel(DummyGatedDeltaNet(), _cp_mesh())
        query = torch.zeros(1, 4, 2, 2)

        # Act: two layers, one checkpoint recompute, then the next micro-batch.
        try:
            publish_varlen_offsets(torch.tensor([0, 8], dtype=torch.int32))
            forward_params = dict(ULYSSES_PARAMS)
            for _layer in range(2):
                gdn_kernels.torch_chunk_gated_delta_rule(query, query, query)
            clear_ulysses_params()
            with PublishedUlyssesParams(forward_params):
                gdn_kernels.torch_chunk_gated_delta_rule(query, query, query)
            publish_varlen_offsets(torch.tensor([0, 16], dtype=torch.int32))
            gdn_kernels.torch_chunk_gated_delta_rule(query, query, query)
        finally:
            clear_ulysses_params()

        # Assert
        assert built == [[0, 8], [0, 16]]


class TestMambaCpIndex:
    def test_selects_the_rank_heads_and_groups(self) -> None:
        # Arrange: x is 8 channels, B and C are 2 groups of 3.
        dims = MambaDims(num_heads=4, head_dim=2, n_groups=2, ssm_state_size=3)

        # Act
        index = mamba_cp_index(dims, 2, 1, "cpu")

        # Assert
        assert index.conv.tolist() == [4, 5, 6, 7, 11, 12, 13, 17, 18, 19]
        assert index.groups.tolist() == [0, 1]

    def test_ranks_sharing_a_group_both_hold_it(self) -> None:
        dims = MambaDims(num_heads=4, head_dim=2, n_groups=1, ssm_state_size=3)

        index = mamba_cp_index(dims, 2, 1, "cpu")

        assert index.conv.tolist() == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        assert index.groups.tolist() == [0, 0]


class TestFlaBuildCpContext:
    def test_passes_cu_seqlens_group_and_kernel_size(self, monkeypatch) -> None:
        def build_cp_context(
            cu_seqlens: torch.Tensor, group: object, conv1d_kernel_size: int
        ) -> tuple[list[int], object, int]:
            return cu_seqlens.tolist(), group, conv1d_kernel_size

        _fake_fla_module(monkeypatch, "fla.ops.cp", build_cp_context=build_cp_context)
        group = object()

        context = _fla_build_cp_context(torch.tensor([0, 8]), group, 4)

        assert context == ([0, 8], group, 4)


class TestFlaCausalConv1d:
    def test_passes_arguments_to_the_fla_convolution_export(self, monkeypatch) -> None:
        # Arrange
        def causal_conv1d(**kwargs: object) -> tuple[dict[str, object], None]:
            return kwargs, None

        _fake_fla_module(
            monkeypatch, "fla.modules.convolution", causal_conv1d=causal_conv1d
        )
        x = torch.zeros(1, 4, 6)
        weight = torch.ones(6, 4)

        # Act
        seen, state = _fla_causal_conv1d(x, weight, None, "silu", "ctx")

        # Assert
        assert seen == {
            "x": x,
            "weight": weight,
            "bias": None,
            "activation": "silu",
            "cp_context": "ctx",
        }
        assert state is None


class TestFlaChunkGatedDeltaRule:
    def test_passes_arguments_through(self, monkeypatch) -> None:
        def chunk_gated_delta_rule(
            *args: object, **kwargs: object
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            return args, kwargs

        _fake_fla_module(
            monkeypatch,
            "fla.ops.gated_delta_rule",
            chunk_gated_delta_rule=chunk_gated_delta_rule,
        )

        result = _fla_chunk_gated_delta_rule("q", "k", cp_context="ctx")

        assert result == (("q", "k"), {"cp_context": "ctx"})


class TestFlaChunkKda:
    def test_passes_arguments_through(self, monkeypatch) -> None:
        def chunk_kda(
            *args: object, **kwargs: object
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            return args, kwargs

        _fake_fla_module(monkeypatch, "fla.ops.kda", chunk_kda=chunk_kda)

        result = _fla_chunk_kda("q", "k", cp_context="ctx", chunk_size=32)

        assert result == (("q", "k"), {"cp_context": "ctx", "chunk_size": 32})


class _LoraInProj(nn.Module):
    """Base layer plus a LoRA pair."""

    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(4, 4, bias=False)
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(x))


class TestCallLocal:
    def test_runs_plain_parameters_as_the_module(self) -> None:
        torch.manual_seed(1)
        layer = _LoraInProj()
        x = torch.randn(2, 3, 4)

        out = call_local(layer, local_param_slots(layer), x)

        assert torch.equal(out, layer(x))

    @pytest.mark.usefixtures("gloo_process_group")
    def test_runs_dtensor_parameters_on_their_local_tensors(self) -> None:
        # Arrange
        torch.manual_seed(2)
        layer = nn.Linear(4, 3)
        weight = layer.weight.detach().clone()
        mesh = init_device_mesh("cpu", (1,))
        layer.weight = nn.Parameter(distribute_tensor(weight, mesh, [Replicate()]))
        x = torch.randn(2, 4)

        # Act
        out = call_local(layer, local_param_slots(layer), x)

        # Assert
        assert not isinstance(out, DTensor)
        assert torch.equal(out, nn.functional.linear(x, weight, layer.bias))

    @pytest.mark.usefixtures("gloo_process_group")
    def test_reads_parameters_swapped_in_after_the_slots_were_resolved(
        self,
    ) -> None:
        # Arrange
        torch.manual_seed(3)
        layer = _LoraInProj()
        slots = local_param_slots(layer)
        weight = torch.randn(4, 4)
        mesh = init_device_mesh("cpu", (1,))
        layer.base_layer.weight = nn.Parameter(
            distribute_tensor(weight, mesh, [Replicate()])
        )
        x = torch.randn(2, 3, 4)

        # Act
        out = call_local(layer, slots, x)

        # Assert
        lora = nn.functional.linear(
            nn.functional.linear(x, layer.lora_A.weight), layer.lora_B.weight
        )
        assert not isinstance(out, DTensor)
        assert torch.allclose(out, nn.functional.linear(x, weight) + lora)


def _granite_mamba(n_groups: int) -> nn.Module:
    config = GraniteMoeHybridConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        mamba_n_heads=8,
        mamba_d_head=4,
        mamba_n_groups=n_groups,
        mamba_d_state=4,
        mamba_d_conv=4,
        mamba_chunk_size=4,
    )
    return GraniteMoeHybridMambaLayer(config, 0)


def _nemotron_mamba(n_groups: int) -> nn.Module:
    config = NemotronHConfig(
        hidden_size=16,
        num_attention_heads=4,
        mamba_num_heads=8,
        mamba_head_dim=4,
        n_groups=n_groups,
        ssm_state_size=4,
        chunk_size=4,
        use_conv_bias=True,
        layers_block_type=["linear_attention"],
    )
    return NemotronHMamba2Mixer(config, 0)


MAMBA_BUILDERS = {
    "granite_1_group": lambda: _granite_mamba(1),
    "granite_2_groups": lambda: _granite_mamba(2),
    "nemotron_2_groups": lambda: _nemotron_mamba(2),
    "nemotron_4_groups": lambda: _nemotron_mamba(4),
}


def _mamba_cp_worker(
    rank: int, world_size: int, builder: str, norm_group_size: int | None
) -> None:
    torch.manual_seed(0)
    mixer = MAMBA_BUILDERS[builder]()
    with torch.no_grad():
        mixer.in_proj.weight.normal_(std=0.5)
        mixer.out_proj.weight.normal_(std=0.5)
        mixer.conv1d.weight.normal_(std=0.5)
        mixer.norm.weight.normal_(mean=1.0, std=0.2)
    reference = MAMBA_BUILDERS[builder]()
    reference.load_state_dict(mixer.state_dict())
    hidden = torch.randn(2, 8, 16)
    probe = torch.randn(2, 8, 16)
    local = slice(rank * 8 // world_size, (rank + 1) * 8 // world_size)
    params = dict(mixer.named_parameters())
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    install_mamba_context_parallel(mixer, mesh, norm_group_size)
    shard = hidden[:, local].clone().requires_grad_(True)
    full = hidden.clone().requires_grad_(True)
    publish_varlen_offsets(torch.arange(3, dtype=torch.int32) * 8)

    try:
        out = mixer(shard)
        (out * probe[:, local]).sum().backward()
        expected = reference(full)
        (expected * probe).sum().backward()
    finally:
        clear_ulysses_params()

    # The mixer's own parameters are registered again once forward returns.
    assert {name: id(param) for name, param in mixer.named_parameters()} == {
        name: id(param) for name, param in params.items()
    }
    assert_exact(out, expected[:, local].detach(), "output")
    assert_exact(shard.grad, full.grad[:, local], "input grad")
    for name, param in reference.named_parameters():
        grad = params[name].grad.clone()
        dist.all_reduce(grad)
        assert_exact(grad, param.grad, f"{name} grad")


def _mamba_tp_cp_worker(rank: int, world_size: int, builder: str) -> None:
    # Arrange
    torch.manual_seed(0)
    mixer = MAMBA_BUILDERS[builder]()
    with torch.no_grad():
        mixer.in_proj.weight.normal_(std=0.5)
        mixer.out_proj.weight.normal_(std=0.5)
        mixer.conv1d.weight.normal_(std=0.5)
        mixer.norm.weight.normal_(mean=1.0, std=0.2)
    reference = MAMBA_BUILDERS[builder]()
    reference.load_state_dict(mixer.state_dict())
    hidden = torch.randn(2, 8, 16)
    probe = torch.randn(2, 8, 16)
    mesh = build_parallel_mesh(world_size=world_size, tp=2, cp=2, device_type="cpu")
    assert mesh is not None
    cp_rank = mesh.cp.get_local_rank()
    local = slice(cp_rank * 4, (cp_rank + 1) * 4)
    norm_group_size = int(mixer.intermediate_size) // int(mixer.n_groups)
    if isinstance(mixer, NemotronHMamba2Mixer):
        assert apply_mamba_tensor_parallel(mixer, mesh.tp) == 1
        orders = mamba_row_orders(mixer)
    else:
        norm_group_size = int(mixer.intermediate_size)
        shard_mamba_mixer(mixer, mesh.tp, norm_group_size)
        orders = mamba_row_orders(mixer, lambda module: module is mixer)
    assert install_mamba_context_parallel(mixer, mesh.cp, norm_group_size)
    shard = hidden[:, local].clone().requires_grad_(True)
    full = hidden.clone().requires_grad_(True)
    publish_varlen_offsets(torch.arange(3, dtype=torch.int32) * 8)

    # Act
    try:
        out = mixer(shard)
        (out * probe[:, local]).sum().backward()
    finally:
        clear_ulysses_params()
    sum_tp_partial_grads(mixer, mesh.tp.get_group())
    expected = reference(full)
    (expected * probe).sum().backward()

    # Assert
    # TP and CP reduces sum partial products in a different order than one
    # matmul; shared B/C conv grads sum over all four ranks.
    tol = {"atol": 1e-5, "rtol": 1e-4}
    torch.testing.assert_close(out, expected[:, local].detach(), **tol)
    torch.testing.assert_close(shard.grad, full.grad[:, local], **tol)
    reference_params = dict(reference.named_parameters())
    for name, param in mixer.named_parameters():
        grad = param.grad
        if isinstance(grad, DTensor):
            summed = grad.to_local().clone()
            dist.all_reduce(summed, group=mesh.cp.get_group())
            grad = DTensor.from_local(
                summed, grad.device_mesh, grad.placements
            ).full_tensor()
            if name in orders:
                grad = grad.index_select(0, torch.argsort(orders[name]))
        else:
            grad = grad.clone()
            dist.all_reduce(grad, group=mesh.cp.get_group())
        torch.testing.assert_close(
            grad, reference_params[name].grad, **tol, msg=f"{name} grad"
        )


class TestInstallMambaContextParallelComposedWithTensorParallel:
    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        "builder",
        ["nemotron_2_groups", "granite_1_group"],
        ids=["groups_split_over_tp", "tp_ranks_share_a_group"],
    )
    def test_output_and_grads_match_unsharded_mixer(self, builder: str) -> None:
        run_ranks(_mamba_tp_cp_worker, 4, builder, timeout=300)


class TestMixerGatedNormThenSeqForward:
    def test_matches_the_gated_norm_in_the_input_dtype(self) -> None:
        # Arrange
        torch.manual_seed(0)
        norm = GraniteMoeHybridRMSNormGated(8).to(torch.bfloat16)
        with torch.no_grad():
            norm.weight.normal_(mean=1.0, std=0.2)
        hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        gate = torch.randn(2, 3, 8, dtype=torch.bfloat16)

        # Act
        out = MixerGatedNormThenSeq(norm, local_param_slots(norm), 1, None)(
            hidden, gate
        )

        # Assert
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, norm(hidden, gate))

    def test_sends_bf16_hidden_states_with_their_gate(self, monkeypatch) -> None:
        # Arrange
        torch.manual_seed(0)
        sent: list[torch.Tensor] = []

        def all_to_all(data: torch.Tensor, _cp: int, _group: object) -> torch.Tensor:
            sent.append(data)
            return data

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            all_to_all,
        )
        norm = GraniteMoeHybridRMSNormGated(8).to(torch.bfloat16)
        hidden = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        gate = torch.randn(2, 3, 8, dtype=torch.bfloat16)

        # Act
        MixerGatedNormThenSeq(norm, local_param_slots(norm), 2, None)(hidden, gate)

        # Assert
        assert len(sent) == 1
        assert torch.equal(sent[0], torch.cat([hidden, gate]))

    def test_sends_the_fp32_gated_product(self, monkeypatch) -> None:
        # Arrange
        torch.manual_seed(0)
        sent: list[torch.Tensor] = []

        def all_to_all(data: torch.Tensor, _cp: int, _group: object) -> torch.Tensor:
            sent.append(data)
            return data

        monkeypatch.setattr(
            "agilerl.distributed.context_parallel.mamba.all_to_all_hidden_to_seq",
            all_to_all,
        )
        norm = GraniteMoeHybridRMSNormGated(8).to(torch.bfloat16)
        hidden = torch.randn(2, 3, 8)
        gate = torch.randn(2, 3, 8, dtype=torch.bfloat16)

        # Act
        out = MixerGatedNormThenSeq(norm, local_param_slots(norm), 2, None)(
            hidden, gate
        )

        # Assert
        assert len(sent) == 1
        assert torch.equal(sent[0], hidden * nn.functional.silu(gate.float()))
        assert torch.equal(out, norm(hidden, gate))


def _gated_norm_cp_worker(rank: int, world_size: int, builder: str) -> None:
    # Arrange
    torch.manual_seed(0)
    norm = MAMBA_BUILDERS[builder]().norm.to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.2)
    width = norm.weight.shape[0]
    hidden = torch.randn(2, 8, width, dtype=torch.bfloat16)
    gate = torch.randn(2, 8, width, dtype=torch.bfloat16)
    probe = torch.randn(2, 8, width, dtype=torch.bfloat16)
    heads = slice(rank * width // world_size, (rank + 1) * width // world_size)
    local = slice(rank * 8 // world_size, (rank + 1) * 8 // world_size)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))
    stand_in = MixerGatedNormThenSeq(
        norm, local_param_slots(norm), world_size, mesh.get_group()
    )
    head_hidden = hidden[..., heads].clone().requires_grad_(True)
    head_gate = gate[..., heads].clone().requires_grad_(True)
    full_hidden = hidden.clone().requires_grad_(True)
    full_gate = gate.clone().requires_grad_(True)

    # Act
    out = stand_in(head_hidden, head_gate)
    (out * probe[:, local]).sum().backward()
    expected = norm(full_hidden, full_gate)
    (expected * probe).sum().backward()

    # Assert
    # The all-to-all only moves rows, so each row runs the same ops as the norm.
    assert torch.equal(out, expected[:, local])
    assert torch.equal(head_hidden.grad, full_hidden.grad[..., heads])
    assert torch.equal(head_gate.grad, full_gate.grad[..., heads])


class TestMixerGatedNormThenSeqMatchesUnsharded:
    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("builder", "world_size"),
        [("granite_1_group", 2), ("nemotron_2_groups", 4)],
        ids=["full_width_norm", "groups_span_ranks"],
    )
    def test_bf16_output_and_grads_match_the_gated_norm(
        self, builder: str, world_size: int
    ) -> None:
        run_ranks(_gated_norm_cp_worker, world_size, builder, timeout=300)


class TestInstallMambaContextParallelMatchesUnsharded:
    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        ("builder", "world_size", "norm_group_size"),
        [
            ("granite_1_group", 2, None),
            ("granite_2_groups", 2, 32),
            ("nemotron_2_groups", 4, None),
            ("nemotron_4_groups", 2, None),
        ],
        ids=[
            "one_group",
            "full_width_norm",
            "groups_span_ranks",
            "groups_per_rank",
        ],
    )
    def test_output_and_grads_match_unsharded_mixer(
        self, builder: str, world_size: int, norm_group_size: int | None
    ) -> None:
        run_ranks(_mamba_cp_worker, world_size, builder, norm_group_size, timeout=300)


def _host_sync_worker(rank: int, world_size: int) -> None:
    """A GDN and a Mamba CP layer run forward and backward with host syncs as errors."""
    # Arrange
    device = torch.device("cuda", rank)
    torch.manual_seed(0)
    # FLA's Triton kernels tl.dot over the head dims, which needs them >= 16.
    gdn = Qwen3_5GatedDeltaNet(
        Qwen3_5TextConfig(
            hidden_size=16,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
        ),
        0,
    ).to(device)
    mamba = _nemotron_mamba(2).to(device)
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("cp",))
    assert install_gdn_context_parallel(gdn, mesh)
    assert install_mamba_context_parallel(mamba, mesh)
    seq = 2 * CP_SCAN_CHUNK * world_size
    hidden = torch.randn(1, seq // world_size, 16, device=device, requires_grad=True)
    update_ulysses_params(
        torch.tensor([0, seq], dtype=torch.int32, device=device), (0, seq)
    )

    def step() -> torch.Tensor:
        loss = (gdn(hidden).float().sum() + mamba(hidden).float().sum()) / seq
        loss.backward()
        return loss.detach()

    # Act: the warm-up compiles kernels and builds the FLA context.
    try:
        step()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            loss = step()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    finally:
        clear_ulysses_params()

    # Assert
    assert torch.isfinite(loss)
    assert torch.isfinite(hidden.grad).all()


class TestContextParallelMixersHostSync:
    @pytest.mark.gpu
    @pytest.mark.multi_rank
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
    def test_gdn_and_mamba_layers_run_without_host_syncs(self) -> None:
        run_ranks(_host_sync_worker, 2, backend="nccl", timeout=600)


class TestCatalogIsolation:
    @pytest.mark.parametrize("family", ["granitemoehybrid", "nemotron_h", "qwen3_5"])
    def test_tensor_parallel_then_context_parallel_import_in_a_fresh_interpreter(
        self, family: str
    ) -> None:
        # Arrange
        code = (
            f"import agilerl.architectures.{family}.tensor_parallel\n"
            f"import agilerl.architectures.{family}.context_parallel\n"
        )

        # Act
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )

        # Assert
        assert result.returncode == 0, result.stderr

    def test_family_context_parallel_plans_return_every_family(self) -> None:
        from agilerl.architectures import family_context_parallel_plans
        from agilerl.architectures.deepseek_v4.context_parallel import (
            CONTEXT_PARALLEL_PLAN as DEEPSEEK_V4_CONTEXT_PARALLEL_PLAN,
        )
        from agilerl.architectures.glm5_next.context_parallel import (
            CONTEXT_PARALLEL_PLAN as GLM5_NEXT_CONTEXT_PARALLEL_PLAN,
        )
        from agilerl.architectures.granitemoehybrid.context_parallel import (
            CONTEXT_PARALLEL_PLAN as GRANITEMOEHYBRID_CONTEXT_PARALLEL_PLAN,
        )
        from agilerl.architectures.nemotron_h.context_parallel import (
            CONTEXT_PARALLEL_PLAN as NEMOTRON_H_CONTEXT_PARALLEL_PLAN,
        )
        from agilerl.architectures.qwen3_5.context_parallel import (
            CONTEXT_PARALLEL_PLAN as QWEN3_5_CONTEXT_PARALLEL_PLAN,
        )

        assert family_context_parallel_plans() == [
            DEEPSEEK_V4_CONTEXT_PARALLEL_PLAN,
            GLM5_NEXT_CONTEXT_PARALLEL_PLAN,
            GRANITEMOEHYBRID_CONTEXT_PARALLEL_PLAN,
            NEMOTRON_H_CONTEXT_PARALLEL_PLAN,
            QWEN3_5_CONTEXT_PARALLEL_PLAN,
        ]
