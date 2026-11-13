# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel NemotronH Mamba2 on the tensor-parallel mesh."""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from transformers import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from agilerl.architectures.nemotron_h import tensor_parallel as mamba_tp
from agilerl.architectures.nemotron_h.layers import _full_parameter_tensor
from agilerl.architectures.nemotron_h.tensor_parallel import (
    apply_mamba_tensor_parallel,
    mark_mamba_tp_params,
    realign_mamba_permuted_shards,
)
from agilerl.distributed import FSDPConfig, build_parallel_mesh, fsdp_blocks
from agilerl.distributed.fsdp import materialize_dtensors
from agilerl.distributed.fsdp_blocks import apply_fsdp2
from agilerl.distributed.tensor_parallel import (
    SharedSeedDropout,
    clear_gathered_rows,
    sum_tp_partial_grads,
    tensor_rows_in_order,
)
from agilerl.utils.llm_utils import get_lora_named_params
from tests.parallel_helpers import broadcast_module
from tests.rank_launch import run_ranks

_DIST_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _clean_dist_state():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    saved = {var: os.environ.pop(var, None) for var in _DIST_ENV}
    yield
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    for var, value in saved.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


def _mixer(
    *,
    num_heads: int,
    n_groups: int,
    head_dim: int = 4,
    ssm_state_size: int = 4,
    hidden_size: int = 8,
) -> NemotronHMamba2Mixer:
    config = NemotronHConfig(
        hidden_size=hidden_size,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        mamba_num_heads=num_heads,
        mamba_head_dim=head_dim,
        n_groups=n_groups,
        ssm_state_size=ssm_state_size,
        use_mamba_kernels=False,
        use_bias=True,
        use_conv_bias=True,
        chunk_size=8,
        layers_block_type=["linear_attention"],
    )
    return NemotronHMamba2Mixer(config, layer_idx=0)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
        diff = (actual - expected).abs().max().item()
        msg = f"{label}: max abs diff {diff}"
        raise AssertionError(msg)


def _lora_mixer() -> NemotronHMamba2Mixer:
    mixer = _mixer(num_heads=4, n_groups=2)
    inject_adapter_in_model(LoraConfig(r=2, target_modules=["in_proj"]), mixer)
    mixer.requires_grad_(True)
    with torch.no_grad():
        for name, param in mixer.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.5)
    return mixer


def _local_expected_grad(
    name: str, param: DTensor, full: torch.Tensor, mixer: NemotronHMamba2Mixer
) -> torch.Tensor:
    permuted = {
        "in_proj.base_layer.weight": "in_proj",
        "in_proj.base_layer.bias": "in_proj",
        "in_proj.lora_B.default.weight": "in_proj",
        "conv1d.weight": "conv",
        "conv1d.bias": "conv",
    }
    kind = permuted.get(name)
    if kind is not None:
        order = mamba_tp._row_order(
            *mamba_tp._mixer_dims(mixer), int(param.device_mesh.size()), kind
        )
        full = full.index_select(0, order)
    local = distribute_tensor(
        full, param.device_mesh, param.placements, src_data_rank=None
    )
    return local.to_local()


def _parity_worker(rank: int, world_size: int) -> None:
    # Arrange
    torch.manual_seed(0)
    mixer = _lora_mixer()
    broadcast_module(mixer)
    reference = copy.deepcopy(mixer)
    source = torch.randn(2, 4, mixer.hidden_size)
    probe = torch.randn(2, 4, mixer.hidden_size)
    dist.broadcast(source, src=0)
    dist.broadcast(probe, src=0)
    mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
    assert mesh is not None
    assert apply_mamba_tensor_parallel(mixer, mesh.tp) == 1
    ref_input = source.clone().requires_grad_(True)
    tp_input = source.clone().requires_grad_(True)

    # Act
    expected = reference(ref_input)
    actual = mixer(tp_input)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    sum_tp_partial_grads(mixer, mesh.tp.get_group())

    # Assert
    lora_b = mixer.in_proj.lora_B["default"].weight
    assert isinstance(lora_b, DTensor)
    assert lora_b.device_mesh == mesh.tp
    assert lora_b.placements == (Shard(0),)
    _assert_close(actual, expected, "output")
    _assert_close(tp_input.grad, ref_input.grad, "input grad")
    reference_params = dict(reference.named_parameters())
    for name, param in mixer.named_parameters():
        full = reference_params[name].grad
        if isinstance(param, DTensor):
            local = _local_expected_grad(name, param, full, mixer)
            _assert_close(param.grad.to_local(), local, name)
        else:
            _assert_close(param.grad, full, name)


def _lora_export_worker(rank: int, world_size: int) -> None:
    # Arrange
    torch.manual_seed(0)
    mixer = _mixer(num_heads=2 * world_size, n_groups=world_size)
    inject_adapter_in_model(LoraConfig(r=2, target_modules=["in_proj"]), mixer)
    with torch.no_grad():
        for name, param in mixer.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.5)
    broadcast_module(mixer)
    reference = copy.deepcopy(mixer)
    source = torch.randn(2, 4, mixer.hidden_size)
    probe = torch.randn(2, 4, mixer.hidden_size)
    dist.broadcast(source, src=0)
    dist.broadcast(probe, src=0)
    mesh = build_parallel_mesh(world_size=world_size, tp=world_size, device_type="cpu")
    assert mesh is not None
    apply_mamba_tensor_parallel(mixer, mesh.tp)

    # Act
    (reference(source) * probe).sum().backward()
    (mixer(source) * probe).sum().backward()
    sum_tp_partial_grads(mixer, mesh.tp.get_group())
    for module in (reference, mixer):
        with torch.no_grad():
            for param in module.parameters():
                if param.grad is not None:
                    param.sub_(0.1 * param.grad)
    named = get_lora_named_params(mixer)
    orders = mamba_tp.mamba_row_orders(mixer)
    with materialize_dtensors(*[param for _, param in named]) as dense:
        exported = {
            name: tensor_rows_in_order(tensor, torch.argsort(orders[name]))
            if name in orders
            else tensor
            for (name, _), tensor in zip(named, dense, strict=True)
        }

    # Assert
    reference_params = dict(reference.named_parameters())
    assert "in_proj.lora_B.default.weight" in orders
    for name, tensor in exported.items():
        _assert_close(tensor, reference_params[name].detach(), name)


def _kernel_tensor_worker(rank: int, world_size: int) -> None:
    mixer = _mixer(num_heads=4, n_groups=2)
    mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
    assert mesh is not None
    apply_mamba_tensor_parallel(mixer, mesh.tp)
    full = torch.arange(8.0).view(4, 2)
    unmarked = distribute_tensor(full, mesh.tp, [Shard(0)])

    shard = _full_parameter_tensor(mixer.conv1d.weight)
    gathered = _full_parameter_tensor(unmarked)

    assert torch.equal(shard, mixer.conv1d.weight.to_local())
    assert torch.equal(gathered, full)


def _dropout_worker(rank: int, world_size: int) -> None:
    torch.manual_seed(rank)
    mixer = _mixer(num_heads=4, n_groups=2)
    config = LoraConfig(r=2, target_modules=["in_proj"], lora_dropout=0.5)
    inject_adapter_in_model(config, mixer)
    mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
    assert mesh is not None
    apply_mamba_tensor_parallel(mixer, mesh.tp)
    dropout = mixer.in_proj.lora_dropout["default"]
    dropout.train()

    mask = dropout(torch.ones(256))
    masks = [torch.empty_like(mask) for _ in range(world_size)]
    dist.all_gather(masks, mask)

    assert isinstance(dropout, SharedSeedDropout)
    assert torch.equal(masks[0], masks[1])
    assert bool((mask == 0).any())
    assert bool((mask == 2).any())


def _realign_worker(rank: int, world_size: int) -> None:
    torch.manual_seed(0)
    tp = 2
    mixer = _mixer(num_heads=4, n_groups=2)
    broadcast_module(mixer)
    reference = copy.deepcopy(mixer)
    in_full = mixer.in_proj.weight.detach().clone()
    in_bias_full = mixer.in_proj.bias.detach().clone()
    conv_full = mixer.conv1d.weight.detach().clone()
    conv_bias_full = mixer.conv1d.bias.detach().clone()
    hidden = mixer.hidden_size
    source = torch.randn(2, 4, hidden)
    dist.broadcast(source, src=0)
    mesh = build_parallel_mesh(world_size=world_size, tp=tp, device_type="cpu")
    assert mesh is not None
    assert apply_mamba_tensor_parallel(mixer, mesh.tp) == 1
    in_weight = mixer.in_proj.weight
    in_bias = mixer.in_proj.bias
    conv_weight = mixer.conv1d.weight
    conv_bias = mixer.conv1d.bias
    assert isinstance(in_weight, DTensor)
    assert isinstance(in_bias, DTensor)
    assert isinstance(conv_weight, DTensor)
    assert isinstance(conv_bias, DTensor)
    in_snapshot = in_weight.to_local().detach().clone()
    conv_snapshot = conv_weight.to_local().detach().clone()
    in_rows = in_snapshot.shape[0]
    conv_rows = conv_snapshot.shape[0]
    with torch.no_grad():
        in_weight.to_local().copy_(in_full[rank * in_rows : (rank + 1) * in_rows])
        in_bias.to_local().copy_(in_bias_full[rank * in_rows : (rank + 1) * in_rows])
        conv_weight.to_local().copy_(
            conv_full[rank * conv_rows : (rank + 1) * conv_rows]
        )
        conv_bias.to_local().copy_(
            conv_bias_full[rank * conv_rows : (rank + 1) * conv_rows]
        )
    if torch.equal(in_weight.to_local(), in_snapshot):
        msg = "in_proj shard overwrite matched the permuted rows"
        raise AssertionError(msg)
    realign_mamba_permuted_shards(mixer)
    if not torch.equal(in_weight.to_local(), in_snapshot):
        msg = "in_proj local rows differ from the permuted snapshot"
        raise AssertionError(msg)
    if not torch.equal(conv_weight.to_local(), conv_snapshot):
        msg = "conv1d local rows differ from the permuted snapshot"
        raise AssertionError(msg)
    with torch.no_grad():
        expected = reference(source)
        actual = mixer(source)
    if not torch.allclose(actual, expected, atol=1e-4, rtol=1e-4):
        diff = (actual - expected).abs().max().item()
        msg = f"max abs diff {diff}"
        raise AssertionError(msg)


def _groups_worker(rank: int, world_size: int) -> None:
    mixer = _mixer(num_heads=4, n_groups=1)
    mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
    assert mesh is not None
    with pytest.raises(ValueError, match=r"n_groups \(1\).*tp \(2\)"):
        apply_mamba_tensor_parallel(mixer, mesh.tp)


def _shared_group_mixers(
    world_size: int,
) -> tuple[NemotronHMamba2Mixer, NemotronHMamba2Mixer, torch.Tensor]:
    """A one-group LoRA mixer sharded over TP, its unsharded copy, and an input."""
    torch.manual_seed(0)
    mixer = _mixer(num_heads=4, n_groups=1)
    inject_adapter_in_model(LoraConfig(r=2, target_modules=["in_proj"]), mixer)
    with torch.no_grad():
        for name, param in mixer.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.5)
    broadcast_module(mixer)
    reference = copy.deepcopy(mixer)
    source = torch.randn(2, 4, mixer.hidden_size)
    dist.broadcast(source, src=0)
    mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
    assert mesh is not None
    mamba_tp.shard_mamba_mixer(mixer, mesh.tp, int(mixer.intermediate_size))
    return mixer, reference, source


def _frozen_rows_cache_worker(rank: int, world_size: int) -> None:
    # Arrange
    mixer, reference, source = _shared_group_mixers(world_size)
    expected = reference(source)

    # Act
    first = mixer(source)
    cached = dict(mixer.tp_frozen_rows)
    second = mixer(source)

    # Assert
    _assert_close(first, expected, "first output")
    assert torch.equal(second, first)
    assert set(cached) == {
        "in_proj.base_layer.weight",
        "in_proj.base_layer.bias",
        "conv1d.weight",
        "conv1d.bias",
    }
    for name, rows in cached.items():
        assert mixer.tp_frozen_rows[name] is rows, name
    clear_gathered_rows(mixer)
    assert "tp_frozen_rows" not in mixer.__dict__


def _trainable_rows_worker(rank: int, world_size: int) -> None:
    # Arrange
    mixer, reference, source = _shared_group_mixers(world_size)
    mixer.requires_grad_(True)
    reference.requires_grad_(True)

    # Act
    actual = mixer(source)
    expected = reference(source)

    # Assert
    _assert_close(actual, expected, "output")
    assert "tp_frozen_rows" not in mixer.__dict__


@pytest.mark.multi_rank
class TestShardMambaMixerSharedGroups:
    def test_frozen_rows_gather_once(self):
        run_ranks(_frozen_rows_cache_worker)

    def test_trainable_rows_gather_every_forward(self):
        run_ranks(_trainable_rows_worker)


class MambaBlock(nn.Module):
    def __init__(self, mixer: NemotronHMamba2Mixer) -> None:
        super().__init__()
        self.mixer = mixer
        self.lin = nn.Linear(mixer.hidden_size, mixer.hidden_size)


class MambaStack(nn.Module):
    _no_split_modules: ClassVar = ["MambaBlock"]

    def __init__(self, mixer: NemotronHMamba2Mixer) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(mixer)])


def _fsdp_worker(rank: int, world_size: int) -> None:
    torch.manual_seed(0)
    tp = 2
    mixer = _mixer(num_heads=4, n_groups=2)
    model = MambaStack(mixer)
    broadcast_module(model)
    mesh = build_parallel_mesh(world_size=world_size, tp=tp, device_type="cpu")
    assert mesh is not None
    assert apply_mamba_tensor_parallel(model, mesh.tp) == 1
    full = {name: param.numel() for name, param in model.named_parameters()}
    tp_sharded = {
        name
        for name, param in model.named_parameters()
        if isinstance(param, DTensor) and param.device_mesh is mesh.tp
    }
    assert tp_sharded, "expected mamba DTensors on the tp mesh"

    with patch.object(fsdp_blocks, "PARAM_PERSISTENCE_THRESHOLD", 0):
        apply_fsdp2(
            model,
            FSDPConfig(tp=tp),
            mesh=mesh.dp_shard,
        )

    for name in sorted(tp_sharded):
        param = model.get_parameter(name)
        assert isinstance(param, DTensor), name
        assert param.device_mesh.mesh_dim_names == ("dp_shard", "tp"), name
        held = torch.tensor(param.to_local().numel())
        dist.all_reduce(held)
        copies = tp if param.placements[-1].is_replicate() else 1
        assert int(held) == full[name] * copies, name
    bias = model.layers[0].mixer.out_proj.bias
    assert isinstance(bias, DTensor)
    assert bias.device_mesh.mesh_dim_names == ("dp_shard", "tp")
    assert bias.placements[-1].is_replicate()
    assert bias.shape[0] == mixer.hidden_size
    lin = model.layers[0].lin.weight
    assert isinstance(lin, DTensor)
    assert lin.device_mesh.mesh_dim_names == ("dp_shard",)


@pytest.mark.multi_rank
class TestApplyMambaTensorParallel:
    def test_forward_and_grads_match_unsharded_mixer(self):
        run_ranks(_parity_worker)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_lora_export_after_step_is_in_hf_row_order(self, world_size):
        run_ranks(_lora_export_worker, world_size)

    def test_kernel_reads_local_rows_of_marked_shards(self):
        run_ranks(_kernel_tensor_worker)

    def test_lora_dropout_mask_matches_across_ranks(self):
        run_ranks(_dropout_worker)

    def test_realign_restores_permuted_local_rows(self):
        run_ranks(_realign_worker)

    def test_n_groups_not_divisible_by_ep_raises(self):
        run_ranks(_groups_worker)

    def test_fsdp_shards_mamba_tp_weights_over_dp_shard(self):
        run_ranks(_fsdp_worker, 4)


class TestMambaTensorParallelGuards:
    def test_vision_head_count_raises_without_heads(self):
        with pytest.raises(ValueError, match="has no num_heads"):
            mamba_tp._vision_head_count(nn.Linear(2, 2))

    def test_shard_vision_attention_rejects_query_rows_that_do_not_divide(self):
        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(4, 6)

        attn = Attn()
        attn.__dict__["num_heads"] = 4
        with pytest.raises(ValueError, match="must be divisible by num_heads"):
            mamba_tp._shard_vision_attention(attn, MagicMock())

    def test_shard_latent_returns_false_when_projections_are_not_linear(self):
        class Latent(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1_latent_proj = nn.Identity()
                self.fc2_latent_proj = nn.Identity()

        assert mamba_tp._shard_latent(Latent(), MagicMock()) is False

    def test_install_mamba_tp_forward_is_idempotent(self):
        mixer = _mixer(num_heads=4, n_groups=2)
        mamba_tp._install_mamba_tp_forward(mixer)
        original = mixer.forward

        mamba_tp._install_mamba_tp_forward(mixer)

        assert mixer.forward is original

    def test_mamba_tp_forward_calls_a_replaced_inner(self):
        # Arrange
        mixer = _mixer(num_heads=4, n_groups=2)
        mixer.out_proj.bias = None
        mamba_tp._install_mamba_tp_forward(mixer)
        object.__setattr__(mixer, "_tp_degree", 2)
        object.__setattr__(mixer, "_tp_group", MagicMock(spec=dist.ProcessGroup))
        seen: list[torch.Size] = []
        reduced = torch.zeros(1, 2, mixer.config.hidden_size)

        def replacement(hidden: torch.Tensor, *_args: object, **_kwargs: object):
            seen.append(hidden.shape)
            return reduced

        object.__setattr__(mixer, "_mamba_inner_forward", replacement)

        # Act
        with (
            patch(
                "agilerl.architectures.nemotron_h.tensor_parallel.copy_input_to_region",
                side_effect=lambda args, kwargs, _group: (args, kwargs),
            ),
            patch(
                "agilerl.architectures.nemotron_h.tensor_parallel._call_local_forward",
                side_effect=lambda _mixer, inner, args, kwargs: inner(*args, **kwargs),
            ),
            patch.object(
                mamba_tp.ReduceFromTPRegion,
                "apply",
                side_effect=lambda tensor, _group: tensor,
            ),
        ):
            out = mixer(torch.ones(1, 2, mixer.config.hidden_size))

        # Assert
        assert seen == [torch.Size((1, 2, mixer.config.hidden_size))]
        assert out is reduced

    def test_mamba_tp_forward_returns_the_reduce_when_out_proj_has_no_bias(self):
        mixer = _mixer(num_heads=4, n_groups=2)
        mixer.out_proj.bias = None
        reduced = torch.ones(1, 2, mixer.config.hidden_size)
        mamba_tp._install_mamba_tp_forward(mixer)
        object.__setattr__(mixer, "_tp_degree", 2)
        object.__setattr__(mixer, "_tp_group", MagicMock(spec=dist.ProcessGroup))
        with (
            patch(
                "agilerl.architectures.nemotron_h.tensor_parallel.copy_input_to_region",
                side_effect=lambda args, kwargs, _group: (args, kwargs),
            ),
            patch(
                "agilerl.architectures.nemotron_h.tensor_parallel._call_local_forward",
                return_value=reduced,
            ),
            patch(
                "agilerl.architectures.nemotron_h.tensor_parallel._local_mixer_sizes",
            ) as sizes,
            patch.object(
                mamba_tp.ReduceFromTPRegion,
                "apply",
                side_effect=lambda tensor, _group: tensor,
            ),
        ):
            sizes.return_value.__enter__.return_value = None
            sizes.return_value.__exit__.return_value = False
            out = mixer(torch.ones(1, 2, mixer.config.hidden_size))

        assert out is reduced

    def test_shard_mixer_installs_forward_when_in_proj_is_already_a_dtensor(self):
        mixer = _mixer(num_heads=4, n_groups=2)
        mesh = MagicMock()
        mesh.size.return_value = 2
        mesh.get_group.return_value = object()
        with patch(
            "agilerl.architectures.nemotron_h.tensor_parallel.DTensor", nn.Parameter
        ):
            mamba_tp.shard_mamba_mixer(
                mixer, mesh, mixer.intermediate_size // mixer.n_groups
            )

        assert getattr(mixer, "_agilerl_mamba_tp", False)

    def test_shard_mixer_rejects_groups_and_tp_that_do_not_divide(self):
        mixer = _mixer(num_heads=6, n_groups=2)
        mesh = MagicMock()
        mesh.size.return_value = 3

        with pytest.raises(
            ValueError, match=r"n_groups \(2\) and tp \(3\) must divide one another"
        ):
            mamba_tp.shard_mamba_mixer(
                mixer, mesh, mixer.intermediate_size // mixer.n_groups
            )

    def test_iter_permuted_mamba_params_lists_in_proj_its_lora_b_and_conv(self):
        # Arrange
        mixer = _lora_mixer()
        object.__setattr__(mixer, "_tp_degree", 2)
        parent = nn.Module()
        parent.mixer = mixer
        parent.stray = nn.Linear(2, 2)

        # Act
        found = mamba_tp.iter_permuted_mamba_params(parent)

        # Assert
        assert all(owner is mixer for _, owner, _ in found)
        assert {fqn: kind for fqn, _, kind in found} == {
            "mixer.in_proj.base_layer.weight": "in_proj",
            "mixer.in_proj.base_layer.bias": "in_proj",
            "mixer.in_proj.lora_B.default.weight": "in_proj",
            "mixer.conv1d.weight": "conv",
            "mixer.conv1d.bias": "conv",
        }

    def test_iter_permuted_mamba_params_skips_unsharded_mixers(self):
        mixer = _lora_mixer()

        assert mamba_tp.iter_permuted_mamba_params(mixer) == []

    def test_mark_mamba_tp_params_skips_mixers_without_a_tp_mesh(self):
        mixer = _mixer(num_heads=4, n_groups=2)
        marked = []
        with patch(
            "agilerl.architectures.nemotron_h.tensor_parallel.mark_mamba_tp_shard",
            side_effect=lambda param: marked.append(param),
        ):
            mark_mamba_tp_params(mixer)

        assert marked == []

    def test_mark_mamba_tp_params_marks_dtensors_on_the_tp_mesh(self):
        mixer = _mixer(num_heads=4, n_groups=2)
        object.__setattr__(mixer, "_tp_mesh", object())
        object.__setattr__(mixer, "_tp_degree", 2)
        param = MagicMock()
        param.device_mesh = SimpleNamespace(mesh_dim_names=("dp_shard", "tp"))
        marked = []
        with (
            patch(
                "agilerl.architectures.nemotron_h.tensor_parallel.DTensor",
                MagicMock,
            ),
            patch.object(type(mixer), "parameters", return_value=iter([param])),
            patch(
                "agilerl.architectures.nemotron_h.tensor_parallel.mark_mamba_tp_shard",
                side_effect=lambda item: marked.append(item),
            ),
        ):
            mark_mamba_tp_params(mixer)

        assert marked == [param]

    def test_apply_mamba_tensor_parallel_is_a_no_op_without_a_mesh(self):
        assert apply_mamba_tensor_parallel(nn.Linear(2, 2), None) == 0
