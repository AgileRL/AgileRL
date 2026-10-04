# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel NemotronH Mamba2 on the tensor-parallel mesh."""

from __future__ import annotations

import copy
import os
import socket
import sys
from typing import Any, ClassVar

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import Shard
from transformers import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from agilerl.architectures.nemotron_h.mamba import _full_parameter_tensor
from agilerl.distributed import FSDPConfig
from agilerl.distributed.expert_parallel import build_parallel_mesh
from agilerl.distributed.fsdp import apply_fsdp2, materialize_dtensors
from agilerl.distributed.mamba_parallel import (
    apply_mamba_tensor_parallel,
    permuted_local_from_full,
    realign_mamba_permuted_shards,
)
from agilerl.distributed.tensor_parallel import SharedSeedDropout
from agilerl.utils.llm_utils import get_lora_named_params

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


def _gloo_available() -> bool:
    if sys.platform == "win32":
        return False
    return dist.is_available()


requires_gloo = pytest.mark.skipif(not _gloo_available(), reason="gloo unavailable")


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_ranks(worker, world_size: int = 2, timeout: float = 300.0) -> None:
    port = _free_port()
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs = [
        ctx.Process(target=worker, args=(rank, world_size, port, queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    results = [queue.get(timeout=timeout) for _ in range(world_size)]
    for proc in procs:
        proc.join(timeout=timeout)
        assert proc.exitcode == 0, f"rank exited {proc.exitcode}"
    for rank, status, err in sorted(results):
        assert status == "ok", f"rank {rank}: {err}"


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


def _broadcast_module(module: nn.Module) -> None:
    for param in module.parameters():
        dist.broadcast(param.data, src=0)


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
        "conv1d.weight": "conv",
        "conv1d.bias": "conv",
    }
    kind = permuted.get(name)
    if kind is not None:
        return permuted_local_from_full(full, mixer, kind, param.device_mesh)
    local = distribute_tensor(
        full, param.device_mesh, param.placements, src_data_rank=None
    )
    return local.to_local()


def _parity_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        # Arrange
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        mixer = _lora_mixer()
        _broadcast_module(mixer)
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

        # Assert
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
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _lora_export_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        # Arrange
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        mixer = _mixer(num_heads=2 * world_size, n_groups=world_size)
        inject_adapter_in_model(LoraConfig(r=2, target_modules=["in_proj"]), mixer)
        with torch.no_grad():
            for name, param in mixer.named_parameters():
                if "lora_" in name:
                    param.normal_(std=0.5)
        _broadcast_module(mixer)
        reference = copy.deepcopy(mixer)
        source = torch.randn(2, 4, mixer.hidden_size)
        probe = torch.randn(2, 4, mixer.hidden_size)
        dist.broadcast(source, src=0)
        dist.broadcast(probe, src=0)
        mesh = build_parallel_mesh(
            world_size=world_size, tp=world_size, device_type="cpu"
        )
        assert mesh is not None
        apply_mamba_tensor_parallel(mixer, mesh.tp)

        # Act
        for module in (reference, mixer):
            (module(source) * probe).sum().backward()
            with torch.no_grad():
                for param in module.parameters():
                    if param.grad is not None:
                        param.sub_(0.1 * param.grad)
        named = get_lora_named_params(mixer)
        with materialize_dtensors(*[param for _, param in named]) as dense:
            exported = dict(zip([name for name, _ in named], dense, strict=True))

        # Assert
        reference_params = dict(reference.named_parameters())
        assert "in_proj.lora_B.default.weight" in exported
        for name, tensor in exported.items():
            _assert_close(tensor, reference_params[name].detach(), name)
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _kernel_tensor_worker(
    rank: int, world_size: int, port: int, result_queue: Any
) -> None:
    try:
        _init_gloo(rank, world_size, port)
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
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _dropout_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
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
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _realign_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        tp = 2
        mixer = _mixer(num_heads=4, n_groups=2)
        _broadcast_module(mixer)
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
            in_bias.to_local().copy_(
                in_bias_full[rank * in_rows : (rank + 1) * in_rows]
            )
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
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _groups_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mixer = _mixer(num_heads=4, n_groups=1)
        mesh = build_parallel_mesh(world_size=world_size, tp=2, device_type="cpu")
        assert mesh is not None
        with pytest.raises(ValueError, match=r"n_groups \(1\).*tp \(2\)"):
            apply_mamba_tensor_parallel(mixer, mesh.tp)
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


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


def _fsdp_worker(rank: int, world_size: int, port: int, result_queue: Any) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.manual_seed(0)
        tp = 2
        mixer = _mixer(num_heads=4, n_groups=2)
        model = MambaStack(mixer)
        _broadcast_module(model)
        mesh = build_parallel_mesh(world_size=world_size, tp=tp, device_type="cpu")
        assert mesh is not None
        assert apply_mamba_tensor_parallel(model, mesh.tp) == 1
        before = {
            id(param)
            for param in model.parameters()
            if isinstance(param, DTensor) and param.device_mesh is mesh.tp
        }
        assert before, "expected mamba DTensors on the tp mesh"
        apply_fsdp2(
            model,
            FSDPConfig(tp=tp, wrap_every_n_blocks=1, param_persistence_threshold=0),
            mesh=mesh.hsdp,
        )
        after = {
            id(param)
            for param in model.parameters()
            if isinstance(param, DTensor) and param.device_mesh is mesh.tp
        }
        assert after == before, "mamba DTensors must stay on the tp mesh"
        bias = model.layers[0].mixer.out_proj.bias
        assert not isinstance(bias, DTensor)
        assert bias.shape[0] == mixer.hidden_size
        lin = model.layers[0].lin.weight
        assert isinstance(lin, DTensor)
        assert lin.device_mesh is not mesh.tp
        result_queue.put((rank, "ok", None))
    except Exception as exc:
        result_queue.put((rank, "err", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_gloo
class TestApplyMambaTensorParallel:
    def test_forward_and_grads_match_unsharded_mixer(self):
        _spawn_ranks(_parity_worker)

    @pytest.mark.parametrize("world_size", [2, 4])
    def test_lora_export_after_step_is_in_hf_row_order(self, world_size):
        _spawn_ranks(_lora_export_worker, world_size=world_size)

    def test_kernel_reads_local_rows_of_marked_shards(self):
        _spawn_ranks(_kernel_tensor_worker)

    def test_lora_dropout_mask_matches_across_ranks(self):
        _spawn_ranks(_dropout_worker)

    def test_realign_restores_permuted_local_rows(self):
        _spawn_ranks(_realign_worker)

    def test_n_groups_not_divisible_by_ep_raises(self):
        _spawn_ranks(_groups_worker)

    def test_fsdp_leaves_mamba_dtensors_on_tp_mesh(self):
        _spawn_ranks(_fsdp_worker)
