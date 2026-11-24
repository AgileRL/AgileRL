# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Stochastic rounding of fp32 master weights to bf16."""

from __future__ import annotations

import os
import sys
import traceback
from collections.abc import Generator
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.fsdp._fully_shard._fsdp_collectives as fsdp_collectives
import torch.multiprocessing as mp
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor

from agilerl.distributed import init_distributed
from agilerl.lora.fused import _apply_lora_factor
from agilerl.utils.stochastic_round import (
    HAS_TRITON,
    activate_bf16_rounding,
    bf16_round_step,
    cast_compute_weight,
    copy_synced_master_,
    stochastic_round_bf16,
    stop_reusing_roundings,
)
from tests.dist_ports import get_free_port
from tests.utils import spawn_new_process_for_each_test


def _bf16_neighbors(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the bf16 grid point at ``value`` and the next grid point toward +inf."""
    grid = torch.tensor(value).to(torch.bfloat16)
    upper = torch.nextafter(grid, torch.tensor(float("inf"), dtype=torch.bfloat16))
    return grid, upper


def _halfway_span(count: int) -> torch.Tensor:
    """Halfway points between distinct bf16 grid values, so each has its own draw."""
    raw = torch.linspace(-0.5, 0.5, count)
    grid = raw.to(torch.bfloat16)
    upper = torch.nextafter(grid, torch.full_like(grid, float("inf")))
    return (grid.float() + upper.float()) / 2


def _sharded_linear(
    values: torch.Tensor, *, cpu_offload: bool, requires_grad: bool = True
) -> nn.Linear:
    """Shard one linear layer on the CPU mesh, optionally with parameter offload."""
    model = nn.Linear(values.shape[1], values.shape[0], bias=False)
    with torch.no_grad():
        model.weight.copy_(values)
    model.weight.requires_grad_(requires_grad)
    offload_policy: OffloadPolicy = (
        CPUOffloadPolicy(pin_memory=False) if cpu_offload else OffloadPolicy()
    )
    fully_shard(
        model,
        mesh=torch.distributed.init_device_mesh("cpu", (1,)),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
        offload_policy=offload_policy,
    )
    return model


def _copy_same_device_transfers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a same-device non-blocking ``Tensor.to`` a new storage."""
    original = torch.Tensor.to

    def _to(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        moved = original(tensor, *args, **kwargs)
        if (
            kwargs.get("non_blocking")
            and moved is tensor
            and tensor.dtype is torch.float32
        ):
            return tensor.clone()
        return moved

    monkeypatch.setattr(torch.Tensor, "to", _to)


def _unsharded_weight(module: nn.Module) -> torch.Tensor:
    weight = module.weight
    if isinstance(weight, DTensor):
        weight = weight.to_local()
    return weight.detach().clone()


def _sub_ulp_shadows(count: int, fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Shadows ``fraction`` of a ulp above their bf16 grid point, and that ulp."""
    raw = torch.linspace(-0.5, 0.5, count)
    grid = raw.to(torch.bfloat16)
    upper = torch.nextafter(grid, torch.full_like(grid, float("inf"))).float()
    ulp = upper - grid.float()
    return grid.float() + fraction * ulp, ulp


@pytest.fixture(autouse=True)
def _reset_bf16_round_step() -> Generator[None, None, None]:
    token = bf16_round_step.set(None)
    yield
    bf16_round_step.reset(token)
    stop_reusing_roundings()


@pytest.fixture
def world_size_one(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Single-process gloo group so FSDP2 can shard on CPU."""
    if sys.platform == "win32":
        pytest.skip("torch gloo process groups are unsupported on Windows wheels")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(29850 + os.getpid() % 100))
    monkeypatch.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
    assert init_distributed() is True
    yield
    dist.destroy_process_group()


class TestStochasticRoundBf16:
    def test_halfway_value_rounds_up_half_the_time(self) -> None:
        # Arrange
        grid, upper = _bf16_neighbors(0.025)
        mid = (grid.float() + upper.float()) / 2
        draws = 4000

        # Act
        rounded = torch.stack(
            [stochastic_round_bf16(mid, step).float() for step in range(draws)]
        )

        # Assert
        went_up = rounded == upper.float()
        assert went_up.any()
        assert (~went_up).any()
        assert abs(float(went_up.float().mean()) - 0.5) < 0.05
        assert abs(float(rounded.mean()) - float(mid)) < 5e-6

    def test_negative_halfway_value_matches_the_shadow(self) -> None:
        # Arrange
        grid, _upper = _bf16_neighbors(-0.025)
        lower = torch.nextafter(grid, torch.tensor(float("-inf"), dtype=torch.bfloat16))
        mid = (grid.float() + lower.float()) / 2
        draws = 4000

        # Act
        rounded = torch.stack(
            [stochastic_round_bf16(mid, step).float() for step in range(draws)]
        )

        # Assert
        assert abs(float(rounded.mean()) - float(mid)) < 5e-6

    def test_exact_bf16_value_stays_put(self) -> None:
        grid, _upper = _bf16_neighbors(0.025)

        rounded = torch.stack(
            [stochastic_round_bf16(grid.float(), step) for step in range(32)]
        )

        assert torch.equal(rounded, grid.expand_as(rounded))

    def test_sub_ulp_move_stays_under_round_to_nearest_and_moves_under_sr(self) -> None:
        # Arrange
        shadow, ulp = _sub_ulp_shadows(2000, 0.2)
        moved = shadow + 0.05 * ulp

        # Act
        nearest_before = shadow.to(torch.bfloat16)
        nearest_after = moved.to(torch.bfloat16)
        before = stochastic_round_bf16(shadow, 0)
        after = stochastic_round_bf16(moved, 1)

        # Assert
        assert torch.equal(nearest_before, nearest_after)
        changed = float((before != after).float().mean())
        assert changed > 0.05

    def test_same_step_returns_the_same_bf16_tensor(self) -> None:
        grid, upper = _bf16_neighbors(0.025)
        values = (grid.float() + upper.float()) / 2
        values = values.expand(16).contiguous()

        first = stochastic_round_bf16(values, 7)
        second = stochastic_round_bf16(values.clone(), 7)

        assert torch.equal(first, second)
        assert not torch.equal(first, stochastic_round_bf16(values, 8))

    def test_nonfinite_values_use_the_ordinary_cast(self) -> None:
        values = torch.tensor([float("nan"), float("inf"), float("-inf")])

        rounded = stochastic_round_bf16(values, 1)

        assert torch.isnan(rounded[0])
        assert rounded[1].item() == float("inf")
        assert rounded[2].item() == float("-inf")

    def test_out_receives_the_rounding(self) -> None:
        values = _halfway_span(32).view(4, 8)
        destination = torch.zeros(4, 8, dtype=torch.bfloat16)

        returned = stochastic_round_bf16(values, 5, out=destination)

        assert returned is destination
        assert torch.equal(destination, stochastic_round_bf16(values, 5))

    def test_non_contiguous_out_receives_the_rounding(self) -> None:
        values = _halfway_span(32).view(8, 4)
        destination = torch.zeros(4, 8, dtype=torch.bfloat16).t()

        stochastic_round_bf16(values, 5, out=destination)

        assert torch.equal(destination, stochastic_round_bf16(values, 5))


def _rounding_inputs() -> torch.Tensor:
    """Halfway points, random magnitudes, negatives, and non-finite values."""
    torch.manual_seed(0)
    return torch.cat(
        [
            _halfway_span(2048),
            torch.randn(4096) * 1e-3,
            -torch.rand(4096) * 1e4,
            torch.tensor([float("nan"), float("inf"), float("-inf"), 0.0, -0.0]),
            torch.tensor([torch.finfo(torch.float32).max]),
        ]
    )


def assert_same_rounding(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert identical bf16 bits, allowing any NaN payload."""
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual[~nan].view(torch.int16), expected[~nan].view(torch.int16))


@pytest.fixture
def triton_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the kernel on CPU tensors through the Triton interpreter.

    Triton reads ``TRITON_INTERPRET`` when it first decorates its own library
    kernels, so the test must run in a fresh process that starts with it set.
    """
    monkeypatch.setenv("TRITON_INTERPRET", "1")


class TestStochasticRoundKernel:
    @spawn_new_process_for_each_test
    @pytest.mark.usefixtures("triton_interpreter")
    @pytest.mark.parametrize("step", [0, 1, 7, 2**20 + 3])
    def test_interpreted_kernel_matches_the_torch_rounding(self, step: int) -> None:
        # Arrange
        kernel = pytest.importorskip("agilerl.utils.stochastic_round_kernel")
        values = _rounding_inputs()
        out = torch.empty(values.shape, dtype=torch.bfloat16)
        salt = (step * 0x9E3779B1) & 0xFFFFFFFF
        salt -= (salt >> 31) << 32

        # Act
        kernel.stochastic_round_bf16_into(values, out, salt)

        # Assert
        assert_same_rounding(out, stochastic_round_bf16(values, step))

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not (torch.cuda.is_available() and HAS_TRITON),
        reason="Triton kernel needs a CUDA device",
    )
    @pytest.mark.parametrize("step", [0, 1, 7, 2**20 + 3])
    def test_cuda_rounding_matches_the_cpu_rounding(self, step: int) -> None:
        values = _rounding_inputs()

        rounded = stochastic_round_bf16(values.cuda(), step)

        assert_same_rounding(rounded.cpu(), stochastic_round_bf16(values, step))


class TestCastComputeWeight:
    def test_flag_off_matches_round_to_nearest(self) -> None:
        grid, upper = _bf16_neighbors(0.025)
        weight = nn.Parameter((grid.float() + upper.float()) / 2)

        cast = cast_compute_weight(weight, torch.bfloat16)

        assert torch.equal(cast, weight.detach().to(torch.bfloat16))
        assert bf16_round_step.get() is None

    def test_forward_reads_the_rounded_master_and_keeps_a_gradient(self) -> None:
        # Arrange
        mid = _halfway_span(32).view(4, 8)

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(8, 4, bias=False)

            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                return self.proj(hidden)

        model = get_peft_model(
            Block(),
            LoraConfig(r=4, lora_alpha=8, target_modules=["proj"], lora_dropout=0.0),
            autocast_adapter_dtype=True,
        )
        layer = model.base_model.model.proj
        factor = layer.lora_A["default"]
        assert factor.weight.dtype is torch.float32
        with torch.no_grad():
            factor.weight.copy_(mid)
        activate_bf16_rounding(model, 2)
        inputs = torch.randn(3, 8, dtype=torch.bfloat16)
        expected = stochastic_round_bf16(factor.weight.detach(), 2)

        # Act
        output = _apply_lora_factor(layer, factor, inputs)
        output.sum().backward()

        # Assert
        assert torch.equal(output, F.linear(inputs, expected))
        assert not torch.equal(expected, factor.weight.detach().to(torch.bfloat16))
        assert factor.weight.grad is not None
        assert torch.isfinite(factor.weight.grad).all()
        assert factor.weight.grad.abs().sum() > 0

    def test_reuse_gives_the_per_call_rounding_and_gradient(self) -> None:
        # Arrange
        master = nn.Parameter(_halfway_span(32).view(4, 8))
        reference = nn.Parameter(master.detach().clone())
        inputs = [torch.randn(3, 8, dtype=torch.bfloat16) for _ in range(3)]
        activate_bf16_rounding(nn.Linear(1, 1), 2)
        for hidden in inputs:
            F.linear(
                hidden, cast_compute_weight(reference, torch.bfloat16)
            ).sum().backward()
        activate_bf16_rounding(nn.ParameterList([master]), 2, reuse_roundings=True)

        # Act
        weights = [cast_compute_weight(master, torch.bfloat16) for _ in inputs]
        for hidden, weight in zip(inputs, weights, strict=True):
            F.linear(hidden, weight).sum().backward()

        # Assert
        assert all(weight is weights[0] for weight in weights)
        assert torch.equal(weights[0], stochastic_round_bf16(master.detach(), 2))
        assert torch.equal(master.grad, reference.grad)

    def test_reuse_after_a_no_grad_read_keeps_the_gradient(self) -> None:
        # Arrange
        master = nn.Parameter(_halfway_span(32).view(4, 8))
        activate_bf16_rounding(nn.ParameterList([master]), 2, reuse_roundings=True)
        with torch.no_grad():
            frozen = cast_compute_weight(master, torch.bfloat16)

        # Act
        weight = cast_compute_weight(master, torch.bfloat16)
        weight.float().sum().backward()

        # Assert
        assert frozen.grad_fn is None
        assert torch.equal(master.grad, torch.ones_like(master))

    def test_reuse_draws_again_on_the_next_step(self) -> None:
        master = nn.Parameter(_halfway_span(32).view(4, 8))
        activate_bf16_rounding(nn.ParameterList([master]), 2, reuse_roundings=True)
        cast_compute_weight(master, torch.bfloat16)
        bf16_round_step.set(3)

        weight = cast_compute_weight(master, torch.bfloat16)

        assert torch.equal(weight, stochastic_round_bf16(master.detach(), 3))

    def test_batched_next_step_rounds_every_master_and_keeps_gradients(
        self,
    ) -> None:
        # Arrange
        masters = [
            nn.Parameter(_halfway_span(32).view(4, 8)),
            nn.Parameter(_halfway_span(48).view(8, 6) * 3),
        ]
        activate_bf16_rounding(nn.ParameterList(masters), 2, reuse_roundings=True)
        for master in masters:
            cast_compute_weight(master, torch.bfloat16)
        bf16_round_step.set(3)

        # Act
        weights = [cast_compute_weight(master, torch.bfloat16) for master in masters]
        for _ in range(2):
            sum(weight.float().sum() for weight in weights).backward()

        # Assert
        for master, weight in zip(masters, weights, strict=True):
            assert torch.equal(weight, stochastic_round_bf16(master.detach(), 3))
            assert torch.equal(master.grad, torch.full_like(master, 2.0))

    def test_reuse_skips_an_unregistered_fp32_copy(self) -> None:
        # Arrange
        master = nn.Parameter(_halfway_span(32).view(4, 8))
        unsharded = nn.Parameter(master.detach().clone())
        activate_bf16_rounding(nn.ParameterList([master]), 2, reuse_roundings=True)

        # Act
        first = cast_compute_weight(unsharded, torch.bfloat16)
        second = cast_compute_weight(unsharded, torch.bfloat16)

        # Assert
        assert first is not second
        assert torch.equal(first, stochastic_round_bf16(master.detach(), 2))

    def test_frozen_master_stays_on_round_to_nearest(self) -> None:
        weight = nn.Parameter(_halfway_span(32).view(8, 4))
        weight.requires_grad = False
        activate_bf16_rounding(nn.Linear(1, 1), 2)

        cast = cast_compute_weight(weight, torch.bfloat16)

        assert torch.equal(cast, weight.detach().to(torch.bfloat16))
        assert not torch.equal(cast, stochastic_round_bf16(weight.detach(), 2))


class TestCopySyncedMaster:
    def test_active_step_writes_the_rounding_into_a_bf16_destination(self) -> None:
        # Arrange
        torch.manual_seed(0)
        master = torch.randn(4, 8)
        destination = torch.empty(4, 8, dtype=torch.bfloat16)
        bf16_round_step.set(5)

        # Act
        copy_synced_master_(destination, master)

        # Assert
        assert torch.equal(destination, stochastic_round_bf16(master, 5))

    def test_without_a_step_is_a_plain_copy(self) -> None:
        # Arrange
        torch.manual_seed(0)
        master = torch.randn(4, 8)
        destination = torch.empty(4, 8, dtype=torch.bfloat16)
        bf16_round_step.set(None)

        # Act
        copy_synced_master_(destination, master)

        # Assert
        assert torch.equal(destination, master.to(torch.bfloat16))


class TestFSDPForwardReadsRoundedMaster:
    def test_unsharded_weight_is_the_stochastic_round(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            model.weight.copy_(mid)
        fully_shard(
            model,
            mesh=torch.distributed.init_device_mesh("cpu", (1,)),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        master = model.weight.detach().to_local().float()
        activate_bf16_rounding(model, 3)
        expected = stochastic_round_bf16(master, 3)
        seen: dict[str, torch.Tensor] = {}

        def _capture(_module: nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
            weight = _module.weight
            if isinstance(weight, DTensor):
                weight = weight.to_local()
            seen["weight"] = weight.detach().clone()

        handle = model.register_forward_pre_hook(_capture)

        # Act
        model(torch.randn(2, 4))
        handle.remove()

        # Assert
        assert seen["weight"].dtype is torch.bfloat16
        assert torch.equal(seen["weight"], expected)
        assert not torch.equal(seen["weight"], master.to(torch.bfloat16))

    def test_cpu_offload_unsharded_weight_is_the_stochastic_round(
        self, world_size_one: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=True)
        activate_bf16_rounding(model, 3)
        master = model.weight.detach().to_local().float()
        expected = stochastic_round_bf16(master, 3)
        seen: dict[str, torch.Tensor] = {}
        _copy_same_device_transfers(monkeypatch)

        def _capture(_module: nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
            seen["weight"] = _unsharded_weight(_module)

        handle = model.register_forward_pre_hook(_capture)

        # Act
        model(torch.randn(2, 4))
        handle.remove()

        # Assert
        assert seen["weight"].dtype is torch.bfloat16
        assert torch.equal(seen["weight"], expected)
        assert not torch.equal(seen["weight"], master.to(torch.bfloat16))

    def test_cpu_offload_frozen_weight_stays_on_round_to_nearest(
        self, world_size_one: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=True, requires_grad=False)
        activate_bf16_rounding(model, 3)
        master = model.weight.detach().to_local().float()
        seen: dict[str, torch.Tensor] = {}
        _copy_same_device_transfers(monkeypatch)

        def _capture(_module: nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
            seen["weight"] = _unsharded_weight(_module)

        handle = model.register_forward_pre_hook(_capture)

        # Act
        model(torch.randn(2, 4))
        handle.remove()

        # Assert
        assert torch.equal(seen["weight"], master.to(torch.bfloat16))
        assert not torch.equal(seen["weight"], stochastic_round_bf16(master, 3))

    def test_multi_rank_all_gather_input_is_the_stochastic_round(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=False)
        activate_bf16_rounding(model, 3)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        master = model.weight.detach().to_local().float()

        # Act
        inputs = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        gathered = inputs[0][0]
        assert gathered.dtype is torch.bfloat16
        assert torch.equal(gathered, stochastic_round_bf16(master, 3).reshape(-1))
        assert not torch.equal(gathered, master.to(torch.bfloat16).reshape(-1))

    def test_multi_rank_all_gather_input_is_nearest_when_rounding_is_off(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=False)
        activate_bf16_rounding(model, 3)
        bf16_round_step.set(None)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        master = model.weight.detach().to_local().float()

        # Act
        inputs = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        gathered = inputs[0][0]
        assert torch.equal(gathered, master.to(torch.bfloat16).reshape(-1))

    def test_multi_rank_all_gather_rounds_masters_and_casts_frozen_params(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = nn.Linear(4, 4)
        with torch.no_grad():
            model.weight.copy_(mid)
            model.bias.copy_(mid[0])
        model.bias.requires_grad_(False)
        fully_shard(
            model,
            mesh=torch.distributed.init_device_mesh("cpu", (1,)),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        activate_bf16_rounding(model, 3)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        weight = model.weight.detach().to_local().float()
        bias = model.bias.detach().to_local().float()

        # Act
        inputs = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        assert torch.equal(inputs[0][0], stochastic_round_bf16(weight, 3).reshape(-1))
        assert torch.equal(inputs[1][0], bias.to(torch.bfloat16))

    def test_multi_rank_all_gather_follows_a_master_update_without_reuse(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=False)
        activate_bf16_rounding(model, 3)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        before = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)
        before = before[0][0].clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        master = model.weight.detach().to_local().float()

        # Act
        after = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        assert torch.equal(after[0][0], stochastic_round_bf16(master, 3).reshape(-1))
        assert not torch.equal(after[0][0], before)

    def test_multi_rank_all_gather_rounds_a_group_into_one_buffer(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = nn.Linear(4, 4)
        with torch.no_grad():
            model.weight.copy_(mid)
            model.bias.copy_(mid[0])
        fully_shard(
            model,
            mesh=torch.distributed.init_device_mesh("cpu", (1,)),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        activate_bf16_rounding(model, 3)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        weight = model.weight.detach().to_local().float()
        bias = model.bias.detach().to_local().float()

        # Act
        inputs = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        assert torch.equal(inputs[0][0], stochastic_round_bf16(weight, 3).reshape(-1))
        assert torch.equal(inputs[1][0], stochastic_round_bf16(bias, 3))
        assert (
            inputs[0][0].untyped_storage().data_ptr()
            == inputs[1][0].untyped_storage().data_ptr()
        )

    def test_reuse_keeps_no_shard_rounding_between_all_gathers(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=False)
        activate_bf16_rounding(model, 3, reuse_roundings=True)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        master = model.weight.detach().to_local().float()

        # Act
        first = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)
        second = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        assert second[0][0] is not first[0][0]
        assert torch.equal(second[0][0], first[0][0])
        assert torch.equal(first[0][0], stochastic_round_bf16(master, 3).reshape(-1))

    def test_stop_reusing_follows_a_master_update(self, world_size_one: None) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=False)
        activate_bf16_rounding(model, 3, reuse_roundings=True)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)
        stop_reusing_roundings()
        with torch.no_grad():
            model.weight.add_(1.0)
        master = model.weight.detach().to_local().float()

        # Act
        inputs = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        assert torch.equal(inputs[0][0], stochastic_round_bf16(master, 3).reshape(-1))

    def test_multi_rank_all_gather_redraws_on_the_next_step(
        self, world_size_one: None
    ) -> None:
        # Arrange
        mid = _halfway_span(16).view(4, 4)
        model = _sharded_linear(mid, cpu_offload=False)
        activate_bf16_rounding(model, 3)
        group = model._get_fsdp_state()._fsdp_param_group
        group.lazy_init()
        master = model.weight.detach().to_local().float()
        fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)
        bf16_round_step.set(4)

        # Act
        inputs = fsdp_collectives._get_param_all_gather_inputs(group.fsdp_params)

        # Assert
        assert torch.equal(inputs[0][0], stochastic_round_bf16(master, 4).reshape(-1))
        assert not torch.equal(
            inputs[0][0], stochastic_round_bf16(master, 3).reshape(-1)
        )


def _two_rank_worker(rank: int, port: int, result_queue: Any) -> None:
    try:
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": "2",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
            }
        )
        dist.init_process_group("gloo", rank=rank, world_size=2)
        master = _halfway_span(32).view(4, 8)
        model = nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            model.weight.copy_(master)
        fully_shard(
            model,
            mesh=torch.distributed.init_device_mesh("cpu", (2,)),
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        activate_bf16_rounding(model, 3)
        seen: dict[str, torch.Tensor] = {}

        def _capture(_module: nn.Module, _args: tuple[torch.Tensor, ...]) -> None:
            seen["weight"] = _unsharded_weight(_module)

        model.register_forward_pre_hook(_capture)
        model(torch.randn(2, 8))

        assert torch.equal(seen["weight"], stochastic_round_bf16(master, 3))
        assert not torch.equal(seen["weight"], master.to(torch.bfloat16))
        result_queue.put((rank, "ok", None))
    except Exception:
        result_queue.put((rank, "err", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestFSDPTwoRankAllGather:
    def test_each_rank_gathers_the_rounding_of_the_full_master(self) -> None:
        # Arrange
        if sys.platform == "win32":
            pytest.skip("torch gloo process groups are unsupported on Windows wheels")
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        port = get_free_port()
        procs = [
            ctx.Process(target=_two_rank_worker, args=(rank, port, queue))
            for rank in range(2)
        ]

        # Act
        for proc in procs:
            proc.start()
        try:
            results = sorted(queue.get(timeout=300) for _ in procs)
            for proc in procs:
                proc.join(timeout=60)
        finally:
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)

        # Assert
        for rank, status, err in results:
            assert status == "ok", f"rank {rank}: {err}"
