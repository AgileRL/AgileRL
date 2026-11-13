# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Stochastic rounding of fp32 master weights to bf16."""

from __future__ import annotations

import os
import sys
from collections.abc import Generator

import pytest
import torch
import torch.distributed as dist
import torch.distributed.fsdp._fully_shard._fsdp_collectives as fsdp_collectives
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
    activate_bf16_rounding,
    begin_master_step,
    bf16_round_step,
    cast_compute_weight,
    finish_master_step,
    stochastic_round_bf16,
)


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

    def test_frozen_master_stays_on_round_to_nearest(self) -> None:
        weight = nn.Parameter(_halfway_span(32).view(8, 4))
        weight.requires_grad = False
        activate_bf16_rounding(nn.Linear(1, 1), 2)

        cast = cast_compute_weight(weight, torch.bfloat16)

        assert torch.equal(cast, weight.detach().to(torch.bfloat16))
        assert not torch.equal(cast, stochastic_round_bf16(weight.detach(), 2))


class TestBeginMasterStep:
    def test_sub_ulp_step_changes_bf16_only_when_rounding_is_on(self) -> None:
        # Arrange
        shadow, ulp = _sub_ulp_shadows(2000, 0.2)

        class Masters(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(shadow.clone())

        module = Masters()
        nearest_before = module.weight.detach().to(torch.bfloat16)
        activate_bf16_rounding(module, 0)

        # Act
        snaps = begin_master_step(module)
        with torch.no_grad():
            module.weight.add_(0.05 * ulp)
        changed = finish_master_step(snaps)

        # Assert
        assert torch.equal(nearest_before, module.weight.detach().to(torch.bfloat16))
        assert changed is not None
        assert changed > 0.05
        assert bf16_round_step.get() == 1

    def test_off_skips_the_snapshot(self) -> None:
        module = nn.Linear(2, 2, bias=False)

        assert begin_master_step(module) == []
        assert finish_master_step([]) is None


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
