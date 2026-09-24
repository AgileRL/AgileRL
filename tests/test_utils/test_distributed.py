# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for ``agilerl.distributed``.

Classes are named ``Test<FunctionName>`` and assert API-visible outcomes
(return values, side effects, raised errors). ``fully_shard`` is stubbed
only as an expensive boundary; assertions still target which modules were
sharded and which config policies were requested, not call counts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule

from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.distributed import (
    CPUOffloadOptimizer,
    FSDPConfig,
    aggregate_metrics_across_gpus,
    aggregate_metrics_dict,
    all_ranks,
    allreduce_minmax_int,
    any_rank,
    apply_fsdp2,
    barrier,
    broadcast_object_list,
    distributed_env_present,
    full_shape_views,
    gather_objects,
    gather_tensor,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    materialize_dtensors,
    resolve_device,
    set_seed,
    sync_grads,
)
from agilerl.distributed import fsdp as dmod
from agilerl.distributed import process as pmod
from agilerl.distributed import runtime as rmod
from agilerl.distributed.fsdp import (
    reshard_fsdp_modules,
    set_full_model_state_dict,
)
from agilerl.distributed.runtime import (
    DPRuntime,
    FSDPRuntime,
    OptimizerStep,
    clip_param_group_grad_norm_,
)
from agilerl.utils.algo_utils import CosineLRScheduleConfig

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for state round-trip"
)


class TestLlmUtilsImportCycle:
    def test_fresh_process_can_import_llm_utils_before_runtime(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agilerl.utils.llm_utils import is_rollout_prompt, make_llm_optimizer",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0, proc.stderr


class TestClipParamGroupGradNorm:
    def test_plain_tensors_clip_to_max_norm(self):
        first = nn.Parameter(torch.ones(4))
        first.grad = torch.ones(4) * 10
        second = nn.Parameter(torch.ones(4))
        second.grad = torch.ones(4) * 10
        total = clip_param_group_grad_norm_([first, second], max_norm=1.0)
        assert float(total) > 1.0
        stacked = torch.cat([first.grad.flatten(), second.grad.flatten()])
        assert torch.linalg.vector_norm(stacked).item() == pytest.approx(1.0, rel=1e-4)

    def test_no_grads_returns_zero(self):
        param = nn.Parameter(torch.ones(2))
        assert clip_param_group_grad_norm_([param], max_norm=1.0).item() == 0.0

    def test_mixed_dtensor_and_tensor_clips_separately(self, monkeypatch):
        class FakeDTensor(nn.Parameter):
            pass

        monkeypatch.setattr("agilerl.distributed.runtime.DTensor", FakeDTensor)
        seen: list[tuple[frozenset[str], float]] = []

        def fake_clip(params, max_norm):
            kinds = frozenset(type(param).__name__ for param in params)
            seen.append((kinds, float(max_norm)))
            return torch.tensor(3.0)

        monkeypatch.setattr("agilerl.distributed.runtime.clip_grad_norm_", fake_clip)
        sharded = FakeDTensor(torch.ones(2))
        sharded.grad = torch.ones(2)
        replicated = nn.Parameter(torch.ones(2))
        replicated.grad = torch.ones(2)
        total = clip_param_group_grad_norm_([sharded, replicated], max_norm=1.0)
        assert len(seen) == 2
        assert all(len(kinds) == 1 for kinds, _ in seen)
        assert float(total) == pytest.approx(18.0**0.5)
        scale = 1.0 / (18.0**0.5 + 1e-6)
        assert sharded.grad[0].item() == pytest.approx(scale, rel=1e-4)
        assert replicated.grad[0].item() == pytest.approx(scale, rel=1e-4)

    def test_sharded_only_delegates_to_torch(self, monkeypatch):
        class FakeDTensor(nn.Parameter):
            pass

        monkeypatch.setattr("agilerl.distributed.runtime.DTensor", FakeDTensor)
        seen: list = []

        def fake_clip(params, max_norm):
            seen.append((list(params), float(max_norm)))
            return torch.tensor(2.0)

        monkeypatch.setattr("agilerl.distributed.runtime.clip_grad_norm_", fake_clip)
        sharded = FakeDTensor(torch.ones(2))
        sharded.grad = torch.ones(2)

        total = clip_param_group_grad_norm_([sharded], max_norm=1.0)

        assert float(total) == 2.0
        assert seen[0][0] == [sharded]
        assert seen[0][1] == 1.0

    def test_scalar_norm_gathers_dtensor(self, monkeypatch):
        monkeypatch.setattr("agilerl.distributed.runtime.DTensor", _FakeDTensor)
        norm = _FakeDTensor(torch.tensor(3.0))

        out = rmod._scalar_grad_norm(norm)

        assert norm.full_tensor_calls == 1
        assert out.shape == torch.Size([])
        assert float(out) == 3.0


class _FakeDTensor:
    """Lightweight DTensor stand-in: carries a ``_local_tensor`` and records
    ``full_tensor()`` calls so tests can assert DTensor-aware state movement
    without a real distributed tensor.
    """

    def __init__(self, local_tensor: torch.Tensor):
        self._local_tensor = local_tensor
        self.full_tensor_calls = 0

    def full_tensor(self) -> torch.Tensor:
        self.full_tensor_calls += 1
        return self._local_tensor


_DIST_ENV = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def _clean_dist_state():
    """Reset process-group + launcher env so tests cannot leak into each other.

    Autouse is intentional: a leaked group or ``WORLD_SIZE`` breaks later
    tests that expect a single-device no-op path.
    """
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


@pytest.fixture
def world_size_one(_clean_dist_state):
    """Initialise a real single-process gloo group (distributed helpers active)."""
    if sys.platform == "win32":
        # Windows torch wheels ship gloo without the TCP transport used for
        # collectives ("makeDeviceForHostname(): unsupported gloo device").
        pytest.skip("torch gloo process groups are unsupported on Windows wheels")

    # Arrange
    os.environ.update(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(29650 + os.getpid() % 200),
        }
    )

    # Act
    started = init_distributed()

    # Assert (fixture invariant)
    assert started is True
    yield
    dist.destroy_process_group()


class TestDistributedPackageImport:
    def test_materialize_dtensors_importable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from agilerl.distributed import materialize_dtensors",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr


class TestInitDistributed:
    def test_returns_false_without_launcher_env(self):
        # Act / Assert
        assert distributed_env_present() is False
        assert init_distributed() is False
        assert is_distributed() is False

    def test_initialises_and_is_idempotent(self, world_size_one):
        # Assert
        assert is_distributed() is True
        assert init_distributed() is True


class TestGetRank:
    def test_zero_without_process_group(self):
        assert get_rank() == 0

    def test_zero_with_world_size_one(self, world_size_one):
        assert get_rank() == 0


class TestGetLocalRank:
    def test_reads_local_rank_env(self):
        # Arrange
        os.environ["LOCAL_RANK"] = "3"

        # Act / Assert
        assert get_local_rank() == 3

    def test_falls_back_to_zero_without_env_or_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert get_local_rank() == 0


class TestGetWorldSize:
    def test_one_without_process_group(self):
        assert get_world_size() == 1

    def test_one_with_world_size_one(self, world_size_one):
        assert get_world_size() == 1


class TestIsMainProcess:
    def test_true_without_process_group(self):
        assert is_main_process() is True

    def test_true_on_rank_zero(self, world_size_one):
        assert is_main_process() is True


class TestBarrier:
    def test_noop_without_process_group(self):
        barrier()

    def test_completes_with_process_group(self, world_size_one):
        barrier()

    def test_calls_dist_barrier_when_world_above_one(self):
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.dist.barrier") as mock_barrier,
        ):
            barrier()

        mock_barrier.assert_called_once_with()


class TestBroadcastObjectList:
    def test_passthrough_without_process_group(self):
        # Arrange
        objects = [1, "two", {"three": 3}]

        # Act
        out = broadcast_object_list(objects)

        # Assert
        assert out is objects

    def test_round_trip_with_process_group(self, world_size_one):
        # Arrange
        objects = ["x", 7]

        # Act
        out = broadcast_object_list(objects)

        # Assert
        assert out == ["x", 7]

    def test_dispatches_when_world_above_one(self):
        # Arrange
        objects = ["x", 7]

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch(
                "agilerl.distributed.process.dist.broadcast_object_list",
            ) as mock_broadcast,
        ):
            out = broadcast_object_list(objects, src=1)

        # Assert
        assert out is objects
        mock_broadcast.assert_called_once_with(objects, src=1)


class TestGatherTensor:
    def test_with_tensor_input(self):
        input_tensor = torch.tensor([1, 2, 3])

        gathered = gather_tensor(input_tensor)

        assert isinstance(gathered, torch.Tensor)
        assert torch.equal(gathered, input_tensor)

    def test_with_non_tensor_input(self):
        input_list = [1, 2, 3]

        gathered = gather_tensor(input_list)

        assert isinstance(gathered, torch.Tensor)
        assert torch.equal(gathered, torch.tensor(input_list))

    def test_concatenates_ranks_when_world_above_one(self):
        # Arrange
        def fake_all_gather(gathered: list, tensor: torch.Tensor) -> None:
            for slot in gathered:
                slot.copy_(tensor)

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.get_world_size", return_value=2),
            patch(
                "agilerl.distributed.process.dist.all_gather",
                side_effect=fake_all_gather,
            ),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
        ):
            out = gather_tensor(torch.tensor([1, 2]))

        # Assert
        assert torch.equal(out, torch.tensor([1, 2, 1, 2]))


class TestGatherObjects:
    def test_passthrough_without_process_group(self):
        # Arrange
        objects = [1, "two", {"three": 3}]

        # Act
        out = gather_objects(objects)

        # Assert
        assert out is objects

    def test_identity_with_world_size_one(self, world_size_one):
        # Arrange
        objects = [{"cpu": 1.0}, "local"]

        # Act
        out = gather_objects(objects)

        # Assert
        assert out == objects

    def test_flattens_objects_from_every_rank(self):
        # Arrange
        def fake_all_gather_object(gathered: list, objects: list) -> None:
            gathered[0] = objects
            gathered[1] = [{"gpu": 1.0}]

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.get_world_size", return_value=2),
            patch(
                "agilerl.distributed.process.dist.all_gather_object",
                side_effect=fake_all_gather_object,
            ),
        ):
            out = gather_objects([{"gpu": 0.0}])

        # Assert
        assert out == [{"gpu": 0.0}, {"gpu": 1.0}]


class TestAllreduceMinmaxInt:
    def test_identity_without_process_group(self):
        min_v, max_v = allreduce_minmax_int(3)
        assert (min_v, max_v) == (3, 3)

    def test_single_value_with_world_size_one(self, world_size_one):
        min_v, max_v = allreduce_minmax_int(3)
        assert (min_v, max_v) == (3, 3)

    def test_reduces_min_and_max_across_ranks(self):
        # Arrange — a peer holds 9, so the MAX all-reduce sees [9, -3] from it
        def fake_max(t: torch.Tensor, op: object) -> None:
            assert op == dist.ReduceOp.MAX
            t.copy_(torch.maximum(t, torch.tensor([9, -3])))

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
            patch("agilerl.distributed.process.dist.all_reduce", side_effect=fake_max),
        ):
            bounds = allreduce_minmax_int(5)

        # Assert
        assert bounds == (3, 9)


class TestAnyRank:
    def test_returns_local_flag_without_process_group(self):
        assert any_rank(False) is False
        assert any_rank(True) is True

    def test_is_true_when_any_peer_sets_the_flag(self):
        with patch(
            "agilerl.distributed.process.allreduce_minmax_int",
            return_value=(0, 1),
        ):
            assert any_rank(False) is True

    def test_is_false_when_every_rank_is_clear(self):
        with patch(
            "agilerl.distributed.process.allreduce_minmax_int",
            return_value=(0, 0),
        ):
            assert any_rank(False) is False


class TestAllRanks:
    def test_returns_local_flag_without_process_group(self):
        assert all_ranks(False) is False
        assert all_ranks(True) is True

    def test_is_false_when_any_peer_is_clear(self):
        with patch(
            "agilerl.distributed.process.allreduce_minmax_int",
            return_value=(0, 1),
        ):
            assert all_ranks(True) is False

    def test_is_true_when_every_rank_sets_the_flag(self):
        with patch(
            "agilerl.distributed.process.allreduce_minmax_int",
            return_value=(1, 1),
        ):
            assert all_ranks(True) is True


class TestAggregateMetricsAcrossGpus:
    def test_single_process(self):
        result = aggregate_metrics_across_gpus(torch.tensor([1.0, 2.0, 3.0]))

        assert result == 2.0
        assert isinstance(result, float)

    def test_with_scalar(self):
        result = aggregate_metrics_across_gpus(5.0)

        assert result == 5.0
        assert isinstance(result, float)

    def test_with_ndarray(self):
        result = aggregate_metrics_across_gpus(np.array([1.0, 3.0]))

        assert result == 2.0
        assert isinstance(result, float)

    def test_with_negative_values(self):
        result = aggregate_metrics_across_gpus(torch.tensor([-1.0, -2.0, -3.0]))

        assert result == -2.0
        assert isinstance(result, float)

    def test_with_zero_values(self):
        result = aggregate_metrics_across_gpus(torch.tensor([0.0, 0.0, 0.0]))

        assert result == 0.0
        assert isinstance(result, float)

    def test_averages_across_ranks_when_distributed(self):
        # Arrange
        def halve(tensor, op):
            assert op == dist.ReduceOp.AVG
            tensor.div_(2)

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
            patch("agilerl.distributed.process.dist.all_reduce", side_effect=halve),
        ):
            result = aggregate_metrics_across_gpus(torch.tensor([2.0, 6.0]))

        # Assert
        assert result == 2.0

    def test_dict_aggregates_each_value(self):
        out = aggregate_metrics_dict(
            {"a": torch.tensor([1.0, 3.0]), "b": 4.0},
        )
        assert out == {"a": 2.0, "b": 4.0}


class TestSyncGrads:
    def test_leaves_grads_unchanged_without_process_group(self):
        # Arrange
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.full((2,), 5.0)

        # Act
        sync_grads([param])

        # Assert
        assert torch.equal(param.grad, torch.full((2,), 5.0))

    def test_averages_only_params_with_grad(self, world_size_one):
        # Arrange
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.full((2,), 3.0)
        no_grad_param = nn.Parameter(torch.ones(2))

        # Act
        sync_grads([param, no_grad_param])

        # Assert — world size 1: mean is the local grad
        assert torch.equal(param.grad, torch.full((2,), 3.0))
        assert no_grad_param.grad is None

    def test_empty_params_is_noop_when_distributed(self):
        # Arrange
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.dist.all_reduce") as mock_all_reduce,
        ):
            # Act
            sync_grads([])

        # Assert
        mock_all_reduce.assert_not_called()

    def test_raises_when_any_rank_is_missing_a_grad(self):
        # Arrange
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.ones(2)
        missing = nn.Parameter(torch.ones(2))

        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.get_rank", return_value=3),
            patch(
                "agilerl.distributed.process.allreduce_minmax_int",
                return_value=(1, 1),
            ),
            patch("agilerl.distributed.process.dist.all_reduce") as mock_all_reduce,
            pytest.raises(RuntimeError, match="1 params have no grad on rank 3"),
        ):
            # Act
            sync_grads([param, missing])

        # Assert
        mock_all_reduce.assert_not_called()
        assert missing.grad is None

    def test_coalesces_and_averages_grads_across_world_size(self):
        # Arrange
        p1 = nn.Parameter(torch.ones(2))
        p1.grad = torch.full((2,), 4.0)
        p2 = nn.Parameter(torch.ones(3))
        p2.grad = torch.full((3,), 6.0)

        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch(
                "agilerl.distributed.process.allreduce_minmax_int",
                return_value=(0, 0),
            ),
            patch(
                "agilerl.distributed.process.dist.all_reduce",
                autospec=True,
            ) as mock_all_reduce,
        ):
            mock_all_reduce.side_effect = lambda tensor, op=None: None

            # Act
            sync_grads([p1, p2])

        # Assert — identity SUM, then divide by world size 2
        mock_all_reduce.assert_called_once()
        assert torch.equal(p1.grad, torch.full((2,), 2.0))
        assert torch.equal(p2.grad, torch.full((3,), 3.0))


class TestMaterializeDtensors:
    def test_full_tensors_without_installing_on_module(self):
        linear = nn.Linear(2, 3, bias=False)
        full = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        class FakeDTensor:
            def full_tensor(self):
                return full

        fake = FakeDTensor()
        linear._parameters["weight"] = fake
        with patch.object(dmod, "DTensor", FakeDTensor):
            with materialize_dtensors(fake, None) as gathered:
                assert gathered[0] is full
                assert gathered[1] is None
                assert linear._parameters["weight"] is fake
        assert linear._parameters["weight"] is fake

    def test_noops_on_plain_tensors(self):
        weight = torch.randn(4, 2)
        entered = False
        with materialize_dtensors(weight, None):
            entered = True
        assert entered


class TestFullShapeViews:
    def test_leaves_plain_tensors_unchanged(self):
        linear = nn.Linear(2, 2)
        weight_before = linear.weight.data.clone()

        with full_shape_views(linear, [linear.weight, linear.bias, None]):
            assert torch.equal(linear.weight.data, weight_before)

        assert torch.equal(linear.weight.data, weight_before)

    def test_installs_global_shape_without_full_tensor(self):
        linear = nn.Linear(2, 3, bias=False)
        global_shape = torch.Size([4, 12, 8])

        class FakeDTensor:
            requires_grad = True
            shape = global_shape
            dtype = torch.float32
            device = torch.device("cpu")

            def full_tensor(self):
                msg = "packed experts must not be gathered"
                raise AssertionError(msg)

        fake_dtensor = FakeDTensor()
        with patch.object(dmod, "DTensor", FakeDTensor):
            linear._parameters["weight"] = fake_dtensor

            with full_shape_views(linear, [fake_dtensor, fake_dtensor, None]):
                assert linear.weight.shape == global_shape
                assert (
                    linear.weight.untyped_storage().nbytes()
                    == linear.weight.element_size()
                )

            assert linear._parameters["weight"] is fake_dtensor

    def test_skips_unowned_dtensor_without_gathering(self):
        class FakeDTensor:
            def full_tensor(self):
                msg = "packed experts must not be gathered"
                raise AssertionError(msg)

        fake_dtensor = FakeDTensor()
        with (
            patch.object(dmod, "DTensor", FakeDTensor),
            full_shape_views(nn.Linear(2, 2), [fake_dtensor]),
        ):
            pass

    def test_restores_shard_when_body_raises(self):
        linear = nn.Linear(2, 3, bias=False)

        class FakeDTensor:
            requires_grad = False
            shape = torch.Size([4, 12, 8])
            dtype = torch.float32
            device = torch.device("cpu")

        fake_dtensor = FakeDTensor()
        with patch.object(dmod, "DTensor", FakeDTensor):
            linear._parameters["weight"] = fake_dtensor

            boom = RuntimeError("boom")
            with (
                pytest.raises(RuntimeError, match="boom"),
                full_shape_views(linear, [fake_dtensor]),
            ):
                raise boom

            assert linear._parameters["weight"] is fake_dtensor


class TestSetFullModelStateDict:
    def test_copies_replicated_parameters(self):
        model = nn.Linear(2, 2)
        new_weight = torch.ones_like(model.weight)
        new_bias = torch.zeros_like(model.bias)

        set_full_model_state_dict(
            model, {"weight": new_weight, "bias": new_bias}, strict=True
        )

        assert torch.equal(model.weight.data, new_weight)
        assert torch.equal(model.bias.data, new_bias)

    def test_dtensor_params_scatter_via_distribute_tensor(self, monkeypatch):
        class FakeDTensor(nn.Parameter):
            def to_local(self):
                return self.data

        class FakeSharded:
            def __init__(self, tensor):
                self._tensor = tensor

            def to_local(self):
                return self._tensor

        monkeypatch.setattr("agilerl.distributed.fsdp.DTensor", FakeDTensor)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                weight = FakeDTensor(torch.zeros(2, 2))
                weight.device_mesh = "mesh"
                weight.placements = "placements"
                self.weight = weight
                self.bias = nn.Parameter(torch.zeros(2))

        model = Tiny()
        captured: dict = {}

        def fake_distribute(full, mesh, placements):
            captured["full"] = full.detach().clone()
            captured["mesh"] = mesh
            captured["placements"] = placements
            return FakeSharded(torch.ones(2, 2))

        monkeypatch.setattr(
            "agilerl.distributed.fsdp.distribute_tensor", fake_distribute
        )
        source = torch.full((2, 2), 7.0)
        bias = torch.full((2,), 3.0)
        set_full_model_state_dict(model, {"weight": source, "bias": bias}, strict=True)

        assert captured["mesh"] == "mesh"
        assert captured["placements"] == "placements"
        assert torch.equal(captured["full"], source)
        assert torch.equal(model.weight.data, torch.ones(2, 2))
        assert torch.equal(model.bias.data, bias)

    def test_strict_raises_on_missing_keys(self):
        model = nn.Linear(2, 2)

        with pytest.raises(RuntimeError, match="Missing keys"):
            set_full_model_state_dict(model, {"weight": torch.ones(2, 2)}, strict=True)

    def test_loads_checkpoint_wrapped_parameter_names(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self._checkpoint_wrapped_module = nn.Linear(2, 2)

        model = Tiny()
        source_weight = torch.full((2, 2), 7.0)
        source_bias = torch.full((2,), 3.0)

        set_full_model_state_dict(
            model,
            {"weight": source_weight, "bias": source_bias},
            strict=True,
        )

        assert torch.equal(model._checkpoint_wrapped_module.weight.data, source_weight)
        assert torch.equal(model._checkpoint_wrapped_module.bias.data, source_bias)

    def test_skips_missing_parameters_when_not_strict(self):
        model = nn.Linear(2, 2)
        before = model.bias.data.clone()

        set_full_model_state_dict(model, {"weight": torch.ones(2, 2)})

        assert torch.equal(model.weight.data, torch.ones(2, 2))
        assert torch.equal(model.bias.data, before)

    def test_copies_buffers(self):
        model = nn.Linear(2, 2)
        model.register_buffer("running", torch.zeros(2))

        set_full_model_state_dict(
            model,
            {
                "weight": torch.ones(2, 2),
                "bias": torch.zeros(2),
                "running": torch.ones(2),
            },
        )

        assert torch.equal(model.running, torch.ones(2))

    def test_rejects_non_tensor_values(self):
        model = nn.Linear(2, 2)

        with pytest.raises(TypeError, match="must be a Tensor"):
            set_full_model_state_dict(model, {"weight": 5, "bias": 6})


class TestReshardFsdpModules:
    def test_reshards_fsdp_units(self):
        class ShardUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.reshard = MagicMock()

        model = ShardUnit()

        with patch("agilerl.distributed.fsdp.FSDPModule", ShardUnit):
            reshard_fsdp_modules(model)

        model.reshard.assert_called_once_with()


class TestParameterOwners:
    def test_maps_each_registered_param_to_module_and_name(self):
        # Arrange
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        # Act
        owners = dmod.parameter_owners(
            model, [model[0].weight, model[0].bias, model[1].weight]
        )

        # Assert
        assert owners == {
            id(model[0].weight): (model[0], "weight"),
            id(model[0].bias): (model[0], "bias"),
            id(model[1].weight): (model[1], "weight"),
        }

    def test_empty_input_returns_empty_map(self):
        assert dmod.parameter_owners(nn.Linear(2, 2), []) == {}

    def test_omits_tensors_held_outside_modules(self):
        # Arrange
        listed = [torch.tensor([1.0])]
        keyed_tensor = torch.tensor([1.0])
        keyed = {keyed_tensor: "value"}
        holder = {"tensor": torch.tensor([1.0])}

        # Act
        owners = dmod.parameter_owners(
            nn.Linear(2, 2), [listed[0], keyed_tensor, holder["tensor"]]
        )

        # Assert
        assert owners == {}
        assert keyed[keyed_tensor] == "value"


class TestGatherParamsOwnerless:
    def test_skips_dtensor_without_owner(self):
        class FakeDTensor:
            def full_tensor(self):
                return torch.ones(2, 2)

        fake = FakeDTensor()
        with (
            patch.object(dmod, "DTensor", FakeDTensor),
            dmod.gather_params(nn.Linear(2, 2), [fake, None]) as out,
        ):
            assert torch.equal(out[0], torch.ones(2, 2))
            assert out[1] is None


class TestReplaceConsecutiveBlocks:
    def test_rejects_empty_blocks(self):
        with pytest.raises(ValueError, match="non-empty"):
            dmod._replace_consecutive_blocks(nn.Sequential(nn.Linear(2, 2)), [])

    def test_raises_when_span_not_found(self):
        model = nn.Sequential(nn.Linear(2, 2))

        with pytest.raises(RuntimeError, match="consecutive ModuleList span"):
            dmod._replace_consecutive_blocks(model, [nn.Linear(2, 2)])


class TestGroupTransformerUnits:
    def test_single_unit_returns_units_unchanged(self):
        model = nn.Sequential(nn.Linear(2, 2))
        units = [nn.Linear(2, 2)]

        assert dmod._group_transformer_units(model, units, 2) == units

    def test_trailing_single_chunk_stays_ungrouped(self):
        first, second, third = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
        model = nn.Sequential(nn.ModuleList([first, second, third]))

        grouped = dmod._group_transformer_units(model, [first, second, third], 2)

        assert len(grouped) == 2
        assert grouped[1] is third


class TestReplaceChild:
    def test_raises_when_old_not_found(self):
        model = nn.Sequential(nn.Linear(2, 2))

        with pytest.raises(RuntimeError, match="Could not find parent"):
            dmod._replace_child(model, nn.Linear(2, 2), nn.Linear(2, 2))


class TestResolveCausalLm:
    def test_unwraps_pretrained_model_shell(self):
        inner = nn.Linear(2, 2)
        shell = nn.Linear(2, 2)
        shell.pretrained_model = inner

        assert dmod._resolve_causal_lm(shell) is inner

    def test_unwraps_base_model_shell(self):
        inner = nn.Linear(2, 2)
        shell = nn.Linear(2, 2)
        shell.base_model = inner

        assert dmod._resolve_causal_lm(shell) is inner

    def test_returns_inner_model_with_lm_head(self):
        head = nn.Linear(2, 2)
        inner = nn.Linear(2, 2)
        inner.lm_head = head
        shell = nn.Linear(2, 2)
        shell.model = inner

        assert dmod._resolve_causal_lm(shell) is inner


class TestLoadModelState:
    def test_dense_loads_plain_state_dict(self):
        model = nn.Linear(2, 2)
        state = {
            "weight": torch.full((2, 2), 3.0),
            "bias": torch.full((2,), 4.0),
        }

        DPRuntime().import_model_state(model, state, strict=True)

        assert torch.equal(model.weight.data, state["weight"])
        assert torch.equal(model.bias.data, state["bias"])

    def test_fsdp2_uses_set_full_model_state_dict(self):
        model = nn.Linear(2, 2)
        state = model.state_dict()
        runtime = FSDPRuntime(FSDPConfig())

        with patch("agilerl.distributed.runtime.set_full_model_state_dict") as set_full:
            runtime.import_model_state(model, state, strict=True)

        set_full.assert_called_once_with(model, state, strict=True)


class TestGatherModelState:
    def test_dense_returns_module_state_dict(self):
        model = nn.Linear(2, 2)
        out = DPRuntime().export_model_state(model)
        assert set(out) == {"weight", "bias"}
        assert torch.equal(out["weight"], model.weight.data)

    def test_fsdp2_gathers_full_state_on_cpu(self):
        model = nn.Linear(2, 2)
        expected = {"weight": torch.zeros(2, 2)}
        runtime = FSDPRuntime(FSDPConfig())

        with patch(
            "agilerl.distributed.runtime.get_model_state_dict",
            return_value=expected,
        ) as get_state:
            out = runtime.export_model_state(model)

        assert out is expected
        options = get_state.call_args.kwargs["options"]
        assert options.full_state_dict is True
        assert options.cpu_offload is True


class TestExportOptimizerState:
    def test_rejects_list_shaped_state(self):
        optimizer = MagicMock()
        optimizer.state_dict.return_value = [{"state": {}}]

        with pytest.raises(TypeError, match="must return a dict"):
            DPRuntime().export_optimizer_state(nn.Linear(2, 2), optimizer)

    def test_returns_optimizer_state_dict(self):
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"state": {}, "param_groups": []}

        out = DPRuntime().export_optimizer_state(nn.Linear(2, 2), optimizer)

        assert out == {"state": {}, "param_groups": []}


class TestFSDPOptimizerState:
    def test_export_passthrough_when_not_sharded(self):
        optimizer = MagicMock()
        optimizer.state_dict.return_value = {"state": {}, "param_groups": []}

        out = FSDPRuntime(FSDPConfig()).export_optimizer_state(
            nn.Linear(2, 2), optimizer
        )

        assert out == {"state": {}, "param_groups": []}

    def test_export_unwraps_cpu_offload_optimizer(self):
        inner = MagicMock()
        offload = CPUOffloadOptimizer(inner, pin_memory=False)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = offload
        actor = nn.Linear(2, 2)
        expected = {"state": {}}

        with (
            patch("agilerl.distributed.runtime.FSDPModule", object),
            patch(
                "agilerl.distributed.runtime.get_optimizer_state_dict",
                return_value=expected,
            ) as mock_get,
        ):
            out = FSDPRuntime(FSDPConfig()).export_optimizer_state(actor, optimizer)

        assert out is expected
        _, kwargs = mock_get.call_args
        assert mock_get.call_args.args == (actor, inner)
        assert kwargs["options"].full_state_dict is True
        assert kwargs["options"].cpu_offload is True

    def test_import_scatters_state_and_reoffloads(self):
        model = nn.Linear(2, 2)
        inner = torch.optim.AdamW(model.parameters(), lr=1e-3)
        offload = CPUOffloadOptimizer(inner, pin_memory=False)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = offload
        saved_state = {
            name: {
                "step": torch.tensor(1),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }
            for name, param in model.named_parameters()
        }

        with patch("agilerl.distributed.runtime.FSDPModule", object):
            FSDPRuntime(FSDPConfig()).import_optimizer_state(
                model, optimizer, {"state": saved_state}
            )

        assert set(inner.state) == set(model.parameters())
        assert offload._initialized is True
        for state in inner.state.values():
            assert state["step"].device.type == "cpu"

    def test_import_passthrough_when_not_sharded(self):
        optimizer = MagicMock()
        saved = {"state": {}}

        FSDPRuntime(FSDPConfig()).import_optimizer_state(
            nn.Linear(2, 2), optimizer, saved
        )

        optimizer.load_state_dict.assert_called_once_with(saved)


class TestFSDPRuntimeImportOptimizerState:
    @staticmethod
    def _import(model: nn.Module, inner: torch.optim.Optimizer, state: dict) -> None:
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        with patch("agilerl.distributed.runtime.FSDPModule", object):
            FSDPRuntime(FSDPConfig()).import_optimizer_state(
                model, optimizer, {"state": state}
            )

    def test_loads_every_saved_key_by_canonical_fqn(self):
        # Arrange
        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self._checkpoint_wrapped_module = nn.Linear(2, 1, bias=False)

        model = Wrapped()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
        saved = {
            "weight": {
                "step": torch.tensor(4.0),
                "exp_avg": torch.full((1, 2), 0.5),
                "exp_avg_sq": torch.full((1, 2), 0.25),
                "max_exp_avg_sq": torch.full((1, 2), 0.75),
            }
        }

        # Act
        self._import(model, inner, saved)

        # Assert
        loaded = inner.state[model._checkpoint_wrapped_module.weight]
        assert set(loaded) == {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        assert torch.equal(loaded["max_exp_avg_sq"], torch.full((1, 2), 0.75))

    def test_scatters_dtensor_state_onto_mesh(self):
        # Arrange
        class FakeDTensor(nn.Parameter):
            pass

        model = nn.Module()
        model.weight = FakeDTensor(torch.zeros(2, 2))
        model.weight.device_mesh = SimpleNamespace(device_type="cpu")
        model.weight.placements = ["shard"]
        inner = torch.optim.AdamW([model.weight], lr=1e-3)
        saved_avg = torch.ones(2, 2)
        scattered = torch.full((2, 2), 7.0)

        # Act
        with (
            patch.object(rmod, "DTensor", FakeDTensor),
            patch.object(
                rmod, "distribute_tensor", return_value=scattered
            ) as mock_scatter,
        ):
            self._import(model, inner, {"weight": {"exp_avg": saved_avg}})

        # Assert
        mock_scatter.assert_called_once_with(
            saved_avg, model.weight.device_mesh, model.weight.placements
        )
        assert inner.state[model.weight]["exp_avg"] is scattered

    def test_keeps_non_tensor_state_values(self):
        model = nn.Linear(2, 1, bias=False)
        inner = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

        self._import(model, inner, {"weight": {"momentum_buffer": None}})

        assert inner.state[model.weight] == {"momentum_buffer": None}

    def test_skips_frozen_params(self):
        # Arrange
        model = nn.Linear(2, 2)
        model.bias.requires_grad = False
        inner = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Act
        self._import(model, inner, {"weight": {"exp_avg": torch.zeros(2, 2)}})

        # Assert
        assert set(inner.state) == {model.weight}

    def test_raises_when_trainable_param_has_no_saved_state(self):
        model = nn.Linear(2, 2)
        inner = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with pytest.raises(RuntimeError, match=r"missing state .* 'bias'"):
            self._import(model, inner, {"weight": {"exp_avg": torch.zeros(2, 2)}})

    def test_raises_for_optimizer_param_not_on_actor(self):
        stray = nn.Parameter(torch.ones(2))
        inner = torch.optim.AdamW([stray], lr=1e-3)

        with pytest.raises(RuntimeError, match="not a named parameter"):
            self._import(nn.Linear(2, 2), inner, {})


class TestGatherLayer:
    def test_moves_gathered_weights_to_target_device(self):
        layer = nn.Linear(2, 2)

        with FSDPRuntime(FSDPConfig()).gather_layer(layer, "meta") as (w, b):
            assert w.device.type == "meta"
            assert b is not None
            assert b.device.type == "meta"


class TestCopyAdapterTensors:
    def test_copies_source_lora_into_target(self):
        class LoraMod(nn.Module):
            def __init__(self, value: float):
                super().__init__()
                self.lora_A = nn.Parameter(torch.full((2, 2), value))

        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Module()
                self.block.src = LoraMod(1.0)
                self.block.tgt = LoraMod(0.0)

        actor = Actor()

        FSDPRuntime(FSDPConfig()).copy_adapter_tensors(actor, "src", "tgt")

        assert torch.equal(actor.block.tgt.lora_A.data, torch.ones(2, 2))


class TestImportAdapterTensors:
    def test_loads_adapters_onto_actor_device(self):
        actor = nn.Linear(2, 2)

        with patch(
            "agilerl.utils.llm_utils.load_lora_adapters",
        ) as mock_load:
            FSDPRuntime(FSDPConfig()).import_adapter_tensors(
                actor, nn.Linear(2, 2), "/ckpt", "adapter"
            )

        mock_load.assert_called_once_with(actor, "/ckpt", "adapter", device="cpu")


class TestFSDPPrepareActorOffload:
    def test_rejects_already_offloaded_optimizer(self):
        actor = nn.Linear(2, 2)
        inner = CPUOffloadOptimizer(MagicMock(), pin_memory=False)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner

        with (
            patch(
                "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
                return_value=actor,
            ),
            patch(
                "agilerl.utils.llm_utils.make_llm_optimizer",
                return_value=optimizer,
            ),
        ):
            with pytest.raises(TypeError, match="already CPU-offloaded"):
                FSDPRuntime(FSDPConfig()).prepare_actor(
                    actor,
                    device="cpu",
                    colocated=False,
                    cosine_lr_schedule_config=None,
                    lr=1e-4,
                    lr_critic=None,
                    restore_adapter_trainability=MagicMock(),
                )


class _LoraActor(nn.Module):
    """Tiny module whose trainable param matches ``init_llm_optimizer`` LoRA names."""

    def __init__(self) -> None:
        super().__init__()
        self.actor_lora_A = nn.Parameter(torch.ones(2, 2))
        self.checkpointing_kwargs: dict | None = None

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        self.checkpointing_kwargs = kwargs


def _prepare_dp_actor(
    actor: nn.Module,
    device: str = "cpu",
    cosine_lr_schedule_config: CosineLRScheduleConfig | None = None,
    gradient_checkpointing: bool = False,
    lr: float = 1e-4,
) -> object:
    return DPRuntime().prepare_actor(
        actor,
        device=device,
        colocated=False,
        cosine_lr_schedule_config=cosine_lr_schedule_config,
        lr=lr,
        lr_critic=None,
        restore_adapter_trainability=lambda _: None,
        gradient_checkpointing=gradient_checkpointing,
    )


class TestDenseWrap:
    def test_builds_optimizer_on_placed_actor(self):
        actor = _LoraActor()

        result = _prepare_dp_actor(actor)

        assert result.actor is actor
        assert all(param.device.type == "cpu" for param in result.actor.parameters())
        assert isinstance(result.optimizer, OptimizerWrapper)
        opt_params = [
            param
            for group in result.optimizer.optimizer.param_groups
            for param in group["params"]
        ]
        assert actor.actor_lora_A in opt_params
        assert result.lr_scheduler is None

    def test_enables_hf_gradient_checkpointing(self):
        actor = _LoraActor()

        _prepare_dp_actor(actor, gradient_checkpointing=True)

        assert actor.checkpointing_kwargs == {
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }

    def test_builds_cosine_scheduler_when_configured(self):
        actor = _LoraActor()
        config = CosineLRScheduleConfig(num_epochs=10, warmup_proportion=0.1)

        result = _prepare_dp_actor(actor, cosine_lr_schedule_config=config)

        assert result.lr_scheduler is not None
        assert result.lr_scheduler.get_last_lr()[0] == pytest.approx(1e-8)

    def test_rejects_actor_to_that_returns_non_module(self):
        actor = nn.Linear(2, 2)
        actor.to = MagicMock(return_value="not-a-module")

        with pytest.raises(TypeError, match=r"must return nn\.Module"):
            _prepare_dp_actor(actor, device="meta")

    def test_rejects_gradient_checkpointing_without_support(self):
        with pytest.raises(TypeError, match="does not support"):
            _prepare_dp_actor(nn.Linear(2, 2), gradient_checkpointing=True)


class TestActorComputeDevice:
    def test_dense_uses_param_device(self):
        actor = nn.Linear(2, 2)
        fallback = torch.device("cuda:0")
        assert DPRuntime().actor_compute_device(actor, fallback) == torch.device("cpu")

    def test_dense_empty_actor_uses_fallback(self):
        fallback = torch.device("cpu")
        assert DPRuntime().actor_compute_device(nn.Sequential(), fallback) is fallback

    def test_fsdp2_uses_fallback(self):
        actor = nn.Linear(2, 2)
        fallback = torch.device("cuda:0")
        runtime = FSDPRuntime(FSDPConfig())
        assert runtime.actor_compute_device(actor, fallback) is fallback


class TestIsSharded:
    def test_dense_is_not_sharded(self):
        assert DPRuntime().is_sharded is False

    def test_fsdp2_is_sharded(self):
        assert FSDPRuntime(FSDPConfig()).is_sharded is True


class TestDPRuntimeBackward:
    def test_steps_and_returns_lr_on_accumulation_boundary(self):
        # Arrange
        model = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(model.weight)
        inner = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(inner, step_size=1, gamma=0.5)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        optimizer.step.side_effect = inner.step
        optimizer.zero_grad.side_effect = inner.zero_grad
        runtime = DPRuntime()
        x = torch.ones(1, 2)

        # Act
        first = runtime.backward(
            model(x).sum(), optimizer, 2, model, lr_scheduler=scheduler
        )
        weight_mid_window = model.weight.detach().clone()
        second = runtime.backward(
            model(x).sum(), optimizer, 2, model, lr_scheduler=scheduler
        )

        # Assert — two half-scaled grads of 1.0 sum to 1.0; SGD lr 0.1
        assert first is None
        assert torch.equal(weight_mid_window, torch.zeros(1, 2))
        assert torch.allclose(model.weight, torch.full((1, 2), -0.1))
        assert second == OptimizerStep(
            grad_norm_pre=pytest.approx(2**0.5),
            grad_norm_post=pytest.approx(2**0.5),
            lr=pytest.approx(0.05),
        )
        assert model.weight.grad is None


class TestFSDPRuntimeBackward:
    def test_syncs_replicated_grads_not_dtensors(self, monkeypatch):
        class FakeDTensor(nn.Parameter):
            pass

        monkeypatch.setattr("agilerl.distributed.runtime.DTensor", FakeDTensor)
        replicated = nn.Parameter(torch.ones(2))
        replicated.grad = torch.ones(2)
        sharded = FakeDTensor(torch.ones(2))
        sharded.grad = torch.ones(2)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value.param_groups = [
            {"params": [replicated, sharded]}
        ]
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=False))
        synced: list[list[nn.Parameter]] = []

        def capture(params):
            synced.append(list(params))

        monkeypatch.setattr("agilerl.distributed.runtime.sync_grads", capture)
        runtime.backward(MagicMock(), optimizer, 1, actor=MagicMock())

        assert synced == [[replicated]]
        optimizer.step.assert_called_once()

    def test_does_not_sync_before_step_boundary(self, monkeypatch):
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.ones(2)
        optimizer = MagicMock()
        optimizer.optimizer.param_groups = [{"params": [param]}]
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=False))
        synced: list[object] = []
        monkeypatch.setattr(
            "agilerl.distributed.runtime.sync_grads",
            lambda params: synced.append(list(params)),
        )

        runtime.backward(MagicMock(), optimizer, 2, actor=MagicMock())

        assert synced == []
        optimizer.step.assert_not_called()

    def test_rejects_defer_without_sharded_actor(self):
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=True))

        with pytest.raises(TypeError, match="FSDP2-sharded actor"):
            runtime.backward(MagicMock(), MagicMock(), 1, actor=nn.Linear(2, 2))

    def test_clips_and_steps_scheduler_at_boundary(self):
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.ones(2) * 10
        inner = MagicMock()
        inner.param_groups = [{"params": [param]}]
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [0.5]
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=False))

        step = runtime.backward(
            MagicMock(),
            optimizer,
            1,
            actor=MagicMock(),
            max_grad_norm=1.0,
            lr_scheduler=scheduler,
        )

        assert torch.linalg.vector_norm(param.grad).item() == pytest.approx(1.0)
        assert step == OptimizerStep(
            grad_norm_pre=pytest.approx(200**0.5),
            grad_norm_post=pytest.approx(1.0),
            lr=0.5,
        )
        scheduler.step.assert_called_once_with()
        optimizer.step.assert_called_once_with()
        optimizer.zero_grad.assert_called_once_with()


class TestDeferGradSync:
    def test_fsdp_config_defaults_on(self):
        assert FSDPConfig().defer_grad_sync is True

    def test_toggles_sync_across_accumulation_window(self):
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=True))
        optimizer = MagicMock()
        loss = MagicMock()
        model = MagicMock(spec=FSDPModule)

        runtime.backward(loss, optimizer, 2, actor=model)
        model.set_requires_gradient_sync.assert_called_once_with(False, recurse=True)
        runtime.backward(loss, optimizer, 2, actor=model)
        model.set_requires_gradient_sync.assert_called_with(True, recurse=True)
        assert model.set_requires_gradient_sync.call_count == 2

        optimizer.step.assert_called_once()

    def test_disabled_does_not_toggle(self):
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=False))
        optimizer = MagicMock()
        loss = MagicMock()
        model = MagicMock()

        runtime.backward(loss, optimizer, 1, actor=model)

        model.set_requires_gradient_sync.assert_not_called()
        optimizer.step.assert_called_once()


class TestRaiseOnAnyRank:
    def test_runs_block_without_process_group(self):
        with pmod.raise_on_any_rank():
            x = 7

        assert x == 7

    def test_reraises_local_error_without_process_group(self):
        msg = "boom"
        with pytest.raises(ValueError, match=msg):
            with pmod.raise_on_any_rank():
                raise ValueError(msg)

    def test_runs_as_decorator_without_process_group(self):
        @pmod.raise_on_any_rank()
        def body() -> int:
            return 7

        assert body() == 7

    def test_decorator_reraises_local_error_without_process_group(self):
        msg = "boom"

        @pmod.raise_on_any_rank()
        def body() -> None:
            raise ValueError(msg)

        with pytest.raises(ValueError, match=msg):
            body()

    def test_raises_on_healthy_rank_when_peer_failed(self):
        def peer_failed(t: torch.Tensor, op: object) -> None:
            t.fill_(1)

        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch(
                "agilerl.distributed.process.dist.all_reduce", side_effect=peer_failed
            ),
            pytest.raises(RuntimeError, match="Peer rank failed"),
        ):
            with pmod.raise_on_any_rank():
                pass

    def test_reraises_local_error_after_joining_peers(self):
        msg = "local"
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.dist.all_reduce") as mock_all_reduce,
            pytest.raises(ValueError, match=msg),
        ):
            with pmod.raise_on_any_rank():
                raise ValueError(msg)

        assert mock_all_reduce.call_args.args[0].item() == 1

    def test_reraises_base_exception_without_collective(self):
        with pytest.raises(KeyboardInterrupt):
            with pmod.raise_on_any_rank():
                raise KeyboardInterrupt


class TestSetSeed:
    def test_reproducible_torch_draws(self):
        # Arrange / Act
        set_seed(123)
        first = torch.rand(3)
        set_seed(123)
        second = torch.rand(3)

        # Assert
        assert torch.equal(first, second)


class TestResolveDevice:
    def test_honours_requested_device_when_not_distributed_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert resolve_device("cpu") == "cpu"

    def test_falls_back_to_cpu_when_no_accelerators(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert resolve_device(None) == "cpu"

    def test_prefers_mps_over_cpu(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert resolve_device(None) == "mps"

    def test_uses_cpu_not_mps_when_distributed_without_cuda(self, world_size_one):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert resolve_device(None) == "cpu"

    def test_pins_cuda_local_rank_when_distributed(self, world_size_one):
        # Act / Assert
        if torch.cuda.is_available():
            assert resolve_device("cpu") == "cuda:0"
        else:
            assert resolve_device("cpu") == "cpu"


class TestFSDPBlockGroup:
    def test_dispatches_dict_masks_by_inner_block_type(self):
        class Block(nn.Module):
            def __init__(self, block_type: str):
                super().__init__()
                self.block_type = block_type
                self.lin = nn.Linear(2, 2)
                self.seen: list[object] = []

            def forward(self, hidden_states, attention_mask=None, **_kwargs):
                self.seen.append(attention_mask)
                return self.lin(hidden_states)

        mamba = Block("linear_attention")
        moe = Block("moe")
        group = dmod.FSDPBlockGroup([mamba, moe])
        hidden = torch.zeros(1, 2)

        out = group(
            hidden,
            attention_mask={
                "linear_attention": "linear",
                "full_attention": "causal",
            },
        )

        assert out.shape == hidden.shape
        assert mamba.seen == ["linear"]
        assert moe.seen == [None]
        assert group.block_type == "linear_attention"


class TestSetPrefetch:
    def test_chains_embed_block_and_lm_head_units(self):
        class FakeUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self.set_modules_to_forward_prefetch = MagicMock()
                self.set_modules_to_backward_prefetch = MagicMock()

        class Language(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = FakeUnit()
                self.layers = nn.ModuleList([nn.Linear(2, 2)])

        class Causal(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Language()
                self.lm_head = FakeUnit()

        model = Causal()
        block = FakeUnit()

        with patch.object(dmod, "FSDPModule", FakeUnit):
            dmod._set_prefetch(model, [block], prefetch_units=1)

        embed = model.model.embed_tokens
        head = model.lm_head
        embed.set_modules_to_forward_prefetch.assert_called_once_with([block])
        block.set_modules_to_forward_prefetch.assert_called_once_with([head])
        block.set_modules_to_backward_prefetch.assert_called_once_with([embed])
        head.set_modules_to_backward_prefetch.assert_called_once_with([block])


class TestApplyFsdp2:
    per_block = FSDPConfig(wrap_every_n_blocks=1)

    def test_shards_transformer_blocks_and_root(self, world_size_one):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        # Arrange
        model = Model()
        blocks = list(model.layers)
        sharded: list[nn.Module] = []

        def _record(module, **_kwargs):
            sharded.append(module)
            return module

        # Act
        with patch.object(dmod, "fully_shard", side_effect=_record):
            out = apply_fsdp2(model, self.per_block)

        # Assert — same object; each block and the root are sharded
        assert out is model
        assert set(sharded) == {blocks[0], blocks[1], model}

    def test_replicates_embed_and_shards_untied_lm_head_before_root(
        self, world_size_one
    ):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

        class LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(4, 2)
                self.layers = nn.ModuleList([Block(), Block()])
                self.norm = nn.LayerNorm(2)

        class CausalLM(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self, tie: bool):
                super().__init__()
                self.config = type("Cfg", (), {"tie_word_embeddings": tie})()
                self.model = LanguageModel()
                self.lm_head = nn.Linear(2, 4, bias=False)

        # Arrange — untied: embed + lm_head + blocks + root
        untied = CausalLM(tie=False)
        sharded: list[nn.Module] = []
        shard_kwargs: dict[int, dict] = {}

        def _record(module, **kwargs):
            sharded.append(module)
            shard_kwargs[id(module)] = kwargs
            return module

        # Act
        with patch.object(dmod, "fully_shard", side_effect=_record):
            apply_fsdp2(untied, self.per_block)

        # Assert — embeddings stay replicated; untied lm_head is its own unit
        assert untied.model.embed_tokens not in sharded
        assert untied.lm_head in sharded
        assert untied.model.norm not in sharded
        assert sharded[-1] is untied
        assert set(sharded) == {
            untied.model.layers[0],
            untied.model.layers[1],
            untied.lm_head,
            untied,
        }
        assert shard_kwargs[id(untied.lm_head)]["reshard_after_forward"] is False
        assert shard_kwargs[id(untied)]["reshard_after_forward"] is True
        assert (
            untied.model.embed_tokens.weight
            in shard_kwargs[id(untied)]["ignored_params"]
        )

        # Arrange — tied: blocks + root; skip embed and lm_head units
        tied = CausalLM(tie=True)
        sharded.clear()
        shard_kwargs.clear()
        with patch.object(dmod, "fully_shard", side_effect=_record):
            apply_fsdp2(tied, self.per_block)

        # Assert
        assert tied.model.embed_tokens not in sharded
        assert tied.lm_head not in sharded
        assert sharded[-1] is tied
        assert (
            tied.model.embed_tokens.weight in shard_kwargs[id(tied)]["ignored_params"]
        )
        assert tied.lm_head.weight in shard_kwargs[id(tied)]["ignored_params"]

    def test_checkpoint_wraps_blocks_then_shards_the_wrapper(self, world_size_one):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        model = Model()
        raw_blocks = list(model.layers)
        sharded: list[nn.Module] = []
        captured_blocks: list[nn.Module] = []

        def _record(module, **_kwargs):
            sharded.append(module)
            return module

        def _capture(_model, block_units, prefetch_count=1):
            captured_blocks.extend(block_units)

        with (
            patch.object(dmod, "fully_shard", side_effect=_record),
            patch.object(dmod, "_set_prefetch", side_effect=_capture),
        ):
            apply_fsdp2(model, self.per_block, gradient_checkpointing=True)

        assert isinstance(model.layers[0], CheckpointWrapper)
        assert isinstance(model.layers[1], CheckpointWrapper)
        assert next(iter(model.layers[0].children())) is raw_blocks[0]
        assert next(iter(model.layers[1].children())) is raw_blocks[1]
        assert raw_blocks[0] not in sharded
        assert raw_blocks[1] not in sharded
        assert sharded[0] is model.layers[0]
        assert sharded[1] is model.layers[1]
        assert captured_blocks == [model.layers[0], model.layers[1]]

    def test_sets_forward_prefetch_on_consecutive_blocks(self, world_size_one):
        class PrefetchBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self.prefetch: list[list[nn.Module]] = []
                self.backward_prefetch: list[list[nn.Module]] = []

            def set_modules_to_forward_prefetch(self, modules):
                self.prefetch.append(list(modules))

            def set_modules_to_backward_prefetch(self, modules):
                self.backward_prefetch.append(list(modules))

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["PrefetchBlock"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [PrefetchBlock(), PrefetchBlock(), PrefetchBlock()]
                )

        model = Model()
        blocks = list(model.layers)

        with (
            patch.object(dmod, "fully_shard", side_effect=lambda m, **_kw: m),
            patch.object(dmod, "FSDPModule", PrefetchBlock),
        ):
            apply_fsdp2(model, self.per_block)

        assert [block.prefetch for block in blocks] == [
            [[blocks[1]]],
            [[blocks[2]]],
            [],
        ]
        assert [block.backward_prefetch for block in blocks] == [
            [],
            [[blocks[0]]],
            [[blocks[1]]],
        ]

    def test_prefetch_units_two_prefetches_two_neighbours(self, world_size_one):
        class PrefetchBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self.prefetch: list[list[nn.Module]] = []
                self.backward_prefetch: list[list[nn.Module]] = []

            def set_modules_to_forward_prefetch(self, modules):
                self.prefetch.append(list(modules))

            def set_modules_to_backward_prefetch(self, modules):
                self.backward_prefetch.append(list(modules))

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["PrefetchBlock"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [PrefetchBlock(), PrefetchBlock(), PrefetchBlock()]
                )

        model = Model()
        blocks = list(model.layers)

        with (
            patch.object(dmod, "fully_shard", side_effect=lambda m, **_kw: m),
            patch.object(dmod, "FSDPModule", PrefetchBlock),
        ):
            apply_fsdp2(model, FSDPConfig(prefetch_units=2, wrap_every_n_blocks=1))

        assert [block.prefetch for block in blocks] == [
            [[blocks[1], blocks[2]]],
            [[blocks[2]]],
            [],
        ]
        assert [block.backward_prefetch for block in blocks] == [
            [],
            [[blocks[0]]],
            [[blocks[1], blocks[0]]],
        ]

    def test_forwards_mesh_to_fully_shard(self, world_size_one):
        model = nn.Linear(2, 2)
        mesh = object()
        seen: dict[int, dict] = {}

        def _record(module, **kwargs):
            seen[id(module)] = kwargs
            return module

        with patch.object(dmod, "fully_shard", side_effect=_record):
            apply_fsdp2(model, self.per_block, mesh=mesh)

        assert seen[id(model)]["mesh"] is mesh

    def test_skips_top_level_packed_experts(self, world_size_one):
        class PackedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.randn(2, 4, 2))
                self.down_proj = nn.Parameter(torch.randn(2, 2, 2))
                self.act_fn = nn.SiLU()

            def forward(self, hidden_states, top_k_index, top_k_weights):
                return hidden_states

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["PackedExperts"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([PackedExperts()])

        model = Model()
        sharded: list[nn.Module] = []

        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model, self.per_block)

        assert model.layers[0] not in sharded
        assert model in sharded

    def test_persists_params_below_threshold(self, world_size_one):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(64, 64)
                self.norm = nn.LayerNorm(64)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block()])

        model = Model()
        block = model.layers[0]
        seen: dict[int, dict] = {}

        def _record(module, **kwargs):
            seen[id(module)] = kwargs
            return module

        with patch.object(dmod, "fully_shard", side_effect=_record):
            apply_fsdp2(
                model,
                FSDPConfig(
                    wrap_every_n_blocks=1,
                    param_persistence_threshold=200,
                ),
            )

        ignored = seen[id(block)]["ignored_params"]
        assert block.norm.weight in ignored
        assert block.norm.bias in ignored
        assert block.lin.bias in ignored
        assert block.lin.weight not in ignored

    def test_casts_persisted_params_to_mixed_precision_dtype(self, world_size_one):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(64, 64)
                self.norm = nn.LayerNorm(64)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block()])

        model = Model()
        block = model.layers[0]
        assert block.norm.weight.dtype == torch.float32

        with patch.object(dmod, "fully_shard", side_effect=lambda m, **_kw: m):
            apply_fsdp2(
                model,
                FSDPConfig(
                    wrap_every_n_blocks=1,
                    param_persistence_threshold=200,
                ),
            )

        assert block.norm.weight.dtype == torch.bfloat16
        assert block.norm.bias.dtype == torch.bfloat16
        assert block.lin.bias.dtype == torch.bfloat16
        assert block.lin.weight.dtype == torch.float32

    def test_threshold_zero_shards_every_parameter(self, world_size_one):
        model = nn.Linear(2, 2)
        seen_kwargs: list[dict] = []

        with patch.object(
            dmod,
            "fully_shard",
            side_effect=lambda m, **kw: seen_kwargs.append(kw) or m,
        ):
            apply_fsdp2(model, FSDPConfig(param_persistence_threshold=0))

        assert seen_kwargs
        assert "ignored_params" not in seen_kwargs[-1]

    def test_shards_root_only_without_no_split_metadata(self, world_size_one):
        # Arrange
        model = nn.Linear(2, 2)
        sharded: list[nn.Module] = []

        # Act
        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            out = apply_fsdp2(model)

        # Assert
        assert out is model
        assert sharded == [model]

    def test_requests_cpu_offload_policy_when_configured(self, world_size_one):
        # Arrange
        model = nn.Linear(2, 2)
        seen_kwargs: list[dict] = []

        # Act
        with patch.object(
            dmod,
            "fully_shard",
            side_effect=lambda m, **kw: seen_kwargs.append(kw) or m,
        ):
            apply_fsdp2(model, FSDPConfig(cpu_offload=True))

        # Assert — FSDPConfig.cpu_offload surfaces as offload_policy
        assert seen_kwargs
        assert "offload_policy" in seen_kwargs[-1]

    def test_applies_default_mixed_precision_policy(self, world_size_one):
        # Arrange
        model = nn.Linear(2, 2)
        seen_kwargs: list[dict] = []

        # Act
        with patch.object(
            dmod,
            "fully_shard",
            side_effect=lambda m, **kw: seen_kwargs.append(kw) or m,
        ):
            apply_fsdp2(model)

        # Assert — bf16 params / fp32 reduce
        assert seen_kwargs
        policy = seen_kwargs[-1]["mp_policy"]
        assert policy.param_dtype == torch.bfloat16
        assert policy.reduce_dtype == torch.float32

    def test_threads_explicit_mixed_precision_dtypes(self, world_size_one):
        model = nn.Linear(2, 2)
        seen_kwargs: list[dict] = []

        with patch.object(
            dmod,
            "fully_shard",
            side_effect=lambda m, **kw: seen_kwargs.append(kw) or m,
        ):
            apply_fsdp2(
                model,
                FSDPConfig(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                ),
            )

        policy = seen_kwargs[-1]["mp_policy"]
        assert policy.param_dtype == torch.bfloat16
        assert policy.reduce_dtype == torch.float32

    def test_registers_generate_as_fsdp_forward_method(self, world_size_one):
        # Arrange — PEFT/HF actors expose generate; FSDP must all-gather for it
        class GenModel(nn.Module):
            def forward(self, x):
                return x

            def generate(self, *args, **kwargs):
                return args

        model = GenModel()
        registered: list[tuple[nn.Module, str]] = []

        # Act
        with (
            patch.object(dmod, "fully_shard", side_effect=lambda m, **_kw: m),
            patch.object(
                dmod,
                "register_fsdp_forward_method",
                side_effect=lambda m, name: registered.append((m, name)),
            ),
        ):
            apply_fsdp2(model)

        # Assert — PEFT generate skips root ``__call__``; bare ``.forward``
        # skips it too. Both names must be registered.
        assert registered == [(model, "generate"), (model, "forward")]

    def test_does_not_shard_packed_experts(self, world_size_one):
        class PackedExperts(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.randn(2, 4, 2))
                self.down_proj = nn.Parameter(torch.randn(2, 2, 2))
                self.act_fn = nn.SiLU()

            def forward(self, hidden_states, top_k_index, top_k_weights):
                return hidden_states

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self.experts = PackedExperts()

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block", "PackedExperts"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        model = Model()
        sharded: list[nn.Module] = []

        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model, self.per_block)

        assert model.layers[0] in sharded
        assert model.layers[1] in sharded
        assert model in sharded
        assert model.layers[0].experts not in sharded
        assert model.layers[1].experts not in sharded

    def test_wraps_outermost_no_split_only(self, world_size_one):
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.lin = nn.Linear(2, 2)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Outer", "Inner"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Outer(), Outer()])

        model = Model()
        sharded: list[nn.Module] = []

        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model, self.per_block)

        assert model.layers[0] in sharded
        assert model.layers[1] in sharded
        assert model in sharded
        assert model.layers[0].inner not in sharded
        assert model.layers[1].inner not in sharded

    def test_grouped_forward_routes_mixed_block_type_masks(
        self, world_size_one, monkeypatch
    ):
        class TypedBlock(nn.Module):
            def __init__(self, block_type: str):
                super().__init__()
                self.block_type = block_type
                self.lin = nn.Linear(2, 2)
                self.seen: list[object] = []

            def forward(self, hidden_states, attention_mask=None, **_kwargs):
                self.seen.append(attention_mask)
                return self.lin(hidden_states)

        class Language(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(4, 2)
                self.layers = nn.ModuleList(
                    [TypedBlock("linear_attention"), TypedBlock("moe")]
                )

            def forward(
                self,
                input_ids=None,
                inputs_embeds=None,
                attention_mask=None,
                **_kwargs,
            ):
                if (input_ids is None) ^ (inputs_embeds is not None):
                    msg = "You must specify exactly one of input_ids or inputs_embeds"
                    raise ValueError(msg)
                hidden = (
                    inputs_embeds
                    if inputs_embeds is not None
                    else self.embeddings(input_ids)
                )
                mapping = attention_mask if isinstance(attention_mask, dict) else {}
                for layer in self.layers:
                    hidden = layer(
                        hidden,
                        attention_mask=mapping.get(layer.block_type),
                    )
                return hidden

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["TypedBlock"]

            def __init__(self):
                super().__init__()
                self.model = Language()

            def forward(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        def fake_mapping(_module, **kwargs):
            return (
                {"full_attention": "causal", "linear_attention": "linear"},
                kwargs.get("position_ids"),
            )

        monkeypatch.setattr(
            "agilerl.architectures.nemotron_h.mamba.block_type_mask_mapping",
            fake_mapping,
        )
        model = Model()
        mamba = model.model.layers[0]
        moe = model.model.layers[1]

        with patch.object(dmod, "fully_shard", side_effect=lambda m, **_kw: m):
            apply_fsdp2(model, FSDPConfig(wrap_every_n_blocks=2, prefetch_units=1))

        hidden = model.model(input_ids=torch.zeros(1, 2, dtype=torch.long))
        group = model.model.layers[0]

        assert isinstance(group, dmod.FSDPBlockGroup)
        assert hidden.shape == (1, 2, 2)
        assert mamba.seen == ["linear"]
        assert moe.seen == [None]
        assert group.block_type == "linear_attention"

    def test_uniform_groups_do_not_wrap_language_forward(self, world_size_one):
        class TypedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_type = "moe"
                self.lin = nn.Linear(2, 2)

        class Language(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(4, 2)
                self.layers = nn.ModuleList([TypedBlock(), TypedBlock()])

            def forward(self, hidden_states=None, **_kwargs):
                return hidden_states

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["TypedBlock"]

            def __init__(self):
                super().__init__()
                self.model = Language()

        model = Model()
        original = type(model.model).forward

        with patch.object(dmod, "fully_shard", side_effect=lambda m, **_kw: m):
            apply_fsdp2(model, FSDPConfig(wrap_every_n_blocks=2, prefetch_units=1))

        assert type(model.model).forward is original

    def test_grouped_mask_forward_is_idempotent(self):
        class Language(nn.Module):
            def forward(self, **kwargs):
                return kwargs

        language = Language()
        dmod._install_grouped_mask_forward(language)
        first = type(language).forward
        dmod._install_grouped_mask_forward(language)

        assert type(language).forward is first

    def test_groups_every_n_transformer_blocks(self, world_size_one):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

            def forward(self, hidden_states):
                return self.lin(hidden_states)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block() for _ in range(4)])

        model = Model()
        raw = list(model.layers)
        sharded: list[nn.Module] = []

        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model, FSDPConfig(wrap_every_n_blocks=2, prefetch_units=1))

        assert len(model.layers) == 2
        assert isinstance(model.layers[0], dmod.FSDPBlockGroup)
        assert isinstance(model.layers[1], dmod.FSDPBlockGroup)
        assert list(model.layers[0].blocks) == raw[:2]
        assert list(model.layers[1].blocks) == raw[2:]
        assert sharded[0] is model.layers[0]
        assert sharded[1] is model.layers[1]
        assert raw[0] not in sharded
        hidden = torch.zeros(1, 2, dtype=torch.bfloat16)
        assert model.layers[0](hidden).shape == hidden.shape

    def test_default_wrap_keeps_one_unit_per_block(self, world_size_one):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        model = Model()
        raw = list(model.layers)
        sharded: list[nn.Module] = []

        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model)

        assert list(model.layers) == raw
        assert raw[0] in sharded
        assert raw[1] in sharded
        assert sharded[-1] is model

    def test_raises_without_process_group(self):
        with pytest.raises(RuntimeError, match="initialised process group"):
            apply_fsdp2(nn.Linear(2, 2), FSDPConfig())


class TestRestoreNonpersistentBuffers:
    def test_restores_via_canonical_name(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self._checkpoint_wrapped_module = nn.Linear(2, 2)
                self._checkpoint_wrapped_module.register_buffer(
                    "running", torch.zeros(2)
                )

        model = Tiny()

        restored = dmod._restore_nonpersistent_buffers(
            model, {"running": torch.ones(2)}
        )

        assert restored == 1
        assert torch.equal(model._checkpoint_wrapped_module.running, torch.ones(2))

    def test_skips_buffers_missing_from_snapshot(self):
        model = nn.Linear(2, 2)
        model.register_buffer("running", torch.zeros(2))

        restored = dmod._restore_nonpersistent_buffers(model, {})

        assert restored == 0
        assert torch.equal(model.running, torch.zeros(2))


class TestShareFsdpCommStreams:
    def test_returns_early_for_meta_parameters(self):
        class FakeUnit(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2).to("meta")

        model = FakeUnit()

        with patch.object(dmod, "FSDPModule", FakeUnit):
            dmod._share_fsdp_comm_streams(model)

    def test_lazy_inits_dtensor_param_groups(self):
        class FakeUnit(nn.Module):
            def __init__(self, param_group):
                super().__init__()
                self.lin = nn.Linear(2, 2)
                self._state = SimpleNamespace(
                    _comm_ctx=MagicMock(),
                    _fsdp_param_group=param_group,
                )

            def _get_fsdp_state(self):
                return self._state

        mixed_group = SimpleNamespace(
            fsdp_params=[SimpleNamespace(sharded_param=torch.ones(2))],
            lazy_init=MagicMock(),
        )
        dense_group = SimpleNamespace(
            fsdp_params=[SimpleNamespace(sharded_param=_FakeDTensor(torch.ones(2)))],
            lazy_init=MagicMock(),
        )

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.mixed = FakeUnit(mixed_group)
                self.dense = FakeUnit(dense_group)

        model = Model()

        with (
            patch.object(dmod, "FSDPModule", FakeUnit),
            patch.object(dmod, "DTensor", _FakeDTensor),
            patch.object(dmod, "share_comm_ctx") as mock_share,
        ):
            dmod._share_fsdp_comm_streams(model)

        mock_share.assert_called_once()
        mixed_group.lazy_init.assert_not_called()
        dense_group.lazy_init.assert_called_once_with()


class TestCPUOffloadOptimizer:
    """Behavior of the optimizer-state CPU offload wrapper.

    Uses a real ``nn.Linear`` + ``torch.optim.AdamW`` so state tensors are real;
    ``fully_shard`` is never involved. CUDA-only round-trips are skipped on
    CPU-only hosts.
    """

    def _make_optimizer(self, device: str) -> tuple[nn.Module, torch.optim.AdamW]:
        model = nn.Linear(4, 4).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, opt

    def _step_once(self, model: nn.Module, opt: torch.optim.Optimizer, device: str):
        x = torch.randn(2, 4, device=device)
        y = model(x).sum()
        y.backward()
        opt.step()
        opt.zero_grad()

    def test_step_first_call_runs_underlying_and_moves_states_to_cpu(self):
        # Arrange — CPU model; first step creates states, then moves to CPU
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)

        # Act
        self._step_once(model, offload, "cpu")

        # Assert — AdamW states exist and live on CPU
        assert offload._initialized is True
        for state in offload.state.values():
            for v in state.values():
                assert isinstance(v, torch.Tensor)
                assert v.device.type == "cpu"

    @cuda_required
    def test_step_subsequent_calls_round_trip_via_cuda(self):
        # Arrange — params on CUDA; first step parks states on CPU
        model, opt = self._make_optimizer("cuda")
        offload = CPUOffloadOptimizer(opt, pin_memory=True)
        self._step_once(model, offload, "cuda")
        # states now on CPU (pinned)
        first_state = next(iter(offload.state.values()))
        assert first_state["exp_avg"].device.type == "cpu"

        # Act — second step moves states to CUDA, runs, moves back to CPU
        self._step_once(model, offload, "cuda")

        # Assert — states back on CPU after the round-trip
        for state in offload.state.values():
            for v in state.values():
                assert v.device.type == "cpu"

    def test_step_with_pin_memory_false_does_not_pin(self):
        # Arrange
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)

        # Act
        self._step_once(model, offload, "cpu")

        # Assert — states on CPU but not pinned
        for state in offload.state.values():
            for v in state.values():
                assert v.is_pinned() is False

    def test_step_with_pin_memory_true_pins_cpu_states(self):
        # Arrange
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=True)
        pinned = torch.ones(2, 2)

        # Act
        with patch.object(torch.Tensor, "pin_memory", return_value=pinned) as mock_pin:
            self._step_once(model, offload, "cpu")

        # Assert — every moment tensor went through pin_memory(); ``step`` stays put
        assert mock_pin.call_count > 0
        for state in offload.state.values():
            for key, v in state.items():
                if key != "step":
                    assert v is pinned

    def test_to_device_moves_cuda_with_non_blocking(self):
        _model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        tensor = MagicMock()

        out = offload._to_device(tensor, "cuda")

        tensor.to.assert_called_once_with("cuda", non_blocking=True)
        assert out is tensor.to.return_value

    def _record_moves(self, offload):
        devices: list[str] = []
        real_to_device = CPUOffloadOptimizer._to_device

        def record(self, tensor, device):
            devices.append(device)
            return real_to_device(self, tensor, "cpu")

        return devices, patch.object(CPUOffloadOptimizer, "_to_device", record)

    def test_step_after_init_moves_states_around_step(self):
        # Arrange — stepped once so states exist and live on CPU
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        self._step_once(model, offload, "cpu")

        # Act — moves are recorded; tensors stay on CPU in this process
        devices, move_patch = self._record_moves(offload)
        with move_patch:
            self._step_once(model, offload, "cpu")

        # Assert — every state goes to CUDA for step(), then back to CPU
        # Adam's scalar ``step`` stays on CPU; only moment tensors move.
        n_states = sum(
            1 for state in opt.state.values() for key in state if key != "step"
        )
        assert n_states > 0
        assert devices == ["cuda"] * n_states + ["cpu"] * n_states

    def test_state_dict_after_init_moves_states_around_snapshot(self):
        # Arrange
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        self._step_once(model, offload, "cpu")

        # Act
        devices, move_patch = self._record_moves(offload)
        with move_patch:
            sd = offload.state_dict()

        # Assert — snapshot captured; every state round-tripped via CUDA
        assert sd["state"], "state dict should contain optimizer states"
        # Adam's scalar ``step`` stays on CPU; only moment tensors move.
        n_states = sum(
            1 for state in opt.state.values() for key in state if key != "step"
        )
        assert devices == ["cuda"] * n_states + ["cpu"] * n_states

    def test_load_state_dict_cpu_moves_states_and_marks_initialized(self):
        # Arrange — capture a state dict from a stepped optimizer
        model_a, opt_a = self._make_optimizer("cpu")
        offload_a = CPUOffloadOptimizer(opt_a, pin_memory=False)
        self._step_once(model_a, offload_a, "cpu")
        _devices_a, snapshot_patch = self._record_moves(offload_a)
        with snapshot_patch:
            sd = offload_a.state_dict()

        # Act — load into a fresh optimizer
        model_b, opt_b = self._make_optimizer("cpu")
        offload_b = CPUOffloadOptimizer(opt_b, pin_memory=False)
        offload_b.load_state_dict(sd)

        # Assert — initialized without stepping; states live on CPU
        assert offload_b._initialized is True
        for state in offload_b.state.values():
            for v in state.values():
                assert v.device.type == "cpu"

        # Act — the next step round-trips instead of re-running init
        devices, move_patch = self._record_moves(offload_b)
        with move_patch:
            self._step_once(model_b, offload_b, "cpu")

        # Assert
        # Adam's scalar ``step`` stays on CPU; only moment tensors move.
        n_states = sum(
            1 for state in opt_b.state.values() for key in state if key != "step"
        )
        assert devices == ["cuda"] * n_states + ["cpu"] * n_states

    @cuda_required
    def test_state_dict_returns_snapshot_and_round_trips_back_to_cpu(self):
        # Arrange
        model, opt = self._make_optimizer("cuda")
        offload = CPUOffloadOptimizer(opt, pin_memory=True)
        self._step_once(model, offload, "cuda")

        # Act
        sd = offload.state_dict()

        # Assert — snapshot captured; live states back on CPU after the call
        assert "state" in sd
        assert sd["state"], "state dict should contain optimizer states"
        for state in offload.state.values():
            for v in state.values():
                assert v.device.type == "cpu"

    def test_state_dict_before_init_returns_inner_state_without_move(self):
        # Arrange — no step yet; _initialized is False
        _model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)

        # Act
        sd = offload.state_dict()

        # Assert — returns inner state dict; state is empty (no step yet)
        assert offload._initialized is False
        assert sd["state"] == {}
        assert len(sd["param_groups"]) == len(opt.param_groups)
        assert sd["param_groups"][0]["lr"] == opt.param_groups[0]["lr"]

    @cuda_required
    def test_load_state_dict_marks_initialized_and_moves_to_cpu(self):
        # Arrange — capture a state dict from a stepped optimizer
        model_a, opt_a = self._make_optimizer("cuda")
        offload_a = CPUOffloadOptimizer(opt_a, pin_memory=True)
        self._step_once(model_a, offload_a, "cuda")
        sd = offload_a.state_dict()

        # Act — load into a fresh optimizer, then step (skips init branch)
        model_b, opt_b = self._make_optimizer("cuda")
        offload_b = CPUOffloadOptimizer(opt_b, pin_memory=True)
        offload_b.load_state_dict(sd)
        assert offload_b._initialized is True
        # states loaded then moved to CPU
        for state in offload_b.state.values():
            for v in state.values():
                assert v.device.type == "cpu"

        # subsequent step round-trips via CUDA (init branch skipped)
        self._step_once(model_b, offload_b, "cuda")
        for state in offload_b.state.values():
            for v in state.values():
                assert v.device.type == "cpu"

    def test_move_states_handles_dtensor_local_tensor(self, monkeypatch):
        # Arrange — point dmod.DTensor at our stub so isinstance() passes
        monkeypatch.setattr(dmod, "DTensor", _FakeDTensor)
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)

        # Act — run a real step so AdamW creates states, then inject a
        # FakeDTensor for one state entry and move to CPU
        self._step_once(model, offload, "cpu")
        p = next(iter(offload.state))
        original_local = offload.state[p]["exp_avg"].clone()
        original_dt = _FakeDTensor(original_local)
        offload.state[p]["exp_avg"] = original_dt

        offload._move_states("cpu")

        # Assert — new wrapper instance, local moved, original untouched
        new_dt = offload.state[p]["exp_avg"]
        assert isinstance(new_dt, _FakeDTensor)
        assert new_dt is not original_dt
        assert new_dt._local_tensor.device.type == "cpu"
        assert torch.equal(original_dt._local_tensor, original_local)

    def test_zero_grad_delegates_to_inner(self):
        # Arrange
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        x = torch.randn(2, 4)
        model(x).sum().backward()
        assert any(p.grad is not None for p in model.parameters())

        # Act
        offload.zero_grad()

        # Assert — inner optimizer grads cleared
        assert all(p.grad is None for p in model.parameters())

    @cuda_required
    def test_step_does_not_call_cuda_synchronize(self):
        model, opt = self._make_optimizer("cuda")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        self._step_once(model, offload, "cuda")

        with patch.object(torch.cuda, "synchronize") as sync:
            self._step_once(model, offload, "cuda")

        sync.assert_not_called()
        for state in offload.state.values():
            for v in state.values():
                assert isinstance(v, torch.Tensor)
                assert v.device.type == "cpu"

    def test_base_optimizer_property_returns_inner(self):
        # Arrange
        _, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)

        # Act / Assert
        assert offload.base_optimizer is opt

    def test_param_groups_property_reads_and_writes_inner(self):
        # Arrange
        _, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)

        # Act / Assert — read reflects inner groups
        assert offload.param_groups is opt.param_groups

        # Act / Assert — setter writes through to inner
        sentinel = [{"lr": 9e-1}]
        offload.param_groups = sentinel
        assert opt.param_groups is sentinel
