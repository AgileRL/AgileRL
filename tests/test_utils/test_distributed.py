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
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from agilerl.distributed import (
    CPUOffloadOptimizer,
    FSDPConfig,
    aggregate_metrics_across_gpus,
    aggregate_metrics_dict,
    all_ranks,
    all_reduce_mean,
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
    shard_dataloader_kwargs,
    sync_grads,
)
from agilerl.distributed import process as dmod
from agilerl.distributed.process import (
    get_full_model_state_dict,
    set_full_model_state_dict,
)
from agilerl.distributed.runtime import DPRuntime, FSDPRuntime

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for state round-trip"
)


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


class TestAllReduceMean:
    def test_passthrough_without_process_group(self):
        # Arrange
        tensor = torch.tensor([1.0, 2.0])

        # Act
        out = all_reduce_mean(tensor)

        # Assert
        assert torch.equal(out, torch.tensor([1.0, 2.0]))

    def test_identity_with_world_size_one(self, world_size_one):
        # Arrange
        tensor = torch.tensor([4.0])

        # Act
        out = all_reduce_mean(tensor)

        # Assert
        assert torch.equal(out, torch.tensor([4.0]))


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

    def test_with_negative_values(self):
        result = aggregate_metrics_across_gpus(torch.tensor([-1.0, -2.0, -3.0]))

        assert result == -2.0
        assert isinstance(result, float)

    def test_with_zero_values(self):
        result = aggregate_metrics_across_gpus(torch.tensor([0.0, 0.0, 0.0]))

        assert result == 0.0
        assert isinstance(result, float)

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

        with full_shape_views([linear.weight, linear.bias, None]):
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
        with (
            patch.object(dmod, "DTensor", FakeDTensor),
            patch.object(
                dmod,
                "parameter_owner",
                return_value=(linear, "weight"),
            ),
        ):
            linear._parameters["weight"] = fake_dtensor

            with full_shape_views([fake_dtensor, fake_dtensor, None]):
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
            patch.object(dmod, "parameter_owner", return_value=None),
            full_shape_views([fake_dtensor]),
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
        with (
            patch.object(dmod, "DTensor", FakeDTensor),
            patch.object(
                dmod,
                "parameter_owner",
                return_value=(linear, "weight"),
            ),
        ):
            linear._parameters["weight"] = fake_dtensor

            boom = RuntimeError("boom")
            with (
                pytest.raises(RuntimeError, match="boom"),
                full_shape_views([fake_dtensor]),
            ):
                raise boom

            assert linear._parameters["weight"] is fake_dtensor


class TestSetFullModelStateDict:
    def test_forwards_to_distributed_checkpoint(self):
        model = nn.Linear(2, 2)
        state = model.state_dict()
        applied: dict = {}

        def _fake_set_model_state_dict(module, state_dict, options=None):
            applied["module"] = module
            applied["state_dict"] = state_dict
            applied["options"] = options

        with patch(
            "agilerl.distributed.process.set_model_state_dict",
            _fake_set_model_state_dict,
        ):
            set_full_model_state_dict(model, state, strict=True)

        assert applied["module"] is model
        assert applied["state_dict"] is state
        assert applied["options"].full_state_dict is True
        assert applied["options"].strict is True


class TestGetFullModelStateDict:
    def test_forwards_to_distributed_checkpoint(self):
        model = nn.Linear(2, 2)
        expected = {"weight": torch.zeros(2, 2)}

        with patch(
            "agilerl.distributed.process.get_model_state_dict",
            return_value=expected,
        ):
            out = get_full_model_state_dict(model)

        assert out is expected

    def test_rejects_gpu_full_state(self):
        model = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="cpu_offload=False"):
            get_full_model_state_dict(model, cpu_offload=False)


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

    def test_fsdp2_uses_get_full_model_state_dict(self):
        model = nn.Linear(2, 2)
        expected = {"weight": torch.zeros(2, 2)}
        runtime = FSDPRuntime(FSDPConfig())

        with patch(
            "agilerl.distributed.runtime.get_full_model_state_dict",
            return_value=expected,
        ) as get_full:
            out = runtime.export_model_state(model, cpu_offload=True)

        get_full.assert_called_once_with(model, cpu_offload=True)
        assert out is expected


class TestDenseWrap:
    def test_moves_actor_to_device(self):
        actor = nn.Linear(2, 2)
        optimizer = MagicMock()
        result = DPRuntime().prepare_actor(
            actor,
            optimizer,
            None,
            device="cpu",
            use_vllm=False,
            cosine_lr_schedule_config=None,
            lr=1e-4,
            lr_critic=None,
            restore_adapter_trainability=lambda _: None,
        )
        assert result.actor is actor
        assert all(param.device.type == "cpu" for param in result.actor.parameters())
        assert result.optimizer is optimizer


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


class TestDeferGradSync:
    def test_fsdp_config_defaults_on(self):
        assert FSDPConfig().defer_grad_sync is True

    def test_toggles_sync_across_accumulation_window(self):
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=True))
        optimizer = MagicMock()
        loss = MagicMock()
        model = MagicMock()

        runtime.backward(loss, optimizer, 1, 2, actor=model)
        model.set_requires_gradient_sync.assert_called_once_with(False, recurse=True)
        runtime.backward(loss, optimizer, 1, 2, actor=model)
        model.set_requires_gradient_sync.assert_called_with(True, recurse=True)
        assert model.set_requires_gradient_sync.call_count == 2

        optimizer.step.assert_called_once()

    def test_disabled_does_not_toggle(self):
        runtime = FSDPRuntime(FSDPConfig(defer_grad_sync=False))
        optimizer = MagicMock()
        loss = MagicMock()
        model = MagicMock()

        runtime.backward(loss, optimizer, 1, 1, actor=model)

        model.set_requires_gradient_sync.assert_not_called()
        optimizer.step.assert_called_once()


class TestRaiseOnAnyRank:
    def test_runs_block_without_process_group(self):
        with dmod.raise_on_any_rank():
            x = 7

        assert x == 7

    def test_reraises_local_error_without_process_group(self):
        msg = "boom"
        with pytest.raises(ValueError, match=msg):
            with dmod.raise_on_any_rank():
                raise ValueError(msg)

    def test_runs_as_decorator_without_process_group(self):
        @dmod.raise_on_any_rank()
        def body() -> int:
            return 7

        assert body() == 7

    def test_decorator_reraises_local_error_without_process_group(self):
        msg = "boom"

        @dmod.raise_on_any_rank()
        def body() -> None:
            raise ValueError(msg)

        with pytest.raises(ValueError, match=msg):
            body()


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

    def test_pins_cuda_local_rank_when_distributed(self, world_size_one):
        # Act / Assert
        if torch.cuda.is_available():
            assert resolve_device("cpu") == "cuda:0"
        else:
            assert resolve_device("cpu") == "cpu"


class TestShardDataloaderKwargs:
    def test_returns_shuffle_flag_on_single_device(self):
        # Act
        kwargs = shard_dataloader_kwargs(dataset=[1, 2, 3], shuffle=False)

        # Assert
        assert kwargs == {"shuffle": False}

    def test_returns_sampler_matching_rank_topology(self, world_size_one):
        # Arrange
        dataset = list(range(8))

        # Act
        kwargs = shard_dataloader_kwargs(dataset=dataset, shuffle=False)
        sampler = kwargs["sampler"]

        # Assert
        assert sampler.num_replicas == 1
        assert sampler.rank == 0
        assert list(sampler) == list(range(8))


class TestApplyFsdp2:
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
            out = apply_fsdp2(model)

        # Assert — same object; each block and the root are sharded
        assert out is model
        assert set(sharded) == {blocks[0], blocks[1], model}

    def test_shards_embed_and_untied_lm_head_before_root(self, world_size_one):
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

            def __init__(self, *, tie: bool):
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
            apply_fsdp2(untied)

        # Assert — child units before root; lm_head alone (not joint with norm)
        assert untied.model.embed_tokens in sharded
        assert untied.lm_head in sharded
        assert untied.model.norm not in sharded
        assert sharded[-1] is untied
        assert set(sharded) == {
            untied.model.layers[0],
            untied.model.layers[1],
            untied.model.embed_tokens,
            untied.lm_head,
            untied,
        }
        assert shard_kwargs[id(untied.lm_head)]["reshard_after_forward"] is False
        assert (
            shard_kwargs[id(untied.model.embed_tokens)]["reshard_after_forward"] is True
        )
        assert shard_kwargs[id(untied)]["reshard_after_forward"] is True

        # Arrange — tied: embed + blocks + root; skip separate lm_head unit
        tied = CausalLM(tie=True)
        sharded.clear()
        with patch.object(dmod, "fully_shard", side_effect=_record):
            apply_fsdp2(tied)

        # Assert
        assert tied.model.embed_tokens in sharded
        assert tied.lm_head not in sharded
        assert sharded[-1] is tied

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
        prefetch_units: list[nn.Module] = []

        def _record(module, **_kwargs):
            sharded.append(module)
            return module

        def _capture(_model, block_units):
            prefetch_units.extend(block_units)

        with (
            patch.object(dmod, "fully_shard", side_effect=_record),
            patch.object(dmod, "_set_prefetch", side_effect=_capture),
        ):
            apply_fsdp2(model, gradient_checkpointing=True)

        assert isinstance(model.layers[0], CheckpointWrapper)
        assert isinstance(model.layers[1], CheckpointWrapper)
        assert next(iter(model.layers[0].children())) is raw_blocks[0]
        assert next(iter(model.layers[1].children())) is raw_blocks[1]
        assert raw_blocks[0] not in sharded
        assert raw_blocks[1] not in sharded
        assert sharded[0] is model.layers[0]
        assert sharded[1] is model.layers[1]
        assert prefetch_units == [model.layers[0], model.layers[1]]

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
            apply_fsdp2(model)

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
            apply_fsdp2(model)

        assert model.layers[0] in sharded
        assert model.layers[1] in sharded
        assert model in sharded
        assert model.layers[0].experts not in sharded
        assert model.layers[1].experts not in sharded

    def test_raises_without_process_group(self):
        with pytest.raises(RuntimeError, match="initialised process group"):
            apply_fsdp2(nn.Linear(2, 2), FSDPConfig())


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
