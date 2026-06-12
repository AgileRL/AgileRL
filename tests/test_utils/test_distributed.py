import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from agilerl.utils import distributed as dmod
from agilerl.utils.distributed import (
    FSDPConfig,
    all_reduce_mean,
    apply_fsdp2,
    barrier,
    broadcast_object_list,
    distributed_env_present,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    resolve_device,
    set_seed,
    shard_dataloader_kwargs,
    sync_grads,
)

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


@pytest.fixture
def world_size_one(_clean_dist_state):
    """A real single-process gloo process group."""
    if sys.platform == "win32":
        # Windows torch wheels ship gloo without the TCP transport used for
        # collectives ("makeDeviceForHostname(): unsupported gloo device").
        # These paths are exercised on the Linux/macOS CI runners.
        pytest.skip("torch gloo process groups are unsupported on Windows wheels")
    os.environ.update(
        {
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(29650 + os.getpid() % 200),
        }
    )
    assert init_distributed() is True
    yield
    dist.destroy_process_group()


class TestSingleDeviceNoOps:
    def test_topology_helpers_without_distributed(self):
        assert distributed_env_present() is False
        assert init_distributed() is False
        assert is_distributed() is False
        assert get_rank() == 0
        assert get_world_size() == 1
        assert is_main_process() is True
        barrier()  # no-op

    def test_local_rank_from_env(self):
        os.environ["LOCAL_RANK"] = "3"
        assert get_local_rank() == 3

    def test_local_rank_fallback_zero_without_env_or_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert get_local_rank() == 0

    def test_broadcast_object_list_passthrough(self):
        objects = [1, "two", {"three": 3}]
        assert broadcast_object_list(objects) is objects

    def test_all_reduce_mean_passthrough(self):
        t = torch.tensor([1.0, 2.0])
        assert torch.equal(all_reduce_mean(t), torch.tensor([1.0, 2.0]))

    def test_sync_grads_noop(self):
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.full((2,), 5.0)
        sync_grads([param])
        assert torch.equal(param.grad, torch.full((2,), 5.0))

    def test_set_seed_reproducible(self):
        set_seed(123)
        a = torch.rand(3)
        set_seed(123)
        assert torch.equal(a, torch.rand(3))

    def test_resolve_device_prefers_requested(self):
        assert resolve_device("cpu") == "cpu"

    def test_resolve_device_fallback_order(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert resolve_device(None) == "cpu"

    def test_resolve_device_prefers_mps_over_cpu(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert resolve_device(None) == "mps"

    def test_shard_dataloader_kwargs_single_device(self):
        assert shard_dataloader_kwargs(dataset=[1, 2, 3], shuffle=False) == {
            "shuffle": False
        }


class TestWorldSizeOne:
    def test_init_and_topology(self, world_size_one):
        assert is_distributed() is True
        assert get_rank() == 0
        assert get_world_size() == 1
        assert is_main_process() is True
        barrier()
        # idempotent re-init reuses the group
        assert init_distributed() is True

    def test_broadcast_and_reduce(self, world_size_one):
        assert broadcast_object_list(["x", 7]) == ["x", 7]
        t = torch.tensor([4.0])
        assert torch.equal(all_reduce_mean(t), torch.tensor([4.0]))

    def test_sync_grads_world_one(self, world_size_one):
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.full((2,), 3.0)
        no_grad_param = nn.Parameter(torch.ones(2))
        sync_grads([param, no_grad_param])
        assert torch.equal(param.grad, torch.full((2,), 3.0))

    def test_resolve_device_cpu_when_no_cuda(self, world_size_one):
        if torch.cuda.is_available():
            assert resolve_device("cpu") == "cuda:0"
        else:
            assert resolve_device("cpu") == "cpu"

    def test_shard_dataloader_kwargs_distributed(self, world_size_one):
        kwargs = shard_dataloader_kwargs(dataset=list(range(8)), shuffle=True)
        sampler = kwargs["sampler"]
        assert sampler.num_replicas == 1
        assert sampler.rank == 0
        assert len(list(sampler)) == 8


class TestApplyFsdp2:
    def test_requires_process_group(self):
        with pytest.raises(RuntimeError, match="initialised process group"):
            apply_fsdp2(nn.Linear(2, 2), FSDPConfig())

    def test_requires_fsdp_build(self):
        with (
            patch.object(dmod, "HAS_FSDP2", False),
            pytest.raises(RuntimeError, match="distributed support"),
        ):
            apply_fsdp2(nn.Linear(2, 2))

    def test_wraps_blocks_then_root(self, world_size_one):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

        class Model(nn.Module):
            _no_split_modules = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block(), Block()])

        model = Model()
        calls = []
        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **kw: calls.append((m, kw))
        ):
            out = apply_fsdp2(
                model,
                FSDPConfig(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            )
        assert out is model
        # two blocks first, root last
        assert len(calls) == 3
        assert calls[-1][0] is model
        assert all(isinstance(c[0], Block) for c in calls[:-1])
        assert "mp_policy" in calls[0][1]

    def test_cpu_offload_policy_forwarded(self, world_size_one):
        calls = []
        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **kw: calls.append(kw)
        ):
            apply_fsdp2(nn.Linear(2, 2), FSDPConfig(cpu_offload=True))
        assert "offload_policy" in calls[-1]

    def test_root_only_without_no_split_metadata(self, world_size_one):
        calls = []
        with patch.object(
            dmod, "fully_shard", side_effect=lambda m, **kw: calls.append(m)
        ):
            apply_fsdp2(nn.Linear(2, 2))
        assert len(calls) == 1


class TestShardedSeamBranches:
    """Cover the FSDP2-sharded branches of the llm_utils seam with fakes."""

    def test_gather_full_params_unshards_and_reshards_in_order(self):
        from agilerl.utils import llm_utils

        class FakeFSDPModule(nn.Module):
            def __init__(self, log, name):
                super().__init__()
                self.log = log
                self.name = name

            def unshard(self):
                self.log.append(("unshard", self.name))

            def reshard(self):
                self.log.append(("reshard", self.name))

        log = []
        root = FakeFSDPModule(log, "root")
        root.child = FakeFSDPModule(log, "child")
        with patch.object(llm_utils, "FSDPModule", FakeFSDPModule):
            with llm_utils.gather_full_params(root):
                assert log == [("unshard", "root"), ("unshard", "child")]
        assert log[2:] == [("reshard", "child"), ("reshard", "root")]

    def test_gather_full_params_rejects_fsdp1(self):
        from agilerl.utils import llm_utils

        class FakeFSDP1(nn.Module):
            pass

        model = FakeFSDP1()
        with (
            patch.object(llm_utils, "FullyShardedDataParallel", FakeFSDP1),
            pytest.raises(NotImplementedError, match="only supports FSDP2"),
        ):
            with llm_utils.gather_full_params(model):
                pass

    def test_load_full_state_dict_sharded_uses_dcp(self):
        from agilerl.utils import llm_utils

        model = nn.Linear(2, 2)
        sd = model.state_dict()
        set_mock = MagicMock()
        with (
            patch.object(llm_utils, "is_fsdp_sharded", return_value=True),
            patch(
                "torch.distributed.checkpoint.state_dict.set_model_state_dict",
                set_mock,
            ),
        ):
            llm_utils.load_full_state_dict(model, sd, strict=True)
        set_mock.assert_called_once()

    def test_get_state_dict_sharded_uses_dcp(self):
        from agilerl.utils import llm_utils

        model = nn.Linear(2, 2)
        get_mock = MagicMock(return_value={"weight": torch.zeros(2, 2)})
        with (
            patch.object(llm_utils, "is_fsdp_sharded", return_value=True),
            patch(
                "torch.distributed.checkpoint.state_dict.get_model_state_dict",
                get_mock,
            ),
        ):
            out = llm_utils.get_state_dict(model)
        get_mock.assert_called_once()
        assert "weight" in out
