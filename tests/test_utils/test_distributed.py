# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for ``agilerl.distributed``.

Classes are named ``Test<FunctionName>`` and assert API-visible outcomes
(return values, side effects, raised errors). ``fully_shard`` is stubbed
only as an expensive boundary; assertions still target which modules were
sharded and which config policies were requested, not call counts.
"""

from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    GptOssConfig,
    GraniteMoeConfig,
    MellumConfig,
    MiniMaxConfig,
    MixtralConfig,
    OlmoeConfig,
    Qwen2MoeConfig,
    Qwen3MoeConfig,
    Qwen4ExpTextConfig,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4TextRotaryEmbedding,
)
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.distributed import (
    CPUOffloadOptimizer,
    FSDPConfig,
    ParallelMesh,
    aggregate_metrics_across_gpus,
    aggregate_metrics_dict,
    all_ranks,
    allreduce_max_ints,
    allreduce_minmax_int,
    allreduce_sum_ints,
    any_rank,
    apply_fsdp2,
    barrier,
    broadcast_object_list,
    distributed_env_present,
    ep_data_parallel_size,
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
    materialize_fsdp2_from_cpu_state,
    packed_expert_counts,
    reference_dispatch_combine,
    resolve_device,
    set_seed,
    sync_grads,
    validate_actor_ep,
    validate_ep_degree,
)
from agilerl.distributed import fsdp as dmod
from agilerl.distributed import fsdp_blocks as bmod
from agilerl.distributed import process as pmod
from agilerl.distributed import runtime as rmod
from agilerl.distributed.fsdp import (
    canonical_fsdp_param_fqn,
    reshard_fsdp_modules,
    set_full_model_state_dict,
)
from agilerl.distributed.fsdp_meta import (
    init_embed_scale_buffers,
    init_rope_buffers,
)
from agilerl.distributed.runtime import (
    DPRuntime,
    FSDPRuntime,
    OptimizerStep,
    PrepareResult,
    clip_param_group_grad_norm_,
    clip_param_groups,
)
from agilerl.utils.algo_utils import CosineLRScheduleConfig
from agilerl.utils.llm_utils import save_lora_adapters
from agilerl.utils.ppo_value_head import AutoModelForCausalLMWithValueHead
from tests.helper_functions import restore_moe_routers
from tests.rank_launch import run_ranks
from tests.test_algorithms.test_llms.llm_helpers import (
    DummyConfig,
    DummyMLPPreTrainedModel,
)

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


ACTOR_CRITIC_CLIP_GROUPS = (frozenset({"actor"}), frozenset({"critic"}))


class TestClipParamGroups:
    @staticmethod
    def two_groups(first_grad: torch.Tensor, second_grad: torch.Tensor):
        first = nn.Parameter(torch.zeros_like(first_grad))
        first.grad = first_grad.clone()
        second = nn.Parameter(torch.zeros_like(second_grad))
        second.grad = second_grad.clone()
        groups = [
            {"params": [first], "group": "actor"},
            {"params": [second], "group": "critic"},
        ]
        return groups, first, second

    def test_one_coefficient_matches_torch_over_all_grads(self):
        # Arrange
        torch.manual_seed(0)
        first_grad = torch.randn(4) * 10
        second_grad = torch.randn(3)
        groups, first, second = self.two_groups(first_grad, second_grad)
        reference = nn.Parameter(torch.zeros(7))
        reference.grad = torch.cat([first_grad, second_grad])
        reference_norm = clip_grad_norm_([reference], max_norm=1.0)

        # Act
        norms, clip_coefs = clip_param_groups(groups, 1.0, clip_param_group_grad_norm_)

        # Assert
        assert norms == pytest.approx(
            (
                torch.linalg.vector_norm(first_grad).item(),
                torch.linalg.vector_norm(second_grad).item(),
            ),
            rel=1e-6,
        )
        assert math.hypot(*norms) == pytest.approx(reference_norm.item(), rel=1e-6)
        assert clip_coefs[0] == clip_coefs[1]
        assert math.hypot(*norms) * clip_coefs[0] == pytest.approx(1.0, rel=1e-5)
        # rtol covers the float64 vs float32 clip coefficient
        assert torch.allclose(
            torch.cat([first.grad, second.grad]), reference.grad, rtol=1e-6, atol=0
        )

    def test_below_threshold_leaves_grads_unchanged(self):
        # Arrange
        groups, first, second = self.two_groups(torch.full((2,), 0.1), torch.zeros(2))

        # Act
        norms, clip_coefs = clip_param_groups(groups, 1.0, clip_param_group_grad_norm_)

        # Assert
        assert norms == pytest.approx((0.02**0.5, 0.0))
        assert clip_coefs == (1.0, 1.0)
        assert torch.equal(first.grad, torch.full((2,), 0.1))
        assert torch.equal(second.grad, torch.zeros(2))

    def test_none_max_norm_only_measures(self):
        # Arrange
        groups, first, second = self.two_groups(torch.full((2,), 10.0), torch.ones(2))

        # Act
        norms, clip_coefs = clip_param_groups(
            groups, None, clip_param_group_grad_norm_, ACTOR_CRITIC_CLIP_GROUPS
        )

        # Assert
        assert norms == pytest.approx((200**0.5, 2**0.5))
        assert clip_coefs == (1.0, 1.0)
        assert torch.equal(first.grad, torch.full((2,), 10.0))
        assert torch.equal(second.grad, torch.ones(2))

    def test_nan_total_propagates_like_torch(self):
        # Arrange
        groups, first, second = self.two_groups(
            torch.tensor([float("nan"), 1.0]), torch.ones(2)
        )

        # Act
        norms, clip_coefs = clip_param_groups(groups, 1.0, clip_param_group_grad_norm_)

        # Assert
        assert np.isnan(norms[0])
        assert np.isnan(clip_coefs).all()
        assert torch.isnan(first.grad).all()
        assert torch.isnan(second.grad).all()

    def test_clip_groups_leave_a_set_below_threshold_unchanged(self):
        # Arrange: actor norm 0.5, critic norm 50.
        groups, actor, critic = self.two_groups(
            torch.tensor([0.3, 0.4]), torch.tensor([30.0, 40.0])
        )

        # Act
        norms, clip_coefs = clip_param_groups(
            groups, 1.0, clip_param_group_grad_norm_, ACTOR_CRITIC_CLIP_GROUPS
        )

        # Assert
        assert norms == pytest.approx((0.5, 50.0), rel=1e-6)
        assert clip_coefs == pytest.approx((1.0, 1.0 / 50.0), rel=1e-6)
        assert torch.equal(actor.grad, torch.tensor([0.3, 0.4]))
        # rtol covers the 1e-6 epsilon in the clip coefficient
        assert torch.allclose(critic.grad, torch.tensor([0.6, 0.8]), rtol=1e-6, atol=0)

    def test_clip_groups_scale_each_set_above_threshold_to_max_norm(self):
        # Arrange: actor norm 5, critic norm 50.
        groups, actor, critic = self.two_groups(
            torch.tensor([3.0, 4.0]), torch.tensor([30.0, 40.0])
        )

        # Act
        _, clip_coefs = clip_param_groups(
            groups, 1.0, clip_param_group_grad_norm_, ACTOR_CRITIC_CLIP_GROUPS
        )

        # Assert
        assert clip_coefs == pytest.approx((1.0 / 5.0, 1.0 / 50.0), rel=1e-6)
        # rtol covers the 1e-6 epsilon in the clip coefficient
        assert torch.allclose(actor.grad, torch.tensor([0.6, 0.8]), rtol=1e-6, atol=0)
        assert torch.allclose(critic.grad, torch.tensor([0.6, 0.8]), rtol=1e-6, atol=0)

    def test_default_scales_every_group_by_the_global_norm(self):
        # Arrange: actor norm 0.5, critic norm 50.
        groups, actor, critic = self.two_groups(
            torch.tensor([0.3, 0.4]), torch.tensor([30.0, 40.0])
        )
        coef = 1.0 / math.hypot(0.5, 50.0)

        # Act
        _, clip_coefs = clip_param_groups(groups, 1.0, clip_param_group_grad_norm_)

        # Assert
        assert clip_coefs == pytest.approx((coef, coef), rel=1e-6)
        # rtol covers the 1e-6 epsilon in the clip coefficient
        assert torch.allclose(
            actor.grad, torch.tensor([0.3, 0.4]) * coef, rtol=1e-6, atol=0
        )
        assert torch.allclose(
            critic.grad, torch.tensor([30.0, 40.0]) * coef, rtol=1e-6, atol=0
        )

    @pytest.mark.parametrize(
        "clip_groups",
        [
            (frozenset({"actor"}),),
            (frozenset({"actor", "critic"}), frozenset({"critic"})),
        ],
        ids=["missing", "repeated"],
    )
    def test_clip_groups_must_hold_each_group_once(self, clip_groups):
        # Arrange
        groups, actor, critic = self.two_groups(torch.ones(2), torch.ones(2))

        # Act / Assert
        with pytest.raises(ValueError, match="exactly once"):
            clip_param_groups(groups, 1.0, clip_param_group_grad_norm_, clip_groups)
        assert torch.equal(actor.grad, torch.ones(2))
        assert torch.equal(critic.grad, torch.ones(2))


class TestOptimizerStepGradNormProperties:
    def test_totals_span_every_group_with_a_shared_clip_coefficient(self):
        # Arrange
        step = OptimizerStep(group_grad_norms=(3.0, 4.0), clip_coefs=(0.2, 0.2))

        # Act
        pre, post = step.grad_norm_pre, step.grad_norm_post

        # Assert
        assert pre == pytest.approx(5.0)
        assert post == pytest.approx(1.0)

    def test_post_applies_each_group_coefficient(self):
        # Arrange
        step = OptimizerStep(group_grad_norms=(0.5, 50.0), clip_coefs=(1.0, 0.02))

        # Act
        pre, post = step.grad_norm_pre, step.grad_norm_post

        # Assert
        assert pre == pytest.approx(math.hypot(0.5, 50.0))
        assert post == pytest.approx(math.hypot(0.5, 1.0))

    def test_nan_group_norm_makes_both_totals_nan(self):
        # Arrange
        step = OptimizerStep(
            group_grad_norms=(float("nan"), 1.0),
            clip_coefs=(float("nan"), float("nan")),
        )

        # Act
        pre, post = step.grad_norm_pre, step.grad_norm_post

        # Assert
        assert np.isnan(pre)
        assert np.isnan(post)


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

    def test_reads_launcher_rank_before_init(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("RANK", "2")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")

        # Act / Assert
        assert is_distributed() is False
        assert get_rank() == 2
        assert is_main_process() is False


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

    def test_reads_launcher_world_size_before_init(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")

        assert is_distributed() is False
        assert get_world_size() == 4


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


class TestAllreduceSumInts:
    def test_identity_without_process_group(self):
        assert allreduce_sum_ints([3, 0, 2]) == [3, 0, 2]

    def test_sums_each_value_across_ranks(self):
        # Arrange — a peer holds [4, 1]
        def fake_sum(t: torch.Tensor, op: object) -> None:
            assert op == dist.ReduceOp.SUM
            t.add_(torch.tensor([4, 1]))

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
            patch("agilerl.distributed.process.dist.all_reduce", side_effect=fake_sum),
        ):
            sums = allreduce_sum_ints([2, 3])

        # Assert
        assert sums == [6, 4]


class TestAllreduceMaxInts:
    def test_identity_without_process_group(self):
        assert allreduce_max_ints([3, 0, 2]) == [3, 0, 2]

    def test_takes_the_max_of_each_value_across_ranks(self):
        # Arrange — a peer holds [4, 1]
        def fake_max(t: torch.Tensor, op: object) -> None:
            assert op == dist.ReduceOp.MAX
            t.copy_(torch.maximum(t, torch.tensor([4, 1])))

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
            patch("agilerl.distributed.process.dist.all_reduce", side_effect=fake_max),
        ):
            maxima = allreduce_max_ints([2, 3])

        # Assert
        assert maxima == [4, 3]


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


class TestAggregateMetricsDict:
    def test_single_process_returns_local_means(self):
        out = aggregate_metrics_dict(
            {"a": torch.tensor([1.0, 3.0]), "b": 4.0, "c": np.array([2.0, 6.0])},
        )

        assert out == {"a": 2.0, "b": 4.0, "c": 4.0}
        assert all(isinstance(value, float) for value in out.values())

    def test_averages_every_metric_in_one_all_reduce(self):
        # Arrange: the peer rank reports loss 3, kl 0 and clip 1.
        reduced: list[torch.Tensor] = []

        def sum_with_peer(tensor, op):
            assert op == dist.ReduceOp.SUM
            reduced.append(tensor.clone())
            tensor.add_(torch.tensor([[3.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
            patch(
                "agilerl.distributed.process.dist.all_reduce",
                side_effect=sum_with_peer,
            ),
        ):
            out = aggregate_metrics_dict(
                {"loss": torch.tensor([1.0, 3.0]), "kl": float("nan"), "clip": 0.5},
            )

        # Assert
        assert len(reduced) == 1
        assert torch.equal(reduced[0][:, [0, 2]], torch.tensor([[2.0, 0.5], [1, 1]]))
        assert list(out) == ["loss", "kl", "clip"]
        assert out["loss"] == 2.5
        assert np.isnan(out["kl"])
        assert out["clip"] == 0.75

    def test_an_excluded_metric_averages_over_the_other_ranks_only(self):
        # Arrange: the peer rank counts both metrics, with loss 3 and kl 0.4.
        def sum_with_peer(tensor, op):
            tensor.add_(torch.tensor([[3.0, 0.4], [1.0, 1.0]]))

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
            patch(
                "agilerl.distributed.process.dist.all_reduce",
                side_effect=sum_with_peer,
            ),
        ):
            out = aggregate_metrics_dict(
                {"loss": 1.0, "kl": float("nan")}, excluded=("kl",)
            )

        # Assert
        assert out["loss"] == 2.0
        assert out["kl"] == pytest.approx(0.4)

    def test_a_metric_every_rank_excludes_is_nan(self):
        out = aggregate_metrics_dict({"loss": 1.0, "kl": 0.2}, excluded=("kl",))

        assert out["loss"] == 1.0
        assert np.isnan(out["kl"])

    def test_empty_dict_returns_empty(self):
        assert aggregate_metrics_dict({}) == {}

    def test_single_process_maximized_metric_is_its_local_mean(self):
        out = aggregate_metrics_dict(
            {"loss": 1.0, "peak_gib": torch.tensor([2.0, 4.0])},
            maximized=("peak_gib",),
        )

        assert out == {"loss": 1.0, "peak_gib": 3.0}

    def test_maximized_metrics_take_the_largest_rank_value_in_the_same_all_reduce(
        self,
    ):
        # Arrange: this is rank 0 of 2; rank 1 reports loss 3, peak 70, retries 0.
        reduced: list[torch.Tensor] = []

        def sum_with_peer(tensor, op):
            assert op == dist.ReduceOp.SUM
            reduced.append(tensor.clone())
            peer = torch.zeros_like(tensor)
            peer[0] = torch.tensor([3.0, 70.0, 0.0])
            peer[1] = torch.tensor([1.0, 1.0, 1.0])
            peer[3] = torch.tensor([0.0, 70.0, 0.0])
            tensor.add_(peer)

        # Act
        with (
            patch("agilerl.distributed.process.is_distributed", return_value=True),
            patch("agilerl.distributed.process.dist.get_world_size", return_value=2),
            patch("agilerl.distributed.process.dist.get_rank", return_value=0),
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
            patch(
                "agilerl.distributed.process.dist.all_reduce",
                side_effect=sum_with_peer,
            ),
        ):
            out = aggregate_metrics_dict(
                {"loss": 1.0, "peak_gib": 75.5, "retries": -1.0},
                maximized=("peak_gib", "retries"),
            )

        # Assert
        assert len(reduced) == 1
        assert reduced[0].shape == (4, 3)
        assert out == {"loss": 2.0, "peak_gib": 75.5, "retries": 0.0}


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

    @pytest.mark.multi_rank
    def test_averages_grads_and_skips_params_with_no_grad_on_every_rank(self):
        run_ranks(_sync_grads_skips_unused_worker)

    @pytest.mark.multi_rank
    def test_leaves_every_grad_none_when_no_rank_has_one(self):
        run_ranks(_sync_grads_all_unused_worker)

    @pytest.mark.multi_rank
    def test_raises_on_every_rank_when_ranks_disagree_on_a_grad(self):
        run_ranks(_sync_grads_mismatch_worker)


def _sync_grads_skips_unused_worker(rank: int, world_size: int) -> None:
    # Arrange — ranks hold grads 1 and 3
    first = nn.Parameter(torch.zeros(2))
    first.grad = torch.full((2,), 2.0 * rank + 1)
    unused = nn.Parameter(torch.zeros(4))
    second = nn.Parameter(torch.zeros(3))
    second.grad = torch.full((3,), 2.0 * rank + 1)
    optimizer = torch.optim.AdamW([first, unused, second], lr=0.1)

    # Act
    sync_grads([first, unused, second])
    optimizer.step()

    # Assert
    assert torch.equal(first.grad, torch.full((2,), 2.0))
    assert torch.equal(second.grad, torch.full((3,), 2.0))
    assert unused.grad is None
    assert torch.equal(unused.detach(), torch.zeros(4))
    assert unused not in optimizer.state


def _sync_grads_all_unused_worker(rank: int, world_size: int) -> None:
    # Arrange
    params = [nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(3))]

    # Act
    sync_grads(params)

    # Assert
    assert all(param.grad is None for param in params)


def _sync_grads_mismatch_worker(rank: int, world_size: int) -> None:
    # Arrange — only rank 0 has a grad for ``partial``
    shared = nn.Parameter(torch.zeros(2))
    shared.grad = torch.full((2,), float(rank + 1))
    partial = nn.Parameter(torch.zeros(2))
    if rank == 0:
        partial.grad = torch.ones(2)

    # Act
    with pytest.raises(
        RuntimeError, match="1 params have a grad on some ranks but not others"
    ):
        sync_grads([shared, partial])

    # Assert
    assert torch.equal(shared.grad, torch.full((2,), float(rank + 1)))


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


class TestReplaceChild:
    def test_raises_when_old_not_found(self):
        model = nn.Sequential(nn.Linear(2, 2))

        with pytest.raises(RuntimeError, match="Could not find parent"):
            bmod._replace_child(model, nn.Linear(2, 2), nn.Linear(2, 2))


class TestResolveCausalLm:
    def test_unwraps_pretrained_model_shell(self):
        inner = nn.Linear(2, 2)
        shell = nn.Linear(2, 2)
        shell.pretrained_model = inner

        assert bmod.resolve_causal_lm(shell) is inner

    def test_unwraps_base_model_shell(self):
        inner = nn.Linear(2, 2)
        shell = nn.Linear(2, 2)
        shell.base_model = inner

        assert bmod.resolve_causal_lm(shell) is inner

    def test_returns_inner_model_with_lm_head(self):
        head = nn.Linear(2, 2)
        inner = nn.Linear(2, 2)
        inner.lm_head = head
        shell = nn.Linear(2, 2)
        shell.model = inner

        assert bmod.resolve_causal_lm(shell) is inner

    def test_returns_language_tower_with_lm_head(self):
        head = nn.Linear(2, 2)
        tower = nn.Linear(2, 2)
        tower.lm_head = head
        shell = nn.Linear(2, 2)
        shell.language_model = tower

        assert bmod.resolve_causal_lm(shell) is tower


class TestLanguageModel:
    def test_returns_backbone_under_language_model_tower(self):
        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2)])

        class Tower(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = Backbone()

        class Omni(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = Tower()

        omni = Omni()
        assert bmod._language_model(omni) is omni.language_model.backbone
        assert bmod.resolve_causal_lm(omni) is omni.language_model


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

        set_full.assert_called_once_with(model, state, strict=True, skip=frozenset())


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

    def test_later_steps_do_not_change_exported_state(self):
        # Arrange
        torch.manual_seed(0)
        model = nn.Linear(2, 2)
        inner = torch.optim.Adam(model.parameters(), lr=1e-2)
        optimizer = MagicMock()
        optimizer.state_dict.side_effect = inner.state_dict
        model(torch.randn(4, 2)).sum().backward()
        inner.step()

        # Act
        out = DPRuntime().export_optimizer_state(model, optimizer)
        exported = copy.deepcopy(out)
        model(torch.randn(4, 2)).sum().backward()
        inner.step()

        # Assert
        for param_id, param_state in exported["state"].items():
            assert torch.equal(out["state"][param_id]["step"], param_state["step"])
            assert torch.equal(
                out["state"][param_id]["exp_avg"], param_state["exp_avg"]
            )
        assert int(inner.state[model.weight]["step"]) == 2
        assert int(out["state"][0]["step"]) == 1


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

        assert out == expected
        _, kwargs = mock_get.call_args
        assert mock_get.call_args.args == (actor, inner)
        assert kwargs["options"].full_state_dict is True
        assert kwargs["options"].cpu_offload is True

    def test_export_does_not_alias_live_step_tensor(self):
        # Arrange
        torch.manual_seed(0)
        model = nn.Linear(2, 2)
        inner = torch.optim.Adam(model.parameters(), lr=1e-2)
        model(torch.randn(4, 2)).sum().backward()
        inner.step()
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        live_step = inner.state[model.weight]["step"]
        gathered = {"state": {"weight": {"step": live_step}}, "param_groups": []}

        # Act
        with (
            patch("agilerl.distributed.runtime.FSDPModule", object),
            patch(
                "agilerl.distributed.runtime.get_optimizer_state_dict",
                return_value=gathered,
            ),
        ):
            out = FSDPRuntime(FSDPConfig()).export_optimizer_state(model, optimizer)
        model(torch.randn(4, 2)).sum().backward()
        inner.step()

        # Assert
        assert int(inner.state[model.weight]["step"]) == 2
        assert int(out["state"]["weight"]["step"]) == 1

    def test_export_keeps_tensors_that_do_not_alias_live_state(self):
        # Arrange
        model = nn.Linear(2, 2)
        inner = torch.optim.Adam(model.parameters(), lr=1e-2)
        model(torch.randn(4, 2)).sum().backward()
        inner.step()
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        host_exp_avg = inner.state[model.weight]["exp_avg"].clone()
        gathered = {"state": {"weight": {"exp_avg": host_exp_avg}}, "param_groups": []}

        # Act
        with (
            patch("agilerl.distributed.runtime.FSDPModule", object),
            patch(
                "agilerl.distributed.runtime.get_optimizer_state_dict",
                return_value=gathered,
            ),
        ):
            out = FSDPRuntime(FSDPConfig()).export_optimizer_state(model, optimizer)

        # Assert
        assert out["state"]["weight"]["exp_avg"] is host_exp_avg

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


class TestFSDPRuntimeImportAdapterTensors:
    @pytest.mark.parametrize("use_value_head", [False, True])
    def test_loads_saved_actor_and_critic_adapters(self, tmp_path, use_value_head):
        # Arrange
        torch.manual_seed(0)
        lora_config = LoraConfig(
            r=2, lora_alpha=4, target_modules=["linear_1"], lora_dropout=0.0
        )
        peft_model = get_peft_model(
            DummyMLPPreTrainedModel(DummyConfig(vocab_size=8, hidden_size=32)),
            lora_config,
            adapter_name="actor",
        )
        peft_model.add_adapter("critic", lora_config)
        actor = (
            AutoModelForCausalLMWithValueHead(peft_model)
            if use_value_head
            else peft_model
        )
        lora_params = {
            name: param for name, param in actor.named_parameters() if "lora_" in name
        }
        with torch.no_grad():
            for param in lora_params.values():
                param.normal_()
        saved = {name: param.detach().clone() for name, param in lora_params.items()}
        save_lora_adapters(
            actor, tmp_path, ["actor", "critic"], use_value_head=use_value_head
        )
        with torch.no_grad():
            for param in lora_params.values():
                param.zero_()

        # Act
        for adapter in ("actor", "critic"):
            FSDPRuntime(FSDPConfig()).import_adapter_tensors(
                actor, peft_model, str(tmp_path), adapter
            )

        # Assert
        assert {".actor.", ".critic."} == {
            role for role in (".actor.", ".critic.") for name in saved if role in name
        }
        for name, param in lora_params.items():
            assert torch.equal(param, saved[name]), name


class TestFSDPPrepareActorOffload:
    @pytest.mark.parametrize(
        ("device", "cpu_offload", "optim_cpu_offload", "fused"),
        [
            ("cuda:0", False, False, True),
            ("cuda:0", False, True, False),
            ("cuda:0", True, False, False),
            ("cpu", False, False, False),
        ],
        ids=["gpu", "optim-cpu-offload", "cpu-offload", "cpu"],
    )
    def test_only_a_gpu_optimizer_steps_fused(
        self, device, cpu_offload, optim_cpu_offload, fused
    ):
        # Arrange
        actor = _LoraActor()
        config = FSDPConfig(
            reshard_after_forward=True,
            cpu_offload=cpu_offload,
            optim_cpu_offload=optim_cpu_offload,
        )

        # Act
        with patch(
            "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
            return_value=actor,
        ):
            result = FSDPRuntime(config).prepare_actor(
                actor,
                device=device,
                colocated=device != "cpu",
                cosine_lr_schedule_config=None,
                lr=1e-4,
                lr_critic=None,
                restore_adapter_trainability=MagicMock(),
            )

        # Assert
        optimizer = result.optimizer.optimizer
        if optim_cpu_offload:
            assert isinstance(optimizer, CPUOffloadOptimizer)
            optimizer = optimizer.optimizer
        assert bool(optimizer.defaults["fused"]) is fused

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
                FSDPRuntime(
                    FSDPConfig(reshard_after_forward=True, optim_cpu_offload=True)
                ).prepare_actor(
                    actor,
                    device="cpu",
                    colocated=False,
                    cosine_lr_schedule_config=None,
                    lr=1e-4,
                    lr_critic=None,
                    restore_adapter_trainability=MagicMock(),
                )

    @pytest.mark.parametrize(
        ("reshard", "optim_cpu_offload", "unset"),
        [
            (True, None, "optim_cpu_offload"),
            (None, False, "reshard_after_forward"),
        ],
    )
    def test_rejects_an_unresolved_memory_field(
        self, reshard, optim_cpu_offload, unset
    ):
        with pytest.raises(ValueError, match=rf"FSDPConfig\.{unset} is unset"):
            FSDPRuntime(
                FSDPConfig(
                    reshard_after_forward=reshard,
                    optim_cpu_offload=optim_cpu_offload,
                )
            ).prepare_actor(
                nn.Linear(2, 2),
                device="cpu",
                colocated=False,
                cosine_lr_schedule_config=None,
                lr=1e-4,
                lr_critic=None,
                restore_adapter_trainability=MagicMock(),
            )

    def test_rejects_mutually_exclusive_offload_flags(self):
        # Arrange
        runtime = FSDPRuntime(
            FSDPConfig(
                reshard_after_forward=True, cpu_offload=True, optim_cpu_offload=True
            )
        )

        # Act / Assert
        with pytest.raises(ValueError, match="mutually exclusive"):
            runtime.prepare_actor(
                nn.Linear(2, 2),
                device="cpu",
                colocated=True,
                cosine_lr_schedule_config=None,
                lr=1e-4,
                lr_critic=None,
                restore_adapter_trainability=MagicMock(),
            )

    def test_rejects_cpu_offload_without_colocated_vllm(self):
        # Arrange
        runtime = FSDPRuntime(
            FSDPConfig(
                reshard_after_forward=True, cpu_offload=True, optim_cpu_offload=False
            )
        )

        # Act / Assert
        with pytest.raises(ValueError, match="requires colocated vLLM"):
            runtime.prepare_actor(
                nn.Linear(2, 2),
                device="cpu",
                colocated=False,
                cosine_lr_schedule_config=None,
                lr=1e-4,
                lr_critic=None,
                restore_adapter_trainability=MagicMock(),
            )


class _LoraActor(nn.Module):
    """Tiny module whose trainable param matches ``init_llm_optimizer`` LoRA names."""

    supports_gradient_checkpointing = True

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

    def test_cpu_optimizer_steps_unfused(self):
        result = _prepare_dp_actor(_LoraActor())

        assert not result.optimizer.optimizer.defaults["fused"]

    @pytest.mark.gpu
    def test_cuda_optimizer_steps_fused(self):
        result = _prepare_dp_actor(_LoraActor(), device="cuda")

        assert result.optimizer.optimizer.defaults["fused"] is True

    def test_enables_hf_gradient_checkpointing(self):
        actor = _LoraActor()

        _prepare_dp_actor(actor, gradient_checkpointing=True)

        assert actor.checkpointing_kwargs == {
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }

    def test_enables_checkpointing_on_value_head_pretrained_model(self):
        inner = _LoraActor()

        class ValueHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pretrained_model = inner
                self.v_head = nn.Linear(2, 1)

        _prepare_dp_actor(ValueHead(), gradient_checkpointing=True)

        assert inner.checkpointing_kwargs == {
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }

    def test_enables_checkpointing_through_a_peft_value_head(self):
        inner = _LoraActor()

        class ValueHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pretrained_model = inner

        class PeftShell(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = ValueHead()

            def get_base_model(self) -> nn.Module:
                return self.base

        _prepare_dp_actor(PeftShell(), gradient_checkpointing=True)

        assert inner.checkpointing_kwargs == {
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }

    def test_builds_cosine_scheduler_when_configured(self):
        actor = _LoraActor()
        config = CosineLRScheduleConfig(num_steps=10, warmup_proportion=0.5)

        result = _prepare_dp_actor(actor, cosine_lr_schedule_config=config, lr=1e-4)

        assert result.lr_scheduler is not None
        assert result.lr_scheduler.get_last_lr() == pytest.approx([2e-5])

    def test_rejects_actor_to_that_returns_non_module(self):
        actor = nn.Linear(2, 2)
        actor.to = MagicMock(return_value="not-a-module")

        with pytest.raises(TypeError, match=r"must return nn\.Module"):
            _prepare_dp_actor(actor, device="meta")

    def test_rejects_gradient_checkpointing_without_support(self):
        with pytest.raises(TypeError, match="does not support gradient checkpointing"):
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
    def test_steps_on_accumulation_boundary(self):
        # Arrange
        model = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(model.weight)
        inner = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        optimizer.step.side_effect = inner.step
        optimizer.zero_grad.side_effect = inner.zero_grad
        runtime = DPRuntime()
        x = torch.ones(1, 2)

        # Act
        first = runtime.backward(model(x).sum(), optimizer, 2, model)
        weight_mid_window = model.weight.detach().clone()
        second = runtime.backward(model(x).sum(), optimizer, 2, model)

        # Assert — two half-scaled grads of 1.0 sum to 1.0; SGD lr 0.1
        assert first is None
        assert torch.equal(weight_mid_window, torch.zeros(1, 2))
        assert torch.allclose(model.weight, torch.full((1, 2), -0.1))
        assert second == OptimizerStep(
            group_grad_norms=(pytest.approx(2**0.5),),
            clip_coefs=(1.0,),
        )
        assert model.weight.grad is None

    def test_open_phase_timer_records_backward_sync_and_optim_phases(self):
        # Arrange
        model = nn.Linear(2, 1, bias=False)
        inner = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        runtime = DPRuntime()
        x = torch.ones(1, 2)

        # Act
        runtime.phase_timer.start("cpu")
        runtime.backward(model(x).sum(), optimizer, 2, model)
        accumulate_only = set(runtime.phase_timer.stop())
        runtime.phase_timer.start("cpu")
        runtime.backward(model(x).sum(), optimizer, 2, model)
        stepped = runtime.phase_timer.stop()

        # Assert
        assert accumulate_only == {"backward"}
        assert set(stepped) == {"backward", "grad_sync", "optim"}
        assert all(seconds >= 0.0 for seconds in stepped.values())


class TestBaseRuntimeMicroBatchesUntilStep:
    def test_counts_down_to_each_optimizer_step(self):
        # Arrange
        model = nn.Linear(2, 1, bias=False)
        inner = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        runtime = DPRuntime()
        x = torch.ones(1, 2)

        # Act
        remaining = [runtime.micro_batches_until_step(3)]
        for _ in range(4):
            runtime.backward(model(x).sum(), optimizer, 3, model)
            remaining.append(runtime.micro_batches_until_step(3))

        # Assert
        assert remaining == [3, 2, 1, 3, 2]


class VisionBlock(nn.Module):
    """Frozen base projection plus an actor LoRA projection."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)
        self.actor_lora = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(x) + self.actor_lora(x))


class DecoderBlock(nn.Module):
    """Frozen base projection plus a trainable actor LoRA projection."""

    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(8, 8)
        self.actor_lora = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.base(x) + self.actor_lora(x))


class VisionLanguageActor(nn.Module):
    """Runs the vision blocks only when ``use_vision`` is set, like a VLM batch without images."""

    _no_split_modules: ClassVar[list[str]] = ["VisionBlock", "DecoderBlock"]

    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.ModuleList([VisionBlock() for _ in range(2)])
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(2)])

    def forward(self, x: torch.Tensor, use_vision: bool = False) -> torch.Tensor:
        if use_vision:
            for block in self.visual:
                x = block(x)
        for layer in self.layers:
            x = layer(x)
        return x


def _vision_language_actor(train_vision: bool = True) -> VisionLanguageActor:
    torch.manual_seed(0)
    actor = VisionLanguageActor()
    for name, param in actor.named_parameters():
        trainable = "actor_lora" in name and (
            train_vision or not name.startswith("visual.")
        )
        param.requires_grad_(trainable)
    return actor


def _prepare_fsdp_actor(
    actor: nn.Module,
    gradient_checkpointing: bool = False,
    param_dtype: str = "bfloat16",
    persistence_threshold: int = 0,
) -> tuple[FSDPRuntime, PrepareResult]:
    # A reduce dtype distinct from the param dtype makes post-backward upcast
    # deferred grads, which reads each param's unsharded copy.
    runtime = FSDPRuntime(
        FSDPConfig(
            reshard_after_forward=True,
            param_dtype=param_dtype,
            reduce_dtype="float32",
            routed_expert_chunk_mib=64,
            optim_cpu_offload=False,
            checkpoint_every_n_blocks=1,
        )
    )
    # FSDP's default mesh uses the process accelerator (MPS on macOS).
    world = init_device_mesh(
        "cpu", (1, dist.get_world_size()), mesh_dim_names=("replicate", "shard")
    )
    with (
        patch(
            "agilerl.distributed.runtime.build_parallel_mesh",
            return_value=ParallelMesh(world=world, hsdp=world["shard"]),
        ),
        patch(
            "agilerl.distributed.fsdp_blocks.PARAM_PERSISTENCE_THRESHOLD",
            persistence_threshold,
        ),
    ):
        result = runtime.prepare_actor(
            actor,
            device="cpu",
            colocated=False,
            cosine_lr_schedule_config=None,
            lr=1e-2,
            lr_critic=None,
            restore_adapter_trainability=lambda _: None,
            gradient_checkpointing=gradient_checkpointing,
        )
    return runtime, result


def _accumulated_step(
    deferred: bool,
    micro_batches: list[torch.Tensor],
    use_vision: list[bool],
    train_vision: bool = True,
) -> tuple[list[OptimizerStep | None], dict[str, torch.Tensor]]:
    """Backward each micro-batch of one window; full grads the optimizer stepped on.

    :param deferred: Reduce-scatter on the last micro-batch only; False
        reduce-scatters every micro-batch.
    """
    runtime, result = _prepare_fsdp_actor(_vision_language_actor(train_vision))
    names = {param: name for name, param in result.actor.named_parameters()}
    stepped: dict[str, torch.Tensor] = {}

    def capture_grads(optimizer, _args, _kwargs):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    stepped[names[param]] = full_grad(param).clone()

    result.optimizer._single_optimizer().register_step_pre_hook(capture_grads)
    sync_every_micro_batch = (
        nullcontext()
        if deferred
        else patch.object(result.actor, "set_requires_gradient_sync")
    )
    with sync_every_micro_batch:
        steps = [
            runtime.backward(
                result.actor(x, vision).sum(),
                result.optimizer,
                len(micro_batches),
                result.actor,
            )
            for x, vision in zip(micro_batches, use_vision, strict=True)
        ]
    return steps, stepped


def _deferred_and_synced_grads_worker(
    rank: int, world_size: int, use_vision: list[bool]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    torch.manual_seed(1 + rank)
    micro_batches = [torch.randn(4, 8) for _ in use_vision]
    _, deferred = _accumulated_step(True, micro_batches, use_vision)
    _, synced = _accumulated_step(False, micro_batches, use_vision)
    # NumPy arrays: a torch tensor sent to the parent shares memory that dies
    # with this process before the parent reads it.
    return (
        {name: grad.numpy() for name, grad in deferred.items()},
        {name: grad.numpy() for name, grad in synced.items()},
    )


DECODER_LORA = {"layers.0.actor_lora.weight", "layers.1.actor_lora.weight"}
VISION_LORA = {"visual.0.actor_lora.weight", "visual.1.actor_lora.weight"}


class TestUnforwardedFSDPUnits:
    def test_tracks_every_unit_after_wrap(self, world_size_one):
        # Arrange
        actor = _vision_language_actor()

        # Act
        runtime, _ = _prepare_fsdp_actor(actor)

        # Assert
        assert list(runtime.unforwarded_units) == [
            actor,
            *actor.visual,
            *actor.layers,
        ]

    def test_drops_units_that_ran_forward(self, world_size_one):
        # Arrange
        actor = _vision_language_actor()
        runtime, result = _prepare_fsdp_actor(actor)

        # Act
        result.actor(torch.randn(4, 8))

        # Assert
        assert list(runtime.unforwarded_units) == list(actor.visual)

    def test_empty_after_every_unit_ran_forward(self, world_size_one):
        # Arrange
        runtime, result = _prepare_fsdp_actor(_vision_language_actor())

        # Act
        result.actor(torch.randn(4, 8), True)
        result.actor(torch.randn(4, 8), True)

        # Assert
        assert list(runtime.unforwarded_units) == []


class TestFSDPRuntimeBackward:
    @pytest.mark.parametrize("train_vision", [False, True])
    def test_deferred_sync_with_unit_skipping_forward_matches_synced_grads(
        self, world_size_one, train_vision
    ):
        # Arrange
        torch.manual_seed(1)
        micro_batches = [torch.randn(4, 8) for _ in range(2)]
        _, expected = _accumulated_step(
            False, micro_batches, [False, False], train_vision
        )

        # Act
        steps, stepped = _accumulated_step(
            True, micro_batches, [False, False], train_vision
        )

        # Assert
        assert steps[0] is None
        assert isinstance(steps[1], OptimizerStep)
        assert expected.keys() == DECODER_LORA
        assert stepped.keys() == expected.keys()
        for name, grad in expected.items():
            assert torch.equal(stepped[name], grad), name

    def test_deferred_sync_with_unit_first_running_mid_window_matches_synced_grads(
        self, world_size_one
    ):
        # Arrange
        torch.manual_seed(1)
        micro_batches = [torch.randn(4, 8) for _ in range(3)]
        use_vision = [False, True, True]
        _, expected = _accumulated_step(False, micro_batches, use_vision)

        # Act
        steps, stepped = _accumulated_step(True, micro_batches, use_vision)

        # Assert
        assert steps[:2] == [None, None]
        assert isinstance(steps[2], OptimizerStep)
        assert expected.keys() == DECODER_LORA | VISION_LORA
        assert stepped.keys() == expected.keys()
        for name, grad in expected.items():
            assert torch.equal(stepped[name], grad), name

    @pytest.mark.multi_rank
    @pytest.mark.parametrize(
        "use_vision",
        [[False, False], [False, True]],
        ids=["never_forwarded", "first_forward_on_second_micro_batch"],
    )
    def test_deferred_sync_across_ranks_matches_synced_grads(self, use_vision):
        # Arrange
        expected_names = DECODER_LORA | (VISION_LORA if any(use_vision) else set())

        # Act
        results = run_ranks(
            _deferred_and_synced_grads_worker, 2, use_vision, timeout=120.0
        )

        # Assert
        for deferred, synced in results:
            assert synced.keys() == expected_names
            assert deferred.keys() == synced.keys()
            for name, grad in synced.items():
                assert np.array_equal(deferred[name], grad), name
        for name, grad in results[0][1].items():
            assert np.array_equal(results[1][1][name], grad), name

    def test_deferred_sync_holds_trainable_grads_until_step_boundary(
        self, world_size_one
    ):
        # Arrange
        runtime, result = _prepare_fsdp_actor(_vision_language_actor())
        trainable = [p for p in result.actor.parameters() if p.requires_grad]
        x = torch.randn(4, 8)

        # Act
        runtime.backward(result.actor(x, True).sum(), result.optimizer, 3, result.actor)

        # Assert
        assert len(trainable) == 4
        assert all(param.grad is None for param in trainable)

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
        runtime = FSDPRuntime(FSDPConfig())
        synced: list[list[nn.Parameter]] = []

        def capture(params):
            synced.append(list(params))

        monkeypatch.setattr("agilerl.distributed.runtime.sync_grads", capture)
        runtime.backward(MagicMock(), optimizer, 1, actor=MagicMock(spec=FSDPModule))

        assert synced == [[replicated]]
        optimizer.step.assert_called_once()

    def test_does_not_sync_before_step_boundary(self, monkeypatch):
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.ones(2)
        optimizer = MagicMock()
        optimizer.optimizer.param_groups = [{"params": [param]}]
        runtime = FSDPRuntime(FSDPConfig())
        synced: list[object] = []
        monkeypatch.setattr(
            "agilerl.distributed.runtime.sync_grads",
            lambda params: synced.append(list(params)),
        )

        runtime.backward(MagicMock(), optimizer, 2, actor=MagicMock(spec=FSDPModule))

        assert synced == []
        optimizer.step.assert_not_called()

    def test_rejects_unsharded_actor(self):
        runtime = FSDPRuntime(FSDPConfig())

        with pytest.raises(TypeError, match="FSDP2-sharded actor"):
            runtime.backward(MagicMock(), MagicMock(), 1, actor=nn.Linear(2, 2))

    def test_clips_and_steps_at_boundary(self):
        param = nn.Parameter(torch.ones(2))
        param.grad = torch.ones(2) * 10
        inner = MagicMock()
        inner.param_groups = [{"params": [param]}]
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        runtime = FSDPRuntime(FSDPConfig())

        step = runtime.backward(
            MagicMock(),
            optimizer,
            1,
            actor=MagicMock(spec=FSDPModule),
            max_grad_norm=1.0,
        )

        assert torch.linalg.vector_norm(param.grad).item() == pytest.approx(1.0)
        assert step.group_grad_norms == pytest.approx((200**0.5,))
        assert step.grad_norm_post == pytest.approx(1.0)
        optimizer.step.assert_called_once_with()
        optimizer.zero_grad.assert_called_once_with()

    def test_clip_groups_clip_each_set_by_its_own_norm(self):
        # Arrange: actor norm 0.5, critic norm 50.
        actor = nn.Parameter(torch.zeros(2))
        actor.grad = torch.tensor([0.3, 0.4])
        critic = nn.Parameter(torch.zeros(2))
        critic.grad = torch.tensor([30.0, 40.0])
        inner = MagicMock()
        inner.param_groups = [
            {"params": [actor], "group": "actor"},
            {"params": [critic], "group": "critic"},
        ]
        optimizer = MagicMock()
        optimizer._single_optimizer.return_value = inner
        runtime = FSDPRuntime(FSDPConfig())

        # Act
        step = runtime.backward(
            MagicMock(),
            optimizer,
            1,
            actor=MagicMock(spec=FSDPModule),
            max_grad_norm=1.0,
            clip_groups=ACTOR_CRITIC_CLIP_GROUPS,
        )

        # Assert
        assert step.clip_coefs == pytest.approx((1.0, 1.0 / 50.0), rel=1e-6)
        assert torch.equal(actor.grad, torch.tensor([0.3, 0.4]))
        # rtol covers the 1e-6 epsilon in the clip coefficient
        assert torch.allclose(critic.grad, torch.tensor([0.6, 0.8]), rtol=1e-6, atol=0)


def _full_params(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        canonical_fsdp_param_fqn(name): (
            param.full_tensor() if isinstance(param, DTensor) else param
        )
        .detach()
        .float()
        .numpy()
        .copy()
        for name, param in model.named_parameters()
    }


def _prepare_ddp_actor(actor: nn.Module) -> tuple[DPRuntime, PrepareResult]:
    runtime = DPRuntime()
    result = runtime.prepare_actor(
        actor,
        device="cpu",
        colocated=False,
        cosine_lr_schedule_config=None,
        lr=1e-2,
        lr_critic=None,
        restore_adapter_trainability=lambda _: None,
    )
    return runtime, result


def _step_and_capture(
    runtime: DPRuntime | FSDPRuntime,
    result: PrepareResult,
    micro_batches: list[torch.Tensor],
) -> dict[str, Any]:
    """One accumulated step; the stepped grads, full params, and trainable dtypes."""
    actor = result.actor
    optimizer = result.optimizer._single_optimizer()
    names = {
        param: canonical_fsdp_param_fqn(name)
        for name, param in actor.named_parameters()
    }
    stepped: dict[str, np.ndarray] = {}
    grad_dtypes: set[str] = set()

    def capture_grads(opt, _args, _kwargs):
        for group in opt.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    grad_dtypes.add(str(param.grad.dtype))
                    stepped[names[param]] = full_grad(param).numpy().copy()

    optimizer.register_step_pre_hook(capture_grads)
    for x in micro_batches:
        runtime.backward(
            actor(x, True).float().sum(), result.optimizer, len(micro_batches), actor
        )
    trainable = [param for param in actor.parameters() if param.requires_grad]
    return {
        "grads": stepped,
        "params": _full_params(actor),
        "dtypes": {
            "params": {str(param.dtype) for param in trainable},
            "grads": grad_dtypes,
            "adam": {
                str(state.dtype)
                for param in trainable
                for key, state in optimizer.state[param].items()
                if key != "step"
            },
            "frozen": {
                str(param.dtype)
                for param in actor.parameters()
                if not param.requires_grad
            },
        },
    }


def _ddp_and_fsdp_step_worker(rank: int, world_size: int) -> dict[str, Any]:
    """One accumulated step per runtime on this rank's batch, and the unsynced local grads."""
    torch.manual_seed(1 + rank)
    micro_batches = [torch.randn(4, 8) for _ in range(2)]
    local = _vision_language_actor()
    for x in micro_batches:
        (local(x, True).sum() / len(micro_batches)).backward()
    ddp, ddp_result = _prepare_ddp_actor(_vision_language_actor())
    fsdp, fsdp_result = _prepare_fsdp_actor(
        _vision_language_actor(), param_dtype="float32"
    )
    return {
        "local": {
            name: param.grad.numpy().copy()
            for name, param in local.named_parameters()
            if param.grad is not None
        },
        "ddp": _step_and_capture(ddp, ddp_result, micro_batches),
        "fsdp": _step_and_capture(fsdp, fsdp_result, micro_batches),
    }


def _lora_over_bf16_base() -> VisionLanguageActor:
    """PEFT LoRA in fp32 on every block projection; the rest frozen in bf16."""
    torch.manual_seed(0)
    actor = VisionLanguageActor()
    # The optimizer trains every parameter named for actor LoRA.
    for block in (*actor.visual, *actor.layers):
        block.actor_lora = nn.Identity()
    actor = inject_adapter_in_model(
        LoraConfig(r=4, target_modules=["base", "proj"], init_lora_weights=False),
        actor,
        adapter_name="actor",
    )
    for name, param in actor.named_parameters():
        trainable = ".lora_" in name
        param.requires_grad_(trainable)
        if not trainable:
            param.data = param.data.to(torch.bfloat16)
    return actor


def _mixed_precision_step_worker(rank: int, world_size: int) -> dict[str, Any]:
    """DDP and bf16-compute FSDP steps of fp32 LoRA over a bf16 base."""
    torch.manual_seed(1 + rank)
    micro_batches = [torch.randn(4, 8, dtype=torch.bfloat16) for _ in range(2)]
    ddp, ddp_result = _prepare_ddp_actor(_lora_over_bf16_base())
    fsdp, fsdp_result = _prepare_fsdp_actor(
        _lora_over_bf16_base(),
        param_dtype="bfloat16",
        persistence_threshold=bmod.PARAM_PERSISTENCE_THRESHOLD,
    )
    return {
        "ddp": _step_and_capture(ddp, ddp_result, micro_batches),
        "fsdp": _step_and_capture(fsdp, fsdp_result, micro_batches),
    }


FP32_TRAINABLE_DTYPES = {
    "params": {"torch.float32"},
    "grads": {"torch.float32"},
    "adam": {"torch.float32"},
}


class TestDataParallelMatchesFsdpGradients:
    @pytest.mark.multi_rank
    def test_ddp_and_fsdp_step_on_the_rank_mean_gradient(self):
        # Act
        results = run_ranks(_ddp_and_fsdp_step_worker, 2, timeout=120.0)

        # Assert: fp32 on CPU, and halving a two-rank sum is exact.
        lora = DECODER_LORA | VISION_LORA
        expected = {
            name: (results[0]["local"][name] + results[1]["local"][name]) / 2
            for name in lora
        }
        for result in results:
            for mode in ("ddp", "fsdp"):
                grads = result[mode]["grads"]
                assert grads.keys() == lora, mode
                for name, grad in expected.items():
                    assert np.array_equal(grads[name], grad), (mode, name)
                dtypes = result[mode]["dtypes"]
                assert {key: dtypes[key] for key in FP32_TRAINABLE_DTYPES} == (
                    FP32_TRAINABLE_DTYPES
                ), mode
        for name, value in results[0]["ddp"]["params"].items():
            assert np.array_equal(results[1]["ddp"]["params"][name], value), name
            assert np.array_equal(results[0]["fsdp"]["params"][name], value), name

    @pytest.mark.multi_rank
    def test_fp32_lora_over_a_bf16_base_trains_in_fp32_on_both_paths(self):
        # Act
        results = run_ranks(_mixed_precision_step_worker, 2, timeout=120.0)

        # Assert: FSDP runs LoRA in bf16 and DDP in fp32 from bf16 inputs.
        # bf16 rounds each op to ~4e-3 relative; through four blocks the
        # first vision block's grad measures 1.1e-2 off in norm.
        for result in results:
            ddp = result["ddp"]
            fsdp = result["fsdp"]
            for mode in ("ddp", "fsdp"):
                dtypes = result[mode]["dtypes"]
                assert {key: dtypes[key] for key in FP32_TRAINABLE_DTYPES} == (
                    FP32_TRAINABLE_DTYPES
                ), mode
                assert dtypes["frozen"] == {"torch.bfloat16"}, mode
            assert fsdp["grads"].keys() == ddp["grads"].keys()
            assert len(ddp["grads"]) == 8
            for name, grad in ddp["grads"].items():
                diff = fsdp["grads"][name] - grad
                assert np.linalg.norm(diff) / np.linalg.norm(grad) < 2e-2, name
        for name, grad in results[0]["fsdp"]["grads"].items():
            assert np.array_equal(results[1]["fsdp"]["grads"][name], grad), name


class TestFSDPRuntimeBackwardGradSync:
    def test_toggles_sync_across_accumulation_window(self):
        runtime = FSDPRuntime(FSDPConfig())
        optimizer = MagicMock()
        loss = MagicMock()
        model = MagicMock(spec=FSDPModule)

        runtime.backward(loss, optimizer, 2, actor=model)
        model.set_requires_gradient_sync.assert_called_once_with(False, recurse=True)
        runtime.backward(loss, optimizer, 2, actor=model)
        model.set_requires_gradient_sync.assert_called_with(True, recurse=True)
        assert model.set_requires_gradient_sync.call_count == 2

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
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
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
            patch("agilerl.distributed.process.resolve_device", return_value="cpu"),
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


class PrefetchUnit(nn.Module):
    """Stand-in FSDP unit that records the prefetch lists it is given."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.forward_prefetch: list[nn.Module] = []
        self.backward_prefetch: list[nn.Module] = []

    def set_modules_to_forward_prefetch(self, modules):
        self.forward_prefetch = list(modules)

    def set_modules_to_backward_prefetch(self, modules):
        self.backward_prefetch = list(modules)


class PrefetchLanguage(nn.Module):
    def __init__(self, n_blocks: int):
        super().__init__()
        self.embed_tokens = PrefetchUnit()
        self.layers = nn.ModuleList([PrefetchUnit() for _ in range(n_blocks)])


class PrefetchVision(nn.Module):
    def __init__(self, n_blocks: int):
        super().__init__()
        self.blocks = nn.Sequential(*[PrefetchUnit() for _ in range(n_blocks)])


class PrefetchVisionLM(nn.Module):
    """Language tower registered before the vision tower, as in some VLMs."""

    def __init__(self):
        super().__init__()
        self.model = PrefetchLanguage(2)
        self.vision = PrefetchVision(2)
        self.lm_head = PrefetchUnit()


class TestSetPrefetch:
    def test_chains_embed_block_and_lm_head_units(self):
        # Arrange
        model = PrefetchVisionLM()
        del model.vision
        embed = model.model.embed_tokens
        first, second = model.model.layers
        head = model.lm_head

        # Act
        with patch.object(bmod, "FSDPModule", PrefetchUnit):
            bmod._set_prefetch(model, [first, second])

        # Assert
        assert embed.forward_prefetch == [first]
        assert first.forward_prefetch == [second]
        assert second.forward_prefetch == [head]
        assert head.forward_prefetch == []
        assert first.backward_prefetch == [embed]
        assert second.backward_prefetch == [first]
        assert head.backward_prefetch == [second]

    def test_towers_prefetch_only_their_own_blocks(self):
        # Arrange
        model = PrefetchVisionLM()
        language_blocks = list(model.model.layers)
        vision_blocks = list(model.vision.blocks)

        # Act
        with patch.object(bmod, "FSDPModule", PrefetchUnit):
            bmod._set_prefetch(model, language_blocks + vision_blocks)

        # Assert
        assert language_blocks[1].forward_prefetch == [model.lm_head]
        assert vision_blocks[0].forward_prefetch == [vision_blocks[1]]
        assert vision_blocks[1].forward_prefetch == []
        assert vision_blocks[0].backward_prefetch == []
        assert vision_blocks[1].backward_prefetch == [vision_blocks[0]]
        assert model.lm_head.backward_prefetch == [language_blocks[1]]

    def test_chains_ple_between_embed_and_language_blocks(self):
        # Arrange
        model = PrefetchVisionLM()
        del model.vision
        model.model.embed_tokens_per_layer = PrefetchUnit()
        embed = model.model.embed_tokens
        ple = model.model.embed_tokens_per_layer
        first, second = model.model.layers
        head = model.lm_head

        # Act
        with patch.object(bmod, "FSDPModule", PrefetchUnit):
            bmod._set_prefetch(model, [first, second])

        # Assert
        assert embed.forward_prefetch == [ple]
        assert ple.forward_prefetch == [first]
        assert first.forward_prefetch == [second]
        assert second.forward_prefetch == [head]
        assert ple.backward_prefetch == [embed]
        assert first.backward_prefetch == [ple]
        assert second.backward_prefetch == [first]
        assert head.backward_prefetch == [second]


class TypedBlock(nn.Module):
    """Hybrid block stand-in: ``block_type`` names its mixer."""

    def __init__(self, block_type: str):
        super().__init__()
        self.block_type = block_type
        self.lin = nn.Linear(2, 2)


class HybridStack(nn.Module):
    _no_split_modules: ClassVar = ["TypedBlock"]

    def __init__(self, block_types: list[str]):
        super().__init__()
        self.layers = nn.ModuleList([TypedBlock(kind) for kind in block_types])


def checkpointed_kinds(body: nn.Module) -> list[str]:
    """``block_type`` of each checkpoint-wrapped entry in ``body.layers``."""
    return [
        layer._checkpoint_wrapped_module.block_type
        for layer in body.layers
        if isinstance(layer, CheckpointWrapper)
    ]


def tiny_hybrid_lm() -> NemotronHForCausalLM:
    """fp32 Nemotron-H with one Mamba, attention, MoE, and MLP block."""
    config = NemotronHConfig(
        vocab_size=64,
        hidden_size=16,
        layers_block_type=["mamba", "attention", "moe", "mlp"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        use_mamba_kernels=False,
        ssm_state_size=8,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=1,
        conv_kernel=2,
        chunk_size=8,
        max_position_embeddings=64,
        use_cache=False,
        attn_implementation="eager",
    )
    return NemotronHForCausalLM(config).float()


def tiny_dense_lm() -> Qwen3ForCausalLM:
    """fp32 Qwen3 with four decoder layers."""
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_cache=False,
    )
    return Qwen3ForCausalLM(config)


def tiny_gated_delta_lm() -> Qwen3_5MoeForCausalLM:
    """fp32 Qwen3.5-MoE alternating GDN and full-attention layers."""
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"] * 2,
        use_cache=False,
    )
    return Qwen3_5MoeForCausalLM(config)


def cpu_parallel_mesh() -> ParallelMesh:
    # FSDP's default mesh uses the process accelerator (MPS on macOS).
    world = init_device_mesh("cpu", (1, 1), mesh_dim_names=("replicate", "shard"))
    return ParallelMesh(world=world, hsdp=world["shard"])


def full_grad(param: torch.Tensor) -> torch.Tensor:
    grad = param.grad
    return grad.full_tensor() if isinstance(grad, DTensor) else grad


class TestMaterializeFsdp2FromCpuStateBlocks:
    config = FSDPConfig(
        param_dtype="float32",
        reduce_dtype="float32",
        routed_expert_chunk_mib=64,
        checkpoint_every_n_blocks=1,
    )

    @pytest.mark.parametrize(
        "build", [tiny_dense_lm, tiny_gated_delta_lm, tiny_hybrid_lm]
    )
    def test_dense_model_keeps_checkpoint_names_and_weights(
        self, world_size_one, build
    ):
        # Arrange
        torch.manual_seed(0)
        reference = build()
        model = copy.deepcopy(reference)
        ids = torch.randint(0, 64, (1, 9))
        with torch.no_grad():
            expected = reference(input_ids=ids).logits

        # Act
        materialize_fsdp2_from_cpu_state(
            model, "cpu", self.config, parallel_mesh=cpu_parallel_mesh()
        )

        # Assert
        assert list(model.state_dict()) == list(reference.state_dict())
        with torch.no_grad():
            assert torch.equal(model(input_ids=ids).logits, expected)

    def test_meta_model_loads_blocks_from_safetensors(self, world_size_one, tmp_path):
        # Arrange
        torch.manual_seed(0)
        reference = tiny_dense_lm()
        reference.save_pretrained(tmp_path)
        with torch.device("meta"):
            model = Qwen3ForCausalLM(Qwen3Config.from_pretrained(tmp_path))
        model.config.name_or_path = str(tmp_path)
        ids = torch.randint(0, 64, (1, 9))
        with torch.no_grad():
            expected = reference(input_ids=ids).logits

        # Act
        materialize_fsdp2_from_cpu_state(
            model, "cpu", self.config, parallel_mesh=cpu_parallel_mesh()
        )

        # Assert
        with torch.no_grad():
            assert torch.equal(model(input_ids=ids).logits, expected)

    def test_checkpointed_blocks_match_reference_grads(self, world_size_one):
        # Arrange
        torch.manual_seed(0)
        reference = tiny_hybrid_lm()
        model = copy.deepcopy(reference)
        ids = torch.randint(0, 64, (1, 9))
        reference(input_ids=ids, labels=ids).loss.backward()
        materialize_fsdp2_from_cpu_state(
            model,
            "cpu",
            self.config,
            parallel_mesh=cpu_parallel_mesh(),
            gradient_checkpointing=True,
        )

        # Act
        model(input_ids=ids, labels=ids).loss.backward()

        # Assert
        grads = {
            dmod.canonical_fsdp_param_fqn(name): full_grad(param)
            for name, param in model.named_parameters()
        }
        assert any(isinstance(layer, CheckpointWrapper) for layer in model.model.layers)
        for name, param in reference.named_parameters():
            # fp32 recompute of the same ops; tolerance covers reduction order.
            torch.testing.assert_close(grads[name], param.grad, rtol=0, atol=1e-6)


class TestApplyFsdp2:
    per_block = FSDPConfig(reshard_after_forward=True, checkpoint_every_n_blocks=1)

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
        with patch.object(bmod, "fully_shard", side_effect=_record):
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
        with patch.object(bmod, "fully_shard", side_effect=_record):
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
        with patch.object(bmod, "fully_shard", side_effect=_record):
            apply_fsdp2(tied, self.per_block)

        # Assert
        assert tied.model.embed_tokens not in sharded
        assert tied.lm_head not in sharded
        assert sharded[-1] is tied
        assert (
            tied.model.embed_tokens.weight in shard_kwargs[id(tied)]["ignored_params"]
        )
        assert tied.lm_head.weight in shard_kwargs[id(tied)]["ignored_params"]

    def test_shards_per_layer_embeddings_as_their_own_unit(self, world_size_one):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(2, 2)

        class LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(4, 2)
                self.embed_tokens_per_layer = nn.Embedding(4, 6)
                self.layers = nn.ModuleList([Block(), Block()])
                self.norm = nn.LayerNorm(2)

        class CausalLM(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {"tie_word_embeddings": True})()
                self.model = LanguageModel()
                self.lm_head = nn.Linear(2, 4, bias=False)

        # Arrange
        model = CausalLM()
        sharded: list[nn.Module] = []
        shard_kwargs: dict[int, dict] = {}

        def _record(module, **kwargs):
            sharded.append(module)
            shard_kwargs[id(module)] = kwargs
            return module

        # Act
        with patch.object(bmod, "fully_shard", side_effect=_record):
            apply_fsdp2(model, self.per_block)

        # Assert — token embeddings stay replicated; PLE is its own unit
        ple = model.model.embed_tokens_per_layer
        assert model.model.embed_tokens not in sharded
        assert ple in sharded
        assert model.lm_head not in sharded
        assert sharded[-1] is model
        assert set(sharded) == {
            model.model.layers[0],
            model.model.layers[1],
            ple,
            model,
        }
        assert shard_kwargs[id(ple)]["reshard_after_forward"] is True
        assert (
            model.model.embed_tokens.weight in shard_kwargs[id(model)]["ignored_params"]
        )

    def test_layer0_lora_stays_finite_after_fsdp_train_forward(self, world_size_one):
        torch.manual_seed(0)
        config = Gemma4TextConfig(
            vocab_size=64,
            vocab_size_per_layer_input=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            global_head_dim=8,
            hidden_size_per_layer_input=8,
            max_position_embeddings=32,
            sliding_window=16,
            pad_token_id=0,
            tie_word_embeddings=True,
            layer_types=["sliding_attention", "full_attention"],
            num_kv_shared_layers=0,
        )
        model = get_peft_model(
            Gemma4ForCausalLM(config),
            LoraConfig(
                r=4,
                target_modules=["q_proj", "v_proj"],
                lora_alpha=8,
                lora_dropout=0.0,
            ),
            adapter_name="actor",
        )
        # FSDP's default mesh uses the process accelerator (MPS on macOS).
        world = init_device_mesh("cpu", (1, 1), mesh_dim_names=("replicate", "shard"))

        with patch.object(bmod, "PARAM_PERSISTENCE_THRESHOLD", 0):
            materialize_fsdp2_from_cpu_state(
                model,
                "cpu",
                FSDPConfig(
                    param_dtype="bfloat16",
                    reduce_dtype="float32",
                    routed_expert_chunk_mib=64,
                ),
                parallel_mesh=ParallelMesh(world=world, hsdp=world["shard"]),
            )

        language = model.get_base_model().model
        assert isinstance(language.embed_tokens_per_layer, FSDPModule)
        layer0 = language.layers[0].self_attn.q_proj.lora_A["actor"]
        seen: dict[str, bool] = {}

        def _hook(module, _args, _output):
            weight = module.weight.detach()
            if hasattr(weight, "full_tensor"):
                weight = weight.full_tensor()
            seen["finite"] = bool(torch.isfinite(weight).all())

        handle = layer0.register_forward_hook(_hook)
        ids = torch.randint(1, 64, (2, 8))
        with torch.enable_grad():
            logits = model(input_ids=ids).logits
            logits.mean().backward()
        handle.remove()

        assert seen["finite"] is True
        assert bool(torch.isfinite(logits).all())

    def test_checkpoint_wraps_blocks_then_shards_the_wrapper(self, world_size_one):

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

        def _capture(_model, block_units):
            captured_blocks.extend(block_units)

        with (
            patch.object(bmod, "fully_shard", side_effect=_record),
            patch.object(bmod, "_set_prefetch", side_effect=_capture),
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

    def test_checkpoint_every_n_wraps_every_nth_block(self, world_size_one):
        # Arrange
        model = HybridStack(["mamba", "attention", "moe", "mamba", "moe"])
        config = FSDPConfig(checkpoint_every_n_blocks=2)

        # Act
        with patch.object(bmod, "fully_shard", side_effect=lambda m, **_kw: m):
            apply_fsdp2(model, config, gradient_checkpointing=True)

        # Assert
        wrapped = [isinstance(layer, CheckpointWrapper) for layer in model.layers]
        assert wrapped == [True, False, True, False, True]

    def test_unset_checkpoint_stride_raises(self, world_size_one):
        model = HybridStack(["mamba", "attention", "moe"])

        with (
            patch.object(bmod, "fully_shard", side_effect=lambda m, **_kw: m),
            pytest.raises(ValueError, match="checkpoint_every_n_blocks is unset"),
        ):
            apply_fsdp2(model, FSDPConfig(), gradient_checkpointing=True)

    def test_checkpoint_wraps_every_block_kind(self, world_size_one):
        # Arrange
        model = HybridStack(["mamba", "attention", "moe"])

        # Act
        with patch.object(bmod, "fully_shard", side_effect=lambda m, **_kw: m):
            apply_fsdp2(
                model,
                FSDPConfig(checkpoint_every_n_blocks=1),
                gradient_checkpointing=True,
            )

        # Assert
        assert checkpointed_kinds(model) == ["mamba", "attention", "moe"]

    def test_no_block_is_checkpointed_without_gradient_checkpointing(
        self, world_size_one
    ):
        model = HybridStack(["mamba", "moe"])

        with patch.object(bmod, "fully_shard", side_effect=lambda m, **_kw: m):
            apply_fsdp2(model, FSDPConfig())

        assert checkpointed_kinds(model) == []

    def test_checkpoint_keeps_hybrid_model_gradients(self, world_size_one):
        # Arrange
        torch.manual_seed(0)
        reference = tiny_hybrid_lm()
        input_ids = torch.randint(0, reference.config.vocab_size, (2, 8))
        fp32 = FSDPConfig(param_dtype="float32", checkpoint_every_n_blocks=1)
        mesh = init_device_mesh("cpu", (1,))
        runs = {"none": False, "full": True}

        # Act
        grads: dict[str, dict[str, torch.Tensor]] = {}
        wrapped: dict[str, list[str]] = {}
        for name, checkpointing in runs.items():
            model = copy.deepcopy(reference)
            apply_fsdp2(model, fp32, mesh=mesh, gradient_checkpointing=checkpointing)
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            wrapped[name] = checkpointed_kinds(model.model)
            grads[name] = {
                dmod.canonical_fsdp_param_fqn(key): full_grad(param)
                for key, param in model.named_parameters()
                if param.grad is not None
            }

        # Assert
        assert wrapped == {
            "none": [],
            "full": ["linear_attention", "full_attention", "moe", "mlp"],
        }
        assert len(grads["none"]) > 0
        assert grads["none"].keys() == grads["full"].keys()
        for key, expected in grads["none"].items():
            # fp32 on CPU; recompute replays the same kernels, so drift stays at rounding.
            assert torch.allclose(grads["full"][key], expected, atol=1e-6, rtol=1e-5)

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
            patch.object(bmod, "fully_shard", side_effect=lambda m, **_kw: m),
            patch.object(bmod, "FSDPModule", PrefetchBlock),
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

    def test_forwards_mesh_to_fully_shard(self, world_size_one):
        model = nn.Linear(2, 2)
        mesh = object()
        seen: dict[int, dict] = {}

        def _record(module, **kwargs):
            seen[id(module)] = kwargs
            return module

        with patch.object(bmod, "fully_shard", side_effect=_record):
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
            bmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model, self.per_block)

        assert model.layers[0] not in sharded
        assert model in sharded

    def test_persists_frozen_params_below_threshold(self, world_size_one):
        # Arrange
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(512, 256)
                self.norm = nn.LayerNorm(512)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block()])

        model = Model().requires_grad_(False)
        block = model.layers[0]
        block.lin.bias.requires_grad_(True)
        seen: dict[int, dict] = {}

        def _record(module, **kwargs):
            seen[id(module)] = kwargs
            return module

        # Act
        with patch.object(bmod, "fully_shard", side_effect=_record):
            apply_fsdp2(model, FSDPConfig())

        # Assert: small trainable parameters shard like large ones.
        ignored = seen[id(block)]["ignored_params"]
        assert block.norm.weight in ignored
        assert block.norm.bias in ignored
        assert block.lin.bias not in ignored
        assert block.lin.weight not in ignored

    def test_casts_persisted_frozen_params_to_mixed_precision_dtype(
        self, world_size_one
    ):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(512, 256)
                self.norm = nn.LayerNorm(512)

        class Model(nn.Module):
            _no_split_modules: ClassVar = ["Block"]

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Block()])

        model = Model().requires_grad_(False)
        block = model.layers[0]
        block.lin.bias.requires_grad_(True)
        assert block.norm.weight.dtype == torch.float32

        with patch.object(bmod, "fully_shard", side_effect=lambda m, **_kw: m):
            apply_fsdp2(model, FSDPConfig())

        assert block.norm.weight.dtype == torch.bfloat16
        assert block.norm.bias.dtype == torch.bfloat16
        assert block.lin.bias.dtype == torch.float32
        assert block.lin.weight.dtype == torch.float32

    def test_shards_root_only_without_no_split_metadata(self, world_size_one):
        # Arrange
        model = nn.Linear(2, 2)
        sharded: list[nn.Module] = []

        # Act
        with patch.object(
            bmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
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
            bmod,
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
            bmod,
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
            bmod,
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
            patch.object(bmod, "fully_shard", side_effect=lambda m, **_kw: m),
            patch.object(
                bmod,
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
            bmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
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
            bmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model, self.per_block)

        assert model.layers[0] in sharded
        assert model.layers[1] in sharded
        assert model in sharded
        assert model.layers[0].inner not in sharded
        assert model.layers[1].inner not in sharded

    def test_wraps_one_unit_per_block(self, world_size_one):
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
            bmod, "fully_shard", side_effect=lambda m, **_kw: sharded.append(m) or m
        ):
            apply_fsdp2(model)

        assert list(model.layers) == raw
        assert raw[0] in sharded
        assert raw[1] in sharded
        assert sharded[-1] is model

    def test_raises_without_process_group(self):
        with pytest.raises(RuntimeError, match="initialised process group"):
            apply_fsdp2(nn.Linear(2, 2), FSDPConfig())


MOE_COMMON: dict[str, Any] = {
    "hidden_size": 32,
    "intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 64,
    "max_position_embeddings": 64,
    "num_hidden_layers": 2,
    "use_cache": False,
    "experts_implementation": "eager",
}

MOE_CONFIGS = {
    "qwen3_moe": lambda: Qwen3MoeConfig(
        **MOE_COMMON,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    ),
    "qwen3_5_moe_text": lambda: Qwen3_5MoeTextConfig(
        **MOE_COMMON,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "full_attention"],
    ),
    "gemma4_text": lambda: Gemma4TextConfig(
        **MOE_COMMON,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=16,
        layer_types=["sliding_attention", "sliding_attention"],
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    ),
    "granitemoe": lambda: GraniteMoeConfig(
        **MOE_COMMON, num_local_experts=4, num_experts_per_tok=2
    ),
    "nemotron_h": lambda: NemotronHConfig(
        **MOE_COMMON,
        layers_block_type=["attention", "moe"],
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    ),
    "gpt_oss": lambda: GptOssConfig(
        **MOE_COMMON,
        num_local_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "full_attention"],
    ),
    "qwen2_moe": lambda: Qwen2MoeConfig(
        **MOE_COMMON,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    ),
    "olmoe": lambda: OlmoeConfig(
        **MOE_COMMON, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True
    ),
    "mellum": lambda: MellumConfig(
        **MOE_COMMON,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    ),
    "qwen4_exp_text": lambda: Qwen4ExpTextConfig(
        **MOE_COMMON,
        moe_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    ),
    "mixtral": lambda: MixtralConfig(
        **MOE_COMMON, num_local_experts=4, num_experts_per_tok=2
    ),
    "minimax": lambda: MiniMaxConfig(
        **MOE_COMMON, num_local_experts=4, num_experts_per_tok=2
    ),
}


def tiny_moe_lm(model_type: str, dtype: torch.dtype) -> nn.Module:
    """Two-block causal LM of ``model_type`` with seeded, non-trivial weights."""
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(MOE_CONFIGS[model_type]())
    for param in model.parameters():
        nn.init.normal_(param, std=0.05)
    return model.to(dtype)


def loss_and_full_grads(
    model: nn.Module, input_ids: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss and full gradients keyed by canonical parameter name."""
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    grads = {
        dmod.canonical_fsdp_param_fqn(key): full_grad(param)
        for key, param in model.named_parameters()
        if param.grad is not None
    }
    return loss.detach(), grads


class TestApplyFsdp2GradientCheckpointing:
    @pytest.fixture(autouse=True)
    def pristine_moe_routers(self):
        with restore_moe_routers():
            yield

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
    )
    @pytest.mark.parametrize("model_type", list(MOE_CONFIGS))
    def test_moe_loss_and_grads_match_uncheckpointed(
        self, world_size_one, model_type, dtype
    ):
        # Arrange
        reference = tiny_moe_lm(model_type, dtype)
        model = copy.deepcopy(reference)
        input_ids = torch.randint(
            0, 64, (2, 12), generator=torch.Generator().manual_seed(1)
        )
        # Before apply_fsdp2, so the reference runs the unpatched routers.
        expected_loss, expected_grads = loss_and_full_grads(reference, input_ids)
        dtype_name = str(dtype).removeprefix("torch.")
        config = FSDPConfig(
            param_dtype=dtype_name, reduce_dtype=dtype_name, checkpoint_every_n_blocks=1
        )
        apply_fsdp2(
            model,
            config,
            mesh=init_device_mesh("cpu", (1,)),
            gradient_checkpointing=True,
        )

        # Act
        loss, grads = loss_and_full_grads(model, input_ids)

        # Assert
        assert torch.equal(loss, expected_loss)
        assert grads.keys() == expected_grads.keys()
        for key, expected in expected_grads.items():
            assert torch.equal(grads[key], expected), key


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


class TestInitRopeBuffers:
    def test_fills_inv_freq_and_original_inv_freq(self):
        class RotaryEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope_type = "default"
                self.config = SimpleNamespace()
                self.register_buffer("inv_freq", torch.zeros(4))
                self.register_buffer("original_inv_freq", torch.zeros(4))

            @staticmethod
            def compute_default_rope_parameters(_config, device=None):
                return torch.arange(4, dtype=torch.float32, device=device), 1.0

        module = RotaryEmbedding()
        root = nn.Sequential(module)

        init_rope_buffers(root)

        expected = torch.arange(4, dtype=torch.float32)
        assert torch.equal(module.inv_freq, expected)
        assert torch.equal(module.original_inv_freq, expected)

    def test_fills_axial_vision_rope(self):
        # Arrange
        modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
        config_module = pytest.importorskip(
            "transformers.models.qwen3_5.configuration_qwen3_5"
        )
        module = modeling.Qwen3_5VisionRotaryEmbedding(
            config_module.Qwen3_5VisionConfig()
        )
        expected = module.inv_freq.clone()
        module.inv_freq.zero_()
        module.original_inv_freq.zero_()

        # Act
        init_rope_buffers(nn.Sequential(module))

        # Assert
        assert module.rope_type == "axial"
        assert torch.equal(module.inv_freq, expected)
        assert torch.equal(module.original_inv_freq, expected)

    def test_fills_per_layer_type_inv_freq_with_each_type_head_dim(self):
        # Arrange
        # As in a checkpoint's config.json: hidden_size / heads is neither head dim.
        config = Gemma4TextConfig.from_dict(
            {
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "global_head_dim": 32,
                "intermediate_size": 32,
                "vocab_size": 64,
                "num_hidden_layers": 2,
                "layer_types": ["sliding_attention", "full_attention"],
                "hidden_size_per_layer_input": 0,
                "num_kv_shared_layers": 0,
            }
        )
        module = Gemma4TextRotaryEmbedding(config)
        names = [
            f"{layer_type}_{kind}"
            for layer_type in ("sliding_attention", "full_attention")
            for kind in ("inv_freq", "original_inv_freq")
        ]
        expected = {name: getattr(module, name).clone() for name in names}
        for name in names:
            getattr(module, name).zero_()

        # Act
        init_rope_buffers(nn.Sequential(module))

        # Assert
        assert expected["full_attention_inv_freq"].shape == (16,)
        for name in names:
            assert torch.equal(getattr(module, name), expected[name]), name


class TestInitEmbedScaleBuffers:
    def test_fills_embed_scale_from_scalar(self):
        class ScaledWordEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar_embed_scale = 3.5
                self.register_buffer("embed_scale", torch.tensor(0.0), persistent=False)

        module = ScaledWordEmbedding()

        init_embed_scale_buffers(module)

        assert float(module.embed_scale) == 3.5

    def test_skips_meta_embed_scale(self):
        class ScaledWordEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar_embed_scale = 3.5
                self.register_buffer(
                    "embed_scale",
                    torch.empty((), device="meta"),
                    persistent=False,
                )

        module = ScaledWordEmbedding()

        init_embed_scale_buffers(module)

        assert module.embed_scale.is_meta


class TestDeferredSplitPassReduceDtype:
    """Two backward passes over different params of one FSDP unit with deferred sync.

    A split actor/critic learn holds the policy pass's gradients while the
    value pass runs. Held gradients wait in ``reduce_dtype``; the step
    boundary then collects them next to the fresh value gradients and
    reduce-scatter requires one dtype across the unit.
    """

    def _split_pass_model(self) -> nn.Module:
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).to(torch.bfloat16)
        for param in model.parameters():
            param.requires_grad_(True)
        return model

    def _shard_single_unit(self, model: nn.Module, reduce_dtype: str) -> None:
        # A CPU mesh keeps the test off the default device (MPS on macOS).
        mesh = DeviceMesh("cpu", [0])
        with patch.object(bmod, "PARAM_PERSISTENCE_THRESHOLD", 0):
            apply_fsdp2(model, FSDPConfig(reduce_dtype=reduce_dtype), mesh=mesh)

    def _run_split_passes(self, model: nn.Module) -> None:
        # PEFT adapter switching enables one adapter's gradients per pass;
        # toggling requires_grad reproduces that attribution.
        inputs = torch.randn(2, 8, dtype=torch.bfloat16)
        model[1].weight.requires_grad_(False)
        model[1].bias.requires_grad_(False)
        model.set_requires_gradient_sync(False, recurse=True)
        model(inputs).pow(2).mean().backward()
        model[0].weight.requires_grad_(False)
        model[0].bias.requires_grad_(False)
        model[1].weight.requires_grad_(True)
        model[1].bias.requires_grad_(True)
        model.set_requires_gradient_sync(True, recurse=True)
        model(inputs).pow(2).mean().backward()

    def test_split_passes_mix_held_and_fresh_gradients(self, world_size_one):
        # Arrange
        model = self._split_pass_model()
        self._shard_single_unit(model, reduce_dtype="float32")

        # Act / Assert
        with pytest.raises(AssertionError, match="uniform gradient dtype"):
            self._run_split_passes(model)

    def test_param_dtype_reduce_keeps_split_passes_uniform(self, world_size_one):
        # Arrange
        model = self._split_pass_model()
        self._shard_single_unit(model, reduce_dtype="bfloat16")

        # Act
        self._run_split_passes(model)

        # Assert
        grads = [param.grad for param in model.parameters()]
        assert all(grad is not None for grad in grads)
        assert {grad.dtype for grad in grads} == {torch.bfloat16}


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

    @pytest.mark.gpu
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

    def test_pinned_states_reuse_their_host_buffers_and_match_adamw(self):
        # Arrange
        torch.manual_seed(0)
        model, opt = self._make_optimizer("cpu")
        reference = copy.deepcopy(model)
        reference_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        offload = CPUOffloadOptimizer(opt, pin_memory=True)
        pin_patch = patch.object(torch.Tensor, "pin_memory", torch.Tensor.clone)
        torch.manual_seed(1)
        with pin_patch:
            self._step_once(model, offload, "cpu")
        torch.manual_seed(1)
        self._step_once(reference, reference_opt, "cpu")

        def moments():
            return [
                value
                for state in offload.state.values()
                for key, value in state.items()
                if key != "step"
            ]

        host_states = moments()

        # Act
        devices, move_patch = self._record_moves()
        torch.manual_seed(2)
        with pin_patch, move_patch:
            self._step_once(model, offload, "cpu")
        torch.manual_seed(2)
        self._step_once(reference, reference_opt, "cpu")

        # Assert: the moments return to the host tensors they left from, and
        # only the device copy is a ``.to`` call.
        assert devices == ["cuda"] * len(host_states)
        assert all(
            after is before
            for after, before in zip(moments(), host_states, strict=True)
        )
        for param, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(param, expected, atol=0, rtol=0)

    def _record_moves(self):
        """Record ``.to("cuda" | "cpu")`` calls; tensors stay on the CPU."""
        devices: list[str] = []
        real_to = torch.Tensor.to

        def record(tensor, *args, **kwargs):
            if args and args[0] in ("cuda", "cpu"):
                devices.append(args[0])
                return real_to(tensor, "cpu")
            return real_to(tensor, *args, **kwargs)

        return devices, patch.object(torch.Tensor, "to", record)

    def test_step_after_init_moves_states_around_step(self):
        # Arrange — stepped once so states exist and live on CPU
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        self._step_once(model, offload, "cpu")

        # Act — moves are recorded; tensors stay on CPU in this process
        devices, move_patch = self._record_moves()
        with move_patch:
            self._step_once(model, offload, "cpu")

        # Assert — every state goes to CUDA for step(), then back to CPU
        # Adam's scalar ``step`` stays on CPU; only moment tensors move.
        n_states = sum(
            1 for state in opt.state.values() for key in state if key != "step"
        )
        assert n_states > 0
        assert devices == ["cuda"] * n_states + ["cpu"] * n_states

    def test_step_leaves_non_tensor_states_in_place(self):
        # Arrange
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        self._step_once(model, offload, "cpu")
        param = next(iter(offload.state))
        offload.state[param]["note"] = 0.5

        # Act
        devices, move_patch = self._record_moves()
        with move_patch:
            self._step_once(model, offload, "cpu")

        # Assert
        moments = sum(
            1
            for state in opt.state.values()
            for key, value in state.items()
            if key != "step" and isinstance(value, torch.Tensor)
        )
        assert offload.state[param]["note"] == 0.5
        assert devices == ["cuda"] * moments + ["cpu"] * moments

    def test_state_dict_after_init_moves_states_around_snapshot(self):
        # Arrange
        model, opt = self._make_optimizer("cpu")
        offload = CPUOffloadOptimizer(opt, pin_memory=False)
        self._step_once(model, offload, "cpu")

        # Act
        devices, move_patch = self._record_moves()
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
        _devices_a, snapshot_patch = self._record_moves()
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
        devices, move_patch = self._record_moves()
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


class _StubRoutedExperts(nn.Module):
    """Minimal gated packed experts (Nemotron-style ungated layout)."""

    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.up_proj = nn.Parameter(torch.randn(num_experts, 8, 16))
        self.down_proj = nn.Parameter(torch.randn(num_experts, 16, 8))
        self.act_fn = torch.nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states


class _StubSortedExperts(nn.Module):
    """Minimal grouped linear over expert-sorted rows."""

    def __init__(self, num_experts: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, 8, 16))

    def forward(self, inputs: torch.Tensor, expert_size: list[int]) -> torch.Tensor:
        return inputs


class TestPackedExpertCounts:
    def test_empty_on_dense_model(self) -> None:
        # Arrange
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))

        # Act
        counts = packed_expert_counts(model)

        # Assert
        assert counts == []

    def test_reports_routed_and_sorted_counts(self) -> None:
        # Arrange
        model = nn.Sequential(_StubRoutedExperts(4), _StubSortedExperts(8))

        # Act
        counts = packed_expert_counts(model)

        # Assert
        assert counts == [4, 8]


class TestValidateActorEp:
    def test_ep_one_passes_dense_model(self) -> None:
        # Arrange
        model = nn.Sequential(nn.Linear(8, 8))

        # Act
        counts = validate_actor_ep(model, 1)

        # Assert
        assert counts == []

    def test_ep_one_returns_counts_unchanged(self) -> None:
        # Arrange
        model = nn.Sequential(_StubRoutedExperts(4))

        # Act
        counts = validate_actor_ep(model, 1, world_size=4)

        # Assert
        assert counts == [4]

    def test_accepts_divisible_packed_model(self) -> None:
        # Arrange
        model = nn.Sequential(_StubRoutedExperts(4), _StubSortedExperts(8))

        # Act
        counts = validate_actor_ep(model, 2, world_size=4)

        # Assert
        assert counts == [4, 8]

    def test_rejects_ep_below_one(self) -> None:
        # Arrange
        model = nn.Sequential(_StubRoutedExperts(4))

        # Act / Assert
        with pytest.raises(ValueError, match="ep must be >= 1"):
            validate_actor_ep(model, 0)

    def test_rejects_dense_model(self) -> None:
        # Arrange
        model = nn.Sequential(nn.Linear(8, 8))

        # Act / Assert
        with pytest.raises(ValueError, match="packed-expert"):
            validate_actor_ep(model, 2, world_size=2)

    def test_rejects_indivisible_world_size(self) -> None:
        # Arrange
        model = nn.Sequential(_StubRoutedExperts(4))

        # Act / Assert
        with pytest.raises(ValueError, match="world size 3 is not divisible"):
            validate_actor_ep(model, 2, world_size=3)

    def test_rejects_indivisible_expert_count(self) -> None:
        # Arrange
        model = nn.Sequential(_StubRoutedExperts(3))

        # Act / Assert
        with pytest.raises(ValueError, match="expert count 3 is not divisible"):
            validate_actor_ep(model, 2, world_size=4)


class TestValidateEpDegree:
    @pytest.mark.parametrize("ep", [1, 2, 8])
    def test_accepts_positive_ints(self, ep: int) -> None:
        # Act / Assert — returns nothing on valid input
        assert validate_ep_degree(ep) is None

    def test_rejects_zero_and_negative(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="ep must be >= 1"):
            validate_ep_degree(0)

    @pytest.mark.parametrize("ep", [True, 2.0, "2"])
    def test_rejects_non_ints(self, ep: object) -> None:
        # Act / Assert
        with pytest.raises(TypeError, match="ep must be an int"):
            validate_ep_degree(ep)  # type: ignore[arg-type]


class TestEpDataParallelSize:
    def test_ep_one_keeps_world(self) -> None:
        # Act / Assert
        assert ep_data_parallel_size(4, 1) == 4

    def test_folds_world_by_ep(self) -> None:
        # Act / Assert
        assert ep_data_parallel_size(8, 4) == 2

    def test_rejects_indivisible_world(self) -> None:
        # Act / Assert
        with pytest.raises(ValueError, match="must be divisible by ep"):
            ep_data_parallel_size(3, 2)


class TestReferenceDispatchCombine:
    def test_round_trip_restores_token_order(self) -> None:
        # Arrange
        torch.manual_seed(0)
        tokens = torch.randn(12, 8)
        expert_ids = torch.tensor([3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2])

        # Act
        restored = reference_dispatch_combine(
            tokens, expert_ids, ep_degree=2, num_experts=4
        )

        # Assert
        assert torch.equal(restored, tokens)

    def test_rejects_indivisible_experts(self) -> None:
        # Arrange
        tokens = torch.randn(4, 8)
        expert_ids = torch.zeros(4, dtype=torch.long)

        # Act / Assert
        with pytest.raises(ValueError, match="must be divisible by ep"):
            reference_dispatch_combine(tokens, expert_ids, ep_degree=2, num_experts=3)


class TestFSDPPrepareActorExpertParallel:
    def test_validates_packed_experts_when_ep_is_above_one(self):
        actor = nn.Linear(2, 2)
        restore = MagicMock()
        with (
            patch("agilerl.distributed.runtime.validate_actor_ep") as validate,
            patch("agilerl.distributed.runtime.build_parallel_mesh", return_value=None),
            patch(
                "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
                return_value=actor,
            ),
            patch("agilerl.distributed.runtime.get_world_size", return_value=2),
            patch(
                "agilerl.utils.llm_utils.make_llm_optimizer",
                return_value=MagicMock(),
            ),
            patch("agilerl.utils.llm_utils.make_llm_scheduler", return_value=None),
        ):
            FSDPRuntime(
                FSDPConfig(reshard_after_forward=True, ep=2, optim_cpu_offload=False)
            ).prepare_actor(
                actor,
                device="cpu",
                colocated=False,
                cosine_lr_schedule_config=None,
                lr=1e-4,
                lr_critic=None,
                restore_adapter_trainability=restore,
            )

        restore.assert_called_once_with(["actor", "critic"])
        validate.assert_called_once_with(actor, 2, 2)
