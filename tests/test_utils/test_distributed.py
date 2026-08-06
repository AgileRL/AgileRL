"""Behavior tests for ``agilerl.utils.distributed``.

Classes are named ``Test<FunctionName>`` and assert API-visible outcomes
(return values, side effects, raised errors). ``fully_shard`` is stubbed
only as an expensive boundary; assertions still target which modules were
sharded and which config policies were requested, not call counts.
"""

from __future__ import annotations

import os
import sys
from typing import ClassVar
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn

from agilerl.utils import distributed as dmod
from agilerl.utils.distributed import (
    CPUOffloadOptimizer,
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

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for state round-trip"
)


class _FakeDTensor:
    """Lightweight DTensor stand-in: carries a ``_local_tensor`` and records
    ``full_tensor()`` calls so tests can assert DTensor-aware state movement
    without a real distributed tensor."""

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
            out = apply_fsdp2(
                model,
                FSDPConfig(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            )

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

        def _record(module, **_kwargs):
            sharded.append(module)
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

        # Arrange — tied: embed + blocks + root; skip separate lm_head unit
        tied = CausalLM(tie=True)
        sharded.clear()
        with patch.object(dmod, "fully_shard", side_effect=_record):
            apply_fsdp2(tied)

        # Assert
        assert tied.model.embed_tokens in sharded
        assert tied.lm_head not in sharded
        assert sharded[-1] is tied

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

        # Assert — bf16 params / fp32 reduce by default (Prime-RL style)
        assert seen_kwargs
        assert "mp_policy" in seen_kwargs[-1]

    def test_requests_mixed_precision_policy_when_dtypes_set(self, world_size_one):
        # Arrange
        model = nn.Linear(2, 2)
        seen_kwargs: list[dict] = []

        # Act
        with patch.object(
            dmod,
            "fully_shard",
            side_effect=lambda m, **kw: seen_kwargs.append(kw) or m,
        ):
            apply_fsdp2(
                model,
                FSDPConfig(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            )

        # Assert
        assert seen_kwargs
        assert "mp_policy" in seen_kwargs[-1]

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

        # Assert
        assert registered == [(model, "generate")]

    def test_raises_without_process_group(self):
        with pytest.raises(RuntimeError, match="initialised process group"):
            apply_fsdp2(nn.Linear(2, 2), FSDPConfig())

    def test_raises_when_fsdp2_unavailable(self):
        with (
            patch.object(dmod, "HAS_FSDP2", False),
            pytest.raises(RuntimeError, match="distributed support"),
        ):
            apply_fsdp2(nn.Linear(2, 2))


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
        model, opt = self._make_optimizer("cpu")
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
