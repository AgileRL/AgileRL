# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for FSDP2 wrap, gather, and optimizer offload."""

from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.distributed.tensor import Replicate, Shard

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.grpo import GRPO
from agilerl.algorithms.ppo_llm import PPO
from agilerl.algorithms.reinforce_llm import REINFORCE
from agilerl.distributed import CPUOffloadOptimizer, FSDPConfig
from agilerl.distributed.runtime import DPRuntime, FSDPRuntime

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for device-move path"
)


# ---------------------------------------------------------------------------
# 2. wrap_models offload validation
# ---------------------------------------------------------------------------


def _make_wrap_stub(**overrides) -> MagicMock:
    """Minimal agent stub for ``LLMAlgorithm.wrap_models``."""
    agent = MagicMock()
    agent.actor = nn.Linear(2, 2)
    agent.distributed = True
    agent.fsdp_config = FSDPConfig()
    agent.colocated = False
    agent.ep = 1
    agent.cp = 1
    agent.gradient_checkpointing = False
    agent.cosine_lr_schedule_config = None
    agent.lr = 1e-4
    agent.optimizer = MagicMock()
    agent.optimizer._single_optimizer.return_value = torch.optim.AdamW(
        [torch.tensor([1.0], requires_grad=True)], lr=1e-3
    )
    agent.lr_critic = None
    agent.device = "cpu"
    agent.lr_scheduler = None
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


class TestLLMWrapModelsOffloadValidation:
    """Config-time validation of FSDP2 offload flags in ``wrap_models``."""

    def test_wrap_raises_when_cpu_offload_and_optim_cpu_offload_both_set(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=True),
            colocated=True,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="mutually exclusive"):
            LLMAlgorithm.wrap_models(agent)

    def test_wrap_raises_when_cpu_offload_without_vllm(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
            colocated=False,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="requires colocated vLLM"):
            LLMAlgorithm.wrap_models(agent)

    def test_wrap_accepts_cpu_offload_with_vllm(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
            colocated=True,
        )

        # Act
        with patch(
            "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
            side_effect=lambda m, _d, _c, **_k: m,
        ):
            LLMAlgorithm.wrap_models(agent)

        # Assert
        assert agent.actor is not None

    def test_wrap_wraps_inner_optimizer_when_optim_cpu_offload(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(optim_cpu_offload=True),
            colocated=False,
        )

        # Act
        with patch(
            "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
            side_effect=lambda m, _d, _c, **_k: m,
        ):
            LLMAlgorithm.wrap_models(agent)

        # Assert
        assert isinstance(agent.optimizer.optimizer, CPUOffloadOptimizer)

    def test_wrap_does_not_wrap_optimizer_when_no_offload(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(optim_cpu_offload=False), colocated=False
        )

        # Act
        with patch(
            "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
            side_effect=lambda m, _d, _c, **_k: m,
        ):
            LLMAlgorithm.wrap_models(agent)

        # Assert
        assert not isinstance(agent.optimizer.optimizer, CPUOffloadOptimizer)


class TestUntiedLmHeadNoReshard:
    """Untied ``lm_head`` is its own FSDP unit and stays gathered after forward."""

    def test_untied_lm_head_fully_shard_disables_reshard_after_forward(self):
        from agilerl.distributed.fsdp import _shard_embed_and_lm_head

        class LanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(4, 2)
                self.norm = nn.LayerNorm(2)

        class CausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("Cfg", (), {"tie_word_embeddings": False})()
                self.model = LanguageModel()
                self.lm_head = nn.Linear(2, 4, bias=False)

        model = CausalLM()
        seen: list[tuple[nn.Module, dict]] = []

        def _record(module, **kwargs):
            seen.append((module, kwargs))
            return module

        with patch("agilerl.distributed.fsdp.fully_shard", side_effect=_record):
            _shard_embed_and_lm_head(
                model, {"reshard_after_forward": True}, persistence_threshold=0
            )

        modules = [module for module, _ in seen]
        assert model.lm_head in modules
        assert model.model.embed_tokens not in modules
        assert model.model.norm not in modules
        head_kwargs = next(kwargs for module, kwargs in seen if module is model.lm_head)
        assert head_kwargs["reshard_after_forward"] is False


# ---------------------------------------------------------------------------
# 4. gather_layer
# ---------------------------------------------------------------------------


def _linear_like(weight: torch.Tensor, bias: torch.Tensor | None) -> SimpleNamespace:
    return SimpleNamespace(weight=weight, bias=bias)


class TestGatherLayer:
    """FSDP2 ``gather_layer`` materialises DTensors and moves them onto device."""

    def test_passthrough_yields_original_weight_when_dense(self):
        # Arrange
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        layer = _linear_like(weight, bias)
        hidden = torch.randn(2, 8)

        # Act
        with DPRuntime().gather_layer(layer, device=hidden.device) as (w, b):
            # Assert
            assert w is weight
            assert b is bias

    def test_materializes_when_fsdp_without_cpu_offload(self):
        # Arrange — plain tensors: materialize is a no-op identity
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        layer = _linear_like(weight, bias)
        runtime = FSDPRuntime(FSDPConfig(cpu_offload=False))
        hidden = torch.randn(2, 8)
        called = []

        def tracking_materialize(*tensors):
            called.append(tensors)
            return _fake_materialize(tensors)

        # Act
        with patch(
            "agilerl.distributed.runtime.materialize_dtensors",
            side_effect=tracking_materialize,
        ):
            with runtime.gather_layer(layer, device=hidden.device) as (w, b):
                # Assert
                assert w is weight
                assert b is bias
        assert called == [(weight, bias)]

    def test_materializes_and_moves_weight_to_device_under_cpu_offload(self):
        # Arrange — weight on CPU, target a (possibly same) compute device
        cpu_weight = torch.randn(4, 8)
        cpu_bias = torch.randn(4)
        layer = _linear_like(cpu_weight, cpu_bias)
        runtime = FSDPRuntime(FSDPConfig(cpu_offload=True, optim_cpu_offload=False))
        device = torch.device("cpu")

        # Act — patch materialize_dtensors to yield the plain CPU tensors
        with patch(
            "agilerl.distributed.runtime.materialize_dtensors",
            side_effect=lambda *tensors: _fake_materialize(tensors),
        ):
            with runtime.gather_layer(layer, device=device) as (w, b):
                assert w.device == device
                assert b.device == device

    def test_no_move_when_weight_already_on_device(self):
        # Arrange — weight already on the target device
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        layer = _linear_like(weight, bias)
        runtime = FSDPRuntime(FSDPConfig(cpu_offload=True, optim_cpu_offload=False))
        move_calls = []
        original_to = torch.Tensor.to

        def tracking_to(self_tensor, device, **kwargs):
            move_calls.append(device)
            return original_to(self_tensor, device, **kwargs)

        device = torch.device("cpu")

        # Act
        with (
            patch(
                "agilerl.distributed.runtime.materialize_dtensors",
                side_effect=lambda *tensors: _fake_materialize(tensors),
            ),
            patch("torch.Tensor.to", tracking_to),
        ):
            with runtime.gather_layer(layer, device=device) as (_w, _b):
                assert len(move_calls) == 0

    def test_handles_none_bias(self):
        # Arrange — bias is None
        weight = torch.randn(4, 8)
        layer = _linear_like(weight, None)
        runtime = FSDPRuntime(FSDPConfig(cpu_offload=True, optim_cpu_offload=False))
        device = torch.device("cpu")

        # Act
        with patch(
            "agilerl.distributed.runtime.materialize_dtensors",
            side_effect=lambda *tensors: _fake_materialize(tensors),
        ):
            with runtime.gather_layer(layer, device=device) as (w, b):
                assert w.device == device
                assert b is None


def _fake_materialize(tensors):
    """Context manager simulating ``materialize_dtensors``: yields the
    input tensors as-is (already plain CPU tensors in tests).
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        yield list(tensors)

    return _ctx()


# ---------------------------------------------------------------------------
# 3. actor_device guard in get_action (GRPO / PPO / REINFORCE)
# ---------------------------------------------------------------------------


def _make_actor_agent(shard_runtime=None, device="cuda:0"):
    """MagicMock agent for ``get_action`` HF-generate path tests.

    ``device`` is set to a CUDA string even on CPU-only hosts so the FSDP2
    branch (``fallback`` = ``self.device``) and the dense branch
    (param device = CPU) produce distinguishable devices.
    ``prepare_prompt_hf_generate`` is patched by the caller to capture the
    device without actually moving tensors.
    """
    agent = MagicMock()
    agent.colocated = False
    agent.device = device
    agent.shard_runtime = shard_runtime or DPRuntime()
    agent.hf_generate_chunk_size = 1
    agent.group_size = 1
    agent.pad_token_id = 0
    agent.vllm_importance_sampling_correction = False
    agent.actor = nn.Linear(2, 2)
    agent.actor.generate = MagicMock(return_value=torch.ones(1, 8, dtype=torch.long))
    return agent


def _dummy_prompts():
    return [
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
    ]


@contextmanager
def _patch_hf_generate_path(module_path, captured_devices):
    """Patch the HF-generate helpers to capture ``actor_device``
    and return dummy data without moving tensors.
    """

    def capture_prepare(prompt_dict, device):
        captured_devices.append(device)
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

    with (
        patch(f"{module_path}.prepare_prompt_hf_generate", side_effect=capture_prepare),
        patch(f"{module_path}.hf_turn_generation_config"),
        patch(f"{module_path}.hf_completion_lengths"),
        patch(
            f"{module_path}.build_completion_mask",
            return_value=torch.ones(1, 8, dtype=torch.bool),
        ),
    ):
        yield


class TestGRPOGetActionActorDevice:
    """``actor_device`` guard: FSDP2 uses ``self.device``, non-FSDP2 probes params."""

    def test_uses_self_device_when_fsdp_runtime(self):
        # Arrange — FSDP2 branch: actor_device = fallback (self.device)
        agent = _make_actor_agent(
            shard_runtime=FSDPRuntime(FSDPConfig()), device="cuda:0"
        )
        captured: list = []

        # Act
        with _patch_hf_generate_path("agilerl.algorithms.grpo", captured):
            GRPO.get_action(agent, _dummy_prompts(), training=False)

        # Assert — prompts sent to self.device, not the CPU param device
        assert captured[0] == torch.device("cuda:0")

    def test_uses_param_device_when_dense_runtime(self):
        # Arrange — dense branch: actor_device = next(params).device
        agent = _make_actor_agent(device="cuda:0")
        captured: list = []

        # Act
        with _patch_hf_generate_path("agilerl.algorithms.grpo", captured):
            GRPO.get_action(agent, _dummy_prompts(), training=False)

        # Assert — prompts sent to the param device (CPU), not self.device
        assert captured[0] == torch.device("cpu")


class TestPPOGetActionActorDevice:
    def test_uses_self_device_when_fsdp_runtime(self):
        # Arrange
        agent = _make_actor_agent(
            shard_runtime=FSDPRuntime(FSDPConfig()), device="cuda:0"
        )
        captured: list = []

        # Act
        with _patch_hf_generate_path("agilerl.algorithms.ppo_llm", captured):
            PPO.get_action(agent, _dummy_prompts(), training=False)

        # Assert
        assert captured[0] == torch.device("cuda:0")


class TestREINFORCEGetActionActorDevice:
    def test_uses_self_device_when_fsdp_runtime(self):
        # Arrange
        agent = _make_actor_agent(
            shard_runtime=FSDPRuntime(FSDPConfig()), device="cuda:0"
        )
        captured: list = []

        # Act
        with _patch_hf_generate_path("agilerl.algorithms.reinforce_llm", captured):
            REINFORCE.get_action(agent, _dummy_prompts(), training=False)

        # Assert
        assert captured[0] == torch.device("cuda:0")


# ---------------------------------------------------------------------------
# 5. load_lora_adapters — DTensor scatter into sharded params
# ---------------------------------------------------------------------------


class TestLoadLoraAdapters:
    """``load_lora_adapters`` scatters full tensors into DTensor local shards."""

    def test_scatters_full_tensor_into_dtensor_param(self, tmp_path, monkeypatch):
        from agilerl.utils.llm_utils import load_lora_adapters

        class FakeDTensor(nn.Parameter):
            def to_local(self) -> torch.Tensor:
                return self.data

        monkeypatch.setattr("agilerl.utils.llm_utils.DTensor", FakeDTensor)

        # Arrange — a fake adapter param that looks like a DTensor
        param = FakeDTensor(torch.zeros(2, 4))
        param.device_mesh = MagicMock()
        param.device_mesh.device_type = "cpu"
        param.placements = (None,)

        model = MagicMock()
        model.named_parameters.return_value = [
            ("base_model.model.layer.lora_A.actor.weight", param),
        ]

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()
        full_tensor = torch.randn(2, 4)
        from safetensors.torch import save_file

        save_file(
            {"base_model.model.layer.lora_A.weight": full_tensor},
            str(adapter_dir / "adapter_model.safetensors"),
        )

        # Act
        sharded = MagicMock()
        sharded.to_local.return_value = full_tensor
        with patch("agilerl.utils.llm_utils.distribute_tensor", return_value=sharded):
            load_lora_adapters(model, str(tmp_path), "actor", device="cpu")

        # Assert — local shard was overwritten with the scattered value
        assert torch.equal(param.data, full_tensor)

    def test_copies_directly_for_plain_tensor(self, tmp_path):
        from agilerl.utils.llm_utils import load_lora_adapters

        # Arrange — a plain (non-DTensor) param
        param = nn.Parameter(torch.zeros(2, 4))
        # Ensure no device_mesh attribute so it takes the plain path
        assert not hasattr(param, "device_mesh")

        model = MagicMock()
        model.named_parameters.return_value = [
            ("base_model.model.layer.lora_A.actor.weight", param),
        ]

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()
        full_tensor = torch.randn(2, 4)
        from safetensors.torch import save_file

        save_file(
            {"base_model.model.layer.lora_A.weight": full_tensor},
            str(adapter_dir / "adapter_model.safetensors"),
        )

        # Act
        load_lora_adapters(model, str(tmp_path), "actor", device="cpu")

        # Assert
        assert torch.equal(param.data, full_tensor)

    def test_skips_non_adapter_params(self, tmp_path):
        from agilerl.utils.llm_utils import load_lora_adapters

        # Arrange — a non-adapter param that should not be touched
        base_param = nn.Parameter(torch.randn(2, 4))
        lora_param = nn.Parameter(torch.zeros(2, 4))
        model = MagicMock()
        model.named_parameters.return_value = [
            ("base_model.model.layer.weight", base_param),
            ("base_model.model.layer.lora_A.actor.weight", lora_param),
        ]

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()
        original_base = base_param.data.clone()
        from safetensors.torch import save_file

        save_file(
            {"base_model.model.layer.lora_A.weight": torch.randn(2, 4)},
            str(adapter_dir / "adapter_model.safetensors"),
        )

        # Act
        load_lora_adapters(model, str(tmp_path), "actor", device="cpu")

        # Assert — base param unchanged
        assert torch.equal(base_param.data, original_base)


# ---------------------------------------------------------------------------
# 6. import_optimizer_state — manual DTensor-aware load
# ---------------------------------------------------------------------------


class TestLoadGatheredOptimizerStateDict:
    """``import_optimizer_state`` manually scatters optimizer
    state into DTensor shards instead of calling ``set_optimizer_state_dict``.
    """

    def test_non_sharded_uses_load_state_dict(self):
        agent = _make_llm_agent_for_ckpt()
        agent.optimizer = MagicMock()

        agent.shard_runtime.import_optimizer_state(
            agent.actor, agent.optimizer, {"state": {}}
        )

        agent.optimizer.load_state_dict.assert_called_once_with({"state": {}})

    def test_sharded_does_not_call_set_optimizer_state_dict(self):
        agent = _make_llm_agent_for_ckpt()

        inner_opt = MagicMock()
        inner_opt.state = {}
        inner_opt.param_groups = [{"params": [], "lr": 0.001}]
        agent.optimizer = MagicMock()
        agent.optimizer._single_optimizer.return_value = inner_opt

        set_opt_mock = MagicMock()

        with (
            patch(
                "torch.distributed.checkpoint.state_dict.set_optimizer_state_dict",
                set_opt_mock,
            ),
            patch("agilerl.distributed.runtime.FSDPModule", object),
        ):
            agent.shard_runtime.import_optimizer_state(
                agent.actor, agent.optimizer, {"state": {}}
            )

        # Assert — the old DCP path is NOT used
        set_opt_mock.assert_not_called()

    def test_sharded_clears_stale_state_before_load(self):
        agent = _make_llm_agent_for_ckpt()

        # A trainable DTensor-like param
        param = MagicMock()
        param.requires_grad = True
        param.device_mesh = MagicMock()
        param.device_mesh.device_type = "cpu"
        param.placements = (None,)
        param.device = torch.device("cpu")

        inner_opt = MagicMock()
        inner_opt.state = {param: {"step": torch.tensor(999.0)}}
        inner_opt.param_groups = [{"params": [param], "lr": 0.001}]
        agent.optimizer = MagicMock()
        agent.optimizer._single_optimizer.return_value = inner_opt

        saved_state = {
            "state": {
                "layer.lora_A.actor.weight": {
                    "step": torch.tensor(1.0),
                    "exp_avg": torch.randn(2, 4),
                    "exp_avg_sq": torch.randn(2, 4),
                }
            },
            "param_groups": [{"lr": 0.001}],
        }

        actor = agent.actor
        actor.named_parameters.return_value = [
            ("layer.lora_A.actor.weight", param),
        ]

        with (
            patch("agilerl.distributed.runtime.distribute_tensor") as mock_dist,
            patch("agilerl.distributed.runtime.FSDPModule", object),
        ):
            mock_dist.return_value = MagicMock(_local_tensor=torch.randn(2, 4))
            agent.shard_runtime.import_optimizer_state(
                agent.actor, agent.optimizer, saved_state
            )

        # Assert — stale state was cleared and replaced with checkpoint state
        assert param in inner_opt.state
        assert inner_opt.state[param]["step"].item() == 1.0

    def test_sharded_matches_checkpoint_wrapper_fqn_to_canonical_saved_key(self):
        agent = _make_llm_agent_for_ckpt()

        param = MagicMock()
        param.requires_grad = True
        param.device_mesh = MagicMock()
        param.device_mesh.device_type = "cpu"
        param.placements = (None,)
        param.device = torch.device("cpu")

        inner_opt = MagicMock()
        inner_opt.state = {}
        inner_opt.param_groups = [{"params": [param], "lr": 0.001}]
        agent.optimizer = MagicMock()
        agent.optimizer._single_optimizer.return_value = inner_opt
        agent.actor.named_parameters.return_value = [
            (
                "h.0._checkpoint_wrapped_module.attn.lora_A.actor.weight",
                param,
            ),
        ]
        saved_state = {
            "state": {
                "h.0.attn.lora_A.actor.weight": {
                    "step": torch.tensor(4.0),
                    "exp_avg": torch.randn(2, 4),
                    "exp_avg_sq": torch.randn(2, 4),
                }
            }
        }

        with (
            patch("agilerl.distributed.runtime.distribute_tensor") as mock_dist,
            patch("agilerl.distributed.runtime.FSDPModule", object),
        ):
            mock_dist.return_value = MagicMock(_local_tensor=torch.randn(2, 4))
            agent.shard_runtime.import_optimizer_state(
                agent.actor, agent.optimizer, saved_state
            )

        assert param in inner_opt.state
        assert inner_opt.state[param]["step"].item() == 4.0

    def test_sharded_raises_when_trainable_param_missing_from_saved_state(self):
        agent = _make_llm_agent_for_ckpt()

        param = MagicMock()
        param.requires_grad = True
        param.device = torch.device("cpu")

        inner_opt = MagicMock()
        inner_opt.state = {}
        inner_opt.param_groups = [{"params": [param], "lr": 0.001}]
        agent.optimizer = MagicMock()
        agent.optimizer._single_optimizer.return_value = inner_opt
        agent.actor.named_parameters.return_value = [
            ("layer.lora_A.actor.weight", param),
        ]

        with (
            patch("agilerl.distributed.runtime.FSDPModule", object),
            pytest.raises(
                RuntimeError,
                match="missing state for trainable parameter",
            ),
        ):
            agent.shard_runtime.import_optimizer_state(
                agent.actor, agent.optimizer, {"state": {}}
            )


# ---------------------------------------------------------------------------
# 8. export_optimizer_state gathers onto CPU
# ---------------------------------------------------------------------------


class TestGatheredOptimizerStateDictCpuOffload:
    """``export_optimizer_state`` gathers FSDP2 state onto CPU."""

    def test_gathers_full_state_on_cpu(self):
        agent = _make_llm_agent_for_ckpt()
        agent.optimizer = MagicMock()
        agent.optimizer.optimizer = MagicMock()

        with (
            patch("agilerl.distributed.runtime.get_optimizer_state_dict") as mock_get,
            patch("agilerl.distributed.runtime.FSDPModule", object),
        ):
            agent.shard_runtime.export_optimizer_state(agent.actor, agent.optimizer)

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert kwargs["options"].cpu_offload is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 9. No full-model-on-GPU residency guards
# ---------------------------------------------------------------------------


class TestFsdpResidencyGuards:
    """Control-flow guards for the no-full-GPU-model invariant."""

    def test_wrap_uses_materialize_helper(self):
        agent = _make_wrap_stub(device="cuda:0")
        with patch(
            "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
            side_effect=lambda m, _d, _c, **_k: m,
        ) as mock_mat:
            LLMAlgorithm.wrap_models(agent)
        mock_mat.assert_called_once_with(
            agent.actor,
            "cuda:0",
            agent.fsdp_config,
            gradient_checkpointing=False,
        )

    def test_fsdp_clone_uses_cpu_state_path(self):
        agent = MagicMock()
        agent.fsdp_config = FSDPConfig()
        agent.quantization_config = None
        clone = MagicMock()
        clone.mutation_hook = MagicMock()
        clone.wrap_models = MagicMock()
        agent._create_clone_instance = MagicMock(return_value=clone)
        agent._copy_clone_attributes = MagicMock(return_value=clone)
        agent._restore_clone_optimizer_and_scheduler = MagicMock()
        agent._clone_actor_network = MagicMock()

        with patch("agilerl.algorithms.core.base.barrier"):
            out = LLMAlgorithm.clone(agent, index=1)

        assert out is clone
        agent._create_clone_instance.assert_called_once()
        clone.wrap_models.assert_called_once()
        agent._restore_clone_optimizer_and_scheduler.assert_called_once_with(clone)

    def test_quantized_clone_rebuilds_from_adapters(self):
        agent = MagicMock()
        agent.quantization_config = object()
        agent.use_value_head = False
        clone = MagicMock()
        clone.mutation_hook = MagicMock()
        clone.wrap_models = MagicMock()
        clone._load_clone_adapter_weights = MagicMock()
        agent._resolve_clone_work_dir = MagicMock(return_value="/tmp/clone")
        agent._save_clone_adapter_weights = MagicMock()
        agent._create_clone_instance = MagicMock(return_value=clone)
        agent._copy_clone_attributes = MagicMock(return_value=clone)
        agent._restore_clone_optimizer_and_scheduler = MagicMock()

        with (
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.tempfile.TemporaryDirectory"
            ) as mock_td,
        ):
            mock_td.return_value.__enter__.return_value = "/tmp/local"
            mock_td.return_value.__exit__.return_value = False
            out = LLMAlgorithm.clone(agent, index=2)

        assert out is clone
        agent._save_clone_adapter_weights.assert_called_once_with("/tmp/clone")
        clone._load_clone_adapter_weights.assert_called_once_with("/tmp/clone")
        clone.wrap_models.assert_called_once()
        agent._restore_clone_optimizer_and_scheduler.assert_called_once_with(clone)


class TestCloneLlmRejectsFsdp:
    def test_raises_when_fsdp_source_lacks_state_dict(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        from agilerl.utils.algo_utils import clone_llm

        model = GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=32)
        )
        with (
            patch("agilerl.utils.algo_utils.FSDPModule", object),
            pytest.raises(RuntimeError, match="CPU state_dict"),
        ):
            clone_llm(model)


class TestFsdpSafetensorsShardHelpers:
    def test_global_shard_slices_match_torch_chunk_divisible(self):
        from torch.distributed.tensor.placement_types import Shard

        from agilerl.distributed.fsdp import global_shard_slices

        global_shape = (8, 4)
        placements = (Shard(0),)
        tensor = torch.arange(32).reshape(8, 4)
        mesh = MagicMock()
        mesh.size.return_value = 4
        for rank in range(4):
            mesh.get_coordinate.return_value = (rank,)
            slices = global_shard_slices(global_shape, placements, mesh)
            expected = torch.chunk(tensor, 4, dim=0)[rank]
            assert torch.equal(tensor[slices], expected)

    def test_global_shard_slices_match_torch_chunk_remainder(self):
        from torch.distributed.tensor.placement_types import Shard

        from agilerl.distributed.fsdp import global_shard_slices

        global_shape = (10,)
        placements = (Shard(0),)
        tensor = torch.arange(10)
        mesh = MagicMock()
        mesh.size.return_value = 3
        chunks = list(torch.chunk(tensor, 3, dim=0))
        for rank in range(3):
            mesh.get_coordinate.return_value = (rank,)
            slices = global_shard_slices(global_shape, placements, mesh)
            assert torch.equal(tensor[slices], chunks[rank])

    def test_global_shard_slices_two_dimensional_shard(self):
        from torch.distributed.tensor.placement_types import Shard

        from agilerl.distributed.fsdp import global_shard_slices

        global_shape = (6, 4)
        placements = (Shard(0), Shard(1))
        tensor = torch.arange(24).reshape(6, 4)
        mesh = MagicMock()
        mesh.size.return_value = 2
        for row_rank in range(2):
            for col_rank in range(2):
                mesh.get_coordinate.return_value = (row_rank, col_rank)
                slices = global_shard_slices(global_shape, placements, mesh)
                row_piece = torch.chunk(tensor, 2, dim=0)[row_rank]
                expected = torch.chunk(row_piece, 2, dim=1)[col_rank]
                assert torch.equal(tensor[slices], expected)

    def test_copy_safetensors_local_slice_with_dtype_cast(self, tmp_path):
        from safetensors import safe_open
        from safetensors.torch import save_file

        from agilerl.distributed.fsdp import _copy_safetensors_slice

        weights = {"layer.weight": torch.ones(6, 4, dtype=torch.float32)}
        path = tmp_path / "model.safetensors"
        save_file(weights, str(path))
        dest = torch.empty(2, 4, dtype=torch.bfloat16)
        path_str = str(path)
        with safe_open(path_str, framework="pt", device="cpu") as handle:
            _copy_safetensors_slice(
                handle,
                "layer.weight",
                (slice(2, 4), slice(0, 4)),
                dest,
            )
        assert dest.dtype == torch.bfloat16
        assert torch.all(dest == torch.ones(2, 4, dtype=torch.bfloat16))

    def test_lora_a_seed_stable_and_lora_b_zeros(self):
        from agilerl.distributed.fsdp import _init_lora_parameter

        param_a = nn.Parameter(torch.empty(2, 3))
        param_b = nn.Parameter(torch.empty(2, 3))
        name = "layers.0.q_proj.lora_A.default.weight"
        _init_lora_parameter(
            param_a,
            name,
            (2, 3),
            (),
            None,
        )
        _init_lora_parameter(
            param_b,
            name.replace("lora_A", "lora_B"),
            (2, 3),
            (),
            None,
        )
        param_a_other = nn.Parameter(torch.empty(2, 3))
        _init_lora_parameter(
            param_a_other,
            name,
            (2, 3),
            (),
            None,
        )
        assert torch.equal(param_a, param_a_other)
        assert torch.equal(param_b, torch.zeros_like(param_b))

    def test_checkpoint_key_maps_peft_base_layer(self):
        from agilerl.distributed.fsdp import checkpoint_key_for_parameter

        live = "base_model.model.layers.0.q_proj.base_layer.weight"
        assert checkpoint_key_for_parameter(live) == "layers.0.q_proj.weight"

    def test_resolve_checkpoint_directory_outer_module_config(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file
        from transformers import GPT2Config, GPT2LMHeadModel

        from agilerl.distributed.fsdp import _resolve_checkpoint_source

        save_file({"w": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
        causal = GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_embd=8, n_head=2, vocab_size=16)
        )
        object.__setattr__(causal.config, "_name_or_path", str(tmp_path))
        language = nn.Module()
        language.config = SimpleNamespace(_name_or_path="")
        causal.language_model = language
        shell = nn.Module()
        shell.base_model = causal

        assert _resolve_checkpoint_source(shell) == str(tmp_path)

    def test_tied_weight_target_missing_from_checkpoint_skips(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        class TinyCausal(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wte = nn.Embedding(16, 8)
                self.lm_head = nn.Linear(8, 16, bias=False)
                self.config = SimpleNamespace(_name_or_path="")
                self.all_tied_weights_keys = {"lm_head.weight": "wte.weight"}

        model = TinyCausal()
        source = model.wte.weight.detach().clone()
        source.fill_(1.5)
        save_file({"wte.weight": source}, str(tmp_path / "model.safetensors"))
        model.config._name_or_path = str(tmp_path)
        head_before = model.lm_head.weight.detach().clone()
        _load_sharded_weights_from_safetensors(model)

        assert torch.allclose(model.wte.weight, torch.full_like(source, 1.5))
        assert torch.equal(model.lm_head.weight, head_before)

    def test_tied_word_embeddings_skip_a_missing_lm_head(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        class TinyCausal(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(4, 2)
                self.lm_head = nn.Linear(2, 4, bias=False)
                self.config = SimpleNamespace(
                    tie_word_embeddings=True,
                    _name_or_path="",
                )

        model = TinyCausal()
        source = torch.full((4, 2), 1.5)
        save_file(
            {"embed_tokens.weight": source},
            str(tmp_path / "model.safetensors"),
        )
        model.config._name_or_path = str(tmp_path)
        head_before = model.lm_head.weight.detach().clone()

        _load_sharded_weights_from_safetensors(model)

        assert torch.equal(model.embed_tokens.weight.detach(), source)
        assert torch.equal(model.lm_head.weight.detach(), head_before)

    def test_missing_value_head_is_initialized(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        class Summary(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.summary = nn.Linear(2, 1)

        class TinyCausal(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(4, 2)
                self.v_head = Summary()
                self.config = SimpleNamespace(_name_or_path="")

        model = TinyCausal()
        source = torch.full((4, 2), 1.5)
        save_file(
            {"embed_tokens.weight": source},
            str(tmp_path / "model.safetensors"),
        )
        model.config._name_or_path = str(tmp_path)

        _load_sharded_weights_from_safetensors(model)

        assert torch.equal(model.embed_tokens.weight.detach(), source)
        assert torch.isfinite(model.v_head.summary.weight.detach()).all()
        assert torch.count_nonzero(model.v_head.summary.weight.detach()) > 0
        assert torch.equal(
            model.v_head.summary.bias.detach(),
            torch.zeros_like(model.v_head.summary.bias),
        )

    def test_missing_non_lora_checkpoint_key_raises(self, tmp_path):
        from safetensors.torch import save_file
        from transformers import GPT2Config, GPT2LMHeadModel

        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        save_file(
            {"layers.0.weight": torch.zeros(2)}, str(tmp_path / "model.safetensors")
        )
        model = GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_embd=8, n_head=2, vocab_size=16)
        )
        model.config._name_or_path = str(tmp_path)
        with pytest.raises(RuntimeError, match="Missing checkpoint weight"):
            _load_sharded_weights_from_safetensors(model)


class TestMaterializeFsdp2FromCpuState:
    """``materialize_fsdp2_from_cpu_state`` fills FSDP shards from meta or a dense module."""

    def test_dense_model_scatters_its_parameters(self):
        from agilerl.distributed import materialize_fsdp2_from_cpu_state

        torch.manual_seed(0)
        model = nn.Linear(4, 4)
        expected = model.weight.detach().cpu().clone()

        with (
            patch(
                "agilerl.distributed.fsdp.apply_fsdp2",
                side_effect=lambda module, *_a, **_k: module,
            ),
            patch("agilerl.distributed.fsdp._restore_after_to_empty"),
            patch("agilerl.distributed.fsdp._share_fsdp_comm_streams"),
        ):
            out = materialize_fsdp2_from_cpu_state(
                model, "cpu", FSDPConfig(cpu_offload=True)
            )

        assert out is model
        assert torch.equal(model.weight.detach().cpu(), expected)

    def test_world_size_one_meta_loads_local_safetensors(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed import materialize_fsdp2_from_cpu_state

        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        save_file({"weight": expected}, str(tmp_path / "model.safetensors"))

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(2, 2))
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        with torch.device("meta"):
            model = Tiny()

        with (
            patch(
                "agilerl.distributed.fsdp.apply_fsdp2",
                side_effect=lambda module, *_a, **_k: module,
            ),
            patch("agilerl.distributed.fsdp._restore_after_to_empty"),
            patch("agilerl.distributed.fsdp._share_fsdp_comm_streams"),
        ):
            out = materialize_fsdp2_from_cpu_state(
                model, "cpu", FSDPConfig(cpu_offload=True)
            )

        assert out is model
        assert torch.equal(model.weight, expected)

    def test_hub_id_resolves_through_cached_file(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed import materialize_fsdp2_from_cpu_state

        hub_id = "org/tiny-checkpoint"
        expected = torch.full((2, 2), 7.0)
        weights_path = str(tmp_path / "model.safetensors")
        save_file({"weight": expected}, weights_path)

        def fake_cached_file(
            path_or_repo: str,
            filename: str,
            **_kwargs: object,
        ) -> str | None:
            if path_or_repo != hub_id:
                return None
            if filename == "model.safetensors":
                return weights_path
            return None

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(2, 2))
                self.config = SimpleNamespace(_name_or_path=hub_id)

        with torch.device("meta"):
            model = Tiny()

        with (
            patch(
                "agilerl.distributed.fsdp.cached_file",
                side_effect=fake_cached_file,
            ),
            patch(
                "agilerl.distributed.fsdp.apply_fsdp2",
                side_effect=lambda module, *_a, **_k: module,
            ),
            patch("agilerl.distributed.fsdp._restore_after_to_empty"),
            patch("agilerl.distributed.fsdp._share_fsdp_comm_streams"),
        ):
            materialize_fsdp2_from_cpu_state(model, "cpu", FSDPConfig(cpu_offload=True))

        assert torch.equal(model.weight, expected)

    def test_distributed_meta_model_loads_safetensor_shards(self):
        from types import SimpleNamespace

        from agilerl.distributed import materialize_fsdp2_from_cpu_state

        with torch.device("meta"):
            model = nn.Linear(4, 4)
        object.__setattr__(
            model,
            "config",
            SimpleNamespace(_name_or_path="/unused"),
        )
        loaded: list[str] = []
        with (
            patch("agilerl.distributed.fsdp.is_distributed", return_value=True),
            patch(
                "agilerl.distributed.fsdp.apply_fsdp2",
                side_effect=lambda module, *_a, **_k: module,
            ),
            patch(
                "agilerl.distributed.fsdp._load_sharded_weights_from_safetensors",
                side_effect=lambda *_a, **_k: loaded.append("shard_load"),
            ),
            patch("agilerl.distributed.fsdp._restore_after_to_empty"),
            patch("agilerl.distributed.fsdp._share_fsdp_comm_streams"),
        ):
            materialize_fsdp2_from_cpu_state(model, "cpu", FSDPConfig(cpu_offload=True))

        assert loaded == ["shard_load"]


class TestLoraConfigsEquivalent:
    """Checkpoint adapter_config.json round-trips None/empty and path fields."""

    def test_none_list_and_base_model_path_are_equivalent(self, tmp_path):
        from peft import LoraConfig

        live = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"],
            task_type="CAUSAL_LM",
        )
        ckpt_dir = tmp_path / "actor"
        ckpt_dir.mkdir()
        live.save_pretrained(str(ckpt_dir))
        loaded = LoraConfig.from_pretrained(str(ckpt_dir))
        assert LLMAlgorithm._lora_configs_equivalent(live, loaded)
        assert not LLMAlgorithm._lora_configs_equivalent(
            LoraConfig(r=4, target_modules=["q_proj"]), loaded
        )


class _CoordMesh:
    """Device mesh stand-in whose coordinate is unset."""

    def get_coordinate(self) -> None:
        return None

    def size(self, mesh_dim: int) -> int:
        return 2

    def get_local_rank(self, mesh_dim: int) -> int:
        return 0


class TestSafetensorsShardKeys:
    """Checkpoint key lookup and the safetensors shard copy."""

    def test_base_layer_bias_maps_to_bias(self):
        from agilerl.distributed.fsdp import checkpoint_key_for_parameter

        assert checkpoint_key_for_parameter("block.base_layer.bias") == "block.bias"

    def test_nested_base_layer_modules_drop_out_of_the_checkpoint_key(self):
        from agilerl.distributed.fsdp import checkpoint_key_for_parameter

        live = (
            "model.layers.0.block_sparse_moe.experts.base_layer.base_layer.gate_up_proj"
        )
        assert checkpoint_key_for_parameter(live) == (
            "model.layers.0.block_sparse_moe.experts.gate_up_proj"
        )

    def test_value_head_peft_fqn_maps_to_hf_embed_key(self):
        from agilerl.distributed.fsdp import checkpoint_key_for_parameter

        live = "pretrained_model.base_model.model.model.embed_tokens.weight"
        assert checkpoint_key_for_parameter(live) == "model.embed_tokens.weight"

    def test_loads_embed_tokens_through_value_head_peft_fqn(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        expected = torch.arange(8, dtype=torch.float32).reshape(4, 2)

        class NemotronBody(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_tokens = nn.Embedding(4, 2)

        class HFModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = NemotronBody()

        class PeftInner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = HFModel()

        class PeftWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = PeftInner()

        class ValueHeadShell(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pretrained_model = PeftWrapper()
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        save_file(
            {"model.embed_tokens.weight": expected},
            str(tmp_path / "model.safetensors"),
        )
        model = ValueHeadShell()
        _load_sharded_weights_from_safetensors(model)

        embed = model.pretrained_model.base_model.model.model.embed_tokens.weight
        assert torch.equal(embed.detach(), expected)

    def test_loads_granite_router_from_the_legacy_checkpoint_key(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)

        class Router(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, 2))

        class Moe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.router = Router()

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block_sparse_moe = Moe()

        class Body(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([Layer()])

        class HFModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Body()
                self.config = SimpleNamespace(model_type="granitemoe")

        class PeftInner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = HFModel()

        class PeftWrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_model = PeftInner()

        class ValueHeadShell(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pretrained_model = PeftWrapper()
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        save_file(
            {
                "model.layers.0.block_sparse_moe.router.layer.weight": expected,
            },
            str(tmp_path / "model.safetensors"),
        )
        model = ValueHeadShell()
        _load_sharded_weights_from_safetensors(model)

        router = model.pretrained_model.base_model.model.model.layers[
            0
        ].block_sparse_moe.router.weight
        assert torch.equal(router.detach(), expected)

    def test_loads_granite_packed_experts_through_nested_base_layers(self, tmp_path):
        from types import SimpleNamespace

        from safetensors.torch import save_file

        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        expected = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)

        class Packed(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.zeros(2, 2, 2))

        class Shell(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_layer = Packed()

        class Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_layer = Shell()

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block_sparse_moe = nn.Module()
                self.block_sparse_moe.experts = Experts()
                self.config = SimpleNamespace(
                    model_type="granitemoe",
                    _name_or_path=str(tmp_path),
                )

        save_file(
            {"block_sparse_moe.input_linear.weight": expected},
            str(tmp_path / "model.safetensors"),
        )
        model = Root()
        _load_sharded_weights_from_safetensors(model)

        loaded = model.block_sparse_moe.experts.base_layer.base_layer.gate_up_proj
        assert torch.equal(loaded.detach(), expected)

    def test_loads_a_split_source_and_a_renamed_buffer(self, tmp_path, monkeypatch):
        from transformers.conversion_mapping import (
            Chunk,
            WeightConverter,
            WeightRenaming,
        )

        from agilerl.distributed import fsdp as fsdp_mod

        query = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        key = query + 10
        value = query + 20
        scale = torch.tensor([3.0, 4.0])
        save_file(
            {
                "qkv.weight": torch.cat([query, key, value], dim=0),
                "old_scale": scale,
            },
            str(tmp_path / "model.safetensors"),
        )

        class Qkv(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.query = nn.Linear(2, 2, bias=False)
                self.key = nn.Linear(2, 2, bias=False)
                self.value = nn.Linear(2, 2, bias=False)

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attention = Qkv()
                self.register_buffer("scale", torch.zeros(2))
                self.register_buffer("ignored", torch.zeros(1))
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        transforms = [
            WeightRenaming("old_scale", "scale"),
            WeightConverter(
                source_patterns="qkv",
                target_patterns=[
                    "attention.query",
                    "attention.key",
                    "attention.value",
                ],
                operations=[Chunk(dim=0)],
            ),
        ]
        monkeypatch.setattr(
            fsdp_mod,
            "_checkpoint_transform_groups",
            lambda _model: [
                ("other", transforms),
                ("attention", [WeightRenaming("zzz", "yyy")]),
                ("", [WeightRenaming("zzz", "yyy")]),
                ("", transforms),
            ],
        )
        model = Root()

        fsdp_mod._load_sharded_weights_from_safetensors(model)

        assert torch.equal(model.attention.query.weight.detach(), query)
        assert torch.equal(model.attention.key.weight.detach(), key)
        assert torch.equal(model.attention.value.weight.detach(), value)
        assert torch.equal(model.scale.detach(), scale)

    def test_language_body_alias_uses_backbone_key(self):
        from agilerl.distributed.fsdp import checkpoint_key_candidates

        assert checkpoint_key_candidates("language_model.model.embeddings.weight") == (
            "language_model.model.embeddings.weight",
            "language_model.backbone.embeddings.weight",
        )

    def test_shard_slices_when_mesh_coordinate_is_unset(self):
        from agilerl.distributed.fsdp import global_shard_slices

        slices = global_shard_slices((4, 2), (Replicate(), Shard(0)), _CoordMesh())

        assert slices[0] == slice(0, 2)
        assert slices[1] == slice(0, 2)

    def test_local_dest_of_a_dtensor(self, monkeypatch: pytest.MonkeyPatch):
        from agilerl.distributed import fsdp as fsdp_mod

        local = torch.ones(2)

        class FakeDTensor(nn.Parameter):
            def to_local(self) -> torch.Tensor:
                return local

        monkeypatch.setattr(fsdp_mod, "DTensor", FakeDTensor)
        param = FakeDTensor(torch.zeros(2))

        assert fsdp_mod._parameter_dest_local(param) is local

    def test_lora_init_copies_this_ranks_slice(self, monkeypatch: pytest.MonkeyPatch):
        from agilerl.distributed import fsdp as fsdp_mod

        class FakeDTensor(nn.Parameter):
            def to_local(self) -> torch.Tensor:
                return self.data

        monkeypatch.setattr(fsdp_mod, "DTensor", FakeDTensor)
        param = FakeDTensor(torch.zeros(2, 2))

        fsdp_mod._init_lora_parameter(
            param,
            "block.lora_B.weight",
            (4, 2),
            (Shard(0),),
            _CoordMesh(),
        )

        assert torch.equal(param.data, torch.zeros(2, 2))

    def test_value_head_init_copies_this_ranks_slice(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from agilerl.distributed import fsdp as fsdp_mod

        class FakeDTensor(nn.Parameter):
            def to_local(self) -> torch.Tensor:
                return self.data

        monkeypatch.setattr(fsdp_mod, "DTensor", FakeDTensor)
        param = FakeDTensor(torch.zeros(2))

        fsdp_mod._init_value_head_parameter(
            param,
            "v_head.summary.bias",
            (4,),
            (Shard(0),),
            _CoordMesh(),
        )

        assert torch.equal(param.data, torch.zeros(2))

    def test_empty_config_path_is_not_a_checkpoint(self):
        from agilerl.distributed.fsdp import _checkpoint_source_from_config

        assert _checkpoint_source_from_config(SimpleNamespace()) is None

    def test_index_file_is_a_checkpoint_source(self, tmp_path):
        from agilerl.distributed.fsdp import _checkpoint_source_from_config

        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map": {}}', encoding="utf-8"
        )

        assert _checkpoint_source_from_config(
            SimpleNamespace(_name_or_path=str(tmp_path))
        ) == str(tmp_path)

    def test_missing_weight_files_are_not_a_checkpoint(self, tmp_path):
        from agilerl.distributed.fsdp import _checkpoint_source_from_config

        (tmp_path / "config.json").write_text("{}", encoding="utf-8")

        assert (
            _checkpoint_source_from_config(SimpleNamespace(_name_or_path=str(tmp_path)))
            is None
        )

    def test_unwraps_pretrained_base_and_inner_model(self):
        from agilerl.distributed.fsdp import _next_unwrap_module

        inner = nn.Linear(2, 2)
        pretrained = nn.Linear(2, 2)
        shell = nn.Linear(2, 2)
        shell.pretrained_model = pretrained
        assert _next_unwrap_module(shell) is pretrained

        class WithBase(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def get_base_model(self) -> nn.Module:
                return inner

        assert _next_unwrap_module(WithBase()) is inner

        wrapped = nn.Linear(2, 2)
        wrapped.base_model = inner
        assert _next_unwrap_module(wrapped) is inner

        holder = nn.Module()
        holder.model = inner
        assert _next_unwrap_module(holder) is inner
        assert _next_unwrap_module(nn.Linear(2, 2)) is None

    def test_resolve_raises_without_a_checkpoint(self):
        from agilerl.distributed.fsdp import _resolve_checkpoint_source

        with pytest.raises(RuntimeError, match="FSDP shard load requires"):
            _resolve_checkpoint_source(nn.Linear(2, 2))

    def test_index_maps_each_key_to_its_shard(self, tmp_path):
        from agilerl.distributed.fsdp import _build_safetensors_key_files

        shard = tmp_path / "model-00001-of-00001.safetensors"
        save_file({"weight": torch.ones(2)}, str(shard))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": shard.name}}),
            encoding="utf-8",
        )

        key_files = _build_safetensors_key_files(str(tmp_path))

        assert key_files == {"weight": str(shard)}

    def test_index_missing_shard_raises(self, tmp_path):
        from agilerl.distributed.fsdp import _build_safetensors_key_files

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "missing.safetensors"}}),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="missing shard"):
            _build_safetensors_key_files(str(tmp_path))

    def test_directory_without_weights_raises(self, tmp_path):
        from agilerl.distributed.fsdp import _build_safetensors_key_files

        with pytest.raises(RuntimeError, match=r"model\.safetensors"):
            _build_safetensors_key_files(str(tmp_path))

    def test_loads_backbone_alias_lora_tied_weight_and_buffer(self, tmp_path):
        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        save_file(
            {
                "language_model.backbone.embeddings.weight": expected,
                "language_model.scale": torch.tensor([3.0, 4.0]),
            },
            str(tmp_path / "model.safetensors"),
        )

        class Body(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embeddings = nn.Embedding(2, 2)
                self.lora_A = nn.Parameter(torch.empty(2, 2))

        class Language(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Body()
                self.all_tied_weights_keys = {
                    "language_model.tied": "embeddings.weight"
                }
                self.tied = nn.Parameter(torch.ones(2, 2))
                self.register_buffer("scale", torch.zeros(2))
                self.register_buffer("unused", torch.ones(2))

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = Language()
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        model = Root()
        _load_sharded_weights_from_safetensors(model)

        embed = model.language_model.model.embeddings.weight
        assert torch.equal(embed.detach(), expected)
        assert torch.equal(model.language_model.scale, torch.tensor([3.0, 4.0]))
        assert torch.equal(model.language_model.unused, torch.ones(2))
        assert torch.equal(model.language_model.tied, torch.ones(2, 2))
        assert torch.isfinite(model.language_model.model.lora_A).all()

    def test_leaves_parameters_outside_the_language_tower(self, tmp_path):
        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        save_file(
            {"language_model.weight": expected},
            str(tmp_path / "model.safetensors"),
        )

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = nn.Linear(2, 2, bias=False)
                self.vision_model = nn.Linear(2, 2, bias=False)
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        model = Root()
        with torch.no_grad():
            model.vision_model.weight.copy_(torch.ones(2, 2))
        _load_sharded_weights_from_safetensors(model)

        assert torch.equal(model.language_model.weight.detach(), expected)
        assert torch.equal(model.vision_model.weight, torch.ones(2, 2))

    def test_missing_key_raises_without_a_language_tower(self, tmp_path):
        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        save_file(
            {"other.weight": torch.ones(2, 2)},
            str(tmp_path / "model.safetensors"),
        )

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Linear(2, 2)
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        with pytest.raises(RuntimeError, match="Missing checkpoint weight"):
            _load_sharded_weights_from_safetensors(Root())

    def test_indexed_keys_skip_a_non_numeric_middle(self):
        from agilerl.distributed.fsdp import _contiguous_indexed_weight_keys

        assert _contiguous_indexed_weight_keys("up_proj", {}) is None
        assert _contiguous_indexed_weight_keys(
            "block.experts.up_proj",
            {
                "block.experts.extra.up_proj.weight": "shard-a",
                "block.experts.0.up_proj.weight": "shard-b",
            },
        ) == ["block.experts.0.up_proj.weight"]

    def test_stacks_indexed_expert_weights(self, tmp_path):
        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        first = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        second = torch.arange(4, 8, dtype=torch.float32).reshape(2, 2)
        save_file(
            {
                "language_model.backbone.layers.0.mixer.experts.0.up_proj.weight": first,
                "language_model.backbone.layers.0.mixer.experts.1.up_proj.weight": second,
            },
            str(tmp_path / "model.safetensors"),
        )

        class Projection(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.up_proj = nn.Parameter(torch.zeros(2, 2, 2))

        class Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_layer = Projection()

        class Experts(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_layer = Inner()

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mixer = nn.Module()
                self.mixer.experts = Experts()

        class Body(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([Layer()])

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.language_model = nn.Module()
                self.language_model.model = Body()
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        model = Root()
        _load_sharded_weights_from_safetensors(model)

        stacked = model.language_model.model.layers[
            0
        ].mixer.experts.base_layer.base_layer.up_proj
        assert torch.equal(stacked.detach(), torch.stack([first, second]))

    def test_indexed_expert_gap_raises(self, tmp_path):
        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        save_file(
            {
                "block.experts.0.up_proj.weight": torch.ones(2, 2),
                "block.experts.2.up_proj.weight": torch.ones(2, 2),
            },
            str(tmp_path / "model.safetensors"),
        )

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Module()
                self.block.experts = nn.Module()
                self.block.experts.up_proj = nn.Parameter(torch.zeros(2, 2, 2))
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        with pytest.raises(RuntimeError, match="not contiguous"):
            _load_sharded_weights_from_safetensors(Root())

    def test_stacked_shape_mismatch_raises(self, tmp_path):
        from agilerl.distributed.fsdp import _load_sharded_weights_from_safetensors

        save_file(
            {"block.experts.0.up_proj.weight": torch.ones(2, 2)},
            str(tmp_path / "model.safetensors"),
        )

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Module()
                self.block.experts = nn.Module()
                self.block.experts.up_proj = nn.Parameter(torch.zeros(3, 2, 2))
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        with pytest.raises(RuntimeError, match="does not match"):
            _load_sharded_weights_from_safetensors(Root())

    def test_stacks_indexed_expert_weights_on_a_dtensor(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        from agilerl.distributed import fsdp as fsdp_mod

        class FakeDTensor(nn.Parameter):
            def to_local(self) -> torch.Tensor:
                return self.data

        class OneChunkMesh:
            def get_coordinate(self) -> None:
                return None

            def size(self, mesh_dim: int) -> int:
                return 1

            def get_local_rank(self, mesh_dim: int) -> int:
                return 0

        monkeypatch.setattr(fsdp_mod, "DTensor", FakeDTensor)
        first = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        second = first + 4
        save_file(
            {
                "block.experts.0.up_proj.weight": first,
                "block.experts.1.up_proj.weight": second,
            },
            str(tmp_path / "model.safetensors"),
        )

        class Root(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.block = nn.Module()
                self.block.experts = nn.Module()
                self.block.experts.up_proj = FakeDTensor(torch.zeros(2, 2, 2))
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        model = Root()
        model.block.experts.up_proj.placements = (Shard(0),)
        model.block.experts.up_proj.device_mesh = OneChunkMesh()

        fsdp_mod._load_sharded_weights_from_safetensors(model)

        assert torch.equal(
            model.block.experts.up_proj.data, torch.stack([first, second])
        )

    def test_loads_a_dtensor_shard(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        from agilerl.distributed import fsdp as fsdp_mod

        class FakeDTensor(nn.Parameter):
            def to_local(self) -> torch.Tensor:
                return self.data

        monkeypatch.setattr(fsdp_mod, "DTensor", FakeDTensor)
        full = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        save_file({"weight": full}, str(tmp_path / "model.safetensors"))

        class OneChunkMesh:
            def get_coordinate(self) -> None:
                return None

            def size(self, mesh_dim: int) -> int:
                return 1

            def get_local_rank(self, mesh_dim: int) -> int:
                return 0

        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = FakeDTensor(torch.zeros(4, 2))
                self.config = SimpleNamespace(_name_or_path=str(tmp_path))

        model = Tiny()
        model.weight.placements = (Shard(0),)
        model.weight.device_mesh = OneChunkMesh()

        fsdp_mod._load_sharded_weights_from_safetensors(model)

        assert torch.equal(model.weight.data, full)

    def test_share_streams_returns_when_nothing_is_sharded(self):
        from agilerl.distributed.fsdp import _share_fsdp_comm_streams

        _share_fsdp_comm_streams(nn.Linear(2, 2))


def _make_llm_agent_for_ckpt() -> MagicMock:
    """Minimal agent stub for checkpoint load/optimizer tests."""
    agent = MagicMock()
    agent.use_value_head = False
    agent.selected_adapters = ("actor",)
    agent.device = "cpu"
    actor = MagicMock()
    actor.named_parameters.return_value = []
    agent.actor = actor
    agent.shard_runtime = FSDPRuntime(FSDPConfig())
    return agent
