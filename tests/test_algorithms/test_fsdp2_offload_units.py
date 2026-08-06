"""Behavior tests for the FSDP2 offload changes in commit a331bd6f.

Covers three units that have no test coverage in the existing suite:
  - ``wrap_models`` config-time validation of cpu_offload / optim_cpu_offload flags
  - ``_lm_head_matmul_ctx`` materialisation of the lm_head DTensor under cpu_offload
  - ``actor_device`` guard in GRPO / LLMPPO / LLMREINFORCE ``get_action``

The LLM algorithm test files have pre-existing import breakage, so these
tests live in a standalone module with minimal stubs at the API boundary.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from agilerl.algorithms.core.base import LLMAlgorithm
from agilerl.algorithms.grpo import GRPO
from agilerl.algorithms.ppo_llm import PPO
from agilerl.algorithms.reinforce_llm import REINFORCE
from agilerl.utils.distributed import CPUOffloadOptimizer, FSDPConfig

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
    agent.use_vllm = False
    agent.gradient_checkpointing = False
    agent.cosine_lr_schedule_config = None
    agent.lr = 1e-4
    agent.optimizer = MagicMock()
    agent.optimizer._single_optimizer.return_value = torch.optim.AdamW(
        [torch.tensor([1.0], requires_grad=True)], lr=1e-3
    )
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


class TestLLMWrapModelsOffloadValidation:
    """Config-time validation of FSDP2 offload flags in ``wrap_models``."""

    def test_wrap_raises_when_cpu_offload_and_optim_cpu_offload_both_set(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=True),
            use_vllm=True,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="mutually exclusive"):
            LLMAlgorithm.wrap_models(agent)

    def test_wrap_raises_when_cpu_offload_without_vllm(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
            use_vllm=False,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="requires vLLM"):
            LLMAlgorithm.wrap_models(agent)

    def test_wrap_accepts_cpu_offload_with_vllm(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
            use_vllm=True,
        )

        # Act
        with (
            patch(
                "agilerl.algorithms.core.base.materialize_fsdp2_from_cpu_state",
                side_effect=lambda m, _d, _c: m,
            ),
            patch.object(agent, "_rebuild_optimizer_after_load"),
        ):
            LLMAlgorithm.wrap_models(agent)

        # Assert
        assert agent.actor is not None

    def test_wrap_wraps_inner_optimizer_when_optim_cpu_offload(self):
        # Arrange
        inner = torch.optim.AdamW([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(optim_cpu_offload=True),
            use_vllm=False,
        )
        agent.optimizer._single_optimizer.return_value = inner

        # Act
        with (
            patch(
                "agilerl.algorithms.core.base.materialize_fsdp2_from_cpu_state",
                side_effect=lambda m, _d, _c: m,
            ),
            patch.object(agent, "_rebuild_optimizer_after_load"),
        ):
            LLMAlgorithm.wrap_models(agent)

        # Assert
        assert isinstance(agent.optimizer.optimizer, CPUOffloadOptimizer)
        assert agent.optimizer.optimizer.base_optimizer is inner

    def test_wrap_does_not_wrap_optimizer_when_no_offload(self):
        # Arrange
        agent = _make_wrap_stub(
            fsdp_config=FSDPConfig(optim_cpu_offload=False), use_vllm=False
        )
        original_opt = agent.optimizer.optimizer

        # Act
        with (
            patch(
                "agilerl.algorithms.core.base.materialize_fsdp2_from_cpu_state",
                side_effect=lambda m, _d, _c: m,
            ),
            patch.object(agent, "_rebuild_optimizer_after_load"),
        ):
            LLMAlgorithm.wrap_models(agent)

        # Assert
        assert not isinstance(agent.optimizer.optimizer, CPUOffloadOptimizer)
        assert agent.optimizer.optimizer is original_opt


# ---------------------------------------------------------------------------
# 4. _lm_head_matmul_ctx
# ---------------------------------------------------------------------------


def _make_lm_head_stub(
    *,
    distributed: bool,
    fsdp_config: FSDPConfig | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> SimpleNamespace:
    """Minimal agent stub for ``LLMAlgorithm._lm_head_matmul_ctx``."""
    fused_fn = MagicMock()
    agent = SimpleNamespace(
        distributed=distributed,
        fsdp_config=fsdp_config,
    )
    agent._fused_logprob_fn_and_head = lambda: (fused_fn, weight, bias)
    return agent


class TestLLMLmHeadMatmulCtx:
    """``_lm_head_matmul_ctx`` materialises the lm_head DTensor and moves it
    to the compute device under FSDP2 cpu_offload; passthrough otherwise.
    """

    def test_passthrough_yields_original_weight_when_no_fsdp(self):
        # Arrange
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        agent = _make_lm_head_stub(
            distributed=False, fsdp_config=None, weight=weight, bias=bias
        )
        hidden = torch.randn(2, 8)

        # Act
        with LLMAlgorithm._lm_head_matmul_ctx(agent, hidden) as (w, b):
            # Assert
            assert w is weight
            assert b is bias

    def test_passthrough_when_fsdp_without_cpu_offload(self):
        # Arrange
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        agent = _make_lm_head_stub(
            distributed=True,
            fsdp_config=FSDPConfig(cpu_offload=False),
            weight=weight,
            bias=bias,
        )
        hidden = torch.randn(2, 8)

        # Act
        with LLMAlgorithm._lm_head_matmul_ctx(agent, hidden) as (w, b):
            # Assert
            assert w is weight
            assert b is bias

    def test_materializes_and_moves_weight_to_hidden_device_under_cpu_offload(self):
        # Arrange — weight on CPU, hidden on a different device
        cpu_weight = torch.randn(4, 8)
        cpu_bias = torch.randn(4)
        agent = _make_lm_head_stub(
            distributed=True,
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
            weight=cpu_weight,
            bias=cpu_bias,
        )
        hidden = torch.randn(2, 8, device="cpu")

        # Act — patch materialize_dtensors to yield the plain CPU tensors
        with patch(
            "agilerl.algorithms.core.base.materialize_dtensors",
            side_effect=lambda *tensors: _fake_materialize(tensors),
        ):
            with LLMAlgorithm._lm_head_matmul_ctx(agent, hidden) as (w, b):
                # Assert — weight and bias on hidden.device
                assert w.device == hidden.device
                assert b.device == hidden.device

    def test_no_move_when_weight_already_on_hidden_device(self):
        # Arrange — weight already on the same device as hidden
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        move_calls = []
        original_to = torch.Tensor.to

        def tracking_to(self_tensor, device, **kwargs):
            move_calls.append(device)
            return original_to(self_tensor, device, **kwargs)

        agent = _make_lm_head_stub(
            distributed=True,
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
            weight=weight,
            bias=bias,
        )
        hidden = torch.randn(2, 8)

        # Act
        with (
            patch(
                "agilerl.algorithms.core.base.materialize_dtensors",
                side_effect=lambda *tensors: _fake_materialize(tensors),
            ),
            patch("torch.Tensor.to", tracking_to),
        ):
            with LLMAlgorithm._lm_head_matmul_ctx(agent, hidden) as (w, b):
                # Assert — no device move needed
                assert len(move_calls) == 0

    def test_handles_none_bias(self):
        # Arrange — bias is None
        weight = torch.randn(4, 8)
        agent = _make_lm_head_stub(
            distributed=True,
            fsdp_config=FSDPConfig(cpu_offload=True, optim_cpu_offload=False),
            weight=weight,
            bias=None,
        )
        hidden = torch.randn(2, 8)

        # Act
        with patch(
            "agilerl.algorithms.core.base.materialize_dtensors",
            side_effect=lambda *tensors: _fake_materialize(tensors),
        ):
            with LLMAlgorithm._lm_head_matmul_ctx(agent, hidden) as (w, b):
                # Assert — weight moved, bias stays None
                assert w.device == hidden.device
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


def _make_actor_agent(*, distributed=True, fsdp_config=None, device="cuda:0"):
    """MagicMock agent for ``get_action`` HF-generate path tests.

    ``device`` is set to a CUDA string even on CPU-only hosts so the FSDP2
    branch (``torch.device(self.device)``) and the non-FSDP2 branch
    (``next(params).device`` = CPU) produce distinguishable devices.
    ``prepare_prompt_hf_generate`` is patched by the caller to capture the
    device without actually moving tensors.
    """
    agent = MagicMock()
    agent.use_vllm = False
    agent.distributed = distributed
    agent.fsdp_config = fsdp_config
    agent.device = device
    agent.hf_generate_chunk_size = 1
    agent.group_size = 1
    agent.pad_token_id = 0
    agent.vllm_importance_sampling_correction = False
    agent._get_unwrapped_actor.return_value = nn.Linear(2, 2)
    agent.actor.generate.return_value = torch.ones(1, 8, dtype=torch.long)
    return agent


def _dummy_prompts():
    return [
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
    ]


def _patch_hf_generate_path(module_path, captured_devices):
    """Patch the HF-generate helper functions to capture ``actor_device``
    and return dummy data without moving tensors.
    """

    def capture_prepare(prompt_dict, device):
        captured_devices.append(device)
        return {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "stitch_prefix_ids": None,
            "initial_prompt_len": 4,
        }

    return (
        patch(f"{module_path}.prepare_prompt_hf_generate", side_effect=capture_prepare),
        patch(
            f"{module_path}.build_hf_completion_mask",
            return_value=(
                torch.ones(1, 8, dtype=torch.long),
                torch.ones(1, 8, dtype=torch.bool),
            ),
        ),
    )


class TestGRPOGetActionActorDevice:
    """``actor_device`` guard: FSDP2 uses ``self.device``, non-FSDP2 probes params."""

    def test_uses_self_device_when_fsdp_config_set(self):
        # Arrange — FSDP2 branch: actor_device = torch.device(self.device)
        agent = _make_actor_agent(fsdp_config=FSDPConfig(), device="cuda:0")
        captured: list = []

        # Act
        patches = _patch_hf_generate_path("agilerl.algorithms.grpo", captured)
        with (
            patches[0],
            patches[1],
        ):
            GRPO.get_action(agent, _dummy_prompts(), training=False)

        # Assert — prompts sent to self.device, not the CPU param device
        assert captured[0] == torch.device("cuda:0")

    def test_uses_param_device_when_no_fsdp_config(self):
        # Arrange — non-FSDP2 branch: actor_device = next(params).device
        agent = _make_actor_agent(fsdp_config=None, device="cuda:0")
        captured: list = []

        # Act
        patches = _patch_hf_generate_path("agilerl.algorithms.grpo", captured)
        with (
            patches[0],
            patches[1],
        ):
            GRPO.get_action(agent, _dummy_prompts(), training=False)

        # Assert — prompts sent to the param device (CPU), not self.device
        assert captured[0] == torch.device("cpu")


class TestPPOGetActionActorDevice:
    def test_uses_self_device_when_fsdp_config_set(self):
        # Arrange
        agent = _make_actor_agent(fsdp_config=FSDPConfig(), device="cuda:0")
        captured: list = []

        # Act
        patches = _patch_hf_generate_path("agilerl.algorithms.ppo_llm", captured)
        with (
            patches[0],
            patches[1],
        ):
            PPO.get_action(agent, _dummy_prompts(), training=False)

        # Assert
        assert captured[0] == torch.device("cuda:0")


class TestREINFORCEGetActionActorDevice:
    def test_uses_self_device_when_fsdp_config_set(self):
        # Arrange
        agent = _make_actor_agent(fsdp_config=FSDPConfig(), device="cuda:0")
        captured: list = []

        # Act
        patches = _patch_hf_generate_path("agilerl.algorithms.reinforce_llm", captured)
        with (
            patches[0],
            patches[1],
        ):
            REINFORCE.get_action(agent, _dummy_prompts(), training=False)

        # Assert
        assert captured[0] == torch.device("cuda:0")


# ---------------------------------------------------------------------------
# 5. load_lora_adapters — DTensor scatter into sharded params
# ---------------------------------------------------------------------------


class TestLoadLoraAdapters:
    """``load_lora_adapters`` scatters full tensors into DTensor local shards."""

    def test_scatters_full_tensor_into_dtensor_param(self, tmp_path):
        from agilerl.utils.llm_utils import load_lora_adapters

        # Arrange — a fake adapter param that looks like a DTensor
        local_shard = torch.zeros(2, 4)
        param = MagicMock()
        param.data._local_tensor = local_shard
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
        with patch("torch.distributed.tensor.distribute_tensor") as mock_dist:
            mock_dist.return_value = MagicMock(_local_tensor=full_tensor)
            load_lora_adapters(model, str(tmp_path), "actor", device="cpu")

        # Assert — local shard was overwritten with the scattered value
        assert torch.equal(local_shard, full_tensor)

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
# 6. _load_gathered_optimizer_state_dict — manual DTensor-aware load
# ---------------------------------------------------------------------------


class TestLoadGatheredOptimizerStateDict:
    """``_load_gathered_optimizer_state_dict`` manually scatters optimizer
    state into DTensor shards instead of calling ``set_optimizer_state_dict``.
    """

    def test_non_sharded_uses_load_state_dict(self):
        agent = _make_llm_agent_for_ckpt()
        agent.optimizer = MagicMock()

        with patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=False):
            LLMAlgorithm._load_gathered_optimizer_state_dict(agent, {"state": {}})

        agent.optimizer.load_state_dict.assert_called_once_with({"state": {}})

    def test_sharded_does_not_call_set_optimizer_state_dict(self):
        agent = _make_llm_agent_for_ckpt()

        inner_opt = MagicMock()
        inner_opt.state = {}
        inner_opt.param_groups = [{"params": [], "lr": 0.001}]
        agent.optimizer = MagicMock()
        agent.optimizer.optimizer = inner_opt

        set_opt_mock = MagicMock()

        with (
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
            patch(
                "torch.distributed.checkpoint.state_dict.set_optimizer_state_dict",
                set_opt_mock,
            ),
        ):
            LLMAlgorithm._load_gathered_optimizer_state_dict(agent, {"state": {}})

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
        inner_opt.optimizer = (
            inner_opt  # so getattr(opt, "optimizer", opt) returns self
        )
        inner_opt.state = {param: {"step": torch.tensor(999.0)}}
        inner_opt.param_groups = [{"params": [param], "lr": 0.001}]
        agent.optimizer = MagicMock()
        agent.optimizer.optimizer = inner_opt

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

        actor = agent._get_unwrapped_actor.return_value
        actor.named_parameters.return_value = [
            ("layer.lora_A.actor.weight", param),
        ]

        with (
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
            patch("torch.distributed.tensor.distribute_tensor") as mock_dist,
        ):
            mock_dist.return_value = MagicMock(_local_tensor=torch.randn(2, 4))
            LLMAlgorithm._load_gathered_optimizer_state_dict(agent, saved_state)

        # Assert — stale state was cleared and replaced with checkpoint state
        assert param in inner_opt.state
        assert inner_opt.state[param]["step"].item() == 1.0


# ---------------------------------------------------------------------------
# 7. get_state_dict cpu_offload parameter
# ---------------------------------------------------------------------------


class TestGetStateDictCpuOffload:
    """``get_state_dict`` passes ``cpu_offload`` through to
    ``get_model_state_dict`` for FSDP2-sharded models.
    """

    def test_default_cpu_offload_true(self):
        from agilerl.utils.llm_utils import get_state_dict

        model = MagicMock()
        with (
            patch("agilerl.utils.llm_utils.is_fsdp_sharded", return_value=True),
            patch(
                "torch.distributed.checkpoint.state_dict.get_model_state_dict"
            ) as mock_get,
        ):
            get_state_dict(model)

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert kwargs["options"].cpu_offload is True

    def test_cpu_offload_false_rejected_for_fsdp(self):
        from agilerl.utils.llm_utils import get_state_dict

        model = MagicMock()
        with (
            patch("agilerl.utils.llm_utils.is_fsdp_sharded", return_value=True),
            pytest.raises(ValueError, match="cpu_offload=False"),
        ):
            get_state_dict(model, cpu_offload=False)

    def test_non_sharded_ignores_cpu_offload(self):
        from agilerl.utils.llm_utils import get_state_dict

        model = nn.Linear(4, 4)
        with patch("agilerl.utils.llm_utils.is_fsdp_sharded", return_value=False):
            sd = get_state_dict(model, cpu_offload=False)
        assert "weight" in sd


# ---------------------------------------------------------------------------
# 8. _gathered_optimizer_state_dict cpu_offload parameter
# ---------------------------------------------------------------------------


class TestGatheredOptimizerStateDictCpuOffload:
    """``_gathered_optimizer_state_dict`` requires CPU offload under FSDP2."""

    def test_default_cpu_offload_true(self):
        agent = _make_llm_agent_for_ckpt()
        agent.optimizer = MagicMock()
        agent.optimizer.optimizer = MagicMock()

        with (
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
            patch(
                "torch.distributed.checkpoint.state_dict.get_optimizer_state_dict"
            ) as mock_get,
        ):
            LLMAlgorithm._gathered_optimizer_state_dict(agent)

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert kwargs["options"].cpu_offload is True

    def test_cpu_offload_false_rejected_for_fsdp(self):
        agent = _make_llm_agent_for_ckpt()
        agent.optimizer = MagicMock()
        agent.optimizer.optimizer = MagicMock()

        with (
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
            pytest.raises(ValueError, match="cpu_offload=False"),
        ):
            LLMAlgorithm._gathered_optimizer_state_dict(agent, cpu_offload=False)


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
        with (
            patch(
                "agilerl.algorithms.core.base.materialize_fsdp2_from_cpu_state",
                side_effect=lambda m, _d, _c: m,
            ) as mock_mat,
            patch.object(agent, "_rebuild_optimizer_after_load"),
        ):
            LLMAlgorithm.wrap_models(agent)
        mock_mat.assert_called_once_with(agent.actor, "cuda:0", agent.fsdp_config)

    def test_adapter_clone_rebuild_false_for_fsdp_without_quant(self):
        agent = _make_wrap_stub(fsdp_config=FSDPConfig(), quantization_config=None)
        assert LLMAlgorithm._uses_adapter_clone_rebuild(agent) is False

    def test_adapter_clone_rebuild_true_for_quant(self):
        agent = _make_wrap_stub(fsdp_config=None, quantization_config=object())
        assert LLMAlgorithm._uses_adapter_clone_rebuild(agent) is True

    def test_fsdp_clone_uses_cpu_state_path(self):
        agent = MagicMock()
        agent.fsdp_config = FSDPConfig()
        agent.quantization_config = None
        clone = MagicMock()
        clone.mutation_hook = MagicMock()
        clone.wrap_models = MagicMock()
        agent._uses_adapter_clone_rebuild = MagicMock(return_value=False)
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


class TestCloneLlmRejectsFsdp:
    def test_raises_when_fsdp_source_lacks_state_dict(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        from agilerl.utils.algo_utils import clone_llm

        model = GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_embd=16, n_head=2, vocab_size=32)
        )
        with (
            patch("agilerl.utils.algo_utils.is_fsdp_sharded", return_value=True),
            pytest.raises(RuntimeError, match="CPU state_dict"),
        ):
            clone_llm(model)


class TestMaterializeFsdp2FromCpuState:
    """``materialize_fsdp2_from_cpu_state`` never densifies on CUDA before shard."""

    def test_sequence_meta_shard_empty_load(self):
        from agilerl.utils.distributed import materialize_fsdp2_from_cpu_state

        model = nn.Linear(4, 4)
        calls: list[str] = []

        def _to_empty(device):
            calls.append(f"to_empty:{device}")
            return model

        model.to_empty = MagicMock(side_effect=_to_empty)  # type: ignore[method-assign]
        with (
            patch(
                "agilerl.utils.distributed.apply_fsdp2",
                side_effect=lambda m, _c: calls.append("apply_fsdp2") or m,
            ),
            patch(
                "agilerl.utils.llm_utils.load_full_state_dict",
                side_effect=lambda *_a, **_k: calls.append("load"),
            ),
            patch(
                "agilerl.utils.distributed._restore_after_to_empty",
                side_effect=lambda _m: calls.append("restore"),
            ),
        ):
            out = materialize_fsdp2_from_cpu_state(
                model, "cuda:0", FSDPConfig(cpu_offload=False)
            )

        assert out is model
        assert calls[0].startswith("to_empty:meta")
        assert "apply_fsdp2" in calls
        assert any(c.startswith("to_empty:") and "cuda" in c for c in calls)
        assert calls.index("apply_fsdp2") < next(
            i for i, c in enumerate(calls) if c.startswith("to_empty:") and "cuda" in c
        )
        assert "load" in calls


def _make_llm_agent_for_ckpt() -> MagicMock:
    """Minimal agent stub for checkpoint load/optimizer tests."""
    agent = MagicMock()
    agent.use_value_head = False
    agent.selected_adapters = ("actor",)
    agent.device = "cpu"
    actor = MagicMock()
    actor.named_parameters.return_value = []
    agent._get_unwrapped_actor.return_value = actor
    return agent
