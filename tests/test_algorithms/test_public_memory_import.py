# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The public framework imports without the private memory package."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from agilerl.algorithms.core.base import (
    LLMAlgorithm,
    _training_memory,
    training_device_spec,
)
from agilerl.arena.llm_runtime import (
    context_parallel_blocker,
    expert_gemm_backend,
    grouped_gemm_supported,
    leaks_recurrent_state,
    mamba_scan_is_unfused,
    padded_expert_lora_rank,
    resolve_attn_implementation,
    resolve_chunk_rows,
    resolve_max_lora_rank,
    resolve_mini_batch_size,
    resolve_moe_lora_recompute,
    sequence_packing_blocker,
)
from agilerl.arena.models.fsdp import FSDPConfig
from agilerl.utils.algo_utils import VLLMConfig


class TestLlmRuntime:
    def test_batch_chunk_and_attention(self) -> None:
        assert resolve_mini_batch_size(4, 8) == 4
        assert resolve_mini_batch_size(None, 8) == 8
        assert resolve_chunk_rows(128) == 4096
        assert resolve_chunk_rows(1_000_000) == 128
        assert resolve_attn_implementation("nemotron_h", cuda=False) == (
            "flash_attention_2"
        )
        assert resolve_attn_implementation("llama", cuda=True) == "flash_attention_2"
        assert resolve_attn_implementation(None, cuda=False) == "sdpa"

    def test_context_parallel_and_packing(self) -> None:
        assert context_parallel_blocker("sdpa", 1) is None
        assert context_parallel_blocker("flash_attention_2", 2) is None
        assert context_parallel_blocker("sdpa", 2, "glm5_next") is None
        assert context_parallel_blocker("sdpa", 2, "llama") is not None
        assert leaks_recurrent_state("llama", True) is True
        assert leaks_recurrent_state("qwen3_5", True) is False
        assert (
            sequence_packing_blocker(False, "sdpa", 1, False, False, "llama")
            is not None
        )
        assert (
            sequence_packing_blocker(True, "sdpa", 1, False, False, "llama") is not None
        )
        assert (
            sequence_packing_blocker(True, "flex_attention", 1, False, False, "gpt_oss")
            is not None
        )
        assert (
            sequence_packing_blocker(True, "flash_attention_2", 2, False, True, "llama")
            is not None
        )
        assert (
            sequence_packing_blocker(True, "flash_attention_2", 1, True, False, "llama")
            is not None
        )
        assert (
            sequence_packing_blocker(
                True, "flash_attention_2", 1, False, False, "llama"
            )
            is None
        )

    def test_lora_rank_mamba_and_expert_gemm(self) -> None:
        assert resolve_max_lora_rank(12) == 16
        with pytest.raises(ValueError, match="max_lora_rank"):
            resolve_max_lora_rank(1024)
        assert mamba_scan_is_unfused((8, 9)) is True
        assert mamba_scan_is_unfused((9, 0)) is False
        assert grouped_gemm_supported((9, 0), "bfloat16") is True
        assert grouped_gemm_supported((9, 0), "float16") is False
        assert expert_gemm_backend((9, 0), "bfloat16", True) == "triton"
        assert expert_gemm_backend((10, 0), "bfloat16", True) == "grouped_mm"
        assert expert_gemm_backend((8, 0), "float16", True) == "triton"
        assert expert_gemm_backend((10, 0), "bfloat16", False) == "grouped_mm"
        with pytest.raises(RuntimeError, match="without Triton"):
            expert_gemm_backend((8, 0), "float32", False)
        assert padded_expert_lora_rank(8, 2) == 8
        assert padded_expert_lora_rank(9, 2) == 16
        assert resolve_moe_lora_recompute(True) is False
        assert resolve_moe_lora_recompute(False) is True


class TestPublicModulesImportWithoutAgilerlMemory:
    def test_framework_modules_import(self) -> None:
        script = textwrap.dedent(
            """
            import sys

            class Block:
                def find_spec(self, name, path, target=None):
                    if name == "agilerl_memory" or name.startswith("agilerl_memory."):
                        raise ModuleNotFoundError(name)
                    return None

            sys.meta_path.insert(0, Block())
            modules = [
                "agilerl.algorithms.core.base",
                "agilerl.utils.llm_utils",
                "agilerl.utils.algo_utils",
                "agilerl.utils.lora_merge",
                "agilerl.builders.llm",
                "agilerl.distributed.runtime",
                "agilerl.distributed.expert_parallel",
                "agilerl.architectures.nemotron_h.layers",
                "agilerl.lora.moe.adapters",
                "agilerl.lora.moe.grouped_gemm",
                "agilerl.lora.moe.model",
                "agilerl.lora.moe.routed",
                "agilerl.lora.moe.wrappers",
            ]
            for name in modules:
                __import__(name)
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr


def _install_memory(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Put stand-in estimator modules where the lazy import looks them up."""

    class DeviceSpec:
        def __init__(self, total_bytes, name, compute_capability, gpus_per_node):
            self.total_bytes = total_bytes
            self.name = name
            self.compute_capability = compute_capability
            self.gpus_per_node = gpus_per_node
            self.usable_bytes = total_bytes
            self.available_bytes = None

        def model_copy(self, update):
            copied = DeviceSpec(
                self.total_bytes,
                self.name,
                self.compute_capability,
                self.gpus_per_node,
            )
            copied.usable_bytes = self.usable_bytes
            copied.__dict__.update(update)
            return copied

    class Settings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def model_copy(self, update):
            copied = Settings(**self.__dict__)
            copied.__dict__.update(update)
            return copied

    class ModelArch:
        @staticmethod
        def from_hf_config(config):
            return {"config": config}

    class ModelSpec:
        def __init__(self, model_id, arch):
            self.model_id = model_id
            self.arch = arch

    class Plan:
        def __init__(self, fsdp):
            self.gradient_checkpointing = True
            self.fsdp = fsdp
            self.routed_expert_chunk_mib = 32
            self.breakdown = SimpleNamespace(total_bytes=10, device_usable_bytes=20)

        def describe(self):
            return "fake plan"

    class TrainingMemoryUnfitError(RuntimeError):
        pass

    calls = {"resolve": 0}

    def resolve_training_memory(*_args):
        calls["resolve"] += 1
        if calls["raise"]:
            msg = "nothing fits"
            raise TrainingMemoryUnfitError(msg)
        return Plan(calls["fsdp"])

    calls["raise"] = False
    calls["fsdp"] = None

    fsdp = ModuleType("agilerl_memory.resolve.fsdp")
    fsdp.MemoryPins = lambda **kwargs: kwargs
    fsdp.TrainingMemoryUnfitError = TrainingMemoryUnfitError
    fsdp.resolve_training_memory = resolve_training_memory
    training = ModuleType("agilerl_memory.resolve.training")
    training.liger_kernel_arch = lambda arch, _liger: arch
    training.lora_target_scope = lambda target: target
    training.packed_moe_dispatch = lambda _params, _materializes: "none"
    training.sampling_support_width = lambda _records, width: width
    specs = ModuleType("agilerl_memory.specs")
    specs.DeviceSpec = DeviceSpec
    specs.ModelArch = ModelArch
    specs.ModelSpec = ModelSpec
    specs.TrainingSettings = Settings
    manifest = ModuleType("agilerl_memory.manifest")
    manifest.ALGORITHM_NAMES = {"grpo": "grpo"}

    for name in ("agilerl_memory", "agilerl_memory.resolve"):
        if name not in sys.modules:
            parent = ModuleType(name)
            parent.__path__ = []
            monkeypatch.setitem(sys.modules, name, parent)
    monkeypatch.setitem(sys.modules, "agilerl_memory.resolve.fsdp", fsdp)
    monkeypatch.setitem(sys.modules, "agilerl_memory.resolve.training", training)
    monkeypatch.setitem(sys.modules, "agilerl_memory.specs", specs)
    monkeypatch.setitem(sys.modules, "agilerl_memory.manifest", manifest)
    return calls


class _StubAlgorithm(LLMAlgorithm):
    def get_action(self, obs):
        """Unused."""
        return obs

    def learn(self, experiences):
        """Unused."""
        return experiences


def _agent(target_modules: str | list[str]) -> _StubAlgorithm:
    head = nn.Linear(2, 2)
    agent = _StubAlgorithm.__new__(_StubAlgorithm)
    agent.algo = "grpo"
    agent.beta = 0.0
    agent.lora_config = type(
        "Lora",
        (),
        {
            "target_modules": target_modules,
            "r": 8,
            "lora_dropout": 0.0,
            "target_parameters": None,
        },
    )()
    agent.group_size = 1
    agent.batch_size_per_process = 2
    agent.micro_batch_size_per_gpu = 1
    agent.gradient_checkpointing = None
    agent.use_separate_reference_adapter = False
    agent.activation_offload = False
    agent.top_k = None
    agent.fsdp_config = None
    agent.routed_expert_chunk_mib = None
    agent.sequence_packing = False
    agent.pretrained_model_name_or_path = "stub/model"
    agent.actor = nn.Linear(2, 2)
    agent.device = "cuda:0"
    agent.mini_batch_size = 1
    agent.vllm_config = None
    agent.colocated = False
    agent.shard_runtime = type(
        "Shard", (), {"data_parallel_world": staticmethod(lambda _world: 1)}
    )()
    agent._row_token_budget = lambda: 32
    agent._micro_batch_token_budget = lambda: None
    agent._config_records_sampling_supports = lambda: False
    agent._get_lm_head = lambda: head
    return agent


class TestTrainingMemoryLoad:
    def test_device_budget_keeps_or_drops_a_resident_engine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_memory(monkeypatch)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _device: SimpleNamespace(
                total_memory=1000, name="L4", major=8, minor=9
            ),
        )
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)

        free = training_device_spec("cuda:0")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
        shared = training_device_spec(
            "cuda:0",
            VLLMConfig(gpu_memory_utilization=0.4, sleep_mode=False),
            colocated=True,
        )

        assert free.gpus_per_node is None
        assert free.available_bytes is None
        assert shared.gpus_per_node == 4
        assert shared.available_bytes == 1000 - 400

    def test_estimate_inputs_read_a_module_list_and_a_regex(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_memory(monkeypatch)
        monkeypatch.setattr("agilerl.algorithms.core.base.get_world_size", lambda: 1)
        monkeypatch.setattr(
            "agilerl.algorithms.core.base.PretrainedConfig.get_config_dict",
            lambda _name: ({"model_type": "qwen3"},),
        )

        listed, _settings = _agent(["q_proj"])._training_estimate_inputs("grpo", 0.1)
        named, _settings = _agent("q_proj|k_proj")._training_estimate_inputs(
            "grpo", 0.1
        )
        linear, linear_settings = _agent("all-linear")._training_estimate_inputs(
            "grpo", 0.1
        )

        assert listed.model_id == "stub/model"
        assert named.arch == {"config": {"model_type": "qwen3"}}
        assert linear.model_id == "stub/model"
        assert linear_settings.lora_target_scope == "all-linear"

    def test_search_returns_rank_zero_plan_and_raises_when_unfit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _install_memory(monkeypatch)
        monkeypatch.setattr("agilerl.algorithms.core.base.get_rank", lambda: 0)
        monkeypatch.setattr("agilerl.algorithms.core.base.get_world_size", lambda: 1)
        monkeypatch.setattr(
            "agilerl.algorithms.core.base.broadcast_object_list",
            lambda objects, src=0: objects,
        )
        monkeypatch.setattr(
            "agilerl.algorithms.core.base.PretrainedConfig.get_config_dict",
            lambda _name: ({"model_type": "qwen3"},),
        )
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _device: SimpleNamespace(
                total_memory=1000, name="L4", major=8, minor=9
            ),
        )
        agent = _agent(["q_proj"])
        picked_fsdp = FSDPConfig(ep=1, reshard_after_forward=True)
        calls["fsdp"] = picked_fsdp

        resolved = agent._resolve_fsdp_config(FSDPConfig())

        assert resolved is picked_fsdp
        assert agent.gradient_checkpointing is True
        assert agent.routed_expert_chunk_mib == 32
        agent.gradient_checkpointing = True
        agent._resolve_fsdp_config(
            FSDPConfig(
                reshard_after_forward=True,
                routed_expert_chunk_mib=32,
                optim_cpu_offload=False,
                activation_offload=False,
                cpu_offload=False,
            )
        )
        calls["raise"] = True
        with pytest.raises(RuntimeError, match="nothing fits"):
            agent._search_training_memory(FSDPConfig())

    def test_other_ranks_raise_the_broadcast_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_memory(monkeypatch)
        monkeypatch.setattr("agilerl.algorithms.core.base.get_rank", lambda: 1)
        monkeypatch.setattr(
            "agilerl.algorithms.core.base.broadcast_object_list",
            lambda objects, src=0: [None, "no fit"],
        )
        agent = _agent(["q_proj"])

        with pytest.raises(ValueError, match="no fit"):
            agent._search_training_memory(None)

    def test_loader_reads_the_installed_modules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_memory(monkeypatch)

        memory = _training_memory()

        assert memory.algorithm_names["grpo"] == "grpo"
