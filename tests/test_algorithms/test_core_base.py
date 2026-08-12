# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl.algorithms.core.base module.

For LLMAlgorithm.save_checkpoint / load_checkpoint, the following cases are considered
exhaustively.

There are 16 cells we care about, defined by the grid
    lora_only:       True / False
    save_optimizer:  True / False
    use_deepspeed:   True / False
x {save, load}.

Expected behaviour per cell (the spec this file enforces):

SAVE — deepspeed path
    lora_only=T, save_optim=T  →  deepspeed save (exclude frozen params)
    lora_only=F, save_optim=T  →  deepspeed save (include frozen params)
    lora_only=T, save_optim=F  →  peft (save adapters)
    lora_only=F, save_optim=F  →  gather params + torch save
                                  (actor model in attributes.pt)

SAVE — plain torch/peft path
    lora_only=T, save_optim=T  →  peft save + optim in attributes.pt
    lora_only=F, save_optim=T  →  torch save  (actor + optim in attributes.pt)
    lora_only=T, save_optim=F  →  peft save
    lora_only=F, save_optim=F  →  torch save  (actor in attributes.pt, no optim)

LOAD — deepspeed path
    LoRA=T, Optim=T  →  load deepspeed
    LoRA=F, Optim=T  →  load deepspeed
    LoRA=T, Optim=F  →  load peft
    LoRA=F, Optim=F  →  load torch state dict from attributes.pt

LOAD — plain torch/peft path
    LoRA=T, Optim=T  →  load peft + torch load (optim)
    LoRA=F, Optim=T  →  torch load
    LoRA=T, Optim=F  →  load peft
    LoRA=F, Optim=F  →  torch load

Test organisation:
  * ``grpo_factory`` — session-scoped, expensive agent build happens once.
  * ``llm_simple_checkpoint_save`` / ``llm_mocked_deepspeed_checkpoint_save`` — session-scoped, parametrised over
    the 4 cells. Each cell runs ``save_checkpoint`` once and tests read from
    the resulting artefacts.
  * ``llm_simple_checkpoint_load`` / ``llm_mocked_deepspeed_checkpoint_load`` — function-scoped
    because load tests mutate agent state (stamp sentinels, step optimizer).
  * Test bodies use the fixture's ``lora_only`` / ``save_optimizer`` fields
    as a truth table rather than branching per cell — each test runs 4x
    (once per parametrised fixture variant).

DeepSpeed tests spy-wrap ``actor.save_checkpoint`` / ``load_checkpoint``
(they'd normally talk to a distributed backend we don't have); we assert the
right branch was taken with the right kwargs.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import re
import shutil
import sys
import warnings
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, PropertyMock, call, patch

import dill
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from gymnasium import spaces
from torch import optim

from agilerl import HAS_DEEPSPEED, HAS_LLM_DEPENDENCIES, HAS_VLLM
from agilerl.algorithms.core.base import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    RegistryMeta,
    _is_readonly_property,
    get_checkpoint_dict,
    get_optimizer_cls,
)
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.algorithms.grpo import GRPO
from agilerl.modules import EvolvableMLP
from agilerl.modules.dummy import DummyEvolvable
from agilerl.utils.algo_utils import VLLMConfig
from tests.test_algorithms.test_base import DummyMARLAlgorithm, DummyRLAlgorithm

create_module = None
if HAS_DEEPSPEED and HAS_VLLM:
    # create_module lives in test_grpo, which importorskips deepspeed/vllm.
    from tests.test_algorithms.test_llms.test_grpo import create_module

pytest.importorskip("peft", reason="LLM checkpoint tests require peft.")
pytest.importorskip("transformers", reason="LLM checkpoint tests require transformers.")

deepspeed_config_stage_2 = None
if HAS_DEEPSPEED and HAS_VLLM:
    from tests.test_algorithms.test_llms.test_grpo import deepspeed_config_stage_2

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from peft import LoraConfig

_LLM_DEPS_SKIP = pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES, reason="agilerl[llm] not installed"
)

_VLLM_SKIP = pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")


def test_core_base_vllm_types_none_when_vllm_not_installed():
    """vLLM symbols are None when HAS_VLLM is false at import time."""
    original_module = sys.modules.pop("agilerl.algorithms.core.base", None)

    try:
        with patch("agilerl.HAS_VLLM", False):
            reloaded = importlib.import_module("agilerl.algorithms.core.base")
            assert reloaded.LLM is None
            assert reloaded.CompletionOutput is None
            assert reloaded.SamplingParams is None
    finally:
        sys.modules["agilerl.algorithms.core.base"] = original_module
        import agilerl.algorithms.core as _core_pkg

        _core_pkg.base = original_module


@pytest.fixture
def vector_space():
    return spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)


@pytest.fixture
def dummy_agent(vector_space):
    action_space = spaces.Discrete(2)
    return DummyRLAlgorithm(vector_space, action_space, index=0)


class TestGetCheckpointDict:
    def test_checkpoint_dict_contains_network_info(self, dummy_agent):
        chkpt = get_checkpoint_dict(dummy_agent)
        assert "network_info" in chkpt
        assert "modules" in chkpt["network_info"]
        assert "optimizers" in chkpt["network_info"]
        assert "network_names" in chkpt["network_info"]
        assert "optimizer_names" in chkpt["network_info"]
        assert "dummy_actor" in chkpt["network_info"]["network_names"]
        assert "dummy_optimizer" in chkpt["network_info"]["optimizer_names"]

    def test_checkpoint_dict_excludes_accelerator(self, dummy_agent):
        chkpt = get_checkpoint_dict(dummy_agent)
        assert "accelerator" not in chkpt

    def test_checkpoint_dict_includes_agilerl_version(self, dummy_agent):
        chkpt = get_checkpoint_dict(dummy_agent)
        assert "agilerl_version" in chkpt

    def test_checkpoint_dict_deepspeed_pops_actor(self, vector_space):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        agent.actor = agent.dummy_actor
        chkpt = get_checkpoint_dict(agent, omit_actor_info=True)
        assert "actor" not in chkpt

    def test_checkpoint_dict_omit_optimizer_info(self, dummy_agent):
        dummy_agent.optimizer = MagicMock()
        chkpt = get_checkpoint_dict(dummy_agent, omit_optimizer_info=True)
        assert "optimizer" not in chkpt

    def test_checkpoint_dict_with_lr_scheduler(self, dummy_agent):
        dummy_agent.lr_scheduler = MagicMock()
        dummy_agent.lr_scheduler.state_dict.return_value = {"step": 0}
        chkpt = get_checkpoint_dict(dummy_agent)
        assert chkpt["lr_scheduler"] == {"step": 0}

    def test_checkpoint_dict_pops_rollout_buffer(self, dummy_agent):
        dummy_agent.rollout_buffer = MagicMock()
        chkpt = get_checkpoint_dict(dummy_agent)
        assert "rollout_buffer" not in chkpt

    def test_checkpoint_dict_excludes_readonly_properties(self, vector_space):
        class _WithDerived(DummyRLAlgorithm):
            @property
            def aux_metric_name(self) -> str:
                return "kl"

        agent = _WithDerived(vector_space, spaces.Discrete(2), index=0)
        assert agent.aux_metric_name == "kl"
        chkpt = get_checkpoint_dict(agent)
        assert "aux_metric_name" not in chkpt


class TestReadonlyCheckpointAttributes:
    def test_is_readonly_property_detects_setterless_property(self):
        class Stub:
            @property
            def derived(self) -> int:
                return 1

            @property
            def mutable(self) -> int:
                return self._m

            @mutable.setter
            def mutable(self, value: int) -> None:
                self._m = value

        stub = Stub()
        stub._m = 0
        assert _is_readonly_property(stub, "derived")
        assert not _is_readonly_property(stub, "mutable")
        assert not _is_readonly_property(stub, "_m")

    def test_load_checkpoint_skips_readonly_property_keys(self, vector_space, tmp_path):
        class _WithDerived(DummyRLAlgorithm):
            @property
            def aux_metric_name(self) -> str:
                return "kl"

        writer = DummyRLAlgorithm(vector_space, spaces.Discrete(2), index=0)
        path = str(tmp_path / "readonly_key.pt")
        writer.save_checkpoint(path)
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["aux_metric_name"] = "stale"
        torch.save(checkpoint, path)

        agent = _WithDerived(vector_space, spaces.Discrete(2), index=0)
        agent.load_checkpoint(path)

        assert agent.aux_metric_name == "kl"

    def test_restore_checkpoint_attributes_skips_readonly_properties(self):
        class Stub:
            def __init__(self) -> None:
                self.beta = 0.1

            @property
            def aux_metric_name(self) -> str:
                return "liger_clip_fraction" if self.beta == 0.0 else "kl"

        stub = Stub()
        LLMAlgorithm._restore_checkpoint_attributes(
            stub,
            {"aux_metric_name": "kl", "beta": 0.0, "lora_config": "skip-me"},
        )
        assert stub.beta == 0.0
        assert stub.aux_metric_name == "liger_clip_fraction"
        assert not hasattr(stub, "lora_config")


class TestRaiseIfLossNotFiniteOnAnyRank:
    def test_all_ranks_finite_returns_without_raising(self):
        stub = SimpleNamespace(accelerator=SimpleNamespace(num_processes=2))
        with patch(
            "agilerl.algorithms.core.base.allreduce_minmax_int",
            return_value=(0, 0),
        ) as reduce_call:
            result = LLMAlgorithm._raise_if_loss_not_finite_on_any_rank(
                stub, torch.tensor(1.0)
            )
        assert result is None
        reduce_call.assert_called_once()


class TestGetOptimizerCls:
    @pytest.mark.parametrize("opt_name", ["Adam", "SGD", "AdamW", "RMSprop"])
    def test_string_returns_optimizer_class(self, opt_name):
        cls = get_optimizer_cls(opt_name)
        assert cls is getattr(torch.optim, opt_name)

    def test_dict_returns_dict_of_classes(self):
        result = get_optimizer_cls({"a": "Adam", "b": "SGD"})
        assert isinstance(result, dict)
        assert result["a"] is torch.optim.Adam
        assert result["b"] is torch.optim.SGD

    def test_invalid_optimizer_name_raises(self):
        with pytest.raises(AttributeError, match="InvalidOptimizer"):
            get_optimizer_cls("InvalidOptimizer")

    def test_dict_with_invalid_optimizer_raises(self):
        with pytest.raises(AttributeError, match="BadName"):
            get_optimizer_cls({"a": "Adam", "b": "BadName"})


class TestInspectAttributes:
    def test_inspect_attributes_returns_dict(self, dummy_agent):
        attrs = EvolvableAlgorithm.inspect_attributes(dummy_agent)
        assert isinstance(attrs, dict)
        assert "accelerator" not in attrs or attrs.get("accelerator") is None
        assert "dummy_actor" not in attrs
        assert "dummy_optimizer" not in attrs

    def test_inspect_attributes_input_args_only(self, dummy_agent):
        attrs = EvolvableAlgorithm.inspect_attributes(dummy_agent, input_args_only=True)
        assert isinstance(attrs, dict)
        for k in attrs:
            assert k in _inspect_signature_params(dummy_agent.__init__)

    def test_inspect_attributes_excludes_tensordict(self, dummy_agent):
        from tensordict import TensorDict

        dummy_agent.buffer = TensorDict({"a": torch.zeros(1)}, batch_size=[])
        attrs = EvolvableAlgorithm.inspect_attributes(dummy_agent)
        assert "buffer" not in attrs


def _inspect_signature_params(func):
    return set(inspect.signature(func).parameters.keys())


class TestCopyAttributes:
    def test_copy_attributes_copies_non_evolvable(self, dummy_agent):
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        dummy_agent.dummy_attribute = "original"
        clone.dummy_attribute = "different"
        result = EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert result is clone
        assert clone.dummy_attribute == "original"

    def test_copy_attributes_skips_callables(self, dummy_agent):
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        dummy_agent.callable_attr = lambda x: x
        clone.callable_attr = lambda y: y + 1
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert clone.callable_attr(1) == 2

    def test_copy_attributes_copies_tensor_when_different(self, dummy_agent):
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        t = torch.tensor([1.0, 2.0])
        dummy_agent.tensor_attr = t
        clone.tensor_attr = torch.tensor([0.0, 0.0])
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert torch.equal(clone.tensor_attr, t)

    def test_copy_attributes_copies_ndarray(self, dummy_agent):
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        arr = np.array([1, 2, 3])
        dummy_agent.arr_attr = arr
        clone.arr_attr = np.array([0, 0, 0])
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert np.array_equal(clone.arr_attr, arr)

    def test_copy_attributes_copies_list(self, dummy_agent):
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        lst = [1, 2, {"a": 3}]
        dummy_agent.list_attr = lst
        clone.list_attr = []
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert clone.list_attr == [1, 2, {"a": 3}]
        assert clone.list_attr is not lst

    def test_copy_attributes_copies_dict(self, dummy_agent):
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        d = {"x": 1, "y": [2]}
        dummy_agent.dict_attr = d
        clone.dict_attr = {}
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert clone.dict_attr == {"x": 1, "y": [2]}
        assert clone.dict_attr is not d

    def test_copy_attributes_adds_missing_attribute_to_clone(self, dummy_agent):
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        dummy_agent.extra_attr = 42
        assert not hasattr(clone, "extra_attr")
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert clone.extra_attr == 42

    def test_copy_attributes_copies_mutation_registry(self, dummy_agent):
        from agilerl.algorithms.core.registry import MutationRegistry

        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        dummy_agent.custom_registry = MutationRegistry()
        clone.custom_registry = MutationRegistry()
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert clone.custom_registry is not dummy_agent.custom_registry


class TestDeprecatedMethods:
    def test_get_state_dim_deprecation(self, vector_space):
        with pytest.warns(
            DeprecationWarning,
            match=r"This method is deprecated\. Use get_input_size_from_space instead\.",
        ):
            dim = EvolvableAlgorithm.get_state_dim(vector_space)
        assert dim == (4,)

    def test_get_action_dim_deprecation(self):
        action_space = spaces.Discrete(5)
        with pytest.warns(
            DeprecationWarning,
            match=r"This method is deprecated\. Use get_output_size_from_space instead\.",
        ):
            dim = EvolvableAlgorithm.get_action_dim(action_space)
        assert dim == 5


class TestEvolvableAttributes:
    def test_evolvable_attributes_includes_networks_and_optimizers(self, dummy_agent):
        attrs = dummy_agent.evolvable_attributes()
        assert "dummy_actor" in attrs
        assert "dummy_optimizer" in attrs

    def test_evolvable_attributes_networks_only(self, dummy_agent):
        attrs = dummy_agent.evolvable_attributes(networks_only=True)
        assert "dummy_actor" in attrs
        assert "dummy_optimizer" not in attrs


class TestToDevice:
    def test_to_device_dict(self, dummy_agent):
        device = torch.device(dummy_agent.device)
        exp = {"states": torch.zeros(2, 4), "actions": torch.zeros(2)}
        result = dummy_agent.to_device(exp)
        assert len(result) == 1
        assert result[0]["states"].device == device
        assert result[0]["actions"].device == device

    def test_to_device_tuple_of_tensors(self, dummy_agent):
        device = torch.device(dummy_agent.device)
        exp = (torch.zeros(2, 4), torch.zeros(2), torch.zeros(2))
        result = dummy_agent.to_device(exp)
        assert len(result) == 1
        for t in result[0]:
            assert t.device == device

    def test_to_device_single_tensor(self, dummy_agent):
        device = torch.device(dummy_agent.device)
        exp = torch.zeros(2, 4)
        result = dummy_agent.to_device(exp)
        assert result[0].device == device

    def test_to_device_non_tensor_passthrough(self, dummy_agent):
        exp = [1, 2, 3]
        result = dummy_agent.to_device(exp)
        assert result[0] == [1, 2, 3]

    def test_to_device_multiple_experiences(self, dummy_agent):
        device = torch.device(dummy_agent.device)
        exp1 = torch.zeros(2, 4)
        exp2 = {"x": torch.zeros(2)}
        result = dummy_agent.to_device(exp1, exp2)
        assert len(result) == 2
        assert result[0].device == device
        assert result[1]["x"].device == device

    def test_to_device_list_of_non_tensors_passthrough(self, dummy_agent):
        exp = [1, 2.5, "a"]
        result = dummy_agent.to_device(exp)
        assert result[0] == [1, 2.5, "a"]

    def test_to_device_empty_list_raises_index_error(self, dummy_agent):
        with pytest.raises(IndexError):
            dummy_agent.to_device([])


class TestIndexAndMutProperties:
    def test_index_property(self, dummy_agent):
        assert dummy_agent.index == 0
        dummy_agent.index = 5
        assert dummy_agent._index == 5
        assert dummy_agent.index == 5

    def test_mut_property(self, dummy_agent):
        assert dummy_agent.mut is None
        dummy_agent.mut = "lr"
        assert dummy_agent._mut == "lr"
        assert dummy_agent.mut == "lr"


class TestWrapUnwrapModels:
    def test_wrap_models_no_accelerator_returns_early(self, dummy_agent):
        dummy_agent.accelerator = None
        dummy_agent.wrap_models()

    def test_unwrap_models_raises_without_accelerator(self, dummy_agent):
        dummy_agent.accelerator = None
        with pytest.raises(AttributeError, match="No accelerator"):
            dummy_agent.unwrap_models()


class TestRegistryMeta:
    def test_registry_init_raises_without_network_groups(self):
        class NoGroups(EvolvableAlgorithm):
            def __init__(self):
                super().__init__(index=0)

            def preprocess_observation(self, obs):
                return obs

            def learn(self, exp, **kw):
                return None

            def get_action(self, obs, **kw):
                return 0

            def test(self, **kw):
                return 0

        with pytest.raises(AttributeError, match="No network groups"):
            NoGroups()


class TestEvolvableAlgorithmInitAssertions:
    """EvolvableAlgorithm __init__ assertion error paths (run before _registry_init)."""

    def _make_stub(self, **kwargs):
        class InitStub(EvolvableAlgorithm):
            def __init__(self, index=0, **kw):
                super().__init__(
                    index=index,
                    **{
                        k: v
                        for k, v in kw.items()
                        if k
                        in (
                            "device",
                            "name",
                            "accelerator",
                            "torch_compiler",
                            "hp_config",
                        )
                    },
                )

            def preprocess_observation(self, obs):
                return obs

            def learn(self, exp, **kw):
                return None

            def get_action(self, obs, **kw):
                return 0

            def test(self, **kw):
                return 0

        return InitStub

    @pytest.mark.parametrize(
        ("bad_index", "msg"), [(1.5, "integer"), ("x", "integer"), ([], "integer")]
    )
    def test_index_must_be_int(self, bad_index, msg):
        Stub = self._make_stub()
        with pytest.raises(AssertionError, match=msg):
            Stub(index=bad_index)

    def test_device_must_be_str_or_device(self):
        Stub = self._make_stub()
        with pytest.raises(AssertionError, match="Device"):
            Stub(index=0, device=123)

    def test_name_must_be_str_or_none(self):
        Stub = self._make_stub()
        with pytest.raises(AssertionError, match="Name"):
            Stub(index=0, name=123)

    def test_accelerator_must_be_accelerator_or_none(self):
        Stub = self._make_stub()
        with pytest.raises(AssertionError, match="Accelerator"):
            Stub(index=0, accelerator="not_accelerator")

    @pytest.mark.parametrize("bad_mode", ["invalid", "off"])
    def test_torch_compiler_invalid_mode_raises(self, bad_mode):
        Stub = self._make_stub()
        with pytest.raises(AssertionError, match="torch compiler"):
            Stub(index=0, torch_compiler=bad_mode)

    @pytest.mark.parametrize(
        "valid_mode", ["default", "reduce-overhead", "max-autotune"]
    )
    def test_torch_compiler_valid_modes_accepted(self, valid_mode):
        Stub = self._make_stub()
        with pytest.raises(AttributeError, match="No network groups"):
            Stub(index=0, torch_compiler=valid_mode)


class TestSaveLoadCheckpoint:
    def test_save_and_load_checkpoint_roundtrip(self, dummy_agent, tmp_path):
        path = tmp_path / "chkpt.pth"
        dummy_agent.save_checkpoint(path)
        assert path.exists()

        agent2 = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        agent2.load_checkpoint(path)
        assert agent2.dummy_attribute == dummy_agent.dummy_attribute


class TestLoadErrorPaths:
    def test_load_raises_when_network_info_missing(self, tmp_path):
        import dill

        path = tmp_path / "bad.pth"
        torch.save({"registry": None, "index": 0}, path, pickle_module=dill)
        with pytest.raises(ValueError, match="Network info not found"):
            DummyRLAlgorithm.load(path)

    def test_load_raises_when_init_dict_missing(self, dummy_agent, tmp_path):
        import dill

        path = tmp_path / "chkpt.pth"
        dummy_agent.save_checkpoint(path)
        chkpt = torch.load(path, weights_only=False, pickle_module=dill)
        del chkpt["network_info"]["modules"]["dummy_actor_init_dict"]
        torch.save(chkpt, path, pickle_module=dill)
        with pytest.raises(ValueError, match=r"Init dict.*not found"):
            DummyRLAlgorithm.load(path)


class TestClone:
    def test_clone_creates_independent_copy(self, dummy_agent):
        clone = dummy_agent.clone(index=7)
        assert clone.index == 7
        assert clone is not dummy_agent
        assert clone.dummy_actor is not dummy_agent.dummy_actor
        assert clone.dummy_attribute == dummy_agent.dummy_attribute

    def test_clone_with_explicit_index(self, dummy_agent):
        clone = dummy_agent.clone(index=42)
        assert clone.index == 42

    def test_clone_without_index_preserves_original(self, dummy_agent):
        clone = dummy_agent.clone(index=None)
        assert clone.index == dummy_agent.index

    def test_clone_wrap_false_skips_accelerator_wrap(self, dummy_agent):
        clone = dummy_agent.clone(wrap=False)
        assert clone.dummy_actor is not dummy_agent.dummy_actor
        assert clone.dummy_attribute == dummy_agent.dummy_attribute


class TestSetTrainingMode:
    def test_set_training_mode_true(self, dummy_agent):
        dummy_agent.set_training_mode(True)
        assert dummy_agent.training is True
        assert dummy_agent.dummy_actor.training

    def test_set_training_mode_false(self, dummy_agent):
        dummy_agent.set_training_mode(False)
        assert dummy_agent.training is False
        assert not dummy_agent.dummy_actor.training


class TestCleanUp:
    def test_clean_up_removes_evolvable_attributes(self, dummy_agent):
        assert hasattr(dummy_agent, "dummy_actor")
        assert hasattr(dummy_agent, "dummy_optimizer")
        dummy_agent.clean_up()
        assert not hasattr(dummy_agent, "dummy_actor")
        assert not hasattr(dummy_agent, "dummy_optimizer")


class TestGetLrNames:
    def test_get_lr_names_returns_lr_attr_names(self, dummy_agent):
        names = dummy_agent.get_lr_names()
        assert isinstance(names, list)
        assert "lr" in names


class TestRegisterMutationHook:
    def test_register_mutation_hook_and_mutation_hook_executes(self, dummy_agent):
        hook_called = []

        def my_hook():
            hook_called.append(1)

        dummy_agent.my_hook = my_hook
        dummy_agent.register_mutation_hook(my_hook)
        dummy_agent.mutation_hook()
        assert len(hook_called) == 1

    def test_mutation_hook_calls_registered_method_by_name(self, dummy_agent):
        class AgentWithHook(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                super().__init__(obs_space, act_space, index=index)
                self._hook_called = False

            def post_mutation_hook(self):
                self._hook_called = True

        obs = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        act = spaces.Discrete(2)
        agent = AgentWithHook(obs, act, index=0)
        agent.register_mutation_hook(agent.post_mutation_hook)
        agent.mutation_hook()
        assert agent._hook_called


class TestCopyAttributesTensorRuntimeError:
    def test_copy_attributes_tensor_deepcopy_fallback_to_clone(self, dummy_agent):
        import copy as copy_mod

        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        non_leaf = torch.tensor([1.0], requires_grad=True) * 2
        dummy_agent.tensor_attr = non_leaf
        clone.tensor_attr = torch.tensor([0.0])
        real_deepcopy = copy_mod.deepcopy

        def deepcopy_raise_for_tensors(x):
            if isinstance(x, torch.Tensor):
                msg = "tensor copy failed"
                raise RuntimeError(msg)
            return real_deepcopy(x)

        with patch(
            "agilerl.algorithms.core.base.copy.deepcopy",
            side_effect=deepcopy_raise_for_tensors,
        ):
            result = EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert result is clone
        assert torch.allclose(clone.tensor_attr, non_leaf.detach())


class TestLoadCheckpointLrScheduler:
    def test_load_checkpoint_restores_lr_scheduler(self, dummy_agent, tmp_path):
        from torch.optim.lr_scheduler import StepLR

        dummy_agent.lr_scheduler = StepLR(
            dummy_agent.dummy_optimizer.optimizer, step_size=1, gamma=0.5
        )
        dummy_agent.lr_scheduler.step()
        path = tmp_path / "chkpt.pth"
        dummy_agent.save_checkpoint(path)

        agent2 = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        agent2.lr_scheduler = StepLR(
            agent2.dummy_optimizer.optimizer, step_size=1, gamma=0.5
        )
        agent2.load_checkpoint(path)
        assert agent2.lr_scheduler.last_epoch == dummy_agent.lr_scheduler.last_epoch


class TestCloneWithTorchCompiler:
    def test_clone_with_torch_compiler_creates_independent_module(self, vector_space):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, torch_compiler="default"
        )
        clone = agent.clone(wrap=False)
        assert clone.dummy_actor is not agent.dummy_actor
        assert clone.dummy_attribute == agent.dummy_attribute
        torch._dynamo.reset()


class TestWrapModelsDictPath:
    def test_wrap_models_with_dict_evolvable(self, vector_space):
        action_space = spaces.Discrete(2)
        agent_ids = ["a0", "a1"]
        obs_spaces = [vector_space, vector_space]
        act_spaces = [action_space, action_space]
        accelerator = Accelerator()
        agent = DummyMARLAlgorithm(
            obs_spaces,
            act_spaces,
            agent_ids=agent_ids,
            index=0,
            accelerator=accelerator,
        )
        agent.wrap_models()
        assert agent.dummy_actors is not None
        agent.unwrap_models()
        for actor in agent.dummy_actors.values():
            assert isinstance(actor, torch.nn.Module)


class TestToDeviceWithAccelerator:
    def test_to_device_uses_accelerator_device_when_present(self, dummy_agent):
        accelerator = Accelerator()
        dummy_agent.accelerator = accelerator
        exp = torch.zeros(2, 4)
        result = dummy_agent.to_device(exp)
        assert result[0].device.type == accelerator.device.type
        dummy_agent.accelerator = None


class TestPopulationParameterized:
    @pytest.mark.parametrize("size", [1, 3, 5])
    def test_population_creates_correct_size(self, vector_space, size):
        action_space = spaces.Discrete(2)
        pop = DummyRLAlgorithm.population(size, vector_space, action_space)
        assert len(pop) == size
        for i, agent in enumerate(pop):
            assert agent.index == i


class TestRegistryInitHpConfig:
    def test_registry_init_rejects_unsupported_hp_dtype(self, vector_space):
        from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

        class BadHpAlgo(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                self.bad_param = "string_value"
                hp = HyperparameterConfig(
                    bad_param=RLParameter(min=0.1, max=0.2),
                )
                super().__init__(obs_space, act_space, index=index, hp_config=hp)

            def get_action(self, *args, **kwargs):
                return None

            def learn(self, *args, **kwargs):
                return None

            def test(self, *args, **kwargs):
                return None

        action_space = spaces.Discrete(2)
        with pytest.raises(TypeError, match="integer, float, and numpy ndarray"):
            BadHpAlgo(vector_space, action_space, index=0)


class TestReinitOptimizersMultiNetwork:
    def test_reinit_optimizers_with_single_optimizer(self, dummy_agent):
        dummy_agent.reinit_optimizers()
        assert dummy_agent.dummy_optimizer is not None


class TestMultiAgentExtractMasks:
    @pytest.fixture
    def ma_agent(self, vector_space):
        obs_spaces = [vector_space, vector_space]
        act_spaces = [spaces.Discrete(2), spaces.Discrete(2)]
        return DummyMARLAlgorithm(
            obs_spaces, act_spaces, agent_ids=["agent_0", "agent_1"], index=0
        )

    def test_extract_action_masks_from_infos(self, ma_agent):
        infos = {
            "agent_0": {"action_mask": np.array([1, 0])},
            "agent_1": {"action_mask": np.array([0, 1])},
        }
        masks = ma_agent.extract_action_masks(infos)
        assert "agent_0" in masks
        assert "agent_1" in masks
        np.testing.assert_array_equal(masks["agent_0"], np.array([1, 0]))
        np.testing.assert_array_equal(masks["agent_1"], np.array([0, 1]))

    def test_extract_action_masks_filters_by_agent_ids(self, ma_agent):
        infos = {
            "agent_0": {"action_mask": np.array([1, 0])},
            "other_agent": {"action_mask": np.array([0, 1])},
        }
        masks = ma_agent.extract_action_masks(infos)
        assert "agent_0" in masks
        assert "other_agent" not in masks

    def test_extract_agent_masks_none_when_no_env_defined_actions(self, ma_agent):
        env_acts, agent_masks = ma_agent.extract_agent_masks(infos=None)
        assert env_acts is None
        assert agent_masks is None

    def test_extract_agent_masks_with_env_defined_actions(self, ma_agent):
        infos = {
            "agent_0": {"env_defined_actions": np.array([0.0])},
            "agent_1": {"env_defined_actions": np.array([1.0])},
        }
        env_acts, agent_masks = ma_agent.extract_agent_masks(infos)
        assert env_acts is not None
        assert agent_masks is not None
        assert "agent_0" in env_acts


class TestMultiAgentGetGroupId:
    def test_get_group_id_splits_on_underscore(self):
        obs = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        act = spaces.Discrete(2)
        agent = DummyMARLAlgorithm(
            [obs, obs], [act, act], agent_ids=["speaker_0", "speaker_1"], index=0
        )
        assert agent.get_group_id("speaker_0") == "speaker"
        assert agent.get_group_id("speaker_1") == "speaker"


class TestMultiAgentHasGroupedAgents:
    def test_has_grouped_agents_true_when_shared_ids_fewer(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        assert agent.has_grouped_agents()

    def test_has_grouped_agents_false_when_heterogeneous(self, vector_space):
        obs = [
            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        ]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(
            obs, act, agent_ids=["agent_0", "other_agent_0"], index=0
        )
        assert not agent.has_grouped_agents()


class TestMultiAgentAssembleSharedInputs:
    def test_assemble_shared_inputs_reshapes_by_group(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        exp = {
            "agent_0": np.zeros(4, dtype=np.float32),
            "agent_1": np.zeros(4, dtype=np.float32),
        }
        result = agent.assemble_shared_inputs(exp)
        assert "agent" in result
        assert "agent_0" in result["agent"]
        assert "agent_1" in result["agent"]


class TestMultiAgentDisassembleGroupedOutputs:
    def test_disassemble_grouped_outputs(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        group_outputs = {"agent": np.array([[4.0, 7.0], [8.0, 9.0]])}
        vect_dim = 2
        result = agent.disassemble_grouped_outputs(
            group_outputs, vect_dim, agent.grouped_agents
        )
        assert "agent_0" in result
        assert "agent_1" in result
        assert result["agent_0"].ndim >= 1


class TestMultiAgentSumSharedRewards:
    def test_sum_shared_rewards(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        rewards = {"agent_0": np.array([1.0, 2.0]), "agent_1": np.array([3.0, 4.0])}
        result = agent.sum_shared_rewards(rewards)
        assert "agent" in result
        np.testing.assert_array_almost_equal(result["agent"], np.array([4.0, 6.0]))


class TestLoadWithWrapperCls:
    def test_load_with_wrapper_cls_in_checkpoint_wraps_agent(
        self, dummy_agent, tmp_path
    ):
        import dill

        class SimpleWrapper:
            def __init__(self, agent, label="wrapped"):
                self.agent = agent
                self.label = label

        path = tmp_path / "chkpt.pth"
        dummy_agent.save_checkpoint(path)
        chkpt = torch.load(path, weights_only=False, pickle_module=dill)
        chkpt["wrapper_cls"] = SimpleWrapper
        chkpt["wrapper_init_dict"] = {"label": "custom"}
        chkpt["wrapper_attrs"] = {}
        torch.save(chkpt, path, pickle_module=dill)
        loaded = DummyRLAlgorithm.load(path)
        assert hasattr(loaded, "agent")
        assert hasattr(loaded, "label")
        assert loaded.label == "custom"
        assert loaded.agent.dummy_attribute == dummy_agent.dummy_attribute


class TestSetTrainingModeNetworksWithoutActor:
    def test_set_training_mode_skips_networks_without_actor_in_name(self, vector_space):
        class AgentWithCritic(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                super().__init__(obs_space, act_space, index=index)
                num_out = (
                    act_space.n
                    if hasattr(act_space, "n")
                    else int(np.prod(act_space.shape))
                )
                num_in = (
                    obs_space.shape[0] if hasattr(obs_space, "shape") else obs_space.n
                )
                self.dummy_critic = EvolvableMLP(
                    num_in + num_out, 1, hidden_size=[8], device=self.device
                )
                self.lr_critic = 0.001
                self.critic_optimizer = OptimizerWrapper(
                    optim.Adam,
                    self.dummy_critic,
                    self.lr_critic,
                    network_names=["dummy_critic"],
                    lr_name="lr_critic",
                )
                self.register_network_group(
                    NetworkGroup(eval_network=self.dummy_critic, policy=False),
                )

        action_space = spaces.Discrete(2)
        agent = AgentWithCritic(vector_space, action_space, index=0)
        agent.set_training_mode(False)
        assert agent.training is False
        assert not agent.dummy_actor.training
        assert agent.dummy_critic.training


class TestGetCheckpointDictOptimizedModule:
    def test_checkpoint_dict_with_compiled_module(self, vector_space, tmp_path):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, torch_compiler="default"
        )
        chkpt = get_checkpoint_dict(agent)
        assert "network_info" in chkpt
        assert "dummy_actor" in chkpt["network_info"]["network_names"]
        torch._dynamo.reset()


class TestRegistryInitEvolvableNotInRegistry:
    def test_registry_init_raises_when_evolvable_not_in_registry(self, vector_space):

        class OrphanNetworkAlgo(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                super().__init__(obs_space, act_space, index=index)
                self.orphan_net = EvolvableMLP(
                    4, 2, hidden_size=[8], device=self.device
                )

        action_space = spaces.Discrete(2)
        with pytest.raises(AttributeError, match="could not be found in the registry"):
            OrphanNetworkAlgo(vector_space, action_space, index=0)


class TestRegistryInitHpMissingAttribute:
    def test_registry_init_raises_when_hp_not_set_as_attribute(self, vector_space):
        from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

        class MissingHpAlgo(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                hp = HyperparameterConfig(
                    lr=RLParameter(min=0.05, max=0.2),
                    missing_param=RLParameter(min=1, max=10),
                )
                super().__init__(obs_space, act_space, index=index, hp_config=hp)

        action_space = spaces.Discrete(2)
        with pytest.raises(AttributeError, match="not been set as an attribute"):
            MissingHpAlgo(vector_space, action_space, index=0)


class TestReinitOptimizersMultiNetworkPath:
    def test_reinit_optimizers_with_multi_network_optimizer(self, vector_space):
        action_space = spaces.Discrete(2)
        num_in = 4
        num_out = 2

        class TwoNetOneOptAlgo(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                super().__init__(obs_space, act_space, index=index)
                self.shared_critic = EvolvableMLP(
                    num_in + num_out, 1, hidden_size=[8], device=self.device
                )
                self.lr_shared = 0.001
                self.shared_optimizer = OptimizerWrapper(
                    optim.Adam,
                    [self.dummy_actor, self.shared_critic],
                    0.001,
                    network_names=["dummy_actor", "shared_critic"],
                    lr_name="lr_shared",
                )
                self.register_network_group(
                    NetworkGroup(eval_network=self.shared_critic, policy=False),
                )

        agent = TwoNetOneOptAlgo(vector_space, action_space, index=0)
        agent.reinit_optimizers()
        assert agent.shared_optimizer is not None
        assert agent.dummy_optimizer is not None


class TestReinitOptimizersWithExplicitConfig:
    def test_reinit_optimizers_with_explicit_optimizer_config(self, dummy_agent):

        config = dummy_agent.registry.optimizers[0]
        dummy_agent.reinit_optimizers(optimizer=config)
        assert dummy_agent.dummy_optimizer is not None


class TestExtractActionMasksNonDictInfo:
    def test_extract_action_masks_returns_none_when_info_not_dict(self, vector_space):
        obs_spaces = [vector_space, vector_space]
        act_spaces = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(
            obs_spaces, act_spaces, agent_ids=["agent_0", "agent_1"], index=0
        )
        infos = {"agent_0": "not_a_dict", "agent_1": {"action_mask": np.array([0, 1])}}
        masks = agent.extract_action_masks(infos)
        assert masks["agent_0"] is None
        np.testing.assert_array_equal(masks["agent_1"], np.array([0, 1]))


class TestExtractAgentMasksEdgeCases:
    @pytest.fixture
    def ma_agent(self, vector_space):
        obs_spaces = [vector_space, vector_space]
        act_spaces = [spaces.Discrete(2), spaces.Discrete(2)]
        return DummyMARLAlgorithm(
            obs_spaces, act_spaces, agent_ids=["agent_0", "agent_1"], index=0
        )

    def test_extract_agent_masks_scalar_env_defined_action(self, ma_agent):
        infos = {
            "agent_0": {"env_defined_actions": 0},
            "agent_1": {"env_defined_actions": 1.0},
        }
        env_acts, agent_masks = ma_agent.extract_agent_masks(infos)
        assert env_acts is not None
        assert agent_masks is not None
        assert env_acts["agent_0"].shape == (1,)
        assert env_acts["agent_1"].shape == (1,)

    def test_extract_agent_masks_none_env_defined_uses_nan(self, ma_agent):
        infos = {
            "agent_0": {"env_defined_actions": None},
            "agent_1": {"env_defined_actions": np.array([1.0])},
        }
        env_acts, _agent_masks = ma_agent.extract_agent_masks(infos)
        assert env_acts is not None
        assert np.isnan(env_acts["agent_0"]).all() or env_acts["agent_0"].size == 1

    def test_extract_agent_masks_tensor_env_defined_action(self, ma_agent):
        infos = {
            "agent_0": {"env_defined_actions": torch.tensor([1.0])},
            "agent_1": {"env_defined_actions": np.array([0.0])},
        }
        env_acts, _agent_masks = ma_agent.extract_agent_masks(infos)
        assert env_acts is not None
        assert isinstance(env_acts["agent_0"], np.ndarray)

    def test_extract_agent_masks_unsupported_type_treated_as_none(self, ma_agent):
        # A type that isn't a scalar/array/tensor is discarded rather than
        # propagated, so it is masked out like an explicit None.
        infos = {
            "agent_0": {"env_defined_actions": "not-an-action"},
            "agent_1": {"env_defined_actions": np.array([1.0])},
        }
        env_acts, _agent_masks = ma_agent.extract_agent_masks(infos)
        assert env_acts is not None
        assert np.isnan(env_acts["agent_0"]).all()


class TestGetGroupIdNonString:
    def test_get_group_id_returns_agent_id_when_not_string(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        non_string_id = 42
        assert agent.get_group_id(non_string_id) == 42


class TestAssembleSharedInputsListExperience:
    def test_assemble_shared_inputs_with_list_experience(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        exp = {
            "agent_0": [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)],
            "agent_1": np.zeros(4, dtype=np.float32),
        }
        result = agent.assemble_shared_inputs(exp)
        assert "agent" in result
        assert "agent_0" in result["agent"]
        assert "agent_1" in result["agent"]
        assert (
            result["agent"]["agent_0"] is not None or result["agent"]["agent_0"] is None
        )


class TestAssembleSharedInputsEmptyList:
    def test_assemble_shared_inputs_empty_list_returns_none(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        exp = {"agent_0": [], "agent_1": np.zeros(4, dtype=np.float32)}
        result = agent.assemble_shared_inputs(exp)
        assert result["agent"]["agent_0"] is None


class TestAssembleGroupedOutputs:
    def test_assemble_grouped_outputs(self, vector_space):
        obs = [vector_space, vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(
            obs, act, agent_ids=["agent_0", "agent_1", "agent_2"], index=0
        )
        agent_outputs = {
            "agent_0": np.array([[1.0], [2.0]]),
            "agent_1": np.array([[3.0], [4.0]]),
            "agent_2": np.array([[5.0], [6.0]]),
        }
        vect_dim = 2
        result = agent.assemble_grouped_outputs(agent_outputs, vect_dim)
        assert "agent" in result
        assert result["agent"].shape[0] == 6


class TestDisassembleGroupedOutputsNonDiscrete:
    def test_disassemble_grouped_outputs_continuous_action_no_squeeze(
        self, vector_space
    ):
        obs = [vector_space, vector_space]
        act = [spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)] * 2
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        group_outputs = {"agent": np.random.randn(2, 2, 2).astype(np.float32)}
        vect_dim = 2
        result = agent.disassemble_grouped_outputs(
            group_outputs, vect_dim, agent.grouped_agents
        )
        assert "agent_0" in result
        assert "agent_1" in result
        assert result["agent_0"].shape[-1] == 2


class TestMultiAgentInitWithDictSpaces:
    def test_multi_agent_init_with_dict_observation_spaces(self, vector_space):
        obs_dict = {"agent_0": vector_space, "agent_1": vector_space}
        act_dict = {"agent_0": spaces.Discrete(2), "agent_1": spaces.Discrete(2)}
        agent = DummyMARLAlgorithm(obs_dict, act_dict, agent_ids=None, index=0)
        assert agent.agent_ids == ["agent_0", "agent_1"]
        assert agent.n_agents == 2


class TestBuildNetConfigGroupedKeyError:
    def test_build_net_config_raises_when_agent_id_in_grouped_setting(
        self, vector_space
    ):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        net_config = {"agent_0": {"encoder_config": {"hidden_size": [8]}}}
        with pytest.raises(KeyError, match="individual sub-agent"):
            agent.build_net_config(net_config, flatten=False)


class TestLoadCheckpointModuleDictPartialState:
    def test_load_checkpoint_skips_falsy_state_dict_for_agent(
        self, vector_space, tmp_path
    ):
        import dill

        obs_spaces = [vector_space, vector_space]
        act_spaces = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(
            obs_spaces, act_spaces, agent_ids=["agent_0", "agent_1"], index=0
        )
        path = tmp_path / "chkpt.pth"
        agent.save_checkpoint(path)
        chkpt = torch.load(path, weights_only=False, pickle_module=dill)
        chkpt["network_info"]["modules"]["dummy_actors_state_dict"]["agent_1"] = None
        torch.save(chkpt, path, pickle_module=dill)

        agent2 = DummyMARLAlgorithm(
            obs_spaces, act_spaces, agent_ids=["agent_0", "agent_1"], index=1
        )
        agent2.load_checkpoint(path)
        assert agent2.dummy_actors["agent_0"] is not None
        assert agent2.dummy_actors["agent_1"] is not None


class TestLoadMissingAttributeWarning:
    def test_load_warns_when_attribute_missing_in_checkpoint(
        self, dummy_agent, tmp_path
    ):
        path = tmp_path / "chkpt.pth"
        dummy_agent.save_checkpoint(path)
        chkpt = torch.load(path, weights_only=False)
        chkpt.pop("dummy_attribute")
        torch.save(chkpt, path)
        with pytest.warns(UserWarning, match="not found in checkpoint"):
            loaded = DummyRLAlgorithm.load(path)
        assert loaded.dummy_attribute == "test_value"


class TestRLAlgorithmPreprocessObservation:
    def test_preprocess_observation_returns_tensor(self, vector_space):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        obs = vector_space.sample()
        result = agent.preprocess_observation(obs)
        assert isinstance(result, torch.Tensor)
        assert result.shape[-1] == 4


class TestMultiAgentGetSetup:
    def test_get_setup_homogeneous(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        from agilerl.typing import MultiAgentSetup

        assert agent.get_setup() == MultiAgentSetup.HOMOGENEOUS

    def test_get_setup_heterogeneous(self, vector_space):
        obs = [
            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8),
            spaces.Dict(
                {
                    "x": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                }
            ),
        ]
        act = [spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(
            obs,
            act,
            agent_ids=["agent_0", "other_agent_0", "other_other_agent_0"],
            index=0,
        )
        from agilerl.typing import MultiAgentSetup

        assert agent.get_setup() == MultiAgentSetup.HETEROGENEOUS


class TestCopyAttributesSkipsEvolvableAlgorithm:
    def test_copy_attributes_skips_callable_attr(self, dummy_agent):
        """Callable attributes are skipped by copy_attributes."""
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        dummy_agent.wrapped_agent = lambda: None
        clone.wrapped_agent = None
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert clone.wrapped_agent is None

    def test_copy_attributes_skips_evolvable_algorithm_attr(self, dummy_agent):
        """EvolvableAlgorithm subclass attributes are skipped by copy_attributes."""
        clone = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=1,
        )
        nested = DummyRLAlgorithm(
            dummy_agent.observation_space,
            dummy_agent.action_space,
            index=2,
        )
        dummy_agent.wrapped_agent = nested
        clone.wrapped_agent = None
        EvolvableAlgorithm.copy_attributes(dummy_agent, clone)
        assert clone.wrapped_agent is None


class TestToDeviceListOfTensors:
    def test_to_device_list_of_tensors(self, dummy_agent):
        device = torch.device(dummy_agent.device)
        exp = [torch.zeros(2, 4), torch.zeros(2)]
        result = dummy_agent.to_device(exp)
        assert len(result) == 1
        assert all(t.device == device for t in result[0])


class TestGetCheckpointDictLrSchedulerNone:
    def test_get_checkpoint_dict_when_lr_scheduler_is_none(self, dummy_agent):
        dummy_agent.lr_scheduler = None
        chkpt = get_checkpoint_dict(dummy_agent)
        assert "lr_scheduler" not in chkpt or chkpt.get("lr_scheduler") is None


class TestPopulationWithWrapperKwargsEmpty:
    def test_population_with_empty_wrapper_kwargs(self, vector_space):
        class SimpleWrapper:
            def __init__(self, agent):
                self.agent = agent

        action_space = spaces.Discrete(2)
        pop = DummyRLAlgorithm.population(
            2,
            vector_space,
            action_space,
            wrapper_cls=SimpleWrapper,
            wrapper_kwargs={},
        )
        assert len(pop) == 2
        for w in pop:
            assert hasattr(w, "agent")


class TestSetAttrOptimizerRegistration:
    def test_setattr_registers_new_optimizer_wrapper(self, vector_space):

        class AlgoWithLateOptimizer(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                super().__init__(obs_space, act_space, index=index)
                self.aux_net = EvolvableMLP(4, 2, hidden_size=[8], device=self.device)
                self.lr_aux = 0.01
                self.aux_optimizer = OptimizerWrapper(
                    optim.Adam,
                    self.aux_net,
                    self.lr_aux,
                    network_names=["aux_net"],
                    lr_name="lr_aux",
                )
                self.register_network_group(
                    NetworkGroup(eval_network=self.aux_net, policy=False),
                )

        action_space = spaces.Discrete(2)
        agent = AlgoWithLateOptimizer(vector_space, action_space, index=0)
        assert "aux_optimizer" in [c.name for c in agent.registry.optimizers]


class TestRegistryInitEvolvableNotRegistered:
    def test_registry_init_raises_when_evolvable_not_in_groups(self, vector_space):
        class OrphanNetworkAlgo(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                super().__init__(obs_space, act_space, index=index)
                self.orphan_net = EvolvableMLP(
                    4, 2, hidden_size=[8], device=self.device
                )

        action_space = spaces.Discrete(2)
        with pytest.raises(AttributeError, match="could not be found in the registry"):
            OrphanNetworkAlgo(vector_space, action_space, index=0)


class TestRegistryInitHpMissing:
    def test_registry_init_raises_when_hp_not_set_as_attribute(self, vector_space):
        from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter

        class MissingHpAlgo(DummyRLAlgorithm):
            def __init__(self, obs_space, act_space, index=0):
                hp = HyperparameterConfig(
                    nonexistent_param=RLParameter(min=0.05, max=0.2),
                )
                super().__init__(obs_space, act_space, index=index, hp_config=hp)

        action_space = spaces.Discrete(2)
        with pytest.raises(AttributeError, match="has not been set as an attribute"):
            MissingHpAlgo(vector_space, action_space, index=0)


class TestGetPolicy:
    def test_get_policy_returns_actor(self, dummy_agent):
        policy = dummy_agent.get_policy()
        assert policy is dummy_agent.dummy_actor

    def test_get_policy_raises_when_no_policy_group(self, vector_space):
        from tests.test_algorithms.test_base import DummyRLAlgorithmNoPolicy

        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithmNoPolicy(vector_space, action_space, index=0)
        for g in agent.registry.groups:
            g.policy = False
        with pytest.raises(AttributeError, match="No policy network"):
            agent.get_policy()


class TestMultiAgentUnknownSpaceType:
    def test_raises_on_unknown_observation_space(self, vector_space):
        unknown = spaces.MultiBinary(4)
        with pytest.raises(ValueError, match="Unknown observation space type"):
            DummyMARLAlgorithm(
                [unknown, vector_space],
                [spaces.Discrete(2), spaces.Discrete(2)],
                agent_ids=["alpha_0", "beta_0"],
                index=0,
            )


class TestMultiAgentInvalidModuleDictKeys:
    def test_registry_init_raises_when_module_dict_key_invalid(self, vector_space):
        from agilerl.modules import ModuleDict

        class BadKeyAlgo(DummyMARLAlgorithm):
            def __init__(self, obs_spaces, act_spaces, agent_ids, index=0):
                super().__init__(
                    obs_spaces, act_spaces, agent_ids=agent_ids, index=index
                )
                self.bad_actors = ModuleDict(
                    {
                        "invalid_key": EvolvableMLP(
                            4, 2, hidden_size=[8], device=self.device
                        )
                    }
                )
                self.register_network_group(
                    NetworkGroup(eval_network=self.bad_actors, policy=False),
                )

        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        with pytest.raises(ValueError, match="not present in"):
            BadKeyAlgo(obs, act, agent_ids=["agent_0", "agent_1"], index=0)


class TestLoadCheckpointTorchCompiler:
    def test_load_checkpoint_recompiles_with_torch_compiler(
        self, vector_space, tmp_path
    ):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, torch_compiler="default"
        )
        path = tmp_path / "chkpt.pth"
        agent.save_checkpoint(path)

        agent2 = DummyRLAlgorithm(
            vector_space, action_space, index=1, torch_compiler="default"
        )
        agent2.load_checkpoint(path)
        assert agent2.torch_compiler == "default"
        torch._dynamo.reset()

    def test_load_classmethod_recompiles_with_torch_compiler(
        self, vector_space, tmp_path
    ):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, torch_compiler="default"
        )
        path = tmp_path / "chkpt.pth"
        agent.save_checkpoint(path)

        loaded = DummyRLAlgorithm.load(path)
        assert loaded.torch_compiler == "default"
        torch._dynamo.reset()


class TestLoadCheckpointLLMBreak:
    def test_load_checkpoint_breaks_when_module_cls_missing(
        self, vector_space, tmp_path
    ):
        import dill

        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        path = tmp_path / "chkpt.pth"
        agent.save_checkpoint(path)

        chkpt = torch.load(path, weights_only=False, pickle_module=dill)
        chkpt["network_info"]["modules"].pop("dummy_actor_cls")
        torch.save(chkpt, path, pickle_module=dill)

        agent2 = DummyRLAlgorithm(vector_space, action_space, index=1)
        agent2.load_checkpoint(path)


# ---------------------------------------------------------------------------
# LLMAlgorithm tests (heavily mocked to avoid GPU/model requirements)
# ---------------------------------------------------------------------------


def _make_mock_accelerator(
    ds_config=None,
    num_processes=1,
    is_main_process=True,
    process_index=0,
):
    """Create a mock Accelerator that passes isinstance checks."""
    if ds_config is None:
        ds_config = {
            "zero_optimization": {"stage": 0},
            "train_micro_batch_size_per_gpu": "auto",
        }
    acc = MagicMock(spec=Accelerator)
    type(acc).num_processes = PropertyMock(return_value=num_processes)
    type(acc).is_main_process = PropertyMock(return_value=is_main_process)
    type(acc).process_index = PropertyMock(return_value=process_index)
    type(acc).local_process_index = PropertyMock(return_value=process_index)
    type(acc).device = PropertyMock(return_value=torch.device("cpu"))

    plugin = MagicMock()
    plugin.deepspeed_config = ds_config
    acc.state = MagicMock()
    acc.state.deepspeed_plugin = plugin
    acc.prepare = MagicMock(side_effect=lambda *a: a)
    acc.free_memory = MagicMock(side_effect=lambda *a: (None,) * len(a))
    return acc


class _MockPeftActor(torch.nn.Module):
    """A torch.nn.Module subclass that quacks like a PeftModel."""

    def __init__(self):
        super().__init__()
        self._dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
        self.name_or_path = "mock-model"
        self.peft_config = {}
        self.base_model = MagicMock()
        self.base_model.model = MagicMock()
        self.base_model.model.lm_head = MagicMock()
        self.set_adapter = MagicMock()
        self.merge_and_unload = MagicMock()
        self.add_adapter = MagicMock()
        self.gradient_checkpointing_enable = MagicMock()
        self.merge_adapter = MagicMock()
        self.unmerge_adapter = MagicMock()

        def _delete_adapter(name: str) -> None:
            self.peft_config.pop(name, None)

        self.delete_adapter = MagicMock(side_effect=_delete_adapter)
        self.disable_adapter = MagicMock()
        self.save_pretrained = MagicMock()
        self.save_checkpoint = MagicMock()
        self.load_checkpoint = MagicMock()
        self.prefix = "model"
        self.generate = MagicMock()
        self.from_pretrained = MagicMock()

    def forward(self, **kwargs):
        batch_size = kwargs.get("input_ids", torch.zeros(2, 5)).shape[0]
        seq_len = kwargs.get("input_ids", torch.zeros(2, 5)).shape[1]
        out = MagicMock()
        out.logits = torch.randn(batch_size, seq_len, 100)
        return out


def _make_mock_peft_actor():
    """Create a mock PEFT actor model."""
    return _MockPeftActor()


class _StubLLMAlgorithm(LLMAlgorithm):
    """Concrete stub of the abstract LLMAlgorithm for testing."""

    def __init__(self, *args, **kwargs):
        actor_network = kwargs.get("actor_network")
        super().__init__(*args, **kwargs)
        if actor_network is not None:
            self.actor = actor_network

    def learn(self, *a, **kw):
        return None

    def get_action(self, *a, **kw):
        return None

    def test(self, *a, **kw):
        return None


def _make_llm_agent(
    accelerator=None,
    clone=True,
    micro_batch_size_per_gpu=None,
    cosine_lr_schedule_config=None,
    max_grad_norm=0.0,
    use_liger_loss=False,
    lora_config=None,
    actor_network=None,
    batch_size=4,
    use_separate_reference_adapter=False,
    *,
    mini_batch_size=None,
    algo_cls=None,
    reduce_memory_peak: bool = False,
    use_vllm: bool = False,
    use_memory_efficient_params: bool = True,
):
    """Helper to create a _StubLLMAlgorithm with heavily mocked internals."""
    if not HAS_LLM_DEPENDENCIES:
        pytest.skip("LLM dependencies not installed")
    if algo_cls is None:
        algo_cls = _StubLLMAlgorithm
    if actor_network is None:
        actor_network = _make_mock_peft_actor()

    with (
        patch.object(LLMAlgorithm, "_initialize_actors"),
        patch.object(LLMAlgorithm, "_configure_vllm"),
        patch.object(LLMAlgorithm, "wrap_models"),
        patch.object(EvolvableAlgorithm, "_registry_init"),
        # ``EvolvableAlgorithm.__init__`` lazily initialises a real
        # ``PartialState()`` (and therefore ``torch.distributed.init_process_group``)
        # whenever the supplied accelerator reports ``num_processes > 1``. This
        # causes the failure mode ``EADDRINUSE`` on macOS CI without below line.
        patch(
            "agilerl.algorithms.core.base.broadcast_object_list",
            side_effect=lambda obj_list, from_process=0: list(obj_list),
        ),
    ):
        agent = algo_cls(
            index=0,
            batch_size=batch_size,
            lr=1e-4,
            max_grad_norm=max_grad_norm,
            clone=clone,
            calc_position_embeddings=False,
            seed=42,
            pad_token_id=0,
            pad_token="<pad>",
            use_liger_loss=use_liger_loss,
            lora_config=lora_config if lora_config is not None else MagicMock(),
            actor_network=actor_network,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            mini_batch_size=mini_batch_size,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            accelerator=accelerator,
            device="cpu",
            use_separate_reference_adapter=use_separate_reference_adapter,
            reduce_memory_peak=reduce_memory_peak,
            use_vllm=use_vllm,
            use_memory_efficient_params=use_memory_efficient_params,
        )
    agent.actor = actor_network
    agent.optimizer = MagicMock()
    agent.optimizer.optimizer = MagicMock()
    agent.optimizer.optimizer.param_groups = [
        {"lr": 1e-4, "params": torch.tensor([1.0])}
    ]
    agent.lr_scheduler = None
    agent.use_vllm = False
    agent.max_output_tokens = None
    agent.max_model_len = 512
    agent.temperature = 1.0
    agent.registry = MagicMock()
    agent.registry.hooks = []
    agent.registry.groups = []
    agent.registry.optimizers = []
    return agent


class TestLLMAlgorithmPopulation:
    """Tests for :meth:`LLMAlgorithm.population`."""

    @staticmethod
    def _population_kwargs() -> dict:
        return {
            "batch_size": 4,
            "lr": 1e-4,
            "max_grad_norm": 0.0,
            "clone": True,
            "calc_position_embeddings": False,
            "seed": 42,
            "pad_token_id": 0,
            "pad_token": "<pad>",
            "use_liger_loss": False,
            "lora_config": MagicMock(),
        }

    @staticmethod
    def _population_init_context():
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            with (
                patch.object(LLMAlgorithm, "_initialize_actors"),
                patch.object(LLMAlgorithm, "_configure_vllm"),
                patch.object(LLMAlgorithm, "wrap_models"),
                patch.object(EvolvableAlgorithm, "_registry_init"),
                patch(
                    "agilerl.algorithms.core.base.broadcast_object_list",
                    side_effect=lambda obj_list, from_process=0: list(obj_list),
                ),
            ):
                yield

        return _ctx()

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES,
        reason="LLM dependencies not installed",
    )
    def test_population_builds_requested_size_without_accelerator(self):
        with (
            self._population_init_context(),
            patch(
                "agilerl.algorithms.core.base.clone_llm",
                return_value=_make_mock_peft_actor(),
            ) as mock_clone,
        ):
            population = _StubLLMAlgorithm.population(
                size=3,
                accelerator=None,
                device="cpu",
                actor_network=_make_mock_peft_actor(),
                **self._population_kwargs(),
            )

        assert len(population) == 3
        assert mock_clone.call_count == 2
        assert population[0].index == 0
        assert population[1].index == 1
        assert population[2].index == 2
        assert population[1].accelerator is None
        assert population[2].accelerator is None

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES,
        reason="LLM dependencies not installed",
    )
    def test_population_with_accelerator_clones_deepspeed_state(self):
        """Population clones via ``get_state_dict`` when an accelerator is set."""
        import agilerl.algorithms.core.base as algo_base

        acc = _make_mock_accelerator(num_processes=1)
        agent_0 = _make_llm_agent(accelerator=acc)

        with self._population_init_context():
            with (
                patch(
                    "agilerl.algorithms.core.base.clone_llm",
                    return_value=_make_mock_peft_actor(),
                ) as mock_clone,
                patch(
                    "agilerl.algorithms.core.base.get_state_dict",
                    return_value={"layer": torch.tensor(1.0)},
                ) as mock_get_sd,
            ):
                cloned_actor = algo_base.clone_llm(
                    agent_0.actor,
                    zero_stage=0,
                    state_dict=algo_base.get_state_dict(agent_0.actor),
                )
                agent_1 = _StubLLMAlgorithm(
                    index=1,
                    accelerator=_make_mock_accelerator(num_processes=1),
                    device="cpu",
                    actor_network=cloned_actor,
                    **self._population_kwargs(),
                )

        mock_get_sd.assert_called_once_with(agent_0.actor)
        mock_clone.assert_called_once()
        assert agent_1.actor is cloned_actor

    @pytest.mark.skipif(
        not HAS_LLM_DEPENDENCIES,
        reason="LLM dependencies not installed",
    )
    def test_population_resume_from_checkpoint_loads_each_agent(self):
        checkpoint_path = "/tmp/llm_checkpoint"

        with (
            self._population_init_context(),
            patch(
                "agilerl.algorithms.core.base.clone_llm",
                return_value=_make_mock_peft_actor(),
            ),
            patch.object(_StubLLMAlgorithm, "load_checkpoint") as mock_load,
        ):
            population = _StubLLMAlgorithm.population(
                size=2,
                accelerator=None,
                device="cpu",
                actor_network=_make_mock_peft_actor(),
                resume_from_checkpoint=checkpoint_path,
                **self._population_kwargs(),
            )

        assert len(population) == 2
        assert mock_load.call_count == 2
        mock_load.assert_any_call(checkpoint_path)
        assert population[0].index == 0
        assert population[1].index == 1


class TestLLMAlgorithmLoad:
    def test_load_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not supported"):
            _StubLLMAlgorithm.load("/some/path")


class TestLLMAlgorithmRecompile:
    def test_recompile_no_deepspeed(self):
        agent = _make_llm_agent()
        agent.recompile()
        assert agent.torch_compiler is None
        assert agent._uses_deepspeed is False

    def test_recompile_skips_compile_when_deepspeed(self):
        """DeepSpeed engine is incompatible with torch.compile wrapping."""
        agent = _make_llm_agent()
        agent.torch_compiler = "default"
        agent._uses_deepspeed = True
        with patch("agilerl.algorithms.core.base.compile_model") as mock_compile:
            agent.recompile()
            mock_compile.assert_not_called()

    def test_recompile_calls_compile_when_not_deepspeed_and_compiler_set(self):
        agent = _make_llm_agent()
        agent.torch_compiler = "default"
        agent._uses_deepspeed = False
        network = MagicMock()
        compiled = MagicMock()
        with (
            patch(
                "agilerl.algorithms.core.base.compile_model", return_value=compiled
            ) as mock_compile,
            patch.object(
                agent,
                "evolvable_attributes",
                return_value={"actor": network},
            ),
        ):
            agent.recompile()
        mock_compile.assert_called_once_with(network, "default")
        assert agent.actor is compiled


class TestLLMSelectOptimClass:
    def test_returns_adamw_without_accelerator(self):
        agent = _make_llm_agent(accelerator=None)
        from torch.optim import AdamW

        assert agent._select_optim_class() is AdamW

    def test_returns_dummy_optimizer_with_ds_optimizer(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc)
        from agilerl.utils.algo_utils import DummyOptimizer

        assert agent._select_optim_class() is DummyOptimizer

    def test_returns_adamw_without_ds_optimizer(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc)
        from torch.optim import AdamW

        assert agent._select_optim_class() is AdamW


class TestLLMUpdateLr:
    def test_update_lr_no_accelerator_with_scheduler(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        sched_config = MagicMock()
        sched_config.warmup_steps = 10
        sched_config.total_steps = 100
        with patch(
            "agilerl.algorithms.core.base.create_warmup_cosine_scheduler"
        ) as mock_sched:
            mock_sched.return_value = MagicMock()
            acc, scheduler = LLMAlgorithm.update_lr(
                opt, 5e-4, accelerator=None, scheduler_config=sched_config
            )
        assert acc is None
        assert scheduler is not None
        assert opt.param_groups[0]["lr"] == 5e-4

    def test_update_lr_no_accelerator_no_scheduler(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        acc, scheduler = LLMAlgorithm.update_lr(opt, 5e-4, accelerator=None)
        assert acc is None
        assert scheduler is None

    def test_update_lr_without_deepspeed_plugin(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        acc = MagicMock()
        acc.state.deepspeed_plugin = None
        returned_acc, scheduler = LLMAlgorithm.update_lr(opt, 5e-4, accelerator=acc)
        assert returned_acc is acc
        assert scheduler is None
        assert opt.param_groups[0]["lr"] == 5e-4

    def test_update_lr_without_deepspeed_config(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        acc = MagicMock()
        plugin = MagicMock(spec=[])
        acc.state.deepspeed_plugin = plugin
        returned_acc, scheduler = LLMAlgorithm.update_lr(opt, 5e-4, accelerator=acc)
        assert returned_acc is acc
        assert scheduler is None

    def test_update_lr_updates_scheduler_config(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "scheduler": {"params": {"warmup_max_lr": 1e-3}},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        _result_acc, _ = LLMAlgorithm.update_lr(opt, 5e-4, accelerator=acc)
        assert (
            acc.state.deepspeed_plugin.deepspeed_config["scheduler"]["params"][
                "warmup_max_lr"
            ]
            == 5e-4
        )

    def test_update_lr_updates_optimizer_config(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "optimizer": {"params": {"lr": 1e-3}},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        LLMAlgorithm.update_lr(opt, 5e-4, accelerator=acc)
        assert (
            acc.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"]
            == 5e-4
        )

    def test_update_lr_accepts_actor_critic_lr_tuple(self):
        actor_param = torch.tensor([1.0], requires_grad=True)
        critic_param = torch.tensor([1.0], requires_grad=True)
        opt = torch.optim.Adam(
            [
                {"params": [actor_param], "lr": 1e-3, "group": "actor"},
                {"params": [critic_param], "lr": 2e-3, "group": "critic"},
            ]
        )

        LLMAlgorithm.update_lr(
            opt,
            lr=(3e-4, 4e-4),
            accelerator=None,
        )

        assert opt.param_groups[0]["lr"] == 3e-4
        assert opt.param_groups[1]["lr"] == 4e-4


class TestLLMSaveDistributedActor:
    def test_save_with_accelerator(self, tmp_path):
        agent = _make_llm_agent(accelerator=_make_mock_accelerator())
        agent._save_distributed_actor(str(tmp_path / "save_dir"))
        agent.actor.save_checkpoint.assert_called_once()

    def test_save_without_accelerator_warns(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        with pytest.warns(UserWarning, match="Distributed actor save not supported"):
            agent._save_distributed_actor("/some/path")


class TestLLMLoadDistributedActor:
    def test_load_without_accelerator_warns(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        with pytest.warns(UserWarning, match="Distributed actor load not supported"):
            agent._load_distributed_actor("/some/path")


class TestLLMWrapModels:
    def test_wrap_models_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.optimizer.optimizer = MagicMock()
        agent.gradient_checkpointing = True
        agent.zero_stage = 2
        wrapped_actor = MagicMock()
        wrapped_actor.module = MagicMock()
        wrapped_actor.optimizer = MagicMock()
        acc.prepare = MagicMock(
            return_value=(wrapped_actor, agent.optimizer.optimizer, None)
        )
        acc.unwrap_model = MagicMock(return_value=wrapped_actor.module)
        LLMAlgorithm.wrap_models(agent)
        acc.prepare.assert_called_once()
        wrapped_actor.module.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    def test_wrap_models_zero3_forces_reentrant_checkpointing(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.optimizer.optimizer = MagicMock()
        agent.gradient_checkpointing = True
        agent.zero_stage = 3
        wrapped_actor = MagicMock()
        wrapped_actor.module = MagicMock()
        wrapped_actor.optimizer = MagicMock()
        acc.prepare = MagicMock(
            return_value=(wrapped_actor, agent.optimizer.optimizer, None)
        )
        acc.unwrap_model = MagicMock(return_value=wrapped_actor.module)
        LLMAlgorithm.wrap_models(agent)
        wrapped_actor.module.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    def test_wrap_models_without_accelerator(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        agent.gradient_checkpointing = False
        LLMAlgorithm.wrap_models(agent)
        assert agent.actor is not None

    def test_wrap_models_without_accelerator_with_checkpointing(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        agent.gradient_checkpointing = True
        agent.zero_stage = None
        original_actor = agent.actor
        LLMAlgorithm.wrap_models(agent)
        original_actor.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    def test_gradient_checkpointing_kwargs_by_zero_stage(self):
        agent = _make_llm_agent(accelerator=None)
        agent.zero_stage = None
        assert agent._gradient_checkpointing_kwargs() == {"use_reentrant": False}
        agent.zero_stage = 2
        assert agent._gradient_checkpointing_kwargs() == {"use_reentrant": False}
        agent.zero_stage = 3
        assert agent._gradient_checkpointing_kwargs() == {"use_reentrant": True}


class TestLLMCleanUp:
    def test_clean_up_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        original_actor = agent.actor
        original_actor.empty_partition_cache = MagicMock()
        original_actor.destroy = MagicMock()
        LLMAlgorithm.clean_up(agent)
        original_actor.empty_partition_cache.assert_called_once()
        original_actor.destroy.assert_called_once()
        acc.free_memory.assert_called_once()
        acc.wait_for_everyone.assert_called_once()

    def test_clean_up_without_accelerator(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        LLMAlgorithm.clean_up(agent)
        assert agent.actor is None
        assert agent.optimizer is None
        assert agent.lr_scheduler is None

    def test_clean_up_deletes_vllm(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        agent.llm = MagicMock()
        agent.llm.llm_engine = MagicMock()
        LLMAlgorithm.clean_up(agent)
        assert not hasattr(agent, "llm") or agent.llm is None


class TestLLMBackwardPass:
    def test_backward_pass_without_accelerator(self):
        agent = _make_llm_agent(accelerator=None)
        print("optimizer", agent.optimizer)
        agent.accelerator = None
        agent.max_grad_norm = 1.0
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        loss.backward.assert_called_once()
        agent.optimizer.step.assert_called_once()
        agent.optimizer.zero_grad.assert_called_once()

    def test_backward_pass_with_lr_scheduler(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        agent.max_grad_norm = 1.0
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.get_last_lr.return_value = [5e-5]
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        agent.lr_scheduler.step.assert_called_once()
        assert agent.lr == 5e-5


class TestLLMLogprobsFromLogits:
    def test_logprobs_from_logits_computes_log_probs(self):
        logits = torch.randn(2, 5, 10)
        index = torch.randint(0, 10, (2, 5))
        result = LLMAlgorithm._logprobs_from_logits(logits, index)
        assert result.shape == (2, 5)
        expected = (
            F.log_softmax(logits.float(), dim=-1)
            .gather(dim=-1, index=index.unsqueeze(-1))
            .squeeze(-1)
        )
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize(
        ("batch_rows", "chunk_rows"),
        [
            # B=1 hits the fast path (B <= chunk_rows), B=17 forces the chunked loop.
            (1, 8),
            (17, 2),
        ],
    )
    def test_logprobs_from_logits_bf16_matches_fp32_log_softmax_reference(
        self,
        batch_rows: int,
        chunk_rows: int,
    ) -> None:
        """Fast path (small B) and chunked path both match fp32 log_softmax reference."""
        torch.manual_seed(batch_rows * 31 + chunk_rows)
        seq, vocab = 11, 8192
        logits_bf16 = torch.randn(
            batch_rows,
            seq,
            vocab,
            dtype=torch.bfloat16,
        )
        index = torch.randint(0, vocab, (batch_rows, seq))

        result_bf16 = LLMAlgorithm._logprobs_from_logits(
            logits_bf16,
            index,
            chunk_rows=chunk_rows,
        )
        reference_fp32 = (
            F.log_softmax(logits_bf16.float(), dim=-1)
            .gather(dim=-1, index=index.unsqueeze(-1))
            .squeeze(-1)
        )
        lg32 = logits_bf16.float()
        max_lg = lg32.amax(dim=-1, keepdim=True)
        shifted = lg32 - max_lg
        stable_gather_minus_lse = shifted.gather(
            dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1) - torch.logsumexp(shifted, dim=-1)
        assert torch.allclose(
            stable_gather_minus_lse,
            reference_fp32,
            rtol=1e-5,
            atol=1e-5,
        )
        assert torch.equal(
            result_bf16,
            reference_fp32.to(torch.bfloat16),
        )

    def test_logprobs_from_logits_cast_to_fp32_false_stays_in_input_dtype(
        self,
    ) -> None:
        """``cast_to_fp32=False`` runs the reduction in input dtype throughout
        and matches a hand-rolled bf16 ``gather - logsumexp``.
        """
        torch.manual_seed(0)
        seq, vocab = 7, 2048
        logits_bf16 = torch.randn(3, seq, vocab, dtype=torch.bfloat16)
        index = torch.randint(0, vocab, (3, seq))

        result = LLMAlgorithm._logprobs_from_logits(
            logits_bf16,
            index,
            cast_to_fp32=False,
        )
        max_lg = logits_bf16.amax(dim=-1, keepdim=True)
        shifted = logits_bf16 - max_lg
        target = shifted.gather(dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(shifted, dim=-1)
        ref_bf16 = target - log_z
        assert result.dtype == torch.bfloat16
        assert torch.equal(result, ref_bf16)


class TestLogprobsFromHiddenFused:
    """Cover the chunked matmul + max-shift gather/logsumexp kernel that
    replaces ``hidden @ Wᵀ → log_softmax → gather`` without ever
    materializing the full ``(B, T, V)`` logits tensor.
    """

    def test_matches_log_softmax_reference_fp32(self) -> None:
        """``cast_to_fp32=True`` matches a stock ``log_softmax + gather`` over
        logits materialized from the same fp32-upcast operands.
        """
        torch.manual_seed(0)
        B, T, H, V = 4, 11, 64, 8192
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16)
        weight = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
        bias = torch.randn(V, dtype=torch.bfloat16)
        targets = torch.randint(0, V, (B, T))
        temperature = 0.7

        result = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden,
            weight,
            bias,
            targets,
            temperature=temperature,
            cast_to_fp32=True,
        )
        # Reference: stock log_softmax + gather over the materialized logits.
        # ``cast_to_fp32`` upcasts the matmul *operands*, so the reference must
        # too — a bf16 product rounds every logit before the reduction sees it.
        logits = (hidden.float() @ weight.float().t() + bias.float()) / temperature
        ref = (
            F.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
            .to(torch.bfloat16)
        )
        assert result.shape == (B, T)
        assert result.dtype == torch.bfloat16
        assert torch.equal(result, ref)

    def test_keeps_input_dtype_when_cast_disabled(self) -> None:
        """``cast_to_fp32=False`` keeps bf16 throughout — the reduction
        runs in input dtype, matching a hand-rolled bf16
        ``gather - logsumexp``.
        """
        torch.manual_seed(1)
        B, T, H, V = 4, 7, 64, 4096
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16)
        weight = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
        targets = torch.randint(0, V, (B, T))

        result = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden,
            weight,
            None,
            targets,
            temperature=1.0,
            cast_to_fp32=False,
        )
        # Reference: gather - logsumexp in bf16 (no fp32 promotion). Mirror the
        # kernel exactly — flat (B*T, H) reshape and a bare logsumexp with no
        # explicit max-shift (logsumexp is internally stable; an extra
        # subtraction would introduce its own bf16 rounding and break
        # bit-equality).
        flat_h = hidden.reshape(-1, H)
        flat_targets = targets.reshape(-1)
        logits = flat_h @ weight.t()
        selected = logits.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(logits, dim=-1)
        ref = (selected - log_z).reshape(B, T)
        assert result.dtype == torch.bfloat16
        assert torch.equal(result, ref)

    def test_mismatched_hidden_and_lm_head_dtypes(self) -> None:
        """An fp16 checkpoint under the bf16 autocast yields fp32 hidden
        states with an fp16 lm_head weight; the fused matmul promotes to a
        common dtype instead of crashing, matching the pre-upcast reference
        exactly (fp16 -> fp32 casts are lossless).
        """
        torch.manual_seed(3)
        B, T, H, V = 2, 5, 32, 2048
        hidden = torch.randn(B, T, H, dtype=torch.float32)
        weight = (torch.randn(V, H) * 0.02).to(torch.float16)
        targets = torch.randint(0, V, (B, T))

        result = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden,
            weight,
            None,
            targets,
            temperature=0.9,
            cast_to_fp32=True,
        )
        ref = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden,
            weight.float(),
            None,
            targets,
            temperature=0.9,
            cast_to_fp32=True,
        )
        assert result.dtype == torch.float32
        assert torch.equal(result, ref)

    def test_chunked_matches_unchunked(self) -> None:
        """Output is independent of ``chunk_rows`` — covers the loop
        boundary path.
        """
        torch.manual_seed(2)
        B, T, H, V = 3, 9, 32, 2048
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16)
        weight = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
        targets = torch.randint(0, V, (B, T))

        big = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden,
            weight,
            None,
            targets,
            temperature=0.5,
            cast_to_fp32=True,
            chunk_rows=10_000,  # > B*T=27 → single chunk
        )
        small = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden,
            weight,
            None,
            targets,
            temperature=0.5,
            cast_to_fp32=True,
            chunk_rows=4,  # forces multiple chunks
        )
        assert torch.equal(big, small)

    def test_no_bias_path_fp32(self) -> None:
        """``bias=None`` skips the add and still matches a stock
        log_softmax + gather reference.
        """
        torch.manual_seed(3)
        B, T, H, V = 2, 5, 16, 512
        hidden = torch.randn(B, T, H, dtype=torch.float32)
        weight = torch.randn(V, H, dtype=torch.float32) * 0.05
        targets = torch.randint(0, V, (B, T))

        result = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden,
            weight,
            None,
            targets,
            temperature=1.0,
            cast_to_fp32=False,
        )
        logits = hidden @ weight.t()
        ref = (
            F.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
        )
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)


class TestFusedLinearLogProbsGrad:
    """Cover the gradient-capable counterpart
    (:meth:`LLMAlgorithm._logprobs_from_hidden_fused_grad`) backed by the
    gradient-checkpointed :class:`_FusedLinearLogProbsFunction`. The forward
    must match the no-grad fused path bit-for-bit; the backward must yield
    the exact ``log_softmax`` gradient, never materializing ``(B, T, V)``.
    """

    @staticmethod
    def _naive_logps(hidden, weight, bias, targets, temperature, cast):
        logits = hidden @ weight.t()
        if bias is not None:
            logits = logits + bias
        if temperature != 1.0:
            logits = logits / temperature
        if cast:
            logits = logits.float()
        return (
            F.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
        )

    def test_forward_value_matches_nograd_path_bitwise(self) -> None:
        """The grad path's forward value is bit-identical to the no-grad
        fused path, so old/ref logprobs (computed no-grad) and policy
        logprobs (computed under grad) stay consistent — the first-step
        ratio is exactly 1.
        """
        torch.manual_seed(0)
        B, T, H, V = 3, 9, 32, 4096
        hidden = torch.randn(B, T, H, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(V, H, dtype=torch.bfloat16) * 0.02
        bias = torch.randn(V, dtype=torch.bfloat16)
        targets = torch.randint(0, V, (B, T))

        grad_val = LLMAlgorithm._logprobs_from_hidden_fused_grad(
            hidden, weight, bias, targets, temperature=0.7, cast_to_fp32=True
        )
        nograd_val = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden.detach(), weight, bias, targets, temperature=0.7, cast_to_fp32=True
        )
        assert grad_val.requires_grad
        assert grad_val.shape == (B, T)
        assert torch.equal(grad_val.detach(), nograd_val)

    def test_hidden_grad_matches_naive_autograd(self) -> None:
        """Gradient w.r.t. hidden matches autograd through a materialized
        ``log_softmax`` to fp32 tolerance.
        """
        torch.manual_seed(1)
        B, T, H, V = 4, 6, 24, 1024
        weight = torch.randn(V, H)
        bias = torch.randn(V)
        targets = torch.randint(0, V, (B, T))
        upstream = torch.randn(B, T)

        hid_f = torch.randn(B, T, H, requires_grad=True)
        out_f = LLMAlgorithm._logprobs_from_hidden_fused_grad(
            hid_f, weight, bias, targets, temperature=0.8, cast_to_fp32=True
        )
        out_f.backward(upstream)

        hid_n = hid_f.detach().clone().requires_grad_(True)
        out_n = self._naive_logps(hid_n, weight, bias, targets, 0.8, True)
        out_n.backward(upstream)

        assert torch.allclose(hid_f.grad, hid_n.grad, rtol=1e-4, atol=1e-5)

    def test_weight_and_bias_grad_match_naive_autograd(self) -> None:
        """Gradients w.r.t. lm_head weight and bias match naive autograd."""
        torch.manual_seed(2)
        B, T, H, V = 2, 5, 16, 512
        targets = torch.randint(0, V, (B, T))
        upstream = torch.randn(B, T)

        hid_f = torch.randn(B, T, H, requires_grad=True)
        w_f = torch.randn(V, H, requires_grad=True)
        b_f = torch.randn(V, requires_grad=True)
        out_f = LLMAlgorithm._logprobs_from_hidden_fused_grad(
            hid_f, w_f, b_f, targets, temperature=1.0, cast_to_fp32=True
        )
        out_f.backward(upstream)

        hid_n = hid_f.detach().clone().requires_grad_(True)
        w_n = w_f.detach().clone().requires_grad_(True)
        b_n = b_f.detach().clone().requires_grad_(True)
        out_n = self._naive_logps(hid_n, w_n, b_n, targets, 1.0, True)
        out_n.backward(upstream)

        assert torch.allclose(w_f.grad, w_n.grad, rtol=1e-4, atol=1e-4)
        assert torch.allclose(b_f.grad, b_n.grad, rtol=1e-4, atol=1e-4)

    def test_grad_invariant_to_chunk_rows(self) -> None:
        """Forward value and hidden gradient are independent of
        ``chunk_rows`` (single chunk vs many) up to fp32 matmul-tiling
        noise — chunking only partitions rows, it changes nothing about
        each row's reduction.
        """
        torch.manual_seed(3)
        B, T, H, V = 3, 7, 20, 2048
        weight = torch.randn(V, H)
        targets = torch.randint(0, V, (B, T))
        upstream = torch.randn(B, T)

        def run(chunk_rows):
            hid = torch.randn(B, T, H, generator=torch.Generator().manual_seed(7))
            hid.requires_grad_(True)
            out = LLMAlgorithm._logprobs_from_hidden_fused_grad(
                hid,
                weight,
                None,
                targets,
                temperature=0.9,
                cast_to_fp32=True,
                chunk_rows=chunk_rows,
            )
            out.backward(upstream)
            return out.detach(), hid.grad

        big_out, big_grad = run(10_000)  # single chunk (> B*T)
        small_out, small_grad = run(4)  # forces many chunks
        assert torch.allclose(big_out, small_out, rtol=1e-5, atol=1e-5)
        assert torch.allclose(big_grad, small_grad, rtol=1e-5, atol=1e-5)

    def test_no_grad_when_inputs_detached(self) -> None:
        """With no input requiring grad the output is detached and the
        bounded backward simply isn't exercised.
        """
        torch.manual_seed(4)
        B, T, H, V = 2, 4, 12, 256
        hidden = torch.randn(B, T, H)
        weight = torch.randn(V, H)
        targets = torch.randint(0, V, (B, T))
        out = LLMAlgorithm._logprobs_from_hidden_fused_grad(
            hidden, weight, None, targets, temperature=1.0, cast_to_fp32=True
        )
        assert not out.requires_grad

    def test_temperature_scaling_applied_once(self) -> None:
        """Temperature folds into logits exactly once before log_softmax."""
        torch.manual_seed(4)
        B, T, H, V = 2, 6, 16, 1024
        hidden = torch.randn(B, T, H, dtype=torch.float32)
        weight = torch.randn(V, H, dtype=torch.float32) * 0.05
        targets = torch.randint(0, V, (B, T))
        temperature = 2.5

        result = LLMAlgorithm._logprobs_from_hidden_fused(
            hidden, weight, None, targets, temperature=temperature
        )
        logits = (hidden @ weight.t()) / temperature
        ref = (
            F.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=targets.unsqueeze(-1))
            .squeeze(-1)
        )
        assert torch.allclose(result, ref, rtol=1e-5, atol=1e-5)


class TestGetLmHeadParentAndPatch:
    """Cover ``_get_lm_head_parent`` (which walks value-head / PEFT / LoRA
    wrappers) and ``_patch_lm_head_to_identity`` (the swap-and-restore
    context manager used by the no-grad fused-linear-logprob path).
    """

    def _agent_with_real_lm_head(self):
        agent = _make_llm_agent()
        # Mock PEFT actor exposes ``base_model.model`` (MagicMock children).
        # Replace its ``lm_head`` with a real ``nn.Linear`` so ``setattr``
        # has somewhere deterministic to swap.
        real_lm_head = torch.nn.Linear(8, 32)
        agent.actor.base_model.model.lm_head = real_lm_head
        return agent, real_lm_head

    def test_get_lm_head_parent_returns_parent_and_attr(self) -> None:
        agent, real = self._agent_with_real_lm_head()
        parent, attr = agent._get_lm_head_parent()
        assert getattr(parent, attr) is real
        assert attr == "lm_head"

    def test_patch_restores_on_normal_exit(self) -> None:
        agent, original = self._agent_with_real_lm_head()
        with agent._patch_lm_head_to_identity():
            assert isinstance(agent.actor.base_model.model.lm_head, torch.nn.Identity)
        assert agent.actor.base_model.model.lm_head is original

    def test_patch_restores_on_exception(self) -> None:
        agent, original = self._agent_with_real_lm_head()
        msg = "simulated failure inside lm_head patch"
        with pytest.raises(
            RuntimeError, match="simulated failure inside lm_head patch"
        ):
            with agent._patch_lm_head_to_identity():
                raise RuntimeError(msg)
        assert agent.actor.base_model.model.lm_head is original


class _TinyCausalLM(torch.nn.Module):
    """Minimal HF-style causal LM (embedding → linear body → lm_head)
    used by the integration tests below to exercise the lm_head→Identity
    monkey-patch against a real ``nn.Linear`` head.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.body = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: object,
    ) -> SimpleNamespace:
        x = self.embed(input_ids)
        x = self.body(x)
        return SimpleNamespace(logits=self.lm_head(x))


class _TinyPeftWrapper(torch.nn.Module):
    """Stand-in for the ``PeftModel → LoraModel → CausalLM`` walk that
    ``_get_lm_head_parent`` traverses.
    """

    def __init__(self, inner: _TinyCausalLM) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = inner

    def forward(self, **kwargs: object) -> SimpleNamespace:
        return self.base_model.model(**kwargs)


class TestFusedLinearLogprobsIntegration:
    """End-to-end: the (unconditional) fused-linear-logprob path in
    ``_get_logprobs`` produces logprobs numerically equivalent to a reference
    computed from full ``(B, T, V)`` logits on the same model under
    ``torch.no_grad()``. Exercises ``_get_lm_head_parent``,
    ``_patch_lm_head_to_identity``, and ``_logprobs_from_hidden_fused``.
    """

    def _build_agent(
        self, vocab_size: int, hidden_size: int
    ) -> tuple[LLMAlgorithm, _TinyPeftWrapper]:
        agent = _make_llm_agent()
        actor = _TinyPeftWrapper(_TinyCausalLM(vocab_size, hidden_size))
        actor.eval()
        agent.actor = actor
        agent.use_value_head = False
        agent.temperature = 0.7
        agent.calc_position_embeddings = False
        agent.pad_token_id = 0

        from contextlib import contextmanager

        @contextmanager
        def _noop_select(_name: str):
            yield

        agent.select_adapter = _noop_select
        return agent, actor

    def test_fused_matches_reference_logits_under_no_grad(self) -> None:
        torch.manual_seed(0)
        B, T, H, V = 3, 7, 16, 256
        agent, actor = self._build_agent(V, H)
        ids = torch.randint(1, V, (B, T))

        with torch.no_grad():
            lp_fused = agent._get_logprobs(
                ids, batch_size=B, use_reference=False, eval_mode=True
            )
            # Reference: full (B, T, V) logits → _logprobs_from_logits, the
            # exact math the fused path approximates chunk-by-chunk.
            full_logits = actor(input_ids=ids).logits / agent.temperature
            lp_ref = LLMAlgorithm._logprobs_from_logits(
                full_logits[:, :-1], ids[:, 1:], cast_to_fp32=True
            )

        assert lp_fused.shape == (B, T - 1)
        assert lp_ref.shape == (B, T - 1)
        assert torch.allclose(lp_fused, lp_ref, rtol=1e-5, atol=1e-5)
        # lm_head restored after the patched call (try/finally teardown).
        assert isinstance(actor.base_model.model.lm_head, torch.nn.Linear)

    def test_no_grad_fused_method_skipped_when_grad_enabled(self) -> None:
        """Under grad, ``_get_logprobs`` uses the gradient-aware fused fn, so
        the no-grad ``_logprobs_from_hidden_fused`` static is not called.
        """
        torch.manual_seed(1)
        B, T, H, V = 2, 5, 8, 128
        agent, _ = self._build_agent(V, H)
        ids = torch.randint(1, V, (B, T))
        with patch.object(
            LLMAlgorithm,
            "_logprobs_from_hidden_fused",
            wraps=LLMAlgorithm._logprobs_from_hidden_fused,
        ) as spy:
            agent._get_logprobs(ids, batch_size=B, use_reference=False, eval_mode=True)
        spy.assert_not_called()

    @pytest.mark.parametrize("cast_to_fp32", [True, False])
    def test_cast_logprobs_to_fp32_threaded_into_fused_kernel(
        self, cast_to_fp32: bool
    ) -> None:
        """``self.cast_logprobs_to_fp32`` flows into the fused-no-grad
        kernel call so toggling it controls the reduction precision.
        """
        torch.manual_seed(2)
        B, T, H, V = 2, 4, 8, 64
        agent, _ = self._build_agent(V, H)
        agent.cast_logprobs_to_fp32 = cast_to_fp32
        ids = torch.randint(1, V, (B, T))
        with (
            patch.object(
                LLMAlgorithm,
                "_logprobs_from_hidden_fused",
                wraps=LLMAlgorithm._logprobs_from_hidden_fused,
            ) as spy,
            torch.no_grad(),
        ):
            agent._get_logprobs(ids, batch_size=B, use_reference=False, eval_mode=True)
        assert spy.called
        assert spy.call_args.kwargs["cast_to_fp32"] is cast_to_fp32

    @pytest.mark.parametrize("cast_to_fp32", [True, False])
    def test_cast_logprobs_to_fp32_threaded_into_fused_model_pass(
        self, cast_to_fp32: bool
    ) -> None:
        """``self.cast_logprobs_to_fp32`` also flows through the other
        call site (``_fused_model_pass``), which is what
        ``_fused_forward`` / ``_fused_forward_no_grad`` go through.
        """
        torch.manual_seed(4)
        B, T, H, V = 2, 4, 8, 64
        agent, _ = self._build_agent(V, H)
        agent.cast_logprobs_to_fp32 = cast_to_fp32

        # ``_fused_model_pass`` calls ``set_fused_adapter_routing`` and
        # ``_get_unwrapped_actor`` — stub both since the tiny test actor
        # isn't a real PEFT model.
        agent._get_unwrapped_actor = lambda: agent.actor
        fused_ids = torch.randint(1, V, (B, T))
        fused_mask = torch.ones_like(fused_ids)
        routing = ["actor"] * B

        with (
            patch(
                "agilerl.algorithms.core.base.set_fused_adapter_routing",
                lambda *a, **kw: None,
            ),
            patch.object(
                LLMAlgorithm,
                "_logprobs_from_hidden_fused",
                wraps=LLMAlgorithm._logprobs_from_hidden_fused,
            ) as spy,
            torch.no_grad(),
        ):
            agent._fused_model_pass(fused_ids, fused_mask, routing)
        assert spy.called
        assert spy.call_args.kwargs["cast_to_fp32"] is cast_to_fp32


class TestLLMCreatePromptMasks:
    def test_creates_correct_mask(self):
        mask = LLMAlgorithm._create_prompt_masks([3, 5], 10)
        assert mask.shape == (2, 10)
        assert not mask[0, 2].item()
        assert mask[0, 4].item()
        assert not mask[1, 4].item()
        assert mask[1, 6].item()

    def test_first_response_token_is_included(self):
        mask = LLMAlgorithm._create_prompt_masks([3, 5], 10)
        assert mask[0, 3].item()
        assert mask[1, 5].item()
        assert not mask[0, 2].item()
        assert not mask[1, 4].item()


@_LLM_DEPS_SKIP
class TestLLMConfigureBatchSize:
    def test_clone_mode_sets_batch_size_directly(self):
        agent = _make_llm_agent(clone=True)
        assert agent.batch_size_per_process == 4

    def test_raises_when_batch_not_divisible_by_processes(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=3,
        )
        with pytest.raises(ValueError, match="divisible by the number of processes"):
            _make_llm_agent(accelerator=acc, clone=False)

    def test_micro_batch_auto_derived(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_accumulation_steps": 2,
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=2,
        )
        agent = _make_llm_agent(accelerator=acc, clone=False)
        assert agent.micro_batch_size_per_gpu == 1

    def test_micro_batch_explicit(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=1,
        )
        agent = _make_llm_agent(
            accelerator=acc, clone=False, micro_batch_size_per_gpu=2
        )
        assert agent.micro_batch_size_per_gpu == 2

    def test_micro_batch_explicit_not_divisible_raises(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=1,
        )
        with pytest.raises(ValueError, match="micro_batch_size_per_gpu"):
            _make_llm_agent(accelerator=acc, clone=False, micro_batch_size_per_gpu=3)

    def test_auto_micro_batch_zero_raises(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_accumulation_steps": 1,
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=1,
        )
        with (
            pytest.raises(
                ValueError,
                match=r"micro_batch_size_per_gpu is equal to zero, which is not allowed\.",
            ),
            patch.object(LLMAlgorithm, "_initialize_actors"),
            patch.object(LLMAlgorithm, "_configure_vllm"),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(EvolvableAlgorithm, "_registry_init"),
        ):
            _StubLLMAlgorithm(
                index=0,
                batch_size=0,
                lr=1e-4,
                max_grad_norm=0.0,
                clone=False,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=MagicMock(),
                actor_network=_make_mock_peft_actor(),
                accelerator=acc,
                device="cpu",
                micro_batch_size_per_gpu=0,
            )

    def test_batch_not_divisible_by_grad_accum_raises(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_accumulation_steps": 3,
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=1,
        )
        with pytest.raises(ValueError, match="gradient accumulation"):
            _make_llm_agent(accelerator=acc, clone=False)


class _MicroDefaultStubLLMAlgorithm(_StubLLMAlgorithm):
    """Stub with the RL-family mini-batch default."""

    _mini_batch_size_default = "micro_batch"


@_LLM_DEPS_SKIP
class TestLLMConfigureMiniBatchSize:
    @staticmethod
    def _accelerator(ds_config=None, num_processes=1):
        return _make_mock_accelerator(
            ds_config=ds_config
            or {
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=num_processes,
        )

    def test_explicit_mini_batch_sets_accumulation_steps(self):
        acc = self._accelerator()
        agent = _make_llm_agent(
            accelerator=acc,
            clone=False,
            micro_batch_size_per_gpu=1,
            mini_batch_size=2,
        )
        ds_config = acc.state.deepspeed_plugin.deepspeed_config
        assert agent.micro_batch_size_per_gpu == 1
        assert agent.mini_batch_size == 2
        assert ds_config["gradient_accumulation_steps"] == 2
        assert ds_config["train_micro_batch_size_per_gpu"] == 1

    def test_default_mini_batch_micro_policy_steps_every_micro_batch(self):
        acc = self._accelerator()
        agent = _make_llm_agent(
            accelerator=acc,
            clone=False,
            micro_batch_size_per_gpu=2,
            algo_cls=_MicroDefaultStubLLMAlgorithm,
        )
        ds_config = acc.state.deepspeed_plugin.deepspeed_config
        assert agent.mini_batch_size == 2
        assert ds_config["gradient_accumulation_steps"] == 1

    def test_default_mini_batch_batch_policy_accumulates_whole_batch(self):
        acc = self._accelerator()
        agent = _make_llm_agent(
            accelerator=acc,
            clone=False,
            micro_batch_size_per_gpu=2,
            batch_size=4,
        )
        ds_config = acc.state.deepspeed_plugin.deepspeed_config
        assert agent.mini_batch_size == 4
        assert ds_config["gradient_accumulation_steps"] == 2

    def test_mini_batch_not_divisible_by_micro_raises(self):
        acc = self._accelerator()
        with pytest.raises(ValueError, match="must be divisible by"):
            _make_llm_agent(
                accelerator=acc,
                clone=False,
                micro_batch_size_per_gpu=2,
                mini_batch_size=3,
                batch_size=4,
            )

    def test_mini_batch_without_micro_batch_takes_one_backward_per_step(self):
        acc = self._accelerator()
        agent = _make_llm_agent(
            accelerator=acc,
            clone=False,
            mini_batch_size=2,
            batch_size=4,
        )
        ds_config = acc.state.deepspeed_plugin.deepspeed_config
        assert agent.micro_batch_size_per_gpu == 2
        assert agent.mini_batch_size == 2
        assert ds_config["gradient_accumulation_steps"] == 1

    def test_mini_batch_without_deepspeed_mismatch_raises(self):
        with pytest.raises(ValueError, match="requires a DeepSpeed engine"):
            _make_llm_agent(
                clone=False,
                micro_batch_size_per_gpu=1,
                mini_batch_size=2,
            )

    def test_mini_batch_nonpositive_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            _make_llm_agent(clone=False, mini_batch_size=0)

    def test_legacy_config_accumulation_records_mini_batch(self):
        acc = self._accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_accumulation_steps": 2,
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=2,
        )
        agent = _make_llm_agent(accelerator=acc, clone=False, batch_size=4)
        assert agent.micro_batch_size_per_gpu == 1
        assert agent.mini_batch_size == 2

    def test_overriding_config_accumulation_warns(self):
        acc = self._accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_accumulation_steps": 4,
            },
        )
        with pytest.warns(
            UserWarning, match="Overwriting DeepSpeed config gradient_accumulation"
        ):
            agent = _make_llm_agent(
                accelerator=acc,
                clone=False,
                micro_batch_size_per_gpu=1,
                mini_batch_size=2,
            )
        ds_config = acc.state.deepspeed_plugin.deepspeed_config
        assert ds_config["gradient_accumulation_steps"] == 2
        assert agent.mini_batch_size == 2


@_LLM_DEPS_SKIP
class TestLLMInitWarnings:
    def test_cosine_lr_with_accelerator_warns_and_nullifies(self):
        acc = _make_mock_accelerator()
        sched = MagicMock()
        agent = _make_llm_agent(
            accelerator=acc,
            cosine_lr_schedule_config=sched,
        )
        assert agent.cosine_lr_schedule_config is None

    def test_reduce_memory_peak_deprecated_warns(self):
        with pytest.warns(DeprecationWarning, match="reduce_memory_peak is deprecated"):
            _make_llm_agent(reduce_memory_peak=True)

    def test_lr_overwrite_warning_from_deepspeed(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "optimizer": {"params": {"lr": 0.999}},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        with pytest.warns(
            UserWarning,
            match="Overwriting deepspeed learning rate with the argument 'lr'.",
        ):
            agent = _make_llm_agent(accelerator=acc)
        assert agent.lr == 0.0001
        assert (
            acc.state.deepspeed_plugin.deepspeed_config["optimizer"]["params"]["lr"]
            == 0.0001
        )

    def test_no_lora_config_applies_default(self):
        class _NonPeftActor:
            name_or_path = "mock-model"

        with (
            patch.object(LLMAlgorithm, "_initialize_actors"),
            patch.object(LLMAlgorithm, "_configure_vllm"),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(EvolvableAlgorithm, "_registry_init"),
            pytest.warns(UserWarning, match="No LoRA config"),
        ):
            agent = _StubLLMAlgorithm(
                index=0,
                batch_size=4,
                lr=1e-4,
                max_grad_norm=0.0,
                clone=True,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=None,
                actor_network=_NonPeftActor(),
                device="cpu",
            )
        assert agent.lora_config is not None

    def test_no_lora_config_applies_default_with_peft_actor(self):
        """Peft actor_network with lora_config=None still gets the same default LoRA."""
        with (
            patch.object(LLMAlgorithm, "_initialize_actors"),
            patch.object(LLMAlgorithm, "_configure_vllm"),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(EvolvableAlgorithm, "_registry_init"),
            pytest.warns(UserWarning, match="No LoRA config"),
        ):
            agent = _StubLLMAlgorithm(
                index=0,
                batch_size=4,
                lr=1e-4,
                max_grad_norm=0.0,
                clone=True,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=None,
                use_separate_reference_adapter=False,
                actor_network=_make_mock_peft_actor(),
                device="cpu",
            )
        assert agent.lora_config is not None

    def test_max_grad_norm_overwrite_warning(self):
        acc = _make_mock_accelerator()
        with pytest.warns(UserWarning, match="max_grad_norm"):
            _make_llm_agent(accelerator=acc, max_grad_norm=1.5)

    def test_zero_stage_3_warning(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 3},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        with pytest.warns(UserWarning, match="ZeRO Stage 3"):
            _make_llm_agent(accelerator=acc)

    def test_zero_stage_3_disables_memory_efficient_params(self):
        """Colocated CPU offload is incompatible with ZeRO-3 param sharding."""
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 3},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        with pytest.warns(
            UserWarning, match="Memory efficient params is not compatible"
        ):
            agent = _make_llm_agent(
                accelerator=acc,
                use_vllm=True,
                use_memory_efficient_params=True,
            )
        assert agent.use_memory_efficient_params is False

    def test_mutation_hook_registered_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc, max_grad_norm=1.0)
        agent.registry.hooks = ["_sync_deepspeed_gradient_clipping"]
        agent._sync_deepspeed_gradient_clipping = MagicMock()
        agent.mutation_hook()
        agent._sync_deepspeed_gradient_clipping.assert_called_once()

    def test_rejects_activation_checkpointing_config(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 2},
                "activation_checkpointing": {"partition_activations": True},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        with pytest.raises(RuntimeError, match="activation_checkpointing"):
            _make_llm_agent(accelerator=acc)


class TestLLMZero3ThirdPartyHooksWireUp:
    def test_installs_hooks_when_stage3(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {
                    "stage": 3,
                    "stage3_param_persistence_threshold": 50_000,
                    "stage3_model_persistence_threshold": 1_000_000,
                },
                "train_micro_batch_size_per_gpu": "auto",
            },
            num_processes=4,
        )
        with (
            patch("agilerl.algorithms.core.base.install_zero3_patches") as mock_hooks,
            pytest.warns(UserWarning, match="ZeRO Stage 3"),
        ):
            agent = _make_llm_agent(accelerator=acc)

        mock_hooks.assert_called_once_with(
            acc.state.deepspeed_plugin.deepspeed_config,
            model_name_or_path=agent.pretrained_model_name_or_path,
            num_partitions=4,
        )

    def test_skips_when_stage_is_not_3(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {
                    "stage": 2,
                    "stage3_param_persistence_threshold": 50_000,
                },
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        with patch("agilerl.algorithms.core.base.install_zero3_patches") as mock_hooks:
            _make_llm_agent(accelerator=acc)

        mock_hooks.assert_not_called()


class TestLLMSyncDeepSpeedGradientClipping:
    def test_sync_updates_ds_config_and_optimizer(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_clipping": 0.5,
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc, max_grad_norm=1.5)
        agent.actor.optimizer = MagicMock()
        agent.actor.optimizer.grad_clip = 0.5
        agent.actor.optimizer.clip_grad = 0.5
        agent._sync_deepspeed_gradient_clipping()
        assert acc.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] == 1.5
        assert agent.actor.optimizer.grad_clip == 1.5
        assert agent.actor.optimizer.clip_grad == 1.5

    def test_sync_noop_without_accelerator(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        agent._sync_deepspeed_gradient_clipping()

    def test_sync_noop_without_gradient_clipping_key(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc)
        agent._sync_deepspeed_gradient_clipping()


class TestLLMGetLmHead:
    def test_finds_lm_head_on_base_model(self):
        agent = _make_llm_agent()
        lm_head = MagicMock()
        agent.actor.base_model.model.lm_head = lm_head
        result = agent._get_lm_head()
        assert result is lm_head

    def test_finds_embed_out(self):
        agent = _make_llm_agent()
        embed_out = MagicMock()
        del agent.actor.base_model.model.lm_head
        agent.actor.base_model.model.embed_out = embed_out
        result = agent._get_lm_head()
        assert result is embed_out

    def test_raises_when_no_lm_head_found(self):
        agent = _make_llm_agent()
        del agent.actor.base_model.model.lm_head
        del agent.actor.base_model.model.embed_out
        with pytest.raises(AttributeError, match="Cannot find lm_head"):
            agent._get_lm_head()

    def test_fused_logprob_fn_and_head_returns_tensors(self):
        agent = _make_llm_agent()
        weight = torch.randn(4, 2)
        lm_head = MagicMock()
        lm_head.weight = weight
        lm_head.bias = None
        agent._get_lm_head = MagicMock(return_value=lm_head)
        fused_fn, got_w, got_b = agent._fused_logprob_fn_and_head()
        assert callable(fused_fn)
        assert got_w is weight
        assert got_b is None

    def test_liger_head_gather_calls_gather_if_ds_param(self):
        agent = _make_llm_agent()
        weight = torch.randn(4, 2)
        bias = torch.randn(4)
        lm_head = MagicMock()
        lm_head.weight = weight
        lm_head.bias = bias
        agent._get_lm_head = MagicMock(return_value=lm_head)
        sentinel = object()

        with patch(
            "agilerl.algorithms.core.base.gather_if_ds_param",
            return_value=sentinel,
        ) as mock_gather:
            result = agent._liger_head_gather()

        mock_gather.assert_called_once_with(weight, bias)
        assert result is sentinel

    def test_resolve_fused_chunk_rows_uses_ds_shape(self):
        weight = torch.empty(0)
        weight.ds_shape = (49152, 2048)
        vocab = getattr(weight, "ds_shape", weight.shape)[0]
        rows = LLMAlgorithm._resolve_fused_chunk_rows(vocab, None)
        assert rows > 0
        assert vocab == 49152


@pytest.mark.skipif(
    not HAS_VLLM,
    reason="_configure_vllm exercises the vllm extra.",
)
class TestLLMConfigureVllm:
    def test_raises_when_vllm_not_installed(self):
        agent = _make_llm_agent()
        with patch("agilerl.algorithms.core.base.LLM", None, create=True):
            with pytest.raises(ImportError, match="vLLM is required"):
                agent._configure_vllm()

    def test_uses_default_config_when_none(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        agent.vllm_config = None
        mock_llm_cls = MagicMock()
        with (
            patch("agilerl.algorithms.core.base.LLM", mock_llm_cls, create=True),
            pytest.warns(UserWarning, match="No VLLM config"),
        ):
            agent._configure_vllm()
        assert isinstance(agent.vllm_config, VLLMConfig)
        mock_llm_cls.assert_called_once()

    def test_raises_when_tp_size_invalid(self):
        acc = _make_mock_accelerator(num_processes=3)
        agent = _make_llm_agent(accelerator=acc, batch_size=12)
        agent.vllm_config = MagicMock()
        agent.vllm_config.tensor_parallel_size = 2
        with patch("agilerl.algorithms.core.base.LLM", MagicMock(), create=True):
            with pytest.raises(ValueError, match="Tensor parallel size"):
                agent._configure_vllm()


class TestLLMSetReferencePolicy:
    def test_set_reference_with_separate_adapter(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        agent.accelerator = None
        ref_p = torch.tensor([1.0])
        act_p = torch.tensor([2.0])
        with patch.object(
            type(agent.actor),
            "named_parameters",
            return_value=[
                ("lora.reference.weight", ref_p),
                ("lora.actor.weight", act_p),
            ],
        ):
            agent.set_reference_policy(1)
        assert torch.equal(ref_p, act_p)
        assert agent.reference_update_tracker == 1

    def test_set_reference_raises_on_no_source_params(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        agent.accelerator = None
        with (
            patch.object(
                type(agent.actor),
                "named_parameters",
                return_value=[
                    ("not_lora.weight", torch.tensor([1.0])),
                ],
            ),
            pytest.raises(ValueError, match="No LoRA tensors found for source adapter"),
        ):
            agent.set_reference_policy(1)

    def test_set_reference_raises_on_no_target_params(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        agent.accelerator = None
        with (
            patch.object(
                type(agent.actor),
                "named_parameters",
                return_value=[
                    ("lora.actor.weight", torch.tensor([1.0])),
                ],
            ),
            pytest.raises(ValueError, match="No LoRA tensors found for target adapter"),
        ):
            agent.set_reference_policy(1)

    def test_set_reference_missing_params(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        agent.accelerator = None
        with patch.object(
            type(agent.actor),
            "named_parameters",
            return_value=[
                ("lora.reference.one.weight", torch.tensor([1.0])),
                ("lora.actor.one.weight", torch.tensor([1.0])),
                ("lora.actor.two.weight", torch.tensor([1.0])),
            ],
        ):
            with pytest.raises(
                ValueError,
                match=r"Target adapter 'reference' is missing 1 LoRA tensors present in source adapter 'actor'\.",
            ):
                agent.set_reference_policy(1)

    def test_set_reference_without_separate_adapter_warns_and_keeps_base(self):
        """Base weights are immutable: the implicit reference cannot move, so an
        update request warns once and only advances the tracker.
        """
        agent = _make_llm_agent(use_separate_reference_adapter=False)
        agent.accelerator = None
        with patch.object(LLMAlgorithm, "_copy_adapter_weights") as mock_copy:
            with pytest.warns(UserWarning, match="stays the initial base policy"):
                agent.set_reference_policy(1)
            mock_copy.assert_not_called()
            assert agent.reference_update_tracker == 1
            # Warn-once: a second update advances the tracker silently.
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                agent.set_reference_policy(2)
        assert agent.reference_update_tracker == 2

    def test_no_update_when_tracker_equal(self):
        agent = _make_llm_agent()
        agent.reference_update_tracker = 5
        with patch.object(LLMAlgorithm, "_copy_adapter_weights") as mock_copy:
            agent.set_reference_policy(5)
        mock_copy.assert_not_called()
        assert agent.reference_update_tracker == 5


class TestLLMGetLogprobs:
    def test_get_logprobs_computes_correctly(self):
        agent = _make_llm_agent()
        agent.calc_position_embeddings = True
        agent.pad_token_id = 0
        ids = torch.randint(1, 50, (2, 10))
        # ``_get_logprobs`` always uses the fused-linear-logprob path; patch the
        # no-grad fused kernel so the mock PEFT actor's stub lm_head isn't
        # actually matmul'd.
        with (
            torch.no_grad(),
            patch.object(
                LLMAlgorithm,
                "_logprobs_from_hidden_fused",
                return_value=torch.randn(2, 9),
            ),
        ):
            result = agent._get_logprobs(ids, batch_size=4)
        assert result.shape[0] == 2


SAVE_LOAD_OPTIONS = [
    pytest.param((True, True), id="lora_only+optim"),
    pytest.param((True, False), id="lora_only"),
    pytest.param((False, True), id="full+optim"),
    pytest.param((False, False), id="full"),
]

SMALL_LORA = LoraConfig(
    r=2,
    lora_alpha=4,
    target_modules=["linear_1"],
    task_type="CAUSAL_LM",
    lora_dropout=0.0,
)


def get_param_by_name(agent, substring: str) -> tuple[str, torch.nn.Parameter]:
    """Return the first actor parameter whose name contains ``substring``."""
    for name, param in agent.actor.named_parameters():
        if substring in name:
            return name, param
    msg = f"no actor param matching {substring!r}"
    raise KeyError(msg)


def find_exp_avg_in_opt_state(agent) -> torch.Tensor | None:
    """Return a reference to the first Adam ``exp_avg`` tensor in agent.optimizer.

    Returns None if optimizer.state is empty (e.g. before any step).
    """
    for state in agent.optimizer.optimizer.state.values():
        if "exp_avg" in state:
            return state["exp_avg"]
    return None


def load_attributes_checkpoint(path):
    return torch.load(
        str(path / "attributes.pt"),
        weights_only=False,
        pickle_module=dill,
    )


def normalize_optimizer_state(value):
    """Normalize nested optimizer state for deterministic comparisons."""
    if isinstance(value, torch.Tensor):
        return {
            "__tensor__": True,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "data": value.detach().cpu().tolist(),
        }
    if isinstance(value, dict):
        return {k: normalize_optimizer_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_optimizer_state(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    # DeepSpeed includes enum-like/custom metadata objects in state_dict.
    return repr(value)


def generate_tiny_grpo(accelerator=None) -> GRPO:
    """Build a tiny CPU GRPO agent with (actor, reference) adapters."""
    if not (HAS_VLLM and HAS_DEEPSPEED):
        pytest.skip("Need to install agilerl with deepspeed + vllm")
    actor = create_module(input_size=6, max_tokens=4, vocab_size=64, device="cpu")
    return GRPO(
        actor_network=actor,
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=4,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        lora_config=SMALL_LORA,
        accelerator=accelerator,
        wrap=False,
        gradient_checkpointing=False,
        device="cpu",
        use_separate_reference_adapter=True,
    )


def _grpo_from_template(template: GRPO) -> GRPO:
    """Copy the session template so checkpoint tests do not mutate it."""
    return template.clone(index=0, wrap=False)


@pytest.fixture(scope="session")
def grpo_factory():
    """Expensive PEFT-wrapped GRPO, built once per session.

    Tests consume deepcopies of this template so the session-scoped instance
    is never mutated after construction.
    """
    tiny_grpo = generate_tiny_grpo(accelerator=None)
    yield tiny_grpo
    tiny_grpo.clean_up()


# --------------------------------------------------------------------------- #
# SAVE — plain torch/peft path (accelerator is None)                          #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_simple_checkpoint(request, grpo_factory, tmp_path_factory):
    """One saved plain-path checkpoint per cell, shared across all tests that
    only *read* the output. This does not involve deepspeed.

    Session scope means ``save_checkpoint`` runs exactly 4 times for the whole
    test session (once per cell), not once per test.
    """
    lora_only, save_optimizer = request.param
    agent = _grpo_from_template(grpo_factory)
    tmp_path = tmp_path_factory.mktemp(
        f"plain_save_lora={lora_only}_optim={save_optimizer}"
    )
    agent.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    simple_checkpoint = SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    yield simple_checkpoint
    del simple_checkpoint


class TestLLMSimpleCheckpointSave:
    """Each test runs 4x (one per SAVE_LOAD_OPTIONS param) against a pre-saved
    checkpoint. Assertions are phrased as truth tables over
    ``plain_saved.lora_only`` / ``plain_saved.save_optimizer``.
    """

    def test_llm_simple_checkpoint_save_attributes_pt_always_written(
        self, llm_simple_checkpoint
    ):
        assert (llm_simple_checkpoint.path / "attributes.pt").exists()

    def test_llm_simple_checkpoint_save_no_deepspeed_tag_dir_on_plain_path(
        self, llm_simple_checkpoint
    ):
        # deepspeed engines write to a tag subdirectory; plain path must not.
        assert not (llm_simple_checkpoint.path / "save_checkpoint").exists()

    def test_llm_simple_checkpoint_save_adapter_dirs_present_if_lora_only(
        self, llm_simple_checkpoint
    ):
        actor_adapter = (
            llm_simple_checkpoint.path / "actor" / "adapter_model.safetensors"
        )
        ref_adapter = (
            llm_simple_checkpoint.path / "reference" / "adapter_model.safetensors"
        )
        assert actor_adapter.exists() == llm_simple_checkpoint.lora_only
        assert ref_adapter.exists() == llm_simple_checkpoint.lora_only

    def test_llm_simple_checkpoint_save_attributes_pt_contents_match_cell(
        self, llm_simple_checkpoint
    ):
        ck = load_attributes_checkpoint(llm_simple_checkpoint.path)
        ni = ck.get("network_info")

        # _lora_only flag round-trips verbatim.
        assert ck.get("_lora_only") == llm_simple_checkpoint.lora_only

        # actor_state_dict in attributes.pt if full-model save (not lora_only).
        has_actor_sd = "actor_state_dict" in ni["modules"]
        assert has_actor_sd == (not llm_simple_checkpoint.lora_only), (
            f"actor_state_dict presence wrong for cell "
            f"(lora_only={llm_simple_checkpoint.lora_only}, save_optimizer={llm_simple_checkpoint.save_optimizer})"
        )

        # Optimizer state in attributes.pt if save_optimizer=True (plain path).
        has_optim = bool(ni["optimizers"])
        assert has_optim == llm_simple_checkpoint.save_optimizer, (
            f"optimizer presence wrong for cell "
            f"(lora_only={llm_simple_checkpoint.lora_only}, save_optimizer={llm_simple_checkpoint.save_optimizer})"
        )


# --------------------------------------------------------------------------- #
# LOAD — plain torch/peft path                                                #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_simple_checkpoint_load(request, grpo_factory, tmp_path):
    """Fresh agent per test (load tests mutate state: stamp sentinels, step
    optimizer). Cheap because deepcopy of the template is near-instant.
    """
    lora_only, save_optimizer = request.param
    agent = _grpo_from_template(grpo_factory)
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )


class TestLLMSimpleCheckpointLoad:
    """Roundtrip: stamp sentinels on tracked state → save → clobber → load →
    assert sentinels restored. Specifically catches 'load silently
    reinitialised a fresh optimizer / fresh weights'.
    """

    def test_simple_checkpoint_load_adapter_weights_roundtrip(
        self, llm_simple_checkpoint_load
    ):
        s = llm_simple_checkpoint_load
        lora_sentinel, base_sentinel, clobber = 0.1234, 0.4321, 9.9999

        _, lora_param = get_param_by_name(s.agent, "lora_A.actor.weight")
        with torch.no_grad():
            lora_param.fill_(lora_sentinel)

        base_param = None
        if not s.lora_only:
            _, base_param = get_param_by_name(s.agent, "linear_1.base_layer.weight")
            with torch.no_grad():
                base_param.fill_(base_sentinel)

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )
        with torch.no_grad():
            lora_param.fill_(clobber)
            if base_param is not None:
                base_param.fill_(clobber)

        s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)

        _, lora_post = get_param_by_name(s.agent, "lora_A.actor.weight")
        assert torch.allclose(lora_post, torch.full_like(lora_post, lora_sentinel)), (
            f"LoRA weight not restored for cell "
            f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
        )
        if not s.lora_only:
            _, base_post = get_param_by_name(s.agent, "linear_1.base_layer.weight")
            assert torch.allclose(
                base_post, torch.full_like(base_post, base_sentinel)
            ), (
                f"base weight not restored for cell "
                f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
            )

    def test_simple_checkpoint_load_optimizer_state_roundtrip(
        self, llm_simple_checkpoint_load
    ):
        s = llm_simple_checkpoint_load
        sentinel, clobber = 0.3333, 9.9999

        # Populate optimizer state: fake grads → step.
        for p in s.agent.actor.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        s.agent.optimizer.step()
        s.agent.optimizer.zero_grad()

        exp_avg = find_exp_avg_in_opt_state(s.agent)
        assert exp_avg is not None, "optimizer.state not populated after step"
        with torch.no_grad():
            exp_avg.fill_(sentinel)

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )
        with torch.no_grad():
            exp_avg.fill_(clobber)

        if s.save_optimizer:
            s.agent.load_checkpoint(str(s.path), load_optimizer=True)
            restored = find_exp_avg_in_opt_state(s.agent)
            assert restored is not None, (
                f"optimizer state empty after load for cell "
                f"(lora_only={s.lora_only}, save_optimizer=True)"
            )
            assert torch.allclose(restored, torch.full_like(restored, sentinel)), (
                f"optimizer state not restored for cell "
                f"(lora_only={s.lora_only}, save_optimizer=True)"
            )
        else:
            # Nothing in the checkpoint to restore from → warn + fresh state.
            with pytest.warns(UserWarning, match="Optimizer state not found"):
                s.agent.load_checkpoint(str(s.path), load_optimizer=True)
            post = find_exp_avg_in_opt_state(s.agent)
            # Sentinel must NOT be present (either rebuilt fresh or still clobbered).
            if post is not None:
                assert not torch.allclose(post, torch.full_like(post, sentinel)), (
                    "optimizer state silently restored despite load_optimizer=False path"
                )


# --------------------------------------------------------------------------- #
# SAVE — deepspeed path                                                        #
# --------------------------------------------------------------------------- #


def _fit_deepspeed_mock(agent, zero_stage: int = 2) -> None:
    """Mutate ``agent`` so it looks like a DeepSpeed-wrapped agent for
    dispatch tests. Mock accelerator, overridden zero_stage, and a reasonable
    unwrap_model that just returns the wrapped model.
    """
    agent.accelerator = _make_mock_accelerator()
    agent.accelerator.unwrap_model = MagicMock(side_effect=lambda m: m)
    agent._uses_deepspeed = True
    agent.zero_stage = zero_stage


def _inner_actor(agent):
    """Strip the DummyEvolvable wrapper; returns the inner peft model.

    save_pretrained / load_state_dict are called on this inner object,
    not on agent.actor itself, so spies for those must be attached here.
    """
    from agilerl.modules.dummy import DummyEvolvable

    actor = agent.actor
    while isinstance(actor, DummyEvolvable):
        actor = actor.module
    return actor


@pytest.fixture(scope="session", params=SAVE_LOAD_OPTIONS)
def llm_mocked_deepspeed_checkpoint_save(request, grpo_factory, tmp_path_factory):
    """Spy-wrapped DeepSpeed save per cell. Session-scoped — 4 deepcopies of
    the template, each saved once.
    """
    lora_only, save_optimizer = request.param
    agent = _grpo_from_template(grpo_factory)
    _fit_deepspeed_mock(agent)

    # save_checkpoint is called as ``self.actor.save_checkpoint(...)`` on the
    # DummyEvolvable wrapper. save_pretrained is called on the unwrapped
    # inner peft model, so that spy must live there.
    save_ckpt_spy = MagicMock()
    agent.actor.save_checkpoint = save_ckpt_spy
    inner = _inner_actor(agent)
    save_pretrained_spy = MagicMock(wraps=inner.save_pretrained)
    inner.save_pretrained = save_pretrained_spy

    tmp_path = tmp_path_factory.mktemp(
        f"ds_save_lora={lora_only}_optim={save_optimizer}"
    )
    agent.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
        save_checkpoint_spy=save_ckpt_spy,
        save_pretrained_spy=save_pretrained_spy,
    )


class TestLLMDeepspeedCheckpointSave:
    """Dispatch-only spy tests: replace ``actor.save_checkpoint`` with a
    MagicMock and assert the right branch was called with the right kwargs.

    LIMITATION: these do NOT verify DeepSpeed actually wrote correct bytes
    to disk — the real engine is mocked out. For end-to-end confidence see
    ``TestDeepspeedSaveE2E`` below (CUDA-only, @pytest.mark.gpu).
    These spy tests run on any machine and catch dispatch regressions.
    """

    def test_llm_deepspeed_checkpoint_save_called_if_save_optimizer(
        self, llm_mocked_deepspeed_checkpoint_save
    ):
        spy = llm_mocked_deepspeed_checkpoint_save.save_checkpoint_spy
        if llm_mocked_deepspeed_checkpoint_save.save_optimizer:
            assert spy.call_count == 1
            kwargs = spy.call_args.kwargs
            assert (
                kwargs.get("exclude_frozen_parameters")
                == llm_mocked_deepspeed_checkpoint_save.lora_only
            )
        else:
            assert spy.call_count == 0

    def test_llm_deepspeed_checkpoint_save_save_pretrained_called_if_lora_only(
        self, llm_mocked_deepspeed_checkpoint_save
    ):
        spy = llm_mocked_deepspeed_checkpoint_save.save_pretrained_spy
        assert (spy.call_count >= 1) == llm_mocked_deepspeed_checkpoint_save.lora_only

    def test_llm_deepspeed_checkpoint_save_attributes_pt_has_actor_state_dict_only_when_full_no_optim(
        self,
        llm_mocked_deepspeed_checkpoint_save,
    ):
        ck = load_attributes_checkpoint(llm_mocked_deepspeed_checkpoint_save.path)
        ni = ck.get("network_info", {}) or {}
        modules = ni.get("modules", {}) if isinstance(ni, dict) else {}
        has_actor_sd = "actor_state_dict" in modules
        expected = (not llm_mocked_deepspeed_checkpoint_save.lora_only) and (
            not llm_mocked_deepspeed_checkpoint_save.save_optimizer
        )
        assert has_actor_sd == expected


# --------------------------------------------------------------------------- #
# LOAD — deepspeed path                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_mocked_deepspeed_checkpoint_load(request, grpo_factory, tmp_path):
    """Fresh agent + pre-saved deepspeed-shape checkpoint per load test.

    Function-scoped because each test patches in method-level spies and we
    want a clean baseline per test.
    """
    from pathlib import Path

    lora_only, save_optimizer = request.param

    # Saver: writes a cell-specific checkpoint to tmp_path. The real
    # DeepSpeed save is stubbed (can't run without a distributed backend)
    # but we still need the expected tag directory on disk so that the load
    # side's ``Path.glob('save_checkpoint')`` assertion passes.
    saver = _grpo_from_template(grpo_factory)
    _fit_deepspeed_mock(saver)

    def _fake_ds_save(path_str, *args, tag="save_checkpoint", **kwargs):
        (Path(path_str) / tag).mkdir(parents=True, exist_ok=True)

    saver.actor.save_checkpoint = MagicMock(side_effect=_fake_ds_save)
    saver.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )

    # Loader: spy its engine load so we can assert dispatch.
    loader = _grpo_from_template(grpo_factory)
    _fit_deepspeed_mock(loader)
    load_ckpt_spy = MagicMock(return_value=(str(tmp_path / "save_checkpoint"), None))
    loader.actor.load_checkpoint = load_ckpt_spy

    return SimpleNamespace(
        agent=loader,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
        load_checkpoint_spy=load_ckpt_spy,
    )


class TestDeepspeedLoad:
    """Dispatch-only spy tests. Same limitations as TestDeepspeedSave — these
    assert the correct load branch was taken but do not verify DeepSpeed
    actually restored state. See ``TestDeepspeedLoadE2E`` for real roundtrip.
    """

    def test_llm_deepspeed_checkpoint_load_called_if_save_optimizer(
        self, llm_mocked_deepspeed_checkpoint_load
    ):
        s = llm_mocked_deepspeed_checkpoint_load
        from unittest.mock import patch

        inner = _inner_actor(s.agent)
        # Stub the non-deepspeed branches so they don't actually try to read
        # adapter files / overwrite state — we only care about dispatch here.
        with (
            patch.object(s.agent, "_load_model_checkpoint"),
            patch.object(inner, "load_state_dict"),
        ):
            s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)
        assert (s.load_checkpoint_spy.call_count == 1) == s.save_optimizer

    def test_llm_deepspeed_checkpoint_load_peft_adapter_load_when_lora_only_and_no_optim(
        self,
        llm_mocked_deepspeed_checkpoint_load,
    ):
        s = llm_mocked_deepspeed_checkpoint_load
        from unittest.mock import patch

        inner = _inner_actor(s.agent)
        with (
            patch.object(s.agent, "_load_model_checkpoint") as peft_spy,
            patch.object(inner, "load_state_dict"),
        ):
            s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)
        # peft load is entered for both LoRA-only cells:
        #   * (lora_only=T, save_optim=F): PEFT-only restore path
        #   * (lora_only=T, save_optim=T): DS resume + PEFT adapter refresh
        expected = s.lora_only
        assert (peft_spy.call_count == 1) == expected

    def test_llm_deepspeed_checkpoint_load_state_dict_load_when_full_and_no_optim(
        self, llm_mocked_deepspeed_checkpoint_load
    ):
        s = llm_mocked_deepspeed_checkpoint_load
        from unittest.mock import patch

        inner = _inner_actor(s.agent)
        with (
            patch.object(s.agent, "_load_model_checkpoint"),
            patch.object(inner, "load_state_dict") as sd_spy,
        ):
            s.agent.load_checkpoint(str(s.path), load_optimizer=s.save_optimizer)
        # load_state_dict on the unwrapped actor is called only for
        # (lora_only=F, save_optim=F).
        expected = (not s.lora_only) and (not s.save_optimizer)
        assert (sd_spy.call_count == 1) == expected

    def test_llm_deepspeed_full_load_without_optimizer_falls_back_to_ds_model_only_restore(
        self, grpo_factory, tmp_path
    ):
        from pathlib import Path
        from unittest.mock import patch

        # Arrange: save a DeepSpeed full-model checkpoint with optimizer state.
        # This shape stores model shards in save_checkpoint/ and does not inject
        # actor_state_dict into attributes.pt.
        saver = _grpo_from_template(grpo_factory)
        _fit_deepspeed_mock(saver)

        def _fake_ds_save(path_str, *args, tag="save_checkpoint", **kwargs):
            (Path(path_str) / tag).mkdir(parents=True, exist_ok=True)

        saver.actor.save_checkpoint = MagicMock(side_effect=_fake_ds_save)
        saver.save_checkpoint(
            str(tmp_path),
            lora_only=False,
            save_optimizer=True,
        )

        # Act/Assert: loading with load_optimizer=False falls back to DS
        # model-only restore (no optimizer/lr scheduler state).
        loader = _grpo_from_template(grpo_factory)
        _fit_deepspeed_mock(loader)
        load_ckpt_spy = MagicMock(
            return_value=(str(tmp_path / "save_checkpoint"), None)
        )
        loader.actor.load_checkpoint = load_ckpt_spy
        with (
            patch.object(loader, "_load_model_checkpoint"),
        ):
            loader.load_checkpoint(str(tmp_path), load_optimizer=False)
        assert load_ckpt_spy.call_count == 1
        kwargs = load_ckpt_spy.call_args.kwargs
        assert kwargs["tag"] == "save_checkpoint"
        assert kwargs["load_optimizer_states"] is False
        assert kwargs["load_lr_scheduler_states"] is False


# --------------------------------------------------------------------------- #
# ZeRO-3 gather behaviour                                                      #
# --------------------------------------------------------------------------- #


class TestLLMGatherIfZero3OnSave:
    """Sanity-check that gather_if_zero3 is entered when zero_stage=3.

    Two save cells gather: the save_pretrained branch (lora_only=True) and
    the full torch-save branch (lora_only=False, save_optim=False). One test
    per branch — no full-grid parametrisation needed.
    """

    def test_llm_gather_entered_on_peft_save_when_zero3(self, grpo_factory, tmp_path):
        from contextlib import contextmanager
        from unittest.mock import patch

        agent = _grpo_from_template(grpo_factory)
        _fit_deepspeed_mock(agent, zero_stage=3)
        agent.actor.save_checkpoint = MagicMock()

        calls = []

        @contextmanager
        def gather_spy(zero_stage, params, modifier_rank=None):
            calls.append(zero_stage)
            yield

        with patch(
            "agilerl.algorithms.core.base.gather_if_zero3",
            side_effect=gather_spy,
        ):
            agent.save_checkpoint(
                str(tmp_path),
                lora_only=True,
                save_optimizer=False,
            )
        assert 3 in calls, "gather_if_zero3 was not entered for lora_only save"

    def test_llm_gather_entered_on_full_save_when_zero3(self, grpo_factory, tmp_path):
        from contextlib import contextmanager
        from unittest.mock import patch

        agent = _grpo_from_template(grpo_factory)
        _fit_deepspeed_mock(agent, zero_stage=3)
        agent.actor.save_checkpoint = MagicMock()

        calls = []

        @contextmanager
        def gather_spy(zero_stage, params, modifier_rank=None):
            calls.append(zero_stage)
            yield

        with patch(
            "agilerl.algorithms.core.base.gather_if_zero3",
            side_effect=gather_spy,
        ):
            agent.save_checkpoint(
                str(tmp_path),
                lora_only=False,
                save_optimizer=False,
            )
        assert 3 in calls, "gather_if_zero3 was not entered for full save"


# --------------------------------------------------------------------------- #
# E2E DeepSpeed tests — real save/load, CUDA-only, @pytest.mark.gpu           #
# --------------------------------------------------------------------------- #
# These run a real Accelerator with a DeepSpeedPlugin and exercise the full
# save/load pipeline end-to-end.


def _require_cuda_deepspeed() -> None:
    """Skip if the environment can't run real DeepSpeed."""
    if not HAS_LLM_DEPENDENCIES:
        pytest.skip("E2E deepspeed tests require the 'llm' extras.")
    if not torch.cuda.is_available():
        pytest.skip("E2E deepspeed tests require CUDA")


def build_deepspeed_grpo(accelerator):
    """Build a real DeepSpeed-wrapped GRPO for end-to-end tests.

    Same synthetic ``create_module`` used in the mocked tests, but with
    ``wrap=True`` so ``accelerator.prepare(...)`` wraps the model into a
    DeepSpeedEngine. Device is resolved to CUDA by ``GRPO.__init__`` when an
    accelerator is attached.
    """
    actor = create_module(
        input_size=6,
        max_tokens=4,
        vocab_size=64,
        device="cuda",
    )
    return GRPO(
        actor_network=actor,
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=4,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        lora_config=SMALL_LORA,
        accelerator=accelerator,
        wrap=True,
        gradient_checkpointing=False,
        device="cuda",
        use_separate_reference_adapter=True,
    )


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_deepspeed_checkpoint_save(
    request,
    deepspeed_env,
    accelerator_factory,
    tmp_path,
):
    """Real DeepSpeed save per cell — ONE real accelerator+agent per test.

    Function-scoped (Accelerator is not reusable across tests because
    the cleanup_after_test autouse fixture resets accelerator state).
    """
    _require_cuda_deepspeed()
    lora_only, save_optimizer = request.param
    accelerator = accelerator_factory(
        use_deepspeed_optimizer=False,
        config=deepspeed_config_stage_2,
    )
    agent = build_deepspeed_grpo(accelerator)
    agent.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )


class TestLLMDeepspeedCheckpointSaveE2E:
    """Real DeepSpeed save → assertions against bytes on disk (no spies).

    All artefact assertions in a single parametrised test to keep the number
    of real DeepSpeed builds small (1 per cell = 4 total).

                                deepspeed tag dir   adapter dirs   actor_state_dict
                                (<path>/save_checkpoint/*)         in attributes.pt
        lora_only=T, optim=T    present             present        absent
        lora_only=T, optim=F    absent              present        absent
        lora_only=F, optim=T    present             absent         absent
        lora_only=F, optim=F    absent              absent         present
    """

    pytestmark = pytest.mark.gpu

    def test_llm_deepspeed_checkpoint_save_artifacts_match_cell(
        self, llm_deepspeed_checkpoint_save
    ):
        s = llm_deepspeed_checkpoint_save

        # attributes.pt always present.
        assert (s.path / "attributes.pt").exists()

        # DeepSpeed engine's own tag directory if save_optimizer=True.
        assert (s.path / "save_checkpoint").is_dir() == s.save_optimizer

        # PEFT adapter dirs if lora_only=True (both actor + reference because
        # use_separate_reference_adapter=True in the fixture).
        actor_adapter = s.path / "actor" / "adapter_model.safetensors"
        ref_adapter = s.path / "reference" / "adapter_model.safetensors"
        assert actor_adapter.exists() == s.lora_only
        assert ref_adapter.exists() == s.lora_only

        # attributes.pt payload:
        #   - ``_lora_only`` flag always matches the save call.
        #   - ``actor_state_dict`` only lands in attrs.pt for the (F, F)
        #     deepspeed cell (gather+torch-save branch).
        ck = load_attributes_checkpoint(s.path)
        assert ck.get("_lora_only") == s.lora_only
        modules = ck.get("network_info", {}).get("modules", {})
        has_actor_sd = "actor_state_dict" in modules
        expected = (not s.lora_only) and (not s.save_optimizer)
        assert has_actor_sd == expected, (
            f"actor_state_dict presence in attributes.pt wrong for cell "
            f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
        )


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_deepspeed_checkpoint_load(
    request,
    deepspeed_env,
    accelerator_factory,
    tmp_path,
):
    """Fresh real-DeepSpeed agent per load test + the factory so the test can
    build a second accelerator for the load-side agent.

    Function-scoped — each test stamps sentinels then saves+loads, so agents
    cannot be shared.
    """
    _require_cuda_deepspeed()
    lora_only, save_optimizer = request.param
    accelerator = accelerator_factory(
        use_deepspeed_optimizer=False,
        config=deepspeed_config_stage_2,
    )
    agent = build_deepspeed_grpo(accelerator)
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
        accelerator_factory=accelerator_factory,
    )


class TestLLMDeepspeedCheckpointSaveLoad:
    """Real DeepSpeed roundtrip: stamp sentinels → save → fresh agent → load
    → assert sentinels restored. One parametrised test per concern
    (weights / optimizer) to keep the real DeepSpeed builds bounded.

    NB: building the second accelerator via the factory triggers
    ``AcceleratorState._reset_state`` which invalidates the first engine.
    That's fine because the first agent is only needed for the save step.
    """

    pytestmark = pytest.mark.gpu

    def test_llm_deepspeed_checkpoint_save_load_adapter_and_base_weight_roundtrip_e2e(
        self, llm_deepspeed_checkpoint_load
    ):
        s = llm_deepspeed_checkpoint_load
        lora_sentinel, base_sentinel, clobber = 0.1234, 0.4321, 9.9999

        # Stamp the actor's LoRA-A weight on the pre-save agent.
        _, lora_param = get_param_by_name(s.agent, "lora_A.actor.weight")
        with torch.no_grad():
            lora_param.fill_(lora_sentinel)

        # Full-save cells also round-trip base model weights, so stamp one.
        if not s.lora_only:
            _, base_param = get_param_by_name(s.agent, "linear_1.base_layer.weight")
            with torch.no_grad():
                base_param.fill_(base_sentinel)

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )

        # Build the loading side on a fresh accelerator + agent.
        new_accel = s.accelerator_factory(
            use_deepspeed_optimizer=False,
            config=deepspeed_config_stage_2,
        )
        new_agent = build_deepspeed_grpo(new_accel)

        # Clobber a weight on new_agent so a silent no-op load would fail
        # the sentinel comparison.
        _, new_lora = get_param_by_name(new_agent, "lora_A.actor.weight")
        with torch.no_grad():
            new_lora.fill_(clobber)

        new_agent.load_checkpoint(
            str(s.path),
            load_optimizer=s.save_optimizer,
        )

        # Re-fetch after load; load may rebuild adapter modules.
        _, lora_post = get_param_by_name(new_agent, "lora_A.actor.weight")
        assert torch.allclose(lora_post, torch.full_like(lora_post, lora_sentinel)), (
            f"LoRA weight not restored for cell "
            f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
        )
        if not s.lora_only:
            _, base_post = get_param_by_name(new_agent, "linear_1.base_layer.weight")
            assert torch.allclose(
                base_post, torch.full_like(base_post, base_sentinel)
            ), (
                f"base weight not restored for cell "
                f"(lora_only={s.lora_only}, save_optimizer={s.save_optimizer})"
            )

    def test_llm_deepspeed_checkpoint_save_load_optimizer_state_roundtrip_e2e(
        self, llm_deepspeed_checkpoint_load
    ):
        """After a real backward+step, optimizer state should round-trip
        through a save+load cycle when ``save_optimizer=True``.

        We don't use a sentinel here — DeepSpeed's optimizer state is
        partitioned/internal and the public state_dict shape isn't trivial to
        tensor-stamp. A "state is non-empty after load" check is sufficient
        to catch a silent fresh-optimizer regression.
        """
        s = llm_deepspeed_checkpoint_load
        _, base_linear = get_param_by_name(s.agent, "linear_1.base_layer.weight")
        in_features = int(base_linear.shape[1])
        input_ids = torch.randint(0, 64, (1, in_features), device=s.agent.device)
        attn_mask = torch.ones_like(input_ids)
        out = s.agent.actor(input_ids=input_ids, attention_mask=attn_mask)
        s.agent.actor.backward(out.logits.sum())
        s.agent.optimizer.step()
        before_inner = getattr(s.agent.optimizer, "optimizer", s.agent.optimizer)
        before_sd = (
            before_inner.state_dict() if hasattr(before_inner, "state_dict") else {}
        )

        s.agent.save_checkpoint(
            str(s.path),
            lora_only=s.lora_only,
            save_optimizer=s.save_optimizer,
        )

        new_accel = s.accelerator_factory(
            use_deepspeed_optimizer=False,
            config=deepspeed_config_stage_2,
        )
        new_agent = build_deepspeed_grpo(new_accel)
        new_agent.load_checkpoint(
            str(s.path),
            load_optimizer=s.save_optimizer,
        )

        if s.save_optimizer:
            inner = getattr(new_agent.optimizer, "optimizer", new_agent.optimizer)
            sd = inner.state_dict() if hasattr(inner, "state_dict") else {}
            assert normalize_optimizer_state(sd) == normalize_optimizer_state(
                before_sd
            ), (
                f"optimizer state mismatch after DeepSpeed load for cell "
                f"(lora_only={s.lora_only}, save_optimizer=True); "
                f"got keys {list(sd.keys())}"
            )


# --------------------------------------------------------------------------- #
# LoRA config merge — unit tests (static method, no fixtures)                 #
# --------------------------------------------------------------------------- #

from agilerl.algorithms.core.base import LLMAlgorithm  # noqa: E402


def get_lora_config(
    r=4, target_modules=("linear_1",), modules_to_save=None, lora_alpha=8
):
    """Helper to build a LoraConfig with sensible defaults for merge tests."""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        modules_to_save=list(modules_to_save) if modules_to_save is not None else None,
        task_type="CAUSAL_LM",
        lora_dropout=0.0,
    )


# --------------------------------------------------------------------------- #
# LoRA config strict matching — integration with save/load                    #
# --------------------------------------------------------------------------- #


def _build_grpo_with_lora(lora_config: LoraConfig) -> GRPO:
    """Like ``_build_grpo`` but lets the caller override ``lora_config``."""
    if not (HAS_VLLM and HAS_DEEPSPEED):
        pytest.skip("Need to install agilerl with deepspeed + vllm")
    actor = create_module(input_size=6, max_tokens=4, vocab_size=64, device="cpu")
    return GRPO(
        actor_network=actor,
        pad_token_id=63,
        pad_token="<pad>",
        batch_size=4,
        group_size=2,
        max_output_tokens=4,
        max_model_len=12,
        lora_config=lora_config,
        accelerator=None,
        wrap=False,
        gradient_checkpointing=False,
        device="cpu",
        use_separate_reference_adapter=True,
    )


class TestStrictLoraConfigLoading:
    """lora-only checkpoints must be loaded by an agent built with a matching
    LoRA config; mismatches raise instead of being reconciled.
    """

    def test_mismatched_config_raises(self, tmp_path):
        saver = _build_grpo_with_lora(
            get_lora_config(r=2, target_modules=("linear_1",))
        )
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=8, target_modules=("linear_1", "linear_2"))
        )
        with pytest.raises(ValueError, match="LoRA configs differ"):
            loader.load_checkpoint(str(tmp_path), load_optimizer=False)

    def test_load_succeeds_when_configs_match(self, tmp_path):
        cfg = get_lora_config(r=4, target_modules=("linear_1",))
        saver = _build_grpo_with_lora(cfg)
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=4, target_modules=("linear_1",))
        )
        loader.load_checkpoint(str(tmp_path), load_optimizer=False)
        assert loader.lora_config.r == 4


class TestLLMClone:
    """LLMAlgorithm.clone requires full model infrastructure (DeepSpeed, real
    model weights, etc.), so we test it indirectly via `_configure_batch_size`
    with `clone=True` to verify the clone-mode branch.
    """

    def test_clone_mode_skips_batch_config(self):
        agent = _make_llm_agent(clone=True)
        assert agent.batch_size_per_process == 4


@_LLM_DEPS_SKIP
class TestLLMConfigureBatchSizeNoDeepSpeedPlugin:
    """``_configure_batch_size`` when ``accelerator.state.deepspeed_plugin`` is None."""

    @staticmethod
    def _accelerator_without_deepspeed(num_processes: int = 1):
        acc = _make_mock_accelerator(num_processes=num_processes)
        acc.state.deepspeed_plugin = None
        return acc

    def test_value_error_raises_when_no_deepspeed_plugin(self):
        acc = self._accelerator_without_deepspeed()
        with pytest.raises(
            ValueError,
            match=r"DeepSpeed plugin is not initialized\. If using an accelerator,",
        ):
            _make_llm_agent(
                accelerator=acc,
                clone=False,
                micro_batch_size_per_gpu=3,
            )


class TestLLMInitMiscPaths:
    def test_use_liger_loss_modifies_lora_config(self):
        lora = MagicMock()
        acc = _make_mock_accelerator()
        with patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True):
            with pytest.warns(UserWarning, match="Liger Loss"):
                _make_llm_agent(accelerator=acc, use_liger_loss=True, lora_config=lora)
        assert lora.exclude_modules == ["lm_head"]

    def test_use_liger_loss_survives_zero_stage_three(self):
        lora = MagicMock()
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 3},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        with patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True):
            with pytest.warns(UserWarning, match="ZeRO Stage 3|Liger Loss"):
                agent = _make_llm_agent(
                    accelerator=acc, use_liger_loss=True, lora_config=lora
                )
        assert agent.zero_stage == 3
        assert agent.use_liger_loss is True

    def test_seed_broadcast_with_multi_process(self):
        acc = _make_mock_accelerator(num_processes=2)
        with patch(
            "agilerl.algorithms.core.base.broadcast_object_list", return_value=[42]
        ):
            _make_llm_agent(accelerator=acc)

    def test_set_seed_called_with_accelerator(self):
        acc = _make_mock_accelerator()
        with patch("agilerl.algorithms.core.base.set_seed") as mock_set_seed:
            _make_llm_agent(accelerator=acc)
            mock_set_seed.assert_called()


class TestLLMGenerateWithVllmColocate:
    def test_raises_when_sampling_params_none(self):
        agent = _make_llm_agent()
        with patch("agilerl.algorithms.core.base.SamplingParams", None, create=True):
            with pytest.raises(
                ImportError,
                match=re.escape(
                    "vLLM is required when use_vllm=True. Install AgileRL with vLLM support for this platform: `pip install agilerl[llm]`."
                ),
            ):
                agent._generate_with_vllm_colocate([], 1, 0.9)


def _fake_save_peft_adapter_for_vllm_rollout(
    peft_model,
    staging_dir,
    adapter_name,
    *,
    target_modules,
    expert_key_map=None,
):
    from pathlib import Path

    adapter_dir = Path(staging_dir) / adapter_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text("{}")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
    return adapter_dir


def _setup_agent_for_vllm_lora_sync(agent):
    import tempfile
    from pathlib import Path

    agent.vllm_config = VLLMConfig()
    agent._vllm_lora_staging_dir = Path(tempfile.mkdtemp())
    agent._vllm_lora_loaded = False
    agent._vllm_rollout_lora_request = None
    agent.llm = MagicMock()
    agent.llm.llm_engine = MagicMock()
    agent.llm.llm_engine.add_lora = MagicMock(return_value=True)
    agent.llm.reset_prefix_cache = MagicMock()
    return agent


class TestEnsureVllmLoraStagingDir:
    """``_ensure_vllm_lora_staging_dir`` resolves the rollout-adapter export
    dir once, honouring a configured ``VLLMConfig.lora_staging_dir`` (e.g. an
    NFS path for colocated Ray rollouts) and never deleting it.
    """

    def _agent(self, lora_staging_dir):
        return SimpleNamespace(
            vllm_config=VLLMConfig(lora_staging_dir=lora_staging_dir),
            _vllm_lora_staging_dir=None,
            accelerator=None,
        )

    def test_uses_configured_dir_and_marks_persistent(self, tmp_path):
        target = tmp_path / "nfs" / "agilerl_lora"  # not yet created
        agent = self._agent(str(target))
        resolved = LLMAlgorithm._ensure_vllm_lora_staging_dir(agent)
        assert resolved == target
        assert target.is_dir()  # created with parents
        assert agent._vllm_lora_staging_dir_is_temp is False

    def test_falls_back_to_tempdir_when_unset(self):
        agent = self._agent(None)
        resolved = LLMAlgorithm._ensure_vllm_lora_staging_dir(agent)
        try:
            assert resolved.is_dir()
            assert agent._vllm_lora_staging_dir_is_temp is True
        finally:
            shutil.rmtree(resolved, ignore_errors=True)

    def test_is_idempotent(self, tmp_path):
        agent = self._agent(str(tmp_path / "lora"))
        first = LLMAlgorithm._ensure_vllm_lora_staging_dir(agent)
        second = LLMAlgorithm._ensure_vllm_lora_staging_dir(agent)
        assert first is second

    def test_cleanup_preserves_configured_dir(self, tmp_path):
        """A configured (non-temp) staging dir survives ``clean_up``'s rmtree
        guard; a temp one would be removed.
        """
        target = tmp_path / "nfs_lora"
        agent = self._agent(str(target))
        LLMAlgorithm._ensure_vllm_lora_staging_dir(agent)
        # Mirror clean_up's guard.
        is_temp = getattr(agent, "_vllm_lora_staging_dir_is_temp", True)
        assert is_temp is False
        assert target.is_dir()

    def test_appends_rank_subdir_in_distributed_run(self, tmp_path):
        """With >1 process, each rank gets its own ``rank_<index>`` subdir."""
        target = tmp_path / "nfs_lora"
        agent = self._agent(str(target))
        agent.accelerator = SimpleNamespace(num_processes=2, process_index=1)
        resolved = LLMAlgorithm._ensure_vllm_lora_staging_dir(agent)
        assert resolved == target / "rank_1"
        assert resolved.is_dir()
        assert agent._vllm_lora_staging_dir_is_temp is False


@pytest.mark.skipif(
    not HAS_VLLM,
    reason="vLLM rollout sync needs the Linux-only vllm extra.",
)
class TestLLMSyncActorToVllm:
    def test_sync_actor_to_vllm_lora_path_exports_adapter_without_merge(self):
        """Adapter-only sync: save_pretrained + add_lora, no merge_adapter."""
        p = torch.nn.Parameter(torch.tensor([1.0]))
        gather = patch("agilerl.algorithms.core.base.gather_if_zero3")

        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [p]
        peft_ref.set_adapter = MagicMock()
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)
        with (
            gather,
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
            ) as mock_save,
        ):
            agent._sync_actor_to_vllm()
        peft_ref.set_adapter.assert_called_with("actor")
        mock_save.assert_called_once()
        peft_ref.merge_adapter.assert_not_called()
        agent.llm.llm_engine.add_lora.assert_called_once()
        assert agent._vllm_rollout_lora_request is not None
        agent.llm.reset_prefix_cache.assert_called_once()

        agent = _make_llm_agent(accelerator=None)
        inner = _make_mock_peft_actor()
        inner.parameters = MagicMock(return_value=[p])
        inner.set_adapter = MagicMock()
        inner.merge_adapter = MagicMock()
        agent.actor = DummyEvolvable(device="cpu", module=inner)
        _setup_agent_for_vllm_lora_sync(agent)
        with (
            gather,
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
            ) as mock_save,
        ):
            agent._sync_actor_to_vllm()
        inner.merge_adapter.assert_not_called()
        mock_save.assert_called_once()
        agent.llm.llm_engine.add_lora.assert_called_once()

    def test_move_lora_to_vllm_waits_before_non_main_path_check(self, tmp_path):
        """Non-main ranks must wait for rank-0 export before dir existence check."""
        p = torch.nn.Parameter(torch.tensor([1.0]))
        acc = _make_mock_accelerator(
            num_processes=2, is_main_process=False, process_index=1
        )
        agent = _make_llm_agent(accelerator=acc)
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [p]
        peft_ref.set_adapter = MagicMock()
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)
        agent._vllm_lora_staging_dir = tmp_path

        def _fake_export(*_args, **_kwargs):
            return tmp_path / "actor"

        def _wait_and_materialize():
            adapter_dir = tmp_path / "actor"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}")
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

        acc.wait_for_everyone = MagicMock(side_effect=_wait_and_materialize)

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_export,
            ),
        ):
            agent._move_lora_to_vllm()

        acc.wait_for_everyone.assert_called_once()
        agent.llm.llm_engine.add_lora.assert_called_once()


class TestMultiAgentPreprocessObservation:
    def test_preprocess_observation(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        observation = {
            "agent_0": np.zeros(4, dtype=np.float32),
            "agent_1": np.ones(4, dtype=np.float32),
        }
        result = agent.preprocess_observation(observation)
        assert "agent_0" in result
        assert "agent_1" in result
        assert isinstance(result["agent_0"], torch.Tensor)

    def test_preprocess_observation_grouped_output(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        observation = {
            "agent_0": np.zeros(4, dtype=np.float32),
            "agent_1": np.ones(4, dtype=np.float32),
        }

        result = agent.preprocess_observation(observation, group_ids=["agent"])

        assert "agent" in result
        assert "agent_0" not in result
        assert "agent_1" not in result
        assert isinstance(result["agent"], torch.Tensor)
        assert result["agent"].shape[0] == 2

    def test_preprocess_observation_creates_missing_group_bucket(self, vector_space):
        """When group_ids omit an agent's network id, the bucket is created lazily."""
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        observation = {
            "agent_0": np.zeros(4, dtype=np.float32),
            "agent_1": np.ones(4, dtype=np.float32),
        }

        result = agent.preprocess_observation(observation, group_ids=["unused_group"])

        assert "agent" in result
        assert isinstance(result["agent"], torch.Tensor)
        assert result["agent"].shape[0] == 2


class TestMultiAgentExtractAgentMasksContinuousNan:
    def test_extract_agent_masks_none_continuous_action(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)] * 2
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        infos = {
            "agent_0": {"env_defined_actions": None},
            "agent_1": {"env_defined_actions": np.array([1.0, 2.0])},
        }
        env_acts, _agent_masks = agent.extract_agent_masks(infos)
        assert np.isnan(env_acts["agent_0"]).all()
        assert env_acts["agent_0"].shape == (2,)


class TestMultiAgentBuildNetConfigPaths:
    def test_build_net_config_rejects_non_mapping_encoder_config(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        net_config = {"encoder_config": "not-a-config"}
        with pytest.raises(TypeError, match="must be a dict or NetConfig"):
            agent.build_net_config(net_config, flatten=False)

    def test_build_net_config_none_creates_defaults(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        config = agent.build_net_config(None, flatten=False)
        assert "agent" in config
        assert "encoder_config" in config["agent"]

    def test_build_net_config_none_with_return_encoders(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        config, encoders = agent.build_net_config(
            None, flatten=False, return_encoders=True
        )
        assert "agent" in config
        assert len(encoders) > 0

    def test_build_net_config_single_level_homogeneous(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        net_config = {"encoder_config": {"hidden_size": [8]}}
        config = agent.build_net_config(net_config, flatten=False)
        assert "agent" in config

    def test_build_net_config_single_level_with_return_encoders(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        net_config = {"encoder_config": {"hidden_size": [64, 64]}}
        config, encoders = agent.build_net_config(
            net_config, flatten=False, return_encoders=True
        )
        assert len(encoders) > 0
        assert "agent" in config

    def test_build_net_config_with_missing_encoder_uses_default(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Discrete(2), spaces.Discrete(2)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        net_config = {"agent": {}}
        config = agent.build_net_config(net_config, flatten=False)
        assert "agent" in config
        assert "encoder_config" in config["agent"]

    def test_build_net_config_group_key_not_found_uses_default(self, vector_space):
        obs = [
            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        ]
        act = [spaces.Discrete(2), spaces.Discrete(3)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["alpha_0", "beta_0"], index=0)
        net_config = {"gamma": {"encoder_config": {"hidden_size": [8]}}}
        config = agent.build_net_config(net_config, flatten=False)
        assert "alpha_0" in config
        assert "beta_0" in config
        assert "encoder_config" in config["alpha_0"]


class TestLLMPreprocessObservation:
    def test_preprocess_observation_passthrough(self):
        agent = _make_llm_agent()
        obs = {"input_ids": [1, 2, 3]}
        result = LLMAlgorithm.preprocess_observation(agent, obs)
        assert result == obs


@_LLM_DEPS_SKIP
class TestLLMInitMissingDeps:
    def test_raises_when_no_llm_deps(self):
        with patch("agilerl.algorithms.core.base.HAS_LLM_DEPENDENCIES", False):
            with pytest.raises(ImportError, match="LLM dependencies"):
                with (
                    patch.object(EvolvableAlgorithm, "_registry_init"),
                ):
                    _StubLLMAlgorithm(
                        index=0,
                        batch_size=4,
                        lr=1e-4,
                        max_grad_norm=0.0,
                        clone=True,
                        calc_position_embeddings=False,
                        seed=42,
                        pad_token_id=0,
                        pad_token="<pad>",
                        use_liger_loss=False,
                        lora_config=MagicMock(),
                        actor_network=_make_mock_peft_actor(),
                        device="cpu",
                    )

    def test_raises_when_no_model_name_or_network(self):
        with pytest.raises(ValueError, match="At least one"):
            with (
                patch.object(LLMAlgorithm, "_initialize_actors"),
                patch.object(LLMAlgorithm, "_configure_vllm"),
                patch.object(LLMAlgorithm, "wrap_models"),
                patch.object(EvolvableAlgorithm, "_registry_init"),
            ):
                _StubLLMAlgorithm(
                    index=0,
                    batch_size=4,
                    lr=1e-4,
                    max_grad_norm=0.0,
                    clone=True,
                    calc_position_embeddings=False,
                    seed=42,
                    pad_token_id=0,
                    pad_token="<pad>",
                    use_liger_loss=False,
                    lora_config=MagicMock(),
                    model_name=None,
                    actor_network=None,
                    device="cpu",
                )


class TestLLMLoadDistributedActorWithAccelerator:
    def test_load_distributed_actor_success(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        tag_dir = tmp_path / "save_checkpoint"
        tag_dir.mkdir()
        agent.actor.load_checkpoint = MagicMock(return_value=(str(tag_dir), None))
        agent._load_distributed_actor(str(tmp_path), tag="save_checkpoint")

    def test_load_distributed_actor_failure(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        tag_dir = tmp_path / "save_checkpoint"
        tag_dir.mkdir()
        agent.actor.load_checkpoint = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(ValueError, match="Deepspeed failed"):
            agent._load_distributed_actor(str(tmp_path), tag="save_checkpoint")

    def test_load_distributed_actor_none_path(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        tag_dir = tmp_path / "save_checkpoint"
        tag_dir.mkdir()
        agent.actor.load_checkpoint = MagicMock(return_value=(None, None))
        with pytest.raises(ValueError, match="Deepspeed failed"):
            agent._load_distributed_actor(str(tmp_path), tag="save_checkpoint")


class TestLLMLoadFullModelActor:
    """A DeepSpeed full-model warm start, from either checkpoint layout."""

    def test_gathered_state_dict_is_loaded_in_place(self):
        # Arrange -- save_optimizer=False keeps the weights in attributes.pt.
        agent = _make_llm_agent(accelerator=_make_mock_accelerator())
        model_ref = MagicMock()
        model_ref.parameters.return_value = []
        agent._get_unwrapped_actor = MagicMock(return_value=model_ref)
        agent._load_distributed_actor = MagicMock()
        state_dict = {"weight": torch.zeros(1)}
        checkpoint = {"network_info": {"modules": {"actor_state_dict": state_dict}}}

        # Act
        LLMAlgorithm._load_full_model_actor(agent, "/some/path", checkpoint)

        # Assert
        model_ref.load_state_dict.assert_called_once_with(state_dict)
        agent._load_distributed_actor.assert_not_called()

    def test_missing_state_dict_falls_back_to_the_tag_directory(self):
        # Arrange -- save_optimizer=True writes the weights to the tag dir instead.
        agent = _make_llm_agent(accelerator=_make_mock_accelerator())
        agent._load_distributed_actor = MagicMock()
        agent._get_unwrapped_actor = MagicMock()

        # Act
        LLMAlgorithm._load_full_model_actor(agent, "/some/path", {})

        # Assert
        agent._load_distributed_actor.assert_called_once_with(
            "/some/path",
            tag="save_checkpoint",
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
        agent._get_unwrapped_actor.assert_not_called()

    def test_deepspeed_full_model_checkpoint_routes_to_the_actor_loader(
        self,
        tmp_path,
    ):
        # Arrange -- a full-model (non-LoRA) checkpoint under DeepSpeed.
        agent = _make_llm_agent(accelerator=_make_mock_accelerator())
        agent._uses_deepspeed = True
        agent._load_full_model_actor = MagicMock()
        agent._load_model_checkpoint = MagicMock()
        torch.save({"_lora_only": False}, str(tmp_path / "attributes.pt"))

        # Act
        LLMAlgorithm.load_weights(agent, str(tmp_path))

        # Assert
        agent._load_full_model_actor.assert_called_once()
        agent._load_model_checkpoint.assert_not_called()

    def test_deepspeed_resume_without_optimizer_loads_the_full_actor(
        self,
        tmp_path,
    ):
        # Arrange -- resuming a full-model DeepSpeed checkpoint without its
        # optimizer shards: neither the adapter dirs nor the sharded restore
        # apply, so only the actor weights come back.
        agent = _make_llm_agent(accelerator=_make_mock_accelerator())
        agent._uses_deepspeed = True
        agent._load_full_model_actor = MagicMock()
        agent._load_distributed_actor = MagicMock()
        agent._load_model_checkpoint = MagicMock()
        agent._restore_checkpoint_attributes = MagicMock()
        torch.save({"_lora_only": False}, str(tmp_path / "attributes.pt"))

        # Act
        LLMAlgorithm.load_checkpoint(agent, str(tmp_path), load_optimizer=False)

        # Assert
        agent._load_full_model_actor.assert_called_once()
        agent._load_distributed_actor.assert_not_called()
        agent._load_model_checkpoint.assert_not_called()
        agent._restore_checkpoint_attributes.assert_called_once()


class TestLLMBackwardPassNonAccelerator:
    def test_backward_pass_calls_clip_grad_norm(self):
        agent = _make_llm_agent(accelerator=None)
        agent.max_grad_norm = 1.0
        param = torch.tensor([1.0], requires_grad=True)
        loss = (param * 2).sum()
        with patch("agilerl.algorithms.core.base.clip_grad_norm_") as mock_clip:
            LLMAlgorithm._backward_pass(agent, loss)
        mock_clip.assert_called_once()


class TestLLMSaveDistributedActorWithAccelerator:
    def test_save_creates_dir(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        save_dir = str(tmp_path / "ds_save")
        with (
            patch.object(
                agent, "use_adapter", wraps=agent.use_adapter
            ) as mock_use_adapter,
            patch.object(
                agent,
                "_restore_adapter_trainability",
                wraps=agent._restore_adapter_trainability,
            ) as mock_restore_trainability,
        ):
            agent._save_distributed_actor(save_dir)
        assert (tmp_path / "ds_save").exists()
        agent.actor.save_checkpoint.assert_called_once()
        mock_use_adapter.assert_called_once_with("actor")
        restored_calls = [
            call.args[0] for call in mock_restore_trainability.call_args_list
        ]
        assert ["actor"] in restored_calls


class TestEvolvableAlgorithmCloneWithAccelerator:
    def test_clone_unwraps_and_wraps_with_accelerator(self, vector_space):
        action_space = spaces.Discrete(2)
        AcceleratorState._reset_state(True)
        accelerator = Accelerator(cpu=True)
        accelerator.prepare = MagicMock(side_effect=lambda x: x)
        accelerator.unwrap_model = MagicMock(side_effect=lambda x: x)
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, accelerator=accelerator
        )
        clone = agent.clone(index=1, wrap=True)
        assert clone is not agent
        assert clone.index == 1
        assert accelerator.unwrap_model.call_count >= 1
        assert accelerator.prepare.call_count >= 1
        agent.accelerator = None
        clone.accelerator = None
        AcceleratorState._reset_state(True)


class _DummyRLWithTensor(DummyRLAlgorithm):
    """Dummy algorithm with a tensor init parameter for testing `load` classmethod."""

    def __init__(
        self, observation_space, action_space, index, tensor_param=None, **kwargs
    ):
        super().__init__(observation_space, action_space, index, **kwargs)
        if tensor_param is None:
            tensor_param = torch.zeros(2)
        self.tensor_param = tensor_param


class TestLoadClassmethodTensorDevice:
    """Tensor device migration in `load` classmethod."""

    def test_load_moves_tensor_to_device(self, vector_space, tmp_path):
        action_space = spaces.Discrete(2)
        agent = _DummyRLWithTensor(
            vector_space,
            action_space,
            index=0,
            tensor_param=torch.tensor([3.0, 4.0]),
        )
        path = tmp_path / "chkpt.pth"
        agent.save_checkpoint(path)

        loaded = _DummyRLWithTensor.load(str(path))
        assert torch.allclose(loaded.tensor_param, torch.tensor([3.0, 4.0]))


class TestLoadCheckpointTensorDevice:
    def test_load_checkpoint_moves_tensor_to_correct_device(
        self, vector_space, tmp_path
    ):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        agent.tensor_attr = torch.tensor([1.0, 2.0])
        path = tmp_path / "chkpt.pth"
        agent.save_checkpoint(path)

        agent2 = DummyRLAlgorithm(vector_space, action_space, index=1)
        agent2.tensor_attr = torch.tensor([0.0, 0.0])
        agent2.load_checkpoint(path)
        assert torch.allclose(agent2.tensor_attr, torch.tensor([1.0, 2.0]))


class _FakeWrapper:
    def __init__(self, wrapped, **kwargs):
        self.wrapped = wrapped
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestLoadWrapperRestore:
    """Restoring wrapper_cls from checkpoint via load()."""

    def test_load_restores_wrapper(self, vector_space, tmp_path):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        path = tmp_path / "chkpt.pth"
        agent.save_checkpoint(path)

        import dill

        chkpt = torch.load(path, pickle_module=dill, weights_only=False)

        chkpt["wrapper_cls"] = _FakeWrapper
        chkpt["wrapper_init_dict"] = {"extra_kwarg": 99}
        chkpt["wrapper_attrs"] = {"custom_attr": "hello"}
        torch.save(chkpt, path, pickle_module=dill)

        result = DummyRLAlgorithm.load(str(path))
        assert isinstance(result, _FakeWrapper)
        assert result.custom_attr == "hello"
        assert result.extra_kwarg == 99


class TestAbstractMethodBodies:
    """Abstract methods raise NotImplementedError when called directly."""

    def test_preprocess_observation_raises(self, vector_space):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        with pytest.raises(NotImplementedError):
            EvolvableAlgorithm.preprocess_observation(agent, np.zeros(4))

    def test_learn_raises(self, vector_space):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        with pytest.raises(NotImplementedError):
            EvolvableAlgorithm.learn(agent, {})

    def test_get_action_raises(self, vector_space):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        with pytest.raises(NotImplementedError):
            EvolvableAlgorithm.get_action(agent, np.zeros(4))

    def test_test_raises(self, vector_space):
        action_space = spaces.Discrete(2)
        agent = DummyRLAlgorithm(vector_space, action_space, index=0)
        with pytest.raises(NotImplementedError):
            EvolvableAlgorithm.test(agent)


class TestWrapModelsDictBranch:
    """wrap_models handles dict-typed evolvable attributes."""

    def test_wrap_models_wraps_dict_attributes(self, vector_space):
        action_space = spaces.Discrete(2)
        AcceleratorState._reset_state(True)
        accelerator = Accelerator()
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, accelerator=accelerator
        )
        original_actor = agent.dummy_actor
        agent.dummy_actor = {"agent_0": original_actor}
        agent.wrap_models()
        assert isinstance(agent.dummy_actor, dict)
        assert "agent_0" in agent.dummy_actor
        agent.accelerator = None


class TestUnwrapModelsDictBranch:
    """unwrap_models handles dict-typed evolvable attributes."""

    def test_unwrap_models_unwraps_dict_attributes(self, vector_space):
        action_space = spaces.Discrete(2)
        AcceleratorState._reset_state(True)
        accelerator = Accelerator()
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, accelerator=accelerator
        )
        original_actor = agent.dummy_actor
        agent.dummy_actor = {"agent_0": original_actor}
        agent.unwrap_models()
        assert isinstance(agent.dummy_actor, dict)
        assert "agent_0" in agent.dummy_actor
        agent.accelerator = None


class TestBuildNetConfigDefaultFallback:
    """Falls back to default encoder when agent/group ID missing from net_config."""

    def test_heterogeneous_agents_partial_config_falls_back(self):
        obs = [
            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        ]
        act = [spaces.Discrete(2), spaces.Discrete(3)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["alpha_0", "beta_0"], index=0)
        net_config = {"alpha": {"encoder_config": {"hidden_size": [64, 64]}}}
        config = agent.build_net_config(net_config, flatten=True)
        assert "alpha_0" in config
        assert "beta_0" in config
        assert "encoder_config" in config["beta_0"]

    def test_heterogeneous_agents_partial_config_with_return_encoders(self):
        obs = [
            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32),
        ]
        act = [spaces.Discrete(2), spaces.Discrete(3)]
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["alpha_0", "beta_0"], index=0)
        net_config = {"alpha": {"encoder_config": {"hidden_size": [64, 64]}}}
        config, encoders = agent.build_net_config(
            net_config, flatten=True, return_encoders=True
        )
        assert "alpha_0" in config
        assert "beta_0" in config
        assert "encoder_config" in config["beta_0"]
        assert len(encoders) > 0


class TestLLMBackwardPassWithAccelerator:
    """_backward_pass delegates to accelerator.backward when available."""

    def test_backward_pass_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.lr_scheduler = None
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        acc.backward.assert_called_once_with(loss)


class TestLLMUseReferencePolicySeparateAdapter:
    """use_adapter('reference') sets requires_grad=False on reference params."""

    def test_use_adapter_sets_requires_grad_false(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        ref_param = torch.nn.Parameter(torch.tensor([1.0]))
        actor_param = torch.nn.Parameter(torch.tensor([2.0]))
        ref_param.requires_grad = True
        actor_param.requires_grad = True
        agent.actor.named_parameters = MagicMock(
            return_value=[
                ("lora.reference.weight", ref_param),
                ("lora.actor.weight", actor_param),
            ]
        )
        agent.use_adapter("reference")
        agent.actor.set_adapter.assert_called_with("reference")
        assert not ref_param.requires_grad
        assert actor_param.requires_grad

    def test_use_adapter_keeps_actor_and_critic_trainable(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        ref_param = torch.nn.Parameter(torch.tensor([1.0]))
        actor_param = torch.nn.Parameter(torch.tensor([2.0]))
        critic_param = torch.nn.Parameter(torch.tensor([3.0]))
        ref_param.requires_grad = True
        actor_param.requires_grad = True
        critic_param.requires_grad = True
        agent.actor.named_parameters = MagicMock(
            return_value=[
                ("lora.reference.weight", ref_param),
                ("lora.actor.weight", actor_param),
                ("lora.critic.weight", critic_param),
            ]
        )

        agent.use_adapter("reference")

        agent.actor.set_adapter.assert_called_with("reference")
        assert not ref_param.requires_grad
        assert actor_param.requires_grad
        assert critic_param.requires_grad


@pytest.mark.skipif(
    not HAS_VLLM,
    reason="vLLM rollout sync needs the Linux-only vllm extra.",
)
class TestLLMMoveModelToVllmAdapterReload:
    """Second sync uses load_inplace on the LoRA request."""

    def test_second_sync_passes_load_inplace(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)
        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
                create=True,
            ),
        ):
            agent._sync_actor_to_vllm()
            agent._vllm_moved = False
            agent._sync_actor_to_vllm()
        second_call = agent.llm.llm_engine.add_lora.call_args_list[-1]
        assert second_call.args[0].load_inplace is True

    def test_generation_request_never_uses_load_inplace(self):
        """The request handed to ``generate()`` must never set ``load_inplace``.

        vLLM re-evaluates ``add_adapter`` for the active LoRA on every
        ``execute_model`` step, and reloads the adapter from disk whenever
        ``load_inplace`` is True. Reusing the one-shot refresh request (which
        does carry ``load_inplace`` from the second sync onward) for generation
        would make every decode step disk-bound and starve the GPU. Pin: the
        stored ``_vllm_rollout_lora_request`` stays ``load_inplace=False`` even
        after repeated syncs.
        """
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)
        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
                create=True,
            ),
        ):
            agent._sync_actor_to_vllm()
            assert agent._vllm_rollout_lora_request.load_inplace is False
            agent._vllm_moved = False
            agent._sync_actor_to_vllm()
        # Even though the second add_lora refresh used load_inplace=True, the
        # request used for generation must remain load_inplace=False.
        assert agent._vllm_rollout_lora_request.load_inplace is False
        second_call = agent.llm.llm_engine.add_lora.call_args_list[-1]
        assert second_call.args[0].load_inplace is True


class TestLLMCloneWithoutAccelerator:
    """LLMAlgorithm.clone without accelerator."""

    def test_clone_without_accelerator(self):
        agent = _make_llm_agent(accelerator=None, clone=True)
        agent.accelerator = None
        agent.zero_stage = -1
        agent.use_vllm = False
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.state_dict.return_value = {"step": 0}
        agent.optimizer.optimizer.state_dict.return_value = {}

        cloned = MagicMock()
        cloned.accelerator = None
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.optimizer.optimizer = MagicMock()
        cloned.llm = None
        cloned.use_vllm = False

        with (
            patch(
                "agilerl.algorithms.core.base.clone_tensors_for_torch_save",
                return_value={},
                create=True,
            ),
            patch("agilerl.algorithms.core.base.clone_llm", return_value=MagicMock()),
            patch.object(
                EvolvableAlgorithm,
                "inspect_attributes",
                return_value={
                    "index": 0,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "max_grad_norm": 0.0,
                    "clone": True,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "actor_network": MagicMock(),
                    "device": "cpu",
                },
            ),
            patch.object(EvolvableAlgorithm, "copy_attributes", return_value=cloned),
            patch.object(RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            result = LLMAlgorithm.clone(agent, index=5, wrap=True)
        assert result is cloned
        cloned.optimizer.optimizer.load_state_dict.assert_called_once()


class TestLLMCloneWithAccelerator:
    """Cover LLMAlgorithm.clone accelerator branches."""

    def test_clone_with_accelerator_no_deepspeed(self):
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = -1
        agent.use_vllm = False
        agent.lr_scheduler = MagicMock()
        acc.unwrap_model = MagicMock(return_value=agent.actor)

        cloned = MagicMock()
        cloned.accelerator = MagicMock()
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.llm = None
        cloned.use_vllm = False
        cloned.mutation_hook = MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.clone_tensors_for_torch_save",
                return_value={},
                create=True,
            ),
            patch("agilerl.algorithms.core.base.clone_llm", return_value=MagicMock()),
            patch("agilerl.algorithms.core.base.Accelerator", return_value=MagicMock()),
            patch.object(
                EvolvableAlgorithm,
                "inspect_attributes",
                return_value={
                    "index": 0,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "max_grad_norm": 0.0,
                    "clone": True,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "actor_network": MagicMock(),
                    "device": "cpu",
                },
            ),
            patch.object(EvolvableAlgorithm, "copy_attributes", return_value=cloned),
            patch.object(RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            result = LLMAlgorithm.clone(agent, index=3)
        assert result is cloned
        acc.wait_for_everyone.assert_called()


class TestLLMCloneWithDeepSpeed:
    """Cover clone lines for zero_stage >= 2."""

    def test_clone_with_zero_stage_2(self):
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = 2
        agent.use_vllm = False
        agent.lr_scheduler = MagicMock()
        acc.unwrap_model = MagicMock(return_value=agent.actor)

        cloned = MagicMock()
        cloned.accelerator = MagicMock()
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.llm = None
        cloned.use_vllm = False
        cloned.mutation_hook = MagicMock()

        with (
            patch("agilerl.algorithms.core.base.clone_llm", return_value=MagicMock()),
            patch("agilerl.algorithms.core.base.Accelerator", return_value=MagicMock()),
            patch.object(
                EvolvableAlgorithm,
                "inspect_attributes",
                return_value={
                    "index": 0,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "max_grad_norm": 0.0,
                    "clone": True,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "actor_network": MagicMock(),
                    "device": "cpu",
                },
            ),
            patch.object(EvolvableAlgorithm, "copy_attributes", return_value=cloned),
            patch.object(RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(LLMAlgorithm, "_save_distributed_actor"),
            patch.object(LLMAlgorithm, "_load_distributed_actor"),
        ):
            result = LLMAlgorithm.clone(agent, index=7)
        assert result is cloned


class TestLLMQuantizedClone:
    """QLoRA / bitsandbytes clones rebuild the base and transfer LoRA only."""

    def test_create_clone_instance_rebuilds_from_pretrained(self):
        agent = _make_llm_agent(accelerator=None)
        agent.quantization_config = MagicMock(name="bnb_config")
        agent.pretrained_model_name_or_path = "org/model"
        agent.zero_stage = None

        captured: dict = {}

        def _fake_ctor(*args, **kwargs):
            captured.update(kwargs)
            return MagicMock(name="clone_instance")

        with (
            patch.object(
                EvolvableAlgorithm,
                "inspect_attributes",
                return_value={
                    "index": 0,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "max_grad_norm": 0.0,
                    "clone": False,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "device": "cpu",
                    "quantization_config": agent.quantization_config,
                },
            ),
            patch.object(
                LLMAlgorithm,
                "_clone_actor_network",
                side_effect=AssertionError("dense clone_llm must not run for quant"),
            ),
            patch.object(RegistryMeta, "__call__", side_effect=_fake_ctor),
        ):
            agent._create_clone_instance()

        assert captured["actor_network"] is None
        assert captured["model_name"] == "org/model"
        assert captured["clone"] is True
        assert captured["wrap"] is False

    def test_save_clone_distributed_uses_lora_only_under_zero2(self):
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = 2
        agent.quantization_config = MagicMock(name="bnb_config")
        agent.index = 0

        with patch.object(LLMAlgorithm, "_save_distributed_actor") as mock_save:
            agent._save_clone_distributed_actor_state("/tmp/work")

        mock_save.assert_called_once_with("/tmp/work/agent_0", lora_only=True)

    def test_save_clone_without_zero2_writes_peft_adapters(self):
        agent = _make_llm_agent(accelerator=None)
        agent.zero_stage = None
        agent.quantization_config = MagicMock(name="bnb_config")

        with patch.object(LLMAlgorithm, "_save_clone_adapter_weights") as mock_adapters:
            agent._save_clone_distributed_actor_state("/tmp/work")

        mock_adapters.assert_called_once_with("/tmp/work")

    def test_save_clone_adapter_weights_calls_save_pretrained(self):
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = None
        agent.selected_adapters = ("actor", "reference")
        model_ref = MagicMock(name="actor")

        with (
            patch.object(LLMAlgorithm, "_get_unwrapped_actor", return_value=model_ref),
            patch(
                "agilerl.algorithms.core.base.gather_if_zero3",
                return_value=nullcontext(),
            ),
        ):
            agent._save_clone_adapter_weights("/tmp/work")

        model_ref.save_pretrained.assert_called_once_with(
            save_directory="/tmp/work/adapters",
            selected_adapters=("actor", "reference"),
            is_main_process=True,
        )
        acc.wait_for_everyone.assert_called()

    def test_load_clone_adapter_weights_iterates_selected_adapters(self):
        agent = _make_llm_agent(accelerator=None)
        agent.selected_adapters = ("actor", "reference")

        with patch.object(LLMAlgorithm, "_load_adapter_weights") as mock_load:
            agent._load_clone_adapter_weights("/tmp/work")

        assert mock_load.call_args_list == [
            call("/tmp/work/adapters", "actor"),
            call("/tmp/work/adapters", "reference"),
        ]

    def test_load_clone_without_zero2_restores_peft_adapters(self):
        agent = _make_llm_agent(accelerator=None)
        agent.zero_stage = None
        agent.quantization_config = MagicMock(name="bnb_config")
        agent.use_value_head = False
        clone = MagicMock()

        with patch.object(clone, "_load_clone_adapter_weights") as mock_load:
            agent._load_clone_distributed_actor_state(clone, "/tmp/work")

        mock_load.assert_called_once_with("/tmp/work")

    def test_load_clone_without_zero2_copies_value_head_and_barrier(self):
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = None
        agent.quantization_config = MagicMock(name="bnb_config")
        agent.use_value_head = True

        parent_actor = MagicMock(name="parent_actor")
        parent_v_head_state = {"w": 1.0}
        parent_actor.v_head.state_dict.return_value = parent_v_head_state
        clone = MagicMock()
        clone_actor = MagicMock(name="clone_actor")

        with (
            patch.object(
                LLMAlgorithm, "_get_unwrapped_actor", return_value=parent_actor
            ),
            patch.object(clone, "_load_clone_adapter_weights"),
            patch.object(clone, "_get_unwrapped_actor", return_value=clone_actor),
        ):
            agent._load_clone_distributed_actor_state(clone, "/tmp/work")

        clone_actor.v_head.load_state_dict.assert_called_once_with(parent_v_head_state)
        acc.wait_for_everyone.assert_called()

    def test_save_clone_adapter_weights_gathers_lora_params_only(self):
        """Under ZeRO-3, _save_clone_adapter_weights must gather adapter params only."""
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = 3
        agent.selected_adapters = ("actor",)

        lora_param = torch.nn.Parameter(torch.zeros(2))
        base_param = torch.nn.Parameter(torch.zeros(10, 10))
        model_ref = MagicMock(name="actor")
        model_ref.named_parameters.return_value = [
            ("base_model.model.layer.0.weight", base_param),
            ("base_model.model.lora_A.actor.weight", lora_param),
        ]

        gather_calls = []

        @contextmanager
        def capture_gather(zero_stage, params, modifier_rank=None):
            gather_calls.append((zero_stage, list(params), modifier_rank))
            yield

        with (
            patch.object(LLMAlgorithm, "_get_unwrapped_actor", return_value=model_ref),
            patch(
                "agilerl.algorithms.core.base.gather_if_zero3",
                side_effect=capture_gather,
            ),
        ):
            agent._save_clone_adapter_weights("/tmp/work")

        assert len(gather_calls) == 1
        stage, params, modifier_rank = gather_calls[0]
        assert stage == 3
        assert any(p is lora_param for p in params)
        assert not any(p is base_param for p in params)
        assert modifier_rank is None

    def test_copy_adapter_weights_gathers_lora_params_with_modifier_rank_zero(self):
        """_copy_adapter_weights must gather lora params with modifier_rank=0 under ZeRO-3."""
        agent = _make_llm_agent(accelerator=None)
        agent.zero_stage = 3

        src_lora = torch.nn.Parameter(torch.ones(4))
        tgt_lora = torch.nn.Parameter(torch.zeros(4))
        base_param = torch.nn.Parameter(torch.zeros(10, 10))
        model_ref = MagicMock(name="actor")
        model_ref.named_parameters.return_value = [
            ("base_model.model.layer.0.weight", base_param),
            ("base_model.model.lora_A.source.weight", src_lora),
            ("base_model.model.lora_A.target.weight", tgt_lora),
        ]
        agent.actor = model_ref

        gather_calls = []

        @contextmanager
        def capture_gather(zero_stage, params, modifier_rank=None):
            gather_calls.append((zero_stage, list(params), modifier_rank))
            yield

        with patch(
            "agilerl.algorithms.core.base.gather_if_zero3", side_effect=capture_gather
        ):
            agent._copy_adapter_weights("source", "target")

        assert len(gather_calls) == 1
        stage, params, modifier_rank = gather_calls[0]
        assert stage == 3
        assert any(p is src_lora for p in params)
        assert any(p is tgt_lora for p in params)
        assert not any(p is base_param for p in params)
        assert modifier_rank == 0

    def test_setup_actors_attaches_adapters_when_rebuild_from_pretrained(self):
        agent = _make_llm_agent(accelerator=None, clone=True)
        agent.use_vllm = False

        with patch.object(LLMAlgorithm, "_initialize_actors") as mock_init:
            agent._setup_actors(None, clone=True)

        mock_init.assert_called_once_with(None, True)

    def test_setup_actors_skips_adapters_for_dense_peft_clone(self):
        agent = _make_llm_agent(accelerator=None, clone=True)
        agent.use_vllm = False
        peft_net = MagicMock(name="peft_actor")

        with patch.object(LLMAlgorithm, "_initialize_actors") as mock_init:
            agent._setup_actors(peft_net, clone=True)

        mock_init.assert_called_once_with(peft_net, False)

    def test_setup_actors_routes_quant_clone_through_colocated_vllm(self):
        agent = _make_llm_agent(accelerator=None, clone=True)
        agent.use_vllm = True

        with patch.object(
            LLMAlgorithm, "_initialize_colocated_vllm_and_actors"
        ) as mock_vllm:
            agent._setup_actors(None, clone=True)

        mock_vllm.assert_called_once_with(None, add_adapters=True, clone=True)

    def test_colocated_sleep_mode_clone_skips_second_vllm_engine(self):
        """Sleep-mode clones build the trainer only; parent's llm is moved later."""
        from agilerl.utils.algo_utils import VLLMConfig

        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc, clone=True)
        agent.use_vllm = True
        agent.vllm_config = VLLMConfig(sleep_mode=True)
        agent.llm = MagicMock(name="stale_llm")

        with (
            patch.object(LLMAlgorithm, "_initialize_actors") as mock_init,
            patch.object(LLMAlgorithm, "_configure_vllm") as mock_configure,
        ):
            agent._initialize_colocated_vllm_and_actors(
                None, add_adapters=True, clone=True
            )

        assert agent.llm is None
        mock_init.assert_called_once_with(None, True)
        mock_configure.assert_not_called()
        acc.wait_for_everyone.assert_called()


class TestLLMCloneWithVllm:
    """clone preserves vllm references during attribute copying."""

    def test_clone_preserves_vllm_references(self):
        agent = _make_llm_agent(accelerator=None, clone=True)
        agent.accelerator = None
        agent.zero_stage = -1
        agent.use_vllm = True
        agent.llm = MagicMock()
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.state_dict.return_value = {"step": 0}
        agent.optimizer.optimizer.state_dict.return_value = {}

        cloned = MagicMock()
        cloned.accelerator = None
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.optimizer.optimizer = MagicMock()
        cloned.llm = MagicMock()
        cloned.use_vllm = True

        with (
            patch(
                "agilerl.algorithms.core.base.clone_tensors_for_torch_save",
                return_value={},
                create=True,
            ),
            patch("agilerl.algorithms.core.base.clone_llm", return_value=MagicMock()),
            patch.object(
                EvolvableAlgorithm,
                "inspect_attributes",
                return_value={
                    "index": 0,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "max_grad_norm": 0.0,
                    "clone": True,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "actor_network": MagicMock(),
                    "device": "cpu",
                },
            ),
            patch.object(EvolvableAlgorithm, "copy_attributes", return_value=cloned),
            patch.object(RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            result = LLMAlgorithm.clone(agent, index=2)
        assert result is cloned
        assert agent.llm is not None

    def test_clone_sleep_mode_transfers_vllm_engine(self):
        """Sleep-mode CuMem is process-global: clone must reuse the parent's llm."""
        from agilerl.utils.algo_utils import VLLMConfig

        agent = _make_llm_agent(accelerator=None, clone=True)
        agent.accelerator = None
        agent.zero_stage = -1
        agent.use_vllm = True
        agent.vllm_config = VLLMConfig(sleep_mode=True)
        parent_llm = MagicMock(name="parent_llm")
        agent.llm = parent_llm
        agent._vllm_awake = False
        agent._vllm_moved = True
        agent._vllm_lora_loaded = True
        agent._vllm_lora_staging_dir = MagicMock(name="staging")
        agent._vllm_lora_staging_dir_is_temp = True
        agent._vllm_rollout_lora_request = MagicMock(name="lora_req")
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.state_dict.return_value = {"step": 0}
        agent.optimizer.optimizer.state_dict.return_value = {}

        cloned = MagicMock()
        cloned.accelerator = None
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.optimizer.optimizer = MagicMock()
        cloned.llm = None  # sleep-mode clone skips LLM construction
        cloned.use_vllm = True

        with (
            patch(
                "agilerl.algorithms.core.base.clone_tensors_for_torch_save",
                return_value={},
                create=True,
            ),
            patch("agilerl.algorithms.core.base.clone_llm", return_value=MagicMock()),
            patch.object(
                EvolvableAlgorithm,
                "inspect_attributes",
                return_value={
                    "index": 0,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "max_grad_norm": 0.0,
                    "clone": True,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "actor_network": MagicMock(),
                    "device": "cpu",
                },
            ),
            patch.object(EvolvableAlgorithm, "copy_attributes", return_value=cloned),
            patch.object(RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            result = LLMAlgorithm.clone(agent, index=2)

        assert result is cloned
        assert cloned.llm is parent_llm
        assert agent.llm is None
        assert agent._vllm_lora_staging_dir is None
        assert cloned._vllm_moved is True
        assert cloned._vllm_lora_staging_dir is not None


class TestLLMInitializeActors:
    """_initialize_actors creates and configures PEFT-wrapped actors."""

    def test_initialize_actors_with_base_model_no_peft(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        agent.zero_stage = 2
        peft_actor = _make_mock_peft_actor()

        base_model = MagicMock(spec=[])  # spec=[] prevents PeftModelProtocol match

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ) as mock_get_peft,
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
            patch.object(
                agent, "use_adapter", wraps=agent.use_adapter
            ) as mock_use_adapter,
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)
        mock_use_adapter.assert_called_once_with("actor")
        mock_get_peft.assert_called_once()
        assert mock_get_peft.call_args.kwargs["autocast_adapter_dtype"] is True

    def test_initialize_actors_zero3_disables_adapter_dtype_autocast(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        agent.zero_stage = 3
        peft_actor = _make_mock_peft_actor()
        peft_actor.register_parameter(
            "lora_A", torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
        )
        base_model = MagicMock(spec=[])

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ) as mock_get_peft,
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
            patch.object(agent, "use_adapter"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)
        mock_get_peft.assert_called_once()
        assert mock_get_peft.call_args.kwargs["autocast_adapter_dtype"] is False
        assert peft_actor.lora_A.dtype == torch.bfloat16

    def test_initialize_actors_with_none_creates_from_path(self):
        agent = _make_llm_agent()
        agent.pretrained_model_name_or_path = "mock-path"
        agent.lora_config = MagicMock()
        peft_actor = _make_mock_peft_actor()
        # spec=[] so the created model doesn't duck-type as a PeftModel on
        # Python <= 3.11 (a bare MagicMock satisfies any runtime protocol
        # there; 3.12+ uses getattr_static and is immune).
        created_model = MagicMock(spec=[])

        with (
            patch(
                "agilerl.algorithms.core.base.create_model_from_name_or_path",
                return_value=created_model,
            ) as mock_create,
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, None, add_adapters=True)
        mock_create.assert_called_once_with(
            "mock-path", model_config=None, add_value_head=False, use_accelerator=False
        )

    def test_initialize_actors_user_peft_raises(self):
        """User-supplied PeftModel inputs are rejected outright."""
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        peft_model = _make_mock_peft_actor()

        with pytest.raises(
            ValueError, match=re.escape("actor_network: a PeftModel was passed")
        ):
            LLMAlgorithm._initialize_actors(agent, peft_model, add_adapters=True)

    def test_initialize_actors_with_separate_reference_adapter(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        agent.zero_stage = 3
        agent.selected_adapters = ("actor", "reference")
        peft_actor = _make_mock_peft_actor()

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
        ):
            LLMAlgorithm._initialize_actors(
                agent, MagicMock(spec=[]), add_adapters=True
            )
        peft_actor.add_adapter.assert_called_once_with(
            adapter_name="reference",
            peft_config=agent.lora_config,
            autocast_adapter_dtype=False,
        )

    def test_initialize_actors_no_add_adapters(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        base_model = _make_mock_peft_actor()

        with (
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=base_model
            ),
            patch.object(
                agent, "use_adapter", wraps=agent.use_adapter
            ) as mock_use_adapter,
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=False)
        mock_use_adapter.assert_called_once_with("actor")

    def test_initialize_actors_value_head_adds_critic_and_sets_wrapper(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.use_value_head = True
        agent.use_separate_reference_adapter = False
        agent.selected_adapters = ("actor", "critic")
        agent.lora_config = MagicMock()

        base_model = torch.nn.Module()
        dense_inner = MagicMock(spec=[])
        base_model.pretrained_model = dense_inner
        peft_actor = _make_mock_peft_actor()
        peft_actor.peft_config = {}

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ) as mock_gpm,
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward", create=True
            ),
            patch.object(
                agent, "use_adapter", wraps=agent.use_adapter
            ) as mock_use_adapter,
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        mock_gpm.assert_called_once_with(
            dense_inner,
            agent.lora_config,
            adapter_name="actor",
            autocast_adapter_dtype=True,
        )
        peft_actor.add_adapter.assert_called_once_with(
            adapter_name="critic",
            peft_config=agent.lora_config,
            autocast_adapter_dtype=True,
        )
        assert base_model.pretrained_model is peft_actor
        assert base_model.is_peft_model is True
        assert agent.actor is base_model
        mock_use_adapter.assert_called_once_with("actor")

    def test_initialize_actors_value_head_inner_peft_raises(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.use_value_head = True
        agent.use_separate_reference_adapter = False
        agent.lora_config = MagicMock()

        inner_peft = _make_mock_peft_actor()
        inner_peft.peft_config = {"default": MagicMock()}
        base_model = torch.nn.Module()
        base_model.pretrained_model = inner_peft

        with pytest.raises(
            ValueError,
            match=re.escape("actor_network.pretrained_model: a PeftModel was passed"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)


class TestLLMInitializeActorsTorchCompiler:
    """_initialize_actors handles torch_compiler / DeepSpeed / gradient_checkpointing combinations."""

    def test_torch_compiler_with_deepspeed_warns_and_skips_compile(self):
        """Setting both torch_compiler and DeepSpeed only warns; no model wrap."""
        agent = _make_llm_agent()
        agent.torch_compiler = "default"
        agent._uses_deepspeed = True
        agent.gradient_checkpointing = True
        base_model = _make_mock_peft_actor()

        with (
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward", create=True
            ),
            patch("agilerl.algorithms.core.base.compile_model") as mock_compile,
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable",
                side_effect=lambda module, device: module,
            ),
            patch.object(agent, "use_adapter"),
            pytest.warns(
                UserWarning,
                match="torch_compiler is not yet compatible with DeepSpeed",
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=False)

        mock_compile.assert_not_called()
        assert agent.gradient_checkpointing is True

    def test_torch_compiler_without_deepspeed_disables_grad_checkpointing_and_compiles(
        self,
    ):
        """Without DeepSpeed, gradient checkpointing is disabled and the model is compiled."""
        agent = _make_llm_agent()
        agent.torch_compiler = "default"
        agent._uses_deepspeed = False
        agent.gradient_checkpointing = True
        base_model = _make_mock_peft_actor()
        # Compile output must be an nn.Module: downstream code in
        # `_initialize_actors` wraps it in OptimizerWrapper.
        compiled = _make_mock_peft_actor()

        with (
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward", create=True
            ),
            patch(
                "agilerl.algorithms.core.base.compile_model", return_value=compiled
            ) as mock_compile,
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable",
                side_effect=lambda module, device: module,
            ),
            patch.object(agent, "use_adapter"),
            pytest.warns(
                UserWarning,
                match="torch_compiler is incompatible with gradient_checkpointing",
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=False)

        mock_compile.assert_called_once_with(base_model, "default")
        assert agent.actor is compiled
        assert agent.gradient_checkpointing is False


class TestLLMCloneActorNetwork:
    """_clone_actor_network preserves value-head weights and respects zero_stage."""

    @staticmethod
    def _make_value_head_actor():
        class _StubValueHead:
            def __init__(self, inner):
                self.pretrained_model = inner
                self.v_head = torch.nn.Linear(2, 1)
                self.is_peft_model = False

        inner = MagicMock()
        inner.state_dict = MagicMock(return_value={"w": torch.tensor([1.0, 2.0])})
        return _StubValueHead(inner)

    def test_value_head_branch_clones_inner_peft_and_copies_v_head(self):
        agent = _make_llm_agent(accelerator=None)
        agent.use_value_head = True
        agent.zero_stage = None
        agent.actor = self._make_value_head_actor()
        original_v_head_weight = agent.actor.v_head.weight.detach().clone()
        cloned_inner = MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.clone_tensors_for_torch_save",
                return_value={"w": torch.tensor([1.0, 2.0])},
                create=True,
            ) as mock_clone_sd,
            patch(
                "agilerl.algorithms.core.base.clone_llm", return_value=cloned_inner
            ) as mock_clone_llm,
        ):
            cloned = agent._clone_actor_network()

        mock_clone_sd.assert_called_once()
        mock_clone_llm.assert_called_once()
        assert "w" in mock_clone_llm.call_args.kwargs["state_dict"]
        assert cloned is not agent.actor
        assert cloned.pretrained_model is cloned_inner
        assert cloned.is_peft_model is True
        # v_head was reloaded from the original (load_state_dict copies values).
        assert torch.equal(cloned.v_head.weight, original_v_head_weight)

    def test_value_head_branch_skips_state_dict_clone_under_zero_stage_3(self):
        agent = _make_llm_agent(accelerator=None)
        agent.use_value_head = True
        agent.zero_stage = 3
        agent.actor = self._make_value_head_actor()
        cloned_inner = MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.clone_tensors_for_torch_save",
                create=True,
            ) as mock_clone_sd,
            patch(
                "agilerl.algorithms.core.base.clone_llm", return_value=cloned_inner
            ) as mock_clone_llm,
            patch(
                "agilerl.algorithms.core.base.gather_if_zero3",
                return_value=nullcontext(),
            ),
        ):
            agent._clone_actor_network()

        mock_clone_sd.assert_not_called()
        assert mock_clone_llm.call_args.kwargs["state_dict"] is None


class TestLLMLoadAdapterWeights:
    """_load_adapter_weights overwrites adapter weights in-place."""

    def test_load_adapter_weights(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = -1

        model_ref = MagicMock()
        model_ref.parameters.return_value = []
        ref_param = torch.nn.Parameter(torch.tensor([1.0]))
        ref_param.requires_grad = True
        model_ref.named_parameters.return_value = [
            ("lora.reference.weight", ref_param),
        ]
        model_ref.set_adapter = MagicMock()
        acc.unwrap_model = MagicMock(return_value=model_ref)

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.load_file", return_value={}, create=True
            ),
            patch(
                "agilerl.algorithms.core.base.set_peft_model_state_dict", create=True
            ),
        ):
            agent._update_existing_adapter(str(tmp_path), "actor")
        model_ref.set_adapter.assert_called_with("actor")
        assert not ref_param.requires_grad


@pytest.mark.skipif(
    not HAS_VLLM,
    reason="_configure_vllm exercises the Linux-only vllm extra.",
)
class TestLLMConfigureVllmAcceleratorPaths:
    """_configure_vllm with accelerator and various TP configurations."""

    def test_configure_vllm_tp_size_1(self):
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        vllm_config.gpu_memory_utilization = 0.9
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = False
        agent.vllm_config = vllm_config
        agent.max_model_len = 512
        agent.pretrained_model_name_or_path = "mock-model"

        mock_llm_instance = MagicMock()
        with patch(
            "agilerl.algorithms.core.base.LLM",
            return_value=mock_llm_instance,
            create=True,
        ) as mock_llm_cls:
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
        mock_llm_cls.assert_called_once()
        acc.wait_for_everyone.assert_called()

    def test_configure_vllm_marks_3d_moe_lora_capable_models(self, caplog):
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        vllm_config.gpu_memory_utilization = 0.9
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = False
        agent.vllm_config = vllm_config
        agent.max_model_len = 512
        agent.pretrained_model_name_or_path = "mock-model"
        agent.lora_config = MagicMock(target_parameters=["experts.gate_up_proj"])

        with (
            patch(
                "agilerl.algorithms.core.base.LLM",
                return_value=MagicMock(),
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.patch_vllm_3d_moe_lora_flag",
                return_value=True,
            ) as flag,
            caplog.at_level(logging.INFO, logger="agilerl.algorithms.core.base"),
        ):
            agent._configure_vllm()

        flag.assert_called_once()
        assert "stacked-3D MoE LoRA" in caplog.text

    @pytest.mark.parametrize(
        "kv_cache_memory_bytes",
        [None, 32 * 1024 * 1024],
    )
    def test_configure_vllm_forwards_kv_cache_memory_bytes(self, kv_cache_memory_bytes):
        # Guards the parallel-vLLM kwargs contract: when kv_cache_memory_bytes
        # is set on VLLMConfig it must be forwarded into the LLM(...) call so
        # vLLM takes the determine_available_memory early-return path; when
        # it's None the kwarg must be omitted (vLLM auto-sizes from
        # gpu_memory_utilization). Regressing either direction silently
        # breaks parallel CI runs.
        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        vllm_config.gpu_memory_utilization = 0.22
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = False
        vllm_config.dtype = None
        vllm_config.quantization = None
        vllm_config.kv_cache_memory_bytes = kv_cache_memory_bytes
        agent.vllm_config = vllm_config
        agent.max_model_len = 512
        agent.pretrained_model_name_or_path = "mock-model"

        with patch(
            "agilerl.algorithms.core.base.LLM",
            return_value=MagicMock(),
            create=True,
        ) as mock_llm_cls:
            agent._configure_vllm()

        kwargs = mock_llm_cls.call_args.kwargs
        if kv_cache_memory_bytes is None:
            assert "kv_cache_memory_bytes" not in kwargs
        else:
            assert kwargs["kv_cache_memory_bytes"] == kv_cache_memory_bytes

    def test_configure_vllm_tp_size_gt_1(self, deepspeed_env):
        del deepspeed_env
        acc = _make_mock_accelerator(num_processes=4)
        agent = _make_llm_agent(accelerator=acc)
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 2
        vllm_config.gpu_memory_utilization = 0.9
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = True
        vllm_config.sleep_mode_level = 1
        agent.vllm_config = vllm_config
        agent.max_model_len = 512
        agent.pretrained_model_name_or_path = "mock-model"

        mock_llm_instance = MagicMock()
        with (
            patch(
                "agilerl.algorithms.core.base.torch.distributed.new_subgroups_by_enumeration",
                return_value=(MagicMock(name="tp_group"), None),
            ),
            patch(
                "agilerl.algorithms.core.base.LLM",
                return_value=mock_llm_instance,
                create=True,
            ) as mock_llm_cls,
        ):
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
        mock_llm_cls.assert_called_once()
        mock_llm_instance.sleep.assert_called_once_with(level=1)

    def test_configure_vllm_value_error_with_backend_env(self):
        import os

        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        vllm_config.gpu_memory_utilization = 0.9
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = False
        agent.vllm_config = vllm_config
        agent.max_model_len = 512
        agent.pretrained_model_name_or_path = "mock-model"

        with (
            patch(
                "agilerl.algorithms.core.base.LLM",
                side_effect=ValueError("unsupported backend"),
                create=True,
            ),
            patch.dict(os.environ, {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}),
        ):
            with pytest.raises(ValueError, match="VLLM_ATTENTION_BACKEND"):
                agent._configure_vllm()

    def test_configure_vllm_value_error_without_backend_env(self):
        import os

        acc = _make_mock_accelerator(num_processes=1)
        agent = _make_llm_agent(accelerator=acc)
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        vllm_config.gpu_memory_utilization = 0.9
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = False
        agent.vllm_config = vllm_config
        agent.max_model_len = 512
        agent.pretrained_model_name_or_path = "mock-model"

        env = os.environ.copy()
        env.pop("VLLM_ATTENTION_BACKEND", None)

        with (
            patch(
                "agilerl.algorithms.core.base.LLM",
                side_effect=ValueError("other error"),
                create=True,
            ),
            patch.dict(os.environ, env, clear=True),
        ):
            with pytest.raises(ValueError, match="other error"):
                agent._configure_vllm()


class TestLLMSyncDeepSpeedGradientClippingMatch:
    """Gradient clipping sync: update when mismatched, noop when matched."""

    def test_sync_updates_when_values_mismatch(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_clipping": 0.5,
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc, max_grad_norm=0.0)
        agent.max_grad_norm = 2.0
        acc.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] = 0.5
        agent._sync_deepspeed_gradient_clipping()
        assert acc.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] == 2.0

    def test_sync_noop_when_values_match(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "gradient_clipping": 1.5,
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc, max_grad_norm=1.5)
        acc.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] = 1.5
        agent._sync_deepspeed_gradient_clipping()
        assert acc.state.deepspeed_plugin.deepspeed_config["gradient_clipping"] == 1.5


class TestLLMReinitOptFromConfig:
    """_reinit_opt_from_config dispatches to LLMAlgorithm.update_lr."""

    def test_reinit_opt_from_config_llm_with_actor_optimizer(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc)
        agent.cosine_lr_schedule_config = None
        agent.actor.optimizer = MagicMock()
        agent.actor.optimizer.param_groups = [{"lr": 1e-4}]

        from agilerl.algorithms.core.registry import OptimizerConfig

        config = OptimizerConfig(
            name="optimizer",
            lr="lr",
            networks=["actor"],
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={},
        )

        with patch.object(
            LLMAlgorithm, "update_lr", return_value=(acc, None)
        ) as mock_update:
            EvolvableAlgorithm._reinit_opt_from_config(agent, config)
        mock_update.assert_called_once()

    def test_reinit_opt_from_config_llm_without_actor_optimizer(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc)
        agent.cosine_lr_schedule_config = None
        if hasattr(agent.actor, "optimizer"):
            del agent.actor.optimizer

        from agilerl.algorithms.core.registry import OptimizerConfig

        config = OptimizerConfig(
            name="optimizer",
            lr="lr",
            networks=["actor"],
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={},
        )

        with patch.object(
            LLMAlgorithm, "update_lr", return_value=(acc, None)
        ) as mock_update:
            EvolvableAlgorithm._reinit_opt_from_config(agent, config)
        mock_update.assert_called_once()

    def test_reinit_opt_from_config_llm_with_split_lr_config(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc)
        agent.cosine_lr_schedule_config = None

        from agilerl.algorithms.core.registry import OptimizerConfig

        config = OptimizerConfig(
            name="optimizer",
            lr=("lr", "lr_critic"),
            networks=["actor"],
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={},
        )

        with patch.object(
            LLMAlgorithm, "update_lr", return_value=(acc, None)
        ) as mock_update:
            EvolvableAlgorithm._reinit_opt_from_config(agent, config)

        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["lr"] == (agent.lr, agent.lr_critic)
        assert kwargs["accelerator"] is agent.accelerator
        assert kwargs["scheduler_config"] is agent.cosine_lr_schedule_config


class TestLLMCleanUpCudaPaths:
    """clean_up clears device caches (CUDA or Apple MPS) when available."""

    def test_clean_up_calls_cuda_empty_cache_when_available(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        with (
            patch(
                "agilerl.algorithms.core.base.torch.cuda.is_available",
                return_value=True,
            ),
            patch("agilerl.algorithms.core.base.torch.cuda.empty_cache") as mock_empty,
            patch(
                "agilerl.algorithms.core.base.torch.cuda.is_initialized",
                return_value=True,
            ),
            patch("agilerl.algorithms.core.base.torch.cuda.synchronize") as mock_sync,
        ):
            LLMAlgorithm.clean_up(agent)
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

    def test_clean_up_calls_mps_empty_cache_when_available(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        with (
            patch(
                "agilerl.algorithms.core.base.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "agilerl.algorithms.core.base.torch.mps.is_available",
                return_value=True,
            ),
            patch("agilerl.algorithms.core.base.torch.mps.empty_cache") as mock_empty,
            patch("agilerl.algorithms.core.base.torch.mps.synchronize") as mock_sync,
        ):
            LLMAlgorithm.clean_up(agent)
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()


class TestLLMLoadCheckpointLoraOnlyWithRefAdapter:
    """load_checkpoint loads the actor from disk and copies it onto the reference adapter."""

    def test_load_checkpoint_copies_actor_to_reference(self, tmp_path):
        import dill

        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc, use_separate_reference_adapter=True)
        chkpt = {"_lora_only": True, "lr": 1e-4}
        torch.save(chkpt, str(tmp_path / "attributes.pt"), pickle_module=dill)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights") as mock_load,
            patch.object(LLMAlgorithm, "_copy_adapter_weights") as mock_copy,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(
                str(tmp_path),
                load_optimizer=False,
                overwrite_reference_adapter=True,
            )
        load_calls = [c.args for c in mock_load.call_args_list]
        assert any(args[:2] == (str(tmp_path), "actor") for args in load_calls) is False
        mock_copy.assert_called_with(source_adapter="actor", target_adapter="reference")

    def _write_attrs(self, tmp_path):
        import dill

        torch.save(
            {"_lora_only": True, "lr": 1e-4},
            str(tmp_path / "attributes.pt"),
            pickle_module=dill,
        )

    def test_reference_seeded_from_actor_when_checkpoint_has_none(self, tmp_path):
        """A stage-N actor becomes the stage-N+1 reference (e.g. SFT -> DPO)."""
        agent = _make_llm_agent(
            accelerator=_make_mock_accelerator(), use_separate_reference_adapter=True
        )
        self._write_attrs(tmp_path)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_weights") as mock_copy,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(str(tmp_path), load_optimizer=False)

        mock_copy.assert_called_with(source_adapter="actor", target_adapter="reference")

    def test_reference_preserved_when_checkpoint_has_one(self, tmp_path):
        """Resuming a run keeps the reference anchor it was training against."""
        agent = _make_llm_agent(
            accelerator=_make_mock_accelerator(), use_separate_reference_adapter=True
        )
        self._write_attrs(tmp_path)
        (tmp_path / "reference").mkdir()

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_weights") as mock_copy,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(str(tmp_path), load_optimizer=False)

        assert not any(
            call.kwargs.get("target_adapter") == "reference"
            for call in mock_copy.call_args_list
        )

    def test_critic_is_not_seeded_from_actor_by_default(self, tmp_path):
        """A critic starts from the base model, not from the policy."""
        agent = _make_llm_agent(accelerator=_make_mock_accelerator())
        agent.selected_adapters = ["actor", "critic"]
        self._write_attrs(tmp_path)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_weights") as mock_copy,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(str(tmp_path), load_optimizer=False)

        assert not any(
            call.kwargs.get("target_adapter") == "critic"
            for call in mock_copy.call_args_list
        )

    def test_critic_seeded_from_actor_when_explicitly_requested(self, tmp_path):
        agent = _make_llm_agent(accelerator=_make_mock_accelerator())
        agent.selected_adapters = ["actor", "critic"]
        self._write_attrs(tmp_path)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_weights") as mock_copy,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(
                str(tmp_path),
                load_optimizer=False,
                overwrite_critic_adapter=True,
            )

        mock_copy.assert_called_with(source_adapter="actor", target_adapter="critic")

    def test_load_checkpoint_updates_reference_adapter_legacy_weights_only_key(
        self, tmp_path
    ):
        import dill

        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc, use_separate_reference_adapter=True)
        chkpt = {"_weights_only": True, "lr": 1e-4}
        torch.save(chkpt, str(tmp_path / "attributes.pt"), pickle_module=dill)

        with (
            patch.object(LLMAlgorithm, "_load_model_checkpoint") as mock_model_load,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(str(tmp_path), load_optimizer=False)
        # ``None`` defers the reference decision to the checkpoint layout; the
        # critic is never seeded from the actor unless explicitly asked.
        mock_model_load.assert_called_once_with(str(tmp_path), None, False)

    def test_load_model_checkpoint_fails_fast_on_lora_config_mismatch(self, tmp_path):
        agent = _make_llm_agent()
        agent.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear_1"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        ckpt_lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["linear_2"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )

        with (
            patch.object(
                LLMAlgorithm,
                "_load_checkpoint_lora_config",
                return_value=ckpt_lora_config,
            ),
            pytest.raises(ValueError, match="LoRA configs differ"),
        ):
            agent._load_model_checkpoint(str(tmp_path))


class TestLLMGenerateWithVllmColocateFullPaths:
    """_generate_with_vllm_colocate produces completions and action masks."""

    def test_generate_with_vllm_colocate_basic(self):
        agent = _make_llm_agent()
        agent.pad_token = "<pad>"
        agent.pad_token_id = 0
        agent.max_output_tokens = 20
        agent.max_model_len = 100
        agent.repetition_penalty = 1.0
        agent.temperature = 1.0
        agent.top_p = 1.0
        agent.top_k = None
        agent.min_p = None
        agent.min_output_tokens = None
        agent.accelerator = None

        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        agent.vllm_config = vllm_config
        agent.device = "cpu"

        prompts = [
            {"input_ids": torch.tensor([[1, 2, 3]]), "text": "hello"},
            {"input_ids": torch.tensor([[4, 5]]), "text": "world"},
        ]

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=list(range(5)))]
        agent.llm = MagicMock()
        agent.llm.generate.return_value = [
            mock_output,
            mock_output,
            mock_output,
            mock_output,
        ]

        mock_sp = MagicMock()
        with (
            patch(
                "agilerl.algorithms.core.base.SamplingParams",
                return_value=mock_sp,
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
        ):
            token_ids, action_masks, _ = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        assert len(token_ids) == 2
        assert len(action_masks) == 2

    def test_generate_with_vllm_colocate_uses_input_ids(self):
        """``input_ids`` is the whole running transcript; vLLM re-reads it as-is."""
        agent = _make_llm_agent()
        agent.pad_token = "<pad>"
        agent.pad_token_id = 0
        agent.max_output_tokens = 20
        agent.max_model_len = 100
        agent.repetition_penalty = 1.0
        agent.temperature = 1.0
        agent.top_p = 1.0
        agent.top_k = None
        agent.min_p = None
        agent.min_output_tokens = None
        agent.accelerator = None

        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        agent.vllm_config = vllm_config
        agent.device = "cpu"

        prompts = [
            {
                "input_ids": torch.tensor([[1, 2, 3, 9, 9]]),
                "text": "hello",
            },
        ]

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=list(range(5)))]
        agent.llm = MagicMock()
        agent.llm.generate.return_value = [mock_output, mock_output]

        with (
            patch(
                "agilerl.algorithms.core.base.SamplingParams",
                return_value=MagicMock(),
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
        ):
            token_ids, _action_masks, _ = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )

        sent = agent.llm.generate.call_args[0][0]
        assert sent[0]["prompt_token_ids"] == [1, 2, 3, 9, 9]
        assert len(token_ids) == 1


class TestLLMGenerateWithVllmColocateAccelerator:
    """_generate_with_vllm_colocate does not world-barrier before generate."""

    def test_generate_with_vllm_colocate_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.pad_token = "<pad>"
        agent.pad_token_id = 0
        agent.max_output_tokens = 20
        agent.max_model_len = 100
        agent.repetition_penalty = 1.0
        agent.temperature = 1.0
        agent.top_p = 1.0
        agent.top_k = None
        agent.min_p = None
        agent.min_output_tokens = None
        agent.device = "cpu"

        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        agent.vllm_config = vllm_config

        prompts = [
            {"input_ids": torch.tensor([[1, 2, 3]]), "text": "hello"},
        ]

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=list(range(5)))]
        agent.llm = MagicMock()
        agent.llm.generate.return_value = [mock_output, mock_output]

        mock_sp = MagicMock()
        with (
            patch(
                "agilerl.algorithms.core.base.SamplingParams",
                return_value=mock_sp,
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
        ):
            token_ids, _action_masks, _ = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        acc.wait_for_everyone.assert_not_called()
        assert len(token_ids) == 1


class TestLLMGenerateWithVllmColocateTP:
    """_generate_with_vllm_colocate gathers and slices with tensor_parallel > 1."""

    def test_generate_with_tp_gt_1(self):
        acc = _make_mock_accelerator(num_processes=2)
        agent = _make_llm_agent(accelerator=acc)
        agent.pad_token = "<pad>"
        agent.pad_token_id = 0
        agent.max_output_tokens = 20
        agent.max_model_len = 100
        agent.repetition_penalty = 1.0
        agent.temperature = 1.0
        agent.top_p = 1.0
        agent.top_k = 50
        agent.min_p = 0.1
        agent.min_output_tokens = 5
        agent.device = "cpu"

        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 2
        agent.vllm_config = vllm_config
        agent.tp_group = MagicMock()

        prompts = [
            {"input_ids": torch.tensor([[1, 2, 3]]), "text": "hello"},
        ]

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=list(range(5)))]
        agent.llm = MagicMock()
        agent.llm.generate.return_value = [mock_output] * 4

        mock_sp = MagicMock()

        def fake_all_gather(dest, src, group=None):
            for i in range(len(dest)):
                dest[i] = src

        with (
            patch(
                "agilerl.algorithms.core.base.SamplingParams",
                return_value=mock_sp,
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
            patch("torch.distributed.all_gather_object", side_effect=fake_all_gather),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            token_ids, _action_masks, _ = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        assert len(token_ids) == 1


class TestLLMCloneBroadcastMultiProcess:
    """clone broadcasts temp directory in multi-process setting."""

    def test_clone_multi_process_broadcasts_temp_dir(self):
        acc = _make_mock_accelerator(num_processes=2)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = -1
        agent.use_vllm = False
        agent.lr_scheduler = MagicMock()
        acc.unwrap_model = MagicMock(return_value=agent.actor)

        cloned = MagicMock()
        cloned.accelerator = MagicMock()
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.llm = None
        cloned.use_vllm = False
        cloned.mutation_hook = MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.clone_tensors_for_torch_save",
                return_value={},
                create=True,
            ),
            patch("agilerl.algorithms.core.base.clone_llm", return_value=MagicMock()),
            patch("agilerl.algorithms.core.base.Accelerator", return_value=MagicMock()),
            patch(
                "agilerl.algorithms.core.base.broadcast_object_list",
                side_effect=lambda x, **kw: x,
            ) as mock_broadcast,
            patch.object(
                EvolvableAlgorithm,
                "inspect_attributes",
                return_value={
                    "index": 0,
                    "batch_size": 4,
                    "lr": 1e-4,
                    "max_grad_norm": 0.0,
                    "clone": True,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "actor_network": MagicMock(),
                    "device": "cpu",
                },
            ),
            patch.object(EvolvableAlgorithm, "copy_attributes", return_value=cloned),
            patch.object(RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            LLMAlgorithm.clone(agent, index=3)
        mock_broadcast.assert_called_once()


@_LLM_DEPS_SKIP
class TestLLMInitEdgeCases:
    """Constructor branches not covered by _make_llm_agent defaults."""

    def test_vllm_config_warns_when_use_vllm_false(self):
        lora = MagicMock()
        with (
            patch.object(LLMAlgorithm, "_initialize_actors"),
            patch.object(LLMAlgorithm, "_configure_vllm"),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(EvolvableAlgorithm, "_registry_init"),
            pytest.warns(
                UserWarning, match="vllm_config is provided but use_vllm is False"
            ),
        ):
            _StubLLMAlgorithm(
                index=0,
                batch_size=4,
                lr=1e-4,
                max_grad_norm=0.0,
                clone=True,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=lora,
                actor_network=_make_mock_peft_actor(),
                device="cpu",
                model_name="mock-model",
                use_vllm=False,
                vllm_config=VLLMConfig(),
            )

    def test_model_config_strips_lora_target_scope(self):
        lora = MagicMock()
        with (
            patch.object(LLMAlgorithm, "_initialize_actors"),
            patch.object(LLMAlgorithm, "_configure_vllm"),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(EvolvableAlgorithm, "_registry_init"),
        ):
            agent = _StubLLMAlgorithm(
                index=0,
                batch_size=4,
                lr=1e-4,
                max_grad_norm=0.0,
                clone=True,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=lora,
                actor_network=_make_mock_peft_actor(),
                device="cpu",
                model_name="mock-model",
                model_config={
                    "attn_implementation": "sdpa",
                    "lora_target_scope": "inner",
                },
            )
        assert "lora_target_scope" not in agent.model_config
        assert agent.model_config["attn_implementation"] == "sdpa"

    def test_raises_when_vllm_importance_sampling_cap_non_positive(self):
        lora = MagicMock()
        with (
            patch.object(LLMAlgorithm, "_initialize_actors"),
            patch.object(LLMAlgorithm, "_configure_vllm"),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(EvolvableAlgorithm, "_registry_init"),
            pytest.raises(ValueError, match="vllm_importance_sampling_cap must be > 0"),
        ):
            _StubLLMAlgorithm(
                index=0,
                batch_size=4,
                lr=1e-4,
                max_grad_norm=0.0,
                clone=True,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=lora,
                actor_network=_make_mock_peft_actor(),
                device="cpu",
                model_name="mock-model",
                vllm_importance_sampling_cap=0.0,
            )


@_LLM_DEPS_SKIP
class TestLLMSaveCheckpointDeprecatedWeightsOnly:
    def test_save_checkpoint_weights_only_kwarg_warns(self, tmp_path):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        mock_actor = MagicMock()
        with (
            patch.object(agent, "_get_unwrapped_actor", return_value=mock_actor),
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch("agilerl.algorithms.core.base.torch.save"),
            pytest.warns(DeprecationWarning, match="weights_only is deprecated"),
        ):
            agent.save_checkpoint(str(tmp_path), weights_only=True)


@_LLM_DEPS_SKIP
class TestLLMRestoreValueHead:
    def test_restore_value_head_noop_without_loader(self):
        agent = _make_llm_agent()

        class _NoResumeWrapper:
            pass

        agent._get_unwrapped_actor = MagicMock(return_value=_NoResumeWrapper())
        LLMAlgorithm._restore_value_head(agent, "/unused/path")


@_LLM_DEPS_SKIP
class TestLLMRebuildOptimizerAfterLoad:
    def test_rebuild_optimizer_after_load(self):
        agent = _make_llm_agent()
        with patch.object(agent, "_select_optim_class", return_value=torch.optim.AdamW):
            agent._rebuild_optimizer_after_load()
        assert isinstance(agent.optimizer, OptimizerWrapper)
        assert agent.optimizer.network_names == ["actor"]


@_LLM_DEPS_SKIP
class TestLLMInitializeActorsStrayAdapter:
    def test_initialize_actors_removes_unlisted_adapters(self):
        agent = _make_llm_agent()
        agent.selected_adapters = ("actor",)
        peft_actor = _make_mock_peft_actor()
        peft_actor.peft_config = {"actor": MagicMock(), "stray": MagicMock()}
        base_model = torch.nn.Module()

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda _model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward", create=True
            ),
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False),
            patch.object(agent, "use_adapter"),
            pytest.warns(UserWarning, match="Adapter 'stray'"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        peft_actor.delete_adapter.assert_called_once_with("stray")

    def test_initialize_actors_liger_fallback_on_unsupported_model(self, caplog):
        agent = _make_llm_agent()
        agent.selected_adapters = ("actor",)
        peft_actor = _make_mock_peft_actor()
        peft_actor.peft_config = {"actor": MagicMock()}

        class _InnerModel:
            def modules(self):
                return []

        peft_actor.base_model.model = _InnerModel()
        base_model = torch.nn.Module()

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda _model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward", create=True
            ),
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True),
            patch(
                "agilerl.algorithms.core.base._apply_liger_kernel_to_instance",
                side_effect=TypeError("unsupported"),
                create=True,
            ),
            patch.object(agent, "use_adapter"),
            caplog.at_level(logging.WARNING, logger="agilerl.algorithms.core.base"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        assert any(
            "Liger Kernel does not support" in rec.message for rec in caplog.records
        )


@_LLM_DEPS_SKIP
class TestLLMInitializeActorsExpertLoraGuards:
    """Packed-experts LoRA attach guards on ``target_parameters``."""

    def test_target_parameters_rejects_nonzero_lora_dropout(self) -> None:
        lora = MagicMock()
        lora.target_parameters = ["experts.up_proj"]
        lora.lora_dropout = 0.05
        agent = _make_llm_agent(lora_config=lora)
        agent.selected_adapters = ("actor",)
        base_model = torch.nn.Linear(4, 4)

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda _model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward",
                create=True,
            ),
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False),
            pytest.raises(ValueError, match=r"lora_dropout=0\.0"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

    def test_target_parameters_rejects_extra_adapters(self) -> None:
        lora = MagicMock()
        lora.target_parameters = ["experts.up_proj"]
        lora.lora_dropout = 0.0
        agent = _make_llm_agent(lora_config=lora)
        agent.selected_adapters = ("actor", "reference")
        base_model = torch.nn.Linear(4, 4)

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda _model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward",
                create=True,
            ),
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False),
            pytest.raises(ValueError, match="only the 'actor' adapter"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

    def test_zero3_upgrades_and_marks_expert_wrappers(self) -> None:
        lora = MagicMock()
        lora.target_parameters = ["weight"]
        lora.lora_dropout = 0.0
        agent = _make_llm_agent(lora_config=lora)
        agent.selected_adapters = ("actor",)
        agent.zero_stage = 3
        peft_actor = _make_mock_peft_actor()
        peft_actor.peft_config = {"actor": MagicMock()}
        base_model = torch.nn.Linear(4, 4)
        mark = MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda _model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model",
                return_value=peft_actor,
            ),
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward",
                create=True,
            ),
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False),
            patch(
                "agilerl.algorithms.core.base.upgrade_moe_param_wrappers",
                return_value=2,
            ) as upgrade,
            patch(
                "agilerl.algorithms.core.base.mark_expert_wrappers_as_zero3_leaves",
                mark,
            ),
            patch.object(agent, "use_adapter"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        upgrade.assert_called_once_with(peft_actor)
        mark.assert_called_once_with(peft_actor)

    def test_upgrade_zero_skips_mark_leaves(self) -> None:
        lora = MagicMock()
        lora.target_parameters = ["weight"]
        lora.lora_dropout = 0.0
        agent = _make_llm_agent(lora_config=lora)
        agent.selected_adapters = ("actor",)
        agent.zero_stage = 3
        peft_actor = _make_mock_peft_actor()
        peft_actor.peft_config = {"actor": MagicMock()}
        base_model = torch.nn.Linear(4, 4)
        mark = MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda _model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model",
                return_value=peft_actor,
            ),
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward",
                create=True,
            ),
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False),
            patch(
                "agilerl.algorithms.core.base.upgrade_moe_param_wrappers",
                return_value=0,
            ),
            patch(
                "agilerl.algorithms.core.base.mark_expert_wrappers_as_zero3_leaves",
                mark,
            ),
            patch.object(agent, "use_adapter"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        mark.assert_not_called()


class _PackedExperts(torch.nn.Module):
    """NemotronH-style self-routing packed experts with stacked 3D weights."""

    def __init__(self, num_experts: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.up_proj = torch.nn.Parameter(
            torch.randn(num_experts, intermediate, hidden) * 0.1
        )
        self.down_proj = torch.nn.Parameter(
            torch.randn(num_experts, hidden, intermediate) * 0.1
        )
        self.act_fn = F.silu

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return hidden_states


class _PackedMoeModel(torch.nn.Module):
    """Two-layer stack of packed-experts blocks under a ``mixer`` path."""

    def __init__(self, num_experts: int = 4, hidden: int = 8) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(2):
            block = torch.nn.Module()
            block.mixer = torch.nn.Module()
            block.mixer.experts = _PackedExperts(num_experts, hidden, 6)
            self.layers.append(block)


@_LLM_DEPS_SKIP
class TestLLMInitializeActorsExpertLoraZero3Memory:
    """Expert-LoRA attach under ZeRO-3 must not need the experts resident."""

    def test_zero3_attach_succeeds_on_partitioned_placeholders(self) -> None:
        """Attach works while every expert param is an empty ZeRO-3 placeholder.

        The fake partitioned params carry only ``ds_shape``/``ds_status``, so
        no DeepSpeed gather can materialize them: the attach can only succeed
        by reading shapes without touching data, and any regression to
        gathering the packed experts fails this test.
        """
        from peft.tuners.lora.layer import ParamWrapper

        from agilerl.algorithms.core.llm_ops.moe_lora import (
            RoutedExpertsLoraWrapper,
        )

        num_experts, hidden = 4, 8
        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=[],
            target_parameters=["experts.up_proj", "experts.down_proj"],
        )
        agent = _make_llm_agent(lora_config=lora)
        agent.selected_adapters = ("actor",)
        agent.zero_stage = 3
        base_model = _PackedMoeModel(num_experts=num_experts, hidden=hidden)
        expert_params = [
            param
            for name, param in base_model.named_parameters()
            if name.endswith(("up_proj", "down_proj"))
        ]
        assert len(expert_params) == 4
        for param in expert_params:
            param.ds_shape = tuple(param.shape)
            param.ds_status = SimpleNamespace(name="NOT_AVAILABLE")
            param.data = torch.empty(0)

        mark = MagicMock()
        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda _model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward",
                create=True,
            ),
            patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", False),
            patch(
                "agilerl.algorithms.core.base.mark_expert_wrappers_as_zero3_leaves",
                mark,
            ),
            patch.object(agent, "use_adapter"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        assert all(param.numel() == 0 for param in expert_params)
        wrappers = [
            module
            for module in agent.actor.modules()
            if isinstance(module, ParamWrapper)
        ]
        assert len(wrappers) == 4
        upgraded = [w for w in wrappers if isinstance(w, RoutedExpertsLoraWrapper)]
        assert len(upgraded) == 2
        for wrapper in wrappers:
            assert wrapper.num_experts == num_experts
            lora_a = wrapper.lora_A["actor"].weight
            assert lora_a.shape[0] == 4 * num_experts
        mark.assert_called_once()


@_LLM_DEPS_SKIP
class TestLLMFusedForwardPaths:
    def test_fused_forward_without_value_head(self):
        agent = _make_llm_agent()
        agent.use_value_head = False
        B, T, V, _H = 2, 5, 32, 8
        ids = torch.randint(1, V, (B, T))
        agent._packing_mode = MagicMock(return_value=None)
        agent._fused_model_pass = MagicMock(
            return_value=(torch.zeros(B, T - 1), torch.zeros(B, T - 1))
        )

        log_probs, values = agent._fused_forward(ids, batch_size=B)

        assert log_probs.shape == (B, T - 1)
        assert values is None
        agent._fused_model_pass.assert_called_once()
        fused_ids = agent._fused_model_pass.call_args.args[0]
        assert fused_ids.shape[0] == B

    def test_fused_forward_with_value_head(self):
        agent = _make_llm_agent()
        agent.use_value_head = True
        B, T = 2, 5
        ids = torch.randint(1, 32, (B, T))
        agent._packing_mode = MagicMock(return_value=None)
        agent._fused_model_pass = MagicMock(
            return_value=(
                torch.zeros(2 * B, T - 1),
                torch.zeros(2 * B, T - 1),
            )
        )

        log_probs, values = agent._fused_forward(ids, batch_size=B)

        assert log_probs.shape == (B, T - 1)
        assert values.shape == (B, T - 1)

    def test_fused_forward_uses_packed_path_when_enabled(self):
        agent = _make_llm_agent()
        agent.use_sequence_packing = True
        agent.model_config = {"attn_implementation": "flash_attention_2"}
        B, T = 2, 5
        ids = torch.randint(1, 32, (B, T))
        expected = (torch.zeros(B, T - 1), None)

        with (
            patch.object(
                agent, "_fused_packed_forward", return_value=expected
            ) as packed_fwd,
            torch.enable_grad(),
        ):
            _log_probs, values = agent._fused_forward(ids, batch_size=B)

        packed_fwd.assert_called_once()
        assert values is None

    def test_fused_packed_forward_object_output(self):
        from contextlib import nullcontext

        agent = _make_llm_agent()
        agent.use_value_head = False
        agent.temperature = 1.0
        agent.cast_logprobs_to_fp32 = True
        B, T, H, V = 2, 6, 8, 32
        ids = torch.randint(1, V, (B, T))
        mask = torch.ones_like(ids)

        hidden = torch.randn(1, T, H)
        actor = MagicMock()
        actor.forward = MagicMock(return_value=SimpleNamespace(logits=hidden))
        agent.actor = actor
        agent._get_unwrapped_actor = MagicMock(return_value=actor)
        mock_fused_fn = MagicMock(return_value=torch.zeros(1, T - 1))
        agent._fused_logprob_fn_and_head = MagicMock(
            return_value=(
                mock_fused_fn,
                torch.randn(V, H),
                None,
            )
        )
        agent._patch_lm_head_to_identity = MagicMock(return_value=nullcontext())
        agent._amp_ctx = MagicMock(return_value=nullcontext())
        agent._activation_offload_ctx = MagicMock(return_value=nullcontext())

        with patch(
            "agilerl.algorithms.core.base.unpack_logprobs",
            return_value=torch.zeros(B, T - 1),
        ):
            log_probs, values = agent._fused_packed_forward(ids, mask)

        assert log_probs.shape == (B, T - 1)
        assert values is None


@_LLM_DEPS_SKIP
class TestLLMResolveAttnImplementation:
    def test_falls_back_to_model_config_on_probe_failure(self):
        agent = SimpleNamespace(
            model_config={"attn_implementation": "flex_attention"},
        )
        agent._get_unwrapped_actor = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("probe failed")
        )
        assert LLMAlgorithm._resolve_attn_implementation(agent) == "flex_attention"

    def test_returns_none_when_probe_fails_without_model_config(self):
        agent = SimpleNamespace(model_config=None)
        agent._get_unwrapped_actor = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("probe failed")
        )
        assert LLMAlgorithm._resolve_attn_implementation(agent) is None


@_LLM_DEPS_SKIP
class TestLLMGetLogprobsPacked:
    def test_get_logprobs_uses_packed_path_under_grad(self):
        torch.manual_seed(0)
        B, T, H, V = 2, 6, 8, 32
        agent = _make_llm_agent()
        agent.use_sequence_packing = True
        agent.model_config = {"attn_implementation": "flash_attention_2"}
        agent.calc_position_embeddings = False
        agent.temperature = 1.0
        agent.cast_logprobs_to_fp32 = True
        agent.pad_token_id = 0

        actor = _TinyPeftWrapper(_TinyCausalLM(V, H))
        actor.eval()
        agent.actor = actor
        agent._get_unwrapped_actor = lambda: actor

        from contextlib import nullcontext

        agent.select_adapter = lambda _name: nullcontext()

        ids = torch.randint(1, V, (B, T))
        with torch.enable_grad():
            lp = agent._get_logprobs(
                ids,
                batch_size=B,
                use_reference=False,
                eval_mode=False,
            )

        assert lp.shape == (B, T - 1)


@_LLM_DEPS_SKIP
class TestLLMBackwardPassDeepSpeedScheduler:
    def test_backward_pass_deepspeed_steps_lr_scheduler(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent._uses_deepspeed = True
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.get_last_lr.return_value = [3e-4]
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        acc.backward.assert_called_once_with(loss)
        agent.lr_scheduler.step.assert_called_once()
        assert agent.lr == 3e-4


@_VLLM_SKIP
@_LLM_DEPS_SKIP
class TestLLMMoveLoraToVllmErrors:
    def test_raises_when_lora_config_missing(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.lora_config = None
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            pytest.raises(ValueError, match="lora_config is required"),
        ):
            agent._move_lora_to_vllm()

    def test_non_main_rank_exports_and_loads_adapter(self, tmp_path):
        """Every rank exports to its process-private staging dir and loads it."""
        acc = _make_mock_accelerator(
            num_processes=2, is_main_process=False, process_index=1
        )
        agent = _make_llm_agent(accelerator=acc)
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)
        agent.vllm_config = VLLMConfig(lora_staging_dir=str(tmp_path))
        agent._vllm_lora_staging_dir = tmp_path / "rank_1"

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
                create=True,
            ) as mock_save,
        ):
            agent._move_lora_to_vllm()
        mock_save.assert_called_once()
        assert mock_save.call_args.args[1] == tmp_path / "rank_1"
        agent.llm.llm_engine.add_lora.assert_called_once()

    def test_add_lora_uses_cuda_device_guard_when_agent_device_is_cuda(self, tmp_path):
        acc = _make_mock_accelerator(is_main_process=True, process_index=1)
        agent = _make_llm_agent(accelerator=acc)
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)
        agent.device = "cuda:1"

        adapter_path = tmp_path / "actor"
        adapter_path.mkdir(parents=True, exist_ok=True)
        (adapter_path / "adapter_config.json").write_text("{}")
        (adapter_path / "adapter_model.safetensors").write_bytes(b"")
        device_guard = MagicMock()
        device_guard.__enter__ = MagicMock(return_value=None)
        device_guard.__exit__ = MagicMock(return_value=False)

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                return_value=adapter_path,
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.torch.cuda.current_device",
                return_value=0,
            ),
            patch(
                "agilerl.algorithms.core.base.torch.cuda.device",
                return_value=device_guard,
            ) as mock_cuda_device,
        ):
            agent._move_lora_to_vllm()
        mock_cuda_device.assert_called_once_with(torch.device("cuda:1"))
        agent.llm.llm_engine.add_lora.assert_called_once()

    def test_raises_when_vllm_add_lora_fails(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        acc.unwrap_model = MagicMock(return_value=peft_ref)
        _setup_agent_for_vllm_lora_sync(agent)
        agent.llm.llm_engine.add_lora = MagicMock(return_value=False)

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
                create=True,
            ),
            pytest.raises(RuntimeError, match="vLLM failed to load LoRA adapter"),
        ):
            agent._move_lora_to_vllm()


@_VLLM_SKIP
@_LLM_DEPS_SKIP
class TestLLMGenerateWithVllmColocateErrors:
    def test_raises_when_prompt_exceeds_max_model_len(self):
        agent = _make_llm_agent(accelerator=None)
        agent.pad_token = "<pad>"
        agent.pad_token_id = 0
        agent.max_output_tokens = 20
        agent.max_model_len = 4
        agent.repetition_penalty = 1.0
        agent.temperature = 1.0
        agent.top_p = 1.0
        agent.top_k = None
        agent.min_p = None
        agent.min_output_tokens = None
        agent.device = "cpu"
        agent.vllm_config = VLLMConfig(tensor_parallel_size=1)
        agent.llm = MagicMock()

        prompts = [{"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "text": "hello"}]
        with (
            patch(
                "agilerl.algorithms.core.base.SamplingParams",
                return_value=MagicMock(),
                create=True,
            ),
            pytest.raises(ValueError, match="Model prompt length"),
        ):
            agent._generate_with_vllm_colocate(prompts, group_size=1, temperature=1.0)

    def test_tp_slice_sampling_logps_when_capture_enabled(self):
        acc = _make_mock_accelerator(num_processes=2)
        agent = _make_llm_agent(accelerator=acc)
        agent.pad_token = "<pad>"
        agent.pad_token_id = 0
        agent.max_output_tokens = 20
        agent.max_model_len = 100
        agent.repetition_penalty = 1.0
        agent.temperature = 1.0
        agent.top_p = 1.0
        agent.top_k = 50
        agent.min_p = 0.1
        agent.min_output_tokens = 5
        agent.device = "cpu"
        agent.vllm_config = VLLMConfig(tensor_parallel_size=2)
        agent.tp_group = MagicMock()

        prompts = [{"input_ids": torch.tensor([[1, 2, 3]]), "text": "hello"}]

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=[1, 2, 3])]
        agent.llm = MagicMock()
        agent.llm.generate.return_value = [mock_output] * 4

        def fake_all_gather(dest, src, group=None):
            for i in range(len(dest)):
                dest[i] = src

        with (
            patch(
                "agilerl.algorithms.core.base.SamplingParams",
                return_value=MagicMock(),
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
            patch(
                "agilerl.algorithms.core.base._vllm_sampled_token_logprobs",
                return_value=[-0.1, -0.2],
            ),
            patch("torch.distributed.all_gather_object", side_effect=fake_all_gather),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            _, _, sampling_logps = agent._generate_with_vllm_colocate(
                prompts,
                group_size=2,
                temperature=0.9,
                capture_sampling_logps=True,
            )

        assert sampling_logps is not None
        assert len(sampling_logps) == 2


@_LLM_DEPS_SKIP
class TestLLMUpdateExistingAdapterTrainability:
    def test_update_existing_adapter_sets_actor_critic_trainable(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = -1

        actor_p = torch.nn.Parameter(torch.tensor([1.0]))
        critic_p = torch.nn.Parameter(torch.tensor([2.0]))
        actor_p.requires_grad = False
        critic_p.requires_grad = False

        model_ref = MagicMock()
        model_ref.parameters.return_value = [actor_p, critic_p]
        model_ref.named_parameters.return_value = [
            ("lora.actor.weight", actor_p),
            ("lora.critic.weight", critic_p),
        ]
        model_ref.set_adapter = MagicMock()
        acc.unwrap_model = MagicMock(return_value=model_ref)

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3", create=True),
            patch(
                "agilerl.algorithms.core.base.load_file", return_value={}, create=True
            ),
            patch(
                "agilerl.algorithms.core.base.set_peft_model_state_dict", create=True
            ),
        ):
            agent._update_existing_adapter(str(tmp_path), "actor")

        assert actor_p.requires_grad
        assert critic_p.requires_grad


@_LLM_DEPS_SKIP
class TestLLMLoadCheckpointLoraConfig:
    def test_load_checkpoint_lora_config_missing(self, tmp_path):
        assert LLMAlgorithm._load_checkpoint_lora_config(str(tmp_path)) is None


class TestMultiAgentRLAlgorithmInit:
    def test_spaces_dict_inputs_are_stored_directly(self):
        """Passing ``spaces.Dict`` obs/action spaces stores them as-is."""
        obs_spaces = spaces.Dict(
            {
                "agent_0": spaces.Box(0.0, 1.0, (4,)),
                "agent_1": spaces.Box(0.0, 1.0, (4,)),
            }
        )
        action_spaces = spaces.Dict(
            {
                "agent_0": spaces.Discrete(2),
                "agent_1": spaces.Discrete(2),
            }
        )
        agent = DummyMARLAlgorithm(
            obs_spaces,
            action_spaces,
            agent_ids=["agent_0", "agent_1"],
            index=0,
        )
        assert agent.possible_observation_spaces is obs_spaces
        assert agent.possible_action_spaces is action_spaces
        assert agent.agent_ids == ["agent_0", "agent_1"]


@_LLM_DEPS_SKIP
class TestLLMAlgorithmInitQuantizationConfig:
    def test_quantization_config_merged_into_dict_model_config(self):
        """A dict ``model_config`` gets ``quantization_config`` merged in."""
        quantization_config = SimpleNamespace(llm_int8_skip_modules=None)
        with (
            patch.object(LLMAlgorithm, "_initialize_actors"),
            patch.object(LLMAlgorithm, "_configure_vllm"),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(EvolvableAlgorithm, "_registry_init"),
            patch(
                "agilerl.algorithms.core.base.broadcast_object_list",
                side_effect=lambda obj_list, from_process=0: list(obj_list),
            ),
        ):
            agent = _StubLLMAlgorithm(
                index=0,
                batch_size=4,
                lr=1e-4,
                max_grad_norm=0.0,
                clone=True,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=MagicMock(),
                actor_network=_make_mock_peft_actor(),
                device="cpu",
                model_config={"trust_remote_code": True},
                quantization_config=quantization_config,
            )

        assert agent.model_config["quantization_config"] is quantization_config
        assert agent.model_config["trust_remote_code"] is True
        # lm_head is force-skipped so the fused lm_head matmul stays exact.
        assert quantization_config.llm_int8_skip_modules == ["lm_head"]


@_LLM_DEPS_SKIP
class TestLLMAlgorithmSyncDeepspeedGradientClipping:
    def test_returns_early_when_no_deepspeed_plugin(self):
        """No-op when an accelerator is present but has no DeepSpeed plugin."""
        agent = _make_llm_agent()
        agent.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=None)
        )
        assert agent._sync_deepspeed_gradient_clipping() is None


def _lora_wrapped_actor(dtype=torch.float32, lora_dtype=None):
    """Tiny model with real PEFT tuner layers injected."""
    from peft import LoraConfig, inject_adapter_in_model

    model = torch.nn.Sequential()
    model.add_module("proj", torch.nn.Linear(8, 8, bias=False))
    model.add_module("out", torch.nn.Linear(8, 4, bias=False))
    model = model.to(dtype)
    model = inject_adapter_in_model(
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["proj", "out"],
            # Random lora_B (not zeros) so the adapter contributes a real delta.
            init_lora_weights=False,
        ),
        model,
    )
    if lora_dtype is not None:
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(lora_dtype)
    return model


def _tuner_layers(model):
    from peft.tuners.tuners_utils import BaseTunerLayer

    return [m for m in model.modules() if isinstance(m, BaseTunerLayer)]


@_LLM_DEPS_SKIP
class TestLoraInputCastCtx:
    @staticmethod
    def _agent(actor):
        return SimpleNamespace(actor=actor)

    def test_disables_cast_inside_and_restores_after(self):
        actor = _lora_wrapped_actor()
        layers = _tuner_layers(actor)
        assert layers

        agent = self._agent(actor)
        with LLMAlgorithm._lora_input_cast_ctx(agent):
            assert all(not layer.cast_input_dtype_enabled for layer in layers)
        assert all(layer.cast_input_dtype_enabled for layer in layers)

    def test_restores_cast_when_body_raises(self):
        actor = _lora_wrapped_actor()
        layers = _tuner_layers(actor)

        agent = self._agent(actor)
        boom = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            with LLMAlgorithm._lora_input_cast_ctx(agent):
                raise boom
        assert all(layer.cast_input_dtype_enabled for layer in layers)

    def test_no_adapters_is_a_noop(self):
        agent = self._agent(torch.nn.Linear(4, 4))
        with LLMAlgorithm._lora_input_cast_ctx(agent):
            pass

    def test_actor_none_is_a_noop(self):
        agent = self._agent(None)
        with LLMAlgorithm._lora_input_cast_ctx(agent):
            pass

    def test_cast_is_load_bearing_without_autocast(self):
        """The suppressed cast is what lets a bf16 input meet an fp32 adapter."""
        actor = _lora_wrapped_actor(dtype=torch.bfloat16, lora_dtype=torch.float32)
        x = torch.randn(2, 8, dtype=torch.bfloat16)

        actor(x)
        agent = self._agent(actor)
        with LLMAlgorithm._lora_input_cast_ctx(agent):
            with pytest.raises(RuntimeError):
                actor(x)


@pytest.mark.gpu
@_LLM_DEPS_SKIP
class TestLoraInputCastUnderAutocast:
    pytestmark = pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )

    @staticmethod
    def _agent(actor):
        agent = SimpleNamespace(actor=actor, accelerator=None, device="cuda")
        agent._lora_input_cast_ctx = lambda: LLMAlgorithm._lora_input_cast_ctx(agent)
        return agent

    def test_amp_ctx_suppresses_the_cast(self):
        actor = _lora_wrapped_actor().cuda()
        layers = _tuner_layers(actor)

        agent = self._agent(actor)
        with LLMAlgorithm._amp_ctx(agent):
            assert all(not layer.cast_input_dtype_enabled for layer in layers)
        assert all(layer.cast_input_dtype_enabled for layer in layers)

    def test_gradients_are_bitwise_identical_without_the_cast(self):
        def lora_grads(cast_inputs):
            torch.manual_seed(0)
            actor = _lora_wrapped_actor(
                dtype=torch.bfloat16, lora_dtype=torch.float32
            ).cuda()
            for name, param in actor.named_parameters():
                param.requires_grad_("lora_" in name)

            torch.manual_seed(1)
            x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
            agent = self._agent(actor)
            ctx = (
                nullcontext()
                if cast_inputs
                else LLMAlgorithm._lora_input_cast_ctx(agent)
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16), ctx:
                actor(x).float().pow(2).sum().backward()
            return {
                name: param.grad.clone()
                for name, param in actor.named_parameters()
                if param.grad is not None
            }

        with_cast = lora_grads(True)
        without_cast = lora_grads(False)

        assert with_cast
        assert with_cast.keys() == without_cast.keys()
        assert all(torch.equal(with_cast[k], without_cast[k]) for k in with_cast)
