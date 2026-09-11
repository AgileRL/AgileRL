# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl.algorithms.core.base module.

For LLMAlgorithm.save_checkpoint / load_checkpoint, the following cases are
considered exhaustively. The same on-disk format is written for plain, DDP
and FSDP2 runs, so the grid is

    lora_only:       True / False
    save_optimizer:  True / False
x {save, load}.

Expected behaviour per cell (the spec this file enforces):

SAVE
    lora_only=T, save_optim=T  →  peft save (adapter dirs) + optim in attributes.pt
    lora_only=F, save_optim=T  →  actor state_dict + optim in attributes.pt
    lora_only=T, save_optim=F  →  peft save (adapter dirs only)
    lora_only=F, save_optim=F  →  actor state_dict in attributes.pt, no optim

LOAD
    LoRA=T, Optim=T  →  load peft adapters + optim state from attributes.pt
    LoRA=F, Optim=T  →  load actor state_dict + optim state from attributes.pt
    LoRA=T, Optim=F  →  load peft adapters
    LoRA=F, Optim=F  →  load actor state_dict
    (Optim=T against a save_optim=F checkpoint warns and keeps the fresh
    optimizer.)

Test organisation:
  * ``grpo_factory`` — function-scoped tiny GRPO build.
  * ``llm_simple_checkpoint`` / ``llm_simple_checkpoint_load`` — parametrised
    over the 4 cells. Each cell runs ``save_checkpoint`` once and tests read
    from the resulting artefacts.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import shutil
import warnings
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import dill
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from gymnasium import spaces
from torch import nn, optim

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import DQN, IPPO, PPO
from agilerl.algorithms.core import base as core_base
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
from agilerl.distributed import FSDPConfig
from agilerl.distributed.runtime import FSDPRuntime
from agilerl.modules import EvolvableMLP
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.mutation_utils import target_activations
from agilerl.wrappers.agent import RSNorm
from tests.helper_functions import capture_grama_snapshot
from tests.test_algorithms.test_base import DummyMARLAlgorithm, DummyRLAlgorithm
from tests.test_algorithms.test_llms.test_grpo import create_module

pytest.importorskip("peft", reason="LLM checkpoint tests require peft.")
pytest.importorskip("transformers", reason="LLM checkpoint tests require transformers.")

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from peft import LoraConfig

_LLM_DEPS_SKIP = pytest.mark.skipif(
    not HAS_LLM_DEPENDENCIES,
    reason="LLM dependencies not installed",
)


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

    def test_checkpoint_dict_includes_agilerl_version(self, dummy_agent):
        chkpt = get_checkpoint_dict(dummy_agent)
        assert "agilerl_version" in chkpt

    def test_checkpoint_dict_omit_actor_info_pops_actor(self, vector_space):
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
        stub = SimpleNamespace()
        with (
            patch("agilerl.algorithms.core.base.get_world_size", return_value=2),
            patch(
                "agilerl.algorithms.core.base.allreduce_minmax_int",
                return_value=(0, 0),
            ) as reduce_call,
        ):
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

    def test_inspect_attributes_exclude_drops_named_attributes(self, dummy_agent):
        attrs = EvolvableAlgorithm.inspect_attributes(
            dummy_agent, exclude=("index", "device")
        )
        assert "index" not in attrs
        assert "device" not in attrs

    def test_inspect_attributes_exclude_is_additive_not_a_default(self, dummy_agent):
        excluded = EvolvableAlgorithm.inspect_attributes(
            dummy_agent, exclude=("index",)
        )
        default = EvolvableAlgorithm.inspect_attributes(dummy_agent)
        assert "index" not in excluded
        assert "index" in default


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

    def test_clone_wrap_false(self, dummy_agent):
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


class _MockPeftActor(torch.nn.Module):
    """A torch.nn.Module subclass that quacks like a PeftModel."""

    def __init__(self):
        super().__init__()
        self._dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
        self.name_or_path = "mock-model"
        self.peft_config = {}
        self.base_model = MagicMock()
        causal = MagicMock()
        causal.lm_head = MagicMock()
        causal.get_output_embeddings = lambda: causal.lm_head
        causal.set_output_embeddings = lambda new: setattr(causal, "lm_head", new)
        self.base_model.model = causal
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
        self.config = SimpleNamespace(_attn_implementation=None)

    def get_base_model(self):
        return self.base_model.model

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

    def learn(self, *a, **kw):
        return None

    def get_action(self, *a, **kw):
        return None

    def test(self, *a, **kw):
        return None


def _make_llm_agent(
    clone=True,
    micro_batch_size_per_gpu=None,
    cosine_lr_schedule_config=None,
    max_grad_norm=0.0,
    use_liger_loss=False,
    lora_config=None,
    actor_network=None,
    batch_size=4,
    use_separate_reference_adapter=False,
    gradient_accumulation_steps=None,
    fsdp_config=None,
    *,
    mini_batch_size=None,
    group_size=1,
    algo_cls=None,
    reduce_memory_peak: bool = False,
    use_vllm: bool = False,
    use_memory_efficient_params: bool = True,
):
    """Helper to create a _StubLLMAlgorithm with heavily mocked internals.

    The stub is single-device (``agent.distributed`` reflects the live
    ``torch.distributed`` state, normally ``False`` in unit tests). Tests
    that need distributed behaviour either patch the
    ``agilerl.algorithms.core.base`` helpers (``get_world_size`` /
    ``init_distributed`` / ...) or use the ``dist_mode_factory`` fixture
    for a real world-size-1 process group.
    """
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
            group_size=group_size,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            device="cpu",
            use_separate_reference_adapter=use_separate_reference_adapter,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fsdp_config=fsdp_config,
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


class TestLLMAlgorithmLoad:
    def test_load_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not supported"):
            _StubLLMAlgorithm.load("/some/path")


class TestLLMAlgorithmRecompile:
    def test_recompile_noop_without_compiler(self):
        agent = _make_llm_agent()
        agent.recompile()
        assert agent.torch_compiler is None

    def test_recompile_skips_compile_when_distributed(self):
        """Compilation is skipped for distributed runs."""
        agent = _make_llm_agent()
        agent.distributed = True
        agent.torch_compiler = "default"
        with patch("agilerl.algorithms.core.base.compile_model") as mock_compile:
            agent.recompile()
            mock_compile.assert_not_called()

    def test_recompile_calls_compile_when_plain_and_compiler_set(self):
        agent = _make_llm_agent()
        agent.torch_compiler = "default"
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


class TestLLMDistributedValidation:
    def test_fsdp_config_without_distributed_raises(self):
        """FSDP2 sharding needs an initialised process group; reject otherwise."""
        with pytest.raises(
            ValueError, match="fsdp_config requires distributed training"
        ):
            _make_llm_agent(fsdp_config=FSDPConfig())

    def test_fsdp_config_accepted_when_distributed(self):
        with patch("agilerl.algorithms.core.base.init_distributed", return_value=True):
            agent = _make_llm_agent(fsdp_config=FSDPConfig())
        assert agent.distributed is True
        assert isinstance(agent.fsdp_config, FSDPConfig)


class TestLLMUpdateLr:
    def test_update_lr_with_scheduler_config_builds_scheduler(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        sched_config = MagicMock()
        sched_config.warmup_steps = 10
        sched_config.total_steps = 100
        with patch(
            "agilerl.algorithms.core.base.create_warmup_cosine_scheduler"
        ) as mock_sched:
            mock_sched.return_value = MagicMock()
            scheduler = LLMAlgorithm.update_lr(opt, 5e-4, scheduler_config=sched_config)
        assert scheduler is not None
        mock_sched.assert_called_once()
        assert opt.param_groups[0]["lr"] == 5e-4

    def test_update_lr_without_scheduler_config_returns_none(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        scheduler = LLMAlgorithm.update_lr(opt, 5e-4)
        assert scheduler is None
        assert opt.param_groups[0]["lr"] == 5e-4

    def test_update_lr_accepts_actor_critic_lr_tuple(self):
        actor_param = torch.tensor([1.0], requires_grad=True)
        critic_param = torch.tensor([1.0], requires_grad=True)
        opt = torch.optim.Adam(
            [
                {"params": [actor_param], "lr": 1e-3, "group": "actor"},
                {"params": [critic_param], "lr": 2e-3, "group": "critic"},
            ]
        )

        LLMAlgorithm.update_lr(opt, lr=(3e-4, 4e-4))

        assert opt.param_groups[0]["lr"] == 3e-4
        assert opt.param_groups[1]["lr"] == 4e-4


class TestLLMWrapModels:
    def test_wrap_models_applies_fsdp2_when_configured(self):
        """FSDP2 materializes shards from CPU state, then rebuilds the optimizer."""
        with patch("agilerl.algorithms.core.base.init_distributed", return_value=True):
            agent = _make_llm_agent(fsdp_config=FSDPConfig())
        agent.gradient_checkpointing = True
        original_actor = agent.actor
        with patch(
            "agilerl.distributed.runtime.materialize_fsdp2_from_cpu_state",
            side_effect=lambda model, device, config, **_k: model,
        ) as mock_fsdp:
            LLMAlgorithm.wrap_models(agent)
        mock_fsdp.assert_called_once_with(
            original_actor,
            agent.device,
            agent.fsdp_config,
            gradient_checkpointing=True,
        )
        original_actor.gradient_checkpointing_enable.assert_not_called()

    def test_wrap_models_plain(self):
        agent = _make_llm_agent()
        agent.gradient_checkpointing = False
        LLMAlgorithm.wrap_models(agent)
        assert agent.actor is not None

    def test_wrap_models_plain_with_checkpointing(self):
        agent = _make_llm_agent()
        agent.gradient_checkpointing = True
        agent.zero_stage = None
        original_actor = agent.actor
        LLMAlgorithm.wrap_models(agent)
        original_actor.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )


class TestLLMCleanUp:
    def test_clean_up_synchronises_processes(self):
        agent = _make_llm_agent()
        with patch("agilerl.algorithms.core.base.barrier") as mock_barrier:
            LLMAlgorithm.clean_up(agent)
        mock_barrier.assert_called_once()

    def test_clean_up_clears_attributes(self):
        agent = _make_llm_agent()
        LLMAlgorithm.clean_up(agent)
        assert agent.actor is None
        assert agent.optimizer is None
        assert agent.lr_scheduler is None

    def test_clean_up_deletes_vllm(self):
        agent = _make_llm_agent()
        agent.llm = MagicMock()
        agent.llm.llm_engine = MagicMock()
        LLMAlgorithm.clean_up(agent)
        assert not hasattr(agent, "llm") or agent.llm is None


class TestLLMBackwardPass:
    def test_backward_pass_steps_optimizer(self):
        agent = _make_llm_agent()
        agent.max_grad_norm = 1.0
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        loss.backward.assert_called_once()
        agent.optimizer.step.assert_called_once()
        agent.optimizer.zero_grad.assert_called_once()

    def test_backward_pass_with_lr_scheduler(self):
        agent = _make_llm_agent()
        agent.max_grad_norm = 1.0
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.get_last_lr.return_value = [5e-5]
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        agent.lr_scheduler.step.assert_called_once()
        assert agent.lr == 5e-5

    def test_backward_pass_accumulates_gradients(self):
        """Optimizer only steps at the gradient-accumulation boundary."""
        agent = _make_llm_agent(batch_size=4, micro_batch_size_per_gpu=2)
        assert agent.gradient_accumulation_steps == 2
        agent.max_grad_norm = None
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        agent.optimizer.step.assert_not_called()
        LLMAlgorithm._backward_pass(agent, loss)
        agent.optimizer.step.assert_called_once()
        agent.optimizer.zero_grad.assert_called_once()


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

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
    """Stand-in PeftModel: ``get_base_model()`` returns the CausalLM."""

    def __init__(self, inner: _TinyCausalLM) -> None:
        super().__init__()
        self.base_model = torch.nn.Module()
        self.base_model.model = inner
        self.config = SimpleNamespace(_attn_implementation=None)

    def get_base_model(self) -> _TinyCausalLM:
        return self.base_model.model

    def forward(self, **kwargs: object) -> SimpleNamespace:
        return self.base_model.model(**kwargs)


class TestFusedLinearLogprobsIntegration:
    """End-to-end: the (unconditional) fused-linear-logprob path in
    ``_get_logprobs`` produces logprobs numerically equivalent to a reference
    computed from full ``(B, T, V)`` logits on the same model under
    ``torch.no_grad()``. Exercises ``_get_lm_head``,
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
            full_logits = actor(input_ids=ids).logits / agent.temperature
            lp_ref = (
                F.log_softmax(full_logits[:, :-1].float(), dim=-1)
                .gather(dim=-1, index=ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .to(full_logits.dtype)
            )

        assert lp_fused.shape == (B, T - 1)
        assert lp_ref.shape == (B, T - 1)
        assert torch.allclose(lp_fused, lp_ref, rtol=1e-5, atol=1e-5)
        assert isinstance(
            actor.get_base_model().get_output_embeddings(), torch.nn.Linear
        )

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

        # ``_fused_model_pass`` calls ``set_fused_adapter_routing`` — stub it
        # since the tiny test actor isn't a real PEFT model.
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
        with (
            patch("agilerl.algorithms.core.base.get_world_size", return_value=3),
            pytest.raises(ValueError, match="divisible by the data-parallel size"),
        ):
            _make_llm_agent(clone=False)

    def test_unset_mini_and_micro_uses_per_rank_batch(self):
        with patch("agilerl.algorithms.core.base.get_world_size", return_value=2):
            agent = _make_llm_agent(clone=False, batch_size=4)

        assert agent.batch_size_per_process == 2
        assert agent.mini_batch_size == 2
        assert agent.micro_batch_size_per_gpu == 2
        assert agent.gradient_accumulation_steps == 1

    def test_unset_mini_uses_per_rank_collect_times_group_size(self):
        with patch("agilerl.algorithms.core.base.get_world_size", return_value=2):
            agent = _make_llm_agent(clone=False, batch_size=4, group_size=5)

        assert agent.batch_size == 4
        assert agent.batch_size_per_process == 10
        assert agent.mini_batch_size == 10
        assert agent.micro_batch_size_per_gpu == 10
        assert agent.gradient_accumulation_steps == 1

    def test_group_size_scales_pinned_micro_accumulation(self):
        agent = _make_llm_agent(
            clone=False,
            batch_size=4,
            group_size=5,
            micro_batch_size_per_gpu=2,
        )

        assert agent.mini_batch_size == 20
        assert agent.micro_batch_size_per_gpu == 2
        assert agent.gradient_accumulation_steps == 10

    def test_micro_batch_explicit(self):
        agent = _make_llm_agent(clone=False, micro_batch_size_per_gpu=2)
        assert agent.micro_batch_size_per_gpu == 2
        assert agent.mini_batch_size == 4
        assert agent.gradient_accumulation_steps == 2

    def test_unset_micro_follows_mini(self):
        agent = _make_llm_agent(clone=False, batch_size=8, mini_batch_size=4)

        assert agent.mini_batch_size == 4
        assert agent.micro_batch_size_per_gpu == 4
        assert agent.gradient_accumulation_steps == 1

    def test_mini_and_micro_derive_accumulation(self):
        agent = _make_llm_agent(
            clone=False,
            batch_size=8,
            mini_batch_size=4,
            micro_batch_size_per_gpu=2,
        )
        assert agent.gradient_accumulation_steps == 2

    def test_passed_gradient_accumulation_steps_is_ignored(self):
        with pytest.warns(
            DeprecationWarning,
            match="gradient_accumulation_steps is ignored",
        ):
            agent = _make_llm_agent(
                clone=False,
                batch_size=4,
                gradient_accumulation_steps=3,
            )

        assert agent.mini_batch_size == 4
        assert agent.micro_batch_size_per_gpu == 4
        assert agent.gradient_accumulation_steps == 1

    def test_micro_batch_explicit_not_divisible_raises(self):
        with pytest.raises(ValueError, match="must be divisible by"):
            _make_llm_agent(clone=False, micro_batch_size_per_gpu=3)

    def test_auto_micro_batch_zero_raises(self):
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
                device="cpu",
                micro_batch_size_per_gpu=0,
            )

    def test_mini_not_divisible_by_micro_raises(self):
        with pytest.raises(
            ValueError,
            match=r"mini_batch_size \(4\) must be divisible by micro_batch_size_per_gpu \(3\)",
        ):
            _make_llm_agent(
                clone=False,
                batch_size=4,
                mini_batch_size=4,
                micro_batch_size_per_gpu=3,
            )

    def test_per_process_not_divisible_by_mini_raises(self):
        with pytest.raises(
            ValueError,
            match=r"batch_size_per_process \(4\) must be divisible by mini_batch_size \(3\)",
        ):
            _make_llm_agent(clone=False, batch_size=4, mini_batch_size=3)

    def test_group_size_below_one_raises(self):
        with pytest.raises(ValueError, match="group_size must be a positive integer"):
            _make_llm_agent(clone=False, group_size=0)


@_LLM_DEPS_SKIP
class TestLLMInitWarnings:
    def test_cosine_lr_schedule_config_is_kept(self):
        sched = MagicMock()
        agent = _make_llm_agent(cosine_lr_schedule_config=sched)
        assert agent.cosine_lr_schedule_config is sched

    def test_reduce_memory_peak_deprecated_warns(self):
        with pytest.warns(DeprecationWarning, match="reduce_memory_peak is deprecated"):
            _make_llm_agent(reduce_memory_peak=True)

    def test_lr_string_coerced_to_float(self):
        """YAML loaders may supply lr as a string; the constructor coerces it."""
        agent = _make_llm_agent()
        assert isinstance(agent.lr, float)
        assert agent.lr == 0.0001

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

    def test_max_grad_norm_kept_as_attribute(self):
        agent = _make_llm_agent(max_grad_norm=1.5)
        assert agent.max_grad_norm == 1.5


class TestLLMGetLmHead:
    def _peft_actor(self):
        inner = _TinyCausalLM(vocab_size=8, hidden_size=4)
        return _TinyPeftWrapper(inner), inner.get_output_embeddings()

    def test_returns_output_embeddings(self):
        agent = _make_llm_agent()
        actor, head = self._peft_actor()
        agent.actor = actor
        agent.use_value_head = False

        result = agent._get_lm_head()

        assert result is head

    def test_uses_pretrained_model_when_value_head(self):
        agent = _make_llm_agent()
        agent.use_value_head = True
        actor, head = self._peft_actor()
        agent.actor = SimpleNamespace(pretrained_model=actor)

        result = agent._get_lm_head()

        assert result is head

    def test_raises_when_no_output_embeddings(self):
        agent = _make_llm_agent()
        actor, _ = self._peft_actor()
        actor.get_base_model().get_output_embeddings = lambda: None
        agent.actor = actor

        with pytest.raises(AttributeError, match="no output embeddings"):
            agent._get_lm_head()

    def test_patch_replaces_head_with_identity_then_restores(self):
        agent = _make_llm_agent()
        actor, original = self._peft_actor()
        agent.actor = actor
        agent.use_value_head = False

        with agent._patch_lm_head_to_identity() as yielded:
            assert isinstance(agent._get_lm_head(), torch.nn.Identity)
            assert yielded is original

        assert agent._get_lm_head() is original

    def test_patch_restores_head_on_exception(self):
        agent = _make_llm_agent()
        actor, original = self._peft_actor()
        agent.actor = actor
        agent.use_value_head = False

        msg = "boom"
        with pytest.raises(RuntimeError, match=msg):
            with agent._patch_lm_head_to_identity():
                raise RuntimeError(msg)

        assert agent._get_lm_head() is original

    def test_patch_raises_when_no_output_embeddings(self):
        agent = _make_llm_agent()
        actor, _ = self._peft_actor()
        actor.get_base_model().get_output_embeddings = lambda: None
        agent.actor = actor

        with pytest.raises(AttributeError, match="no output embeddings"):
            with agent._patch_lm_head_to_identity():
                pass

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

    def test_liger_head_gather_yields_lm_head_tensors(self):
        agent = _make_llm_agent()
        weight = torch.randn(4, 2)
        lm_head = MagicMock()
        lm_head.weight = weight
        lm_head.bias = None
        agent._get_lm_head = MagicMock(return_value=lm_head)

        with agent._liger_head_gather() as (w, b):
            assert w is weight
            assert b is None

    def test_resolve_fused_chunk_rows_uses_ds_shape(self):
        weight = torch.empty(0)
        weight.ds_shape = (49152, 2048)
        vocab = getattr(weight, "ds_shape", weight.shape)[0]
        rows = LLMAlgorithm._resolve_fused_chunk_rows(vocab, None)
        assert rows > 0
        assert vocab == 49152


class TestLLMConfigureVllm:
    def test_raises_when_vllm_not_installed(self):
        agent = _make_llm_agent()
        with patch("agilerl.algorithms.core.base.LLM", None, create=True):
            with pytest.raises(ImportError, match="vLLM is required"):
                agent._configure_vllm()

    def test_uses_default_config_when_none(self):
        agent = _make_llm_agent()
        agent.vllm_config = None
        mock_llm_cls = MagicMock()
        with (
            patch.dict(os.environ, {}),  # _configure_vllm writes rendezvous vars
            patch("agilerl.algorithms.core.base.LLM", mock_llm_cls, create=True),
            pytest.warns(UserWarning, match="No VLLM config"),
        ):
            agent._configure_vllm()
        assert isinstance(agent.vllm_config, VLLMConfig)
        mock_llm_cls.assert_called_once()

    def test_raises_when_tp_size_invalid(self):
        agent = _make_llm_agent(batch_size=12)
        agent.vllm_config = MagicMock()
        agent.vllm_config.tensor_parallel_size = 2
        with (
            patch("agilerl.algorithms.core.base.get_world_size", return_value=3),
            patch("agilerl.algorithms.core.base.LLM", MagicMock(), create=True),
        ):
            with pytest.raises(ValueError, match="Tensor parallel size"):
                agent._configure_vllm()


class TestLLMSetReferencePolicy:
    def test_set_reference_with_separate_adapter(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
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
        with patch.object(LLMAlgorithm, "_copy_adapter_tensors") as mock_copy:
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
        with patch.object(LLMAlgorithm, "_copy_adapter_tensors") as mock_copy:
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
    # state_dicts can include enum-like/custom metadata objects.
    return repr(value)


def generate_tiny_grpo() -> GRPO:
    """Build a tiny CPU GRPO agent with (actor, reference) adapters."""
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
    """Expensive PEFT-wrapped GRPO, built once per test."""
    tiny_grpo = generate_tiny_grpo()
    yield tiny_grpo
    tiny_grpo.clean_up()


# --------------------------------------------------------------------------- #
# SAVE — plain torch/peft path (single device, no process group)              #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_simple_checkpoint(request, grpo_factory, tmp_path_factory):
    """One saved plain-path checkpoint per cell, shared across all tests that
    only *read* the output.
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

    def test_llm_simple_checkpoint_save_no_engine_tag_dir_on_plain_path(
        self, llm_simple_checkpoint
    ):
        # Legacy engine backends wrote to a tag subdirectory; the unified
        # native format must not.
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
# SAVE/LOAD — distributed path (real process group; identical on-disk format) #
# --------------------------------------------------------------------------- #


@pytest.fixture(params=SAVE_LOAD_OPTIONS)
def llm_distributed_checkpoint(request, dist_mode_factory, tmp_path):
    """Save per cell under a real (world-size-1) process group; same unified
    format as the plain path.
    """
    lora_only, save_optimizer = request.param
    dist_mode_factory("dist")
    agent = generate_tiny_grpo()
    assert agent.distributed
    agent.save_checkpoint(
        str(tmp_path),
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    yield SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )
    agent.clean_up()


@pytest.mark.gpu
class TestLLMDistributedCheckpointSaveLoad:
    """The distributed path writes the SAME artifacts as the plain path:
    adapter dirs when lora_only, actor/optimizer state inside attributes.pt,
    and never an engine tag directory.
    """

    def test_distributed_save_attributes_pt_always_written(
        self, llm_distributed_checkpoint
    ):
        assert (llm_distributed_checkpoint.path / "attributes.pt").exists()

    def test_distributed_save_writes_no_engine_tag_dir(
        self, llm_distributed_checkpoint
    ):
        assert not (llm_distributed_checkpoint.path / "save_checkpoint").exists()

    def test_distributed_save_adapter_dirs_present_if_lora_only(
        self, llm_distributed_checkpoint
    ):
        actor_adapter = (
            llm_distributed_checkpoint.path / "actor" / "adapter_model.safetensors"
        )
        assert actor_adapter.exists() == llm_distributed_checkpoint.lora_only

    def test_distributed_save_attributes_pt_contents_match_cell(
        self, llm_distributed_checkpoint
    ):
        ck = load_attributes_checkpoint(llm_distributed_checkpoint.path)
        ni = ck.get("network_info")
        assert ck.get("_lora_only") == llm_distributed_checkpoint.lora_only
        has_actor_sd = "actor_state_dict" in ni["modules"]
        assert has_actor_sd == (not llm_distributed_checkpoint.lora_only)
        has_optim = bool(ni["optimizers"])
        assert has_optim == llm_distributed_checkpoint.save_optimizer

    def test_distributed_load_roundtrip(self, llm_distributed_checkpoint):
        s = llm_distributed_checkpoint
        if s.save_optimizer:
            s.agent.load_checkpoint(str(s.path), load_optimizer=True)
        else:
            with pytest.warns(UserWarning, match="Optimizer state not found"):
                s.agent.load_checkpoint(str(s.path), load_optimizer=True)


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
    """LLMAlgorithm.clone requires full model infrastructure (real model
    weights etc.), so we test it indirectly via `_configure_batch_size`
    with `clone=True` to verify the clone-mode branch.
    """

    def test_clone_mode_skips_batch_config(self):
        agent = _make_llm_agent(clone=True)
        assert agent.batch_size_per_process == 4


class TestLLMInitMiscPaths:
    def test_use_liger_loss_modifies_lora_config(self):
        lora = MagicMock()
        with patch("agilerl.algorithms.core.base.HAS_LIGER_KERNEL", True):
            with pytest.warns(UserWarning, match="Liger Loss"):
                _make_llm_agent(use_liger_loss=True, lora_config=lora)
        assert lora.exclude_modules == ["lm_head"]

    def test_seed_broadcast_with_multi_process(self):
        with (
            patch("agilerl.algorithms.core.base.init_distributed", return_value=True),
            patch("agilerl.algorithms.core.base.get_world_size", return_value=2),
            patch(
                "agilerl.algorithms.core.base.broadcast_object_list",
                return_value=[42],
            ) as mock_broadcast,
        ):
            _make_llm_agent()
        mock_broadcast.assert_called_once()

    def test_set_seed_called_when_distributed(self):
        with (
            patch("agilerl.algorithms.core.base.init_distributed", return_value=True),
            patch("agilerl.algorithms.core.base.set_seed") as mock_set_seed,
        ):
            _make_llm_agent()
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


def _fake_build_vllm_rollout_lora_request(
    lora_path,
    *,
    load_inplace=False,
    lora_name="actor",
    lora_int_id=1,
):
    """Stand-in for the real builder (which imports vLLM's ``LoRARequest``)."""
    return SimpleNamespace(
        lora_name=lora_name,
        lora_int_id=lora_int_id,
        lora_path=str(lora_path),
        load_inplace=load_inplace,
    )


def _setup_agent_for_vllm_lora_sync(agent, peft_ref):
    """Wire the agent for an adapter-only colocated vLLM sync.

    ``peft_ref`` is the PEFT model the adapter-only sync
    touches (it exports the LoRA delta, never base weights).
    """
    import tempfile
    from pathlib import Path

    agent.vllm_config = VLLMConfig()
    agent._vllm_lora_staging_dir = Path(tempfile.mkdtemp())
    agent._vllm_lora_staging_dir_is_temp = True
    agent._vllm_lora_loaded = False
    agent._vllm_moved = False
    agent._vllm_rollout_lora_request = None
    agent.lora_config = SimpleNamespace(target_modules=["q_proj"])
    agent.use_value_head = False
    agent.actor = peft_ref
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
        with (
            patch("agilerl.algorithms.core.base.get_world_size", return_value=2),
            patch("agilerl.algorithms.core.base.get_rank", return_value=1),
        ):
            resolved = LLMAlgorithm._ensure_vllm_lora_staging_dir(agent)
        assert resolved == target / "rank_1"
        assert resolved.is_dir()
        assert agent._vllm_lora_staging_dir_is_temp is False


class TestLLMSyncActorToVllm:
    def test_sync_actor_to_vllm_lora_path_exports_adapter_without_merge(self):
        """Adapter-only sync: set_adapter + export + add_lora, no merge_adapter."""
        p = torch.nn.Parameter(torch.tensor([1.0]))

        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [p]
        peft_ref.named_parameters.return_value = []
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.build_vllm_rollout_lora_request",
                side_effect=_fake_build_vllm_rollout_lora_request,
            ),
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

    def test_move_lora_to_vllm_waits_before_non_main_path_check(self, tmp_path):
        """Non-main ranks barrier after rank-0 export before the dir check."""
        p = torch.nn.Parameter(torch.tensor([1.0]))
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [p]
        peft_ref.named_parameters.return_value = []
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        agent._vllm_lora_staging_dir = tmp_path
        agent._vllm_lora_staging_dir_is_temp = False

        def _fake_export(*_args, **_kwargs):
            return tmp_path / "actor"

        def _barrier_materializes():
            adapter_dir = tmp_path / "actor"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            (adapter_dir / "adapter_config.json").write_text("{}")
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=False),
            patch(
                "agilerl.algorithms.core.base.barrier",
                side_effect=_barrier_materializes,
            ) as mock_barrier,
            patch(
                "agilerl.algorithms.core.base.build_vllm_rollout_lora_request",
                side_effect=_fake_build_vllm_rollout_lora_request,
            ),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_export,
            ),
        ):
            agent._move_lora_to_vllm()

        mock_barrier.assert_called_once()
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


class TestLLMBackwardPassGradClipping:
    def test_backward_pass_calls_clip_grad_norm(self):
        agent = _make_llm_agent()
        agent.max_grad_norm = 1.0
        param = torch.tensor([1.0], requires_grad=True)
        loss = (param * 2).sum()
        with patch("agilerl.algorithms.core.base.clip_grad_norm_") as mock_clip:
            LLMAlgorithm._backward_pass(agent, loss)
        mock_clip.assert_called_once()


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


class TestLLMBackwardPassDistributed:
    """Without an FSDP wrapper, data-parallel runs average the (LoRA-sized)
    trainable gradients explicitly at the step boundary.
    """

    def test_backward_pass_syncs_grads_when_distributed_without_fsdp(self):
        agent = _make_llm_agent()
        agent.distributed = True
        assert agent.fsdp_config is None
        agent.max_grad_norm = None
        agent.lr_scheduler = None
        loss = MagicMock()
        with patch("agilerl.distributed.runtime.sync_grads") as mock_sync:
            LLMAlgorithm._backward_pass(agent, loss)
        mock_sync.assert_called_once()
        loss.backward.assert_called_once()
        agent.optimizer.step.assert_called_once()

    def test_backward_pass_skips_sync_grads_with_fsdp(self):
        """FSDP2 reduces gradients itself — no manual all-reduce."""
        with patch("agilerl.algorithms.core.base.init_distributed", return_value=True):
            agent = _make_llm_agent(fsdp_config=FSDPConfig())
        agent.max_grad_norm = None
        agent.lr_scheduler = None
        agent.actor.set_requires_gradient_sync = MagicMock()
        loss = MagicMock()
        with patch("agilerl.distributed.runtime.sync_grads") as mock_sync:
            LLMAlgorithm._backward_pass(agent, loss)
        mock_sync.assert_not_called()
        agent.optimizer.step.assert_called_once()


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


class TestLLMMoveModelToVllmAdapterReload:
    """Second sync uses load_inplace on the LoRA request."""

    def test_second_sync_passes_load_inplace(self):
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        peft_ref.named_parameters.return_value = []
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.build_vllm_rollout_lora_request",
                side_effect=_fake_build_vllm_rollout_lora_request,
            ),
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
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        peft_ref.named_parameters.return_value = []
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.build_vllm_rollout_lora_request",
                side_effect=_fake_build_vllm_rollout_lora_request,
            ),
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


class TestLLMClonePlain:
    """LLMAlgorithm.clone transfers optimizer state via gathered state dicts
    (no engine-checkpoint disk roundtrip).
    """

    def test_clone_copies_optimizer_state(self):
        agent = _make_llm_agent(clone=True)
        agent.use_vllm = False
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.state_dict.return_value = {"step": 0}
        agent.optimizer.optimizer.state_dict.return_value = {}

        cloned = MagicMock()
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.optimizer.optimizer = MagicMock()
        cloned.llm = None
        cloned.use_vllm = False

        with (
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
        cloned._import_optimizer_state.assert_called_once()
        cloned.wrap_models.assert_called_once()


class TestLLMCloneSynchronisesProcesses:
    """clone ends with a barrier so all ranks finish wrapping together."""

    def test_clone_synchronises_processes(self):
        agent = _make_llm_agent()
        agent.use_vllm = False
        agent.lr_scheduler = MagicMock()

        cloned = MagicMock()
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.llm = None
        cloned.use_vllm = False
        cloned.mutation_hook = MagicMock()

        with (
            patch("agilerl.algorithms.core.base.clone_llm", return_value=MagicMock()),
            patch("agilerl.algorithms.core.base.barrier") as mock_barrier,
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
        mock_barrier.assert_called()


class TestLLMColocatedSleepModeClone:
    def test_colocated_sleep_mode_clone_skips_second_vllm_engine(self):
        """Sleep-mode clones build the trainer only; parent's llm is moved later."""
        from types import SimpleNamespace

        from agilerl.utils.algo_utils import VLLMConfig

        agent = SimpleNamespace(
            vllm_config=VLLMConfig(sleep_mode=True),
            llm=MagicMock(name="stale_llm"),
            _initialize_actors=MagicMock(),
            _configure_vllm=MagicMock(),
        )

        with patch("agilerl.algorithms.core.base.barrier"):
            LLMAlgorithm._initialize_colocated_vllm_and_actors(
                agent, None, add_adapters=True, clone=True
            )

        assert agent.llm is None
        agent._initialize_actors.assert_called_once_with(None, True)
        agent._configure_vllm.assert_not_called()


class TestLLMCloneWithVllm:
    """clone preserves vllm references during attribute copying."""

    def test_clone_preserves_vllm_references(self):
        agent = _make_llm_agent(clone=True)
        agent.use_vllm = True
        agent.llm = MagicMock()
        agent.lr_scheduler = MagicMock()
        agent.lr_scheduler.state_dict.return_value = {"step": 0}
        agent.optimizer.optimizer.state_dict.return_value = {}

        cloned = MagicMock()
        cloned.lr_scheduler = MagicMock()
        cloned.optimizer = MagicMock()
        cloned.optimizer.optimizer = MagicMock()
        cloned.llm = MagicMock()
        cloned.use_vllm = True

        with (
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


class TestLLMInitializeActors:
    """_initialize_actors creates and configures PEFT-wrapped actors."""

    def test_initialize_actors_with_base_model_no_peft(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        peft_actor = _make_mock_peft_actor()

        base_model = MagicMock(spec=[])  # spec=[] prevents PeftModelProtocol match

        with (
            patch(
                "agilerl.algorithms.core.base.adapt_lora_config_for_model",
                side_effect=lambda model, cfg, **kw: cfg,
            ),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch.object(
                agent, "use_adapter", wraps=agent.use_adapter
            ) as mock_use_adapter,
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)
        mock_use_adapter.assert_called_once_with("actor")

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
        ):
            LLMAlgorithm._initialize_actors(agent, None, add_adapters=True)
        mock_create.assert_called_once_with(
            "mock-path",
            model_config={"device_map": "cpu"},
            add_value_head=False,
            use_distributed=False,
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
        ):
            LLMAlgorithm._initialize_actors(
                agent, MagicMock(spec=[]), add_adapters=True
            )
        peft_actor.add_adapter.assert_called_once_with(
            adapter_name="reference",
            peft_config=agent.lora_config,
            autocast_adapter_dtype=True,
        )

    def test_initialize_actors_no_add_adapters(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        base_model = _make_mock_peft_actor()

        with (
            patch.object(
                agent, "use_adapter", wraps=agent.use_adapter
            ) as mock_use_adapter,
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=False)
        mock_use_adapter.assert_called_once_with("actor")

    def test_initialize_actors_value_head_adds_critic_and_sets_wrapper(self):
        agent = _make_llm_agent()
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
        agent = _make_llm_agent()
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
    """_initialize_actors handles torch_compiler / distributed / gradient_checkpointing combinations."""

    def test_torch_compiler_distributed_warns_and_skips_compile(self):
        """torch_compiler with a distributed run only warns; no compile wrap."""
        agent = _make_llm_agent()
        agent.distributed = True
        agent.torch_compiler = "default"
        agent.gradient_checkpointing = True
        base_model = _make_mock_peft_actor()

        with (
            patch(
                "agilerl.algorithms.core.base.patch_lora_for_fused_forward", create=True
            ),
            patch("agilerl.algorithms.core.base.compile_model") as mock_compile,
            patch.object(agent, "use_adapter"),
            pytest.warns(
                UserWarning,
                match="torch_compiler is not yet supported for distributed",
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=False)

        mock_compile.assert_not_called()
        assert agent.gradient_checkpointing is True

    def test_torch_compiler_plain_disables_grad_checkpointing_and_compiles(
        self,
    ):
        """On the plain path, gradient checkpointing is disabled and the model is compiled."""
        agent = _make_llm_agent()
        agent.torch_compiler = "default"
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
    """_clone_actor_network preserves value-head weights."""

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
        agent = _make_llm_agent()
        agent.use_value_head = True
        agent.fsdp_config = None
        agent.actor = self._make_value_head_actor()
        original_v_head_weight = agent.actor.v_head.weight.detach().clone()
        cloned_inner = MagicMock()

        with patch(
            "agilerl.algorithms.core.base.clone_llm", return_value=cloned_inner
        ) as mock_clone_llm:
            cloned = agent._clone_actor_network()

        mock_clone_llm.assert_called_once()
        assert "w" in mock_clone_llm.call_args.kwargs["state_dict"]
        assert cloned is not agent.actor
        assert cloned.pretrained_model is cloned_inner
        assert cloned.is_peft_model is True
        # v_head was reloaded from the original (load_state_dict copies values).
        assert torch.equal(cloned.v_head.weight, original_v_head_weight)

    def test_clone_actor_network_fsdp_uses_cpu_broadcast(self):
        agent = _make_llm_agent()
        agent.fsdp_config = FSDPConfig()
        agent.use_value_head = False
        agent.actor = MagicMock()
        state = {"w": torch.tensor([1.0])}
        cloned = MagicMock()
        with (
            patch.object(agent.shard_runtime, "export_model_state", return_value=state),
            patch.object(
                agent, "_broadcast_cpu_state_dict", return_value=state
            ) as mock_bcast,
            patch(
                "agilerl.algorithms.core.base.clone_llm", return_value=cloned
            ) as mock_clone,
        ):
            out = agent._clone_actor_network()
        mock_bcast.assert_called_once_with(state)
        mock_clone.assert_called_once()
        assert out is cloned


class TestLLMMemoryEfficientParams:
    """_memory_efficient_params context manager moves params on/off GPU when
    the actor is not FSDP-sharded.
    """

    def test_sharded_actor_warns_and_yields_without_moving_params(self):
        agent = _make_llm_agent()
        agent.shard_runtime = FSDPRuntime(FSDPConfig())

        with (
            patch("agilerl.algorithms.core.base.move_params_to_gpu") as mock_to_gpu,
            patch("agilerl.algorithms.core.base.move_params_to_cpu") as mock_to_cpu,
            pytest.warns(
                UserWarning,
                match="Memory efficient params is not compatible with FSDP2",
            ),
            agent._memory_efficient_params(),
        ):
            pass

        mock_to_gpu.assert_not_called()
        mock_to_cpu.assert_not_called()

    def test_default_path_moves_params_to_gpu_and_back_to_cpu(self):
        agent = _make_llm_agent()
        agent.device = "cpu"
        unwrapped = MagicMock()
        agent.actor = unwrapped

        with (
            patch("agilerl.algorithms.core.base.move_params_to_gpu") as mock_to_gpu,
            patch("agilerl.algorithms.core.base.move_params_to_cpu") as mock_to_cpu,
        ):
            with agent._memory_efficient_params():
                mock_to_gpu.assert_called_once_with(unwrapped, torch.device("cpu"))
                mock_to_cpu.assert_not_called()

        mock_to_cpu.assert_called_once_with(unwrapped)


class TestLLMLoadAdapterWeights:
    """_load_adapter_weights overwrites adapter weights in-place."""

    def test_load_adapter_weights(self, tmp_path):
        agent = _make_llm_agent()

        model_ref = MagicMock()
        ref_param = torch.nn.Parameter(torch.tensor([1.0]))
        ref_param.requires_grad = True
        model_ref.named_parameters.return_value = [
            ("lora.reference.weight", ref_param),
        ]
        model_ref.set_adapter = MagicMock()
        agent.actor = model_ref

        with patch.object(agent.shard_runtime, "import_adapter_tensors") as mock_load:
            agent.update_existing_adapter(str(tmp_path), "actor")

        mock_load.assert_called_once_with(model_ref, model_ref, str(tmp_path), "actor")
        model_ref.set_adapter.assert_called_with("actor")
        assert not ref_param.requires_grad

    def test_update_existing_adapter_uses_runtime_when_sharded(self, tmp_path):
        agent = _make_llm_agent()
        agent.shard_runtime = FSDPRuntime(FSDPConfig())
        model_ref = MagicMock()
        model_ref.named_parameters.return_value = []
        model_ref.set_adapter = MagicMock()
        agent.actor = model_ref

        with patch.object(agent.shard_runtime, "import_adapter_tensors") as mock_load:
            agent.update_existing_adapter(str(tmp_path), "actor")

        mock_load.assert_called_once()


class TestLLMConfigureVllmPaths:
    """_configure_vllm under various world-size / TP configurations."""

    def test_configure_vllm_tp_size_1(self):
        agent = _make_llm_agent()
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        vllm_config.gpu_memory_utilization = 0.9
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = False
        agent.vllm_config = vllm_config
        agent.max_model_len = 512
        agent.pretrained_model_name_or_path = "mock-model"

        mock_llm_instance = MagicMock()
        with (
            patch.dict(os.environ, {}),  # _configure_vllm writes rendezvous vars
            patch(
                "agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance
            ) as mock_llm_cls,
            patch("agilerl.algorithms.core.base.barrier") as mock_barrier,
        ):
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
        mock_llm_cls.assert_called_once()
        mock_barrier.assert_called()

    def test_configure_vllm_marks_3d_moe_lora_capable_models(self, caplog):
        agent = _make_llm_agent()
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
        agent = _make_llm_agent()
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

        with (
            patch.dict(os.environ, {}),
            patch(
                "agilerl.algorithms.core.base.LLM", return_value=MagicMock()
            ) as mock_llm_cls,
        ):
            agent._configure_vllm()

        kwargs = mock_llm_cls.call_args.kwargs
        if kv_cache_memory_bytes is None:
            assert "kv_cache_memory_bytes" not in kwargs
        else:
            assert kwargs["kv_cache_memory_bytes"] == kv_cache_memory_bytes

    def test_configure_vllm_tp_size_gt_1(self, distributed_env):
        del distributed_env
        agent = _make_llm_agent()
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
            patch("agilerl.algorithms.core.base.get_world_size", return_value=4),
            patch(
                "agilerl.algorithms.core.base.torch.distributed.new_subgroups_by_enumeration",
                return_value=(MagicMock(name="tp_group"), None),
            ),
            patch(
                "agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance
            ) as mock_llm_cls,
        ):
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
        mock_llm_cls.assert_called_once()
        mock_llm_instance.sleep.assert_called_once_with(level=1)

    def test_configure_vllm_value_error_with_backend_env(self):
        agent = _make_llm_agent()
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
        agent = _make_llm_agent()
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


class TestLLMReinitOptFromConfig:
    """_reinit_opt_from_config dispatches to LLMAlgorithm.update_lr with the
    wrapper's own optimizer (no engine-optimizer fallback).
    """

    def test_reinit_opt_from_config_llm(self):
        agent = _make_llm_agent()
        agent.cosine_lr_schedule_config = None

        from agilerl.algorithms.core.registry import OptimizerConfig

        config = OptimizerConfig(
            name="optimizer",
            lr="lr",
            networks=["actor"],
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={},
        )

        with patch.object(LLMAlgorithm, "update_lr", return_value=None) as mock_update:
            EvolvableAlgorithm._reinit_opt_from_config(agent, config)
        mock_update.assert_called_once()
        args, kwargs = mock_update.call_args
        passed_opt = args[0] if args else kwargs.get("optimizer")
        assert passed_opt is agent.optimizer.optimizer
        assert agent.lr_scheduler is None

    def test_reinit_opt_from_config_llm_with_split_lr_config(self):
        agent = _make_llm_agent()
        agent.cosine_lr_schedule_config = None

        from agilerl.algorithms.core.registry import OptimizerConfig

        config = OptimizerConfig(
            name="optimizer",
            lr=("lr", "lr_critic"),
            networks=["actor"],
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={},
        )

        with patch.object(LLMAlgorithm, "update_lr", return_value=None) as mock_update:
            EvolvableAlgorithm._reinit_opt_from_config(agent, config)

        mock_update.assert_called_once()
        _, kwargs = mock_update.call_args
        assert kwargs["lr"] == (agent.lr, agent.lr_critic)
        assert kwargs["scheduler_config"] is agent.cosine_lr_schedule_config


class TestLLMCleanUpCudaPaths:
    """clean_up clears device caches (CUDA or Apple MPS) when available."""

    def test_clean_up_calls_cuda_empty_cache_when_available(self):
        agent = _make_llm_agent()
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
        agent = _make_llm_agent()
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

        agent = _make_llm_agent(use_separate_reference_adapter=True)
        chkpt = {"_lora_only": True, "lr": 1e-4}
        torch.save(chkpt, str(tmp_path / "attributes.pt"), pickle_module=dill)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights") as mock_load,
            patch.object(LLMAlgorithm, "_copy_adapter_tensors") as mock_copy,
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
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        self._write_attrs(tmp_path)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_tensors") as mock_copy,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(str(tmp_path), load_optimizer=False)

        mock_copy.assert_called_with(source_adapter="actor", target_adapter="reference")

    def test_reference_preserved_when_checkpoint_has_one(self, tmp_path):
        """Resuming a run keeps the reference anchor it was training against."""
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        self._write_attrs(tmp_path)
        (tmp_path / "reference").mkdir()

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_tensors") as mock_copy,
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
        agent = _make_llm_agent()
        agent.selected_adapters = ["actor", "critic"]
        self._write_attrs(tmp_path)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_tensors") as mock_copy,
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
        agent = _make_llm_agent()
        agent.selected_adapters = ["actor", "critic"]
        self._write_attrs(tmp_path)

        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_tensors") as mock_copy,
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

        agent = _make_llm_agent(use_separate_reference_adapter=True)
        chkpt = {"_weights_only": True, "lr": 1e-4}
        torch.save(chkpt, str(tmp_path / "attributes.pt"), pickle_module=dill)

        with (
            patch.object(LLMAlgorithm, "_load_model_checkpoint") as mock_model_load,
            patch.object(
                LLMAlgorithm, "_load_checkpoint_lora_config", return_value=None
            ),
        ):
            agent.load_checkpoint(str(tmp_path), load_optimizer=False)
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
            completion_ids, action_masks, _ = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        assert len(completion_ids) == 2
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

    def test_generate_with_vllm_colocate_clamps_min_tokens_to_remaining(self):
        agent = _make_llm_agent()
        agent.pad_token = "<pad>"
        agent.pad_token_id = 0
        agent.max_output_tokens = 64
        agent.max_model_len = 20
        agent.repetition_penalty = 1.0
        agent.temperature = 1.0
        agent.top_p = 1.0
        agent.top_k = None
        agent.min_p = None
        agent.min_output_tokens = 16

        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 1
        vllm_config.presence_penalty = 0.0
        vllm_config.frequency_penalty = 0.0
        vllm_config.stop_sequences = None
        agent.vllm_config = vllm_config
        agent.device = "cpu"

        prompts = [
            {"input_ids": torch.tensor([[1, 2, 3, 4, 5]]), "text": "hello"},
        ]

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(token_ids=list(range(5)))]
        agent.llm = MagicMock()
        agent.llm.generate.return_value = [mock_output]

        captured: list[dict] = []

        def capture_sampling_params(**kwargs):
            captured.append(kwargs)
            return MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.SamplingParams",
                side_effect=capture_sampling_params,
                create=True,
            ),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(1, 5), None),
            ),
        ):
            agent._generate_with_vllm_colocate(prompts, group_size=1, temperature=0.9)

        assert captured[0]["max_tokens"] == 15
        assert captured[0]["min_tokens"] == 15


class TestLLMGenerateWithVllmColocateTP:
    """_generate_with_vllm_colocate gathers and slices with tensor_parallel > 1."""

    def test_generate_with_tp_gt_1(self):
        agent = _make_llm_agent()
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
            completion_ids, _action_masks, _ = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        assert len(completion_ids) == 1


class TestLLMShardedCheckpointBranches:
    """FSDP2-sharded optimizer-state branches and load_checkpoint guards."""

    def test_export_optimizer_state_sharded_uses_dcp(self):
        agent = _make_llm_agent()
        agent.actor = _make_mock_peft_actor()
        agent.optimizer = MagicMock()
        get_mock = MagicMock(return_value={"state": {}})
        agent.shard_runtime = FSDPRuntime(FSDPConfig())
        with (
            patch(
                "agilerl.distributed.runtime.get_optimizer_state_dict",
                get_mock,
            ),
            patch(
                "agilerl.distributed.runtime.is_fsdp_sharded",
                return_value=True,
            ),
        ):
            out = agent._export_optimizer_state()
        get_mock.assert_called_once()
        assert out == {"state": {}}
        agent.optimizer.state_dict.assert_not_called()

    def test_save_checkpoint_sharded_embeds_gathered_optimizer_state(self, tmp_path):
        grpo = generate_tiny_grpo()
        try:
            with patch.object(grpo, "_export_optimizer_state", return_value={"x": 1}):
                grpo.save_checkpoint(str(tmp_path), save_optimizer=True)
            checkpoint = torch.load(
                str(tmp_path / "attributes.pt"), weights_only=False, pickle_module=dill
            )
            assert checkpoint["network_info"]["optimizers"]["optimizer_state_dict"] == {
                "x": 1
            }
        finally:
            grpo.clean_up()

    def test_load_checkpoint_registry_mismatch_raises(self, tmp_path):
        grpo = generate_tiny_grpo()
        try:
            from agilerl.algorithms.core.registry import MutationRegistry

            torch.save(
                {"registry": MutationRegistry()},
                str(tmp_path / "attributes.pt"),
                pickle_module=dill,
            )
            with pytest.raises(
                ValueError, match="does not match the algorithm's registry"
            ):
                grpo.load_checkpoint(str(tmp_path))
        finally:
            grpo.clean_up()

    def test_load_checkpoint_without_actor_weights_raises(self, tmp_path):
        grpo = generate_tiny_grpo()
        try:
            torch.save(
                {
                    "_lora_only": False,
                    "network_info": {"modules": {}, "optimizers": {}},
                },
                str(tmp_path / "attributes.pt"),
                pickle_module=dill,
            )
            with pytest.raises(ValueError, match="does not contain actor weights"):
                grpo.load_checkpoint(str(tmp_path))
        finally:
            grpo.clean_up()


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
        grpo = generate_tiny_grpo()
        try:
            with pytest.warns(DeprecationWarning, match="weights_only is deprecated"):
                grpo.save_checkpoint(str(tmp_path), weights_only=True)
        finally:
            grpo.clean_up()


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

    def test_initialize_actors_applies_liger_without_fused_lce(self):
        agent = _make_llm_agent()
        agent.selected_adapters = ("actor",)
        peft_actor = _make_mock_peft_actor()
        peft_actor.peft_config = {"actor": MagicMock()}

        class _InnerModel(torch.nn.Module):
            pass

        peft_actor.base_model.model = _InnerModel()
        base_model = torch.nn.Module()
        apply = MagicMock()

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
                "agilerl.algorithms.core.base.register_nemotron_h_liger",
                return_value=True,
            ),
            patch(
                "agilerl.algorithms.core.base._apply_liger_kernel_to_instance",
                apply,
                create=True,
            ),
            patch.object(agent, "use_adapter"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        apply.assert_called_once()
        assert apply.call_args.kwargs["fused_linear_cross_entropy"] is False
        assert apply.call_args.kwargs["model"] is peft_actor.base_model.model

    def test_initialize_actors_installs_packed_expert_grouped_gemm(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        peft_actor = _make_mock_peft_actor()
        install = MagicMock(return_value=0)

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
            patch(
                "agilerl.algorithms.core.base.install_packed_expert_grouped_gemm",
                install,
            ),
            patch.object(agent, "use_adapter"),
        ):
            LLMAlgorithm._initialize_actors(
                agent, MagicMock(spec=[]), add_adapters=True
            )

        install.assert_called_once_with(peft_actor)


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
    def test_prefers_actor_config(self):
        agent = SimpleNamespace(
            actor=SimpleNamespace(
                config=SimpleNamespace(_attn_implementation="flash_attention_2")
            ),
            model_config={"attn_implementation": "sdpa"},
        )

        assert LLMAlgorithm._resolve_attn_implementation(agent) == "flash_attention_2"

    def test_falls_back_to_model_config_when_actor_impl_is_none(self):
        agent = SimpleNamespace(
            actor=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None)),
            model_config={"attn_implementation": "flex_attention"},
        )

        assert LLMAlgorithm._resolve_attn_implementation(agent) == "flex_attention"

    def test_returns_none_when_actor_and_model_config_have_no_impl(self):
        agent = SimpleNamespace(
            actor=SimpleNamespace(config=SimpleNamespace(_attn_implementation=None)),
            model_config=None,
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
class TestLLMMoveLoraToVllmErrors:
    """Error paths of the adapter-only colocated vLLM LoRA sync."""

    def test_raises_when_lora_config_missing(self):
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        peft_ref.named_parameters.return_value = []
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        agent.lora_config = None

        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            pytest.raises(ValueError, match="lora_config is required"),
        ):
            agent._move_lora_to_vllm()

    def test_raises_when_adapter_export_missing(self, tmp_path):
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        peft_ref.named_parameters.return_value = []
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        agent._vllm_lora_staging_dir = tmp_path

        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                return_value=tmp_path / "missing_adapter",
            ),
            pytest.raises(FileNotFoundError, match="PEFT adapter export"),
        ):
            agent._move_lora_to_vllm()

    def test_non_main_rank_exports_and_loads_adapter(self, tmp_path):
        """Every rank exports to its process-private staging dir and loads it."""
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        peft_ref.named_parameters.return_value = []
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        agent.vllm_config = VLLMConfig(lora_staging_dir=str(tmp_path))
        agent._vllm_lora_staging_dir = tmp_path / "rank_1"

        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=False),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.build_vllm_rollout_lora_request",
                side_effect=_fake_build_vllm_rollout_lora_request,
            ),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
            ) as mock_save,
        ):
            agent._move_lora_to_vllm()
        mock_save.assert_called_once()
        assert mock_save.call_args.args[1] == tmp_path / "rank_1"
        agent.llm.llm_engine.add_lora.assert_called_once()

    def test_add_lora_uses_cuda_device_guard_when_agent_device_is_cuda(self, tmp_path):
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        peft_ref.named_parameters.return_value = []
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        agent.device = "cuda:1"

        adapter_path = tmp_path / "actor"
        adapter_path.mkdir(parents=True, exist_ok=True)
        (adapter_path / "adapter_config.json").write_text("{}")
        (adapter_path / "adapter_model.safetensors").write_bytes(b"")
        device_guard = MagicMock()
        device_guard.__enter__ = MagicMock(return_value=None)
        device_guard.__exit__ = MagicMock(return_value=False)

        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                return_value=adapter_path,
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
        agent = _make_llm_agent()
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [torch.tensor([1.0])]
        peft_ref.named_parameters.return_value = []
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)
        agent.llm.llm_engine.add_lora = MagicMock(return_value=False)

        with (
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.build_vllm_rollout_lora_request",
                side_effect=_fake_build_vllm_rollout_lora_request,
            ),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                side_effect=_fake_save_peft_adapter_for_vllm_rollout,
            ),
            pytest.raises(RuntimeError, match="vLLM failed to load LoRA adapter"),
        ):
            agent._move_lora_to_vllm()


@_LLM_DEPS_SKIP
class TestLLMGenerateWithVllmColocateErrors:
    def test_raises_when_prompt_exceeds_max_model_len(self):
        agent = _make_llm_agent()
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
        with pytest.raises(ValueError, match="Model prompt length"):
            agent._generate_with_vllm_colocate(prompts, group_size=1, temperature=1.0)

    def test_tp_slice_sampling_logps_when_capture_enabled(self):
        agent = _make_llm_agent()
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
        agent = _make_llm_agent()

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
        agent.actor = model_ref

        with patch.object(agent.shard_runtime, "import_adapter_tensors"):
            agent.update_existing_adapter(str(tmp_path), "actor")

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
        agent = SimpleNamespace(actor=actor, device="cuda")
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


class TestEvolvableAlgorithmGraMaState:
    """The per-agent GraMa state ReGraMa reads at mutation time."""

    def agent(self):
        return DQN(
            spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            spaces.Discrete(2),
            device="cpu",
        )

    def test_unmeasured_by_default(self):
        agent = self.agent()

        assert agent.grama_scores is None

    def test_training_block_captures_only_when_enabled(self):
        agent = self.agent()

        agent.init_training_step()
        agent.actor(torch.rand(2, 4)).square().mean().backward()
        agent.finalize_training_step(1)

        # No hooks are registered, so capture costs nothing when off.
        assert agent.grama_scores is None

    def test_training_block_stores_a_snapshot_when_enabled(self):
        agent = self.agent()

        agent.init_training_step(capture_grama=True)
        agent.actor(torch.rand(2, 4)).square().mean().backward()
        agent.finalize_training_step(1)

        assert agent.grama_scores
        assert any(entry is not None for entry in agent.grama_scores[0])

    def test_snapshot_travels_to_a_clone(self):
        # This is what lets a child read the gradients captured while its parent
        # trained, under any selection strategy.
        agent = self.agent()
        agent.init_training_step(capture_grama=True)
        agent.actor(torch.rand(2, 4)).square().mean().backward()
        agent.finalize_training_step(1)

        clone = agent.clone(wrap=False)

        assert clone.grama_scores is not None
        assert len(clone.grama_scores) == len(agent.grama_scores)

    def test_snapshot_is_deep_copied_onto_the_clone(self):
        # Mutating the parent's snapshot must not reach the child's.
        agent = self.agent()
        agent.grama_scores = [[torch.ones(3)]]

        clone = agent.clone(wrap=False)
        agent.grama_scores[0][0].fill_(9.0)

        assert torch.equal(clone.grama_scores[0][0], torch.ones(3))

    def test_snapshot_is_kept_out_of_checkpoints(self):
        # Transient training state, recaptured every cycle.
        agent = self.agent()
        agent.grama_scores = [[torch.ones(3)]]

        checkpoint = get_checkpoint_dict(agent)

        assert "grama_scores" not in checkpoint

    def test_resume_restores_without_reporting_a_missing_attribute(
        self,
        tmp_path,
        recwarn,
    ):
        agent = self.agent()
        path = str(tmp_path / "agent.pt")
        agent.save_checkpoint(path)

        restored = DQN.load(path, device="cpu")

        assert restored.grama_scores is None
        assert not [
            warning for warning in recwarn if "grama_scores" in str(warning.message)
        ]

    def registered_hooks(self, agent):
        """Backward hooks currently attached to the agent's measured activations."""
        return sum(
            len(module._backward_hooks)
            for _network_id, network in agent.unrolled_eval_networks()
            for module in target_activations(network)
        )

    def test_reopening_a_block_does_not_stack_a_second_set_of_hooks(self):
        agent = self.agent()

        agent.init_training_step(capture_grama=True)
        after_one = self.registered_hooks(agent)
        agent.init_training_step(capture_grama=True)

        assert after_one > 0
        assert self.registered_hooks(agent) == after_one

    def test_one_finalize_clears_the_hooks_of_a_reopened_block(self):
        agent = self.agent()

        agent.init_training_step(capture_grama=True)
        agent.init_training_step(capture_grama=True)
        agent.finalize_training_step(1)

        assert self.registered_hooks(agent) == 0

    def test_wrapped_agent_stores_the_snapshot_on_the_unwrapped_algorithm(self):
        wrapper = RSNorm(self.agent())

        wrapper.init_training_step(capture_grama=True)
        wrapper.agent.actor(torch.rand(2, 4)).square().mean().backward()
        wrapper.finalize_training_step(1)

        assert wrapper.agent.grama_scores


def grama_mlp_net_config() -> dict:
    """Return a fresh MLP net config.

    Deliberately not the shared encoder_mlp_config fixture.
    """
    return {
        "latent_dim": 8,
        "min_latent_dim": 1,
        "encoder_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
        "head_config": {"hidden_size": [8, 8], "min_mlp_nodes": 1},
    }


@pytest.fixture
def dqn_agent(vector_space, discrete_space):
    return DQN(
        vector_space,
        discrete_space,
        net_config=grama_mlp_net_config(),
        device="cpu",
    )


class TestPerNeuronGrad:
    """Reduce a backward hook's grad_input to one magnitude per neuron."""

    def test_dense_gradient_averages_over_the_batch(self):
        gradient = torch.tensor([[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]])

        result = core_base._per_neuron_grad((gradient,))

        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_conv_gradient_averages_over_batch_and_spatial(self):
        # (batch=2, channels=3, 4, 4), one constant per channel.
        gradient = torch.ones(2, 3, 4, 4)
        gradient[:, 1] = -5.0
        gradient[:, 2] = 2.0

        result = core_base._per_neuron_grad((gradient,))

        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([1.0, 5.0, 2.0]))

    def test_absolute_value_is_taken_before_the_reduction(self):
        # A neuron whose gradient cancels across the batch.
        gradient = torch.tensor([[4.0], [-4.0]])

        result = core_base._per_neuron_grad((gradient,))

        # Assert 4.0, not 0.0.
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_already_per_neuron_gradient_is_returned_unreduced(self):
        gradient = torch.tensor([1.0, -2.0])

        result = core_base._per_neuron_grad((gradient,))

        assert torch.allclose(result, torch.tensor([1.0, 2.0]))

    @pytest.mark.parametrize("grad_input", [(None,), None, ()])
    def test_missing_gradient_is_unmeasured_rather_than_zero(self, grad_input):
        result = core_base._per_neuron_grad(grad_input)

        assert result is None


class TestGraMaCapture:
    """Capture per-neuron pre-activation gradients during a training block."""

    def test_snapshot_layout_matches_the_measured_layers(self, dqn_agent):
        expected = [
            len(target_activations(network))
            for _network_id, network in dqn_agent.unrolled_eval_networks()
        ]

        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))

        assert [len(entry) for entry in dqn_agent.grama_scores] == expected

    def test_captured_widths_match_the_producing_layers(self, dqn_agent):
        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))

        for entry in dqn_agent.grama_scores[0]:
            assert entry is None or entry.dim() == 1

    def test_only_the_last_minibatch_survives(self, dqn_agent):
        # The metric's expectation is taken at fixed parameters, and the reset acts
        # on the network as it stands at the end of the cycle.
        dqn_agent.init_training_step(capture_grama=True)
        dqn_agent.actor(torch.ones(4, 4) * 100.0).square().mean().backward()
        small = torch.rand(4, 4) * 1e-3
        dqn_agent.actor(small).square().mean().backward()
        dqn_agent.finalize_training_step(1)
        captured = dqn_agent.grama_scores[0][0]

        # Reproduce the second minibatch alone.
        dqn_agent.init_training_step(capture_grama=True)
        dqn_agent.actor(small).square().mean().backward()
        dqn_agent.finalize_training_step(1)

        assert torch.allclose(captured, dqn_agent.grama_scores[0][0], atol=1e-8)

    def test_hooks_are_released_when_the_block_completes(self, dqn_agent):
        activation = target_activations(dqn_agent.actor)[0]

        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))

        assert not activation._backward_hooks

    def test_hooks_left_by_an_aborted_block_are_cleared_by_the_next_one(
        self,
        dqn_agent,
    ):
        activation = target_activations(dqn_agent.actor)[0]
        dqn_agent.init_training_step(capture_grama=True)
        assert activation._backward_hooks

        dqn_agent.init_training_step(capture_grama=False)

        assert not activation._backward_hooks

    def test_repeated_captures_do_not_accumulate_hooks(self, dqn_agent):
        activation = target_activations(dqn_agent.actor)[0]

        for _ in range(3):
            capture_grama_snapshot(dqn_agent, torch.rand(4, 4))

        assert not activation._backward_hooks

    def test_layer_outside_the_loss_graph_is_stored_as_unmeasured(
        self,
        vector_space,
        discrete_space,
    ):
        # Only PPO's actor is exercised, so the critic never fires.
        agent = PPO(vector_space, discrete_space, device="cpu")
        capture_grama_snapshot(agent, torch.rand(4, 4))
        networks = [network for _network_id, network in agent.unrolled_eval_networks()]
        critic_index = networks.index(agent.critic)

        # Recorded as None, so it is skipped downstream rather than read
        # as a fully dormant layer and needlessly reset.
        assert all(entry is None for entry in agent.grama_scores[critic_index])
        assert any(entry is not None for entry in agent.grama_scores[0])

    def test_a_failing_hook_propagates(self, dqn_agent, monkeypatch):
        def explode(_grad_input):
            msg = "hook blew up"
            raise RuntimeError(msg)

        monkeypatch.setattr(core_base, "_per_neuron_grad", explode)

        with pytest.raises(RuntimeError, match="hook blew up"):
            capture_grama_snapshot(dqn_agent, torch.rand(4, 4))

    def test_a_layer_whose_gradient_reduces_to_nothing_stays_unmeasured(
        self,
        dqn_agent,
        monkeypatch,
    ):
        monkeypatch.setattr(core_base, "_per_neuron_grad", lambda _grad_input: None)

        capture_grama_snapshot(dqn_agent, torch.rand(4, 4))

        assert all(entry is None for entry in dqn_agent.grama_scores[0])

    def test_a_registration_failure_propagates(self, dqn_agent, monkeypatch):
        def explode(_agent):
            msg = "no networks here"
            raise AttributeError(msg)

        monkeypatch.setattr(EvolvableAlgorithm, "unrolled_eval_networks", explode)

        with pytest.raises(AttributeError, match="no networks here"):
            capture_grama_snapshot(dqn_agent, torch.rand(4, 4))

    def test_compiled_agent_captures(self, dqn_agent, recwarn):
        dqn_agent.torch_compiler = "default"

        dqn_agent.init_training_step(capture_grama=True)
        dqn_agent.finalize_training_step(1)

        assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)


class TestUnrolledEvalNetworks:
    """Enumerate the networks ReGraMa measures and rewrites."""

    def test_target_networks_are_excluded(self, dqn_agent):
        measured = [
            network for _network_id, network in dqn_agent.unrolled_eval_networks()
        ]

        # The frozen copy must never be scored or reset.
        assert dqn_agent.actor in measured
        assert dqn_agent.actor_target not in measured

    def test_actor_and_critic_are_both_measured(self, vector_space, discrete_space):
        agent = PPO(vector_space, discrete_space, device="cpu")

        measured = [network for _network_id, network in agent.unrolled_eval_networks()]

        assert agent.actor in measured
        assert agent.critic in measured

    def test_multi_agent_module_dicts_are_unrolled_per_sub_policy(
        self,
        ma_vector_space,
        ma_discrete_space,
    ):
        agent = IPPO(
            ma_vector_space,
            ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            device="cpu",
        )
        policy = getattr(agent, agent.registry.policy())

        result = agent.unrolled_eval_networks()

        # One entry per sub-policy, each tagged with its own key, so a
        # captured snapshot is never routed to another sub-agent's network.
        for key, sub_network in policy.items():
            assert any(
                network_id == key and network is sub_network
                for network_id, network in result
            )


class TestPolicyEvalNetworkIds:
    """Identify the policy networks whose latent other networks may borrow."""

    def test_the_policy_evaluation_network_is_reported(self, dqn_agent):
        assert dqn_agent.eval_policy_network_ids() == {id(dqn_agent.actor)}

    def test_multi_agent_policies_report_every_sub_policy(
        self,
        ma_vector_space,
        ma_discrete_space,
    ):
        # Each sub-policy owns an encoder its own critic may borrow, so all of
        # them count as policy networks.
        agent = IPPO(
            ma_vector_space,
            ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            device="cpu",
        )
        policy = getattr(agent, agent.registry.policy())

        result = agent.eval_policy_network_ids()

        assert result == {id(sub_network) for _key, sub_network in policy.items()}

    def test_an_agent_without_a_policy_group_reports_no_policy(self, dqn_agent):
        dqn_agent.registry.groups = []

        assert dqn_agent.eval_policy_network_ids() == set()

    def test_a_policy_the_agent_does_not_carry_reports_no_policy(self, dqn_agent):
        dqn_agent.actor = None

        assert dqn_agent.eval_policy_network_ids() == set()

    def test_a_single_agent_policy_is_unwrapped_before_hashing(self, dqn_agent):
        wrapped = object()
        real_actor = dqn_agent.actor
        dqn_agent.actor = wrapped
        dqn_agent.accelerator = MagicMock(spec=Accelerator)
        dqn_agent.accelerator.unwrap_model = MagicMock(return_value=real_actor)

        result = dqn_agent.eval_policy_network_ids()

        dqn_agent.accelerator.unwrap_model.assert_called_once_with(wrapped)
        assert result == {id(real_actor)}

    def test_a_plain_dict_policy_unwraps_every_member_under_accelerator(
        self,
        dqn_agent,
    ):
        real_actor = dqn_agent.actor
        wrapped = object()
        dqn_agent.actor = {"agent_0": wrapped}
        dqn_agent.accelerator = MagicMock(spec=Accelerator)
        dqn_agent.accelerator.unwrap_model = MagicMock(return_value=real_actor)

        result = dqn_agent.eval_policy_network_ids()

        dqn_agent.accelerator.unwrap_model.assert_called_once_with(wrapped)
        assert result == {id(real_actor)}

    def test_a_plain_dict_policy_without_an_accelerator_reports_raw_ids(
        self,
        dqn_agent,
    ):
        member = object()
        dqn_agent.actor = {"agent_0": member}

        assert dqn_agent.eval_policy_network_ids() == {id(member)}


class TestGraMaCaptureUnderAccelerator:
    """Capture on wrapped networks still lines up with the unwrapped ones."""

    def test_snapshot_survives_the_unwrap_before_selection(
        self,
        vector_space,
        discrete_space,
        encoder_mlp_config,
    ):
        agent = DQN(
            vector_space,
            discrete_space,
            net_config=encoder_mlp_config,
            accelerator=Accelerator(cpu=True, device_placement=False),
            device="cpu",
        )
        agent.wrap_models()
        capture_grama_snapshot(agent, torch.rand(4, 4))

        agent.unwrap_models()
        snapshot = agent.grama_scores[0]

        assert len(snapshot) == len(target_activations(agent.actor))
        assert all(entry is not None for entry in snapshot)

    def test_capture_measures_a_distributed_wrapped_network(
        self,
        gloo_process_group,
        vector_space,
        discrete_space,
        encoder_mlp_config,
    ):
        agent = DQN(
            vector_space,
            discrete_space,
            net_config=encoder_mlp_config,
            accelerator=Accelerator(cpu=True, device_placement=False),
            device="cpu",
        )
        agent.actor = nn.parallel.DistributedDataParallel(agent.actor)
        inner = agent.accelerator.unwrap_model(agent.actor)

        agent.init_training_step(capture_grama=True)
        agent.actor(torch.rand(4, 4)).square().mean().backward()
        agent.finalize_training_step(1)

        # Every measured layer of the wrapped network scored a gradient,
        # so the reset acts instead of silently degrading to Gaussian noise.
        measured = agent.grama_scores[0]
        assert len(measured) == len(target_activations(inner))
        assert all(entry is not None for entry in measured)

    def test_distributed_wrapped_module_dicts_are_still_unrolled(
        self,
        gloo_process_group,
        ma_vector_space,
        ma_discrete_space,
    ):
        agent = IPPO(
            ma_vector_space,
            ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "agent_2"],
            accelerator=Accelerator(cpu=True, device_placement=False),
            device="cpu",
        )
        policy_name = agent.registry.policy()
        policy = getattr(agent, policy_name)
        setattr(agent, policy_name, nn.parallel.DistributedDataParallel(policy))

        result = agent.unrolled_eval_networks()

        for key, sub_network in policy.items():
            assert any(
                network_id == key and network is sub_network
                for network_id, network in result
            )
