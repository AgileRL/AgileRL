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
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import dill
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import optim

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.core.base import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    _RegistryMeta,
    get_checkpoint_dict,
    get_optimizer_cls,
)
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.algorithms.grpo import GRPO
from agilerl.modules import EvolvableMLP
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.utils.distributed import FSDPConfig
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
        assert dim == (5,)


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
    gradient_accumulation_steps=1,
    fsdp_config=None,
    *,
    reduce_memory_peak: bool = False,
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
    if actor_network is None:
        actor_network = _make_mock_peft_actor()

    with (
        patch.object(LLMAlgorithm, "_initialize_actors"),
        patch.object(LLMAlgorithm, "_configure_vllm"),
        patch.object(LLMAlgorithm, "wrap_models"),
        patch.object(EvolvableAlgorithm, "_registry_init"),
    ):
        agent = _StubLLMAlgorithm(
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
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            device="cpu",
            use_separate_reference_adapter=use_separate_reference_adapter,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fsdp_config=fsdp_config,
            reduce_memory_peak=reduce_memory_peak,
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

    def test_assert_not_sharded_raises_when_sharded(self):
        agent = _make_llm_agent()
        with (
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
            pytest.raises(NotImplementedError, match="FSDP2-sharded"),
        ):
            agent._assert_not_sharded("Test operation")


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
        """FSDP2 sharding swaps params in place, so the optimizer (and LR
        scheduler) are rebuilt after ``apply_fsdp2``.
        """
        with patch("agilerl.algorithms.core.base.init_distributed", return_value=True):
            agent = _make_llm_agent(fsdp_config=FSDPConfig())
        agent.gradient_checkpointing = True
        original_actor = agent.actor
        with (
            patch(
                "agilerl.algorithms.core.base.apply_fsdp2",
                side_effect=lambda model, config: model,
            ) as mock_fsdp,
            patch.object(agent, "_rebuild_optimizer_after_load") as mock_rebuild,
        ):
            LLMAlgorithm.wrap_models(agent)
        mock_fsdp.assert_called_once_with(original_actor, agent.fsdp_config)
        mock_rebuild.assert_called_once()
        original_actor.gradient_checkpointing_enable.assert_called_once()

    def test_wrap_models_plain(self):
        agent = _make_llm_agent()
        agent.gradient_checkpointing = False
        LLMAlgorithm.wrap_models(agent)
        assert agent.actor is not None

    def test_wrap_models_plain_with_checkpointing(self):
        agent = _make_llm_agent()
        agent.gradient_checkpointing = True
        original_actor = agent.actor
        LLMAlgorithm.wrap_models(agent)
        original_actor.gradient_checkpointing_enable.assert_called_once()


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
        agent = _make_llm_agent(batch_size=4, gradient_accumulation_steps=2)
        assert agent.gradient_accumulation_steps == 2
        agent.max_grad_norm = None
        loss = MagicMock()
        LLMAlgorithm._backward_pass(agent, loss)
        agent.optimizer.step.assert_not_called()
        LLMAlgorithm._backward_pass(agent, loss)
        agent.optimizer.step.assert_called_once()
        agent.optimizer.zero_grad.assert_called_once()


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
        """fp32 reduction matches a stock ``log_softmax + gather`` over
        the materialized logits within bf16 quantisation noise.
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
        # Reference: stock log_softmax + gather over the materialized logits,
        # promoted to fp32 to match the fused kernel's reduction precision.
        logits = (hidden @ weight.t() + bias) / temperature
        ref = (
            F.log_softmax(logits.float(), dim=-1)
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
        msg = "boom"
        with pytest.raises(RuntimeError, match="boom"):
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
        with (
            patch("agilerl.algorithms.core.base.get_world_size", return_value=3),
            pytest.raises(ValueError, match="divisible by the number of processes"),
        ):
            _make_llm_agent(clone=False)

    def test_micro_batch_auto_derived(self):
        with patch("agilerl.algorithms.core.base.get_world_size", return_value=2):
            agent = _make_llm_agent(clone=False, gradient_accumulation_steps=2)
        assert agent.micro_batch_size_per_gpu == 1

    def test_micro_batch_explicit(self):
        agent = _make_llm_agent(clone=False, micro_batch_size_per_gpu=2)
        assert agent.micro_batch_size_per_gpu == 2
        assert agent.gradient_accumulation_steps == 2

    def test_micro_batch_explicit_not_divisible_raises(self):
        with pytest.raises(ValueError, match="micro_batch_size_per_gpu"):
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

    def test_batch_not_divisible_by_grad_accum_raises(self):
        with pytest.raises(ValueError, match="gradient accumulation"):
            _make_llm_agent(clone=False, gradient_accumulation_steps=3)


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

    ``peft_ref`` is the unwrapped PEFT model returned by
    ``_get_peft_model_for_vllm_sync`` — the only model the adapter-only sync
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
    agent._get_peft_model_for_vllm_sync = MagicMock(return_value=peft_ref)
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
        with patch("agilerl.algorithms.core.base.sync_grads") as mock_sync:
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
        loss = MagicMock()
        with patch("agilerl.algorithms.core.base.sync_grads") as mock_sync:
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
            patch.object(_RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            result = LLMAlgorithm.clone(agent, index=5, wrap=True)
        assert result is cloned
        cloned._load_gathered_optimizer_state_dict.assert_called_once()
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
            patch.object(_RegistryMeta, "__call__", return_value=cloned),
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
            patch.object(_RegistryMeta, "__call__", return_value=cloned),
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
            model_config=None,
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
            adapter_name="reference", peft_config=agent.lora_config
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
            dense_inner, agent.lora_config, adapter_name="actor"
        )
        peft_actor.add_adapter.assert_called_once_with(
            adapter_name="critic", peft_config=agent.lora_config
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


class TestLLMMemoryEfficientParams:
    """_memory_efficient_params context manager moves params on/off GPU when
    the actor is not FSDP-sharded.
    """

    def test_sharded_actor_warns_and_yields_without_moving_params(self):
        agent = _make_llm_agent()

        with (
            patch("agilerl.algorithms.core.base.move_params_to_gpu") as mock_to_gpu,
            patch("agilerl.algorithms.core.base.move_params_to_cpu") as mock_to_cpu,
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
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

        with (
            patch.object(agent, "_get_unwrapped_actor", return_value=unwrapped),
            patch("agilerl.algorithms.core.base.move_params_to_gpu") as mock_to_gpu,
            patch("agilerl.algorithms.core.base.move_params_to_cpu") as mock_to_cpu,
        ):
            with agent._memory_efficient_params():
                mock_to_gpu.assert_called_once_with(unwrapped, "cpu")
                mock_to_cpu.assert_not_called()

        mock_to_cpu.assert_called_once_with(unwrapped)


class TestLLMLoadAdapterWeights:
    """_load_adapter_weights overwrites adapter weights in-place."""

    def test_load_adapter_weights(self, tmp_path):
        agent = _make_llm_agent()

        model_ref = MagicMock()
        model_ref.parameters.return_value = []
        ref_param = torch.nn.Parameter(torch.tensor([1.0]))
        ref_param.requires_grad = True
        model_ref.named_parameters.return_value = [
            ("lora.reference.weight", ref_param),
        ]
        model_ref.set_adapter = MagicMock()
        agent.actor = model_ref

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()

        with (
            patch("agilerl.algorithms.core.base.load_file", return_value={}),
            patch("agilerl.algorithms.core.base.set_peft_model_state_dict"),
        ):
            agent._update_existing_adapter(str(tmp_path), "actor")
        model_ref.set_adapter.assert_called_with("actor")
        assert not ref_param.requires_grad


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
        mock_model_load.assert_called_once_with(str(tmp_path), False, True)

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


class TestLLMGenerateWithVllmColocateBarrier:
    """_generate_with_vllm_colocate synchronises ranks before generating."""

    def test_generate_with_vllm_colocate_calls_barrier(self):
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
            patch("agilerl.algorithms.core.base.barrier") as mock_barrier,
        ):
            completion_ids, _action_masks, _ = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        mock_barrier.assert_called()
        assert len(completion_ids) == 1


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

    def test_gathered_optimizer_state_dict_sharded_uses_dcp(self):
        agent = _make_llm_agent()
        agent.actor = _make_mock_peft_actor()
        agent.optimizer = MagicMock()
        get_mock = MagicMock(return_value={"state": {}})
        with (
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
            patch(
                "torch.distributed.checkpoint.state_dict.get_optimizer_state_dict",
                get_mock,
            ),
        ):
            out = agent._gathered_optimizer_state_dict()
        get_mock.assert_called_once()
        assert out == {"state": {}}
        agent.optimizer.state_dict.assert_not_called()

    def test_load_gathered_optimizer_state_dict_sharded_uses_dcp(self):
        agent = _make_llm_agent()
        agent.actor = _make_mock_peft_actor()
        agent.optimizer = MagicMock()
        set_mock = MagicMock()
        with (
            patch("agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True),
            patch(
                "torch.distributed.checkpoint.state_dict.set_optimizer_state_dict",
                set_mock,
            ),
        ):
            agent._load_gathered_optimizer_state_dict({"state": {}})
        set_mock.assert_called_once()
        agent.optimizer.load_state_dict.assert_not_called()

    def test_save_checkpoint_sharded_embeds_gathered_optimizer_state(self, tmp_path):
        grpo = generate_tiny_grpo()
        try:
            with (
                patch(
                    "agilerl.algorithms.core.base.is_fsdp_sharded", return_value=True
                ),
                patch.object(
                    grpo, "_gathered_optimizer_state_dict", return_value={"x": 1}
                ),
            ):
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
            patch("agilerl.algorithms.core.base.is_main_process", return_value=True),
            patch("agilerl.algorithms.core.base.barrier"),
            patch(
                "agilerl.algorithms.core.base.save_peft_adapter_for_vllm_rollout",
                return_value=tmp_path / "missing_adapter",
            ),
            pytest.raises(FileNotFoundError, match="PEFT adapter export"),
        ):
            agent._move_lora_to_vllm()

    def test_logs_debug_lora_norm_on_main_process(self, tmp_path, caplog):
        agent = _make_llm_agent()
        lora_b = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        peft_ref = MagicMock()
        peft_ref.parameters.return_value = [lora_b]
        peft_ref.named_parameters.return_value = [
            ("lora.actor.lora_B.weight", lora_b),
        ]
        peft_ref.set_adapter = MagicMock()
        _setup_agent_for_vllm_lora_sync(agent, peft_ref)

        with (
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
            caplog.at_level(logging.DEBUG, logger="agilerl.algorithms.core.base"),
        ):
            agent._move_lora_to_vllm()

        assert any(
            "lora-sync: actor lora_B L2=" in rec.message for rec in caplog.records
        )

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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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
            patch("agilerl.algorithms.core.base.gather_full_params"),
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

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"")

        with (
            patch("agilerl.algorithms.core.base.load_file", return_value={}),
            patch("agilerl.algorithms.core.base.set_peft_model_state_dict"),
        ):
            agent._update_existing_adapter(str(tmp_path), "actor")

        assert actor_p.requires_grad
        assert critic_p.requires_grad


@_LLM_DEPS_SKIP
class TestLLMLoadCheckpointLoraConfig:
    def test_load_checkpoint_lora_config_missing(self, tmp_path):
        assert LLMAlgorithm._load_checkpoint_lora_config(str(tmp_path)) is None
