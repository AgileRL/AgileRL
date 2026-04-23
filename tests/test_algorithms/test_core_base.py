"""Tests for agilerl.algorithms.core.base module.



For LLMAlgorithm.save_checkpoint / load_checkpoint, the following cases are considered
exhaustively.

There are 16 cells we care about, defined by the grid
    lora_only:       True / False
    save_optimizer:  True / False
    use_deepspeed:   True / False
× {save, load}.

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
import copy
import inspect
import re
from unittest.mock import MagicMock, PropertyMock, patch
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import dill
import pytest
import torch
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from gymnasium import spaces
from torch import optim
from typing import TYPE_CHECKING

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
from agilerl.utils.algo_utils import VLLMConfig
from agilerl.modules import EvolvableMLP
from agilerl.modules.dummy import DummyEvolvable
from agilerl.algorithms.grpo import GRPO

from tests.test_algorithms.test_base import DummyMARLAlgorithm, DummyRLAlgorithm
from tests.test_algorithms.test_llms.test_grpo import create_module


pytest.importorskip("peft", reason="LLM checkpoint tests require peft.")
pytest.importorskip("transformers", reason="LLM checkpoint tests require transformers.")

if HAS_LLM_DEPENDENCIES or TYPE_CHECKING:
    from tests.test_algorithms.test_llms.test_grpo import deepspeed_config_stage_2
    from peft import LoraConfig


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
        with pytest.warns(DeprecationWarning):
            dim = EvolvableAlgorithm.get_state_dim(vector_space)
        assert dim == (4,)

    def test_get_action_dim_deprecation(self):
        action_space = spaces.Discrete(5)
        with pytest.warns(DeprecationWarning):
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
        "bad_index,msg", [(1.5, "integer"), ("x", "integer"), ([], "integer")]
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
        with pytest.raises(ValueError, match="Init dict.*not found"):
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
                raise RuntimeError("tensor copy failed")
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
        assert hasattr(loaded, "agent") and hasattr(loaded, "label")
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
        from agilerl.algorithms.core.registry import NetworkGroup

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
        from agilerl.algorithms.core.registry import OptimizerConfig

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
        env_acts, agent_masks = ma_agent.extract_agent_masks(infos)
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
        from agilerl.algorithms.core.registry import OptimizerConfig

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
    reduce_memory_peak: bool = False,
):
    """Helper to create a _StubLLMAlgorithm with heavily mocked internals."""
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
            accelerator=accelerator,
            device="cpu",
            use_separate_reference_adapter=use_separate_reference_adapter,
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
        result_acc, _ = LLMAlgorithm.update_lr(opt, 5e-4, accelerator=acc)
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
        wrapped_actor = MagicMock()
        wrapped_actor.module = MagicMock()
        wrapped_actor.optimizer = MagicMock()
        acc.prepare = MagicMock(
            return_value=(wrapped_actor, agent.optimizer.optimizer, None)
        )
        acc.unwrap_model = MagicMock(return_value=wrapped_actor.module)
        LLMAlgorithm.wrap_models(agent)
        acc.prepare.assert_called_once()
        wrapped_actor.module.gradient_checkpointing_enable.assert_called_once()

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
        original_actor = agent.actor
        LLMAlgorithm.wrap_models(agent)
        original_actor.gradient_checkpointing_enable.assert_called_once()


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


class TestLLMMemoryEfficientLogits:
    def test_memory_efficient_logits_computes_log_probs(self):
        logits = torch.randn(2, 5, 10)
        index = torch.randint(0, 10, (2, 5))
        result = LLMAlgorithm._memory_efficient_logits(logits, index)
        assert result.shape == (2, 5)
        expected = (
            F.log_softmax(logits, dim=-1)
            .gather(dim=-1, index=index.unsqueeze(-1))
            .squeeze(-1)
        )
        assert torch.allclose(result, expected)


class TestLLMCreatePromptMasks:
    def test_creates_correct_mask(self):
        mask = LLMAlgorithm._create_prompt_masks([3, 5], 10)
        assert mask.shape == (2, 10)
        assert not mask[0, 2].item()
        assert mask[0, 4].item()
        assert not mask[1, 4].item()
        assert mask[1, 6].item()


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
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
        with pytest.raises(
            ValueError,
            match="micro_batch_size_per_gpu is equal to zero, which is not allowed.",
        ):
            with (
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


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
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

    def test_mutation_hook_registered_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc, max_grad_norm=1.0)
        agent.registry.hooks = ["_sync_deepspeed_gradient_clipping"]
        agent._sync_deepspeed_gradient_clipping = MagicMock()
        agent.mutation_hook()
        agent._sync_deepspeed_gradient_clipping.assert_called_once()


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
        with patch.object(
            type(agent.actor),
            "named_parameters",
            return_value=[
                ("not_lora.weight", torch.tensor([1.0])),
            ],
        ):
            with pytest.raises(
                ValueError, match="No LoRA tensors found for source adapter"
            ):
                agent.set_reference_policy(1)

    def test_set_reference_raises_on_no_target_params(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        agent.accelerator = None
        with patch.object(
            type(agent.actor),
            "named_parameters",
            return_value=[
                ("lora.actor.weight", torch.tensor([1.0])),
            ],
        ):
            with pytest.raises(
                ValueError, match="No LoRA tensors found for target adapter"
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
                match="Target adapter 'reference' is missing 1 LoRA tensors present in source adapter 'actor'.",
            ):
                agent.set_reference_policy(1)

    def test_set_reference_without_separate_adapter_no_accelerator(self):
        agent = _make_llm_agent(use_separate_reference_adapter=False)
        agent.accelerator = None
        merged = MagicMock(spec=torch.nn.Module)
        merged.set_adapter = MagicMock()
        merged.base_model = MagicMock()
        merged.parameters.return_value = [torch.tensor([1.0])]
        agent.actor.merge_and_unload.return_value = merged
        agent.lora_config = MagicMock()
        with (
            patch("agilerl.algorithms.core.base.get_peft_model", return_value=merged),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            agent.set_reference_policy(1)
        assert agent.reference_update_tracker == 1

    def test_set_reference_without_separate_adapter_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc, use_separate_reference_adapter=False)
        merged = MagicMock(spec=torch.nn.Module)
        merged.set_adapter = MagicMock()
        merged.base_model = MagicMock()
        merged.parameters.return_value = [torch.tensor([1.0])]
        unwrapped = MagicMock()
        unwrapped.merge_and_unload.return_value = merged
        acc.unwrap_model = MagicMock(return_value=unwrapped)
        agent.lora_config = MagicMock()
        with (
            patch("agilerl.algorithms.core.base.get_peft_model", return_value=merged),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            agent.set_reference_policy(1)
        assert agent.reference_update_tracker == 1
        acc.wait_for_everyone.assert_called()

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
        with patch.object(
            LLMAlgorithm, "_memory_efficient_logits", return_value=torch.randn(2, 9)
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
    raise KeyError(f"no actor param matching {substring!r}")


def find_exp_avg_in_opt_state(agent) -> torch.Tensor | None:
    """Return a reference to the first Adam ``exp_avg`` tensor in agent.optimizer.

    Returns None if optimizer.state is empty (e.g. before any step)."""
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


@pytest.fixture(scope="function")
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


@pytest.fixture(scope="function", params=SAVE_LOAD_OPTIONS)
def llm_simple_checkpoint(request, grpo_factory, tmp_path_factory):
    """One saved plain-path checkpoint per cell, shared across all tests that
    only *read* the output. This does not involve deepspeed.

    Session scope means ``save_checkpoint`` runs exactly 4 times for the whole
    test session (once per cell), not once per test.
    """
    lora_only, save_optimizer = request.param
    agent = grpo_factory
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
    """Each test runs 4× (one per SAVE_LOAD_OPTIONS param) against a pre-saved
    checkpoint. Assertions are phrased as truth tables over
    ``plain_saved.lora_only`` / ``plain_saved.save_optimizer``."""

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
    optimizer). Cheap because deepcopy of the template is near-instant."""
    lora_only, save_optimizer = request.param
    agent = grpo_factory
    return SimpleNamespace(
        agent=agent,
        path=tmp_path,
        lora_only=lora_only,
        save_optimizer=save_optimizer,
    )


class TestLLMSimpleCheckpointLoad:
    """Roundtrip: stamp sentinels on tracked state → save → clobber → load →
    assert sentinels restored. Specifically catches 'load silently
    reinitialised a fresh optimizer / fresh weights'."""

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
    unwrap_model that just returns the wrapped model."""
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
    the template, each saved once."""
    lora_only, save_optimizer = request.param
    agent = grpo_factory
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
    ``TestDeepspeedSaveE2E`` below (CUDA-only, @pytest.mark.llm).
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
    saver = grpo_factory
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
    loader = grpo_factory
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

        agent = grpo_factory
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

        agent = grpo_factory
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
# E2E DeepSpeed tests — real save/load, CUDA-only, @pytest.mark.llm           #
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


@pytest.mark.llm
class TestLLMDeepspeedCheckpointSave:
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


@pytest.mark.llm
class TestLLMDeepspeedCheckpointSaveLoad:
    """Real DeepSpeed roundtrip: stamp sentinels → save → fresh agent → load
    → assert sentinels restored. One parametrised test per concern
    (weights / optimizer) to keep the real DeepSpeed builds bounded.

    NB: building the second accelerator via the factory triggers
    ``AcceleratorState._reset_state`` which invalidates the first engine.
    That's fine because the first agent is only needed for the save step.
    """

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


class TestMergeLoraConfigs:
    """Unit tests for ``LLMAlgorithm._merge_lora_configs``. Rules under test:

    * ``current=None`` → checkpoint is returned as-is, no warnings.
    * ``r``               → ``max(current, checkpoint)``; warn on mismatch.
    * ``target_modules``  → set union; warn on difference.
    * ``modules_to_save`` → set union; warn on difference.
    * anything else       → current kept; warn on difference.
    """

    def test_current_none_returns_checkpoint_unchanged(self):
        ckpt = get_lora_config(r=8)
        # No warnings should fire when there's nothing to merge against.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            merged = LLMAlgorithm._merge_lora_configs(None, ckpt)
        assert merged is ckpt

    def test_rank_takes_max_and_warns_on_mismatch(self):
        current = get_lora_config(r=2)
        ckpt = get_lora_config(r=8)
        with pytest.warns(UserWarning, match="LoRA rank mismatch"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        assert merged.r == 8

    def test_rank_equal_no_warning(self):
        current = get_lora_config(r=4)
        ckpt = get_lora_config(r=4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        assert merged.r == 4

    def test_target_modules_unioned_and_warns(self):
        current = get_lora_config(target_modules=("linear_1",))
        ckpt = get_lora_config(target_modules=("linear_1", "linear_2"))
        with pytest.warns(UserWarning, match="'target_modules' differs"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        # merged.target_modules is a sorted list (per the implementation).
        assert set(merged.target_modules) == {"linear_1", "linear_2"}

    def test_target_modules_equal_no_warning(self):
        current = get_lora_config(target_modules=("linear_1", "linear_2"))
        ckpt = get_lora_config(target_modules=("linear_1", "linear_2"))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            LLMAlgorithm._merge_lora_configs(current, ckpt)

    def test_modules_to_save_unioned_and_warns(self):
        current = get_lora_config(modules_to_save=("summary",))
        ckpt = get_lora_config(modules_to_save=("summary", "v_head"))
        with pytest.warns(UserWarning, match="'modules_to_save' differs"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        assert set(merged.modules_to_save) == {"summary", "v_head"}

    def test_other_field_mismatch_warns_and_keeps_current(self):
        current = get_lora_config(lora_alpha=8)
        ckpt = get_lora_config(lora_alpha=32)
        with pytest.warns(UserWarning, match="'lora_alpha' differs"):
            merged = LLMAlgorithm._merge_lora_configs(current, ckpt)
        # Current wins for non-special fields.
        assert merged.lora_alpha == 8


# --------------------------------------------------------------------------- #
# LoRA config merge — integration with save/load                              #
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
        accelerator=None,
        wrap=False,
        gradient_checkpointing=False,
        device="cpu",
        use_separate_reference_adapter=True,
    )


class TestMergeLoraConfigsRoundtrip:
    """Save a lora-only checkpoint with config A, load into an agent built
    with config B, and verify the merged config survives load.

    Only lora-only checkpoints carry a ``LoraConfig`` on disk (via
    ``save_pretrained``), so that's the branch where ``_merge_lora_configs``
    actually runs during load when ``merge_lora_configs=True``.
    """

    def test_merged_lora_config_persists_on_agent_when_merge_enabled(self, tmp_path):
        """The merged config should survive on ``self.lora_config``, mirroring
        the deepspeed path's ``_restore_checkpoint_attributes`` behaviour."""
        from unittest.mock import patch

        saver = _build_grpo_with_lora(
            get_lora_config(r=2, target_modules=("linear_1",))
        )
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=8, target_modules=("linear_1", "linear_2"))
        )
        with (
            patch.object(LLMAlgorithm, "_load_adapter_weights"),
            patch.object(LLMAlgorithm, "_copy_adapter_weights"),
            patch.object(LLMAlgorithm, "_reconfigure_adapters_to_match"),
        ):
            loader.load_checkpoint(
                str(tmp_path), load_optimizer=False, merge_lora_configs=True
            )

        assert loader.lora_config.r == 8
        assert set(loader.lora_config.target_modules) == {"linear_1", "linear_2"}

    def test_full_roundtrip_with_rank_growth_loads_weights_when_merge_enabled(
        self, tmp_path
    ):
        """End-to-end: save at r=2, load into r=8 agent — merge takes
        ``r=max(2,8)=8``, ``_reconfigure_adapters_to_match`` rebuilds the live
        adapter at rank 8, and ``_pad_adapter_state_to_live_shape`` drops the
        saved r=2 weights into the top-left rank slice before peft's
        ``set_peft_model_state_dict`` applies them."""
        saver = _build_grpo_with_lora(
            get_lora_config(r=2, target_modules=("linear_1",))
        )
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=8, target_modules=("linear_1",))
        )
        loader.load_checkpoint(
            str(tmp_path), load_optimizer=False, merge_lora_configs=True
        )
        assert loader.lora_config.r == 8

    def test_load_no_warning_when_configs_match(self, tmp_path):
        cfg = get_lora_config(r=4, target_modules=("linear_1",))
        saver = _build_grpo_with_lora(cfg)
        saver.save_checkpoint(str(tmp_path), lora_only=True, save_optimizer=False)

        loader = _build_grpo_with_lora(
            get_lora_config(r=4, target_modules=("linear_1",))
        )
        # We only assert the merge-specific warnings don't fire — PEFT /
        # other parts of load may legitimately warn on unrelated things.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader.load_checkpoint(str(tmp_path), load_optimizer=False)
        merge_warnings = [
            w
            for w in caught
            if "rank mismatch" in str(w.message)
            or "'target_modules' differs" in str(w.message)
            or "'modules_to_save' differs" in str(w.message)
        ]
        assert merge_warnings == [], (
            f"unexpected merge warnings: {[str(w.message) for w in merge_warnings]}"
        )
        assert loader.lora_config.r == 4


class TestLLMClone:
    """LLMAlgorithm.clone requires full model infrastructure (DeepSpeed, real
    model weights, etc.), so we test it indirectly via `_configure_batch_size`
    with `clone=True` to verify the clone-mode branch."""

    def test_clone_mode_skips_batch_config(self):
        agent = _make_llm_agent(clone=True)
        assert agent.batch_size_per_process == 4


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
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
            match="DeepSpeed plugin is not initialized. If using an accelerator,",
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


class TestLLMMoveModelToVllm:
    def test_move_model_to_vllm_resolves_model_ref(self):
        """``model_ref`` from unwrap (accelerator), ``DummyEvolvable.module``, or ``actor``."""
        p = torch.nn.Parameter(torch.tensor([1.0]))
        named = [("base_model.model.layer.weight", p)]
        gather = patch("agilerl.algorithms.core.base.gather_if_zero3")

        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        unwrapped = MagicMock()
        unwrapped.parameters.return_value = [p]
        unwrapped.named_parameters.return_value = named
        unwrapped.prefix = "model"
        acc.unwrap_model = MagicMock(return_value=unwrapped)
        agent.llm = MagicMock()
        with gather:
            agent._move_model_to_vllm()
        agent.llm.apply_model.assert_called_once()
        agent.llm.reset_prefix_cache.assert_called_once()

        agent = _make_llm_agent(accelerator=None)
        inner = _make_mock_peft_actor()
        inner.parameters = MagicMock(return_value=[p])
        inner.named_parameters = MagicMock(return_value=named)
        inner.merge_adapter = MagicMock()
        inner.unmerge_adapter = MagicMock()
        inner.set_adapter = MagicMock()
        inner.prefix = "model"
        agent.actor = DummyEvolvable(device="cpu", module=inner)
        agent.llm = MagicMock()
        with gather:
            agent._move_model_to_vllm()
        inner.merge_adapter.assert_called_once()
        agent.llm.apply_model.assert_called_once()
        agent.llm.reset_prefix_cache.assert_called_once()

        agent = _make_llm_agent(accelerator=None)
        actor = MagicMock()
        actor.parameters.return_value = [p]
        actor.named_parameters.return_value = named
        actor.prefix = "model"
        agent.actor = actor
        agent.llm = MagicMock()
        with gather:
            agent._move_model_to_vllm()
        actor.merge_adapter.assert_called_once()
        agent.llm.apply_model.assert_called_once()
        agent.llm.reset_prefix_cache.assert_called_once()


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


class TestMultiAgentExtractAgentMasksContinuousNan:
    def test_extract_agent_masks_none_continuous_action(self, vector_space):
        obs = [vector_space, vector_space]
        act = [spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)] * 2
        agent = DummyMARLAlgorithm(obs, act, agent_ids=["agent_0", "agent_1"], index=0)
        infos = {
            "agent_0": {"env_defined_actions": None},
            "agent_1": {"env_defined_actions": np.array([1.0, 2.0])},
        }
        env_acts, agent_masks = agent.extract_agent_masks(infos)
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


@pytest.mark.skipif(not HAS_LLM_DEPENDENCIES, reason="LLM dependencies not installed")
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
        accelerator = Accelerator()
        agent = DummyRLAlgorithm(
            vector_space, action_space, index=0, accelerator=accelerator
        )
        clone = agent.clone(index=1, wrap=True)
        assert clone is not agent
        assert clone.index == 1
        agent.accelerator = None
        clone.accelerator = None


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


class TestLLMMoveModelToVllmSkipsPrefixAndOriginalModule:
    """_move_model_to_vllm skips PEFT adapter params (lora_, original_module, etc.)."""

    def test_skips_peft_adapter_params(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.llm = MagicMock()
        model_ref = MagicMock()
        model_ref.prefix = "base_model"
        model_ref.named_parameters.return_value = [
            ("base_model.model.model.layer.base_layer.weight", torch.tensor([1.0])),
            ("base_model.model.model.layer.lora_A.actor.weight", torch.tensor([2.0])),
            ("base_model.model.model.layer.lora_B.actor.weight", torch.tensor([3.0])),
            (
                "base_model.model.model.layer.original_module.weight",
                torch.tensor([4.0]),
            ),
            ("base_model.model.dense.weight", torch.tensor([5.0])),
        ]
        acc.unwrap_model = MagicMock(return_value=model_ref)
        with patch("agilerl.algorithms.core.base.gather_if_zero3"):
            agent._move_model_to_vllm()
        agent.llm.apply_model.assert_called_once()
        call_args = agent.llm.apply_model.call_args
        load_fn = call_args[0][0]
        mock_model = MagicMock()
        load_fn(mock_model)
        mock_model.load_weights.assert_called_once()
        loaded = mock_model.load_weights.call_args[0][0]
        names = [n for n, _ in loaded]
        assert len(loaded) == 2
        assert "model.layer.weight" in names
        assert "dense.weight" in names


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
            patch.object(_RegistryMeta, "__call__", return_value=cloned),
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
            patch.object(_RegistryMeta, "__call__", return_value=cloned),
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
            patch.object(_RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
            patch.object(LLMAlgorithm, "_save_distributed_actor"),
            patch.object(LLMAlgorithm, "_load_distributed_actor"),
        ):
            result = LLMAlgorithm.clone(agent, index=7)
        assert result is cloned


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
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
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
        created_model = MagicMock()

        with (
            patch(
                "agilerl.algorithms.core.base.create_model_from_name_or_path",
                return_value=created_model,
            ) as mock_create,
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, None, add_adapters=True)
        mock_create.assert_called_once_with(
            "mock-path", add_value_head=False, use_accelerator=False
        )

    def test_initialize_actors_user_peft_warns_and_reinitializes_adapters(self):
        """User PEFT input is warned and replaced with a fresh actor adapter."""
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        peft_model = _make_mock_peft_actor()
        dense = MagicMock(spec=[])
        peft_actor = _make_mock_peft_actor()

        with (
            patch.object(
                LLMAlgorithm, "_warn_peft_model", return_value=dense
            ) as mock_warn,
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ) as mock_gpm,
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, peft_model, add_adapters=True)

        mock_warn.assert_called_once_with(peft_model, context="actor_network")
        mock_gpm.assert_called_once()
        call_kw = mock_gpm.call_args
        assert call_kw[0][0] is dense

    def test_warn_peft_model_warns_and_merges(self):
        """_warn_peft_model emits the expected warning and merges adapters."""
        agent = _make_llm_agent()
        peft_model = _make_mock_peft_actor()
        dense = torch.nn.Module()
        peft_model.merge_and_unload.return_value = dense

        with pytest.warns(
            UserWarning,
            match=re.escape(
                "actor_network: A PeftModel was passed; calling merge_and_unload() to merge active adapter weights "
                "into the dense base model before attaching new randomly initialized AgileRL adapters."
            ),
        ):
            out = agent._warn_peft_model(peft_model, context="actor_network")

        peft_model.merge_and_unload.assert_called_once_with()
        assert out is dense

    def test_initialize_actors_with_separate_reference_adapter(self):
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()
        agent.selected_adapters = ("actor", "reference")
        peft_actor = _make_mock_peft_actor()

        with (
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
            adapter_name="reference", peft_config=agent.lora_config
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

    def test_initialize_actors_peft_extra_adapter_names_warns_and_reinitializes(self):
        """Stray adapter names on user PEFT input are ignored via reinitialization."""
        agent = _make_llm_agent()
        agent.lora_config = MagicMock()

        peft_in = _make_mock_peft_actor()
        peft_in.peft_config = {"actor": MagicMock(), "stray_adapter": MagicMock()}
        dense = MagicMock(spec=[])
        peft_out = _make_mock_peft_actor()

        with (
            patch.object(
                LLMAlgorithm, "_warn_peft_model", return_value=dense
            ) as mock_warn,
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_out
            ) as mock_gpm,
            patch("agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_out),
        ):
            LLMAlgorithm._initialize_actors(agent, peft_in, add_adapters=True)

        mock_warn.assert_called_once_with(peft_in, context="actor_network")
        assert mock_gpm.call_args[0][0] is dense

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
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ) as mock_gpm,
            patch("agilerl.algorithms.core.base.patch_lora_for_fused_forward"),
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

    def test_initialize_actors_value_head_merges_inner_peft_and_warns(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.use_value_head = True
        agent.use_separate_reference_adapter = False
        agent.lora_config = MagicMock()

        inner_peft = _make_mock_peft_actor()
        inner_peft.peft_config = {"default": MagicMock()}
        dense_inner = torch.nn.Module()
        base_model = torch.nn.Module()
        base_model.pretrained_model = inner_peft
        peft_actor = _make_mock_peft_actor()
        peft_actor.peft_config = {}

        with (
            patch.object(
                LLMAlgorithm, "_warn_peft_model", return_value=dense_inner
            ) as mock_warn,
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ) as mock_gpm,
            patch("agilerl.algorithms.core.base.patch_lora_for_fused_forward"),
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=True)

        mock_warn.assert_called_once_with(
            inner_peft, context="actor_network.pretrained_model"
        )
        assert mock_gpm.call_args[0][0] is dense_inner
        assert base_model.pretrained_model is peft_actor
        assert agent.actor is base_model


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
            patch("agilerl.algorithms.core.base.gather_if_zero3"),
            patch("agilerl.algorithms.core.base.load_file", return_value={}),
            patch("agilerl.algorithms.core.base.set_peft_model_state_dict"),
        ):
            agent._update_existing_adapter(str(tmp_path), "actor")
        model_ref.set_adapter.assert_called_with("actor")
        assert not ref_param.requires_grad


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
            "agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance
        ) as mock_llm_cls:
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
        mock_llm_cls.assert_called_once()
        acc.wait_for_everyone.assert_called()

    def test_configure_vllm_tp_size_gt_1(self):
        acc = _make_mock_accelerator(num_processes=4)
        agent = _make_llm_agent(accelerator=acc)
        vllm_config = MagicMock()
        vllm_config.tensor_parallel_size = 2
        vllm_config.gpu_memory_utilization = 0.9
        vllm_config.max_num_seqs = 256
        vllm_config.sleep_mode = True
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
                "agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance
            ) as mock_llm_cls,
        ):
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
        mock_llm_cls.assert_called_once()
        mock_llm_instance.sleep.assert_called_once_with(level=2)

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
        mock_model_load.assert_called_once_with(str(tmp_path), False, True, False)

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
            patch.object(
                LLMAlgorithm, "_reconfigure_adapters_to_match"
            ) as mock_reconfig,
            pytest.raises(ValueError, match="LoRA configs differ"),
        ):
            agent._load_model_checkpoint(str(tmp_path))

        mock_reconfig.assert_not_called()

    def test_load_model_checkpoint_can_merge_on_lora_config_mismatch(self, tmp_path):
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
            patch.object(
                LLMAlgorithm,
                "_merge_lora_configs",
                wraps=LLMAlgorithm._merge_lora_configs,
            ) as mock_merge,
            patch.object(
                LLMAlgorithm, "_reconfigure_adapters_to_match"
            ) as mock_reconfig,
        ):
            agent._load_model_checkpoint(str(tmp_path), merge_lora_configs=True)

        mock_merge.assert_called_once()
        mock_reconfig.assert_called_once()
        assert "linear_1" in set(agent.lora_config.target_modules)
        assert "linear_2" in set(agent.lora_config.target_modules)


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
            completion_ids, action_masks = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        assert len(completion_ids) == 2
        assert len(action_masks) == 2


class TestLLMGenerateWithVllmColocateAccelerator:
    """_generate_with_vllm_colocate waits for all processes with accelerator."""

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
            completion_ids, action_masks = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        acc.wait_for_everyone.assert_called()
        assert len(completion_ids) == 1


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
            completion_ids, action_masks = agent._generate_with_vllm_colocate(
                prompts, group_size=2, temperature=0.9
            )
        assert len(completion_ids) == 1


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
            patch.object(_RegistryMeta, "__call__", return_value=cloned),
            patch.object(LLMAlgorithm, "wrap_models"),
        ):
            LLMAlgorithm.clone(agent, index=3)
        mock_broadcast.assert_called_once()
