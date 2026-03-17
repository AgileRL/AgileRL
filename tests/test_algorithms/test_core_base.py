"""Tests for agilerl.algorithms.core.base module."""

import inspect
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from gymnasium import spaces
from torch import optim

from agilerl.algorithms.core.base import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    _RegistryMeta,
    get_checkpoint_dict,
    get_optimizer_cls,
)
from agilerl.algorithms.core.optimizer_wrapper import OptimizerWrapper
from agilerl.algorithms.core.registry import NetworkGroup
from agilerl.modules import EvolvableMLP
from tests.test_algorithms.test_base import DummyMARLAlgorithm, DummyRLAlgorithm


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
        chkpt = get_checkpoint_dict(agent, using_deepspeed=True)
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
    reduce_memory_peak=False,
    micro_batch_size_per_gpu=None,
    cosine_lr_schedule_config=None,
    max_grad_norm=0.0,
    use_liger_loss=False,
    lora_config=None,
    use_separate_reference_adapter=False,
    actor_network=None,
):
    """Helper to create a _StubLLMAlgorithm with heavily mocked internals."""
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
            batch_size=4,
            lr=1e-4,
            max_grad_norm=max_grad_norm,
            clone=clone,
            reduce_memory_peak=reduce_memory_peak,
            calc_position_embeddings=False,
            seed=42,
            pad_token_id=0,
            pad_token="<pad>",
            use_liger_loss=use_liger_loss,
            lora_config=lora_config if lora_config is not None else MagicMock(),
            use_separate_reference_adapter=use_separate_reference_adapter,
            actor_network=actor_network,
            micro_batch_size_per_gpu=micro_batch_size_per_gpu,
            cosine_lr_schedule_config=cosine_lr_schedule_config,
            accelerator=accelerator,
            device="cpu",
        )
    agent.actor = actor_network
    agent.optimizer = MagicMock()
    agent.optimizer.optimizer = MagicMock()
    agent.optimizer.optimizer.param_groups = [{"lr": 1e-4}]
    agent.lr_scheduler = None
    agent.use_vllm = False
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
    def test_recompile_raises_not_implemented(self):
        agent = _make_llm_agent()
        with pytest.raises(NotImplementedError, match="not available"):
            agent.recompile()


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

    def test_update_lr_raises_without_deepspeed_plugin(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        acc = MagicMock()
        acc.state.deepspeed_plugin = None
        with pytest.raises(ValueError, match="deepspeed plugin"):
            LLMAlgorithm.update_lr(opt, 5e-4, accelerator=acc)

    def test_update_lr_raises_without_deepspeed_config(self):
        opt = torch.optim.Adam([torch.tensor([1.0], requires_grad=True)], lr=1e-3)
        acc = MagicMock()
        plugin = MagicMock(spec=[])
        acc.state.deepspeed_plugin = plugin
        with pytest.raises(ValueError, match="Deepspeed config not found"):
            LLMAlgorithm.update_lr(opt, 5e-4, accelerator=acc)

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


class TestLLMSelectPolicy:
    def test_select_policy_reference_with_separate_adapter(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        with agent.select_policy(use_reference=True):
            agent.actor.set_adapter.assert_any_call("reference")
        agent.actor.set_adapter.assert_called_with("actor")

    def test_select_policy_reference_without_separate_adapter(self):
        agent = _make_llm_agent(use_separate_reference_adapter=False)
        with agent.select_policy(use_reference=True):
            agent.actor.base_model.disable_adapter_layers.assert_called()
        agent.actor.base_model.enable_adapter_layers.assert_called()

    def test_select_policy_actor_with_separate_adapter(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        with agent.select_policy(use_reference=False):
            pass
        agent.actor.set_adapter.assert_called_with("actor")


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
        mask = LLMAlgorithm.create_prompt_masks([3, 5], 10)
        assert mask.shape == (2, 10)
        assert not mask[0, 2].item()
        assert mask[0, 4].item()
        assert not mask[1, 4].item()
        assert mask[1, 6].item()


class TestLLMConfigureBatchSize:
    def test_clone_mode_sets_batch_size_directly(self):
        agent = _make_llm_agent(clone=True)
        assert agent.batch_size_per_process == 4

    def test_reduce_memory_peak_mode(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        agent = _make_llm_agent(accelerator=acc, clone=False, reduce_memory_peak=True)
        assert agent.batch_size_per_process == 1
        assert agent.micro_batch_size_per_gpu == 1

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
        with pytest.raises(ValueError, match="micro_batch_size_per_gpu is 0"):
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
                    reduce_memory_peak=False,
                    calc_position_embeddings=False,
                    seed=42,
                    pad_token_id=0,
                    pad_token="<pad>",
                    use_liger_loss=False,
                    lora_config=MagicMock(),
                    use_separate_reference_adapter=False,
                    actor_network=_make_mock_peft_actor(),
                    accelerator=acc,
                    device="cpu",
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


class TestLLMInitWarnings:
    def test_cosine_lr_with_accelerator_warns_and_nullifies(self):
        acc = _make_mock_accelerator()
        sched = MagicMock()
        agent = _make_llm_agent(
            accelerator=acc,
            cosine_lr_schedule_config=sched,
        )
        assert agent.cosine_lr_schedule_config is None

    def test_reduce_memory_peak_with_micro_batch_raises(self):
        with pytest.raises(ValueError, match="Cannot specify micro_batch_size_per_gpu"):
            _make_llm_agent(
                reduce_memory_peak=True,
                micro_batch_size_per_gpu=2,
                clone=False,
            )

    def test_lr_overwrite_warning_from_deepspeed(self):
        acc = _make_mock_accelerator(
            ds_config={
                "zero_optimization": {"stage": 0},
                "optimizer": {"params": {"lr": 0.999}},
                "train_micro_batch_size_per_gpu": "auto",
            }
        )
        with pytest.warns(UserWarning, match="overwritten"):
            agent = _make_llm_agent(accelerator=acc)
        assert agent.lr == 0.999

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
                reduce_memory_peak=False,
                calc_position_embeddings=False,
                seed=42,
                pad_token_id=0,
                pad_token="<pad>",
                use_liger_loss=False,
                lora_config=None,
                use_separate_reference_adapter=False,
                actor_network=_NonPeftActor(),
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
        with patch("agilerl.algorithms.core.base.LLM", None):
            with pytest.raises(ImportError, match="vLLM is required"):
                agent._configure_vllm()

    def test_uses_default_config_when_none(self):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        agent.vllm_config = None
        with (
            patch("agilerl.algorithms.core.base.LLM", MagicMock()),
            pytest.warns(UserWarning, match="No VLLM config"),
        ):
            agent._configure_vllm()

    def test_raises_when_tp_size_invalid(self):
        acc = _make_mock_accelerator(num_processes=3)
        agent = _make_llm_agent(accelerator=acc)
        agent.vllm_config = MagicMock()
        agent.vllm_config.tensor_parallel_size = 2
        with patch("agilerl.algorithms.core.base.LLM", MagicMock()):
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

    def test_set_reference_raises_on_unknown_adapter_name(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        agent.accelerator = None
        with patch.object(
            type(agent.actor),
            "named_parameters",
            return_value=[
                ("lora.unknown.weight", torch.tensor([1.0])),
            ],
        ):
            with pytest.raises(ValueError, match="Only adapter names"):
                agent.set_reference_policy(1)

    def test_set_reference_without_separate_adapter_no_accelerator(self):
        agent = _make_llm_agent(use_separate_reference_adapter=False)
        agent.accelerator = None
        merged = MagicMock(spec=torch.nn.Module)
        merged.set_adapter = MagicMock()
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
        agent.set_reference_policy(5)
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


class TestLLMSaveLoadCheckpoint:
    def test_save_checkpoint_weights_only_with_accelerator(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.use_separate_reference_adapter = False
        acc.unwrap_model = MagicMock(return_value=agent.actor)
        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3"),
            patch(
                "agilerl.algorithms.core.base.get_checkpoint_dict",
                return_value={"lr": 1e-4},
            ),
            patch("agilerl.algorithms.core.base.torch.save"),
        ):
            agent.save_checkpoint(str(tmp_path))
        agent.actor.save_pretrained.assert_called_once()

    def test_save_checkpoint_full_with_accelerator(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        with (
            patch.object(LLMAlgorithm, "_save_distributed_actor") as mock_save,
            patch(
                "agilerl.algorithms.core.base.get_checkpoint_dict",
                return_value={"lr": 1e-4},
            ),
            patch("agilerl.algorithms.core.base.torch.save"),
        ):
            agent.save_checkpoint(str(tmp_path), weights_only=False)
        mock_save.assert_called_once()

    def test_load_checkpoint_with_accelerator_weights_only(self, tmp_path):
        import dill

        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        chkpt = {"_weights_only": True, "lr": 1e-4}
        torch.save(chkpt, str(tmp_path / "attributes.pt"), pickle_module=dill)
        with patch.object(LLMAlgorithm, "_update_existing_adapter"):
            agent.load_checkpoint(str(tmp_path))

    def test_load_checkpoint_with_accelerator_full(self, tmp_path):
        import dill

        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        chkpt = {"_weights_only": False, "lr": 1e-4}
        torch.save(chkpt, str(tmp_path / "attributes.pt"), pickle_module=dill)
        with patch.object(LLMAlgorithm, "_load_distributed_actor"):
            agent.load_checkpoint(str(tmp_path))

    def test_load_checkpoint_without_accelerator(self, tmp_path):
        agent = _make_llm_agent(accelerator=None)
        agent.accelerator = None
        with patch.object(EvolvableAlgorithm, "load_checkpoint") as mock_load:
            agent.load_checkpoint(str(tmp_path))
            mock_load.assert_called_once_with(str(tmp_path) + "/attributes.pt")


class TestLLMClone:
    """LLMAlgorithm.clone requires full model infrastructure (DeepSpeed, real
    model weights, etc.), so we test it indirectly via `_configure_batch_size`
    with `clone=True` to verify the clone-mode branch."""

    def test_clone_mode_skips_batch_config(self):
        agent = _make_llm_agent(clone=True)
        assert agent.batch_size_per_process == 4


class TestLLMInitMiscPaths:
    def test_use_liger_loss_modifies_lora_config(self):
        lora = MagicMock()
        acc = _make_mock_accelerator()
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
        with patch("agilerl.algorithms.core.base.SamplingParams", None):
            with pytest.raises(ImportError, match="vLLM is required"):
                agent._generate_with_vllm_colocate([], 1)


class TestLLMMoveModelToVllm:
    def test_move_model_to_vllm_with_accelerator(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.llm = MagicMock()
        agent.llm.llm_engine.model_executor.driver_worker.model_runner.model = (
            MagicMock()
        )
        model_ref = MagicMock()
        model_ref.named_parameters.return_value = [
            ("base_model.model.layer.weight", torch.tensor([1.0])),
        ]
        model_ref.prefix = "model"
        acc.unwrap_model = MagicMock(return_value=model_ref)
        with patch("agilerl.algorithms.core.base.gather_if_zero3"):
            agent._move_model_to_vllm()
        agent.llm.reset_prefix_cache.assert_called_once()


class TestConditionalImportFallbacks:
    def test_clone_tensors_fallback_when_deepspeed_import_fails(self):
        from agilerl.algorithms.core.base import clone_tensors_for_torch_save

        result = clone_tensors_for_torch_save({"a": 1})
        assert isinstance(result, dict)

    def test_vllm_fallback_when_import_fails(self):
        from agilerl.algorithms.core import base as base_mod

        assert hasattr(base_mod, "LLM")
        assert hasattr(base_mod, "SamplingParams")


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
                        reduce_memory_peak=False,
                        calc_position_embeddings=False,
                        seed=42,
                        pad_token_id=0,
                        pad_token="<pad>",
                        use_liger_loss=False,
                        lora_config=MagicMock(),
                        use_separate_reference_adapter=False,
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
                    reduce_memory_peak=False,
                    calc_position_embeddings=False,
                    seed=42,
                    pad_token_id=0,
                    pad_token="<pad>",
                    use_liger_loss=False,
                    lora_config=MagicMock(),
                    use_separate_reference_adapter=False,
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
        agent.accelerator = None
        agent.max_grad_norm = 1.0
        param = torch.tensor([1.0], requires_grad=True)
        loss = (param * 2).sum()
        agent.optimizer = MagicMock()
        agent.actor = MagicMock()
        agent.actor.parameters.return_value = [param]
        with patch("agilerl.algorithms.core.base.clip_grad_norm_") as mock_clip:
            LLMAlgorithm._backward_pass(agent, loss)
        mock_clip.assert_called_once()


class TestLLMSaveDistributedActorWithAccelerator:
    def test_save_creates_dir(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        save_dir = str(tmp_path / "ds_save")
        agent._save_distributed_actor(save_dir)
        assert (tmp_path / "ds_save").exists()
        agent.actor.save_checkpoint.assert_called_once()
        agent.actor.set_adapter.assert_called_with("actor")


class TestEvolvableAlgorithmCloneWithAccelerator:
    def test_clone_unwraps_and_wraps_with_accelerator(self, vector_space):
        action_space = spaces.Discrete(2)
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
    """_use_reference_policy sets requires_grad=False on reference params."""

    def test_use_reference_policy_sets_requires_grad_false(self):
        agent = _make_llm_agent(use_separate_reference_adapter=True)
        ref_param = torch.nn.Parameter(torch.tensor([1.0]))
        ref_param.requires_grad = True
        agent.actor.named_parameters = MagicMock(
            return_value=[
                ("lora.reference.weight", ref_param),
                ("other.weight", torch.nn.Parameter(torch.tensor([2.0]))),
            ]
        )
        LLMAlgorithm._use_reference_policy(agent)
        agent.actor.set_adapter.assert_called_with("reference")
        assert not ref_param.requires_grad


class TestLLMMoveModelToVllmSkipsPrefixAndOriginalModule:
    """_move_model_to_vllm skips prefix and original_module params."""

    def test_skips_prefix_and_original_module(self):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.llm = MagicMock()
        llm_model = MagicMock()
        agent.llm.llm_engine.model_executor.driver_worker.model_runner.model = llm_model
        model_ref = MagicMock()
        model_ref.prefix = "model"
        model_ref.named_parameters.return_value = [
            ("base_model.model.model.layer.weight", torch.tensor([1.0])),
            ("base_model.model.original_module.layer.weight", torch.tensor([2.0])),
            ("base_model.model.dense.weight", torch.tensor([3.0])),
        ]
        acc.unwrap_model = MagicMock(return_value=model_ref)
        with patch("agilerl.algorithms.core.base.gather_if_zero3"):
            agent._move_model_to_vllm()
        llm_model.load_weights.assert_called_once_with(
            [("dense.weight", torch.tensor([3.0]))]
        )


class TestLLMCloneWithoutAccelerator:
    """LLMAlgorithm.clone without accelerator."""

    def test_clone_without_accelerator(self):
        agent = _make_llm_agent(accelerator=None, clone=True)
        agent.accelerator = None
        agent.zero_stage = None
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
                    "reduce_memory_peak": False,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "use_separate_reference_adapter": False,
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
        agent.zero_stage = None
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
                    "reduce_memory_peak": False,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "use_separate_reference_adapter": False,
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
                    "reduce_memory_peak": False,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "use_separate_reference_adapter": False,
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
        agent.zero_stage = None
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
                    "reduce_memory_peak": False,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "use_separate_reference_adapter": False,
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
        agent.use_separate_reference_adapter = False
        agent.lora_config = MagicMock()
        peft_actor = _make_mock_peft_actor()

        with (
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, MagicMock(), add_adapters=True)
        peft_actor.set_adapter.assert_called_with("actor")

    def test_initialize_actors_with_none_creates_from_path(self):
        agent = _make_llm_agent()
        agent.pretrained_model_name_or_path = "mock-path"
        agent.use_separate_reference_adapter = False
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
        mock_create.assert_called_once_with("mock-path")

    def test_initialize_actors_peft_model_input_merges(self):
        agent = _make_llm_agent()
        agent.use_separate_reference_adapter = False
        agent.lora_config = None
        agent.zero_stage = None

        peft_model = _make_mock_peft_actor()
        peft_model.peft_config = {"default": MagicMock()}
        merged = MagicMock()
        merged.peft_config = {"default": MagicMock()}
        peft_model.merge_and_unload = MagicMock(return_value=merged)
        peft_result = _make_mock_peft_actor()

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3"),
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_result
            ),
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_result
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, peft_model, add_adapters=True)
        peft_model.merge_and_unload.assert_called_once()

    def test_initialize_actors_with_separate_reference_adapter(self):
        agent = _make_llm_agent()
        agent.use_separate_reference_adapter = True
        agent.lora_config = MagicMock()
        peft_actor = _make_mock_peft_actor()

        with (
            patch(
                "agilerl.algorithms.core.base.get_peft_model", return_value=peft_actor
            ),
            patch(
                "agilerl.algorithms.core.base.DummyEvolvable", return_value=peft_actor
            ),
        ):
            LLMAlgorithm._initialize_actors(agent, MagicMock(), add_adapters=True)
        peft_actor.add_adapter.assert_called_once_with(
            adapter_name="reference", peft_config=agent.lora_config
        )

    def test_initialize_actors_no_add_adapters(self):
        agent = _make_llm_agent()
        agent.use_separate_reference_adapter = False
        agent.lora_config = MagicMock()
        base_model = _make_mock_peft_actor()

        with patch(
            "agilerl.algorithms.core.base.DummyEvolvable", return_value=base_model
        ):
            LLMAlgorithm._initialize_actors(agent, base_model, add_adapters=False)
        base_model.set_adapter.assert_called_with("actor")


class TestLLMUpdateExistingAdapter:
    """_update_existing_adapter overwrites adapter weights in-place."""

    def test_update_existing_adapter(self, tmp_path):
        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = None

        inner_model = MagicMock()
        inner_model.parameters.return_value = []
        ref_param = torch.nn.Parameter(torch.tensor([1.0]))
        ref_param.requires_grad = True
        inner_model.named_parameters.return_value = [
            ("lora.reference.weight", ref_param),
        ]

        outer_model = MagicMock(spec=[])
        outer_model.module = inner_model
        outer_model.parameters = MagicMock(return_value=[])
        acc.unwrap_model = MagicMock(return_value=outer_model)

        adapter_dir = tmp_path / "actor"
        adapter_dir.mkdir()

        with (
            patch("agilerl.algorithms.core.base.gather_if_zero3"),
            patch("agilerl.algorithms.core.base.load_file", return_value={}),
            patch("agilerl.algorithms.core.base.set_peft_model_state_dict"),
        ):
            agent._update_existing_adapter(str(tmp_path), "actor")
        inner_model.set_adapter.assert_called_with("actor")
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
        with patch("agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance):
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
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
            patch("agilerl.algorithms.core.base.LLM", return_value=mock_llm_instance),
            patch(
                "torch.distributed.new_subgroups_by_enumeration",
                return_value=(MagicMock(), MagicMock()),
            ),
        ):
            agent._configure_vllm()
        assert agent.llm is mock_llm_instance
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
    """clean_up calls CUDA cache/sync when CUDA is available."""

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


class TestLLMLoadCheckpointWeightsOnlyWithRefAdapter:
    """load_checkpoint updates both reference and actor adapters when weights_only."""

    def test_load_checkpoint_updates_reference_adapter(self, tmp_path):
        import dill

        acc = _make_mock_accelerator()
        agent = _make_llm_agent(accelerator=acc, use_separate_reference_adapter=True)
        chkpt = {"_weights_only": True, "lr": 1e-4}
        torch.save(chkpt, str(tmp_path / "attributes.pt"), pickle_module=dill)

        with patch.object(LLMAlgorithm, "_update_existing_adapter") as mock_update:
            agent.load_checkpoint(str(tmp_path))
        calls = [c.args for c in mock_update.call_args_list]
        assert (str(tmp_path), "reference") in calls
        assert (str(tmp_path), "actor") in calls


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
            patch("agilerl.algorithms.core.base.SamplingParams", return_value=mock_sp),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
        ):
            completion_ids, action_masks = agent._generate_with_vllm_colocate(
                prompts, group_size=2
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
            patch("agilerl.algorithms.core.base.SamplingParams", return_value=mock_sp),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
        ):
            completion_ids, action_masks = agent._generate_with_vllm_colocate(
                prompts, group_size=2
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
            patch("agilerl.algorithms.core.base.SamplingParams", return_value=mock_sp),
            patch(
                "agilerl.algorithms.core.base.stack_and_pad_experiences",
                return_value=(torch.zeros(2, 5), None),
            ),
            patch("torch.distributed.all_gather_object", side_effect=fake_all_gather),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            completion_ids, action_masks = agent._generate_with_vllm_colocate(
                prompts, group_size=2
            )
        assert len(completion_ids) == 1


class TestLLMCloneBroadcastMultiProcess:
    """clone broadcasts temp directory in multi-process setting."""

    def test_clone_multi_process_broadcasts_temp_dir(self):
        acc = _make_mock_accelerator(num_processes=2)
        agent = _make_llm_agent(accelerator=acc)
        agent.zero_stage = None
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
                    "reduce_memory_peak": False,
                    "calc_position_embeddings": False,
                    "seed": 42,
                    "pad_token_id": 0,
                    "pad_token": "<pad>",
                    "use_liger_loss": False,
                    "lora_config": MagicMock(),
                    "use_separate_reference_adapter": False,
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
