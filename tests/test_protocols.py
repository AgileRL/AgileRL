# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

from agilerl.algorithms import (
    CQN,
    DDPG,
    DQN,
    IPPO,
    MADDPG,
    MATD3,
    PPO,
    TD3,
    NeuralTS,
    NeuralUCB,
    RainbowDQN,
)
from agilerl.algorithms.core import MultiAgentRLAlgorithm
from agilerl.modules import (
    EvolvableCNN,
    EvolvableLSTM,
    EvolvableMLP,
    EvolvableMultiInput,
    EvolvableResNet,
    EvolvableSimBa,
    ModuleDict,
)
from agilerl.networks import (
    ContinuousQNetwork,
    DeterministicActor,
    QNetwork,
    RainbowQNetwork,
    StochasticActor,
    ValueNetwork,
)
from agilerl.protocols import (
    BanditEnvProtocol,
    EvolvableAlgorithmProtocol,
    EvolvableModuleProtocol,
    EvolvableNetworkProtocol,
    LoraConfigProtocol,
    ModuleDictProtocol,
    MutationMethodProtocol,
    MutationType,
    PeftModelProtocol,
    PreTrainedModelProtocol,
)
from agilerl.wrappers.agent import RSNorm
from agilerl.wrappers.learning import BanditEnv
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multidiscrete_space,
    generate_random_box_space,
)


@pytest.fixture
def action_space(network_cls: type[EvolvableNetworkProtocol]) -> gym.Space | None:
    if issubclass(
        network_cls,
        (DeterministicActor, StochasticActor, ContinuousQNetwork),
    ):
        return generate_random_box_space(shape=(2,))
    if issubclass(network_cls, (RainbowQNetwork, QNetwork)):
        return generate_discrete_space(2)
    return None


class TestEvolvableAlgorithmProtocol:
    """Test that classes implement the EvolvableAlgorithm protocol."""

    @pytest.mark.parametrize(
        ("algorithm_cls", "algo_action_space"),
        [
            (CQN, generate_discrete_space(2)),
            (DQN, generate_discrete_space(2)),
            (RainbowDQN, generate_discrete_space(2)),
            (NeuralTS, generate_discrete_space(2)),
            (NeuralUCB, generate_discrete_space(2)),
            (DDPG, generate_random_box_space(shape=(2,))),
            (TD3, generate_random_box_space(shape=(2,))),
            (PPO, generate_random_box_space(shape=(2,))),
            (PPO, generate_multidiscrete_space(2, 2)),
            (PPO, generate_discrete_space(2)),
            (IPPO, generate_discrete_space(2)),
            (IPPO, generate_random_box_space(shape=(2,))),
            (IPPO, generate_multidiscrete_space(2, 2)),
            (MADDPG, generate_random_box_space(shape=(2,), low=-1.0, high=1.0)),
            (MATD3, generate_random_box_space(shape=(2,), low=-1.0, high=1.0)),
        ],
    )
    @pytest.mark.parametrize(
        "observation_space",
        [
            generate_random_box_space(shape=(4,)),
            generate_random_box_space(shape=(3, 32, 32)),
            generate_dict_or_tuple_space(2, 2),
            generate_discrete_space(6),
            generate_multidiscrete_space(4, 4),
        ],
    )
    def test_algorithm_instances_implement_evolvable_algorithm_protocol(
        self,
        algorithm_cls,
        algo_action_space,
        observation_space,
    ):
        """Test that algorithm instances implement the EvolvableAlgorithm protocol."""
        # Skip actual instantiation, just check the class definitions
        if issubclass(algorithm_cls, MultiAgentRLAlgorithm):
            instance = algorithm_cls(
                observation_spaces=[observation_space],
                action_spaces=[algo_action_space],
                agent_ids=["agent_0"],
            )
        else:
            instance = algorithm_cls(observation_space, algo_action_space)

        assert isinstance(instance, EvolvableAlgorithmProtocol)


class TestEvolvableNetworkProtocol:
    """Test that classes implement the EvolvableNetwork protocol."""

    @pytest.mark.parametrize(
        "network_cls",
        [
            QNetwork,
            RainbowQNetwork,
            ContinuousQNetwork,
            ValueNetwork,
            DeterministicActor,
            StochasticActor,
        ],
    )
    @pytest.mark.parametrize(
        "observation_space",
        [
            generate_random_box_space(shape=(4,)),
            generate_random_box_space(shape=(3, 32, 32)),
            generate_dict_or_tuple_space(2, 2),
            generate_discrete_space(6),
            generate_multidiscrete_space(4, 4),
        ],
    )
    def test_network_instances_implement_evolvable_network_protocol(
        self,
        network_cls,
        observation_space,
        action_space,
    ):
        """Test that network instances implement the EvolvableNetwork protocol."""
        kwargs = {}
        if issubclass(network_cls, RainbowQNetwork):
            kwargs["support"] = torch.linspace(-100, 100, 51)

        # Create network instance
        network = network_cls(observation_space, action_space, **kwargs)
        assert isinstance(network, EvolvableNetworkProtocol)


class TestEvolvableModuleProtocol:
    """Test that classes implement the EvolvableModule protocol."""

    @pytest.mark.parametrize(
        "module_cls",
        [
            EvolvableMLP,
            EvolvableLSTM,
            EvolvableSimBa,
            EvolvableResNet,
            EvolvableCNN,
            EvolvableMultiInput,
        ],
    )
    def test_module_instances_implement_evolvable_module_protocol(self, module_cls):
        """Test that module instances implement the EvolvableModule protocol."""
        kwargs = {"num_outputs": 10}
        if issubclass(module_cls, (EvolvableCNN, EvolvableResNet)):
            kwargs["input_shape"] = (3, 32, 32)
            kwargs["channel_size"] = [8, 8]
            kwargs["kernel_size"] = [3, 3]
            kwargs["stride_size"] = [1, 1]
        elif issubclass(module_cls, (EvolvableMLP, EvolvableSimBa)):
            kwargs["num_inputs"] = 4
            kwargs["hidden_size"] = [16, 32]
        elif issubclass(module_cls, EvolvableLSTM):
            kwargs["input_size"] = 4
            kwargs["hidden_state_size"] = 16
        elif issubclass(module_cls, EvolvableMultiInput):
            kwargs["observation_space"] = generate_dict_or_tuple_space(2, 2)

        if issubclass(module_cls, (EvolvableSimBa, EvolvableResNet)):
            kwargs["num_blocks"] = 2
            if issubclass(module_cls, EvolvableResNet):
                kwargs["channel_size"] = 8
                kwargs["kernel_size"] = 3
                kwargs["stride_size"] = 1
            else:
                kwargs["hidden_size"] = 16

        instance = module_cls(**kwargs)
        assert isinstance(instance, EvolvableModuleProtocol)

        mutation_methods = instance.get_mutation_methods()
        for method_name, method in mutation_methods.items():
            assert isinstance(
                method,
                MutationMethodProtocol,
            ), (
                f"Mutation method {method_name} does not implement MutationMethod protocol"
            )


class TestPreTrainedModelProtocol:
    """Test that HuggingFace PreTrainedModel instances satisfy PreTrainedModelProtocol."""

    def test_pretrained_model_instances_implement_pretrained_model_protocol(self):
        """Test that HuggingFace PreTrainedModel instances satisfy PreTrainedModelProtocol."""
        pytest.importorskip("transformers")
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=100,
            n_positions=64,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
        model = GPT2LMHeadModel(config)
        assert isinstance(model, PreTrainedModelProtocol)


class TestPeftModelProtocol:
    """Test that PEFT PeftModel instances satisfy PeftModelProtocol."""

    def test_peft_model_instances_implement_peft_model_protocol(self):
        """Test that PEFT PeftModel instances satisfy PeftModelProtocol."""
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from peft import LoraConfig, get_peft_model
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=100,
            n_positions=64,
            n_embd=32,
            n_layer=1,
            n_head=2,
        )
        base_model = GPT2LMHeadModel(config)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base_model, lora_config)
        assert isinstance(peft_model, PeftModelProtocol)


class TestModuleDictProtocol:
    def test_module_dict_implements_protocol(self):
        mdl = ModuleDict(
            {"a": EvolvableMLP(num_inputs=4, num_outputs=2, hidden_size=[8])},
        )
        assert isinstance(mdl, ModuleDictProtocol)


class TestNStepAgentAccess:
    """The prioritized-replay path must keep working for wrapped agents."""

    def test_wrapper_is_not_a_rainbow_instance(self):
        rainbow = RainbowDQN(
            generate_random_box_space(shape=(4,)), generate_discrete_space(2)
        )
        assert not isinstance(RSNorm(rainbow), RainbowDQN)

    def test_accessor_passes_wrapped_agents_through(self):
        from agilerl.training.train_off_policy import _as_n_step_agent

        rainbow = RainbowDQN(
            generate_random_box_space(shape=(4,)), generate_discrete_space(2)
        )
        wrapper = RSNorm(rainbow)
        assert _as_n_step_agent(wrapper) is wrapper

    def test_wrapper_proxies_beta_reads_and_writes(self):
        rainbow = RainbowDQN(
            generate_random_box_space(shape=(4,)), generate_discrete_space(2)
        )
        wrapper = RSNorm(rainbow)
        start = rainbow.beta
        wrapper.beta += 0.1
        assert wrapper.beta == pytest.approx(start + 0.1)
        assert rainbow.beta == pytest.approx(start + 0.1)


class TestLoraConfigProtocol:
    def test_lora_config_implements_protocol(self):
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from peft import LoraConfig

        lora = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
        )
        assert isinstance(lora, LoraConfigProtocol)


class TestMutationType:
    def test_mutation_type_enum_values(self):
        assert MutationType.LAYER.value == "layer"
        assert MutationType.NODE.value == "node"
        assert MutationType.ACTIVATION.value == "activation"


def test_protocol_type_aliases_importable():
    from agilerl.protocols import (
        NumpyObsType,
        TorchObsType,
    )

    assert NumpyObsType is not None
    assert TorchObsType is not None


class TestBanditEnvProtocol:
    """Test the reference :class:`~agilerl.wrappers.learning.BanditEnv` implementation."""

    def test_bandit_env_implements_protocol(self):
        features = pd.DataFrame(np.random.uniform(0, 1, size=(10, 1)), columns=["x"])
        targets = pd.DataFrame(np.random.randint(0, 2, size=(10, 1)), columns=["y"])
        env = BanditEnv(features, targets)

        assert isinstance(env, BanditEnvProtocol)
        assert env.arms == 2
        assert env.num_envs == 1

        state = env.reset()
        assert isinstance(state, np.ndarray)
        next_state, reward = env.step(0)
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
