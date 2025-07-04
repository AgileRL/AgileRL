from typing import Optional, Type

import gymnasium as gym
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
    EvolvableAlgorithm,
    EvolvableModule,
    EvolvableNetwork,
    MutationMethod,
)
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multidiscrete_space,
    generate_random_box_space,
)


@pytest.fixture
def action_space(network_cls: Type[EvolvableNetwork]) -> Optional[gym.Space]:
    if issubclass(
        network_cls, (DeterministicActor, StochasticActor, ContinuousQNetwork)
    ):
        return generate_random_box_space(shape=(2,))
    elif issubclass(network_cls, (RainbowQNetwork, QNetwork)):
        return generate_discrete_space(2)
    else:
        return None


class TestProtocols:
    """Test that classes implement their respective protocols."""

    @pytest.mark.parametrize(
        "algorithm_cls, algo_action_space",
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
        self, algorithm_cls, algo_action_space, observation_space
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

        assert isinstance(instance, EvolvableAlgorithm)

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
        self, network_cls, observation_space, action_space
    ):
        """Test that network instances implement the EvolvableNetwork protocol."""
        kwargs = {}
        if issubclass(network_cls, RainbowQNetwork):
            kwargs["support"] = torch.linspace(-100, 100, 51)

        # Create network instance
        network = network_cls(observation_space, action_space, **kwargs)
        assert isinstance(network, EvolvableNetwork)

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
        assert isinstance(instance, EvolvableModule)

        mutation_methods = instance.get_mutation_methods()
        for method_name, method in mutation_methods.items():
            assert isinstance(
                method, MutationMethod
            ), f"Mutation method {method_name} does not implement MutationMethod protocol"
