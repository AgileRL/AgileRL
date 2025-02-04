import pytest
import torch.optim as optim
from gymnasium import spaces

from agilerl.algorithms.core import MultiAgentRLAlgorithm, OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    NetworkGroup,
    RLParameter,
)
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.utils.evolvable_networks import is_image_space
from tests.helper_functions import (
    gen_multi_agent_dict_or_tuple_spaces,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_random_box_space,
)


@pytest.fixture
def mlp_config():
    return {"hidden_size": [8]}


@pytest.fixture
def cnn_config():
    return {"channel_size": [3], "kernel_size": [3]}


@pytest.fixture
def multi_input_config():
    return {"hidden_size": [8], "channel_size": [3], "kernel_size": [3]}


class DummyRLAlgorithm(RLAlgorithm):
    def __init__(self, *args, lr=True, **kwargs):
        super().__init__(*args, **kwargs)

        num_outputs = (
            self.action_space.n
            if isinstance(self.action_space, spaces.Discrete)
            else self.action_space.shape[0]
        )
        if is_image_space(self.observation_space):
            self.dummy_actor = EvolvableCNN(
                self.observation_space.shape,
                num_outputs,
                channel_size=[3],
                kernel_size=[3],
                stride_size=[1],
            )
        elif isinstance(self.observation_space, (spaces.Box, spaces.Discrete)):
            num_inputs = (
                self.observation_space.shape[0]
                if isinstance(self.observation_space, spaces.Box)
                else self.observation_space.n
            )
            self.dummy_actor = EvolvableMLP(num_inputs, num_outputs, hidden_size=[8])
        elif isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            self.dummy_actor = EvolvableMultiInput(
                self.observation_space,
                num_outputs,
                hidden_size=[8],
                channel_size=[3],
                kernel_size=[3],
                stride_size=[1],
            )

        self.lr = 0.1
        self.dummy_optimizer = OptimizerWrapper(optim.Adam, self.dummy_actor, self.lr)

        self.register_network_group(NetworkGroup(eval=self.dummy_actor, policy=True))

    def get_action(self, *args, **kwargs):
        return

    def learn(self, *args, **kwargs):
        return

    def test(self, *args, **kwargs):
        return


class DummyMARLAlgorithm(MultiAgentRLAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def create_actor(idx):
            obs_space = self.observation_spaces[idx]
            action_space = self.action_spaces[idx]
            num_outputs = (
                action_space.n
                if isinstance(action_space, spaces.Discrete)
                else action_space.shape[0]
            )
            if is_image_space(obs_space):
                return EvolvableCNN(
                    obs_space.shape,
                    num_outputs,
                    channel_size=[3],
                    kernel_size=[3],
                    stride_size=[1],
                )
            elif isinstance(obs_space, (spaces.Box, spaces.Discrete)):
                num_inputs = (
                    obs_space.shape[0]
                    if isinstance(obs_space, spaces.Box)
                    else obs_space.n
                )
                return EvolvableMLP(num_inputs, num_outputs, hidden_size=[8])
            elif isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
                return EvolvableMultiInput(
                    obs_space,
                    num_outputs,
                    hidden_size=[8],
                    channel_size=[3],
                    kernel_size=[3],
                    stride_size=[1],
                )

        self.dummy_actors = [create_actor(idx) for idx in range(self.n_agents)]
        self.lr = 0.1
        self.dummy_optimizer = OptimizerWrapper(
            optim.Adam, self.dummy_actors, self.lr, multiagent=True
        )

        self.register_network_group(
            NetworkGroup(eval=self.dummy_actors, policy=True, multiagent=True)
        )

    def get_action(self, *args, **kwargs):
        return

    def learn(self, *args, **kwargs):
        return

    def test(self, *args, **kwargs):
        return


@pytest.mark.parametrize(
    "observation_space",
    [
        generate_dict_or_tuple_space(1, 2),
        generate_discrete_space(4),
        generate_random_box_space((4,)),
        # generate_multidiscrete_space(2, 2)
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        generate_discrete_space(4),
        generate_random_box_space((4,)),
        # generate_multidiscrete_space(2, 2)
    ],
)
def test_initialise_single_agent(observation_space, action_space):
    agent = DummyRLAlgorithm(observation_space, action_space, index=0)
    assert agent is not None


@pytest.mark.parametrize(
    "observation_space",
    [
        generate_multi_agent_box_spaces(2, (2,)),
        generate_multi_agent_discrete_spaces(2, 4),
        gen_multi_agent_dict_or_tuple_spaces(2, 2, 2),
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        generate_multi_agent_discrete_spaces(2, 4),
        generate_multi_agent_box_spaces(2, (2,)),
    ],
)
def test_initialise_multi_agent(observation_space, action_space):
    agent = DummyMARLAlgorithm(
        observation_space, action_space, agent_ids=["agent1", "agent2"], index=0
    )
    assert agent is not None


def test_population_single_agent():
    observation_space = generate_random_box_space((4,))
    action_space = generate_discrete_space(4)
    population = DummyRLAlgorithm.population(10, observation_space, action_space)
    assert len(population) == 10
    for i, agent in enumerate(population):
        assert agent.observation_space == observation_space
        assert agent.action_space == action_space
        assert agent.index == i


def test_population_multi_agent():
    observation_spaces = generate_multi_agent_box_spaces(2, (2,))
    action_spaces = generate_multi_agent_discrete_spaces(2, 4)
    population = DummyMARLAlgorithm.population(
        10, observation_spaces, action_spaces, agent_ids=["agent1", "agent2"]
    )
    assert len(population) == 10
    for i, agent in enumerate(population):
        for j in range(2):
            agent_id = agent.agent_ids[j]
            assert agent.observation_space[agent_id] == observation_spaces[j]
            assert agent.action_space[agent_id] == action_spaces[j]

        assert agent.index == i


def test_incorrect_hp_config():
    with pytest.raises(AttributeError):
        hp_config = HyperparameterConfig(lr_actor=RLParameter(min=0.1, max=0.2))
        agent = DummyRLAlgorithm(
            generate_random_box_space((4,)),
            generate_discrete_space(4),
            index=0,
            hp_config=hp_config,
        )
