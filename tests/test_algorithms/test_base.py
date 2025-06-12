from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim as optim
from gymnasium import spaces
from torch._dynamo.eval_frame import OptimizedModule

from agilerl.algorithms.core import MultiAgentRLAlgorithm, OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    NetworkGroup,
    RLParameter,
)
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.modules.base import ModuleDict
from agilerl.utils.evolvable_networks import is_image_space
from tests.helper_functions import (
    assert_state_dicts_equal,
    gen_multi_agent_dict_or_tuple_spaces,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multi_agent_box_spaces,
    generate_multi_agent_discrete_spaces,
    generate_multidiscrete_space,
    generate_random_box_space,
    is_processed_observation,
)


@pytest.fixture
def mlp_config():
    yield {"hidden_size": [8], "min_mlp_nodes": 8, "max_mlp_nodes": 80}


@pytest.fixture
def cnn_config():
    yield {"channel_size": [3], "kernel_size": [3], "stride_size": [1]}


@pytest.fixture
def multi_input_config():
    yield {
        "latent_dim": 64,
        "mlp_config": {"hidden_size": [8]},
        "cnn_config": {"channel_size": [3], "kernel_size": [3]},
    }


@pytest.fixture
def single_level_net_config(request):
    """Fixture for a single-level net config (one config for all agents)."""
    mlp_config = request.getfixturevalue("mlp_config")
    yield {"encoder_config": mlp_config}


@pytest.fixture
def homogeneous_group_net_config(request):
    """Fixture for a homogeneous group net config with group-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    yield {
        "agent": {"encoder_config": mlp_config},
        "other_agent": {"encoder_config": mlp_config},
    }


@pytest.fixture
def homogeneous_agent_net_config(request):
    """Fixture for a homogeneous nested net config with agent-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    yield {
        "agent_0": {"encoder_config": mlp_config},
        "other_agent_0": {"encoder_config": mlp_config},
    }


@pytest.fixture
def mixed_group_net_config(request):
    """Fixture for a mixed group net config with group-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    cnn_config = request.getfixturevalue("cnn_config")
    yield {
        "agent": {"encoder_config": mlp_config},
        "other_agent": {"encoder_config": cnn_config},
    }


@pytest.fixture
def mixed_agent_net_config(request):
    """Fixture for a mixed nested net config with agent-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    cnn_config = request.getfixturevalue("cnn_config")
    yield {
        "agent_0": {"encoder_config": mlp_config},
        "agent_1": {"encoder_config": mlp_config},
        "other_agent_0": {"encoder_config": cnn_config},
        "other_agent_1": {"encoder_config": cnn_config},
    }


@pytest.fixture
def heterogeneous_agent_net_config(request):
    """Fixture for a heterogeneous nested net config with agent-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    cnn_config = request.getfixturevalue("cnn_config")
    multi_input_config = request.getfixturevalue("multi_input_config")
    yield {
        "agent_0": {"encoder_config": mlp_config},
        "other_agent_0": {"encoder_config": cnn_config},
        "other_other_agent_0": {"encoder_config": multi_input_config},
    }


@pytest.fixture
def homogeneous_agent():
    """Fixture for a homogeneous multi-agent setup where all agents have the same observation space type."""
    # All agents have 1D Box spaces of the same shape
    obs_spaces = [
        spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
    ]
    action_spaces = [
        spaces.Discrete(2),
        spaces.Discrete(2),
    ]
    agent_ids = ["agent_0", "other_agent_0"]
    return DummyMARLAlgorithm(obs_spaces, action_spaces, agent_ids=agent_ids, index=0)


@pytest.fixture
def mixed_agent():
    """Fixture for a mixed multi-agent setup with two distinct groups."""
    # Create two groups with different observation spaces
    # Group 1: 2 agents with 1D Box spaces of same shape
    # Group 2: 2 agents with 3D Box spaces (images) of same shape
    obs_spaces = [
        spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),  # Group 1 (vector)
        spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),  # Group 1 (vector)
        spaces.Box(
            low=0, high=255, shape=(3, 32, 32), dtype=np.uint8
        ),  # Group 2 (image)
        spaces.Box(
            low=0, high=255, shape=(3, 32, 32), dtype=np.uint8
        ),  # Group 2 (image)
    ]
    action_spaces = [
        spaces.Discrete(2),
        spaces.Discrete(2),
        spaces.Discrete(2),
        spaces.Discrete(2),
    ]
    # Use agent IDs that clearly indicate group membership
    agent_ids = ["agent_0", "agent_1", "other_agent_0", "other_agent_1"]
    return DummyMARLAlgorithm(obs_spaces, action_spaces, agent_ids=agent_ids, index=0)


@pytest.fixture
def heterogeneous_agent():
    """Fixture for a heterogeneous multi-agent setup with fundamentally different observation spaces."""
    # Create four agents with different observation space types
    obs_spaces = [
        spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),  # 1D vector
        spaces.Box(low=0, high=255, shape=(3, 32, 32), dtype=np.uint8),  # 3D image
        spaces.Dict(
            {  # Dict space
                "position": spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32),
                "velocity": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            }
        ),
    ]
    action_spaces = [spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)]
    agent_ids = ["agent_0", "other_agent_0", "other_other_agent_0"]
    return DummyMARLAlgorithm(obs_spaces, action_spaces, agent_ids=agent_ids, index=0)


class DummyRLAlgorithm(RLAlgorithm):
    def __init__(self, observation_space, action_space, index, lr=True, **kwargs):
        super().__init__(observation_space, action_space, index, **kwargs)

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
        elif isinstance(self.observation_space, spaces.MultiDiscrete):
            # Handle MultiDiscrete spaces
            num_inputs = len(self.observation_space.nvec)
            self.dummy_actor = EvolvableMLP(num_inputs, num_outputs, hidden_size=[8])
        elif isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            config = {
                "mlp_config": {"hidden_size": [8]},
                "cnn_config": {
                    "channel_size": [3],
                    "kernel_size": [3],
                    "stride_size": [1],
                },
            }
            self.dummy_actor = EvolvableMultiInput(
                self.observation_space, num_outputs, **config
            )

        self.lr = 0.1
        self.dummy_optimizer = OptimizerWrapper(optim.Adam, self.dummy_actor, self.lr)
        self.dummy_attribute = "test_value"

        self.register_network_group(NetworkGroup(eval=self.dummy_actor, policy=True))

    def get_action(self, *args, **kwargs):
        return

    def learn(self, *args, **kwargs):
        return

    def test(self, *args, **kwargs):
        return


class DummyMARLAlgorithm(MultiAgentRLAlgorithm):
    def __init__(self, observation_spaces, action_spaces, agent_ids, index, **kwargs):
        super().__init__(observation_spaces, action_spaces, agent_ids, index, **kwargs)

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
                config = {
                    "mlp_config": {"hidden_size": [8]},
                    "cnn_config": {
                        "channel_size": [3],
                        "kernel_size": [3],
                        "stride_size": [1],
                    },
                }
                return EvolvableMultiInput(obs_space, num_outputs, **config)

        self.dummy_actors = ModuleDict(
            {
                agent_id: create_actor(idx)
                for idx, agent_id in enumerate(self.observation_space.keys())
            }
        )
        self.lr = 0.1
        self.dummy_optimizer = OptimizerWrapper(optim.Adam, self.dummy_actors, self.lr)

        self.register_network_group(NetworkGroup(eval=self.dummy_actors, policy=True))

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


@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space((4,)),
        generate_random_box_space((3, 32, 32)),
        generate_dict_or_tuple_space(1, 1, dict_space=True),
        generate_dict_or_tuple_space(1, 1, dict_space=False),
    ],
)
def test_preprocess_observation(observation_space):
    agent = DummyRLAlgorithm(observation_space, generate_discrete_space(4), index=0)
    observation = agent.preprocess_observation(observation_space.sample())
    assert is_processed_observation(observation, observation_space)


def test_incorrect_hp_config():
    with pytest.raises(AttributeError):
        hp_config = HyperparameterConfig(lr_actor=RLParameter(min=0.1, max=0.2))
        _ = DummyRLAlgorithm(
            generate_random_box_space((4,)),
            generate_discrete_space(4),
            index=0,
            hp_config=hp_config,
        )


def test_recompile():
    agent = DummyMARLAlgorithm(
        generate_multi_agent_box_spaces(2, (4,)),
        generate_multi_agent_discrete_spaces(2, 4),
        agent_ids=["agent1", "agent2"],
        index=0,
        torch_compiler="default",
    )
    agent.recompile()
    assert all(
        isinstance(mod, OptimizedModule) for mod in agent.dummy_actors.values()
    ), agent.dummy_actors.values()


@pytest.mark.parametrize(
    "with_hp_config",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space((4,)),
        generate_discrete_space(4),
        generate_dict_or_tuple_space(1, 1, dict_space=True),
        generate_dict_or_tuple_space(1, 1, dict_space=False),
        generate_multidiscrete_space(2, 2),
    ],
)
def test_save_load_checkpoint_single_agent(tmpdir, with_hp_config, observation_space):
    action_space = generate_discrete_space(4)
    # Initialize the dummy agent
    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))
        agent = DummyRLAlgorithm(
            observation_space, action_space, index=0, hp_config=hp_config
        )
    else:
        agent = DummyRLAlgorithm(observation_space, action_space, index=0)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path)

    # Check if the loaded checkpoint has the correct keys
    assert "dummy_actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "dummy_actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "dummy_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "lr" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Create a new agent with the same hp_config if needed
    new_agent = DummyRLAlgorithm(
        observation_space,
        action_space,
        index=1,  # Different index to verify it gets overwritten
        hp_config=hp_config,
    )

    # Load checkpoint
    new_agent.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(
        new_agent.dummy_actor, (EvolvableMLP, EvolvableCNN, EvolvableMultiInput)
    )
    assert new_agent.lr == agent.lr
    assert_state_dicts_equal(
        new_agent.dummy_actor.state_dict(), agent.dummy_actor.state_dict()
    )
    assert new_agent.index == agent.index
    assert new_agent.scores == agent.scores
    assert new_agent.fitness == agent.fitness
    assert new_agent.steps == agent.steps


@pytest.mark.parametrize(
    "with_hp_config",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (4,)),
        generate_multi_agent_discrete_spaces(2, 4),
        gen_multi_agent_dict_or_tuple_spaces(2, 1, 1, dict_space=True),
        gen_multi_agent_dict_or_tuple_spaces(2, 1, 1, dict_space=False),
    ],
)
def test_save_load_checkpoint_multi_agent(tmpdir, with_hp_config, observation_spaces):
    # Initialize the dummy multi-agent
    agent_ids = ["agent1", "agent2"]
    action_spaces = generate_multi_agent_discrete_spaces(2, 4)

    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))
        agent = DummyMARLAlgorithm(
            observation_spaces,
            action_spaces,
            agent_ids=agent_ids,
            index=0,
            hp_config=hp_config,
        )
    else:
        agent = DummyMARLAlgorithm(
            observation_spaces, action_spaces, agent_ids=agent_ids, index=0
        )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path)

    # Check if the loaded checkpoint has the correct keys
    assert "dummy_actors_init_dict" in checkpoint["network_info"]["modules"]
    assert "dummy_actors_state_dict" in checkpoint["network_info"]["modules"]
    assert "dummy_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "lr" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint
    assert "agent_ids" in checkpoint

    # Create a new agent with the same hp_config if needed
    new_agent = DummyMARLAlgorithm(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        index=1,  # Different index to verify it gets overwritten
        hp_config=hp_config,
    )

    # Load checkpoint
    new_agent.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    for agent_id in agent.agent_ids:
        assert isinstance(
            new_agent.dummy_actors[agent_id],
            (EvolvableMLP, EvolvableCNN, EvolvableMultiInput),
        )
        assert_state_dicts_equal(
            new_agent.dummy_actors[agent_id].state_dict(),
            agent.dummy_actors[agent_id].state_dict(),
        )

    assert new_agent.lr == agent.lr
    assert new_agent.index == agent.index
    assert new_agent.scores == agent.scores
    assert new_agent.fitness == agent.fitness
    assert new_agent.steps == agent.steps
    assert new_agent.agent_ids == agent.agent_ids


@pytest.mark.parametrize(
    "device, with_hp_config",
    [
        ("cpu", False),
        ("cpu", True),
    ],
)
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space((4,)),
        generate_discrete_space(4),
        generate_dict_or_tuple_space(1, 1, dict_space=True),
        generate_dict_or_tuple_space(1, 1, dict_space=False),
        generate_multidiscrete_space(2, 2),
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space((2,)),
        generate_discrete_space(4),
    ],
)
def test_load_from_pretrained_single_agent(
    device, tmpdir, with_hp_config, observation_space, action_space
):
    # Initialize the dummy agent
    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))
        agent = DummyRLAlgorithm(
            observation_space, action_space, index=0, hp_config=hp_config
        )
    else:
        agent = DummyRLAlgorithm(observation_space, action_space, index=0)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create new agent object using the class method
    new_agent = DummyRLAlgorithm.load(checkpoint_path, device=device)

    # Check if properties and weights are loaded correctly
    assert new_agent.observation_space == agent.observation_space
    assert new_agent.action_space == agent.action_space
    assert isinstance(
        new_agent.dummy_actor, (EvolvableMLP, EvolvableCNN, EvolvableMultiInput)
    )
    assert new_agent.lr == agent.lr
    assert_state_dicts_equal(
        new_agent.dummy_actor.to("cpu").state_dict(), agent.dummy_actor.state_dict()
    )
    assert new_agent.index == agent.index
    assert new_agent.scores == agent.scores
    assert new_agent.fitness == agent.fitness
    assert new_agent.steps == agent.steps


@pytest.mark.parametrize(
    "device, with_hp_config",
    [
        ("cpu", False),
        ("cpu", True),
    ],
)
@pytest.mark.parametrize(
    "observation_spaces",
    [
        generate_multi_agent_box_spaces(2, (4,)),
        generate_multi_agent_discrete_spaces(2, 4),
        gen_multi_agent_dict_or_tuple_spaces(2, 1, 1, dict_space=True),
        gen_multi_agent_dict_or_tuple_spaces(2, 1, 1, dict_space=False),
    ],
)
@pytest.mark.parametrize(
    "action_spaces",
    [
        generate_multi_agent_box_spaces(2, (2,)),
        generate_multi_agent_discrete_spaces(2, 4),
    ],
)
def test_load_from_pretrained_multi_agent(
    device, tmpdir, with_hp_config, observation_spaces, action_spaces
):
    # Initialize the dummy multi-agent
    agent_ids = ["agent1", "agent2"]

    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))
        agent = DummyMARLAlgorithm(
            observation_spaces,
            action_spaces,
            agent_ids=agent_ids,
            index=0,
            hp_config=hp_config,
        )
    else:
        agent = DummyMARLAlgorithm(
            observation_spaces, action_spaces, agent_ids=agent_ids, index=0
        )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create new agent object using the class method
    new_agent = DummyMARLAlgorithm.load(checkpoint_path, device=device)

    # Check if properties and weights are loaded correctly
    for i, agent_id in enumerate(agent_ids):
        assert (
            new_agent.observation_space[agent_id] == agent.observation_space[agent_id]
        )
        assert new_agent.action_space[agent_id] == agent.action_space[agent_id]

    for agent_id in agent.agent_ids:
        assert isinstance(
            new_agent.dummy_actors[agent_id],
            (EvolvableMLP, EvolvableCNN, EvolvableMultiInput),
        )
        assert_state_dicts_equal(
            new_agent.dummy_actors[agent_id].to("cpu").state_dict(),
            agent.dummy_actors[agent_id].state_dict(),
        )

    assert new_agent.lr == agent.lr
    assert new_agent.index == agent.index
    assert new_agent.scores == agent.scores
    assert new_agent.fitness == agent.fitness
    assert new_agent.steps == agent.steps
    assert new_agent.agent_ids == agent.agent_ids


@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space((4,)),
    ],
)
def test_missing_attribute_warning(tmpdir, observation_space):
    action_space = generate_discrete_space(4)
    # Initialize the dummy agent
    agent = DummyRLAlgorithm(observation_space, action_space, index=0)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Load and modify the checkpoint to remove an attribute
    checkpoint = torch.load(checkpoint_path)
    checkpoint.pop("dummy_attribute")

    # Save the modified checkpoint
    modified_path = Path(tmpdir) / "modified_checkpoint.pth"
    torch.save(checkpoint, modified_path)

    # Load the modified checkpoint and check if a warning is raised
    with pytest.warns(
        UserWarning, match="Attribute dummy_attribute not found in checkpoint"
    ):
        new_agent = DummyRLAlgorithm.load(modified_path, device="cpu")

    # The attribute should keep its original value
    assert new_agent.dummy_attribute == "test_value"


@pytest.mark.parametrize("flatten", [True, False])
def test_build_net_config_homogeneous_single_level(
    homogeneous_agent, single_level_net_config, flatten
):
    """Test build_net_config with homogeneous setup and single-level config."""
    result = homogeneous_agent.build_net_config(
        single_level_net_config, flatten=flatten
    )

    # Should have entries for all agents
    assert set(result.keys()) == set(homogeneous_agent.agent_ids)

    # All agents should have the same config
    for agent_id in homogeneous_agent.agent_ids:
        assert "encoder_config" in result[agent_id]
        assert result[agent_id]["encoder_config"]["hidden_size"] == [8]
        assert result[agent_id]["encoder_config"]["min_mlp_nodes"] == 8
        assert result[agent_id]["encoder_config"]["max_mlp_nodes"] == 80


@pytest.mark.parametrize("flatten", [True, False])
def test_build_net_config_homogeneous_group_level(
    homogeneous_agent, homogeneous_group_net_config, flatten
):
    """Test build_net_config with homogeneous setup and group-level config."""
    result = homogeneous_agent.build_net_config(
        homogeneous_group_net_config, flatten=flatten
    )

    # Should have entries for all agents
    assert set(result.keys()) == set(homogeneous_agent.agent_ids)

    # Check if each agent has a config based on its group
    for agent_id in homogeneous_agent.agent_ids:
        assert "encoder_config" in result[agent_id]
        # Since no group matches exactly, all should get default configs
        assert "hidden_size" in result[agent_id]["encoder_config"]


@pytest.mark.parametrize("flatten", [True, False])
def test_build_net_config_homogeneous_agent_level(
    homogeneous_agent, homogeneous_agent_net_config, flatten
):
    """Test build_net_config with homogeneous setup and agent-level config."""
    result = homogeneous_agent.build_net_config(
        homogeneous_agent_net_config, flatten=flatten
    )

    # Should have entries for all agents
    assert set(result.keys()) == set(homogeneous_agent.agent_ids)

    # Each agent should have its specified config
    for agent_id in homogeneous_agent.agent_ids:
        assert "encoder_config" in result[agent_id]
        assert (
            result[agent_id]["encoder_config"]["hidden_size"]
            == homogeneous_agent_net_config[agent_id]["encoder_config"]["hidden_size"]
        )


def test_build_net_config_mixed_single_level(mixed_agent, single_level_net_config):
    """Test build_net_config with mixed setup and single-level config."""
    # This should raise an assertion error because we can't use single-level config with non-homogeneous setup
    with pytest.raises(AssertionError):
        mixed_agent.build_net_config(single_level_net_config)


@pytest.mark.parametrize("flatten", [True, False])
def test_build_net_config_mixed_group_level(
    mixed_agent, mixed_group_net_config, flatten
):
    """Test build_net_config with mixed setup and group-level config."""
    result = mixed_agent.build_net_config(mixed_group_net_config, flatten=flatten)

    # Should have entries for all agents
    agent_ids = mixed_agent.shared_agent_ids if not flatten else mixed_agent.agent_ids
    assert set(result.keys()) == set(agent_ids)

    # Check if each agent has a config based on its group
    for agent_id in agent_ids:
        group_id = mixed_agent.get_group_id(agent_id) if flatten else agent_id
        assert "encoder_config" in result[agent_id]
        assert (
            result[agent_id]["encoder_config"]
            == mixed_group_net_config[group_id]["encoder_config"]
        )


@pytest.mark.parametrize("flatten", [True, False])
def test_build_net_config_mixed_agent_level(
    mixed_agent, mixed_agent_net_config, flatten
):
    """Test build_net_config with mixed setup and agent-level config."""
    if not flatten:
        with pytest.raises(KeyError):
            mixed_agent.build_net_config(mixed_agent_net_config, flatten=flatten)
    else:
        result = mixed_agent.build_net_config(mixed_agent_net_config, flatten=flatten)

        # Should have entries for all agents
        assert set(result.keys()) == set(mixed_agent.agent_ids)

        # Each agent should have configs as specified, or defaults if not specified
        for agent_id in mixed_agent.agent_ids:
            assert "encoder_config" in result[agent_id]
            assert (
                result[agent_id]["encoder_config"]
                == mixed_agent_net_config[agent_id]["encoder_config"]
            )


def test_build_net_config_heterogeneous_single_level(
    heterogeneous_agent, single_level_net_config
):
    """Test build_net_config with heterogeneous setup and single-level config."""
    # This should raise an assertion error because we can't use single-level config with non-homogeneous setup
    with pytest.raises(AssertionError):
        heterogeneous_agent.build_net_config(single_level_net_config)


def test_build_net_config_heterogeneous_agent_level(
    heterogeneous_agent, heterogeneous_agent_net_config
):
    """Test build_net_config with heterogeneous setup and agent-level config."""
    result = heterogeneous_agent.build_net_config(heterogeneous_agent_net_config)

    # Should have entries for all agents
    assert set(result.keys()) == set(heterogeneous_agent.agent_ids)

    # Each agent should have configs as specified
    for agent_id in heterogeneous_agent.agent_ids:
        assert "encoder_config" in result[agent_id]
        assert (
            result[agent_id]["encoder_config"]
            == heterogeneous_agent_net_config[agent_id]["encoder_config"]
        )


@pytest.mark.parametrize(
    "setup",
    [
        "homogeneous_agent",
        "heterogeneous_agent",
        "mixed_agent",
    ],
)
def test_build_net_config_return_encoders(request, setup):
    """Test that build_net_config returns unique configs when requested."""
    agent = request.getfixturevalue(setup)

    if setup == "homogeneous_agent":
        setup_net_config = request.getfixturevalue("homogeneous_agent_net_config")
    elif setup == "heterogeneous_agent":
        setup_net_config = request.getfixturevalue("heterogeneous_agent_net_config")
    elif setup == "mixed_agent":
        setup_net_config = request.getfixturevalue("mixed_agent_net_config")

    result, unique_configs = agent.build_net_config(
        setup_net_config, return_encoders=True
    )

    # Check that the result contains configs for all agents
    assert set(result.keys()) == set(agent.agent_ids)

    # Check that unique_configs contains at least one mlp_config
    assert "mlp_config" in unique_configs
    if setup == "heterogeneous_agent":
        assert "other_agent_0" in unique_configs
        assert "other_other_agent_0" in unique_configs
    elif setup == "mixed_agent":
        assert "other_agent_0" in unique_configs


@pytest.mark.parametrize("flatten", [True, False])
def test_build_net_config_grouped_agents(mixed_agent, mixed_group_net_config, flatten):
    """Test build_net_config with grouped_agents=True."""
    result = mixed_agent.build_net_config(mixed_group_net_config, flatten=flatten)

    # Should have entries for shared agent IDs only, not individual agents
    agent_ids = mixed_agent.shared_agent_ids if not flatten else mixed_agent.agent_ids
    assert set(result.keys()) == set(agent_ids)

    # Each group should have its specified config or default
    for agent_id in agent_ids:
        print(agent_id)
        group_id = mixed_agent.get_group_id(agent_id) if flatten else agent_id
        assert "encoder_config" in result[agent_id]
        assert (
            result[agent_id]["encoder_config"]
            == mixed_group_net_config[group_id]["encoder_config"]
        )


@pytest.mark.parametrize("flatten", [True, False])
def test_build_net_config_none(homogeneous_agent, flatten):
    """Test build_net_config with None input."""
    result = homogeneous_agent.build_net_config(None, flatten=flatten)

    # Should have entries for all agents with default configs
    assert set(result.keys()) == set(homogeneous_agent.agent_ids)

    for agent_id in homogeneous_agent.agent_ids:
        assert "encoder_config" in result[agent_id]
        assert "hidden_size" in result[agent_id]["encoder_config"]
