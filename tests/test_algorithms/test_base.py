from pathlib import Path

import numpy as np
import pytest
import torch
import torch.optim as optim
from accelerate import Accelerator
from gymnasium import spaces
from torch._dynamo.eval_frame import OptimizedModule

from agilerl.algorithms.core import MultiAgentRLAlgorithm, OptimizerWrapper, RLAlgorithm
from agilerl.algorithms.core.registry import (
    HyperparameterConfig,
    NetworkGroup,
    RLParameter,
)
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput, ModuleDict
from agilerl.utils.evolvable_networks import is_image_space
from tests.helper_functions import assert_state_dicts_equal, is_processed_observation


@pytest.fixture(scope="module")
def mlp_config():
    return {"hidden_size": [8], "min_mlp_nodes": 8, "max_mlp_nodes": 80}


@pytest.fixture(scope="module")
def cnn_config():
    return {"channel_size": [3], "kernel_size": [3], "stride_size": [1]}


@pytest.fixture(scope="module")
def multi_input_config():
    return {
        "latent_dim": 64,
        "mlp_config": {"hidden_size": [8]},
        "cnn_config": {"channel_size": [3], "kernel_size": [3]},
    }


@pytest.fixture(scope="module")
def single_level_net_config(request):
    """Fixture for a single-level net config (one config for all agents)."""
    mlp_config = request.getfixturevalue("mlp_config")
    return {"encoder_config": mlp_config}


@pytest.fixture(scope="module")
def homogeneous_group_net_config(request):
    """Fixture for a homogeneous group net config with group-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    return {
        "agent": {"encoder_config": mlp_config},
        "other_agent": {"encoder_config": mlp_config},
    }


@pytest.fixture(scope="module")
def homogeneous_agent_net_config(request):
    """Fixture for a homogeneous nested net config with agent-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    return {
        "agent_0": {"encoder_config": mlp_config},
        "other_agent_0": {"encoder_config": mlp_config},
    }


@pytest.fixture(scope="module")
def mixed_group_net_config(request):
    """Fixture for a mixed group net config with group-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    cnn_config = request.getfixturevalue("cnn_config")
    return {
        "agent": {"encoder_config": mlp_config},
        "other_agent": {"encoder_config": cnn_config},
    }


@pytest.fixture(scope="module")
def mixed_agent_net_config(request):
    """Fixture for a mixed nested net config with agent-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    cnn_config = request.getfixturevalue("cnn_config")
    return {
        "agent_0": {"encoder_config": mlp_config},
        "agent_1": {"encoder_config": mlp_config},
        "other_agent_0": {"encoder_config": cnn_config},
        "other_agent_1": {"encoder_config": cnn_config},
    }


@pytest.fixture(scope="module")
def heterogeneous_agent_net_config(request):
    """Fixture for a heterogeneous nested net config with agent-specific configs."""
    mlp_config = request.getfixturevalue("mlp_config")
    cnn_config = request.getfixturevalue("cnn_config")
    multi_input_config = request.getfixturevalue("multi_input_config")
    return {
        "agent_0": {"encoder_config": mlp_config},
        "other_agent_0": {"encoder_config": cnn_config},
        "other_other_agent_0": {"encoder_config": multi_input_config},
    }


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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
    def __init__(
        self, observation_space, action_space, index, lr=True, device="cpu", **kwargs
    ):
        kwargs.pop("wrap", None)
        super().__init__(
            observation_space, action_space, index, device=device, **kwargs
        )

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
                device=self.device,
            )
        elif isinstance(self.observation_space, (spaces.Box, spaces.Discrete)):
            num_inputs = (
                self.observation_space.shape[0]
                if isinstance(self.observation_space, spaces.Box)
                else self.observation_space.n
            )
            self.dummy_actor = EvolvableMLP(
                num_inputs, num_outputs, hidden_size=[8], device=self.device
            )
        elif isinstance(self.observation_space, spaces.MultiDiscrete):
            # Handle MultiDiscrete spaces
            num_inputs = len(self.observation_space.nvec)
            self.dummy_actor = EvolvableMLP(
                num_inputs, num_outputs, hidden_size=[8], device=self.device
            )
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
                self.observation_space, num_outputs, **config, device=self.device
            )

        self.lr = 0.1
        self.dummy_optimizer = OptimizerWrapper(optim.Adam, self.dummy_actor, self.lr)
        self.dummy_attribute = "test_value"

        self.register_network_group(
            NetworkGroup(eval_network=self.dummy_actor, policy=True)
        )

        if self.accelerator is not None:
            self.wrap_models()

    def get_action(self, *args, **kwargs):
        return

    def learn(self, *args, **kwargs):
        return

    def test(self, *args, **kwargs):
        return


class DummyMARLAlgorithm(MultiAgentRLAlgorithm):
    def __init__(
        self,
        observation_spaces,
        action_spaces,
        agent_ids,
        index,
        device="cpu",
        **kwargs,
    ):
        kwargs.pop("wrap", None)
        super().__init__(
            observation_spaces, action_spaces, index, agent_ids, device=device, **kwargs
        )

        def create_actor(idx):
            obs_space = self.possible_observation_spaces[self.agent_ids[idx]]
            action_space = self.possible_action_spaces[self.agent_ids[idx]]
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
                    device=self.device,
                )
            elif isinstance(obs_space, (spaces.Box, spaces.Discrete)):
                num_inputs = (
                    obs_space.shape[0]
                    if isinstance(obs_space, spaces.Box)
                    else obs_space.n
                )
                return EvolvableMLP(
                    num_inputs, num_outputs, hidden_size=[8], device=self.device
                )
            elif isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
                config = {
                    "mlp_config": {"hidden_size": [8]},
                    "cnn_config": {
                        "channel_size": [3],
                        "kernel_size": [3],
                        "stride_size": [1],
                    },
                }
                return EvolvableMultiInput(
                    obs_space, num_outputs, **config, device=self.device
                )

        self.dummy_actors = ModuleDict(
            {
                agent_id: create_actor(idx)
                for idx, agent_id in enumerate(self.possible_observation_spaces.keys())
            }
        )
        self.lr = 0.1
        self.dummy_optimizer = OptimizerWrapper(optim.Adam, self.dummy_actors, self.lr)

        self.register_network_group(
            NetworkGroup(eval_network=self.dummy_actors, policy=True)
        )

        if self.accelerator is not None:
            self.wrap_models()
        elif self.torch_compiler:
            self.recompile()

    def get_action(self, *args, **kwargs):
        return

    def learn(self, *args, **kwargs):
        return

    def test(self, *args, **kwargs):
        return


@pytest.mark.parametrize(
    "observation_space",
    ["dict_space", "discrete_space", "vector_space", "multidiscrete_space"],
)
@pytest.mark.parametrize(
    "action_space",
    ["discrete_space", "vector_space", "multidiscrete_space"],
)
def test_initialise_single_agent(observation_space, action_space, request):
    obs_space = request.getfixturevalue(observation_space)
    act_space = request.getfixturevalue(action_space)
    agent = DummyRLAlgorithm(obs_space, act_space, index=0)
    assert agent is not None


@pytest.mark.parametrize(
    "observation_space",
    [
        "ma_vector_space",
        "ma_image_space",
        "ma_discrete_space",
        "ma_dict_space",
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        "ma_discrete_space",
        "ma_vector_space",
    ],
)
@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "agent_2"], None])
def test_initialise_multi_agent(observation_space, action_space, agent_ids, request):
    obs_spaces = request.getfixturevalue(observation_space)
    act_spaces = request.getfixturevalue(action_space)

    if agent_ids is None:
        obs_spaces = {f"agent_{i}": obs_spaces[i] for i in range(len(obs_spaces))}
        act_spaces = {f"agent_{i}": act_spaces[i] for i in range(len(act_spaces))}

    agent = DummyMARLAlgorithm(obs_spaces, act_spaces, agent_ids=agent_ids, index=0)
    assert agent is not None


def test_population_single_agent(vector_space, discrete_space):
    population = DummyRLAlgorithm.population(10, vector_space, discrete_space)
    assert len(population) == 10
    for i, agent in enumerate(population):
        assert agent.observation_space == vector_space
        assert agent.action_space == discrete_space
        assert agent.index == i


def test_population_multi_agent(ma_vector_space, ma_discrete_space):
    population = DummyMARLAlgorithm.population(
        10,
        ma_vector_space,
        ma_discrete_space,
        agent_ids=["agent_0", "agent_1", "agent_2"],
    )
    assert len(population) == 10
    for i, agent in enumerate(population):
        for j in range(len(agent.agent_ids)):
            agent_id = agent.agent_ids[j]
            assert agent.possible_observation_spaces[agent_id] == ma_vector_space[j]
            assert agent.possible_action_spaces[agent_id] == ma_discrete_space[j]

        assert agent.index == i


@pytest.mark.parametrize(
    "observation_space", ["vector_space", "image_space", "dict_space"]
)
def test_preprocess_observation(observation_space, discrete_space, request):
    obs_space = request.getfixturevalue(observation_space)
    agent = DummyRLAlgorithm(obs_space, discrete_space, index=0)
    observation = agent.preprocess_observation(obs_space.sample())
    assert is_processed_observation(observation, obs_space)


def test_reinit_optimizers_single_agent(vector_space, discrete_space):
    agent = DummyRLAlgorithm(vector_space, discrete_space, index=0)
    clone_agent = agent.clone()
    clone_agent.reinit_optimizers()
    opt_attr = clone_agent.registry.optimizers[0].name
    new_opt = getattr(clone_agent, opt_attr)
    old_opt = getattr(agent, opt_attr)

    assert_state_dicts_equal(new_opt.state_dict(), old_opt.state_dict())


def test_reinit_optimizers_multi_agent(ma_vector_space, ma_discrete_space):
    agent = DummyMARLAlgorithm(
        ma_vector_space,
        ma_discrete_space,
        index=0,
        agent_ids=["agent_0", "agent_1", "agent_2"],
    )
    clone_agent = agent.clone()
    clone_agent.reinit_optimizers()
    opt_attr = clone_agent.registry.optimizers[0].name
    new_opt = getattr(clone_agent, opt_attr)
    old_opt = getattr(agent, opt_attr)

    assert_state_dicts_equal(new_opt.state_dict(), old_opt.state_dict())


def test_incorrect_hp_config(vector_space, discrete_space):
    with pytest.raises(AttributeError):
        hp_config = HyperparameterConfig(lr_actor=RLParameter(min=0.1, max=0.2))
        _ = DummyRLAlgorithm(
            vector_space,
            discrete_space,
            index=0,
            hp_config=hp_config,
        )


def test_recompile(ma_vector_space, ma_discrete_space):
    agent = DummyMARLAlgorithm(
        ma_vector_space,
        ma_discrete_space,
        agent_ids=["agent_0", "agent_1", "agent_2"],
        index=0,
        torch_compiler="default",
    )
    agent.recompile()
    assert all(
        isinstance(mod, OptimizedModule) for mod in agent.dummy_actors.values()
    ), agent.dummy_actors.values()

    # Reset torch compilation state to prevent affecting subsequent tests
    torch._dynamo.reset()


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_unwrap_models_multi_agent(compile_mode, ma_vector_space, ma_discrete_space):
    accelerator = Accelerator()
    agent = DummyMARLAlgorithm(
        ma_vector_space,
        ma_discrete_space,
        index=0,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )

    agent.unwrap_models()

    for _, actor in agent.dummy_actors.items():
        assert isinstance(actor, torch.nn.Module)

    # Reset torch compilation state if compilation was used
    if compile_mode is not None:
        torch._dynamo.reset()


def test_unwrap_models_single_agent(vector_space, discrete_space):
    accelerator = Accelerator()
    agent = DummyRLAlgorithm(
        vector_space, discrete_space, index=0, accelerator=accelerator
    )
    agent.unwrap_models()
    assert isinstance(agent.dummy_actor, torch.nn.Module)


@pytest.mark.parametrize("with_hp_config", [False, True])
@pytest.mark.parametrize(
    "observation_space",
    ["vector_space", "discrete_space", "dict_space", "multidiscrete_space"],
)
def test_save_load_checkpoint_single_agent(
    tmpdir, with_hp_config, observation_space, discrete_space, request
):
    obs_space = request.getfixturevalue(observation_space)
    action_space = discrete_space
    # Initialize the dummy agent
    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))
        agent = DummyRLAlgorithm(obs_space, action_space, index=0, hp_config=hp_config)
    else:
        agent = DummyRLAlgorithm(obs_space, action_space, index=0)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, weights_only=False)

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
        obs_space,
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


@pytest.mark.parametrize("with_hp_config", [False, True])
@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        ("ma_vector_space", EvolvableMLP),
        ("ma_image_space", EvolvableCNN),
        ("ma_dict_space", EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None])
def test_save_load_checkpoint_multi_agent(
    tmpdir,
    with_hp_config,
    observation_spaces,
    encoder_cls,
    ma_discrete_space,
    accelerator,
    compile_mode,
    request,
):
    # Initialize the dummy multi-agent
    obs_spaces = request.getfixturevalue(observation_spaces)
    agent_ids = ["agent_0", "agent_1", "agent_2"]
    action_spaces = ma_discrete_space

    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))

    agent = DummyMARLAlgorithm(
        obs_spaces,
        action_spaces,
        agent_ids=agent_ids,
        index=0,
        hp_config=hp_config,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, weights_only=False)

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
        obs_spaces,
        action_spaces,
        agent_ids=agent_ids,
        index=1,  # Different index to verify it gets overwritten
        hp_config=hp_config,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )

    # Load checkpoint
    new_agent.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    for agent_id in agent.agent_ids:
        if compile_mode is not None and accelerator is None:
            assert isinstance(new_agent.dummy_actors[agent_id], OptimizedModule)
        else:
            assert isinstance(new_agent.dummy_actors[agent_id], encoder_cls)

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

    del new_agent


@pytest.mark.parametrize("device, with_hp_config", [("cpu", False), ("cpu", True)])
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        ("vector_space", EvolvableMLP),
        ("discrete_space", EvolvableMLP),
        ("dict_space", EvolvableMultiInput),
        ("multidiscrete_space", EvolvableMLP),
    ],
)
@pytest.mark.parametrize("action_space", ["vector_space", "discrete_space"])
def test_load_from_pretrained_single_agent(
    device,
    tmpdir,
    with_hp_config,
    observation_space,
    encoder_cls,
    action_space,
    request,
):
    # Initialize the dummy agent
    obs_space = request.getfixturevalue(observation_space)
    act_space = request.getfixturevalue(action_space)
    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))
        agent = DummyRLAlgorithm(obs_space, act_space, index=0, hp_config=hp_config)
    else:
        agent = DummyRLAlgorithm(obs_space, act_space, index=0)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create new agent object using the class method
    new_agent = DummyRLAlgorithm.load(checkpoint_path, device=device)

    # Check if properties and weights are loaded correctly
    assert new_agent.observation_space == agent.observation_space
    assert new_agent.action_space == agent.action_space
    assert isinstance(new_agent.dummy_actor, encoder_cls)
    assert new_agent.lr == agent.lr
    assert_state_dicts_equal(
        new_agent.dummy_actor.to("cpu").state_dict(), agent.dummy_actor.state_dict()
    )
    assert new_agent.index == agent.index
    assert new_agent.scores == agent.scores
    assert new_agent.fitness == agent.fitness
    assert new_agent.steps == agent.steps


@pytest.mark.parametrize("device, with_hp_config", [("cpu", False), ("cpu", True)])
@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        ("ma_vector_space", EvolvableMLP),
        ("ma_image_space", EvolvableCNN),
        ("ma_discrete_space", EvolvableMLP),
        ("ma_dict_space", EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("action_spaces", ["ma_vector_space", "ma_discrete_space"])
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None])
def test_load_from_pretrained_multi_agent(
    device,
    tmpdir,
    with_hp_config,
    observation_spaces,
    encoder_cls,
    action_spaces,
    accelerator,
    compile_mode,
    request,
):
    # Initialize the dummy multi-agent
    obs_spaces = request.getfixturevalue(observation_spaces)
    act_spaces = request.getfixturevalue(action_spaces)
    agent_ids = ["agent_0", "agent_1", "agent_2"]

    hp_config = None
    if with_hp_config:
        hp_config = HyperparameterConfig(lr=RLParameter(min=0.05, max=0.2))

    agent = DummyMARLAlgorithm(
        obs_spaces,
        act_spaces,
        agent_ids=agent_ids,
        index=0,
        hp_config=hp_config,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create new agent object using the class method
    new_agent = DummyMARLAlgorithm.load(
        checkpoint_path, device=device, accelerator=accelerator
    )

    # Check if properties and weights are loaded correctly
    for i, agent_id in enumerate(agent_ids):
        assert (
            new_agent.possible_observation_spaces[agent_id]
            == agent.possible_observation_spaces[agent_id]
        )
        assert (
            new_agent.possible_action_spaces[agent_id]
            == agent.possible_action_spaces[agent_id]
        )

    for agent_id in agent.agent_ids:
        if compile_mode is not None and accelerator is None:
            assert isinstance(new_agent.dummy_actors[agent_id], OptimizedModule)
        else:
            assert isinstance(new_agent.dummy_actors[agent_id], encoder_cls)

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

    del new_agent


def test_gpu_to_no_cuda_transfer_single_agent(tmpdir, vector_space):
    """Test saving agent on GPU and loading when CUDA is completely unavailable."""
    import os
    import subprocess
    import sys

    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU to no-CUDA transfer test")

    # Initialize the dummy agent
    observation_space = vector_space
    action_space = vector_space
    agent = DummyRLAlgorithm(observation_space, action_space, index=0, device="cuda")

    # Verify agent is on GPU
    assert next(agent.dummy_actor.parameters()).device.type == "cuda"

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create a subprocess that runs with CUDA_VISIBLE_DEVICES="" to simulate no GPU environment
    test_script = f"""
import sys
sys.path.insert(0, "{os.getcwd()}")
import torch
from pathlib import Path
from tests.test_algorithms.test_base import DummyRLAlgorithm

# Verify CUDA is not available in this subprocess
assert not torch.cuda.is_available(), "CUDA should not be available"

# Load the checkpoint (should work even though it was saved on GPU)
checkpoint_path = Path("{checkpoint_path}")
new_agent = DummyRLAlgorithm.load(checkpoint_path, device="cpu")

# Verify the agent is on CPU
assert next(new_agent.dummy_actor.parameters()).device.type == "cpu"

print("SUCCESS: GPU-saved checkpoint loaded successfully in no-CUDA environment")
    """

    script_path = Path(tmpdir) / "test_no_cuda.py"
    with open(script_path, "w") as f:
        f.write(test_script)

    # Run the test script with CUDA_VISIBLE_DEVICES="" to hide GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [sys.executable, str(script_path)], env=env, capture_output=True, text=True
    )

    # Check that the subprocess succeeded
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_gpu_to_no_cuda_load_checkpoint_single_agent(tmpdir, vector_space):
    """Test saving agent on GPU and loading checkpoint when CUDA is completely unavailable."""
    import os
    import subprocess
    import sys

    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU to no-CUDA load_checkpoint test")

    # Initialize the dummy agent
    observation_space = vector_space
    action_space = vector_space
    agent = DummyRLAlgorithm(observation_space, action_space, index=0, device="cuda")

    # Verify agent is on GPU
    assert next(agent.dummy_actor.parameters()).device.type == "cuda"

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create a subprocess that runs with CUDA_VISIBLE_DEVICES="" to simulate no GPU environment
    # Serialize observation and action spaces
    obs_space_init = "generate_random_box_space((4,))"
    action_space_init = "generate_random_box_space((4,))"

    test_script = f"""
import sys
sys.path.insert(0, "{os.getcwd()}")
import torch
import numpy as np
from pathlib import Path
from gymnasium import spaces
from tests.test_algorithms.test_base import DummyRLAlgorithm
from tests.helper_functions import generate_random_box_space

# Verify CUDA is not available in this subprocess
assert not torch.cuda.is_available(), "CUDA should not be available"

# Create observation and action spaces
observation_space = {obs_space_init}
action_space = {action_space_init}

# Create a new agent in no-CUDA environment
new_agent = DummyRLAlgorithm(observation_space, action_space, index=1)

# Load checkpoint using instance method
checkpoint_path = Path("{checkpoint_path}")
new_agent.load_checkpoint(checkpoint_path)

# Verify the agent is on CPU
assert next(new_agent.dummy_actor.parameters()).device.type == "cpu"

print("SUCCESS: GPU-saved checkpoint loaded via load_checkpoint in no-CUDA environment")
    """

    script_path = Path(tmpdir) / "test_load_checkpoint_no_cuda.py"
    with open(script_path, "w") as f:
        f.write(test_script)

    # Run the test script with CUDA_VISIBLE_DEVICES="" to hide GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [sys.executable, str(script_path)], env=env, capture_output=True, text=True
    )

    # Check that the subprocess succeeded
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_gpu_to_no_cuda_transfer_multi_agent(tmpdir, ma_vector_space):
    """Test saving multi-agent on GPU and loading when CUDA is completely unavailable."""
    import os
    import subprocess
    import sys

    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU to no-CUDA transfer test")

    # Initialize the dummy multi-agent
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    observation_spaces = ma_vector_space
    action_spaces = ma_vector_space
    agent = DummyMARLAlgorithm(
        observation_spaces, action_spaces, agent_ids=agent_ids, index=0, device="cuda"
    )

    # Verify agents are on GPU
    for actor in agent.dummy_actors.values():
        assert next(actor.parameters()).device.type == "cuda"

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create a subprocess that runs with CUDA_VISIBLE_DEVICES="" to simulate no GPU environment
    test_script = f"""
import sys
sys.path.insert(0, "{os.getcwd()}")
import torch
from pathlib import Path
from tests.test_algorithms.test_base import DummyMARLAlgorithm

# Verify CUDA is not available in this subprocess
assert not torch.cuda.is_available(), "CUDA should not be available"

# Load the checkpoint (should work even though it was saved on GPU)
checkpoint_path = Path("{checkpoint_path}")
new_agent = DummyMARLAlgorithm.load(checkpoint_path, device="cpu")

# Verify all agents are on CPU
agent_ids = ["agent_0", "agent_1", "other_agent_0"]
for actor in new_agent.dummy_actors.values():
    assert next(actor.parameters()).device.type == "cpu"

print("SUCCESS: GPU-saved multi-agent checkpoint loaded successfully in no-CUDA environment")
    """

    script_path = Path(tmpdir) / "test_no_cuda_multi.py"
    with open(script_path, "w") as f:
        f.write(test_script)

    # Run the test script with CUDA_VISIBLE_DEVICES="" to hide GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [sys.executable, str(script_path)], env=env, capture_output=True, text=True
    )

    # Check that the subprocess succeeded
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_gpu_to_no_cuda_load_checkpoint_multi_agent(tmpdir, ma_vector_space):
    """Test saving multi-agent on GPU and loading checkpoint when CUDA is completely unavailable."""
    import os
    import subprocess
    import sys

    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU to no-CUDA load_checkpoint test")

    # Initialize the dummy multi-agent
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    observation_spaces = ma_vector_space
    action_spaces = ma_vector_space
    agent = DummyMARLAlgorithm(
        observation_spaces, action_spaces, agent_ids=agent_ids, index=0, device="cuda"
    )

    # Verify agents are on GPU
    for actor in agent.dummy_actors.values():
        assert next(actor.parameters()).device.type == "cuda"

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Create a subprocess that runs with CUDA_VISIBLE_DEVICES="" to simulate no GPU environment
    obs_spaces_init = "generate_multi_agent_box_spaces(3, (4,))"
    action_spaces_init = "generate_multi_agent_box_spaces(3, (2,))"

    test_script = f"""
import sys
sys.path.insert(0, "{os.getcwd()}")
import torch
import numpy as np
from pathlib import Path
from gymnasium import spaces
from tests.test_algorithms.test_base import DummyMARLAlgorithm
from tests.helper_functions import generate_multi_agent_box_spaces

# Verify CUDA is not available in this subprocess
assert not torch.cuda.is_available(), "CUDA should not be available"

# Create observation and action spaces
observation_spaces = {obs_spaces_init}
action_spaces = {action_spaces_init}

# Create a new multi-agent in no-CUDA environment
agent_ids = ["agent_0", "agent_1", "other_agent_0"]
new_agent = DummyMARLAlgorithm(observation_spaces, action_spaces, agent_ids=agent_ids, index=1)

# Load checkpoint using instance method
checkpoint_path = Path("{checkpoint_path}")
new_agent.load_checkpoint(checkpoint_path)

# Verify all agents are on CPU
for actor in new_agent.dummy_actors.values():
    assert next(actor.parameters()).device.type == "cpu"

print("SUCCESS: GPU-saved multi-agent checkpoint loaded via load_checkpoint in no-CUDA environment")
    """

    script_path = Path(tmpdir) / "test_load_checkpoint_no_cuda_multi.py"
    with open(script_path, "w") as f:
        f.write(test_script)

    # Run the test script with CUDA_VISIBLE_DEVICES="" to hide GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    result = subprocess.run(
        [sys.executable, str(script_path)], env=env, capture_output=True, text=True
    )

    # Check that the subprocess succeeded
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_missing_attribute_warning(tmpdir, vector_space):
    action_space = vector_space
    # Initialize the dummy agent
    agent = DummyRLAlgorithm(vector_space, action_space, index=0)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    agent.save_checkpoint(checkpoint_path)

    # Load and modify the checkpoint to remove an attribute
    checkpoint = torch.load(checkpoint_path, weights_only=False)
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
