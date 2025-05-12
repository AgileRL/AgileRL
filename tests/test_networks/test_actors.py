import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.networks.actors import DeterministicActor, StochasticActor
from agilerl.networks.base import EvolvableNetwork
from tests.helper_functions import (
    assert_close_dict,
    check_equal_params_ind,
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_random_box_space,
)


@pytest.mark.parametrize("action_space", [generate_random_box_space((4,))])
@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        (generate_dict_or_tuple_space(2, 3), "multi_input"),
        (generate_discrete_space(4), "mlp"),
        (generate_random_box_space((8,)), "mlp"),
        (generate_random_box_space((3, 32, 32)), "cnn"),
    ],
)
def test_deterministic_actor_initialization(
    observation_space, action_space, encoder_type
):
    network = DeterministicActor(observation_space, action_space)

    assert network.observation_space == observation_space

    if encoder_type == "multi_input":
        assert isinstance(network.encoder, EvolvableMultiInput)
    elif encoder_type == "mlp":
        assert isinstance(network.encoder, EvolvableMLP)
    elif encoder_type == "cnn":
        assert isinstance(network.encoder, EvolvableCNN)

    evolvable_modules = network.modules()
    assert "encoder" in evolvable_modules
    assert "head_net" in evolvable_modules


@pytest.mark.parametrize("action_space", [generate_random_box_space((4,))])
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_dict_or_tuple_space(2, 3),
        generate_discrete_space(4),
        generate_random_box_space((8,)),
        generate_random_box_space((3, 32, 32)),
    ],
)
def test_deterministic_actor_mutation_methods(observation_space, action_space):
    network = DeterministicActor(observation_space, action_space)

    for method in network.mutation_methods:
        new_network = network.clone()
        getattr(new_network, method)()

        if "." in method:
            net_name = method.split(".")[0]
            mutated_module: EvolvableModule = getattr(new_network, net_name)
            exec_method = new_network.last_mutation_attr.split(".")[-1]

            if isinstance(observation_space, (spaces.Tuple, spaces.Dict)):
                mutated_attr = mutated_module.last_mutation_attr.split(".")[-1]
            else:
                mutated_attr = mutated_module.last_mutation_attr

            assert mutated_attr == exec_method

        check_equal_params_ind(network, new_network)


@pytest.mark.parametrize("action_space", [generate_random_box_space((4,))])
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_dict_or_tuple_space(2, 3),
        generate_discrete_space(4),
        generate_random_box_space((8,)),
        generate_random_box_space((3, 32, 32)),
    ],
)
def test_deterministic_actor_forward(
    observation_space: spaces.Space, action_space: spaces.Space
):
    network = DeterministicActor(observation_space, action_space)

    x_np = observation_space.sample()

    if isinstance(observation_space, spaces.Discrete):
        x_np = (
            F.one_hot(torch.tensor(x_np), num_classes=observation_space.n)
            .float()
            .numpy()
        )

    with torch.no_grad():
        out = network(x_np)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, spaces.flatdim(action_space)])

    if isinstance(observation_space, spaces.Dict):
        x = {key: torch.tensor(value) for key, value in x_np.items()}
    elif isinstance(observation_space, spaces.Tuple):
        x = tuple(torch.tensor(value) for value in x_np)
    else:
        x = torch.tensor(x_np)

    with torch.no_grad():
        out = network(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, spaces.flatdim(action_space)])


@pytest.mark.parametrize("action_space", [generate_random_box_space((4,))])
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_dict_or_tuple_space(2, 3),
        generate_discrete_space(4),
        generate_random_box_space((8,)),
        generate_random_box_space((3, 32, 32)),
    ],
)
def test_deterministic_actor_clone(
    observation_space: spaces.Space, action_space: spaces.Space
):
    network = DeterministicActor(observation_space, action_space)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert str(clone.state_dict()) == str(network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])


def test_deterministic_actor_rescale_action():
    # Test rescaling with different output activations

    # Create a simple action space
    action_low = torch.tensor([-2.0, -1.0])
    action_high = torch.tensor([2.0, 3.0])

    # Test with Tanh activation (default for DeterministicActor)
    action = torch.tensor([[-0.5, 0.5]])  # Action from network between -1 and 1
    rescaled = DeterministicActor.rescale_action(
        action, action_low, action_high, "Tanh"
    )
    # Calculate expected value: low + (high - low) * (action - (-1)) / (1 - (-1))
    expected = torch.tensor([[-1.0, 2.0]])  # Should be mapped to middle of range
    torch.testing.assert_close(rescaled, expected)

    # Test with Sigmoid activation
    action = torch.tensor([[0.25, 0.75]])  # Action from network between 0 and 1
    rescaled = DeterministicActor.rescale_action(
        action, action_low, action_high, "Sigmoid"
    )
    expected = torch.tensor(
        [[-1.0, 2.0]]
    )  # Should be mapped to quarter/three-quarters of range
    torch.testing.assert_close(rescaled, expected)

    # Test with no activation (unbounded)
    action = torch.tensor([[1.0, -3.0]])  # Unbounded action
    rescaled = DeterministicActor.rescale_action(action, action_low, action_high, None)
    # With no activation, action is just passed through as is
    expected = action
    torch.testing.assert_close(rescaled, expected)

    # Test clipping behavior (out-of-bounds values)
    action = torch.tensor([[-2.0, 2.0]])  # Actions outside of tanh range
    rescaled = DeterministicActor.rescale_action(
        action, action_low, action_high, "Tanh"
    )
    # For actions outside the normal range, we still apply the same rescaling formula
    # For -2.0: low + (high - low) * (-2.0 - (-1)) / (1 - (-1)) = low + (high - low) * (-1.0/2.0)
    # For 2.0: low + (high - low) * (2.0 - (-1)) / (1 - (-1)) = low + (high - low) * (3.0/2.0)
    expected_first = action_low + (action_high - action_low) * (
        action[0, 0] - (-1.0)
    ) / (1.0 - (-1.0))
    expected_second = action_low + (action_high - action_low) * (
        action[0, 1] - (-1.0)
    ) / (1.0 - (-1.0))
    expected = torch.tensor([[expected_first[0], expected_second[1]]])
    torch.testing.assert_close(rescaled, expected)


def test_deterministic_actor_forward_rescaling():
    # Test that the forward method rescales actions correctly
    # Create a continuous action space
    action_space = spaces.Box(low=np.array([-2.0, -1.0]), high=np.array([2.0, 3.0]))

    # Test with default Tanh activation
    actor = DeterministicActor(
        observation_space=spaces.Box(low=-1, high=1, shape=(2,)),
        action_space=action_space,
    )

    # Create sample observation
    obs = torch.tensor([[-0.5, 0.5]]).float()

    # Override the head_net to output a known value
    def mock_forward(x):
        return torch.tensor([[-0.5, 0.5]])  # Fixed output for testing

    # Save original function
    original_head_forward = actor.head_net.forward

    try:
        # Mock the head network output
        actor.head_net.forward = mock_forward

        # Get action from forward pass
        with torch.no_grad():
            action = actor(obs)

        # Expected value based on rescaling from [-1,1] to action space
        expected = torch.tensor([[-1.0, 2.0]])  # Middle of the action range
        torch.testing.assert_close(action, expected)

        # Test with explicit Sigmoid activation
        actor = DeterministicActor(
            observation_space=spaces.Box(low=-1, high=1, shape=(2,)),
            action_space=action_space,
            head_config={"output_activation": "Sigmoid", "hidden_size": [32]},
        )

        # Override again for the new actor
        actor.head_net.forward = lambda x: torch.tensor([[0.25, 0.75]])

        with torch.no_grad():
            action = actor(obs)

        expected = torch.tensor(
            [[-1.0, 2.0]]
        )  # Should map to quarter/three-quarters of range
        torch.testing.assert_close(action, expected)

    finally:
        # Restore original function if needed
        if "original_head_forward" in locals():
            actor.head_net.forward = original_head_forward


@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space((4,)),
        generate_discrete_space(4),
        spaces.MultiDiscrete([3, 4, 5]),
        spaces.MultiBinary(4),
    ],
)
@pytest.mark.parametrize(
    "observation_space, encoder_type",
    [
        (generate_dict_or_tuple_space(2, 3), "multi_input"),
        (generate_discrete_space(4), "mlp"),
        (generate_random_box_space((8,)), "mlp"),
        (generate_random_box_space((3, 32, 32)), "cnn"),
    ],
)
def test_stochastic_actor_initialization(observation_space, action_space, encoder_type):
    network = StochasticActor(observation_space, action_space)

    assert network.observation_space == observation_space

    if encoder_type == "multi_input":
        assert isinstance(network.encoder, EvolvableMultiInput)
    elif encoder_type == "mlp":
        assert isinstance(network.encoder, EvolvableMLP)
    elif encoder_type == "cnn":
        assert isinstance(network.encoder, EvolvableCNN)

    evolvable_modules = network.modules()
    assert "encoder" in evolvable_modules
    assert "head_net" in evolvable_modules


@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space((4,)),
        generate_discrete_space(4),
        spaces.MultiDiscrete([3, 4, 5]),
        spaces.MultiBinary(4),
    ],
)
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_dict_or_tuple_space(2, 3),
        generate_discrete_space(4),
        generate_random_box_space((8,)),
        generate_random_box_space((3, 32, 32)),
    ],
)
def test_stochastic_actor_mutation_methods(observation_space, action_space):
    network = StochasticActor(observation_space, action_space)

    for method in network.mutation_methods:
        new_network = network.clone()
        getattr(new_network, method)()

        if "." in method and new_network.last_mutation_attr is not None:
            net_name = method.split(".")[0]
            mutated_module: EvolvableModule = getattr(new_network, net_name)

            exec_method = new_network.last_mutation_attr.split(".")[-1]

            if isinstance(observation_space, (spaces.Tuple, spaces.Dict)):
                mutated_attr = mutated_module.last_mutation_attr.split(".")[-1]
            else:
                mutated_attr = mutated_module.last_mutation_attr

            assert mutated_attr == exec_method

        check_equal_params_ind(network, new_network)


@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space((4,)),
        generate_discrete_space(4),
        spaces.MultiDiscrete([3, 4, 5]),
        spaces.MultiBinary(4),
    ],
)
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_dict_or_tuple_space(2, 3),
        generate_discrete_space(4),
        generate_random_box_space((8,)),
        generate_random_box_space((3, 32, 32)),
    ],
)
def test_stochastic_actor_forward(
    observation_space: spaces.Space, action_space: spaces.Space
):
    # For continuous action spaces, we need to set squash_output=True to ensure actions are scaled
    squash_output = isinstance(action_space, spaces.Box)

    network = StochasticActor(
        observation_space, action_space, squash_output=squash_output
    )

    x_np = observation_space.sample()

    if isinstance(observation_space, spaces.Discrete):
        x_np = (
            F.one_hot(torch.tensor(x_np), num_classes=observation_space.n)
            .float()
            .numpy()
        )

    with torch.no_grad():
        action, log_prob, entropy = network(x_np)

    # Check action shape and values
    assert isinstance(action, torch.Tensor)

    # For continuous action spaces
    if isinstance(action_space, spaces.Box):
        # Check that actions are within bounds
        action_low = torch.tensor(action_space.low, device=action.device)
        action_high = torch.tensor(action_space.high, device=action.device)
        assert torch.all(action >= action_low)
        assert torch.all(action <= action_high)

    # For discrete action spaces
    elif isinstance(action_space, spaces.Discrete):
        assert action.shape == torch.Size([1])  # One scalar action
        assert torch.all(action >= 0)
        assert torch.all(action < action_space.n)

    # For multi-discrete action spaces
    elif isinstance(action_space, spaces.MultiDiscrete):
        assert action.shape == torch.Size([1, len(action_space.nvec)])
        for i, n in enumerate(action_space.nvec):
            assert torch.all(action[:, i] >= 0)
            assert torch.all(action[:, i] < n)

    # For multi-binary action spaces
    elif isinstance(action_space, spaces.MultiBinary):
        assert action.shape == torch.Size([1, action_space.n])
        assert torch.all((action == 0) | (action == 1))

    # Check log_prob
    assert isinstance(log_prob, torch.Tensor)

    # Check entropy - it might be None for some distribution types
    if entropy is not None:
        assert isinstance(entropy, torch.Tensor)

    # Test with tensor inputs
    if isinstance(observation_space, spaces.Dict):
        x = {key: torch.tensor(value).clone().detach() for key, value in x_np.items()}
    elif isinstance(observation_space, spaces.Tuple):
        x = tuple(torch.tensor(value).clone().detach() for value in x_np)
    else:
        x = torch.tensor(x_np).clone().detach()

    with torch.no_grad():
        action, log_prob, entropy = network(x)

    assert isinstance(action, torch.Tensor)
    assert isinstance(log_prob, torch.Tensor)
    # Entropy might be None
    if entropy is not None:
        assert isinstance(entropy, torch.Tensor)


@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space((4,)),
        generate_discrete_space(4),
        spaces.MultiDiscrete([3, 4, 5]),
        spaces.MultiBinary(4),
    ],
)
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_dict_or_tuple_space(2, 3),
        generate_discrete_space(4),
        generate_random_box_space((8,)),
        generate_random_box_space((3, 32, 32)),
    ],
)
def test_stochastic_actor_clone(
    observation_space: spaces.Space, action_space: spaces.Space
):
    network = StochasticActor(observation_space, action_space)

    original_net_dict = dict(network.named_parameters())
    clone = network.clone()
    assert isinstance(clone, EvolvableNetwork)

    assert_close_dict(network.init_dict, clone.init_dict)

    assert str(clone.state_dict()) == str(network.state_dict())
    for key, param in clone.named_parameters():
        torch.testing.assert_close(param, original_net_dict[key])


def test_stochastic_actor_scaling():
    # Test the scale_action method with different squash_output values

    # Create a continuous action space
    action_space = spaces.Box(low=np.array([-2.0, -1.0]), high=np.array([2.0, 3.0]))

    # Test with squash_output=True
    actor = StochasticActor(
        observation_space=spaces.Box(low=-1, high=1, shape=(2,)),
        action_space=action_space,
        squash_output=True,
    )

    # Test rescaling from [-1, 1] to [low, high]
    # Use action with same dimension as action space (2)
    action = torch.tensor([[-1.0, 0.0], [1.0, -1.0]])  # Batch of actions
    scaled = actor.scale_action(action)

    # Expected: For -1 -> low, for 0 -> middle, for 1 -> high
    expected = torch.tensor([[-2.0, 1.0], [2.0, -1.0]])
    torch.testing.assert_close(scaled, expected)

    # Test with forward method when squash_output=True
    obs = torch.tensor([[-0.5, 0.5]]).float()

    # Mock the distribution to return a known action
    original_forward = actor.head_net.forward

    def mock_forward(x, mask=None):
        return torch.tensor([[-0.5, 0.5]]), torch.tensor([0.0]), torch.tensor([1.0])

    try:
        actor.head_net.forward = mock_forward

        # Get action from forward pass
        with torch.no_grad():
            action, _, _ = actor(obs)

        # Expected value based on rescaling from [-1,1] to action space
        # For -0.5: low + (0.5 * (-0.5 + 1) * (high - low)) = low + 0.25 * (high - low)
        # For 0.5: low + (0.5 * (0.5 + 1) * (high - low)) = low + 0.75 * (high - low)
        expected = torch.tensor(
            [[-1.0, 2.0]]
        )  # Should be between low and middle, middle and high
        torch.testing.assert_close(action, expected)

    finally:
        # Restore original function
        actor.head_net.forward = original_forward

    # Test with discrete action space (no scaling needed)
    discrete_space = spaces.Discrete(3)
    actor = StochasticActor(
        observation_space=spaces.Box(low=-1, high=1, shape=(2,)),
        action_space=discrete_space,
    )

    # For discrete spaces, no scaling is applied - verify by testing forward
    def mock_discrete_forward(x, mask=None):
        return torch.tensor([1]), torch.tensor([0.0]), torch.tensor([1.0])

    try:
        actor.head_net.forward = mock_discrete_forward

        with torch.no_grad():
            action, _, _ = actor(obs)

        # Action should remain unchanged
        expected = torch.tensor([1])
        torch.testing.assert_close(action, expected)
    finally:
        # Restore original function
        actor.head_net.forward = mock_discrete_forward


@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space((4,)),
        generate_discrete_space(4),
        spaces.MultiDiscrete([3, 4, 5]),
        spaces.MultiBinary(4),
    ],
)
def test_stochastic_actor_distribution_methods(action_space: spaces.Space):
    """Test StochasticActor's distribution-related methods with different action spaces."""
    observation_space = spaces.Box(low=-1, high=1, shape=(8,))
    network = StochasticActor(observation_space, action_space)

    # Get a sample observation
    obs = torch.tensor(observation_space.sample()).float().unsqueeze(0)

    # Get action, log_prob, and entropy from forward pass
    with torch.no_grad():
        action, log_prob, entropy = network(obs)

    # Test action_entropy matches the entropy from forward
    manual_entropy = network.action_entropy()
    torch.testing.assert_close(entropy, manual_entropy)

    # Specific tests for each action space type
    if isinstance(action_space, spaces.Box):
        # For continuous spaces, log_prob should be scalar
        assert log_prob.shape == torch.Size([1])
        # Entropy should be positive for continuous distributions
        assert torch.all(entropy > 0)

    elif isinstance(action_space, spaces.Discrete):
        # For discrete spaces, log_prob is scalar
        assert log_prob.shape == torch.Size([1])
        # Entropy should be positive and at most log(n)
        assert torch.all(entropy > 0)
        assert torch.all(
            entropy <= torch.log(torch.tensor(action_space.n, dtype=torch.float))
        )

    elif isinstance(action_space, spaces.MultiDiscrete):
        # For multi-discrete spaces, log_prob is scalar (sum of log probs)
        assert log_prob.shape == torch.Size([1])
        # Entropy should be positive
        assert torch.all(entropy > 0)

    elif isinstance(action_space, spaces.MultiBinary):
        # For multi-binary spaces, log_prob is scalar
        assert log_prob.shape == torch.Size([1])
        # Entropy should be positive
        assert torch.all(entropy > 0)

    # Test action_log_prob gives value for the chosen action
    # Note: This may not give exactly the same value as from forward
    # since the distribution is resampled, but we can check it has the right shape
    manual_log_prob = network.action_log_prob(action)
    assert manual_log_prob.shape == log_prob.shape


def test_stochastic_actor_action_masking_discrete():
    """Test action masking in StochasticActor with Discrete action space."""
    # Create discrete action space (4 actions)
    action_space = spaces.Discrete(4)
    observation_space = spaces.Box(low=-1, high=1, shape=(8,))

    # Create actor
    actor = StochasticActor(observation_space, action_space)

    # Sample observation
    obs = torch.tensor(observation_space.sample()).float().unsqueeze(0)

    # Create action mask that only allows actions 1 and 3
    # True = valid action, False = masked action
    action_mask = torch.tensor([[False, True, False, True]])

    # Run forward pass with mask
    with torch.no_grad():
        action, log_prob, entropy = actor(obs, action_mask)

    # Verify that the selected action is either 1 or 3 (unmasked)
    assert action.item() in [1, 3]

    # Run many times to ensure consistent masking
    valid_actions = []
    for _ in range(10):
        with torch.no_grad():
            action, _, _ = actor(obs, action_mask)
        valid_actions.append(action.item())

    # Verify only unmasked actions were selected
    assert all(a in [1, 3] for a in valid_actions)

    # The log probabilities of masked actions should be -inf
    # This is hard to test directly without accessing the distribution
    # but we can verify entropy is smaller with masks
    with torch.no_grad():
        _, _, unmasked_entropy = actor(obs, None)  # No mask
        _, _, masked_entropy = actor(obs, action_mask)  # With mask

    # Entropy should be lower with mask (fewer options)
    assert masked_entropy < unmasked_entropy


def test_stochastic_actor_action_masking_multidiscrete():
    """Test action masking in StochasticActor with MultiDiscrete action space."""
    # Create multi-discrete action space
    action_space = spaces.MultiDiscrete([3, 2, 4])
    observation_space = spaces.Box(low=-1, high=1, shape=(8,))

    # Create actor
    actor = StochasticActor(observation_space, action_space)

    # Sample observation
    obs = torch.tensor(observation_space.sample()).float().unsqueeze(0)

    # Create action masks for each dimension
    # Dimension 1: allow only action 0
    # Dimension 2: allow only action 1
    # Dimension 3: allow actions 1 and 3
    # Format as a flat tensor that will be split by the distribution
    action_mask = torch.tensor(
        [
            [
                True,
                False,
                False,  # 1st dimension (3 options)
                False,
                True,  # 2nd dimension (2 options)
                False,
                True,
                False,
                True,
            ]  # 3rd dimension (4 options)
        ]
    )

    # Run forward pass with mask multiple times
    valid_actions = []
    for _ in range(10):
        with torch.no_grad():
            action, _, _ = actor(obs, action_mask)
        valid_actions.append(action.squeeze().tolist())

    # Verify all actions respect the mask
    for action in valid_actions:
        assert action[0] == 0  # 1st dimension must be 0
        assert action[1] == 1  # 2nd dimension must be 1
        assert action[2] in [1, 3]  # 3rd dimension must be 1 or 3


def test_stochastic_actor_action_masking_multibinary():
    """Test action masking in StochasticActor with MultiBinary action space."""
    # Create multi-binary action space
    action_space = spaces.MultiBinary(4)
    observation_space = spaces.Box(low=-1, high=1, shape=(8,))

    # Create actor
    actor = StochasticActor(observation_space, action_space)

    # Sample observation
    obs = torch.tensor(observation_space.sample()).float().unsqueeze(0)

    # For MultiBinary, the mask should have the same shape as the action space
    # True means this action is allowed, False means it's masked (forbidden)
    # We'll mask the first and last bits to only allow action=0
    action_mask = torch.tensor([[False, True, True, False]])

    # Run forward pass with mask multiple times
    valid_actions = []
    for _ in range(10):
        with torch.no_grad():
            action, _, _ = actor(obs, action_mask)
        valid_actions.append(action.squeeze().tolist())

    # Verify all actions respect the mask
    for action in valid_actions:
        # The first and last bits should be 0 because we masked them to prevent 1
        assert action[0] == 0
        assert action[3] == 0
        # Positions 1 and 2 can be either 0 or 1
