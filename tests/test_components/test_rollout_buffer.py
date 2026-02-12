import warnings

import numpy as np
import pytest
import torch
from gymnasium import spaces
from tensordict import TensorDict

from agilerl.components.rollout_buffer import RolloutBuffer
from agilerl.typing import BPTTSequenceType


class TestRolloutBufferInitialization:
    """Test rollout buffer initialization with different configurations."""

    def test_init_box_observations(self):
        """Test initialization with Box observation space."""
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        assert buffer.capacity == 10
        assert buffer.num_envs == 2
        assert buffer.pos == 0
        assert not buffer.full
        assert buffer.buffer is not None
        assert buffer.buffer["observations"].shape == (10, 2, 4)
        assert buffer.buffer["actions"].shape == (10, 2, 1)

    def test_init_discrete_observations(self):
        """Test initialization with Discrete observation space."""
        obs_space = spaces.Discrete(5)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=3,
            device="cpu",
        )

        assert buffer.buffer["observations"].shape == (5, 3, 1)
        assert buffer.buffer["actions"].shape == (5, 3, 2)

    def test_init_dict_observations(self):
        """Test initialization with Dict observation space."""
        obs_space = spaces.Dict(
            {
                "vector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "discrete": spaces.Discrete(3),
            },
        )
        action_space = spaces.Discrete(2)

        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        assert isinstance(buffer.buffer["observations"], TensorDict)
        assert "vector" in buffer.buffer["observations"]
        assert "discrete" in buffer.buffer["observations"]
        assert buffer.buffer["observations"]["vector"].shape == (5, 2, 2)
        assert buffer.buffer["observations"]["discrete"].shape == (5, 2, 1)

    def test_init_recurrent_mode(self):
        """Test initialization with recurrent mode."""
        observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=100,
            num_envs=1,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            gae_lambda=0.95,
            gamma=0.99,
        )

        assert buffer.capacity == 100
        assert buffer.observation_space == observation_space
        assert buffer.action_space == action_space
        assert buffer.gamma == 0.99
        assert buffer.gae_lambda == 0.95
        assert buffer.recurrent is False
        assert buffer.hidden_state_architecture is None
        assert buffer.device == "cpu"
        assert buffer.pos == 0
        assert buffer.full is False

        # Test with hidden states
        buffer = RolloutBuffer(
            capacity=100,
            num_envs=8,
            observation_space=observation_space,
            action_space=action_space,
            device="cpu",
            gae_lambda=0.95,
            gamma=0.99,
            recurrent=True,
            # (num_layers * directions, num_envs, hidden_size)
            hidden_state_architecture={
                "shared_encoder_h": (1, 8, 64),
                "shared_encoder_c": (1, 8, 64),
            },
        )

        assert buffer.recurrent is True
        assert buffer.hidden_state_architecture is not None
        assert buffer.buffer.get("hidden_states") is not None

    def test_init_error_recurrent_without_architecture(self):
        """Test error when recurrent=True but no hidden_state_architecture provided."""
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        with pytest.raises(
            ValueError,
            match="hidden_state_architecture must be provided",
        ):
            RolloutBuffer(
                capacity=10,
                observation_space=obs_space,
                action_space=action_space,
                num_envs=2,
                recurrent=True,
                hidden_state_architecture=None,
            )


class TestRolloutBufferDataAddition:
    """Test adding data to the rollout buffer."""

    def test_add_box_observations(self):
        """Test adding data with Box observations."""
        observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        device = "cpu"

        buffer = RolloutBuffer(
            capacity=100,
            num_envs=1,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        # Add a single sample
        obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)
        action = np.array([action_space.sample()])  # Ensure action is within space
        reward = 1.0
        done = False
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape).astype(
            observation_space.dtype,
        )

        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

        assert buffer.pos == 1
        assert not buffer.full
        # where X is the environment index, and Y is the position in the buffer (pos-1 for last added)
        # Data is stored at buffer.pos - 1
        current_pos_idx = buffer.pos - 1
        assert np.array_equal(
            buffer.buffer.get("observations")[current_pos_idx, 0].cpu().numpy(),
            obs,
        )
        assert np.array_equal(
            buffer.buffer.get("actions")[current_pos_idx, 0, 0].cpu().numpy(),
            action[0],
        ), f"Expected action {action[0]} at position {current_pos_idx}, but got {buffer.buffer.get('actions')[current_pos_idx, 0].cpu().numpy()}"
        assert buffer.buffer.get("rewards")[current_pos_idx, 0].item() == reward
        assert buffer.buffer.get("dones")[current_pos_idx, 0].item() == float(done)
        assert buffer.buffer.get("values")[current_pos_idx, 0].item() == value
        assert buffer.buffer.get("log_probs")[current_pos_idx, 0].item() == log_prob
        assert np.array_equal(
            buffer.buffer.get("next_observations")[current_pos_idx, 0].cpu().numpy(),
            next_obs,
        )

        # Add samples until buffer is full
        for _ in range(buffer.capacity - 1):  # Already added one sample
            buffer.add(obs, action, reward, done, value, log_prob, next_obs)

        assert buffer.pos == buffer.capacity  # pos is next insertion point
        assert buffer.full is True  # Buffer is full when pos reaches capacity

        buffer.reset()

        assert buffer.pos == 0
        assert buffer.full is False

    def test_add_discrete_observations(self):
        """Test adding data with Discrete observations."""
        observation_space = spaces.Discrete(5)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        device = "cpu"
        num_envs = 8
        buffer = RolloutBuffer(
            capacity=100,
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        # Add a single sample
        obs = np.array([observation_space.sample() for _ in range(num_envs)])
        action = np.array([action_space.sample() for _ in range(num_envs)])
        reward = np.array([1.0 for _ in range(num_envs)])
        done = np.array([False for _ in range(num_envs)])
        value = np.array([0.5 for _ in range(num_envs)])
        log_prob = np.array([-0.5 for _ in range(num_envs)])
        next_obs = np.array([observation_space.sample() for _ in range(num_envs)])

        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

        assert buffer.pos == 1
        assert not buffer.full
        assert buffer.buffer.get("observations").shape == (100, num_envs, 1)
        assert buffer.buffer.get("actions").shape == (100, num_envs, 2)
        assert buffer.buffer.get("rewards").shape == (100, num_envs)
        assert buffer.buffer.get("dones").shape == (100, num_envs)
        assert buffer.buffer.get("values").shape == (100, num_envs)
        assert buffer.buffer.get("log_probs").shape == (100, num_envs)
        assert buffer.buffer.get("next_observations").shape == (100, num_envs, 1)

    def test_add_multidiscrete_observations(self):
        """Test adding data with MultiDiscrete observations."""
        observation_space = spaces.MultiDiscrete([2, 3])
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        device = "cpu"
        num_envs = 8
        buffer = RolloutBuffer(
            capacity=100,
            num_envs=num_envs,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        # Add a single sample
        obs = np.array([observation_space.sample() for _ in range(num_envs)])
        action = np.array([action_space.sample() for _ in range(num_envs)])
        reward = np.array([1.0 for _ in range(num_envs)])
        done = np.array([False for _ in range(num_envs)])
        value = np.array([0.5 for _ in range(num_envs)])
        log_prob = np.array([-0.5 for _ in range(num_envs)])
        next_obs = np.array([observation_space.sample() for _ in range(num_envs)])

        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

        assert buffer.pos == 1
        assert not buffer.full
        assert buffer.buffer.get("observations").shape == (100, num_envs, 2)
        assert buffer.buffer.get("actions").shape == (100, num_envs, 2)
        assert buffer.buffer.get("rewards").shape == (100, num_envs)
        assert buffer.buffer.get("dones").shape == (100, num_envs)
        assert buffer.buffer.get("values").shape == (100, num_envs)
        assert buffer.buffer.get("log_probs").shape == (100, num_envs)
        assert buffer.buffer.get("next_observations").shape == (100, num_envs, 2)

    def test_add_dict_observations(self):
        """Test adding data with Dict observations."""
        obs_space = spaces.Dict(
            {
                "vector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "discrete": spaces.Discrete(3),
            },
        )
        action_space = spaces.Discrete(2)

        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        obs = {
            "vector": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "discrete": np.array([[0], [2]]),
        }
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
        )

        assert buffer.pos == 1
        assert isinstance(buffer.buffer["observations"], TensorDict)
        assert torch.allclose(
            buffer.buffer["observations"]["vector"][0],
            torch.tensor(obs["vector"], dtype=torch.float32),
        )
        assert torch.allclose(
            buffer.buffer["observations"]["discrete"][0],
            torch.tensor(obs["discrete"], dtype=torch.float32),
        )

    def test_add_with_next_obs(self):
        """Test adding data with next observations."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        next_obs = np.array([[1.1, 2.1], [3.1, 4.1]])
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
            next_obs=next_obs,
        )

        assert torch.allclose(
            buffer.buffer["next_observations"][0],
            torch.tensor(next_obs, dtype=torch.float32),
        )

    def test_add_recurrent_data(self):
        """Test adding data with hidden states."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),  # (num_layers, num_envs, hidden_size)
        }

        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])
        hidden_state = {"h_actor": torch.randn(1, 2, 4)}

        buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
            hidden_state=hidden_state,
        )

        assert buffer.pos == 1
        assert buffer.recurrent
        assert "hidden_states" in buffer.buffer
        assert "h_actor" in buffer.buffer["hidden_states"]

    def test_add_capacity_overflow(self):
        """Test buffer behavior when capacity is exceeded."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=2,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
        )

        obs = np.array([[1.0, 2.0]])
        action = np.array([0])
        reward = np.array([1.0])
        done = np.array([False])
        value = np.array([0.5])
        log_prob = np.array([0.1])

        # Fill buffer to capacity
        buffer.add(obs, action, reward, done, value, log_prob)
        buffer.add(obs, action, reward, done, value, log_prob)

        # Should raise error when exceeding capacity
        with pytest.raises(ValueError, match="Buffer has reached capacity"):
            buffer.add(obs, action, reward, done, value, log_prob)

    def test_add_wrap_at_capacity(self):
        """Test buffer wrapping when wrap_at_capacity=True."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=2,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            wrap_at_capacity=True,
        )

        obs = np.array([[1.0, 2.0]])
        action = np.array([0])
        reward = np.array([1.0])
        done = np.array([False])
        value = np.array([0.5])
        log_prob = np.array([0.1])

        # Fill buffer to capacity
        buffer.add(obs, action, reward, done, value, log_prob)
        buffer.add(obs, action, reward, done, value, log_prob)

        # Should wrap around
        buffer.add(obs, action, reward, done, value, log_prob)
        assert buffer.pos == 1
        assert buffer.full


class TestRolloutBufferReturnsAndAdvantages:
    """Test computation of returns and advantages."""

    def test_compute_returns_gae(self):
        """Test computing returns and advantages with GAE."""
        observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        device = "cpu"
        capacity = 5

        buffer = RolloutBuffer(
            capacity=capacity,
            num_envs=1,
            observation_space=observation_space,
            action_space=action_space,
            gamma=0.99,
            gae_lambda=0.95,
            use_gae=True,
            device=device,
        )

        # Add samples
        for i in range(capacity):
            obs = np.random.rand(*observation_space.shape).astype(
                observation_space.dtype,
            )
            action = np.array([action_space.sample()])
            reward = 1.0
            done = i == (capacity - 1)  # Last step is done
            value = 0.5
            log_prob = -0.5
            next_obs = np.random.rand(*observation_space.shape).astype(
                observation_space.dtype,
            )

            buffer.add(obs, action, reward, done, value, log_prob, next_obs)

        # Compute returns and advantages
        last_value = torch.tensor([[0.0]], device=device)  # Shape (num_envs, 1)
        last_done = torch.tensor([[0.0]], device=device)  # Shape (num_envs, 1)
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Check that returns and advantages are computed
        # Slicing [:, 0] gets data for the first (and only) environment
        assert not np.array_equal(
            buffer.buffer.get("returns")[:, 0].cpu().numpy(),
            np.zeros((capacity, 1)),
        )
        assert not np.array_equal(
            buffer.buffer.get("advantages")[:, 0].cpu().numpy(),
            np.zeros((capacity, 1)),
        )

        # Check that returns are higher for earlier steps (due to discounting)
        assert (
            buffer.buffer.get("returns")[0, 0].item()
            > buffer.buffer.get("returns")[capacity - 1, 0].item()
        )

    def test_compute_returns_monte_carlo(self):
        """Test computing returns with Monte Carlo method."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            use_gae=False,
            gamma=0.99,
        )

        # Add some data
        for i in range(3):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([0, 1])
            reward = np.array([1.0, -1.0])
            done = np.array([False, False])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])

            buffer.add(obs, action, reward, done, value, log_prob)

        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, False])

        buffer.compute_returns_and_advantages(last_value, last_done)

        # Check that advantages and returns were computed
        advantages = buffer.buffer["advantages"][:3]
        returns = buffer.buffer["returns"][:3]

        assert advantages.shape == (3, 2)
        assert returns.shape == (3, 2)
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()

    def test_compute_returns_with_dones(self):
        """Test computing returns with done episodes."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        # Add data with some done episodes
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])  # Second env episode ends
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(obs, action, reward, done, value, log_prob)

        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, True])

        buffer.compute_returns_and_advantages(last_value, last_done)

        advantages = buffer.buffer["advantages"][:1]
        returns = buffer.buffer["returns"][:1]

        assert advantages.shape == (1, 2)
        assert returns.shape == (1, 2)
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()


class TestRolloutBufferDataRetrieval:
    """Test retrieving data from the buffer."""

    def test_get_numpy_data(self):
        """Test getting data as numpy arrays."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        # Add test data
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Get data
        data = buffer.get()

        assert isinstance(data, dict)
        assert "observations" in data
        assert "actions" in data
        assert "rewards" in data
        assert "advantages" in data
        assert "returns" in data

        # Check shapes and types
        assert isinstance(data["observations"], np.ndarray)
        assert data["observations"].shape == (2, 2)  # 2 samples, 2 features
        assert data["actions"].shape == (2, 1)

    def test_get_tensor_batch(self):
        """Test getting tensor batch."""
        observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        device = "cpu"
        num_samples = 10

        buffer = RolloutBuffer(
            capacity=100,
            num_envs=1,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )

        # Add samples
        for i in range(num_samples):
            obs = np.random.rand(*observation_space.shape).astype(
                observation_space.dtype,
            )
            action = np.array([action_space.sample()])
            reward = 1.0
            done = i == (num_samples - 1)  # Last step is done
            value = 0.5
            log_prob = -0.5
            next_obs = np.random.rand(*observation_space.shape).astype(
                observation_space.dtype,
            )

            buffer.add(obs, action, reward, done, value, log_prob, next_obs)

        # Compute returns and advantages
        last_value = torch.tensor([[0.0]], device=device)
        last_done = torch.tensor([[0.0]], device=device)
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Get all data (up to current pos)
        batch = buffer.get()  # Gets all data up to buffer.pos

        assert len(batch["observations"]) == num_samples
        assert len(batch["actions"]) == num_samples
        # Rewards, dones, values, log_probs are (num_samples, 1) after get() flattens num_envs
        assert len(batch["rewards"]) == num_samples
        assert len(batch["dones"]) == num_samples
        assert len(batch["values"]) == num_samples
        assert len(batch["log_probs"]) == num_samples
        assert len(batch["advantages"]) == num_samples
        assert len(batch["returns"]) == num_samples

        # Get batch of specific size
        batch_size = 5
        batch = buffer.get(batch_size=batch_size)

        assert len(batch["observations"]) == batch_size
        assert len(batch["actions"]) == batch_size

        # Get tensor batch
        tensor_batch = buffer.get_tensor_batch(batch_size=batch_size)

        assert isinstance(tensor_batch["observations"], torch.Tensor)
        assert isinstance(tensor_batch["actions"], torch.Tensor)
        assert isinstance(tensor_batch["advantages"], torch.Tensor)
        assert tensor_batch["observations"].shape[0] == batch_size

    def test_get_dict_observations(self):
        """Test getting data with dict observations."""
        obs_space = spaces.Dict(
            {
                "vector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "discrete": spaces.Discrete(3),
            },
        )
        action_space = spaces.Discrete(2)

        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        obs = {
            "vector": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "discrete": np.array([[0], [1]]),
        }
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Get data
        data = buffer.get()

        assert isinstance(data["observations"], dict)
        assert "vector" in data["observations"]
        assert "discrete" in data["observations"]
        assert data["observations"]["vector"].shape == (2, 2)
        assert data["observations"]["discrete"].shape == (2, 1)

    def test_get_with_batch_size(self):
        """Test getting data with specific batch size."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        # Add multiple timesteps
        for i in range(3):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([0, 1])
            reward = np.array([1.0, -1.0])
            done = np.array([False, False])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])

            buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, False])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Get data with batch size
        data = buffer.get(batch_size=4)

        assert data["observations"].shape[0] == 4
        assert data["actions"].shape[0] == 4


class TestRolloutBufferSequences:
    """Test sequence methods for BPTT."""

    def test_prepare_sequence_tensors_basic(self):
        """Test basic sequence tensor preparation."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }

        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
            max_seq_len=3,
        )

        # Add multiple timesteps with episodes ending
        for i in range(5):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            done = np.array([i == 4, i == 2])  # Episodes end at different times
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])
            hidden_state = {"h_actor": torch.randn(1, 2, 4)}

            buffer.add(
                obs,
                action,
                reward,
                done,
                value,
                log_prob,
                hidden_state=hidden_state,
            )

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([True, True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer.prepare_sequence_tensors()

        # Check that sequence data was prepared
        assert buffer.padded_data is not None
        assert buffer.unpadded_data is not None
        assert buffer.unpadded_slices is not None
        assert buffer.num_sequences is not None
        assert buffer.max_sequence_length is not None

        # Check that padded data contains expected keys
        assert "observations" in buffer.padded_data
        assert "actions" in buffer.padded_data
        assert "pad_mask" in buffer.padded_data

        # Check that unpadded data contains expected keys
        assert "log_probs" in buffer.unpadded_data
        assert "advantages" in buffer.unpadded_data
        assert "values" in buffer.unpadded_data
        assert "returns" in buffer.unpadded_data

    @pytest.mark.parametrize("recurrent", [True, False])
    def test_prepare_sequence_tensors_with_hidden_states(self, recurrent):
        """Test sequence tensor preparation with hidden states."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
            "h_critic": (2, 2, 8),
        }

        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            recurrent=recurrent,
            hidden_state_architecture=hidden_state_arch,
            max_seq_len=4,
        )

        # Add multiple timesteps
        for i in range(6):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            done = np.array([i == 5, i == 3])  # Episodes end at different times
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])
            hidden_state = {
                "h_actor": torch.randn(1, 2, 4),
                "h_critic": torch.randn(2, 2, 8),
            }

            buffer.add(
                obs,
                action,
                reward,
                done,
                value,
                log_prob,
                hidden_state=hidden_state,
            )

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([True, True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        if recurrent:
            buffer.prepare_sequence_tensors()
        else:
            with pytest.raises(ValueError):
                buffer.prepare_sequence_tensors()
            return

        # Check that initial hidden states are prepared
        try:
            initial_hidden_states = buffer.padded_data.get_non_tensor(
                "initial_hidden_states",
            )
            assert "h_actor" in initial_hidden_states
            assert "h_critic" in initial_hidden_states

            # Check shapes of initial hidden states
            assert initial_hidden_states["h_actor"].shape[1:] == (
                1,
                4,
            )  # (batch, layers, hidden_size)
            assert initial_hidden_states["h_critic"].shape[1:] == (2, 8)
        except KeyError as err:
            # If initial_hidden_states is not found, the test should fail
            msg = "initial_hidden_states should be present for recurrent buffer"
            raise AssertionError(msg) from err

    def test_get_minibatch_sequences(self):
        """Test getting minibatch sequences."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }

        buffer = RolloutBuffer(
            capacity=15,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
            max_seq_len=3,
        )

        # Add multiple episodes
        for i in range(10):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            # Create episodes of different lengths
            done = np.array([i % 4 == 3, i % 5 == 4])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])
            hidden_state = {"h_actor": torch.randn(1, 2, 4)}

            buffer.add(
                obs,
                action,
                reward,
                done,
                value,
                log_prob,
                hidden_state=hidden_state,
            )

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([True, True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer.prepare_sequence_tensors()

        # Get minibatch sequences
        batch_size = 2
        minibatch_generator = buffer.get_minibatch_sequences(batch_size)

        # Test that we can iterate through minibatches
        minibatch_count = 0
        for padded_batch, unpadded_batch in minibatch_generator:
            minibatch_count += 1

            # Check that batches are TensorDicts
            assert isinstance(padded_batch, TensorDict)
            assert isinstance(unpadded_batch, TensorDict)

            # Check that padded batch contains expected keys
            assert "observations" in padded_batch
            assert "actions" in padded_batch
            assert "pad_mask" in padded_batch

            # Check that unpadded batch contains expected keys
            assert "log_probs" in unpadded_batch
            assert "advantages" in unpadded_batch
            assert "values" in unpadded_batch
            assert "returns" in unpadded_batch

            # Check that initial hidden states are present
            try:
                initial_hidden_states = padded_batch.get_non_tensor(
                    "initial_hidden_states",
                )
                assert "h_actor" in initial_hidden_states
            except KeyError:
                # It's okay if initial_hidden_states is not present for this test
                pass

        assert minibatch_count > 0

    def test_get_minibatch_sequences_without_preparation(self):
        """Test that get_minibatch_sequences raises error without preparation."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            max_seq_len=3,
        )

        # Try to get minibatch sequences without preparing
        with pytest.raises(
            ValueError,
            match="Attempting to fetch minibatches before preparing sequences",
        ):
            list(buffer.get_minibatch_sequences(2))

    def test_pad_sequences_static_method(self):
        """Test the static _pad_sequences method."""
        # Test with regular tensors
        sequences = [
            torch.tensor([[1, 2], [3, 4]]),  # Length 2
            torch.tensor([[5, 6], [7, 8], [9, 10]]),  # Length 3
            torch.tensor([[11, 12]]),  # Length 1
        ]

        # Test padding to max length
        padded = RolloutBuffer._pad_sequences(sequences)
        assert padded.shape == (3, 3, 2)  # (batch_size, max_seq_len, feature_dim)

        # Test padding to specific target length
        padded_target = RolloutBuffer._pad_sequences(sequences, target_length=5)
        assert padded_target.shape == (
            3,
            5,
            2,
        )  # (batch_size, target_length, feature_dim)

        # Test with TensorDict sequences
        td_sequences = [
            TensorDict(
                {"obs": torch.tensor([[1, 2]]), "act": torch.tensor([0])},
                batch_size=[1],
            ),
            TensorDict(
                {"obs": torch.tensor([[3, 4], [5, 6]]), "act": torch.tensor([1, 0])},
                batch_size=[2],
            ),
        ]

        padded_td = RolloutBuffer._pad_sequences(td_sequences)
        assert isinstance(padded_td, TensorDict)
        assert padded_td["obs"].shape == (
            2,
            2,
            2,
        )  # (batch_size, max_seq_len, feature_dim)
        assert padded_td["act"].shape == (2, 2)  # (batch_size, max_seq_len)

    def test_different_bptt_sequence_types(self):
        """Test different BPTT sequence types."""
        from agilerl.typing import BPTTSequenceType

        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }

        max_seq_len = 4
        episode_length = 12
        for seq_type in [
            BPTTSequenceType.CHUNKED,
            BPTTSequenceType.MAXIMUM,
            BPTTSequenceType.FIFTY_PERCENT_OVERLAP,
        ]:
            buffer = RolloutBuffer(
                capacity=20,
                observation_space=obs_space,
                action_space=action_space,
                num_envs=1,
                device="cpu",
                max_seq_len=max_seq_len,
                bptt_sequence_type=seq_type,
                recurrent=True,
                hidden_state_architecture=hidden_state_arch,
            )

            # Add a long episode
            for i in range(episode_length):
                obs = np.array([[i, i + 1]])
                action = np.array([i % 2])
                reward = np.array([1.0])
                done = np.array([i == 11])  # Episode ends at the last step
                value = np.array([0.5])
                log_prob = np.array([0.1])

                buffer.add(obs, action, reward, done, value, log_prob)

            # Compute returns
            last_value = np.array([0.0])
            last_done = np.array([True])
            buffer.compute_returns_and_advantages(last_value, last_done)

            # Prepare sequence tensors
            buffer.prepare_sequence_tensors()

            # Check that the number of sequences is correct
            if seq_type == BPTTSequenceType.CHUNKED:
                assert buffer.num_sequences == episode_length // max_seq_len
            elif seq_type == BPTTSequenceType.MAXIMUM:
                assert buffer.num_sequences == episode_length - max_seq_len + 1
            elif seq_type == BPTTSequenceType.FIFTY_PERCENT_OVERLAP:
                assert buffer.num_sequences == 5

            # Check that sequences were created
            assert buffer.num_sequences > 0
            assert buffer.max_sequence_length <= 4

    def test_empty_buffer_sequence_preparation(self):
        """Test sequence preparation with empty buffer."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }

        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            max_seq_len=3,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Try to prepare sequences with empty buffer
        with pytest.raises(ValueError):
            buffer.prepare_sequence_tensors()

    def test_sequence_preparation_with_dict_observations(self):
        """Test sequence preparation with dictionary observations."""
        obs_space = spaces.Dict(
            {
                "vector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "discrete": spaces.Discrete(3),
            },
        )
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }

        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            max_seq_len=3,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add multiple timesteps
        for i in range(5):
            obs = {
                "vector": np.array([[i, i + 1], [i + 2, i + 3]]),
                "discrete": np.array([[i % 3], [(i + 1) % 3]]),
            }
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            done = np.array([i == 4, i == 2])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])

            buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([True, True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer.prepare_sequence_tensors()

        # Check that observations are properly structured
        assert "observations" in buffer.padded_data
        obs_data = buffer.padded_data["observations"]
        assert isinstance(obs_data, TensorDict)
        assert "vector" in obs_data
        assert "discrete" in obs_data

    def test_sequence_preparation_different_max_seq_len(self):
        """Test sequence preparation with different max_seq_len values."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        # Test with max_seq_len=None (should use full episodes)
        buffer_no_limit = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            max_seq_len=None,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add a long episode
        for i in range(8):
            obs = np.array([[i, i + 1]])
            action = np.array([i % 2])
            reward = np.array([1.0])
            done = np.array([i == 7])  # Episode ends at the last step
            value = np.array([0.5])
            log_prob = np.array([0.1])

            buffer_no_limit.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.0])
        last_done = np.array([True])
        buffer_no_limit.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer_no_limit.prepare_sequence_tensors()

        # Should have one sequence of length 8
        assert buffer_no_limit.num_sequences == 1
        assert buffer_no_limit.max_sequence_length == 8

    def test_minibatch_sequences_batch_size_larger_than_sequences(self):
        """Test minibatch sequences when batch size is larger than available sequences."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            max_seq_len=3,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add a short episode
        for i in range(3):
            obs = np.array([[i, i + 1]])
            action = np.array([i % 2])
            reward = np.array([1.0])
            done = np.array([i == 2])
            value = np.array([0.5])
            log_prob = np.array([0.1])

            buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.0])
        last_done = np.array([True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer.prepare_sequence_tensors()

        # Request batch size larger than available sequences
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minibatch_generator = buffer.get_minibatch_sequences(batch_size=10)
            list(minibatch_generator)  # Consume the generator

            # Should have issued a warning
            assert len(w) > 0
            assert "larger than the number of sequences" in str(w[0].message)

    def test_bptt_sequence_type_parameter(self):
        """Test that bptt_sequence_type parameter is properly stored."""
        from agilerl.typing import BPTTSequenceType

        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            max_seq_len=3,
            bptt_sequence_type=BPTTSequenceType.MAXIMUM,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        assert buffer.bptt_sequence_type == BPTTSequenceType.MAXIMUM

    def test_sequence_preparation_device_parameter(self):
        """Test sequence preparation with specific device parameter."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            max_seq_len=3,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add some data
        for i in range(3):
            obs = np.array([[i, i + 1]])
            action = np.array([i % 2])
            reward = np.array([1.0])
            done = np.array([i == 2])
            value = np.array([0.5])
            log_prob = np.array([0.1])

            buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.0])
        last_done = np.array([True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors with specific device
        buffer.prepare_sequence_tensors(device="cpu")

        # Check that data is on the correct device
        assert buffer.padded_data.device.type == "cpu"
        assert buffer.unpadded_data.device.type == "cpu"


class TestRolloutBufferUtilities:
    """Test utility methods and edge cases."""

    @pytest.mark.parametrize(
        "bptt_sequence_type",
        [
            BPTTSequenceType.CHUNKED,
            BPTTSequenceType.MAXIMUM,
            BPTTSequenceType.FIFTY_PERCENT_OVERLAP,
        ],
    )
    @pytest.mark.parametrize("max_seq_len", [None, 3, 5])
    def test_get_complete_sequences_method(self, bptt_sequence_type, max_seq_len):
        """Test the _get_complete_sequences method directly."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            max_seq_len=max_seq_len,
            bptt_sequence_type=bptt_sequence_type,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Create test data
        test_data = torch.arange(20).reshape(10, 2)  # 10 timesteps, 2 envs
        episode_done_indices = [[3, 7, 9], [4, 9]]  # Different episode endings per env

        sequences, max_length = buffer._get_complete_sequences(
            test_data,
            episode_done_indices,
        )

        # Should have multiple sequences
        assert len(sequences) > 0
        assert max_length > 0

        # Check that sequences are torch tensors
        for seq in sequences:
            assert isinstance(seq, torch.Tensor)

    def test_convert_td_to_np_dict_method(self):
        """Test the _convert_td_to_np_dict method."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
        )

        # Create a test TensorDict
        test_td = TensorDict(
            {
                "observations": torch.randn(3, 2),
                "actions": torch.randint(0, 2, (3, 1)),
                "rewards": torch.randn(3),
                "hidden_states": TensorDict(
                    {
                        "h_actor": torch.randn(3, 1, 4),
                    },
                    batch_size=[3],
                ),
            },
            batch_size=[3],
        )

        # Convert to numpy dict
        np_dict = buffer._convert_td_to_np_dict(test_td)

        # Check that conversion worked
        assert isinstance(np_dict, dict)
        assert "observations" in np_dict
        assert "actions" in np_dict
        assert "rewards" in np_dict
        assert "hidden_states" in np_dict

        # Check that arrays are numpy arrays
        assert isinstance(np_dict["observations"], np.ndarray)
        assert isinstance(np_dict["actions"], np.ndarray)
        assert isinstance(np_dict["rewards"], np.ndarray)
        assert isinstance(np_dict["hidden_states"], dict)
        assert isinstance(np_dict["hidden_states"]["h_actor"], np.ndarray)

    def test_sequence_preparation_with_multiple_episodes_per_env(self):
        """Test sequence preparation with multiple episodes per environment."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=20,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            max_seq_len=4,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add multiple episodes with different lengths
        for i in range(15):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            # Create episodes of different lengths for each environment
            done = np.array([i % 6 == 5, i % 4 == 3])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])

            buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([True, True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer.prepare_sequence_tensors()

        # Should have created multiple sequences
        assert buffer.num_sequences > 2
        assert buffer.max_sequence_length <= 4

    def test_sequence_preparation_short_episodes(self):
        """Test sequence preparation with episodes shorter than max_seq_len."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            max_seq_len=5,  # Longer than episodes
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add short episodes
        for episode in range(3):
            for i in range(2):  # Each episode is only 2 steps
                obs = np.array([[episode * 10 + i, episode * 10 + i + 1]])
                action = np.array([i % 2])
                reward = np.array([1.0])
                done = np.array([i == 1])  # Episode ends after 2 steps
                value = np.array([0.5])
                log_prob = np.array([0.1])

                buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.0])
        last_done = np.array([True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer.prepare_sequence_tensors()

        # Should have 3 sequences, each of length 2
        assert buffer.num_sequences == 3
        assert buffer.max_sequence_length == 2

    def test_minibatch_sequences_remainder_handling(self):
        """Test minibatch sequences with remainder sequences."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=15,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
            max_seq_len=2,
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add 5 short episodes (will create 5 sequences)
        for episode in range(5):
            for i in range(2):
                obs = np.array([[episode * 10 + i, episode * 10 + i + 1]])
                action = np.array([i % 2])
                reward = np.array([1.0])
                done = np.array([i == 1])
                value = np.array([0.5])
                log_prob = np.array([0.1])

                buffer.add(obs, action, reward, done, value, log_prob)

        # Compute returns
        last_value = np.array([0.0])
        last_done = np.array([True])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Prepare sequence tensors
        buffer.prepare_sequence_tensors()

        # Get minibatches with batch_size=2 (should have 2 full batches + 1 remainder)
        batch_size = 2
        minibatch_generator = buffer.get_minibatch_sequences(batch_size)

        batch_sizes = []
        for padded_batch, _unpadded_batch in minibatch_generator:
            # Count sequences in this batch by checking the batch dimension
            batch_sequences = padded_batch.batch_size[0] // buffer.max_sequence_length
            batch_sizes.append(batch_sequences)

        # Should have 2 batches of size 2 and 1 batch of size 1
        assert 2 in batch_sizes  # Full batches
        assert 1 in batch_sizes  # Remainder batch

    @pytest.mark.parametrize("recurrent", [True, False])
    def test_reset(self, recurrent):
        """Test buffer reset."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (1, 2, 4),
        }
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            max_seq_len=3,
            recurrent=recurrent,
            hidden_state_architecture=hidden_state_arch,
        )

        # Add some data
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])
        hidden_state = {"h_actor": torch.randn(1, 2, 4)}

        buffer.add(
            obs,
            action,
            reward,
            done,
            value,
            log_prob,
            hidden_state=hidden_state,
        )

        # Prepare sequences
        buffer.compute_returns_and_advantages(
            np.array([0.0, 0.0]),
            np.array([True, True]),
        )

        if recurrent:
            buffer.prepare_sequence_tensors()
        else:
            with pytest.raises(ValueError):
                buffer.prepare_sequence_tensors()

            return

        # Check that buffer has data
        assert buffer.pos > 0
        assert buffer.padded_data is not None
        assert buffer.unpadded_data is not None

        # Reset buffer
        buffer.reset()

        # Check that buffer is reset
        assert buffer.pos == 0
        assert buffer.full is False
        assert buffer.num_sequences is None
        assert buffer.max_sequence_length is None
        assert buffer.unpadded_slices is None
        assert buffer.padded_data is None
        assert buffer.unpadded_data is None

    def test_size_method(self):
        """Test size method."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=3,
            device="cpu",
        )

        assert buffer.size() == 0

        # Add one timestep
        obs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        action = np.array([0, 1, 0])
        reward = np.array([1.0, -1.0, 0.5])
        done = np.array([False, True, False])
        value = np.array([0.5, -0.5, 0.0])
        log_prob = np.array([0.1, 0.2, 0.3])

        buffer.add(obs, action, reward, done, value, log_prob)

        assert buffer.size() == 3  # 3 environments

    def test_serialization(self):
        """Test buffer serialization and deserialization."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        # Add some data
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(obs, action, reward, done, value, log_prob)

        # Get state
        state = buffer.__getstate__()
        assert isinstance(state, dict)
        assert "buffer" in state
        assert "pos" in state
        assert "full" in state

        # Create new buffer and set state
        new_buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        new_buffer.__setstate__(state)

        assert new_buffer.pos == buffer.pos
        assert new_buffer.full == buffer.full
        assert new_buffer.size() == buffer.size()

    def test_different_action_spaces(self):
        """Test with different action spaces."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Test with Box action space
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        action = np.array([[0.5, -0.5], [1.0, -1.0]])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(obs, action, reward, done, value, log_prob)

        assert buffer.pos == 1
        assert buffer.buffer["actions"].shape == (5, 2, 2)

        # Test with MultiDiscrete action space
        action_space = spaces.MultiDiscrete([2, 3])
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        action = np.array([[0, 1], [1, 2]])
        buffer.add(obs, action, reward, done, value, log_prob)

        assert buffer.pos == 1
        assert buffer.buffer["actions"].shape == (5, 2, 2)

    def test_single_environment(self):
        """Test with single environment."""
        obs_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=1,
            device="cpu",
        )

        obs = np.array([1.0, 2.0, 3.0])
        action = np.array(0)
        reward = np.array(1.0)
        done = np.array(False)
        value = np.array(0.5)
        log_prob = np.array(0.1)

        buffer.add(obs, action, reward, done, value, log_prob)

        assert buffer.pos == 1
        assert buffer.size() == 1
        assert buffer.buffer["observations"].shape == (5, 1, 3)
        assert buffer.buffer["actions"].shape == (5, 1, 1)
