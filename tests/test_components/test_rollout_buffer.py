import numpy as np
import pytest
import torch
from gymnasium import spaces
from tensordict import TensorDict

from agilerl.components.rollout_buffer import RolloutBuffer


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
        assert buffer.buffer["actions"].shape == (10, 2)

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
            }
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
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        hidden_state_arch = {
            "h_actor": (2, 2, 64),  # (num_layers, num_envs, hidden_size)
            "h_critic": (1, 2, 32),
        }

        buffer = RolloutBuffer(
            capacity=10,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            recurrent=True,
            hidden_state_architecture=hidden_state_arch,
        )

        assert buffer.recurrent
        assert "hidden_states" in buffer.buffer
        assert "h_actor" in buffer.buffer["hidden_states"]
        assert "h_critic" in buffer.buffer["hidden_states"]
        assert buffer.buffer["hidden_states"]["h_actor"].shape == (10, 2, 2, 64)
        assert buffer.buffer["hidden_states"]["h_critic"].shape == (10, 2, 1, 32)

    def test_init_error_recurrent_without_architecture(self):
        """Test error when recurrent=True but no hidden_state_architecture provided."""
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)

        with pytest.raises(
            ValueError, match="hidden_state_architecture must be provided"
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
        obs_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        obs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
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
        assert buffer.size() == 2
        assert torch.allclose(
            buffer.buffer["observations"][0], torch.tensor(obs, dtype=torch.float32)
        )
        assert torch.allclose(buffer.buffer["actions"][0], torch.tensor(action))
        assert torch.allclose(
            buffer.buffer["rewards"][0], torch.tensor(reward, dtype=torch.float32)
        )

    def test_add_dict_observations(self):
        """Test adding data with Dict observations."""
        obs_space = spaces.Dict(
            {
                "vector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "discrete": spaces.Discrete(3),
            }
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
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
            use_gae=True,
            gamma=0.99,
            gae_lambda=0.95,
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
        assert data["actions"].shape == (2,)

    def test_get_tensor_batch(self):
        """Test getting tensor batch."""
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

        # Get tensor data
        data = buffer.get_tensor_batch(device="cpu")

        assert isinstance(data, TensorDict)
        assert "observations" in data
        assert "actions" in data
        assert data["observations"].device.type == "cpu"
        assert isinstance(data["observations"], torch.Tensor)

    def test_get_dict_observations(self):
        """Test getting data with dict observations."""
        obs_space = spaces.Dict(
            {
                "vector": spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                "discrete": spaces.Discrete(3),
            }
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

    def test_get_sequences_basic(self):
        """Test basic sequence retrieval."""
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

        # Add multiple timesteps
        for i in range(5):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            done = np.array([False, False])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])
            hidden_state = {"h_actor": torch.randn(1, 2, 4)}

            buffer.add(
                obs, action, reward, done, value, log_prob, hidden_state=hidden_state
            )

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, False])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Test sequence methods
        sequences = buffer.get_sequences(seq_len=3, batch_size=2)
        assert isinstance(sequences, dict)

        if sequences:  # If there are sequences
            assert "observations" in sequences
            assert "actions" in sequences

    def test_get_sequence_tensor_batch(self):
        """Test getting sequence tensor batch."""
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

        # Add multiple timesteps
        for i in range(5):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            done = np.array([False, False])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])
            hidden_state = {"h_actor": torch.randn(1, 2, 4)}

            buffer.add(
                obs, action, reward, done, value, log_prob, hidden_state=hidden_state
            )

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, False])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Get sequence tensor batch
        tensor_sequences = buffer.get_sequence_tensor_batch(seq_len=3, batch_size=2)

        # Check if we got sequences (don't use boolean conversion on TensorDict)
        assert isinstance(tensor_sequences, (TensorDict, dict))

        if (
            isinstance(tensor_sequences, TensorDict)
            and len(tensor_sequences.keys()) > 0
        ):
            assert "observations" in tensor_sequences
            assert "actions" in tensor_sequences

    def test_get_specific_sequences(self):
        """Test getting specific sequences."""
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

        # Add multiple timesteps
        for i in range(5):
            obs = np.array([[i, i + 1], [i + 2, i + 3]])
            action = np.array([i % 2, (i + 1) % 2])
            reward = np.array([1.0, -1.0])
            done = np.array([False, False])
            value = np.array([0.5, -0.5])
            log_prob = np.array([0.1, 0.2])
            hidden_state = {"h_actor": torch.randn(1, 2, 4)}

            buffer.add(
                obs, action, reward, done, value, log_prob, hidden_state=hidden_state
            )

        # Compute returns
        last_value = np.array([0.8, 0.0])
        last_done = np.array([False, False])
        buffer.compute_returns_and_advantages(last_value, last_done)

        # Get specific sequences
        sequence_coords = [(0, 0), (1, 1)]  # (env_idx, time_idx)
        sequences = buffer.get_specific_sequences_tensor_batch(
            seq_len=3, sequence_coords=sequence_coords
        )

        assert isinstance(sequences, (TensorDict, dict))

        if isinstance(sequences, TensorDict) and len(sequences.keys()) > 0:
            assert "observations" in sequences
            assert "actions" in sequences

    def test_empty_sequences(self):
        """Test sequence methods with empty buffer."""
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

        # Try to get sequences from empty buffer
        with pytest.raises(ValueError):
            buffer.get_sequences(seq_len=3, batch_size=2)


class TestRolloutBufferUtilities:
    """Test utility methods and edge cases."""

    def test_reset(self):
        """Test buffer reset."""
        obs_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        buffer = RolloutBuffer(
            capacity=5,
            observation_space=obs_space,
            action_space=action_space,
            num_envs=2,
            device="cpu",
        )

        # Add data
        obs = np.array([[1.0, 2.0], [3.0, 4.0]])
        action = np.array([0, 1])
        reward = np.array([1.0, -1.0])
        done = np.array([False, True])
        value = np.array([0.5, -0.5])
        log_prob = np.array([0.1, 0.2])

        buffer.add(obs, action, reward, done, value, log_prob)

        assert buffer.pos == 1
        assert buffer.size() == 2

        # Reset
        buffer.reset()

        assert buffer.pos == 0
        assert not buffer.full
        assert buffer.size() == 0

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
        assert buffer.buffer["actions"].shape == (5, 1)
