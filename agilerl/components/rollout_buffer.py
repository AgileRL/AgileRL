import random  # Added to support random sequence sampling for BPTT
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict  # Add import

from agilerl.typing import ArrayOrTensor, ObservationType

# from agilerl.utils.utils import convert_np_to_torch_dtype # Import the utility


# Define the utility function locally to avoid circular import
def convert_np_to_torch_dtype(np_dtype):
    """Converts a numpy dtype to a torch dtype."""
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.uint8:
        return torch.uint8
    elif np_dtype == np.bool_:
        return torch.bool
    else:
        # Fallback or raise error for unhandled dtypes
        warnings.warn(f"Unhandled numpy dtype {np_dtype}, defaulting to torch.float32")
        return torch.float32


class RolloutBuffer:
    """
    Rollout buffer for collecting experiences and computing advantages for RL algorithms.
    This buffer is designed to handle vectorized environments efficiently.

    :param capacity: Maximum number of timesteps to store in the buffer (per environment).
    :type capacity: int
    :param observation_space: Observation space of the environment.
    :type observation_space: gym.spaces.Space
    :param action_space: Action space of the environment.
    :type action_space: gym.spaces.Space
    :param num_envs: Number of parallel environments.
    :type num_envs: int
    :param device: Device to store tensors on, defaults to "cpu".
    :type device: str, optional
    :param gae_lambda: Lambda parameter for GAE, defaults to 0.95.
    :type gae_lambda: float, optional
    :param gamma: Discount factor, defaults to 0.99.
    :type gamma: float, optional
    :param recurrent: Whether to store hidden states, defaults to False.
    :type recurrent: bool, optional
    :param hidden_state_size: Size of hidden states if used, defaults to None.
    :type hidden_state_size: int, optional
    :param use_gae: Whether to compute GAE advantages, defaults to True.
    :type use_gae: bool, optional
    :param wrap_at_capacity: Whether to wrap the buffer at capacity, defaults to False. This is especially useful for OFF-policy algorithms, ON-policy algorithms should leave this as False in most cases.
    :type wrap_at_capacity: bool, optional
    :param max_seq_len: Maximum sequence length for BPTT, defaults to None.
    :type max_seq_len: int, optional
    """

    hidden_states: Optional[Dict[str, np.ndarray]] = None
    next_hidden_states: Optional[Dict[str, np.ndarray]] = None

    def __init__(
        self,
        capacity: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_envs: int = 1,
        device: str = "cpu",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        recurrent: bool = False,
        hidden_state_architecture: Optional[Dict[str, Tuple[int, int, int]]] = None,
        use_gae: bool = True,
        wrap_at_capacity: bool = False,
        max_seq_len: Optional[int] = None,  # Maximum sequence length for BPTT
    ):
        self.capacity = capacity
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.recurrent = recurrent
        self.hidden_state_architecture = hidden_state_architecture
        self.use_gae = use_gae
        self.wrap_at_capacity = wrap_at_capacity
        self.max_seq_len = (
            max_seq_len  # Store maximum sequence length for BPTT sampling
        )

        self.pos = 0
        self.full = False
        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        """Initialize buffer arrays with correct shapes for vectorized environments."""
        # Determine shapes and dtypes for all expected fields
        if isinstance(self.observation_space, spaces.Discrete):
            obs_shape = (1,)
            obs_dtype = (
                torch.float32
            )  # Or torch.float32 if directly using torch tensors
        elif isinstance(self.observation_space, spaces.Box):
            obs_shape = self.observation_space.shape
            obs_dtype = convert_np_to_torch_dtype(
                self.observation_space.dtype
            )  # Convert numpy dtype to torch dtype
        elif isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            # For Dict/Tuple, we'll store them as nested TensorDicts or handle them
            # by creating multiple flat entries in the main TensorDict.
            # For now, let's assume we'll pre-allocate based on flattened or example structure.
            # This part might need refinement based on how Dict/Tuple observations are handled.
            # If obs are dicts of tensors, TensorDict can handle them naturally.
            obs_shape = ()  # Placeholder, will be determined by actual data
            obs_dtype = torch.float32  # Placeholder
        else:
            obs_shape = self.observation_space.shape
            obs_dtype = torch.float32  # Default

        if isinstance(self.action_space, spaces.Discrete):
            action_shape = ()
            action_dtype = torch.int64
        elif isinstance(self.action_space, spaces.Box):
            action_shape = self.action_space.shape
            action_dtype = convert_np_to_torch_dtype(
                self.action_space.dtype
            )  # Convert numpy dtype to torch dtype
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_shape = (len(self.action_space.nvec),)
            action_dtype = torch.int64
        elif isinstance(self.action_space, spaces.MultiBinary):
            action_shape = (self.action_space.n,)
            action_dtype = torch.int64
        else:
            try:
                action_shape = self.action_space.shape
                action_dtype = convert_np_to_torch_dtype(
                    getattr(self.action_space, "dtype", np.float32)
                )  # Convert numpy dtype to torch dtype
            except AttributeError:
                raise TypeError(
                    f"Unsupported action space type without shape: {type(self.action_space)}"
                )

        # Create a source TensorDict with appropriately sized tensors
        # The tensors will be on the CPU by default, can be moved to device later if needed.
        source_dict = {
            "observations": torch.zeros(
                (self.capacity, self.num_envs, *obs_shape), dtype=obs_dtype
            ),
            "next_observations": torch.zeros(
                (self.capacity, self.num_envs, *obs_shape), dtype=obs_dtype
            ),
            "actions": torch.zeros(
                (self.capacity, self.num_envs, *action_shape), dtype=action_dtype
            ),
            "rewards": torch.zeros((self.capacity, self.num_envs), dtype=torch.float32),
            "dones": torch.zeros((self.capacity, self.num_envs), dtype=torch.bool),
            "values": torch.zeros((self.capacity, self.num_envs), dtype=torch.float32),
            "log_probs": torch.zeros(
                (self.capacity, self.num_envs), dtype=torch.float32
            ),
            "advantages": torch.zeros(
                (self.capacity, self.num_envs), dtype=torch.float32
            ),
            "returns": torch.zeros((self.capacity, self.num_envs), dtype=torch.float32),
            "episode_starts": torch.zeros(
                (self.capacity, self.num_envs), dtype=torch.bool
            ),
        }

        if self.recurrent:
            if self.hidden_state_architecture is None:
                raise ValueError(
                    "hidden_state_architecture must be provided if recurrent=True"
                )
            # self.hidden_state_architecture is Dict[str, Tuple[num_layers, num_envs_at_ppo_init, hidden_size]]
            # For buffer storage, each hidden state key should map to a tensor of shape:
            # (capacity, num_envs, num_layers, hidden_size)
            source_dict["hidden_states"] = {
                key: torch.zeros(
                    (
                        self.capacity,
                        self.num_envs,
                        self.hidden_state_architecture[key][0],  # num_layers
                        self.hidden_state_architecture[key][2],  # hidden_size
                    ),
                    dtype=torch.float32,
                )
                for key in self.hidden_state_architecture.keys()
            }
            # `next_hidden_states` might not be strictly necessary if we only store initial hidden states
            # for sequences. If we store step-by-step next_hidden_states, it would be:
            # source_dict["next_hidden_states"] = { ... similar structure ... }

        # Initialize the main buffer as a TensorDict
        self.buffer = TensorDict(
            source_dict,
            batch_size=[self.capacity, self.num_envs],
            device="cpu",  # Keep buffer on CPU, move to device in get_tensor_batch
        )


    def add(
        self,
        obs: ObservationType,
        action: ArrayOrTensor,
        reward: Union[float, np.ndarray],
        done: Union[bool, np.ndarray],
        value: Union[float, np.ndarray],
        log_prob: Union[float, np.ndarray],
        next_obs: Optional[ObservationType] = None,
        hidden_state: Optional[Dict[str, ArrayOrTensor]] = None,
        next_hidden_state: Optional[
            Dict[str, ArrayOrTensor]
        ] = None,  # Not used if only initial hidden states are stored
        episode_start: Optional[Union[bool, np.ndarray]] = None,
    ) -> None:
        """
        Add a new batch of observations and associated data from vectorized environments to the buffer.

        :param obs: Current observation batch (shape: (num_envs, *obs_shape))
        :param action: Action batch taken (shape: (num_envs, *action_shape))
        :param reward: Reward batch received (shape: (num_envs,))
        :param done: Done flag batch (shape: (num_envs,))
        :param value: Value estimate batch (shape: (num_envs,))
        :param log_prob: Log probability batch of the actions (shape: (num_envs,))
        :param next_obs: Next observation batch (shape: (num_envs, *obs_shape)), defaults to None
        :param hidden_state: Current hidden state batch (shape: (num_envs, hidden_size)), defaults to None
        :param next_hidden_state: Next hidden state batch (shape: (num_envs, hidden_size)), defaults to None
        :param episode_start: Episode start flag batch (shape: (num_envs,)), defaults to None
        """
        if self.pos == self.capacity:
            if self.wrap_at_capacity:
                self.pos = 0
            else:
                raise ValueError(
                    f"Buffer has reached capacity ({self.capacity} transitions) but received more transitions. Either increase buffer capacity or set wrap_at_capacity=True."
                )

        # Prepare data as a dictionary of tensors for the current time step
        current_step_data = {}

        # Convert inputs to tensors and ensure correct device (CPU for buffer storage)
        # Also ensure they have the (num_envs, ...) shape

        # Observations
        if isinstance(
            obs, dict
        ):  # Assuming obs can be a dict for Dict observation spaces
            current_step_data["observations"] = TensorDict(
                {
                    k: (
                        torch.as_tensor(v, device="cpu").unsqueeze(0)
                        if self.num_envs == 1
                        else torch.as_tensor(v, device="cpu")
                    )
                    for k, v in obs.items()
                },
                batch_size=[self.num_envs],
            )
        else:
            obs_tensor = torch.as_tensor(obs, device="cpu")
            if (
                self.num_envs == 1
                and obs_tensor.ndim < len(self.observation_space.shape) + 1
            ):  # Add batch dim for single env
                obs_tensor = obs_tensor.unsqueeze(0)
            current_step_data["observations"] = obs_tensor

        # Actions
        action_tensor = torch.as_tensor(action, device="cpu")
        if (
            self.num_envs == 1 and action_tensor.ndim < len(self.action_space.shape) + 1
        ):  # Add batch dim
            action_tensor = action_tensor.unsqueeze(0)
        current_step_data["actions"] = action_tensor

        # Rewards
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device="cpu")
        current_step_data["rewards"] = reward_tensor.reshape(self.num_envs)

        # Dones
        done_tensor = torch.as_tensor(done, dtype=torch.bool, device="cpu")
        current_step_data["dones"] = done_tensor.reshape(self.num_envs)

        # Values
        value_tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
        current_step_data["values"] = value_tensor.reshape(self.num_envs)

        # Log_probs
        log_prob_tensor = torch.as_tensor(log_prob, dtype=torch.float32, device="cpu")
        current_step_data["log_probs"] = log_prob_tensor.reshape(self.num_envs)

        # Next Observations
        if next_obs is not None:
            if isinstance(next_obs, dict):
                current_step_data["next_observations"] = TensorDict(
                    {
                        k: (
                            torch.as_tensor(v, device="cpu").unsqueeze(0)
                            if self.num_envs == 1
                            else torch.as_tensor(v, device="cpu")
                        )
                        for k, v in next_obs.items()
                    },
                    batch_size=[self.num_envs],
                )
            else:
                next_obs_tensor = torch.as_tensor(next_obs, device="cpu")
                if (
                    self.num_envs == 1
                    and next_obs_tensor.ndim < len(self.observation_space.shape) + 1
                ):  # Add batch dim
                    next_obs_tensor = next_obs_tensor.unsqueeze(0)
                current_step_data["next_observations"] = next_obs_tensor

        # Episode Starts
        if episode_start is not None:
            episode_start_tensor = torch.as_tensor(
                episode_start, dtype=torch.bool, device="cpu"
            )
            current_step_data["episode_starts"] = episode_start_tensor.reshape(
                self.num_envs
            )
        else:
            current_step_data["episode_starts"] = torch.zeros(
                self.num_envs, dtype=torch.bool, device="cpu"
            )

        # Hidden States (assuming they are dictionaries of tensors)
        if self.recurrent and hidden_state is not None:
            # hidden_state is Dict[str, Tensor] from PPO -> {key: (layers, num_envs, size)}
            # We need to populate current_step_data["hidden_states"]
            # with {key: (num_envs, layers, size)}
            current_step_data["hidden_states"] = {}
            for key, ppo_tensor_val in hidden_state.items():
                current_step_data["hidden_states"][key] = ppo_tensor_val.permute(
                    1, 0, 2
                )  # Shape: (num_envs, layers, size)

        # Create a TensorDict for the current step's data
        # This will have batch_size [num_envs]
        current_td_slice = TensorDict(
            current_step_data, batch_size=[self.num_envs], device="cpu"
        )

        # Assign this slice to the buffer at the current position
        # self.buffer has batch_size [capacity, num_envs]
        # self.buffer[self.pos] should be a TensorDict with batch_size [num_envs]
        # This assignment will correctly place the data, including the nested hidden_states TD.
        self.buffer[self.pos] = current_td_slice

        # Update buffer position
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value: ArrayOrTensor, last_done: ArrayOrTensor
    ) -> None:
        """
        Compute returns and advantages for the stored experiences using GAE or Monte Carlo.

        :param last_value: Value estimate for the last observation in each environment (shape: (num_envs,))
        :param last_done: Done flag for the last state in each environment (shape: (num_envs,))
        """
        # Convert inputs to numpy arrays if they are tensors, and ensure correct shape
        if isinstance(last_value, torch.Tensor):
            last_value_np = last_value.cpu().numpy().reshape(self.num_envs)
        else:
            last_value_np = np.asarray(last_value).reshape(self.num_envs)

        if isinstance(last_done, torch.Tensor):
            last_done_np = last_done.cpu().numpy().reshape(self.num_envs)
        else:
            last_done_np = np.asarray(last_done).reshape(self.num_envs)

        buffer_size = self.capacity if self.full else self.pos

        # Temporary numpy arrays for computation, will be assigned back to TensorDict
        advantages_np = np.zeros((buffer_size, self.num_envs), dtype=np.float32)
        returns_np = np.zeros((buffer_size, self.num_envs), dtype=np.float32)

        # Get necessary data from TensorDict as numpy arrays for computation
        # Slicing to buffer_size for all components.
        rewards_np = self.buffer["rewards"][:buffer_size].numpy()
        dones_np = self.buffer["dones"][:buffer_size].numpy()
        values_np = self.buffer["values"][:buffer_size].numpy()

        if self.use_gae:
            last_gae_lambda = np.zeros(self.num_envs, dtype=np.float32)
            for t in reversed(range(buffer_size)):
                if t == buffer_size - 1:
                    next_non_terminal = 1.0 - last_done_np.astype(float)
                    next_values = last_value_np.astype(float)
                else:
                    next_non_terminal = 1.0 - dones_np[t + 1].astype(float)
                    next_values = values_np[t + 1]

                delta = (
                    rewards_np[t]
                    + self.gamma * next_values * next_non_terminal
                    - values_np[t]
                )
                advantages_np[t] = last_gae_lambda = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
                )
            returns_np = advantages_np + values_np
        else:
            # Monte Carlo returns
            last_returns_np = last_value_np.astype(float) * (
                1.0 - last_done_np.astype(float)
            )
            for t in reversed(range(buffer_size)):
                returns_np[t] = last_returns_np = rewards_np[
                    t
                ] + self.gamma * last_returns_np * (1.0 - dones_np[t].astype(float))
            advantages_np = returns_np - values_np

        # Assign computed advantages and returns back to the TensorDict
        self.buffer["advantages"][:buffer_size] = torch.from_numpy(advantages_np)
        self.buffer["returns"][:buffer_size] = torch.from_numpy(returns_np)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Get data from the buffer, flattened and optionally sampled into minibatches.

        :param batch_size: Size of the minibatch to sample. If None, returns all data. Defaults to None.
        :return: Dictionary containing flattened buffer data arrays.
        """
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            # Return empty dictionary or raise error if buffer is empty
            return {}

        # Get a view of the buffer up to the current position and for all envs
        # This slice will have batch_size [buffer_size, num_envs]
        valid_buffer_data = self.buffer[:buffer_size]

        # Reshape to flatten the num_envs dimension into the first batch dimension
        # New batch_size will be [buffer_size * num_envs]
        flattened_td = valid_buffer_data.reshape(-1)  # or .view(-1) if we want a view

        if batch_size is not None:
            if batch_size > total_samples:
                warnings.warn(
                    f"Batch size {batch_size} is larger than buffer size {total_samples}. Returning all data."
                )
                # Convert the flattened TensorDict to the old dictionary format
                # For hidden_states, we need to handle the nested TensorDict structure
                np_dict = {}
                for key, value in flattened_td.items():
                    if key == "hidden_states" and isinstance(value, TensorDict):
                        np_dict[key] = {
                            k_hid: v_hid.cpu().numpy() for k_hid, v_hid in value.items()
                        }
                    else:
                        np_dict[key] = value.cpu().numpy()
                return np_dict

            indices = np.random.choice(total_samples, size=batch_size, replace=False)
            sampled_td = flattened_td[indices]
            # Convert the sampled TensorDict to the old dictionary format
            np_dict = {}
            for key, value in sampled_td.items():
                if key == "hidden_states" and isinstance(value, TensorDict):
                    np_dict[key] = {
                        k_hid: v_hid.cpu().numpy() for k_hid, v_hid in value.items()
                    }
                else:
                    np_dict[key] = value.cpu().numpy()
            return np_dict
        else:
            # Convert the entire flattened TensorDict to the old dictionary format
            np_dict = {}
            for key, value in flattened_td.items():
                if key == "hidden_states" and isinstance(value, TensorDict):
                    # The hidden states in flattened_td might still be (total_samples, layers, size)
                    # if the hidden_states TD was not reshaped along with the main TD.
                    # This needs careful handling of how hidden_states are stored and flattened.

                    # Let's assume hidden_states inside self.buffer[key] are (capacity, num_envs, layers, size)
                    # valid_buffer_data[key] would be (buffer_size, num_envs, layers, size)
                    # flattened_td[key] would be (total_samples, layers, size)

                    np_dict[key] = {
                        # k_hid: v_hid.reshape(total_samples, *v_hid.shape[1:]).cpu().numpy() # if v_hid was (buffer_size, num_envs, ...)
                        k_hid: v_hid.cpu().numpy()  # if v_hid is already (total_samples, ...)
                        for k_hid, v_hid in value.items()  # `value` is the TensorDict for hidden_states
                    }

                else:
                    np_dict[key] = value.cpu().numpy()
            return np_dict

    def get_tensor_batch(
        self, batch_size: Optional[int] = None, device: Optional[str] = None
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Get data from the buffer as PyTorch tensors, flattened and optionally sampled.
        The output is a TensorDict.

        :param batch_size: Size of batch to sample, if None returns all data, defaults to None.
        :param device: Device to put tensors on, defaults to None (uses self.device).
        :return: TensorDict containing the data.
        """
        target_device = device or self.device
        buffer_size = self.capacity if self.full else self.pos
        total_samples = buffer_size * self.num_envs

        if total_samples == 0:
            # Return an empty TensorDict with the correct device and structure if possible,
            # or simply an empty dict if structure is complex to define when empty.
            # For now, returning an empty dict to match previous behavior.
            return {}

        # Get a view of the buffer up to the current position and for all envs
        # This slice will have batch_size [buffer_size, num_envs]
        # All tensors inside self.buffer are already torch.Tensors on CPU
        valid_buffer_data_view = self.buffer[:buffer_size]

        # Reshape to flatten the num_envs dimension into the first batch dimension
        # New batch_size will be [buffer_size * num_envs]
        # .view(-1) is crucial for not creating a copy if possible
        flattened_td = valid_buffer_data_view.reshape(-1)

        if batch_size is not None:
            if batch_size > total_samples:
                warnings.warn(
                    f"Batch size {batch_size} is larger than buffer_size {total_samples}. "
                    "Returning all data."
                )
                # Move the whole flattened_td to the target device
                return flattened_td.to(target_device)

            indices = torch.randperm(total_samples, device="cpu")[
                :batch_size
            ]  # Sample on CPU then move
            sampled_td = flattened_td[indices]
            return sampled_td.to(target_device)
        else:
            # Return all flattened data, moved to the target device
            return flattened_td.to(target_device)

    def size(self) -> int:
        """
        Get current number of transitions stored in the buffer.

        :return: Current number of transitions.
        """
        return (self.capacity if self.full else self.pos) * self.num_envs

    def reset(self) -> None:
        """Reset the buffer pointer and full flag."""
        self.pos = 0
        self.full = False

    # ------------------------------------------------------------------
    # New helper functions for truncated Backpropagation Through Time (BPTT)
    # ------------------------------------------------------------------

    def _sample_sequence_start_indices(
        self, seq_len: int, batch_size: int
    ) -> List[Tuple[int, int]]:
        """Helper to randomly sample (time_idx, env_idx) pairs that can serve as
        the starting positions for sequences of length ``seq_len``.

        :param seq_len: Desired sequence length.
        :param batch_size: Number of sequences to sample.
        :return: List of tuples (time_idx, env_idx) identifying the first element
                 of every sampled sequence.
        """
        buffer_size = self.capacity if self.full else self.pos

        if seq_len > buffer_size:
            raise ValueError(
                f"Requested sequence length {seq_len} exceeds current buffer size {buffer_size}."
            )

        # Compute valid starting indices along the time dimension for each environment
        max_start_time_idx = (
            buffer_size - seq_len
        )  # This is the max start index within one env's rollout

        if max_start_time_idx < 0:  # Not enough data in buffer for even one sequence
            return []

        valid_coords: List[Tuple[int, int]] = [
            (env_idx, t_idx)
            for env_idx in range(self.num_envs)
            for t_idx in range(max_start_time_idx + 1)
        ]

        if not valid_coords:  # Should be caught by max_start_time_idx < 0 check too
            return []

        if batch_size is None or batch_size >= len(valid_coords):
            return valid_coords  # Return all pairs if batch_size too large / None

        return random.sample(valid_coords, batch_size)

    def get_sequences(
        self,
        seq_len: int,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Returns a dictionary with batched sequences suitable for truncated BPTT.

        The returned arrays have an additional leading batch dimension of size
        ``batch_size`` and a time dimension of size ``seq_len``.

        :param seq_len: Length of each sequence in timesteps.
        :param batch_size: Number of sequences to sample. If None, returns all
                           possible sequences.
        :return: Dictionary mirroring :pyfunc:`get`, but with sequences.
        """
        # Sample starting coordinates: list of (env_idx, time_idx_in_env_rollout)
        start_coords = self._sample_sequence_start_indices(seq_len, batch_size)

        actual_batch_size = len(start_coords)

        if actual_batch_size == 0:
            return {}

        # For each key in the buffer, we need to extract sequences.
        # The buffer has shape [capacity, num_envs, ...] for most keys.
        # Or for hidden_states, it's a nested TD with tensors of shape [capacity, layers, envs, size]

        list_of_sequence_tds = []
        for env_idx, time_idx in start_coords:
            # Slice for one sequence: buffer[time_idx : time_idx + seq_len, env_idx]
            # This slice will have batch_size [seq_len]
            sequence_slice = self.buffer[time_idx : time_idx + seq_len, env_idx]
            list_of_sequence_tds.append(sequence_slice)

        # Stack these sequence TensorDicts along a new batch dimension
        # Each TD in list_of_sequence_tds has batch_size [seq_len]
        # torch.stack will create a new TD with batch_size [actual_batch_size, seq_len]
        sequences_td = torch.stack(list_of_sequence_tds, dim=0)

        # Convert the TensorDict to the old dictionary of numpy arrays format
        np_dict = {}
        for key, value in sequences_td.items():
            if key == "hidden_states" and isinstance(value, TensorDict):
                # For hidden states, we want the *initial* hidden state for each sequence.
                # The `value` here would be a TensorDict where each key (e.g., "h_actor")
                # has a tensor of shape (actual_batch_size, seq_len, layers, hidden_size)
                # We need to take the first time step [:, 0]
                # This block will be superseded by the explicit addition of "initial_hidden_states" below
                # but kept for structural reference or if full hidden sequences were needed in np_dict form.
                np_dict[key] = {
                    k_hid: v_hid.cpu().numpy() for k_hid, v_hid in value.items()
                }
            elif isinstance(
                value, TensorDict
            ):  # For nested like observations if it's a Dict space
                np_dict[key] = {
                    k_sub: v_sub.cpu().numpy() for k_sub, v_sub in value.items()
                }
            else:
                np_dict[key] = value.cpu().numpy()

        # Explicitly add initial_hidden_states if recurrent
        if self.recurrent and self.buffer.get("hidden_states", None) is not None:
            initial_hidden_states_for_np_dict = {}
            # sequences_td.get("hidden_states") is a TensorDict itself, where each value is a tensor
            # of shape (actual_batch_size, seq_len, layers, size)
            if sequences_td.get("hidden_states") is not None:
                for h_key_orig, h_val_seq_tensor in sequences_td.get(
                    "hidden_states"
                ).items():
                    initial_hidden_states_for_np_dict[h_key_orig] = (
                        h_val_seq_tensor[:, 0].cpu().numpy()
                    )  # take t=0
            np_dict["initial_hidden_states"] = initial_hidden_states_for_np_dict

            # We might also remove the full "hidden_states" from np_dict if PPO doesn't need it.
            if "hidden_states" in np_dict and "initial_hidden_states" in np_dict:
                del np_dict["hidden_states"]

        return np_dict

    def get_sequence_tensor_batch(
        self,
        seq_len: int,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Same as :pyfunc:`get_sequences` but returns PyTorch tensors on the
        specified device.

        :param seq_len: Length of each sequence.
        :param batch_size: Number of sequences to sample. If None, returns all.
        :param device: Torch device to move tensors to. Defaults to
                       :pyattr:`self.device`.
        :return: Dictionary with tensor sequences.
        """
        target_device = device or self.device
        start_coords = self._sample_sequence_start_indices(seq_len, batch_size)
        actual_batch_size = len(start_coords)

        if actual_batch_size == 0:
            return {}  # Return empty TensorDict or dict

        list_of_sequence_tds = []
        for env_idx, time_idx in start_coords:
            # Slice for one sequence: self.buffer[time_idx : time_idx + seq_len, env_idx]
            # This slice will have batch_size [seq_len] and tensors are on CPU.
            sequence_slice = self.buffer[
                time_idx : time_idx + seq_len, env_idx
            ].clone()  # Clone to make them independent for stack
            list_of_sequence_tds.append(sequence_slice)

        if not list_of_sequence_tds:  # Should be caught by actual_batch_size == 0
            return {}

        # Stack these sequence TensorDicts along a new batch dimension
        # Each TD in list_of_sequence_tds has batch_size [seq_len]
        # torch.stack will create a new TD with batch_size [actual_batch_size, seq_len]
        sequences_td = torch.stack(list_of_sequence_tds, dim=0)

        # Now, create the "initial_hidden_states" part for the output TensorDict
        # This will be a nested TensorDict with batch_size [actual_batch_size]
        # Its keys will be the hidden state keys (e.g. "h_actor"), and values will be tensors
        # of shape (actual_batch_size, num_layers, hidden_size)
        if self.recurrent and sequences_td.get("hidden_states", None) is not None:
            initial_hidden_states_source = {}
            full_hidden_sequences = sequences_td.get("hidden_states")
            for h_key, h_val_tensor_sequences in full_hidden_sequences.items():
                initial_hidden_states_source[h_key] = h_val_tensor_sequences[
                    :, 0
                ].clone()

            # Use set_non_tensor for the dictionary of initial hidden state tensors
            sequences_td.set_non_tensor(
                "initial_hidden_states", initial_hidden_states_source
            )

            if (
                "hidden_states" in sequences_td.keys()
            ):  # Exclude original full hidden_states sequence
                sequences_td = sequences_td.exclude("hidden_states")

        return sequences_td.to(target_device)

    def get_specific_sequences_tensor_batch(
        self,
        seq_len: int,
        sequence_coords: List[
            Tuple[int, int]
        ],  # List of (env_idx, time_idx_in_env_rollout)
        device: Optional[str] = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Returns a dictionary with batched sequences for specific, pre-determined
        starting coordinates, as PyTorch tensors.

        The returned arrays have a leading batch dimension (size = len(sequence_coords))
        and a time dimension of size ``seq_len``.

        :param seq_len: Length of each sequence in timesteps.
        :param sequence_coords: A list of (env_idx, time_idx) tuples.
                                Each tuple specifies the starting environment and time
                                for a sequence within that environment's rollout.
        :param device: Torch device to move tensors to. Defaults to self.device.
        :return: Dictionary with tensor sequences.
        """
        target_device = device or self.device
        actual_batch_size = len(sequence_coords)

        if actual_batch_size == 0:
            return {}  # Return empty TensorDict or dict

        list_of_sequence_tds = []
        for env_idx, time_idx in sequence_coords:
            sequence_slice = self.buffer[time_idx : time_idx + seq_len, env_idx].clone()
            list_of_sequence_tds.append(sequence_slice)

        if not list_of_sequence_tds:
            return {}

        sequences_td = torch.stack(list_of_sequence_tds, dim=0)

        if self.recurrent and sequences_td.get("hidden_states", None) is not None:
            initial_hidden_states_source = {}
            full_hidden_sequences = sequences_td.get("hidden_states")
            for h_key, h_val_tensor_sequences in full_hidden_sequences.items():
                initial_hidden_states_source[h_key] = h_val_tensor_sequences[
                    :, 0
                ].clone().to(target_device)

            # Use set_non_tensor for the dictionary of initial hidden state tensors
            sequences_td.set_non_tensor(
                "initial_hidden_states", initial_hidden_states_source
            )

            if (
                "hidden_states" in sequences_td.keys()
            ):  # Exclude original full hidden_states sequence
                sequences_td = sequences_td.exclude("hidden_states")

        return sequences_td.to(target_device)

    def __getstate__(self) -> Dict[str, Any]:
        """Gets the state dictionary for pickling, ensuring arrays are copied."""
        return self.__dict__.copy()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Sets the state dictionary when unpickling, re-initializing buffers."""
        self.__dict__.update(state)

        # Let's assume self.buffer is correctly loaded by `self.__dict__.update(state)`.
        # We should verify its integrity or re-initialize if it's somehow corrupted/not present.

        if not hasattr(self, "buffer") or not isinstance(self.buffer, TensorDict):
            warnings.warn(
                "TensorDict buffer not found or invalid during unpickling. Re-initializing."
            )
            if not hasattr(self, "observation_space") or not hasattr(
                self, "action_space"
            ):
                warnings.warn(
                    "Observation or action space missing during RolloutBuffer unpickling. Buffer might be invalid."
                )
                return
            self._initialize_buffers()  # Fallback to re-initialize if buffer is not good
        else:
            # Ensure the loaded buffer is on the correct device (CPU) as expected by internal logic
            self.buffer = self.buffer.to("cpu")
            # Verify batch_size and keys if necessary
            expected_batch_size = torch.Size([self.capacity, self.num_envs])
            if self.buffer.batch_size != expected_batch_size:
                warnings.warn(
                    f"Loaded TensorDict has batch_size {self.buffer.batch_size}, expected {expected_batch_size}. Re-initializing."
                )
                self._initialize_buffers()
