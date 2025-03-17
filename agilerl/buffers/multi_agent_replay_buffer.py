from typing import Dict, List, TypeAlias, Union

import torch
from tensordict import TensorDict
from torch import Tensor

# Define a recursive type for TensorDict contents
TensorDictValue: TypeAlias = Union[Tensor, Dict[str, Tensor], "TensorDictContainer"]
TensorDictContainer: TypeAlias = TensorDict[str, TensorDictValue]


class MultiAgentReplayBuffer:
    """A multi-agent circular replay buffer that uses TensorDict objects for storage.

    :param max_size: Maximum number of transitions to store
    :type max_size: int
    :param agent_ids: List of agent IDs
    :type agent_ids: List[str]
    :param device: Device to store the transitions on
    :type device: Union[str, torch.device], optional
    :param dtype: Data type for the tensors
    :type dtype: torch.dtype, optional
    """

    def __init__(
        self,
        max_size: int,
        agent_ids: List[str],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        assert max_size > 0, "Max size must be greater than zero."
        assert len(agent_ids) > 0, "Must specify at least one agent ID."

        self.max_size = max_size
        self.agent_ids = agent_ids
        self.device = device
        self.dtype = dtype
        self.counter = 0
        self.initialized = False

        self._cursor = 0
        self._size = 0
        self._storage = None

    @property
    def storage(self) -> TensorDictContainer:
        """Storage of the buffer."""
        return self._storage

    @property
    def size(self) -> int:
        """Number of transitions in the buffer."""
        return self._size

    @property
    def is_full(self) -> bool:
        """Whether the buffer is full."""
        return len(self) == self.max_size

    def __len__(self) -> int:
        """Returns the current size of internal memory."""
        return self._size

    def _init(self, data: TensorDictContainer, is_vectorised: bool = False) -> None:
        """Initialize the buffer given the passed data.

        :param data: Data to initialize the buffer with
        :type data: TensorDictContainer
        :param is_vectorised: Whether the data is vectorised or not
        :type is_vectorised: bool
        """
        _data = (
            data
            if not is_vectorised
            else TensorDict({}, batch_size=[], device=self.device)
        )
        for key, value in data.items():
            if is_vectorised and data.batch_size[0] > 0:
                # Take the first item from batch for initialization
                if isinstance(value, TensorDict):
                    _data[key] = TensorDict({}, batch_size=[], device=self.device)
                    for agent_id in self.agent_ids:
                        if agent_id in value:
                            _data[key][agent_id] = value[agent_id][0]
                else:
                    _data[key] = value[0]

        # Create storage TensorDict
        self._storage = TensorDict({}, batch_size=[self.max_size], device=self.device)

        # Initialize storage for each field
        for key, value in _data.items():
            if isinstance(value, TensorDict):
                # Agent-specific fields
                agent_dict = {}
                for agent_id in self.agent_ids:
                    if agent_id in value:
                        agent_value = value[agent_id]
                        if isinstance(agent_value, TensorDict):
                            # Nested TensorDict (e.g., for mixed observations)
                            nested_dict = {}
                            for nested_key, nested_val in agent_value.items():
                                # Create storage for each nested field
                                nested_dtype = (
                                    self.dtype
                                    if nested_val.dtype.is_floating_point
                                    else nested_val.dtype
                                )
                                nested_dict[nested_key] = torch.zeros(
                                    (self.max_size,) + tuple(nested_val.shape),
                                    dtype=nested_dtype,
                                    device=self.device,
                                )
                            agent_dict[agent_id] = TensorDict(
                                nested_dict,
                                batch_size=[self.max_size],
                                device=self.device,
                            )
                        else:
                            # Regular tensor
                            agent_dtype = (
                                self.dtype
                                if agent_value.dtype.is_floating_point
                                else agent_value.dtype
                            )
                            agent_dict[agent_id] = torch.zeros(
                                (self.max_size,) + tuple(agent_value.shape),
                                dtype=agent_dtype,
                                device=self.device,
                            )

                self._storage[key] = TensorDict(
                    agent_dict, batch_size=[self.max_size], device=self.device
                )
            else:
                # Non-agent-specific fields
                if value is not None:
                    dtype = self.dtype if value.dtype.is_floating_point else value.dtype
                    self._storage[key] = torch.zeros(
                        (self.max_size,) + tuple(value.shape),
                        dtype=dtype,
                        device=self.device,
                    )

        self.initialized = True

    def add(self, data: TensorDictContainer, is_vectorised: bool = False) -> None:
        """Add a transition to the buffer.

        :param data: Transition to add to the buffer
        :type data: TensorDictContainer
        :param is_vectorised: Whether the data is vectorised or not
        :type is_vectorised: bool
        """
        # Convert to device
        data = data.to(self.device)

        # Initialize storage if needed
        if self._storage is None:
            self._init(data, is_vectorised)

        # Determine batch size and indices
        if is_vectorised:
            batch_size = data.batch_size[0]
        else:
            # Add batch dimension if not vectorized
            data = data.unsqueeze(0)
            batch_size = 1

        # Add to storage considering circularity of buffer
        start = self._cursor
        end = (self._cursor + batch_size) % self.max_size

        # Handle wrap-around case
        if start + batch_size > self.max_size:
            # Number of items that fit before the end
            num_first_part = self.max_size - start
            # Number of items that need to wrap to beginning
            num_second_part = batch_size - num_first_part

            # Split and store both parts
            self._store_batch(data[:num_first_part], start, start + num_first_part)
            if num_second_part > 0:
                self._store_batch(data[num_first_part:], 0, num_second_part)
        else:
            # No wrap-around needed
            self._store_batch(data, start, start + batch_size)

        # Update cursor and size
        self._cursor = end
        self._size = min(self._size + batch_size, self.max_size)
        self.counter += batch_size

    def _store_batch(
        self, data: TensorDictContainer, start_idx: int, end_idx: int
    ) -> None:
        """Store a batch of data in the buffer.

        :param data: Data to store
        :type data: TensorDictContainer
        :param start_idx: Start index in the buffer
        :type start_idx: int
        :param end_idx: End index in the buffer (exclusive)
        :type end_idx: int
        """
        for key, value in data.items():
            if isinstance(value, TensorDict):
                # Handle agent-specific data
                for agent_id in self.agent_ids:
                    if agent_id in value:
                        agent_value = value[agent_id]
                        if isinstance(agent_value, TensorDict):
                            # Handle nested TensorDict
                            for nested_key, nested_val in agent_value.items():
                                self._storage[key][agent_id][nested_key][
                                    start_idx:end_idx
                                ] = nested_val
                        else:
                            # Handle regular tensor
                            self._storage[key][agent_id][
                                start_idx:end_idx
                            ] = agent_value
            else:
                # Handle non-agent-specific data
                self._storage[key][start_idx:end_idx] = value

    def sample(self, batch_size: int, return_idx: bool = False) -> TensorDictContainer:
        """Sample a batch of transitions.

        :param batch_size: Number of samples to return
        :type batch_size: int
        :param return_idx: Boolean flag to return index of samples randomly selected, defaults to False
        :type return_idx: bool, optional
        :return: TensorDict containing sampled experiences
        :rtype: TensorDictContainer
        """
        if batch_size > self._size:
            batch_size = self._size

        if self._size == 0:
            return TensorDict({}, batch_size=[0], device=self.device)

        indices = torch.randint(0, self._size, (batch_size,), device=self.device)
        samples = TensorDict({}, batch_size=[batch_size], device=self.device)

        # Extract data for each field and agent
        for key, value in self._storage.items():
            if isinstance(value, TensorDict):
                # Handle agent-specific data
                agent_dict = {}
                for agent_id in self.agent_ids:
                    if agent_id in value:
                        agent_value = value[agent_id]
                        if isinstance(agent_value, TensorDict):
                            # Handle nested TensorDicts
                            nested_dict = {}
                            for nested_key, nested_val in agent_value.items():
                                nested_dict[nested_key] = nested_val[indices]
                            agent_dict[agent_id] = TensorDict(
                                nested_dict, batch_size=[batch_size], device=self.device
                            )
                        else:
                            # Handle regular tensors
                            agent_dict[agent_id] = agent_value[indices]

                samples[key] = TensorDict(
                    agent_dict, batch_size=[batch_size], device=self.device
                )
            else:
                # Handle non-agent-specific data
                samples[key] = value[indices]

        if return_idx:
            samples["idxs"] = indices

        return samples

    def save_to_memory_single_env(self, *args) -> None:
        """Legacy method for compatibility with previous API.

        :param args: Variable length argument list, expected to be a single TensorDict
        """
        if len(args) == 1 and isinstance(args[0], TensorDict):
            self.add(args[0], is_vectorised=False)
        else:
            raise TypeError("Expected a single TensorDict argument")

    def save_to_memory_vect_envs(self, *args) -> None:
        """Legacy method for compatibility with previous API.

        :param args: Variable length argument list, expected to be a single TensorDict
        """
        if len(args) == 1 and isinstance(args[0], TensorDict):
            self.add(args[0], is_vectorised=True)
        else:
            raise TypeError("Expected a single TensorDict argument")

    def save_to_memory(
        self, data: TensorDictContainer, is_vectorised: bool = False
    ) -> None:
        """Alias for add method for compatibility with previous API.

        :param data: Transition to add to the buffer
        :type data: TensorDictContainer
        :param is_vectorised: Whether the data is vectorised or not
        :type is_vectorised: bool
        """
        self.add(data, is_vectorised)

    def clear(self) -> None:
        """Clear all transitions from the buffer."""
        self._size = 0
        self._cursor = 0
        self._storage = None
        self.initialized = False
