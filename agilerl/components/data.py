import warnings
from collections import OrderedDict
from collections.abc import Iterator
from numbers import Number

import numpy as np
import torch
from tensordict import TensorDict, tensorclass
from torch.utils.data import IterableDataset

from agilerl.components import ReplayBuffer
from agilerl.typing import ArrayOrTensor, MultiAgentObservationType, ObservationType


def to_tensordict(
    data: ObservationType,
    dtype: torch.dtype = torch.float32,
) -> TensorDict:
    """Convert a tuple or dict of torch.Tensor or np.ndarray to a TensorDict.

    :param data: Tuple or dict of torch.Tensor or np.ndarray.
    :type data: ObservationType
    :param dtype: Data type of the TensorDict, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: TensorDict, whether the data was a tuple or not.
    """
    if isinstance(data, tuple):
        assert all(isinstance(el, (torch.Tensor, np.ndarray, Number)) for el in data), (
            "Expected all elements of the tuple to be torch.Tensor or np.ndarray."
        )

        new_data = OrderedDict()
        for i, el in enumerate(data):
            new_data[f"tuple_obs_{i}"] = el

        data = TensorDict(new_data)

    elif isinstance(data, dict):
        assert all(
            isinstance(el, (torch.Tensor, np.ndarray, Number)) for el in data.values()
        ), "Expected all values of the dict to be torch.Tensor or np.ndarray."

        data = TensorDict(data)

    return data.to(dtype=dtype)


def to_torch_tensor(
    data: ArrayOrTensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a numpy array or Python number to a torch tensor.

    :param data: Numpy array or Python number.
    :type data: ArrayOrTensor
    :param dtype: Data type of the torch tensor, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: Torch tensor.
    """
    if isinstance(data, (np.ndarray, Number, bool)):
        return torch.tensor(data, dtype=dtype)
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    # Handle any other types by attempting to convert to tensor
    return torch.tensor(data, dtype=dtype)


@tensorclass
class Transition:
    obs: ObservationType
    action: ArrayOrTensor
    next_obs: ObservationType
    reward: ArrayOrTensor
    done: ArrayOrTensor

    def __post_init__(self) -> None:
        # Convert observations to TensorDict if they are dicts or tuples
        if isinstance(self.obs, (dict, tuple)):
            self.obs = to_tensordict(self.obs)

        if isinstance(self.next_obs, (dict, tuple)):
            self.next_obs = to_tensordict(self.next_obs)

        # Convert all data to torch tensors with proper dtype
        self.action = to_torch_tensor(self.action)
        self.done = to_torch_tensor(self.done)
        self.reward = to_torch_tensor(self.reward)

        if self.done.ndim == 0:
            self.done = self.done.unsqueeze(-1)

        if self.reward.ndim == 0:
            self.reward = self.reward.unsqueeze(-1)


def _to_agent_td(data: dict) -> TensorDict:
    """Convert a per-agent dict to a :class:`TensorDict`.

    Each value can be an array/tensor (flat obs) **or** a dict/tuple
    (dict/tuple observation space), in which case it is recursively
    converted via :func:`to_tensordict`.
    """
    converted = {}
    for agent_id, value in data.items():
        if isinstance(value, (dict, tuple)):
            converted[agent_id] = to_tensordict(value)
        else:
            converted[agent_id] = to_torch_tensor(value)
    return TensorDict(converted)


@tensorclass
class MultiAgentTransition:
    """Multi-agent analogue of :class:`Transition`.

    Each field is a ``dict[agent_id, array | dict]`` that is converted to a
    sub-:class:`TensorDict` on construction.  Dict/tuple observation spaces
    are handled automatically.

    Usage mirrors single-agent :class:`Transition`::

        transition = MultiAgentTransition(
            obs=obs, action=action, reward=reward,
            next_obs=next_obs, done=done,
        )
        td = transition.to_tensordict()
        td.batch_size = [num_envs]
        memory.add(td)
    """

    obs: MultiAgentObservationType
    action: dict[str, ArrayOrTensor]
    reward: dict[str, ArrayOrTensor]
    next_obs: MultiAgentObservationType
    done: dict[str, ArrayOrTensor]

    def __post_init__(self) -> None:
        self.obs = _to_agent_td(self.obs)
        self.next_obs = _to_agent_td(self.next_obs)
        self.action = _to_agent_td(self.action)
        self.reward = _to_agent_td(self.reward)
        self.done = _to_agent_td(self.done)


class ReplayDataset(IterableDataset):
    """Iterable Dataset containing the ReplayBuffer which will be updated with new
    experiences during training.

    :param buffer: Experience replay buffer
    :type buffer: agilerl.components.replay_buffer.ReplayBuffer()
    :param batch_size: Number of experiences to sample at a time, defaults to 256
    :type batch_size: int, optional
    """

    def __init__(self, buffer: ReplayBuffer, batch_size: int = 256) -> None:
        if not isinstance(buffer, ReplayBuffer):
            warnings.warn("Buffer is not an agilerl ReplayBuffer.", stacklevel=2)

        assert batch_size > 0, "Batch size must be greater than zero."
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        samples = self.buffer.sample(self.batch_size)
        yield samples
