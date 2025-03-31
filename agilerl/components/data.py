import warnings
from collections import OrderedDict
from numbers import Number
from typing import Iterator

import numpy as np
import torch
from tensordict import TensorDict, tensorclass
from torch.utils.data import IterableDataset

from agilerl.components import ReplayBuffer
from agilerl.typing import ArrayOrTensor, ObservationType


def to_tensordict(
    data: ObservationType, dtype: torch.dtype = torch.float32
) -> TensorDict:
    """Converts a tuple or dict of torch.Tensor or np.ndarray to a TensorDict.

    :param data: Tuple or dict of torch.Tensor or np.ndarray.
    :type data: ObservationType
    :param dtype: Data type of the TensorDict, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: TensorDict, whether the data was a tuple or not.
    """
    if isinstance(data, tuple):
        assert all(
            isinstance(el, (torch.Tensor, np.ndarray, Number)) for el in data
        ), "Expected all elements of the tuple to be torch.Tensor or np.ndarray."

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


def to_torch_tensor(data: ArrayOrTensor, dtype=torch.float32) -> torch.Tensor:
    """Converts a numpy array or Python number to a torch tensor.

    :param data: Numpy array or Python number.
    :type data: ArrayOrTensor
    :param dtype: Data type of the torch tensor, defaults to torch.float32
    :type dtype: torch.dtype, optional
    :return: Torch tensor.
    """
    if isinstance(data, (np.ndarray, Number, bool, np.bool_)):
        return torch.tensor(data, dtype=dtype)
    elif isinstance(data, torch.Tensor):
        return data.to(dtype=dtype)
    else:
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


class ReplayDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer which will be updated with new
    experiences during training

    :param buffer: Experience replay buffer
    :type buffer: agilerl.components.replay_buffer.ReplayBuffer()
    :param batch_size: Number of experiences to sample at a time, defaults to 256
    :type batch_size: int, optional
    """

    def __init__(self, buffer: ReplayBuffer, batch_size: int = 256) -> None:
        if not isinstance(buffer, ReplayBuffer):
            warnings.warn("Buffer is not an agilerl ReplayBuffer.")

        assert batch_size > 0, "Batch size must be greater than zero."
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        samples = self.buffer.sample(self.batch_size)
        yield samples
