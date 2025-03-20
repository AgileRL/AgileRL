import warnings
from collections import OrderedDict
from typing import Iterator

import numpy as np
import torch
from tensordict import TensorDict, tensorclass
from torch.utils.data import IterableDataset

from agilerl.components import ReplayBuffer
from agilerl.typing import ObservationType, TorchObsType


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
            isinstance(el, (torch.Tensor, np.ndarray)) for el in data
        ), "Expected all elements of the tuple to be torch.Tensor or np.ndarray."

        new_data = OrderedDict()
        for i, el in enumerate(data):
            new_data[f"tuple_obs_{i}"] = el

        data = TensorDict(new_data)

    elif isinstance(data, dict):
        assert all(
            isinstance(el, (torch.Tensor, np.ndarray)) for el in data.values()
        ), "Expected all values of the dict to be torch.Tensor or np.ndarray."

        data = TensorDict(data)

    return data.to(dtype=dtype)


@tensorclass
class Transition:
    obs: TorchObsType
    action: torch.Tensor
    next_obs: TorchObsType
    reward: torch.Tensor
    done: torch.Tensor

    def __post_init__(self) -> None:
        self.action = self.action.to(dtype=torch.float32)
        self.done = self.done.to(dtype=torch.float32).unsqueeze(-1)
        self.reward = self.reward.to(dtype=torch.float32).unsqueeze(-1)


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
        yield self.buffer.sample(self.batch_size)
