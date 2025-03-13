from collections import OrderedDict

import numpy as np
import torch
from tensordict import TensorDict, tensorclass

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
class Observation:
    """TensorDict wrapper for observations."""

    value: TorchObsType

    def __post_init__(self) -> None:
        self.value = to_tensordict(self.value)


@tensorclass
class RecurrentObservation:
    """TensorDict wrapper for recurrent observations."""

    value: TorchObsType
    hidden: TorchObsType

    def __post_init__(self) -> None:
        self.value = to_tensordict(self.value)
        self.hidden = to_tensordict(self.hidden)


@tensorclass
class Transition:
    obs: Observation
    action: torch.Tensor
    next_obs: Observation
    reward: torch.Tensor
    done: torch.Tensor

    def __post_init__(self) -> None:
        self.action = self.action.to(dtype=torch.float32)
        self.done = self.done.to(dtype=torch.float32)
        self.reward = self.reward.to(dtype=torch.float32)
