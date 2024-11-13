from typing import Union, Dict

import torch
from numpy.typing import ArrayLike

ArrayOrTensor = Union[ArrayLike, torch.Tensor]
TensorDict = Dict[str, torch.Tensor]