from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from agilerl.modules import EvolvableModule
from agilerl.typing import DeviceType


def to_evolvable(
    module_fn: Callable[[], nn.Module],
    module_kwargs: Dict[str, Any],
    device: DeviceType,
) -> EvolvableModule:
    return DummyEvolvable(module_fn, module_kwargs, device)


class DummyEvolvable(EvolvableModule):
    """Wrapper to convert a PyTorch nn.Module into an EvolvableModule object.

    .. note::
        This doesn't actually allow the user to mutate its architecture, but rather allows us
        to use nn.Module objects as networks in an ``EvolvableAlgorithm``. If a user wants to
        mutate the architecture of a network, they should create their network using the ``EvolvableModule``
        class hierarchy directly. Please refer to the documentation for more information on how to do this.

    :param module_fn: Function that returns a PyTorch nn.Module object.
    :type module_fn: Callable[[], nn.Module]
    :param module_kwargs: Keyword arguments to pass to the module_fn.
    :type module_kwargs: Dict[str, Any]
    :param device: Device to run the module on.
    :type device: DeviceType
    """

    def __init__(
        self,
        module_fn: Callable[[], nn.Module],
        module_kwargs: Dict[str, Any],
        device: DeviceType,
    ) -> None:

        # Initialize the module
        module = module_fn(**module_kwargs).to(device)

        super().__init__(device)

        self.module_fn = module_fn
        self.module_kwargs = module_kwargs
        self.module = module

    def change_activation(self, activation: str, output: bool) -> None:
        return

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def generate(self, *args, **kwargs) -> torch.Tensor:
        if not hasattr(self.module, "generate"):
            raise AttributeError(
                f"Module {self.module_fn} does not have a generate method."
            )
        return self.module.generate(*args, **kwargs)
