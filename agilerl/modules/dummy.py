from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from agilerl.modules import EvolvableModule
from agilerl.typing import DeviceType


def to_evolvable(
    module_fn: Callable[[], nn.Module],
    module_kwargs: dict[str, Any],
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
    :type module_kwargs: dict[str, Any]
    :param device: Device to run the module on.
    :type device: DeviceType
    """

    def __init__(
        self,
        device: DeviceType,
        module: nn.Module | None = None,
        module_fn: Callable[[], nn.Module] | None = None,
        module_kwargs: Optional[dict[str, Any]] | None = None,
    ) -> None:

        if module is None and module_fn is None:
            raise ValueError("Either module or module_fn must be provided.")

        if module_fn is not None and module_kwargs is None:
            module_kwargs = {}

        if module is None:
            module = module_fn(**module_kwargs).to(device)
        else:
            module = module.to(device)
        # Initialize the module
        super().__init__(device)
        self.module = module
        self.module_fn = module_fn
        self.module_kwargs = module_kwargs

    def change_activation(self, activation: str, output: bool) -> None:
        return

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        module = self.__dict__["_modules"]["module"]
        if name == "module":
            return module
        return getattr(module, name)
