import inspect
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn
from torch.optim import Optimizer

from agilerl.modules.base import EvolvableModule
from agilerl.protocols import EvolvableAlgorithm
from agilerl.typing import OptimizerType, StateDict

ModuleList = List[EvolvableModule]
_Optimizer = Union[OptimizerType, List[OptimizerType]]
_Module = Union[EvolvableModule, ModuleList, List[ModuleList]]


def init_from_multiple(
    networks: ModuleList,
    optimizer_cls: OptimizerType,
    lr: float,
    optimizer_kwargs: Dict[str, Any],
) -> Optimizer:
    """
    Initialize an optimizer from a list of networks.

    :param networks: The list of networks that the optimizer will update.
    :type networks: ModuleList
    :param optimizer_cls: The optimizer class to be initialized.
    :type optimizer_cls: OptimizerType
    :param lr: The learning rate of the optimizer.
    :type lr: float
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer.
    :type optimizer_kwargs: Dict[str, Any]
    """
    opt_args = []
    for i, net in enumerate(networks):
        kwargs = (
            optimizer_kwargs[i]
            if isinstance(optimizer_kwargs, list)
            else optimizer_kwargs
        )
        opt_args.append({"params": net.parameters(), "lr": lr, **kwargs})

    return optimizer_cls(opt_args)


def init_from_single(
    network: EvolvableModule,
    optimizer_cls: OptimizerType,
    lr: float,
    optimizer_kwargs: Dict[str, Any],
) -> Optimizer:
    """
    Initialize an optimizer from a single network.
    """
    return optimizer_cls(network.parameters(), lr=lr, **optimizer_kwargs)


class OptimizerWrapper:
    """Wrapper to initialize optimizer and store metadata relevant for
    evolutionary hyperparameter optimization. In AgileRL algorithms,
    all optimizers should be initialized using this wrapper. This allows
    us to access the relevant networks that they optimize inside `Mutations`
    to be able to reinitialize them after mutating an individual.

    :param optimizer_cls: The optimizer class to be initialized.
    :type optimizer_cls: Type[torch.optim.Optimizer]
    :param networks: The list of networks that the optimizer will update.
    :type networks: List[EvolvableModule]
    :param lr: The learning rate of the optimizer.
    :type lr: float
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer.
    :type optimizer_kwargs: Dict[str, Any]
    :param network_names: The attribute names of the networks in the parent container.
    :type network_names: List[str]
    :param lr_name: The attribute name of the learning rate in the parent container.
    :type lr_name: str
    :param multiagent: Flag to indicate if the optimizer is multi-agent.
    :type multiagent: bool
    """

    optimizer: _Optimizer

    def __init__(
        self,
        optimizer_cls: _Optimizer,
        networks: _Module,
        lr: float,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        network_names: Optional[List[str]] = None,
        lr_name: Optional[str] = None,
        multiagent: bool = False,
    ) -> None:

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.lr = lr
        self.multiagent = multiagent

        if isinstance(networks, nn.Module):
            self.networks = [networks]
        else:
            assert isinstance(networks, list) and all(
                isinstance(net, nn.Module) for net in networks
            ), "Expected a single network or a list of networks."

            self.networks = networks

        # NOTE: This should be passed when reintializing the optimizer
        # when mutating an individual.
        if network_names is not None:
            assert lr_name is not None, "Learning rate attribute name must be passed."
            self.network_names = network_names
            self.lr_name = lr_name
        else:
            parent_container = self._infer_parent_container()
            self.network_names = self._infer_network_attr_names(parent_container)
            self.lr_name = self._infer_lr_name(parent_container)

        assert self.network_names, "No networks found in the parent container."

        # Initialize the optimizer/s
        # NOTE: For multi-agent algorithms, we want to have a different optimizer
        # for each of the networks in the passed list since they correspond to
        # different agents.
        multiple_attrs = len(self.network_names) > 1
        multiple_networks = len(self.networks) > 1
        if multiagent:
            multiple_networks = isinstance(self.networks[0], list)
            self.optimizer = []
            for i, net in enumerate(self.networks):
                optimizer = (
                    optimizer_cls[i]
                    if isinstance(optimizer_cls, list)
                    else optimizer_cls
                )
                kwargs = (
                    optimizer_kwargs[i]
                    if isinstance(self.optimizer_kwargs, list)
                    else self.optimizer_kwargs
                )
                if isinstance(net, list):
                    self.optimizer.append(
                        init_from_multiple(net, optimizer, self.lr, kwargs)
                    )
                else:
                    self.optimizer.append(
                        init_from_single(net, optimizer, self.lr, kwargs)
                    )

        # Single-agent algorithms with multiple networks for a single optimizer
        elif multiple_networks and multiple_attrs:
            assert len(self.networks) == len(
                self.network_names
            ), "Number of networks and network attribute names do not match."
            assert isinstance(
                optimizer_cls, type
            ), "Expected a single optimizer class for multiple networks."
            # Initialize a single optimizer from the combination of network parameters
            self.optimizer = init_from_multiple(
                self.networks, optimizer_cls, self.lr, self.optimizer_kwargs
            )

        # Single-agent algorithms with a single network for a single optimizer
        else:
            assert isinstance(
                optimizer_cls, type
            ), "Expected a single optimizer class for a single network."
            assert isinstance(
                self.optimizer_kwargs, dict
            ), "Expected a single dictionary of optimizer keyword arguments."

            self.optimizer = init_from_single(
                self.networks[0], optimizer_cls, self.lr, self.optimizer_kwargs
            )

    def __getitem__(self, index: int) -> Optimizer:
        try:
            return self.optimizer[index]
        except TypeError:
            raise TypeError(
                f"Can't access item of a single {type(self.optimizer)} object."
            )

    def __iter__(self):
        return iter(self.optimizer)

    def __getattr__(self, name: str):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            opt = object.__getattribute__(self, "optimizer")
            return getattr(opt, name)

    def _infer_parent_container(self) -> EvolvableAlgorithm:
        """
        Infer the parent container dynamically using the stack frame.

        :return: The parent container object
        """
        # Here the assumption is that OptimizerWrapper is used inside the __init__
        # method of the implemented algorithm, such that we can access the defined locals
        # and extract the corresponding attribute names to the passed networks.
        current_frame = inspect.currentframe()
        return current_frame.f_back.f_back.f_locals["self"]

    def _infer_network_attr_names(self, container: Any) -> List[str]:
        """
        Infer attribute names of the networks being optimized.

        :return: List of attribute names for the networks
        """

        def _match_condition(attr_value: Any) -> bool:
            if not self.multiagent:
                return any(id(attr_value) == id(net) for net in self.networks)
            return id(attr_value) == id(self.networks)

        return [
            attr_name
            for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]

    def _infer_lr_name(self, container: Any) -> str:
        """
        Infer the learning rate attribute name from the parent container.

        :return: The learning rate attribute name
        """

        def _match_condition(attr_value: Any) -> bool:
            return self.lr is attr_value

        def _check_lr_names(attr_name: str) -> bool:
            return "lr" in attr_name.lower() or "learning_rate" in attr_name.lower()

        matches = [
            attr_name
            for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            for match in matches:
                if _check_lr_names(match):
                    return match
            raise AttributeError(
                "Multiple attributes matched with the same value as the learning rate. "
                "Please have your attribute contain 'lr' or 'learning_rate' in its name."
            )
        else:
            raise AttributeError(
                "Learning rate attribute not found in the parent container."
            )

    def load_state_dict(self, state_dict: StateDict) -> None:
        """
        Load the state of the optimizer from the passed state dictionary.

        :param state_dict: State dictionary of the optimizer.
        :type state_dict: Dict[str, Any]
        """
        if self.multiagent:
            assert isinstance(state_dict, list) and len(state_dict) == len(
                self.optimizer
            ), "Expected a list of optimizer state dictionaries for multi-agent optimizers."
            optimizers: List[Optimizer] = self.optimizer
            for i, opt in enumerate(optimizers):
                opt.load_state_dict(state_dict[i])
        else:
            assert isinstance(
                state_dict, dict
            ), "Expected a single optimizer state dictionary for single-agent optimizers."
            self.optimizer.load_state_dict(state_dict)

    def state_dict(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Return the state of the optimizer as a dictionary.

        :return: State dictionary of the optimizer.
        :rtype: Dict[str, Any]
        """
        if self.multiagent:
            optimizers: List[Optimizer] = self.optimizer
            return [opt.state_dict() for opt in optimizers]

        return self.optimizer.state_dict()

    def zero_grad(self) -> None:
        """
        Zero the gradients of the optimizer.
        """
        if self.multiagent:
            raise ValueError(
                "Please use the zero_grad() method of the individual optimizer in "
                "a multi-agent algorithm."
            )
        else:
            self.optimizer.zero_grad()

    def step(self) -> None:
        """
        Perform a single optimization step.
        """
        if self.multiagent:
            raise ValueError(
                "Please use the step() method of the individual optimizer in "
                "a multi-agent algorithm."
            )
        else:
            self.optimizer.step()

    def __repr__(self) -> str:
        return (
            f"OptimizerWrapper(\n"
            f"    optimizer={self.optimizer_cls.__name__},\n"
            f"    lr={self.lr},\n"
            f"    networks={self.network_names},\n"
            f"    optimizer_kwargs={self.optimizer_kwargs}\n"
            f"    multiagent={self.multiagent}\n"
            f")"
        )
