from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any, cast

from torch import nn
from torch.optim import Optimizer

from agilerl.algorithms.core.registry import OptimizerFactory
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.protocols import EvolvableAlgorithmProtocol
from agilerl.typing import LrNameType, StateDict

ModuleList = list[EvolvableModule]


def init_from_multiple(
    networks: Sequence[nn.Module],
    optimizer_cls: OptimizerFactory,
    lr: float,
    optimizer_kwargs: dict[str, Any] | list[dict[str, Any]],
    lr_critic: bool = False,
    use_lora: bool = False,
) -> Optimizer:
    """Initialize an optimizer from a list of networks.

    :param networks: The list of networks that the optimizer will update.
    :type networks: Sequence[nn.Module]
    :param optimizer_cls: The optimizer class to be initialized.
    :type optimizer_cls: OptimizerFactory
    :param lr: The learning rate of the optimizer.
    :type lr: float
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer,
        or one dictionary per network.
    :type optimizer_kwargs: dict[str, Any] | list[dict[str, Any]]
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
    network: nn.Module,
    optimizer_cls: OptimizerFactory,
    lr: float,
    optimizer_kwargs: dict[str, Any],
) -> Optimizer:
    """Initialize an optimizer from a single network."""
    return optimizer_cls(network.parameters(), lr=lr, **optimizer_kwargs)


def init_llm_optimizer(
    network: nn.Module,
    optimizer_cls: OptimizerFactory,
    lr_actor: float,
    optimizer_kwargs: dict[str, Any],
    lr_critic: float | None = None,
) -> Optimizer:
    """AdamW-style optimizer with separate param groups for actor LoRA vs critic/value head."""
    for name, param in network.named_parameters():
        name_lower = name.lower()
        if ("actor" in name_lower and "lora" in name_lower) or (
            "critic" in name_lower and "lora" in name_lower
        ):
            param.requires_grad = True

    actor_params = [
        p
        for n, p in network.named_parameters()
        if "actor" in n.lower() and "lora" in n.lower() and p.requires_grad
    ]
    params: list[dict[str, Any]] = [
        {"params": actor_params, "lr": lr_actor, "group": "actor"},
    ]
    if lr_critic is not None:
        critic_params = [
            p
            for n, p in network.named_parameters()
            if (
                ("critic" in n.lower() and "lora" in n.lower() and p.requires_grad)
                or ("v_head.summary" in n.lower() and p.requires_grad)
            )
        ]
        params.append(
            {"params": critic_params, "lr": lr_critic, "group": "critic"},
        )
    return optimizer_cls(params, **optimizer_kwargs)


class OptimizerWrapper:
    """Wrapper to initialize optimizer and store metadata relevant for
    evolutionary hyperparameter optimization. In AgileRL algorithms,
    all optimizers should be initialized using this wrapper. This allows
    us to access the relevant networks that they optimize inside `Mutations`
    to be able to reinitialize them after mutating an individual.

    :param optimizer_cls: The optimizer class to be initialized.
    :type optimizer_cls: type[torch.optim.Optimizer]
    :param networks: The network/s that the optimizer will update.
    :type networks: EvolvableModule, ModuleDict
    :param lr: The learning rate of the optimizer.
    :type lr: float
    :param optimizer_kwargs: The keyword arguments to be passed to the optimizer.
    :type optimizer_kwargs: dict[str, Any]
    :param network_names: The attribute names of the networks in the parent container.
    :type network_names: list[str]
    :param lr_name: Attribute name(s) on the parent for learning rate(s): ``str``
        or ``("lr_actor", "lr_critic")`` when ``is_llm_optimizer`` is True.
    :type lr_name: str | tuple[str, str] | None
    :param is_llm_optimizer: If True, build actor/critic param groups via
        :func:`init_llm_optimizer` (single module only). Requires ``network_names``,
        ``lr_name`` as a 2-tuple, and ``lr_critic``.
    :type is_llm_optimizer: bool
    :param lr_critic: Learning rate for the critic/value-head group when
        ``is_llm_optimizer`` is True.
    :type lr_critic: float | None
    :param is_llm_optimizer: If True, the optimizer is an LLM optimizer.
    :type is_llm_optimizer: bool
    """

    # ``optimizer`` holds the initialized optimizer instance/s; ``optimizer_cls``
    # keeps the constructor/s used to build (and rebuild) them.
    optimizer: Optimizer | dict[str, Optimizer]
    optimizer_cls: OptimizerFactory | dict[str, OptimizerFactory]
    networks: list[nn.Module]
    network_names: list[str]
    lr_name: LrNameType

    def __init__(
        self,
        optimizer_cls: OptimizerFactory | dict[str, OptimizerFactory],
        networks: nn.Module | list[nn.Module],
        lr: float,
        optimizer_kwargs: dict[str, Any] | list[dict[str, Any]] | None = None,
        network_names: list[str] | None = None,
        lr_name: LrNameType | None = None,
        lr_critic: float | None = None,
        is_llm_optimizer: bool = False,
    ) -> None:

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        self.lr = lr
        self.is_llm_optimizer = is_llm_optimizer
        self.lr_critic = lr_critic

        if isinstance(networks, nn.Module):
            self.networks = [networks]
        elif isinstance(networks, list):
            if not all(isinstance(net, nn.Module) for net in networks):
                msg = "Expected a single / list of torch.nn.Module objects."
                raise TypeError(msg)
            self.networks = list(networks)
        else:
            msg = "Expected a single / list of torch.nn.Module objects."
            raise TypeError(msg)

        first_network = self.networks[0]
        if is_llm_optimizer:
            if isinstance(first_network, ModuleDict):
                msg = "is_llm_optimizer=True does not support ModuleDict networks."
                raise TypeError(msg)
            if len(self.networks) != 1:
                msg = "is_llm_optimizer=True expects exactly one network module."
                raise ValueError(msg)
            if network_names is None or lr_name is None:
                msg = (
                    "is_llm_optimizer=True requires explicit network_names and "
                    "lr_name=('lr_actor', 'lr_critic')."
                )
                raise ValueError(msg)
            self.network_names = network_names
            self.lr_name = lr_name
        elif network_names is not None:
            assert lr_name is not None, (
                "Learning rate attribute name must be passed along with the network names."
            )
            self.network_names = network_names
            self.lr_name = lr_name
        else:
            parent_container = self._infer_parent_container()
            self.network_names = self._infer_network_attr_names(parent_container)
            self.lr_name = self._infer_lr_name(parent_container)

        assert self.network_names, "No networks found in the parent container."

        # Initialize the optimizer/s
        # NOTE: For multi-agent algorithms, we want to have a different optimizer
        # for each of the networks in the passed ModuleDict since they correspond to
        # different agents.
        multiple_attrs = len(self.network_names) > 1
        if isinstance(first_network, ModuleDict):
            assert isinstance(self.optimizer_kwargs, dict), (
                "Expected a dictionary of optimizer keyword arguments for "
                "multi-agent optimizers."
            )
            optimizers: dict[str, Optimizer] = {}
            # isinstance narrowing erases the ModuleDict value type to object, so
            # pin it back to iterate members as modules.
            module_dict = cast("ModuleDict[EvolvableModule]", first_network)
            for agent_id, net in module_dict.items():
                if isinstance(optimizer_cls, dict):
                    # isinstance(dict) erases the value type, so pin it to index.
                    single_cls = cast("dict[str, OptimizerFactory]", optimizer_cls)[
                        agent_id
                    ]
                else:
                    single_cls = optimizer_cls
                kwargs = self.optimizer_kwargs.get(agent_id, {})
                optimizers[agent_id] = init_from_single(
                    net,
                    single_cls,
                    self.lr,
                    kwargs,
                )
            self.optimizer = optimizers

        elif is_llm_optimizer:
            assert isinstance(
                optimizer_cls,
                type,
            ), "Expected a single optimizer class for LLM param groups."
            assert isinstance(
                self.optimizer_kwargs,
                dict,
            ), "Expected a single dictionary of optimizer keyword arguments."
            self.optimizer = init_llm_optimizer(
                first_network,
                optimizer_cls,
                self.lr,
                self.optimizer_kwargs,
                lr_critic,
            )

        # Single-agent algorithms with multiple networks for a single optimizer
        elif len(self.networks) > 1 and multiple_attrs:
            assert len(self.networks) == len(
                self.network_names,
            ), "Number of networks and network attribute names do not match."
            assert isinstance(
                optimizer_cls,
                type,
            ), "Expected a single optimizer class for multiple networks."
            # Initialize a single optimizer from the combination of network parameters
            self.optimizer = init_from_multiple(
                self.networks,
                optimizer_cls,
                self.lr,
                self.optimizer_kwargs,
            )

        # Single-agent algorithms with a single network for a single optimizer
        else:
            assert isinstance(
                optimizer_cls,
                type,
            ), "Expected a single optimizer class for a single network."
            assert isinstance(
                self.optimizer_kwargs,
                dict,
            ), "Expected a single dictionary of optimizer keyword arguments."

            self.optimizer = init_from_single(
                first_network,
                optimizer_cls,
                self.lr,
                self.optimizer_kwargs,
            )

    def _optimizers_by_agent(self) -> dict[str, Optimizer]:
        """The per-agent optimizer mapping of a multi-agent wrapper.

        Narrowing ``self.optimizer`` in place keeps an ``Optimizer & dict``
        intersection alive, so the mapping type is pinned here once.
        """
        optimizer = self.optimizer
        assert isinstance(optimizer, dict), "Expected a dictionary of optimizers."
        return cast("dict[str, Optimizer]", optimizer)

    def _single_optimizer(self) -> Optimizer:
        """The lone optimizer of a single-agent wrapper."""
        optimizer = self.optimizer
        assert not isinstance(optimizer, dict), "Expected a single optimizer."
        return optimizer

    def __getitem__(self, agent_id: str) -> Optimizer:
        if not isinstance(self.optimizer, dict):
            msg = f"Can't access item of a single {type(self.optimizer)} object."
            raise TypeError(msg)
        return self._optimizers_by_agent()[agent_id]

    def items(self) -> list[tuple[str, Optimizer]]:
        if isinstance(self.optimizer, dict):
            return list(self._optimizers_by_agent().items())
        msg = f"Can't iterate over a single {type(self.optimizer)} object."
        raise TypeError(
            msg,
        )

    def values(self) -> list[Optimizer]:
        if isinstance(self.optimizer, dict):
            return list(self._optimizers_by_agent().values())
        msg = f"Can't iterate over a single {type(self.optimizer)} object."
        raise TypeError(
            msg,
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- dynamic proxy to wrapped optimizer
        # Never proxy dunder lookups — they must resolve against the class hierarchy,
        # not self.optimizer. Proxying them breaks copy.deepcopy / pickle, because
        # Python ends up calling e.g. self.optimizer.__getstate__ and then
        # reconstructing a wrapper from the *inner* optimizer's state. This problem only comes up during testing on python < 3.12
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            opt = object.__getattribute__(self, "optimizer")
        except AttributeError:
            msg = f"{type(self).__name__!r} object has no attribute {name!r}"
            raise AttributeError(msg) from None
        return getattr(opt, name)

    def _infer_parent_container(self) -> EvolvableAlgorithmProtocol:
        """Infer the parent container dynamically using the stack frame.

        :return: The parent container object
        """
        # Here the assumption is that OptimizerWrapper is used inside the __init__
        # method of the implemented algorithm, such that we can access the defined locals
        # and extract the corresponding attribute names to the passed networks.
        current_frame = inspect.currentframe()
        assert current_frame is not None
        caller_frame = current_frame.f_back
        assert caller_frame is not None
        algorithm_frame = caller_frame.f_back
        assert algorithm_frame is not None
        return algorithm_frame.f_locals["self"]

    def _infer_network_attr_names(
        self, container: EvolvableAlgorithmProtocol
    ) -> list[str]:
        """Infer attribute names of the networks being optimized.

        :return: List of attribute names for the networks
        """

        def _match_condition(attr_value: Any) -> bool:  # noqa: ANN401 -- arbitrary container attribute value
            return any(id(attr_value) == id(net) for net in self.networks)

        return [
            attr_name
            for attr_name, attr_value in vars(container).items()
            if _match_condition(attr_value)
        ]

    def _infer_lr_name(self, container: EvolvableAlgorithmProtocol) -> str:
        """Infer the learning rate attribute name from the parent container.

        :return: The learning rate attribute name
        """

        def _match_condition(attr_value: Any) -> bool:  # noqa: ANN401 -- arbitrary container attribute value
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
        if len(matches) > 1:
            for match in matches:
                if _check_lr_names(match):
                    return match
            msg = (
                "Multiple attributes matched with the same value as the learning rate. "
                "Please have your attribute contain 'lr' or 'learning_rate' in its name."
            )
            raise AttributeError(
                msg,
            )
        msg = "Learning rate attribute not found in the parent container."
        raise AttributeError(
            msg,
        )

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Load the state of the optimizer from the passed state dictionary.

        :param state_dict: State dictionary of the optimizer.
        :type state_dict: dict[str, Any]
        """
        if isinstance(self.networks[0], ModuleDict):
            optimizers = self._optimizers_by_agent()
            assert isinstance(
                state_dict,
                dict,
            ), (
                "Expected a dictionary of optimizer state dictionaries for multi-agent optimizers."
            )
            assert state_dict.keys() == optimizers.keys(), (
                "Expected a dictionary of optimizer state dictionaries for multi-agent optimizers."
            )
            for agent_id, opt in optimizers.items():
                opt.load_state_dict(state_dict[agent_id])
        else:
            assert isinstance(
                state_dict,
                dict,
            ), (
                "Expected a single optimizer state dictionary for single-agent optimizers."
            )

            self._single_optimizer().load_state_dict(state_dict)

    def state_dict(self) -> StateDict:
        """Return the state of the optimizer as a dictionary.

        :return: State dictionary of the optimizer.
        :rtype: StateDict
        """
        if isinstance(self.networks[0], ModuleDict):
            return {
                agent_id: opt.state_dict()
                for agent_id, opt in self._optimizers_by_agent().items()
            }

        return self._single_optimizer().state_dict()

    def optimizer_cls_names(self) -> str | dict[str, str]:
        """Return the names of the optimizers."""
        if isinstance(self.networks[0], ModuleDict):
            if isinstance(self.optimizer_cls, dict):
                # Per-agent optimizer classes; isinstance(dict) erases the value
                # type, so pin it to read each class name.
                cls_map = cast("dict[str, OptimizerFactory]", self.optimizer_cls)
                return {agent_id: cls.__name__ for agent_id, cls in cls_map.items()}
            return dict.fromkeys(
                self._optimizers_by_agent().keys(),
                self.optimizer_cls.__name__,
            )
        assert not isinstance(self.optimizer_cls, dict), (
            "Expected a single optimizer class for single-agent optimizers."
        )
        return self.optimizer_cls.__name__

    def checkpoint_dict(self, name: str) -> dict[str, Any]:
        """Return a dictionary of the optimizer's state and parameters.

        :param name: The name of the optimizer.
        :type name: str

        :return: A dictionary of the optimizer's state and parameters.
        :rtype: dict[str, Any]
        """
        out = {
            f"{name}_cls": self.optimizer_cls_names(),
            f"{name}_state_dict": self.state_dict(),
            f"{name}_networks": self.network_names,
            f"{name}_lr": self.lr_name,
            f"{name}_kwargs": self.optimizer_kwargs,
        }
        if self.is_llm_optimizer:
            out[f"{name}_is_llm_optimizer"] = True
        return out

    def zero_grad(self) -> None:
        """Zero the gradients of the optimizer."""
        if isinstance(self.optimizer, dict):
            msg = (
                "Please use the zero_grad() method of the individual optimizer in "
                "a multi-agent algorithm."
            )
            raise TypeError(msg)
        self._single_optimizer().zero_grad()

    def step(self) -> None:
        """Perform a single optimization step."""
        if isinstance(self.optimizer, dict):
            msg = (
                "Please use the step() method of the individual optimizer in "
                "a multi-agent algorithm."
            )
            raise TypeError(msg)
        self._single_optimizer().step()

    def __repr__(self) -> str:
        extra = ""
        if self.is_llm_optimizer:
            extra = f",\n    lr_critic={self.lr_critic},\n    is_llm_optimizer=True"
        return (
            f"OptimizerWrapper(\n"
            f"    optimizer={self.optimizer_cls_names()},\n"
            f"    lr={self.lr},\n"
            f"    networks={self.network_names},\n"
            f"    lr_name={self.lr_name!r},\n"
            f"    optimizer_kwargs={self.optimizer_kwargs}"
            f"{extra}\n"
            ")"
        )
