# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import inspect
import warnings
from copy import deepcopy
from dataclasses import asdict
from typing import Any, ClassVar, Protocol, TypeVar, overload, runtime_checkable

import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from agilerl.modules import (
    EvolvableCNN,
    EvolvableLSTM,
    EvolvableMLP,
    EvolvableMultiInput,
    EvolvableResNet,
    EvolvableSimBa,
)
from agilerl.modules.base import EvolvableModule, ModuleMeta, mutation
from agilerl.protocols import MutationType
from agilerl.typing import (
    DeviceType,
    MutationApplyDict,
    NetConfigType,
    TorchObsType,
)
from agilerl.utils.algo_utils import get_hidden_states_shape_from_model
from agilerl.utils.evolvable_networks import get_default_encoder_config, is_image_space

SelfEvolvableNetwork = TypeVar("SelfEvolvableNetwork", bound="EvolvableNetwork")
ModuleT = TypeVar("ModuleT", bound=nn.Module)


@runtime_checkable
class SupportsNumOutputs(Protocol):
    """An encoder that reports its output width via ``num_outputs``."""

    num_outputs: int


DefaultEncoderType = (
    EvolvableCNN | EvolvableMLP | EvolvableMultiInput | EvolvableSimBa | EvolvableLSTM
)


def preserve_parameters(old_net: nn.Module, new_net: ModuleT) -> ModuleT:
    """Copy compatible parameters from ``old_net`` into ``new_net`` and return it.

    :param old_net: Old neural network to copy parameters from.
    :type old_net: nn.Module
    :param new_net: New neural network to copy parameters into.
    :type new_net: ModuleT
    :return: The new network with copied parameters.
    :rtype: ModuleT
    """
    EvolvableModule.preserve_parameters(old_net, new_net)
    return new_net


def assert_correct_mlp_net_config(net_config: NetConfigType) -> None:
    """Assert that the MLP network configuration is correct.

    :param net_config: Configuration of the MLP network.
    :type net_config: dict[str, Any]
    """
    assert "hidden_size" in net_config, "Net config must contain hidden_size: int."
    assert isinstance(
        net_config["hidden_size"],
        list,
    ), "Net config hidden_size must be a list."
    assert len(net_config["hidden_size"]) > 0, (
        "Net config hidden_size must contain at least one element."
    )


def assert_correct_simba_net_config(net_config: NetConfigType) -> None:
    """Assert that the MLP network configuration is correct.

    :param net_config: Configuration of the MLP network.
    :type net_config: dict[str, Any]
    """
    assert "hidden_size" in net_config, "Net config must contain hidden_size: int."
    assert isinstance(
        net_config["hidden_size"],
        (int, np.int64),
    ), "Net config hidden_size must be an integer."
    assert "num_blocks" in net_config, "Net config must contain num_blocks: int."
    assert isinstance(
        net_config["num_blocks"],
        int,
    ), "Net config num_blocks must be an integer."


def assert_correct_cnn_net_config(net_config: NetConfigType) -> None:
    """Assert that the CNN network configuration is correct.

    :param net_config: Configuration of the CNN network.
    :type net_config: dict[str, Any]
    """
    for key in [
        "channel_size",
        "kernel_size",
        "stride_size",
    ]:
        assert key in net_config, f"Net config must contain {key}: int."
        assert isinstance(net_config[key], list), f"Net config {key} must be a list."
        assert len(net_config[key]) > 0, (
            f"Net config {key} must contain at least one element."
        )

        if key == "kernel_size":
            assert isinstance(
                net_config[key],
                (int, tuple, list),
            ), "Kernel size must be of type int, list, or tuple."


def assert_correct_lstm_net_config(net_config: NetConfigType) -> None:
    """Assert that the LSTM network configuration is correct.

    :param net_config: Configuration of the LSTM network.
    :type net_config: dict[str, Any]
    """
    assert "hidden_state_size" in net_config, (
        "LSTM net config must contain hidden_state_size: int."
    )
    assert isinstance(net_config["hidden_state_size"], (int, np.int64)), (
        "LSTM net config hidden_state_size must be an integer but is "
        + str(type(net_config["hidden_state_size"]))
        + "."
    )


# TODO: Need to think of a way to do this check without the metaclass
class NetworkMeta(ModuleMeta):
    """Metaclass for evolvable networks. Checks that the network has
    an encoder and a head_net (named as such).
    """

    def __call__(
        cls: type[SelfEvolvableNetwork],
        *args: Any,
        **kwargs: Any,
    ) -> SelfEvolvableNetwork:
        instance: SelfEvolvableNetwork = super().__call__(*args, **kwargs)

        # Check that the mutation methods of the network are correctly defined
        # i.e. only contain underlying methods corresponding to the encoder and head_net
        for mut_method in instance.mutation_methods:
            if "." in mut_method:
                attr = mut_method.split(".")[0]
                if attr not in ["encoder", "head_net"]:
                    msg = (
                        "Mutation methods must reference only 'encoder' or 'head_net' "
                        "so mutations apply consistently across networks (e.g. actor and critic)."
                    )
                    raise AttributeError(
                        msg,
                    )

        return instance


class EvolvableNetwork(EvolvableModule, metaclass=NetworkMeta):
    """Base class for evolvable networks, i.e., evolvable modules that are configured in
    a specific way for a reinforcement learning algorithm, similar to how CNNs are used
    as building blocks in ResNet, VGG, etc. An evolvable network automatically inspects the passed
    observation space to determine the appropriate encoder to build through the AgileRL
    evolvable modules, inheriting the mutation methods of any nested evolvable modules.

    .. note::
        Currently, evolvable networks should only have the encoder (which, if not specified by the user,
        is automatically built from the observation space) and a 'head_net' attribute that processes the
        latent encodings into the desired number of outputs as evolvable components. For example, in
        ``RainbowQNetwork``, we disable mutations for the advantage net and apply the same mutations to it
        as the 'value' net, which is the network head in this case. Users should follow the same philosophy.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: str | type[EvolvableModule] | None
    :param encoder_config: Configuration of the encoder. Defaults to None.
    :type encoder_config: ConfigType | None
    :param action_space: Action space of the environment. Defaults to None.
    :type action_space: spaces.Space | None
    :param min_latent_dim: Minimum dimension of the latent space representation. Defaults to 8.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation. Defaults to 128.
    :type max_latent_dim: int
    :param latent_dim: Dimension of the latent space representation. Defaults to 32.
    :type latent_dim: int
    :param simba: If True, use a SimBa network for the encoder for vector spaces. Defaults to False.
    :type simba: bool
    :param recurrent: If True, use a recurrent network for 2D observations. Defaults to False, whereby
        the encoder is a nn.Flatten() followed by an `EvolvableMLP`.
    :type recurrent: bool
    :param device: Device to use for the network. Defaults to "cpu".
    :type device: DeviceType
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: int | None
    """

    encoder: EvolvableModule
    head_net: EvolvableModule
    encoder_cls: type[EvolvableModule] | None

    # Custom encoder aliases
    _encoder_aliases: ClassVar[dict[str, type[EvolvableModule]]] = {
        "ResNet": EvolvableResNet,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        encoder_cls: str | type[EvolvableModule] | None = None,
        encoder_config: NetConfigType | None = None,
        encoder_name: str = "encoder",
        action_space: spaces.Space | None = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        latent_dim: int = 64,
        simba: bool = False,
        recurrent: bool = False,
        device: DeviceType = "cpu",
        random_seed: int | None = None,
        encoder: EvolvableModule | None = None,
    ) -> None:
        super().__init__(device, random_seed)

        # A pre-built encoder fixes the latent width (and disables latent-space
        # mutations below), so these bounds only constrain a built encoder.
        if encoder is None:
            assert latent_dim <= max_latent_dim, (
                "Latent dimension must be less than or equal to max latent dimension."
            )
            assert latent_dim >= min_latent_dim, (
                "Latent dimension must be greater than or equal to min latent dimension."
            )

        if encoder_config is None:
            encoder_config = get_default_encoder_config(
                observation_space,
                simba=simba,
                recurrent=recurrent,
            )

        # Resolve encoder class aliases (e.g. "ResNet") to the actual class
        if isinstance(encoder_cls, str):
            encoder_cls = self._encoder_aliases[encoder_cls]

        if encoder_cls is not None and not issubclass(encoder_cls, EvolvableModule):
            msg = "Encoder class must be a subclass of EvolvableModule."
            raise TypeError(msg)

        # A pre-built encoder takes precedence over `encoder_cls` (which clone
        # round-trips as ``type(encoder)``) and fixes the latent width.
        if encoder is not None:
            if not isinstance(encoder, SupportsNumOutputs):
                msg = (
                    "A pre-built `encoder` must expose an integer `num_outputs` "
                    "(e.g. a MakeEvolvable network or an AgileRL evolvable module)."
                )
                raise TypeError(msg)
            latent_dim = encoder.num_outputs

        self.observation_space = observation_space
        self.action_space = action_space
        self.latent_dim = latent_dim
        self.min_latent_dim = min_latent_dim
        self.max_latent_dim = max_latent_dim
        self.encoder_cls = type(encoder) if encoder is not None else encoder_cls
        self._encoder_injected = encoder is not None
        self.device = device
        self.simba = simba
        self.recurrent = recurrent
        self.flatten_obs = False
        self.encoder_name = encoder_name
        self.cached_hidden_state = None

        # By default we use same activation for encoder output as for the rest of the network
        output_activation = encoder_config.get("output_activation", None)
        if output_activation is None:
            activation = encoder_config.get("activation", "ReLU")
            encoder_config["output_activation"] = activation

        if encoder is not None:
            # A user-supplied encoder owns its architecture: adopt it directly
            # and disable latent-space mutations, which would otherwise resize an
            # output width we do not control.
            self.encoder = encoder
            self.filter_mutation_methods("latent")
        elif encoder_cls is not None:
            # Check if encoder config contains `num_outputs` as input argument, in which
            # case we can enable latent space mutations. Otherwise, we disable them.
            input_args = inspect.getfullargspec(encoder_cls.__init__).args
            if "num_outputs" not in input_args:
                warnings.warn(
                    f"{encoder_cls.__name__} does not contain `num_outputs` as an "
                    "input argument. Disabling latent space mutations. Make sure to set the number of "
                    "outputs to the latent dimension in the encoder configuration.",
                    stacklevel=2,
                )
                self.filter_mutation_methods("latent")
            else:
                encoder_config["num_outputs"] = self.latent_dim

            # Concrete encoder constructors take arbitrary config kwargs, unlike
            # the (device, random_seed) signature of the EvolvableModule base
            # class; type the factory as Any to admit them. The single mapping
            # (rather than explicit kwargs) lets ``encoder_config`` override the
            # defaults without a duplicate-keyword error.
            encoder_factory: Any = encoder_cls
            self.encoder = encoder_factory(
                **{
                    "observation_space": self.observation_space,
                    "num_outputs": self.latent_dim,
                    "device": self.device,
                    "name": self.encoder_name,
                    **encoder_config,
                },
            )
        else:
            self.encoder = self._build_encoder(encoder_config)

        # NOTE: We disable layer mutations for the encoder since this usually adds a lot
        # of variance to the optimization process
        self.encoder.disable_mutations(MutationType.LAYER)

    def get_init_dict(self) -> dict[str, Any]:
        """Constructor arguments for the network.

        :return: The dictionary of constructor arguments.
        :rtype: dict[str, Any]
        """
        init_dict = super().get_init_dict()
        if "encoder" in init_dict and not self._encoder_injected:
            init_dict["encoder"] = None
        return init_dict

    @property
    def encoder_config(self) -> dict[str, Any]:
        """Net configuration for encoder.

        :return: Initial dictionary for the network.
        :rtype: dict[str, Any]
        """
        return (
            self.encoder.net_config
            if self.encoder_cls is None
            else self.encoder.init_dict
        )

    @property
    def head_config(self) -> dict[str, Any]:
        """Net configuration for head.

        :return: Initial dictionary for the network.
        :rtype: dict[str, Any]
        """
        if not hasattr(self, "head_net"):
            msg = "Network does not have a head_net attribute."
            raise AttributeError(msg)

        return self.head_net.net_config

    @property
    def activation(self) -> str | None:
        """Activation function of the network.

        :return: Activation function.
        :rtype: str | None
        """
        return self.encoder.activation

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the network."""
        return self.forward(*args, **kwargs)

    @overload
    def extract_features(
        self,
        x: TorchObsType,
        hidden_state: None = None,
    ) -> torch.Tensor: ...

    @overload
    def extract_features(
        self,
        x: TorchObsType,
        hidden_state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    def extract_features(
        self,
        x: TorchObsType,
        hidden_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Extract features from the encoder part of the network.

        :param x: Input observation to extract features from
        :type x: TorchObsType
        :param hidden_state: Hidden states for recurrent networks (unused in non-recurrent networks)
        :type hidden_state: dict[str, torch.Tensor], optional
        :return: The encoded features, and (for recurrent networks) the next
            hidden-state dict if a hidden state was passed
        :rtype: torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]
        """
        # For compatibility with both recurrent and non-recurrent networks
        if hidden_state is None:
            return self.encoder(x)
        return self.encoder(x, hidden_state=hidden_state)

    def forward_head(
        self,
        latent: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass of the network head using pre-computed latent encodings.

        :param latent: Latent encodings from the encoder.
        :type latent: torch.Tensor

        :return: Output of the network head.
        :rtype: torch.Tensor
        """
        if not hasattr(self, "head_net"):
            msg = "Network does not have a head_net attribute."
            raise AttributeError(msg)

        return self.head_net(latent, *args, **kwargs)

    def build_network_head(self, net_config: NetConfigType) -> None:
        """Build the head of the network.

        :param net_config: Configuration of the network head.
        :type net_config: NetConfigType
        """
        msg = (
            "Method build_network_head must be implemented in EvolvableNetwork objects."
        )
        raise NotImplementedError(
            msg,
        )

    # NOTE: The base ``EvolvableModule.recreate_network(**kwargs)`` accepts arbitrary
    # kwargs because the mutation wrapper in agilerl/modules/base.py inspects each
    # override's signature and only forwards the kwargs it declares. Networks declare
    # exactly the arguments they accept (none), hence the narrower signature.
    def recreate_network(self) -> None:
        """Recreate the network after a mutation. Implemented by subclasses."""
        super().recreate_network()

    def create_mlp(
        self,
        num_inputs: int,
        num_outputs: int,
        name: str,
        net_config: NetConfigType,
    ) -> EvolvableMLP:
        """Build the head of the network based on the passed configuration.

        :param num_inputs: Number of inputs to the network head.
        :type num_inputs: int
        :param num_outputs: Number of outputs of the network head.
        :type num_outputs: int
        :param name: Name of the network head.
        :type name: str
        :param net_config: Configuration of the network head.
        :type net_config: dict[str, Any]

        :return: Network head.
        :rtype: EvolvableMLP
        """
        return EvolvableMLP(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            device=self.device,
            name=name,
            **net_config,
        )

    def init_weights_gaussian(
        self,
        std_coeff: float = 4.0,
        output_coeff: float = 2.0,
    ) -> None:
        """Initialize the weights of the network with a Gaussian distribution.

        :param std_coeff: Coefficient for the standard deviation of the Gaussian distribution, defaults to 4.0
        :type std_coeff: float, optional
        :param output_coeff: Coefficient for the standard deviation of the Gaussian distribution for the output layer, defaults to 2.0
        :type output_coeff: float, optional
        """
        # Initialize the weights of the encoder
        self.encoder.init_weights_gaussian(std_coeff=std_coeff)

        # Initialize weights of network heads
        # NOTE: We assume the head is an instance of EvolvableMLP
        for attr, module in self.evolvable_modules().items():
            if attr != "encoder":
                module.init_weights_gaussian(
                    std_coeff=std_coeff,
                    output_coeff=output_coeff,
                )

    def initialize_hidden_state(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Initialize the hidden state for the network.

        :param env: The environment to initialize the hidden state for
        :type env: GymEnvType
        """
        if self.recurrent:
            # if the hidden state is not initialized, initialize it
            if (
                self.cached_hidden_state is None
                or len(self.cached_hidden_state) == 0
                or self.cached_hidden_state_batch_size != batch_size
            ):
                self.cached_hidden_state = {}
                self.cached_hidden_state_batch_size = batch_size
                for name, shape in get_hidden_states_shape_from_model(
                    self.encoder,
                ).items():
                    # Replace the BatchDimension sentinel with the runtime batch size
                    shape = tuple(
                        x if isinstance(x, int) else batch_size for x in shape
                    )
                    self.cached_hidden_state[name] = torch.zeros(shape).to(self.device)
            return deepcopy(self.cached_hidden_state)
        msg = "Cannot initialize hidden state for non-recurrent networks."
        raise ValueError(
            msg,
        )

    def change_activation(self, activation: str, output: bool = False) -> None:
        """Change the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: If True, change the output activation function, defaults to False
        :type output: bool, optional
        """
        for attr, module in self.evolvable_modules().items():
            _output = True if attr == "encoder" else output
            module.change_activation(activation, output=_output)

    @mutation(MutationType.NODE)
    def add_latent_node(self, numb_new_nodes: int | None = None) -> MutationApplyDict:
        """Add a latent node to the network.

        :param numb_new_nodes: Number of new nodes to add, defaults to None
        :type numb_new_nodes: int, optional

        :return: Configuration for adding a latent node.
        :rtype: dict[str, Any]
        """
        if numb_new_nodes is None:
            numb_new_nodes = int(self.rng.choice([8, 16, 32]))

        if self.latent_dim + numb_new_nodes < self.max_latent_dim:
            self.latent_dim += numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_latent_node(
        self, numb_new_nodes: int | None = None
    ) -> MutationApplyDict:
        """Remove a latent node from the network.

        :param numb_new_nodes: Number of nodes to remove, defaults to None
        :type numb_new_nodes: int, optional

        :return: Configuration for removing a latent node.
        :rtype: dict[str, Any]
        """
        if numb_new_nodes is None:
            numb_new_nodes = int(self.rng.choice([8, 16, 32]))

        if self.latent_dim - numb_new_nodes > self.min_latent_dim:
            self.latent_dim -= numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    def recreate_encoder(self) -> None:
        """Recreate the encoder of the network."""
        if self.encoder_cls is not None:
            # Need to change `num_outputs` to the latent dimension after a mutation
            init_dict = self.encoder.init_dict
            init_dict["num_outputs"] = self.latent_dim
            encoder = self.encoder_cls(**init_dict)
        else:
            encoder = self._build_encoder(self.encoder.net_config)

        self.encoder = preserve_parameters(self.encoder, encoder)

    def _build_encoder(self, net_config: NetConfigType) -> DefaultEncoderType:
        """Build the encoder for the network based on the environments observation space.

        :return: Encoder module.
        :rtype: EvolvableModule
        """
        net_config = net_config if isinstance(net_config, dict) else asdict(net_config)

        if isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            encoder = EvolvableMultiInput(
                observation_space=self.observation_space,
                num_outputs=self.latent_dim,
                device=self.device,
                name=self.encoder_name,
                **net_config,
            )
        elif is_image_space(self.observation_space):
            assert_correct_cnn_net_config(net_config)

            obs_shape = self.observation_space.shape
            assert obs_shape is not None, "Image observation spaces must have a shape."

            encoder = EvolvableCNN(
                input_shape=list(obs_shape),
                num_outputs=self.latent_dim,
                device=self.device,
                name=self.encoder_name,
                **net_config,
            )
        elif self.recurrent:
            assert_correct_lstm_net_config(net_config)

            encoder = EvolvableLSTM(
                input_size=spaces.flatdim(self.observation_space),
                num_outputs=self.latent_dim,
                device=self.device,
                name=self.encoder_name,
                **net_config,
            )
        else:
            if self.simba:
                assert_correct_simba_net_config(net_config)
                encoder_mlp_cls = EvolvableSimBa
            else:
                assert_correct_mlp_net_config(net_config)
                encoder_mlp_cls = EvolvableMLP

                # For MLP encoders we want to:
                # 1. Be consistent with layer normalization and apply after the final
                # linear layer for further stability (see https://github.com/AgileRL/AgileRL/issues/337)
                # 2. Disable output_vanish
                net_config["output_layernorm"] = net_config.get("layer_norm", True)
                net_config["output_vanish"] = False

            encoder = encoder_mlp_cls(
                num_inputs=spaces.flatdim(self.observation_space),
                num_outputs=self.latent_dim,
                device=self.device,
                name=self.encoder_name,
                **net_config,
            )

            # Need to flatten > 2D observations by default for MLPs
            obs_shape = self.observation_space.shape
            self.flatten_obs = obs_shape is not None and len(obs_shape) > 1

        return encoder
