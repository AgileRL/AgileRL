import inspect
import warnings
from dataclasses import asdict
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
from gymnasium import spaces

from agilerl.modules import (
    EvolvableCNN,
    EvolvableMLP,
    EvolvableMultiInput,
    EvolvableResNet,
    EvolvableSimBa,
)
from agilerl.modules.base import EvolvableModule, ModuleMeta, mutation
from agilerl.protocols import MutationType
from agilerl.typing import ConfigType, DeviceType, TorchObsType
from agilerl.utils.evolvable_networks import get_default_encoder_config, is_image_space

SelfEvolvableNetwork = TypeVar("SelfEvolvableNetwork", bound="EvolvableNetwork")
DefaultEncoderType = Union[
    EvolvableCNN, EvolvableMLP, EvolvableMultiInput, EvolvableSimBa
]


def assert_correct_mlp_net_config(net_config: Dict[str, Any]) -> None:
    """Asserts that the MLP network configuration is correct.

    :param net_config: Configuration of the MLP network.
    :type net_config: Dict[str, Any]
    """
    assert (
        "hidden_size" in net_config.keys()
    ), "Net config must contain hidden_size: int."
    assert isinstance(
        net_config["hidden_size"], list
    ), "Net config hidden_size must be a list."
    assert (
        len(net_config["hidden_size"]) > 0
    ), "Net config hidden_size must contain at least one element."


def assert_correct_simba_net_config(net_config: Dict[str, Any]) -> None:
    """Asserts that the MLP network configuration is correct.

    :param net_config: Configuration of the MLP network.
    :type net_config: Dict[str, Any]
    """
    assert (
        "hidden_size" in net_config.keys()
    ), "Net config must contain hidden_size: int."
    assert isinstance(
        net_config["hidden_size"], (int, np.int64)
    ), "Net config hidden_size must be an integer."
    assert "num_blocks" in net_config.keys(), "Net config must contain num_blocks: int."
    assert isinstance(
        net_config["num_blocks"], int
    ), "Net config num_blocks must be an integer."


def assert_correct_cnn_net_config(net_config: Dict[str, Any]) -> None:
    """Asserts that the CNN network configuration is correct.

    :param net_config: Configuration of the CNN network.
    :type net_config: Dict[str, Any]
    """
    for key in [
        "channel_size",
        "kernel_size",
        "stride_size",
    ]:
        assert key in net_config.keys(), f"Net config must contain {key}: int."
        assert isinstance(net_config[key], list), f"Net config {key} must be a list."
        assert (
            len(net_config[key]) > 0
        ), f"Net config {key} must contain at least one element."

        if key == "kernel_size":
            assert isinstance(
                net_config[key], (int, tuple, list)
            ), "Kernel size must be of type int, list, or tuple."


def assert_correct_multi_input_net_config(net_config: Dict[str, Any]) -> None:
    """Asserts that the MultiInput network configuration is correct.

    :param net_config: Configuration of the MultiInput network.
    :type net_config: Dict[str, Any]
    """
    # Multi-input networks contain at least one image space
    assert_correct_cnn_net_config(net_config)


# TODO: Need to think of a way to do this check without the metaclass
class NetworkMeta(ModuleMeta):
    """Metaclass for evolvable networks. Checks that the network has
    an encoder and a head_net (named as such)."""

    def __call__(cls, *args, **kwargs):
        instance: SelfEvolvableNetwork = super().__call__(*args, **kwargs)

        # Check that the mutation methods of the network are correctly defined
        # i.e. only contain underlying methods corresponding to the encoder and head_net
        for mut_method in instance.mutation_methods:
            if "." in mut_method:
                attr = mut_method.split(".")[0]
                if attr not in ["encoder", "head_net"]:
                    raise AttributeError(
                        "Mutation methods of underlying modules in EvolvableNetwork objects should only correspond "
                        "to the encoder or head_net. This is done to ensure that analogous architecture mutations "
                        "can be applied between different networks. "
                    )

        return instance


class EvolvableNetwork(EvolvableModule, metaclass=NetworkMeta):
    """
    Base class for evolvable networks, i.e., evolvable modules that are configured in
    a specific way for a reinforcement learning algorithm, similar to how CNNs are used
    as building blocks in ResNet, VGG, etc. An evolvable network automatically inspects the passed
    observation space to determine the appropriate encoder to build through the AgileRL
    evolvable modules, inheriting the mutation methods of any underlying evolvable module.

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
    :type encoder_cls: Optional[Union[str, Type[EvolvableModule]]]
    :param encoder_config: Configuration of the encoder. Defaults to None.
    :type encoder_config: Optional[ConfigType]
    :param action_space: Action space of the environment. Defaults to None.
    :type action_space: Optional[spaces.Space]
    :param min_latent_dim: Minimum dimension of the latent space representation. Defaults to 8.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation. Defaults to 128.
    :type max_latent_dim: int
    :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to
        single-agent environments.
    :type n_agents: Optional[int]
    :param encoder_mutations: If True, allow mutations to the encoder. Defaults to False.
    :type encoder_mutations: bool
    :param latent_dim: Dimension of the latent space representation. Defaults to 32.
    :type latent_dim: int
    :param simba: If True, use a SimBa network for the encoder for vector spaces. Defaults to False.
    :type simba: bool
    :param device: Device to use for the network. Defaults to "cpu".
    :type device: DeviceType
    """

    encoder: EvolvableModule
    head_net: EvolvableModule

    # Custom encoder aliases
    _encoder_aliases: Dict[str, Type[EvolvableModule]] = {
        "ResNet": EvolvableResNet,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        encoder_cls: Optional[Union[str, Type[EvolvableModule]]] = None,
        encoder_config: Optional[ConfigType] = None,
        action_space: Optional[spaces.Space] = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        n_agents: Optional[int] = None,
        latent_dim: int = 32,
        simba: bool = False,
        device: DeviceType = "cpu",
    ) -> None:
        super().__init__(device)

        if encoder_config is None:
            encoder_config = get_default_encoder_config(observation_space, simba=simba)

        # NOTE: For multi-agent settings, we use a depth corresponding to that of the
        # sample input for the kernel of the first layer of CNN-based networks
        if n_agents is not None and "kernel_size" in encoder_config.keys():
            encoder_config = EvolvableNetwork.modify_multi_agent_config(
                net_config=encoder_config, observation_space=observation_space
            )

        self.observation_space = observation_space
        self.action_space = action_space
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.min_latent_dim = min_latent_dim
        self.max_latent_dim = max_latent_dim
        self.encoder_cls = encoder_cls
        self.device = device
        self.simba = simba

        encoder_config = (
            encoder_config
            if isinstance(encoder_config, dict)
            else asdict(encoder_config)
        )

        # Encoder processes an observation into a latent vector representation
        output_activation = encoder_config.get("output_activation", None)
        if output_activation is None:
            activation = encoder_config.get("activation", "ReLU")
            encoder_config["output_activation"] = activation

        if encoder_cls is not None:
            if isinstance(encoder_cls, str):
                self.encoder_cls = self._encoder_aliases[encoder_cls]
            elif not issubclass(encoder_cls, EvolvableModule):
                raise TypeError("Encoder class must be a subclass of EvolvableModule.")

            # Check if encoder config contains `num_outputs` as input argument, in which
            # case we can enable latent space mutations. Otherwise, we disable them.
            input_args = inspect.getfullargspec(self.encoder_cls.__init__).args
            if "num_outputs" not in input_args:
                warnings.warn(
                    "Custom encoder does not contain `num_outputs` as an input argument. "
                    "Disabling latent space mutations. Make sure to set the number of "
                    "outputs to the latent dimension in the encoder configuration."
                )
                self.filter_mutation_methods("latent")
            else:
                encoder_config["num_outputs"] = self.latent_dim

            self.encoder = self.encoder_cls(**encoder_config)
        else:
            self.encoder = self._build_encoder(encoder_config)

        # NOTE: We disable layer mutations for the encoder since this usually adds a lot
        # of variance to the optimization process
        self.encoder.disable_mutations(MutationType.LAYER)

    @property
    def encoder_config(self) -> Dict[str, Any]:
        """Net configuration for encoder.

        :return: Initial dictionary for the network.
        :rtype: Dict[str, Any]
        """
        return (
            self.encoder.net_config
            if self.encoder_cls is None
            else self.encoder.init_dict
        )

    @property
    def head_config(self) -> Dict[str, Any]:
        """Net configuration for head.

        :return: Initial dictionary for the network.
        :rtype: Dict[str, Any]
        """
        if not hasattr(self, "head_net"):
            raise AttributeError("Network does not have a head_net attribute.")

        return self.head_net.net_config

    @property
    def activation(self) -> str:
        """Activation function of the network.

        :return: Activation function.
        :rtype: str
        """
        return self.encoder.activation

    def forward(self, x: TorchObsType) -> torch.Tensor:
        """Forward pass of the network.

        :param x: Input to the network.
        :type x: TorchObsType

        :return: Output of the network.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def build_network_head(self, *args, **kwargs) -> None:
        """Build the head of the network."""
        raise NotImplementedError(
            "Method build_network_head must be implemented in EvolvableNetwork objects."
        )

    @staticmethod
    def modify_multi_agent_config(
        net_config: Dict[str, Any],
        observation_space: spaces.Space,
    ) -> Dict[str, Any]:
        """In multi-agent settings, it is not clear what the shape of the input to the
        encoder is based on the passed observation space. If kernel sizes are passed as
        integers, we add a depth dimension of 1 for all layers. Note that for e.g. value
        functions the first layer should have a depth corresponding to the number of agents
        to receive a single output rather than `self.n_agents`
        """
        if isinstance(observation_space, (spaces.Dict, spaces.Tuple)):
            net_config["cnn_block_type"] = "Conv3d"
        else:
            net_config["block_type"] = "Conv3d"

        return net_config

    def create_mlp(
        self, num_inputs: int, num_outputs: int, name: str, net_config: Dict[str, Any]
    ) -> EvolvableMLP:
        """Builds the head of the network based on the passed configuration.

        :param num_inputs: Number of inputs to the network head.
        :type num_inputs: int
        :param num_outputs: Number of outputs of the network head.
        :type num_outputs: int
        :param name: Name of the network head.
        :type name: str
        :param net_config: Configuration of the network head.
        :type net_config: Dict[str, Any]

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

    def modules(self) -> Dict[str, EvolvableModule]:
        """Modules of the network.

        :return: Modules of the network.
        :rtype: Dict[str, EvolvableModule]
        """
        return super().modules()

    def init_weights_gaussian(
        self, std_coeff: float = 4.0, output_coeff: float = 2.0
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
        for attr, module in self.modules().items():
            if attr != "encoder":
                module.init_weights_gaussian(
                    std_coeff=std_coeff, output_coeff=output_coeff
                )

    def change_activation(self, activation: str, output: bool = False) -> None:
        """Change the activation function for the network.

        :param activation: Activation function to use.
        :type activation: str
        :param output: If True, change the output activation function, defaults to False
        :type output: bool, optional
        """
        for attr, module in self.modules().items():
            _output = False if attr == "encoder" else output
            module.change_activation(activation, output=_output)

    @mutation(MutationType.NODE)
    def add_latent_node(self, numb_new_nodes: Optional[int] = None) -> Dict[str, Any]:
        """Add a latent node to the network.

        :param numb_new_nodes: Number of new nodes to add, defaults to None
        :type numb_new_nodes: int, optional

        :return: Configuration for adding a latent node.
        :rtype: Dict[str, Any]
        """
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([8, 16, 32], 1)[0]

        if self.latent_dim + numb_new_nodes < self.max_latent_dim:
            self.latent_dim += numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    @mutation(MutationType.NODE)
    def remove_latent_node(
        self, numb_new_nodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Remove a latent node from the network.

        :param numb_new_nodes: Number of nodes to remove, defaults to None
        :type numb_new_nodes: int, optional

        :return: Configuration for removing a latent node.
        :rtype: Dict[str, Any]
        """
        if numb_new_nodes is None:
            numb_new_nodes = np.random.choice([8, 16, 32], 1)[0]

        if self.latent_dim - numb_new_nodes > self.min_latent_dim:
            self.latent_dim -= numb_new_nodes

        return {"numb_new_nodes": numb_new_nodes}

    def recreate_encoder(self: SelfEvolvableNetwork) -> None:
        """Recreate the encoder of the network."""
        if self.encoder_cls is not None:
            # Need to change `num_outputs` to the latent dimension after a mutation
            init_dict = self.encoder.init_dict
            init_dict["num_outputs"] = self.latent_dim
            encoder = self.encoder_cls(**init_dict)
        else:
            encoder = self._build_encoder(self.encoder.net_config)

        self.encoder = EvolvableModule.preserve_parameters(self.encoder, encoder)

    def _build_encoder(self, net_config: Dict[str, Any]) -> DefaultEncoderType:
        """Builds the encoder for the network based on the environments observation space.

        :return: Encoder module.
        :rtype: EvolvableModule
        """
        if isinstance(self.observation_space, (spaces.Dict, spaces.Tuple)):
            assert_correct_multi_input_net_config(net_config)

            encoder = EvolvableMultiInput(
                observation_space=self.observation_space,
                num_outputs=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config,
            )
        elif is_image_space(self.observation_space):
            assert_correct_cnn_net_config(net_config)

            encoder = EvolvableCNN(
                input_shape=self.observation_space.shape,
                num_outputs=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config,
            )
        else:
            if self.simba:
                assert_correct_simba_net_config(net_config)
                encoder_mlp_cls = EvolvableSimBa
            else:
                assert_correct_mlp_net_config(net_config)
                encoder_mlp_cls = EvolvableMLP

            encoder = encoder_mlp_cls(
                num_inputs=spaces.flatdim(self.observation_space),
                num_outputs=self.latent_dim,
                device=self.device,
                name="encoder",
                **net_config,
            )

        return encoder
