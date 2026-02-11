import warnings
from typing import ClassVar, Optional

import torch
from gymnasium import spaces

from agilerl.modules.base import EvolvableModule
from agilerl.modules.configs import MlpNetConfig
from agilerl.networks.base import EvolvableNetwork
from agilerl.networks.distributions import EvolvableDistribution
from agilerl.typing import ArrayOrTensor, NetConfigType, TorchObsType
from agilerl.utils.algo_utils import get_output_size_from_space


def get_output_bounds(output_activation: str) -> tuple[float, float]:
    """Get the bounds of the output activation function.

    :param output_activation: Output activation function.
    :type output_activation: str
    :return: Bounds of the output activation function.
    :rtype: tuple[float, float]
    """
    if output_activation in ["Tanh", "Softsign"]:
        return -1.0, 1.0
    elif output_activation in ["Sigmoid", "Softmax", "GumbelSoftmax"]:
        return 0.0, 1.0
    else:
        raise ValueError(
            f"Received invalid output activation function: {output_activation}. "
        )


class DeterministicActor(EvolvableNetwork):
    """Deterministic actor network for policy-gradient algorithms. Given an observation,
    it outputs the mean of the action distribution. This is useful for e.g. DDPG, SAC, TD3.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment.
    :type action_space: spaces.Box | spaces.Discrete
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: str | type[EvolvableModule] | None
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: NetConfigType
    :param head_config: Configuration of the network MLP head.
    :type head_config: NetConfigType | None
    :param min_latent_dim: Minimum dimension of the latent space representation.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation.
    :type max_latent_dim: int
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param simba: Whether to use the SimBa architecture for training the network.
    :type simba: bool
    :param recurrent: Whether to use a recurrent network.
    :type recurrent: bool
    :param device: Device to use for the network.
    :type device: str
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: int | None
    :param encoder_name: Name of the encoder network.
    :type encoder_name: str
    """

    supported_spaces: ClassVar[tuple[type[spaces.Space], ...]] = (
        spaces.Box,
        spaces.Discrete,
    )
    _allowed_output_activations: ClassVar[tuple[str, ...]] = (
        "Tanh",
        "Softsign",
        "Sigmoid",
        "Softmax",
        "GumbelSoftmax",
    )

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box | spaces.Discrete,
        encoder_cls: str | type[EvolvableModule] | None = None,
        encoder_config: NetConfigType | None = None,
        head_config: NetConfigType | None = None,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        latent_dim: int = 32,
        simba: bool = False,
        recurrent: bool = False,
        device: str = "cpu",
        random_seed: int | None = None,
        encoder_name: str = "encoder",
    ) -> None:
        super().__init__(
            observation_space,
            encoder_cls=encoder_cls,
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            latent_dim=latent_dim,
            simba=simba,
            recurrent=recurrent,
            device=device,
            random_seed=random_seed,
            encoder_name=encoder_name,
        )

        if isinstance(action_space, spaces.Box):
            self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
            self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        else:
            self.action_low, self.action_high = None, None

        # Set output activation based on action space
        if isinstance(action_space, spaces.Box):
            output_activation = "Tanh"
        elif isinstance(action_space, spaces.Discrete):
            output_activation = "GumbelSoftmax"

        if head_config is not None:
            if "output_activation" in head_config:
                user_output_activation = head_config["output_activation"]
                if user_output_activation not in self._allowed_output_activations:
                    warnings.warn(
                        f"Output activation must be one of the following: {', '.join(self._allowed_output_activations)}. "
                        f"Got {user_output_activation} instead. Using default output activation."
                    )
                else:
                    output_activation = user_output_activation

        self.output_activation = output_activation

        if head_config is None:
            head_config = MlpNetConfig(
                hidden_size=[32], output_activation=output_activation
            )
        else:
            head_config["output_activation"] = output_activation

        self.output_size = get_output_size_from_space(self.action_space)

        self.build_network_head(head_config)  # Build network head

    @staticmethod
    def rescale_action(
        action: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        output_activation: str,
    ) -> torch.Tensor:
        """Rescale an action from the network output bounds to the action space bounds [low, high].

        :param action: Action as outputted by the network.
        :type action: torch.Tensor
        :param low: Minimum action array.
        :type low: torch.Tensor
        :param high: Maximum action array.
        :type high: torch.Tensor
        :param output_activation: Output activation function of the network.
        :type output_activation: str
        :return: Action in space bounds [low, high].
        :rtype: torch.Tensor
        """
        min_output, max_output = get_output_bounds(output_activation)

        # If the action space or network output are unbounded, we just return the action
        if (
            min_output is None
            or max_output is None
            or low.isinf().any()
            or high.isinf().any()
        ):
            return action

        # [prescaled_min, prescaled_max] -> [low, high]
        rescaled_action = low + (high - low) * (
            (action - min_output) / (max_output - min_output)
        )
        return rescaled_action.to(low.dtype)

    def build_network_head(self, net_config: Optional[NetConfigType] = None) -> None:
        """Builds the head of the network.

        :param net_config: Configuration of the head.
        :type net_config: Optional[ConfigType]
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.output_size,
            name="actor",
            net_config=net_config,
        )

    def forward(self, obs: TorchObsType) -> torch.Tensor:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :return: Output of the network.
        :rtype: torch.Tensor
        """
        latent = self.extract_features(obs)
        return self.head_net(latent)

    def recreate_network(self) -> None:
        """Recreates the network."""
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.output_size,
            name="actor",
            net_config=self.head_net.net_config,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)


class StochasticActor(EvolvableNetwork):
    """Stochastic actor network for policy-gradient algorithms. Given an observation, constructs
    a distribution over the action space from the logits output by the network. Contains methods
    to sample actions and compute log probabilities and the entropy of the action distribution,
    relevant for many policy-gradient algorithms such as PPO, A2C, TRPO.

    :param observation_space: Observation space of the environment.
    :type observation_space: spaces.Space
    :param action_space: Action space of the environment
    :type action_space: spaces.Space
    :param encoder_cls: Encoder class to use for the network. Defaults to None, whereby it is
        automatically built using an AgileRL module according the observation space.
    :type encoder_cls: str | type[EvolvableModule] | None
    :param encoder_config: Configuration of the encoder network.
    :type encoder_config: NetConfigType | None
    :param head_config: Configuration of the network MLP head.
    :type head_config: NetConfigType | None
    :param action_std_init: Initial log standard deviation of the action distribution. Defaults to 0.0.
    :type action_std_init: float
    :param squash_output: Whether to squash the output to the action space.
    :type squash_output: bool
    :param min_latent_dim: Minimum dimension of the latent space representation.
    :type min_latent_dim: int
    :param max_latent_dim: Maximum dimension of the latent space representation.
    :type max_latent_dim: int
    :param latent_dim: Dimension of the latent space representation.
    :type latent_dim: int
    :param simba: Whether to use the SimBa architecture for training the network.
    :type simba: bool
    :param recurrent: Whether to use a recurrent network.
    :type recurrent: bool
    :param device: Device to use for the network.
    :type device: str
    :param random_seed: Random seed to use for the network. Defaults to None.
    :type random_seed: int | None
    :param encoder_name: Name of the encoder network.
    :type encoder_name: str
    """

    head_net: EvolvableDistribution
    supported_spaces: ClassVar[tuple[type[spaces.Space], ...]] = (
        spaces.Box,
        spaces.Discrete,
        spaces.MultiDiscrete,
        spaces.MultiBinary,
    )

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        encoder_cls: str | type[EvolvableModule] | None = None,
        encoder_config: NetConfigType | None = None,
        head_config: NetConfigType | None = None,
        action_std_init: float = 0.0,
        squash_output: bool = False,
        min_latent_dim: int = 8,
        max_latent_dim: int = 128,
        latent_dim: int = 32,
        simba: bool = False,
        recurrent: bool = False,
        device: str = "cpu",
        random_seed: int | None = None,
        encoder_name: str = "encoder",
    ) -> None:
        super().__init__(
            observation_space,
            encoder_cls=encoder_cls,
            encoder_config=encoder_config,
            action_space=action_space,
            min_latent_dim=min_latent_dim,
            max_latent_dim=max_latent_dim,
            latent_dim=latent_dim,
            simba=simba,
            recurrent=recurrent,
            device=device,
            random_seed=random_seed,
            encoder_name=encoder_name,
        )

        # Require the head to output logits to parameterize a distribution (i.e. no output activation)
        if head_config is None:
            head_config = MlpNetConfig(hidden_size=[32], output_activation=None)
        else:
            head_config["output_activation"] = None

        self.action_std_init = action_std_init
        self.squash_output = squash_output
        self.action_space = action_space
        self.output_size = get_output_size_from_space(self.action_space)

        if isinstance(action_space, spaces.Box):
            self.action_low = torch.as_tensor(
                action_space.low, device=self.device, dtype=torch.float32
            )
            self.action_high = torch.as_tensor(
                action_space.high, device=self.device, dtype=torch.float32
            )
        else:
            self.action_low, self.action_high = None, None

        self.build_network_head(head_config)
        self.output_activation = None

        self.head_net = EvolvableDistribution(
            action_space=action_space,
            network=self.head_net,
            action_std_init=action_std_init,
            squash_output=squash_output,
            device=device,
        )

    def build_network_head(self, net_config: NetConfigType | None = None) -> None:
        """Builds the head of the network.

        :param net_config: Configuration of the head.
        :type net_config: NetConfigType | None
        """
        self.head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.output_size,
            name="actor",
            net_config=net_config,
        )

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Rescale the action from [-1, 1] to the action space bounds [low, high].

        :param action: Action.
        :type action: torch.Tensor
        :return: Rescaled action.
        :rtype: torch.Tensor
        """
        return self.action_low + (
            0.5 * (action + 1.0) * (self.action_high - self.action_low)
        )

    def forward(
        self, obs: TorchObsType, action_mask: ArrayOrTensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the network.

        :param obs: Observation input.
        :type obs: TorchObsType
        :param action_mask: Action mask.
        :type action_mask: ArrayOrTensor | None
        :return: Action and log probability of the action.
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        latent = self.extract_features(obs)
        action, log_prob, entropy = self.head_net.forward(latent, action_mask)

        # Action scaling only relevant for continuous action spaces with squashing
        if isinstance(self.action_space, spaces.Box) and self.squash_output:
            action = self.rescale_action(action)

        return action, log_prob, entropy

    def action_log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Get the log probability of the action.

        :param action: Action.
        :type action: torch.Tensor
        :return: Log probability of the action.
        :rtype: torch.Tensor
        """
        return self.head_net.log_prob(action)

    def action_entropy(self) -> torch.Tensor:
        """Get the entropy of the action distribution.

        :return: Entropy of the action distribution.
        :rtype: torch.Tensor
        """
        return self.head_net.entropy()

    def recreate_network(self) -> None:
        """Recreates the network with the same parameters as the current network."""
        self.recreate_encoder()

        head_net = self.create_mlp(
            num_inputs=self.latent_dim,
            num_outputs=self.output_size,
            name="actor",
            net_config=self.head_net.net_config,
        )

        head_net = EvolvableDistribution(
            self.action_space,
            head_net,
            action_std_init=self.action_std_init,
            squash_output=self.squash_output,
            device=self.device,
        )

        self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)
