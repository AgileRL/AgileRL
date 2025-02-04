.. _rainbow_dqn_tutorial:

Building a Dueling Distributional Q Network Using AgileRL
=========================================================

`Rainbow DQN <https://arxiv.org/abs/1710.02298>`_ is an extension of DQN that integrates multiple improvements and techniques to achieve state-of-the-art performance. The improvements pertaining to the Q network that is optimized during training are the following:

* **Dueling Networks**: Splits the Q-network into two separate streams — one for estimating the state value function and another for estimating the advantages for each action. They are then combined to produce Q-values.
* **Categorical DQN (C51)**: A specific form of distributional RL where the continuous range of possible cumulative future rewards is discretized into a fixed set of categories.

In order to extend our implementation of :class:`QNetwork <agilerl.networks.q_networks.QNetwork>` to a Dueling Distributional Q Network, we need to make the following changes:

1. Modify the Q network to output a distribution over Q-values instead of a single Q-value.
2. Implement the Dueling Network architecture to separate the Q network into two streams - a value network and an advantage network.

Since these changes only really pertain to the heads of our network, we can take a variety of approaches to implement them appropriately. However, users
should remember that only one head in a class inheriting from :class:`QNetwork <agilerl.networks.q_networks.QNetwork>` should contribute to the global mutation methods of
the network (this can be checked through the ``mutation_methods`` attribute of any :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` object). This is done to reduce the additional variance
incurred by our evolutionary hyperparameter optimization process. So, we want to add an evolvable head for our advantage network but disable mutations for it
directly, and rather mutate its architecture synchronously with the value network (i.e. whenever the value head is mutated, the advantage head is mutated in the
same way).

Approaches:

1. **Adding Head Directly in EvolvableNetwork**: We can copy our simple :class:`QNetwork <agilerl.networks.q_networks.QNetwork>` and add an additional head for our advantage
network as an :class:`EvolvableMLP <agilerl.modules.mlp.EvolvableMLP>`. When we do this, its mutation methods will be added automatically so we need to disable them manually through the
:meth:`EvolvableMLP.disable_mutations() <agilerl.modules.base.EvolvableModule.disable_mutations>` method.

2. **Creating a Custom RainbowMLP**: We can create a custom MLP that inherits from :class:`EvolvableMLP <agilerl.modules.mlp.EvolvableMLP>` and add the advantage head without having to disable the mutations on it.

For either of the above solutions, we need to be able to recreate the network after an architecture mutations such that the same mutation is applied to both the
value and advantage heads. We can do this by overwriting the :meth:`recreate_network() <agilerl.modules.base.EvolvableModule.recreate_network>` method in our custom MLP.
For more information, please refer to the `EvolvableMLP <https://github.com/AgileRL/AgileRL/blob/complex-spaces/agilerl/modules/mlp.py#L9>`_ implementation for an example of
the complete requirements of a :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` object.

RainbowMLP
----------

Below we show our implementation of our custom head with a distributional dueling architecture, using noisy linear layers as described in the paper. In
:meth:`recreate_network() <agilerl.modules.base.EvolvableModule.recreate_network>`, we use the same parameters as the value network to create the advantage network after mutating it.

.. code-block:: python

    from typing import List, Dict, Any
    import torch
    import torch.nn.functional as F

    from agilerl.modules.base import EvolvableModule
    from agilerl.modules.mlp import EvolvableMLP
    from agilerl.utils.evolvable_networks import create_mlp

    class DuelingMLP(EvolvableMLP):
        """A multi-layer perceptron network that calculates state-action values through
        the use of separate advantage and value networks. It outputs a distribution of values
        for both of these networks. Used in the Rainbow DQN algorithm.

        :param num_inputs: Number of input features.
        :type num_inputs: int
        :param num_outputs: Number of output features.
        :type num_outputs: int
        :param hidden_size: List of hidden layer sizes.
        :type hidden_size: List[int]
        :param num_atoms: Number of atoms in the distribution.
        :type num_atoms: int
        :param support: Support of the distribution.
        :type support: torch.Tensor
        :param noise_std: Standard deviation of the noise. Defaults to 0.5.
        :type noise_std: float, optional
        :param activation: Activation layer, defaults to 'ReLU'
        :type activation: str, optional
        :param output_activation: Output activation layer, defaults to None
        :type output_activation: str, optional
        :param min_hidden_layers: Minimum number of hidden layers the network will shrink down to, defaults to 1
        :type min_hidden_layers: int, optional
        :param max_hidden_layers: Maximum number of hidden layers the network will expand to, defaults to 3
        :type max_hidden_layers: int, optional
        :param min_mlp_nodes: Minimum number of nodes a layer can have within the network, defaults to 64
        :type min_mlp_nodes: int, optional
        :param max_mlp_nodes: Maximum number of nodes a layer can have within the network, defaults to 500
        :type max_mlp_nodes: int, optional
        :param layer_norm: Normalization between layers, defaults to True
        :type layer_norm: bool, optional
        :param output_vanish: Vanish output by multiplying by 0.1, defaults to True
        :type output_vanish: bool, optional
        :param init_layers: Initialise network layers, defaults to True
        :type init_layers: bool, optional
        :param new_gelu: Use new GELU activation function, defaults to False
        :type new_gelu: bool, optional
        :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
        :type device: str, optional
        """

        def __init__(
                self,
                num_inputs: int,
                num_outputs: int,
                hidden_size: List[int],
                num_atoms: int,
                support: torch.Tensor,
                noise_std: float = 0.5,
                **kwargs
                ) -> None:

            super().__init__(
                num_inputs,
                num_atoms,
                hidden_size,
                noisy=True,
                init_layers=False,
                layer_norm=True,
                output_vanish=True,
                noise_std=noise_std,
                name="value",
                **kwargs
                )

            self.num_atoms = num_atoms
            self.num_actions = num_outputs
            self.support = support

            self.advantage_net = create_mlp(
                input_size=num_inputs,
                output_size=num_outputs * num_atoms,
                hidden_size=self.hidden_size,
                output_vanish=self.output_vanish,
                output_activation=self.output_activation,
                noisy=self.noisy,
                init_layers=self.init_layers,
                layer_norm=self.layer_norm,
                activation=self.activation,
                noise_std=self.noise_std,
                device=self.device,
                new_gelu=self.new_gelu,
                name="advantage"
            )

        @property
        def net_config(self) -> Dict[str, Any]:
            net_config = super().net_config.copy()
            net_config.pop("num_atoms")
            net_config.pop("support")
            return net_config

        @property
        def init_dict(self) -> Dict[str, Any]:
            mlp_dict = super().init_dict
            mlp_dict["num_atoms"] = self.num_atoms
            mlp_dict['num_outputs'] = self.num_actions
            mlp_dict["support"] = self.support
            mlp_dict.pop("noisy")
            mlp_dict.pop("init_layers")
            mlp_dict.pop("layer_norm")
            mlp_dict.pop("output_vanish")
            mlp_dict.pop("name")
            return mlp_dict

        def forward(self, x: torch.Tensor, q: bool = True, log: bool = False) -> torch.Tensor:
            """Forward pass of the RainbowMLP.

            :param obs: Input to the network.
            :type obs: torch.Tensor, dict[str, torch.Tensor], or list[torch.Tensor]
            :param q: Whether to return Q values. Defaults to True.
            :type q: bool
            :param log: Whether to return log probabilities. Defaults to False.
            :type log: bool

            :return: Output of the network.
            :rtype: torch.Tensor
            """
            value: torch.Tensor = self.model(x)
            advantage: torch.Tensor = self.advantage_net(x)

            batch_size = value.size(0)
            value = value.view(batch_size, 1, self.num_atoms)
            advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

            x = value + advantage - advantage.mean(1, keepdim=True)
            if log:
                x = F.log_softmax(x.view(-1, self.num_atoms), dim=-1)
                return x.view(-1, self.num_actions, self.num_atoms)

            x = F.softmax(x.view(-1, self.num_atoms), dim=-1)
            x = x.view(-1, self.num_actions, self.num_atoms).clamp(min=1e-3)
            if q:
                x = torch.sum(x * self.support, dim=2)

            return x

        def recreate_network(self) -> None:
            """Recreates the network with the same parameters."""
            # Recreate value net with the same parameters
            super().recreate_network()

            advantage_net = create_mlp(
                input_size=self.num_inputs,
                output_size=self.num_actions * self.num_atoms,
                hidden_size=self.hidden_size,
                output_activation=self.output_activation,
                output_vanish=self.output_vanish,
                noisy=self.noisy,
                init_layers=self.init_layers,
                layer_norm=self.layer_norm,
                activation=self.activation,
                noise_std=self.noise_std,
                device=self.device,
                new_gelu=self.new_gelu,
                name="advantage"
            )

            self.advantage_net = EvolvableModule.preserve_parameters(
                self.advantage_net, advantage_net
                )

Creating a Custom :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>`
-----------------------------------------------------------------------------------

Now that we have our custom head, we can create a custom network that inherits from :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>` and uses our custom head. Since we have done most of the work in the head,
the implementation is quite simple and analogous to the :class:`QNetwork <agilerl.networks.q_networks.QNetwork>` implementation. We only need to change the head to our custom head and update the
:meth:`recreate_network() <agilerl.networks.base.EvolvableNetwork.recreate_network>` method.

.. code-block:: python

    from typing import Optional, Dict, Any
    from dataclasses import asdict

    import torch
    from gym import spaces

    from agilerl.networks.base import EvolvableNetwork
    from agilerl.modules.configs import MlpNetConfig

    class RainbowQNetwork(EvolvableNetwork):
        """RainbowQNetwork is an extension of the QNetwork that incorporates the Rainbow DQN improvements
        from "Rainbow: Combining Improvements in Deep Reinforcement Learning" (Hessel et al., 2017).

        Paper: https://arxiv.org/abs/1710.02298

        :param observation_space: Observation space of the environment.
        :type observation_space: spaces.Space
        :param action_space: Action space of the environment
        :type action_space: DiscreteSpace
        :param encoder_config: Configuration of the encoder network.
        :type encoder_config: ConfigType
        :param support: Support for the distributional value function.
        :type support: torch.Tensor
        :param num_atoms: Number of atoms in the distributional value function. Defaults to 51.
        :type num_atoms: int
        :param head_config: Configuration of the network MLP head.
        :type head_config: Optional[ConfigType]
        :param min_latent_dim: Minimum dimension of the latent space representation. Defaults to 8.
        :type min_latent_dim: int
        :param max_latent_dim: Maximum dimension of the latent space representation. Defaults to 128.
        :type max_latent_dim: int
        :param n_agents: Number of agents in the environment. Defaults to None, which corresponds to
            single-agent environments.
        :type n_agents: Optional[int]
        :param latent_dim: Dimension of the latent space representation.
        :type latent_dim: int
        :param device: Device to use for the network.
        :type device: str
        """

        def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Discrete,
            support: torch.Tensor,
            num_atoms: int = 51,
            noise_std: float = 0.5,
            encoder_config: Optional[ConfigType] = None,
            head_config: Optional[ConfigType] = None,
            min_latent_dim: int = 8,
            max_latent_dim: int = 128,
            n_agents: Optional[int] = None,
            latent_dim: int = 32,
            device: str = "cpu",
        ):

            if isinstance(observation_space, spaces.Box) and not is_image_space(
                observation_space
            ):
                if encoder_config is None:
                    encoder_config = asdict(MlpNetConfig(hidden_size=[16]))

                encoder_config["noise_std"] = noise_std
                encoder_config["output_activation"] = encoder_config.get(
                    "activation", "ReLU"
                )
                encoder_config["output_vanish"] = False
                encoder_config["init_layers"] = False
                encoder_config["layer_norm"] = True

            super().__init__(
                observation_space,
                encoder_config=encoder_config,
                action_space=action_space,
                min_latent_dim=min_latent_dim,
                max_latent_dim=max_latent_dim,
                n_agents=n_agents,
                latent_dim=latent_dim,
                device=device,
            )

            if not isinstance(action_space, (spaces.Discrete, spaces.MultiDiscrete)):
                raise ValueError("Action space must be either Discrete or MultiDiscrete")

            if head_config is None:
                head_config = asdict(
                    MlpNetConfig(
                        hidden_size=[16], output_activation=None, noise_std=noise_std
                    )
                )
            elif isinstance(head_config, NetConfig):
                head_config = asdict(head_config)
                head_config["noise_std"] = noise_std

            # The heads should have no output activation
            head_config["output_activation"] = None

            for arg in ["noisy", "init_layers", "layer_norm", "output_vanish"]:
                if head_config.get(arg, None) is not None:
                    head_config.pop(arg)

            self.num_actions = spaces.flatdim(action_space)
            self.num_atoms = num_atoms
            self.support = support
            self.noise_std = noise_std

            # Build value and advantage networks
            self.build_network_head(head_config)

        @property
        def init_dict(self) -> Dict[str, Any]:
            """Initializes the configuration of the Rainbow Q network.

            :return: Configuration of the Rainbow Q network.
            :rtype: Dict[str, Any]
            """
            return {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "support": self.support,
                "num_atoms": self.num_atoms,
                "encoder_config": self.encoder.net_config,
                "head_config": self.head_net.net_config,
                "min_latent_dim": self.min_latent_dim,
                "max_latent_dim": self.max_latent_dim,
                "n_agents": self.n_agents,
                "latent_dim": self.latent_dim,
                "device": self.device,
            }

        def build_network_head(self, net_config: Dict[str, Any]) -> None:
            """Builds the value and advantage heads of the network based on the passed configuration.

            :param net_config: Configuration of the network head.
            :type net_config: Dict[str, Any]
            """
            self.head_net = DuelingMLP(
                num_inputs=self.latent_dim,
                num_outputs=self.num_actions,
                num_atoms=self.num_atoms,
                support=self.support,
                device=self.device,
                **net_config
            )

        def forward(
            self, obs: TorchObsType, q: bool = True, log: bool = False
        ) -> torch.Tensor:
            """Forward pass of the Rainbow Q network.

            :param obs: Input to the network.
            :type obs: torch.Tensor, dict[str, torch.Tensor], or list[torch.Tensor]
            :param q: Whether to return Q values. Defaults to True.
            :type q: bool
            :param log: Whether to return log probabilities. Defaults to False.
            :type log: bool

            :return: Output of the network.
            :rtype: torch.Tensor
            """
            latent = self.encoder(obs)
            return self.head_net(latent, q=q, log=log)

        def recreate_network(self) -> None:
            """Recreates the network"""
            encoder = self._build_encoder(self.encoder.net_config)

            head_net = DuelingMLP(
                num_inputs=self.latent_dim,
                num_outputs=self.num_actions,
                num_atoms=self.num_atoms,
                support=self.support,
                device=self.device,
                **self.head_net.net_config
            )

            self.encoder = EvolvableModule.preserve_parameters(self.encoder, encoder)
            self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)

This network is used by default in the :class:`RainbowDQN <agilerl.algorithms.dqn_rainbow.RainbowDQN>` agent, which is an extension of the
:class:`DQN <agilerl.algorithms.dqn.DQN>` that incorporates the Rainbow DQN improvements. However, users can employ their custom networks in
the implemented algorithms by passing them as ``actor_network`` and/or ``critic_network`` arguments in the agent's constructor. AgileRL will
automatically perform the enabled architecture mutations during training as part of its evolutionary hyperparameter optimization process!
