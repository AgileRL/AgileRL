.. _rainbow_dqn_tutorial:

Building a Dueling Distributional Q Network Using AgileRL
=========================================================

.. note::
    Here we go through the process we followed to develop the network in the :class:`RainbowDQN <agilerl.algorithms.dqn_rainbow.RainbowDQN>` agent.
    However, users can employ their custom networks in the implemented algorithms by passing them as ``actor_network`` and/or ``critic_network``
    arguments in the agent's constructor. AgileRL will automatically perform the enabled architecture mutations during training as part of its
    evolutionary hyperparameter optimization process!


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

2. **Creating a Custom MLP**: We can create a custom MLP that inherits from :class:`EvolvableMLP <agilerl.modules.mlp.EvolvableMLP>` and add the advantage head without having to
disable the mutations on it.

For either of the above solutions, we need to be able to recreate the network after an architecture mutation such that the same mutation is applied to both the
value and advantage heads. We can do this by overwriting the :meth:`recreate_network() <agilerl.modules.base.EvolvableModule.recreate_network>` method in our custom MLP.
For more information, please refer to the `EvolvableMLP <https://github.com/AgileRL/AgileRL/blob/complex-spaces/agilerl/modules/mlp.py#L9>`_ implementation for an example of
the complete requirements of a :class:`EvolvableModule <agilerl.modules.base.EvolvableModule>` object.

DuelingDistributionalMLP
------------------------

Below we show our implementation of our custom head with a distributional dueling architecture, using noisy linear layers as described in the paper. In
:meth:`recreate_network() <agilerl.modules.base.EvolvableModule.recreate_network>`, we use the same parameters as the value network to create the advantage network after mutating it.

.. code-block:: python

    class DuelingDistributionalMLP(EvolvableMLP):
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
            activation: str = "ReLU",
            output_activation: str = None,
            min_hidden_layers: int = 1,
            max_hidden_layers: int = 3,
            min_mlp_nodes: int = 64,
            max_mlp_nodes: int = 500,
            new_gelu: bool = False,
            device: str = "cpu",
        ) -> None:

            super().__init__(
                num_inputs,
                num_atoms,
                hidden_size,
                activation,
                output_activation,
                min_hidden_layers,
                max_hidden_layers,
                min_mlp_nodes,
                max_mlp_nodes,
                layer_norm=True,
                output_vanish=True,
                init_layers=False,
                noisy=True,
                noise_std=noise_std,
                new_gelu=new_gelu,
                device=device,
                name="value",
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
                name="advantage",
            )

        @property
        def net_config(self) -> Dict[str, Any]:
            net_config = super().net_config.copy()
            net_config.pop("num_atoms")
            net_config.pop("support")
            return net_config

        def forward(
            self, x: torch.Tensor, q: bool = True, log: bool = False
        ) -> torch.Tensor:
            """Forward pass of the network.

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
                name="advantage",
            )

            self.advantage_net = EvolvableModule.preserve_parameters(
                self.advantage_net, advantage_net
            )


Creating a Custom Evolvable Network
------------------------------------------------------------------------------------

Now that we have our custom head, we can create a custom network that inherits from :class:`EvolvableNetwork <agilerl.networks.base.EvolvableNetwork>`
and uses our custom head. Since we have done most of the work in the head, the implementation is quite simple and analogous to the
:class:`QNetwork <agilerl.networks.q_networks.QNetwork>` implementation. We only need to change the head to our custom head and update the
:meth:`recreate_network() <agilerl.networks.base.EvolvableNetwork.recreate_network>` method to reflect the changes in the head.

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
            self.head_net = DuelingDistributionalMLP(
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

            head_net = DuelingDistributionalMLP(
                num_inputs=self.latent_dim,
                num_outputs=self.num_actions,
                num_atoms=self.num_atoms,
                support=self.support,
                device=self.device,
                **self.head_net.net_config
            )

            self.encoder = EvolvableModule.preserve_parameters(self.encoder, encoder)
            self.head_net = EvolvableModule.preserve_parameters(self.head_net, head_net)

Available Architecture Mutations in the Network
------------------------------------------------

When defining complex networks with nested ``EvolvabelModule`` objects like the one above, it is useful to inspect the available architecture mutations
that can be applied to the network. This can be done by calling the ``mutation_methods`` attribute of the network object.

.. code-block:: python

    import torch
    from gymnasium import spaces

    # Define an image observation space and a discrete action space
    observation_space = spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
    action_space = spaces.Discrete(4)

    support = torch.linspace(-10, 10, 51)

    network = RainbowQNetwork(
        observation_space=observation_space,
        action_space=action_space,
        support=torch.linspace(-10, 10, 51), # Support for the DuelingDistributionalMLP
        )

    print(network.mutation_methods)

This will output the following list of available mutations. We can recognise the mutation methods of the underlying ``EvolvableCNN`` encoder, the
``DuelingDistributionalMLP`` head, and the **add_latent_node** and **remove_latent_node** mutations that are available for all instances of ``EvolvabelNetwork``.

.. code-block:: text

    [
    'head_net.remove_layer',
    'head_net.add_layer',
    'add_latent_node',
    'remove_latent_node',
    'encoder.remove_channel',
    'encoder.add_channel',
    'encoder.change_kernel',
    'head_net.remove_node',
    'head_net.add_node'
    ]

Training the Rainbow DQN Agent
------------------------------

Now that we have our custom network, we can define it with a specific architecture and pass it to the
:class:`RainbowDQN <agilerl.algorithms.dqn_rainbow.RainbowDQN>` agent as the ``actor_network`` argument. The agent will automatically mutate the architecture
of the network with the corresponding probability specified in the ``architecture`` argument of ::class:`Mutations <agilerl.hpo.mutation.Mutations>`.

End-to-end example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import imageio
    import gymnasium as gym
    import numpy as np
    import torch

    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.algorithms.dqn_rainbow import RainbowDQN
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.networks import RainbowQNetwork
    from agilerl.components.replay_buffer import (
        MultiStepReplayBuffer,
        PrioritizedReplayBuffer,
    )
    from agilerl.training.train_off_policy import train_off_policy
    from agilerl.utils.utils import make_vect_envs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    num_envs=16
    env = make_vect_envs("CartPole-v1", num_envs=num_envs)

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # Hyperparameters
    INIT_HP = {
        "BATCH_SIZE": 64,  # Batch size
        "LR": 0.0001,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 100_000,  # Max memory buffer size
        "LEARN_STEP": 1,  # Learning frequency
        "N_STEP": 3,  # Step number to calculate td error
        "PER": True,  # Use prioritized experience replay buffer
        "ALPHA": 0.6,  # Prioritized replay buffer parameter
        "BETA": 0.4,  # Importance sampling coefficient
        "TAU": 0.001,  # For soft update of target parameters
        "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
        "NUM_ATOMS": 51,  # Unit number of support
        "V_MIN": -200.0,  # Minimum value of support
        "V_MAX": 200.0,  # Maximum value of support
        "NOISY": True,  # Add noise directly to the weights of the network
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "LEARNING_DELAY": 1000,  # Steps before starting learning
        "CHANNELS_LAST": False,  # Use with RGB states
        "TARGET_SCORE": 200.0,  # Target score that will beat the environment
        "MAX_STEPS": 200000,  # Maximum number of steps an agent takes in an environment
        "EVO_STEPS": 10000,  # Evolution frequency
        "EVAL_STEPS": None,  # Number of evaluation steps per episode
        "EVAL_LOOP": 1,  # Number of evaluation episodes
        "TOURN_SIZE": 4,  # Tournament size
        "POP_SIZE": 4,  # Population size
        "ELITISM": True,  # Use elitism in the tournament
    }

    MUTATION_PARAMS = {
        "NO_MUTATION": 0.4,  # Probability of no mutation
        "ARCHITECTURE": 0.2,  # Probability of architecture mutation
        "NEW_LAYER_PROB": 0.2,  # Probability of adding a new layer
        "PARAMETERS": 0.2,  # Probability of changing parameters
        "ACTIVATION": 0.2,  # Probability of changing activation function
        "RL_HP": 0.2,  # Probability of changing RL hyperparameters
        "MUTATION_SD": 0.1,  # Standard deviation of the mutation
        "RAND_SEED": 42,  # Random seed
    }

    # Actor architecture configuration
    NET_CONFIG = {
        "latent_dim": 32, # latent dimension for observation encodings
        "encoder_config": {
            "hidden_size": [64] # Encoder hidden size
        },
        "head_config": {
            "hidden_size": [64] # Head hidden size
        }
    }

    # Define the support for the distributional value function and the custom actor
    support = torch.linspace(INIT_HP['V_MIN'], INIT_HP['V_MAX'], INIT_HP['NUM_ATOMS'], device=device)
    actor = RainbowQNetwork(
        observation_space=observation_space,
        action_space=action_space,
        support=support,
        device=device,
        **NET_CONFIG
    )

    # RL hyperparameters configuration for mutation during training
    hp_config = HyperparameterConfig(
        lr = RLParameter(min=6.25e-5, max=1e-2),
        learn_step = RLParameter(min=1, max=10, dtype=int),
        batch_size = RLParameter(
            min=8, max=512, dtype=int
            )
    )

    # Tournament selection
    tournament = TournamentSelection(
        tournament_size=INIT_HP["TOURN_SIZE"],
        elitism=INIT_HP["ELITISM"],
        population_size=INIT_HP["POP_SIZE"],
        eval_loop=INIT_HP["EVAL_LOOP"],
    )

    # Define the mutation parameters
    mutations = Mutations(
        no_mutation=MUTATION_PARAMS["NO_MUTATION"],
        architecture=MUTATION_PARAMS["ARCHITECTURE"],
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER_PROB"],
        parameters=MUTATION_PARAMS["PARAMETERS"],
        activation=MUTATION_PARAMS["ACTIVATION"],
        rl_hp=MUTATION_PARAMS["RL_HP"],
        mutation_sd=MUTATION_PARAMS["MUTATION_SD"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    # Define a population of agents
    agent_pop = RainbowDQN.population(
        size=INIT_HP['POP_SIZE'], # Number of individuals to mutate
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor,
        hp_config=hp_config,
        batch_size=INIT_HP["BATCH_SIZE"],
        lr=INIT_HP["LR"],
        learn_step=INIT_HP["LEARN_STEP"],
        gamma=INIT_HP["GAMMA"],
        tau=INIT_HP["TAU"],
        beta=INIT_HP["BETA"],
        prior_eps=INIT_HP["PRIOR_EPS"],
        num_atoms=INIT_HP["NUM_ATOMS"],
        v_min=INIT_HP["V_MIN"],
        v_max=INIT_HP["V_MAX"],
        n_step=INIT_HP["N_STEP"],
        device=device
    )

    # Prioritised experience replay with N-step memory
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = PrioritizedReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        num_envs=num_envs,
        alpha=INIT_HP["ALPHA"],
        gamma=INIT_HP["GAMMA"],
        device=device,
    )
    n_step_memory = MultiStepReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        num_envs=num_envs,
        n_step=INIT_HP["N_STEP"],
        gamma=INIT_HP["GAMMA"],
        device=device,
    )

    # Train the agent
    trained_pop, pop_fitnesses = train_off_policy(
        env,
        "CartPole-v1",
        "RainbowDQN",
        agent_pop,
        memory=memory,
        n_step_memory=n_step_memory,
        n_step=True,
        per=True,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
    )
