.. _base_algorithm:

EvolvableAlgorithm Base Class
=============================

We develop a class hierarchy for RL algorithms with a focus on making the evolution of their hyperparameters, and that of their
underlying neural networks, seamless. The base class implements methods and attributes that are used by :class:`Mutations <agilerl.hpo.mutation.Mutations>`
objects to mutate individuals of a population in a general manner. In order to this, we have created a framework for signalling the "network groups" in
an algorithm such that architecture mutations on the networks are applied correctly. Under the hood, all ``EvolvableAlgorithm`` objects create a
:class:`MutationRegistry <agilerl.algorithms.core.registry.MutationRegistry>` object that keeps a log of the network groups, optimizers, and the
hyperparameters of the algorithm that the user wishes to mutate during training.

We have base classes for single-agent and multi-agent algorithms, namely :class:`RLAlgorithm <agilerl.algorithms.core.base.RLAlgorithm>`
and :class:`MultiAgentRLAlgorithm <agilerl.algorithms.core.base.MultiAgentRLAlgorithm>`, respectively.

Network Groups
--------------

Users must specify the :class:`NetworkGroup <agilerl.algorithms.core.registry.NetworkGroup>`'s in their algorithm, which contain at least one
**evaluation** network (i.e. a network that is optimized during training e.g. the Q-network in DQN) and, optionally, "shared" networks that share
parameters with the evaluation network in the group but aren't optimized during training directly (e.g. the target network in DQN). An RL algorithm
must also contain one :class:`NetworkGroup <agilerl.algorithms.core.registry.NetworkGroup>` corresponding to the policy (i.e. the network used to
select actions), signalled by the ``policy`` attribute in the group.

OptimizerWrapper
----------------

The last thing users should do when creating a custom algorithm is wrap their optimizers in an :class:`OptimizerWrapper <agilerl.algorithms.core.wrappers.OptimizerWrapper>`,
specifying the networks that the optimizer is responsible for optimizing. Since we are mutating network architectures during training, we need to have knowledge of
this in order to reinitiliaze the optimizers correctly when we do so.

.. note::
    All of the network groups and optimizers of an algorithm should by convention be defined in the ``__init__`` method of the algorithm.

Example
-------

Below is a simple example of how this is can be done for the DDPG algorithm, which contains a combination of actors and critics. Here we have two network groups,
one for the actor and one for the critic (the actor being flagged as the policy since it is used to select actions), and each with their respective target networks
being flagged as having shared parameters. We also have two separate optimizers (one for each network groups evaluation network) that are wrapped in an
:class:`OptimizerWrapper <agilerl.algorithms.core.wrappers.OptimizerWrapper>`.

.. code-block:: python

    class DDPG(RLAlgorithm):
        """The DDPG algorithm class. DDPG paper: https://arxiv.org/abs/1509.02971

        :param observation_space: Environment observation space
        :type observation_space: gym.spaces.Space
        :param action_space: Environment action space
        :type action_space: gym.spaces.Space
        :param O_U_noise: Use Ornstein Uhlenbeck action noise for exploration. If False, uses Gaussian noise. Defaults to True
        :type O_U_noise: bool, optional
        :param expl_noise: Scale for Ornstein Uhlenbeck action noise, or standard deviation for Gaussian exploration noise, defaults to 0.1
        :type expl_noise: Union[float, ArrayLike], optional
        :param vect_noise_dim: Vectorization dimension of environment for action noise, defaults to 1
        :type vect_noise_dim: int, optional
        :param mean_noise: Mean of exploration noise, defaults to 0.0
        :type mean_noise: float, optional
        :param theta: Rate of mean reversion in Ornstein Uhlenbeck action noise, defaults to 0.15
        :type theta: float, optional
        :param dt: Timestep for Ornstein Uhlenbeck action noise update, defaults to 1e-2
        :type dt: float, optional
        :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
        :type index: int, optional
        :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
        :type hp_config: HyperparameterConfig, optional
        :param net_config: Encoder configuration, defaults to None
        :type net_config: Optional[Dict[str, Any]], optional
        :param head_config: Head configuration, defaults to None
        :type head_config: Optional[Dict[str, Any]], optional
        :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
        :type batch_size: int, optional
        :param lr_actor: Learning rate for actor optimizer, defaults to 1e-4
        :type lr_actor: float, optional
        :param lr_critic: Learning rate for critic optimizer, defaults to 1e-3
        :type lr_critic: float, optional
        :param learn_step: Learning frequency, defaults to 5
        :type learn_step: int, optional
        :param gamma: Discount factor, defaults to 0.99
        :type gamma: float, optional
        :param tau: For soft update of target network parameters, defaults to 1e-3
        :type tau: float, optional
        :param normalize_images: Normalize images flag, defaults to True
        :type normalize_images: bool, optional
        :param mut: Most recent mutation to agent, defaults to None
        :type mut: Optional[str], optional
        :param policy_freq: Frequency of critic network updates compared to policy network, defaults to 2
        :type policy_freq: int, optional
        :param actor_network: Custom actor network, defaults to None
        :type actor_network: Optional[nn.Module], optional
        :param critic_network: Custom critic network, defaults to None
        :type critic_network: Optional[nn.Module], optional
        :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
        :type device: str, optional
        :param accelerator: Accelerator for distributed computing, defaults to None
        :type accelerator: accelerate.Accelerator(), optional
        :param wrap: Wrap models for distributed training upon creation, defaults to True
        :type wrap: bool, optional
        """

        def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            O_U_noise: bool = True,
            expl_noise: Union[float, ArrayLike] = 0.1,
            vect_noise_dim: int = 1,
            mean_noise: float = 0.0,
            theta: float = 0.15,
            dt: float = 1e-2,
            index: int = 0,
            hp_config: Optional[HyperparameterConfig] = None,
            net_config: Optional[Dict[str, Any]] = None,
            batch_size: int = 64,
            lr_actor: float = 1e-4,
            lr_critic: float = 1e-3,
            learn_step: int = 5,
            gamma: float = 0.99,
            tau: float = 1e-3,
            normalize_images: bool = True,
            mut: Optional[str] = None,
            policy_freq: int = 2,
            actor_network: Optional[EvolvableModule] = None,
            critic_network: Optional[EvolvableModule] = None,
            device: str = "cpu",
            accelerator: Optional[Any] = None,
            wrap: bool = True,
        ) -> None:

            super().__init__(
                observation_space,
                action_space,
                index=index,
                hp_config=hp_config,
                device=device,
                accelerator=accelerator,
                normalize_images=normalize_images,
                name="DDPG",
            )

            assert learn_step >= 1, "Learn step must be greater than or equal to one."
            assert isinstance(learn_step, int), "Learn step rate must be an integer."
            assert isinstance(
                action_space, spaces.Box
            ), "DDPG only supports continuous action spaces."
            assert (isinstance(expl_noise, (float, int))) or (
                isinstance(expl_noise, np.ndarray)
                and expl_noise.shape == (vect_noise_dim, self.action_dim)
            ), f"Exploration action noise rate must be a float, or an array of size {self.action_dim}"
            if isinstance(expl_noise, (float, int)):
                assert (
                    expl_noise >= 0
                ), "Exploration noise must be greater than or equal to zero."
            assert isinstance(batch_size, int), "Batch size must be an integer."
            assert batch_size >= 1, "Batch size must be greater than or equal to one."
            assert isinstance(lr_actor, float), "Actor learning rate must be a float."
            assert lr_actor > 0, "Actor learning rate must be greater than zero."
            assert isinstance(lr_critic, float), "Critic learning rate must be a float."
            assert lr_critic > 0, "Critic learning rate must be greater than zero."
            assert isinstance(learn_step, int), "Learn step rate must be an integer."
            assert learn_step >= 1, "Learn step must be greater than or equal to one."
            assert isinstance(gamma, (float, int, torch.Tensor)), "Gamma must be a float."
            assert isinstance(tau, float), "Tau must be a float."
            assert tau > 0, "Tau must be greater than zero."
            assert isinstance(policy_freq, int), "Policy frequency must be an integer."
            assert (
                policy_freq >= 1
            ), "Policy frequency must be greater than or equal to one."

            if (actor_network is not None) != (critic_network is not None):  # XOR operation
                warnings.warn(
                    "Actor and critic networks must both be supplied to use custom networks. Defaulting to net config."
                )
            assert isinstance(
                wrap, bool
            ), "Wrap models flag must be boolean value True or False."

            self.batch_size = batch_size
            self.lr_actor = lr_actor
            self.lr_critic = lr_critic
            self.learn_step = learn_step
            self.net_config = net_config
            self.gamma = gamma
            self.tau = tau
            self.wrap = wrap
            self.mut = mut
            self.policy_freq = policy_freq
            self.O_U_noise = O_U_noise
            self.vect_noise_dim = vect_noise_dim
            self.expl_noise = (
                expl_noise
                if isinstance(expl_noise, np.ndarray)
                else expl_noise * np.ones((vect_noise_dim, self.action_dim))
            )
            self.mean_noise = (
                mean_noise
                if isinstance(mean_noise, np.ndarray)
                else mean_noise * np.ones((vect_noise_dim, self.action_dim))
            )
            self.current_noise = np.zeros((vect_noise_dim, self.action_dim))
            self.theta = theta
            self.dt = dt
            self.learn_counter = 0

            if actor_network is not None and critic_network is not None:
                if not isinstance(actor_network, EvolvableModule):
                    raise TypeError(
                        f"'actor_network' is of type {type(actor_network)}, but must be of type EvolvableModule."
                    )
                if not isinstance(critic_network, EvolvableModule):
                    raise TypeError(
                        f"'critic_network' is of type {type(critic_network)}, but must be of type EvolvableModule."
                    )

                self.actor, self.critic = make_safe_deepcopies(
                    actor_network, critic_network
                )
                self.actor_target, self.critic_target = make_safe_deepcopies(
                    actor_network, critic_network
                )
            else:
                net_config = {} if net_config is None else net_config
                head_config = net_config.get("head_config", None)
                if head_config is not None:
                    critic_head_config = copy.deepcopy(head_config)
                    critic_head_config["output_activation"] = None
                else:
                    critic_head_config = MlpNetConfig(hidden_size=[64])

                critic_net_config = copy.deepcopy(net_config)
                critic_net_config["head_config"] = critic_head_config

                def create_actor():
                    return DeterministicActor(
                        observation_space=observation_space,
                        action_space=action_space,
                        device=device,
                        **net_config,
                    )

                def create_critic():
                    return ContinuousQNetwork(
                        observation_space=observation_space,
                        action_space=action_space,
                        device=device,
                        **critic_net_config,
                    )

                self.actor = create_actor()
                self.actor_target = create_actor()
                self.critic = create_critic()
                self.critic_target = create_critic()

            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

            # Optimizers
            self.actor_optimizer = OptimizerWrapper(
                optim.Adam, networks=self.actor, lr=lr_actor
            )
            self.critic_optimizer = OptimizerWrapper(
                optim.Adam, networks=self.critic, lr=lr_critic
            )

            if self.accelerator is not None and wrap:
                self.wrap_models()

            self.criterion = nn.MSELoss()

            # Register network groups for actors and critics
            self.register_network_group(
                NetworkGroup(eval=self.actor, shared=self.actor_target, policy=True)
            )
            self.register_network_group(
                NetworkGroup(eval=self.critic, shared=self.critic_target)
            )


Parameters
------------

.. autoclass:: agilerl.algorithms.core.base.EvolvableAlgorithm
    :members:

.. autoclass:: agilerl.algorithms.core.base.RLAlgorithm
    :members:

.. autoclass:: agilerl.algorithms.core.base.MultiAgentRLAlgorithm
    :members:
