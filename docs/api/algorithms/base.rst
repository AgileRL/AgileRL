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
    All of the network groups and optimizers of an algorithm should by convention all be defined in the ``__init__`` method of the algorithm.

Example
-------

Below is a simple example of how this is can be done for the DQN algorithm:

.. code-block:: python

    class DQN(RLAlgorithm):
        """The DQN algorithm class. DQN paper: https://arxiv.org/abs/1312.5602

        :param observation_space: Observation space of the environment
        :type observation_space: gymnasium.spaces.Space
        :param action_space: Action space of the environment
        :type action_space: gymnasium.spaces.Space
        :param index: Index to keep track of object instance during tournament selection and mutation, defaults to 0
        :type index: int, optional
        :param hp_config: RL hyperparameter mutation configuration, defaults to None, whereby algorithm mutations are disabled.
        :type hp_config: HyperparameterConfig, optional
        :param net_config: Network configuration, defaults to None
        :type net_config: dict, optional
        :param batch_size: Size of batched sample from replay buffer for learning, defaults to 64
        :type batch_size: int, optional
        :param lr: Learning rate for optimizer, defaults to 1e-4
        :type lr: float, optional
        :param learn_step: Learning frequency, defaults to 5
        :type learn_step: int, optional
        :param gamma: Discount factor, defaults to 0.99
        :type gamma: float, optional
        :param tau: For soft update of target network parameters, defaults to 1e-3
        :type tau: float, optional
        :param mut: Most recent mutation to agent, defaults to None
        :type mut: str, optional
        :param double: Use double Q-learning, defaults to False
        :type double: bool, optional
        :param actor_network: Custom actor network, defaults to None
        :type actor_network: nn.Module, optional
        :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
        :type device: str, optional
        :param accelerator: Accelerator for distributed computing, defaults to None
        :type accelerator: accelerate.Accelerator(), optional
        :param cudagraphs: Use CUDA graphs for optimization, defaults to False
        :type cudagraphs: bool, optional
        :param wrap: Wrap models for distributed training upon creation, defaults to True
        :type wrap: bool, optional
        """

        def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            index: int = 0,
            hp_config: Optional[HyperparameterConfig] = None,
            net_config: Optional[Dict[str, Any]] = None,
            batch_size: int = 64,
            lr: float = 1e-4,
            learn_step: int = 5,
            gamma: float = 0.99,
            tau: float = 1e-3,
            mut: Optional[str] = None,
            double: bool = False,
            normalize_images: bool = True,
            actor_network: Optional[nn.Module] = None,
            device: str = "cpu",
            accelerator: Optional[Any] = None,
            cudagraphs: bool = False,
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
                name="DQN"
            )

            assert learn_step >= 1, "Learn step must be greater than or equal to one."
            assert isinstance(learn_step, int), "Learn step rate must be an integer."
            assert isinstance(batch_size, int), "Batch size must be an integer."
            assert batch_size >= 1, "Batch size must be greater than or equal to one."
            assert isinstance(lr, float), "Learning rate must be a float."
            assert lr > 0, "Learning rate must be greater than zero."
            assert isinstance(gamma, (float, int)), "Gamma must be a float."
            assert isinstance(tau, float), "Tau must be a float."
            assert tau > 0, "Tau must be greater than zero."
            assert isinstance(
                double, bool
            ), "Double Q-learning flag must be boolean value True or False."
            assert isinstance(
                wrap, bool
            ), "Wrap models flag must be boolean value True or False."

            self.batch_size = batch_size
            self.lr = lr
            self.learn_step = learn_step
            self.gamma = gamma
            self.tau = tau
            self.mut = mut
            self.double = double
            self.net_config = net_config
            self.cudagraphs = cudagraphs
            self.capturable = cudagraphs

            if actor_network is not None:
                if not isinstance(actor_network, EvolvableModule):
                    raise TypeError(
                        f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableModule."
                        )

                # Need to make deepcopies for target and detached networks
                self.actor, self.actor_target = make_safe_deepcopies(actor_network, actor_network)
            else:
                net_config = {} if net_config is None else net_config
                create_actor = lambda: QNetwork(
                    observation_space=observation_space,
                    action_space=action_space,
                    device=self.device,
                    **net_config
                )
                self.actor = create_actor()
                self.actor_target = create_actor()

            # Copy over actor weights to target
            self.init_hook()

            # Initialize optimizer with OptimizerWrapper
            self.optimizer = OptimizerWrapper(
                optim.Adam,
                networks=self.actor,
                lr=self.lr,
                optimizer_kwargs={"capturable": self.capturable}
            )

            if self.accelerator is not None and wrap:
                self.wrap_models()

            self.criterion = nn.MSELoss()

            # torch.compile and cuda graph optimizations
            if self.cudagraphs:
                warnings.warn("CUDA graphs for DQN are implemented experimentally and may not work as expected.")
                self.update = torch.compile(self.update, mode=None)
                self._get_action = torch.compile(self._get_action, mode=None, fullgraph=True)
                self.update = CudaGraphModule(self.update)
                self._get_action = CudaGraphModule(self._get_action)

            # Register DQN network groups and mutation hook
            self.register_network_group(
                NetworkGroup(
                    eval=self.actor,
                    shared=[self.actor_target],
                    policy=True
                )
            )
            self.register_init_hook(self.init_hook)


Parameters
------------

.. autoclass:: agilerl.algorithms.core.base.EvolvableAlgorithm
    :members:

.. autoclass:: agilerl.algorithms.core.base.RLAlgorithm
    :members:

.. autoclass:: agilerl.algorithms.core.base.MultiAgentRLAlgorithm
    :members:
