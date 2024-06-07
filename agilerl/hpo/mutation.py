import copy

import fastrand
import numpy as np
import torch

from agilerl.networks.evolvable_mlp import EvolvableMLP


class Mutations:
    """The Mutations class for evolutionary hyperparameter optimization.

    :param algo: RL algorithm. Use str e.g. 'DQN' if using AgileRL algos, or provide a dict with names of agent networks
    :type algo: str or dict
    :param no_mutation: Relative probability of no mutation
    :type no_mutation: float
    :param architecture: Relative probability of architecture mutation
    :type architecture: float
    :param new_layer_prob: Relative probability of new layer mutation (type of architecture mutation)
    :type new_layer_prob: float
    :param parameters: Relative probability of network parameters mutation
    :type parameters: float
    :param activation: Relative probability of activation layer mutation
    :type activation: float
    :param rl_hp: Relative probability of learning hyperparameter mutation
    :type rl_hp: float
    :param rl_hp_selection: Learning hyperparameter mutations to choose from
    :type rl_hp_selection: list[str]
    :param mutation_sd: Mutation strength
    :type mutation_sd: float
    :param activation_selection: Activation functions to choose from, defaults to ["ReLU", "ELU", "GELU"]
    :type activation_selection: list[str], optional
    :param min_lr: Minimum learning rate in the hyperparameter search space
    :type min_lr: float, optional
    :param max_lr: Maximum learning rate in the hyperparameter search space
    :type max_lr: float, optional
    :param min_learn_step: Minimum learn step in the hyperparameter search space
    :type min_learn_step: int, optional
    :param max_learn_step: Maximum learn step in the hyperparameter search space
    :type max_learn_step: int, optional
    :param min_batch_size: Minimum batch size in the hyperparameter search space
    :type min_batch_size: int, optional
    :param max_batch_size: Maximum batch size in the hyperparameter search space
    :type max_batch_size: int, optional
    :param agents_id: List of agent ID's for multi-agent algorithms
    :type agents_id: list[str]
    :param arch: Network architecture type. 'mlp' or 'cnn', defaults to 'mlp'
    :type arch: str, optional
    :param mutate_elite: Mutate elite member of population, defaults to True
    :type mutate_elite: bool, optional
    :param rand_seed: Random seed for repeatability, defaults to None
    :type rand_seed: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: accelerate.Accelerator(), optional
    """

    def __init__(
        self,
        algo,
        no_mutation,
        architecture,
        new_layer_prob,
        parameters,
        activation,
        rl_hp,
        rl_hp_selection=["lr", "batch_size", "learn_step"],
        mutation_sd=0.1,
        activation_selection=["ReLU", "ELU", "GELU"],
        min_lr=0.0001,
        max_lr=0.01,
        min_learn_step=1,
        max_learn_step=120,
        min_batch_size=8,
        max_batch_size=1024,
        agent_ids=None,
        arch="mlp",
        mutate_elite=True,
        rand_seed=None,
        device="cpu",
        accelerator=None,
    ):
        assert isinstance(
            algo, (str, dict)
        ), "Algo must be string e.g. 'DQN' or a dictionary with agent network names."
        assert isinstance(
            no_mutation, (float, int)
        ), "Probability of no mutation must be a float or integer."
        assert (
            no_mutation >= 0
        ), "Probability of no mutation must be greater than or equal to zero."
        assert isinstance(
            architecture, (float, int)
        ), "Probability of architecture mutation must be a float or integer."
        assert (
            architecture >= 0
        ), "Probability of architecture mutation must be greater than or equal to zero."
        assert isinstance(
            new_layer_prob, (float, int)
        ), "Probability of new layer architecture mutation must be a float or integer."
        assert (
            1 >= new_layer_prob >= 0
        ), "Probability of new layer architecture mutation must be between zero and one (inclusive)."
        assert isinstance(
            parameters, (float, int)
        ), "Probability of parameters mutation must be a float or integer."
        assert (
            parameters >= 0
        ), "Probability of parameters mutation must be greater than or equal to zero."
        assert isinstance(
            activation, (float, int)
        ), "Probability of activation mutation must be a float or integer."
        assert (
            activation >= 0
        ), "Probability of activation mutation must be greater than or equal to zero."
        assert isinstance(
            rl_hp, (float, int)
        ), "Probability of reinforcement learning hyperparameter mutation must be a float or integer."
        assert (
            rl_hp >= 0
        ), "Probability of reinforcement learning hyperparameter mutation must be greater than or equal to zero."
        if rl_hp > 0:
            assert isinstance(
                rl_hp_selection, list
            ), "Reinforcement learning hyperparameter mutation options must be a list."
            assert (
                len(rl_hp_selection) >= 0
            ), "Reinforcement learning hyperparameter mutation options list must contain at least one option."
        assert (
            mutation_sd >= 0
        ), "Mutation strength must be greater than or equal to zero."
        assert isinstance(
            mutation_sd, (float, int)
        ), "Mutation strength must be a float or integer."
        assert isinstance(min_lr, float), "Minimum learning rate must be a float."
        assert min_lr > 0, "Minimum learning rate must be greater than zero."
        assert isinstance(max_lr, float), "Maximum learning rate must be a float."
        assert max_lr > 0, "Maximum learning rate must be greater than zero."
        assert isinstance(
            min_learn_step, int
        ), "Minimum learn step rate must be an integer."
        assert (
            min_learn_step >= 1
        ), "Minimum learn step must be greater than or equal to one."
        assert isinstance(
            max_learn_step, int
        ), "Maximum learn step rate must be an integer."
        assert (
            max_learn_step >= 1
        ), "Maximum learn step must be greater than or equal to one."
        assert isinstance(
            min_batch_size, int
        ), "Minimum batch size rate must be an integer."
        assert (
            min_batch_size >= 1
        ), "Minimum batch size must be greater than or equal to one."
        assert isinstance(
            max_batch_size, int
        ), "Maximum batch size rate must be an integer."
        assert (
            max_batch_size >= 1
        ), "Maximum batch size must be greater than or equal to one."
        assert isinstance(
            mutate_elite, bool
        ), "Mutate elite must be boolean value True or False."
        assert (
            isinstance(rand_seed, int) or rand_seed is None
        ), "Random seed must be an integer or None."
        if isinstance(rand_seed, int):
            assert rand_seed >= 0, "Random seed must be greater than or equal to zero."

        # Random seed for repeatability
        self.rng = np.random.RandomState(rand_seed)

        self.arch = arch  # Network architecture type

        # Relative probabilities of mutation
        self.no_mut = no_mutation  # No mutation
        self.architecture_mut = architecture  # Architecture mutation
        # New layer mutation (type of architecture mutation)
        self.new_layer_prob = new_layer_prob
        self.parameters_mut = parameters  # Network parameters mutation
        self.activation_mut = activation  # Activation layer mutation
        self.rl_hp_mut = rl_hp  # Learning HP mutation
        self.activation_selection = activation_selection  # Learning HPs to choose from
        self.rl_hp_selection = rl_hp_selection  # Learning HPs to choose from
        self.mutation_sd = mutation_sd  # Mutation strength
        self.mutate_elite = mutate_elite
        self.device = device
        self.accelerator = accelerator
        self.agent_ids = agent_ids
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_learn_step = min_learn_step
        self.max_learn_step = max_learn_step

        if algo in ["MADDPG", "MATD3"]:
            self.multi_agent = True
        else:
            self.multi_agent = False

        # Set algorithm dictionary with agent network names for mutation
        # Use custom agent dict, or pre-configured agent from API
        if isinstance(algo, dict):
            self.algo = algo
        else:
            self.algo = self.get_algo_nets(algo)

    def no_mutation(self, individual):
        """Returns individual from population without mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        individual.mut = "None"  # No mutation
        return individual

    # Generic mutation function - gather mutation options and select from these
    def mutation(self, population, pre_training_mut=False):
        """Returns mutated population.

        :param population: Population of agents
        :type population: list[object]
        :param pre_training_mut: Boolean flag indicating if the mutation is before the training loop
        :type pre_training_mut: bool, optional
        """
        # Create lists of possible mutation functions and their respective
        # relative probabilities
        mutation_options = []
        mutation_proba = []
        if self.no_mut:
            mutation_options.append(self.no_mutation)
            if pre_training_mut:
                mutation_proba.append(float(0))
            else:
                mutation_proba.append(float(self.no_mut))
        if self.architecture_mut:
            mutation_options.append(self.architecture_mutate)
            mutation_proba.append(float(self.architecture_mut))
        if self.parameters_mut:
            mutation_options.append(self.parameter_mutation)
            mutation_proba.append(float(self.parameters_mut))
        if self.activation_mut:
            mutation_options.append(self.activation_mutation)
            mutation_proba.append(float(self.activation_mut))
        if self.rl_hp_mut:
            mutation_options.append(self.rl_hyperparam_mutation)
            mutation_proba.append(float(self.rl_hp_mut))

        if len(mutation_options) == 0:  # Return if no mutation options
            return population

        mutation_proba = np.array(mutation_proba) / np.sum(
            mutation_proba
        )  # Normalize probs

        # Randomly choose mutation for each agent in population from options with
        # relative probabilities
        mutation_choice = self.rng.choice(
            mutation_options, len(population), p=mutation_proba
        )

        # If not mutating elite member of population (first in list from tournament selection),
        # set this as the first mutation choice
        if not self.mutate_elite:
            mutation_choice[0] = self.no_mutation

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):
            # Call mutation function for each individual
            individual = mutation(individual)

            if self.multi_agent:
                offspring_actors = getattr(individual, self.algo["actor"]["eval"])

                # Reinitialise target network with frozen weights due to potential
                # mutation in architecture of value network
                ind_targets = [
                    type(offspring_actor)(**offspring_actor.init_dict)
                    for offspring_actor in offspring_actors
                ]

                for ind_target, offspring_actor in zip(ind_targets, offspring_actors):
                    ind_target.load_state_dict(offspring_actor.state_dict())

                if self.accelerator is None:
                    ind_targets = [
                        ind_target.to(self.device) for ind_target in ind_targets
                    ]
                setattr(individual, self.algo["actor"]["target"], ind_targets)

                # If algorithm has critics, reinitialize their respective target networks
                # too
                for critics_list in self.algo["critics"]:
                    offspring_critics = getattr(individual, critics_list["eval"])
                    ind_targets = [
                        type(offspring_critic)(**offspring_critic.init_dict)
                        for offspring_critic in offspring_critics
                    ]

                    for ind_target, offspring_critic in zip(
                        ind_targets, offspring_critics
                    ):
                        ind_target.load_state_dict(offspring_critic.state_dict())

                    if self.accelerator is None:
                        ind_targets = [
                            ind_target.to(self.device) for ind_target in ind_targets
                        ]
                    setattr(individual, critics_list["target"], ind_targets)
            else:
                if "target" in self.algo["actor"].keys():
                    offspring_actor = getattr(individual, self.algo["actor"]["eval"])

                    # Reinitialise target network with frozen weights due to potential
                    # mutation in architecture of value network
                    ind_target = type(offspring_actor)(**offspring_actor.init_dict)
                    ind_target.load_state_dict(offspring_actor.state_dict())
                    if self.accelerator is not None:
                        setattr(individual, self.algo["actor"]["target"], ind_target)
                    else:
                        setattr(
                            individual,
                            self.algo["actor"]["target"],
                            ind_target.to(self.device),
                        )

                    # If algorithm has critics, reinitialize their respective target networks
                    # too
                    for critic in self.algo["critics"]:
                        offspring_critic = getattr(individual, critic["eval"])
                        ind_target = type(offspring_critic)(
                            **offspring_critic.init_dict
                        )
                        ind_target.load_state_dict(offspring_critic.state_dict())
                        if self.accelerator is not None:
                            setattr(individual, critic["target"], ind_target)
                        else:
                            setattr(
                                individual, critic["target"], ind_target.to(self.device)
                            )

            mutated_population.append(individual)

        return mutated_population

    def reinit_opt(self, individual):
        if self.multi_agent:
            # Reinitialise optimizer
            actor_opts = getattr(individual, self.algo["actor"]["optimizer"])

            net_params = [
                actor.parameters()
                for actor in getattr(individual, self.algo["actor"]["eval"])
            ]

            offspring_actor_opts = [
                type(actor_opt)(net_param, lr=individual.lr_actor)
                for actor_opt, net_param in zip(actor_opts, net_params)
            ]

            setattr(
                individual,
                self.algo["actor"]["optimizer"],
                offspring_actor_opts,
            )

            for critic_list in self.algo["critics"]:
                critic_opts = getattr(individual, critic_list["optimizer"])

                net_params = [
                    critic.parameters()
                    for critic in getattr(individual, critic_list["eval"])
                ]

                offspring_critic_opts = [
                    type(critic_opt)(net_param, lr=individual.lr_critic)
                    for critic_opt, net_param in zip(critic_opts, net_params)
                ]

                setattr(
                    individual,
                    critic_list["optimizer"],
                    offspring_critic_opts,
                )
        else:
            if individual.algo in ["PPO"]:
                # Reinitialise optimizer
                opt = getattr(individual, self.algo["actor"]["optimizer"])
                actor_net_params = getattr(
                    individual, self.algo["actor"]["eval"]
                ).parameters()
                critic_net_params = getattr(
                    individual, self.algo["critics"][0]["eval"]
                ).parameters()
                opt_args = [
                    {"params": actor_net_params, "lr": individual.lr},
                    {"params": critic_net_params, "lr": individual.lr},
                ]
                setattr(
                    individual,
                    self.algo["actor"]["optimizer"],
                    type(opt)(opt_args),
                )

            else:
                # Reinitialise optimizer
                actor_opt = getattr(individual, self.algo["actor"]["optimizer"])
                net_params = getattr(
                    individual, self.algo["actor"]["eval"]
                ).parameters()
                if individual.algo in ["DDPG", "TD3"]:
                    setattr(
                        individual,
                        self.algo["actor"]["optimizer"],
                        type(actor_opt)(net_params, lr=individual.lr_actor),
                    )
                else:
                    setattr(
                        individual,
                        self.algo["actor"]["optimizer"],
                        type(actor_opt)(net_params, lr=individual.lr),
                    )

                # If algorithm has critics, reinitialise their optimizers too
                for critic in self.algo["critics"]:
                    critic_opt = getattr(individual, critic["optimizer"])
                    net_params = getattr(individual, critic["eval"]).parameters()
                    setattr(
                        individual,
                        critic["optimizer"],
                        type(critic_opt)(net_params, lr=individual.lr_critic),
                    )

    def rl_hyperparam_mutation(self, individual):
        """Returns individual from population with RL hyperparameter mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        # Learning hyperparameter mutation
        rl_params = self.rl_hp_selection
        # Select HP to mutate from options
        mutate_param = self.rng.choice(rl_params, 1)[0]

        # Increase or decrease HP randomly (within clipped limits)
        if mutate_param == "batch_size":
            bs_multiplication_options = [1.2, 0.8]  # Grow or shrink
            bs_probs = [0.5, 0.5]  # Equal probability
            bs_mult = self.rng.choice(bs_multiplication_options, size=1, p=bs_probs)[0]
            individual.batch_size = min(
                self.max_batch_size,
                max(self.min_batch_size, int(individual.batch_size * bs_mult)),
            )
            individual.mut = "bs"

        elif mutate_param == "lr":
            lr_multiplication_options = [1.2, 0.8]  # Grow or shrink
            lr_probs = [0.5, 0.5]  # Equal probability
            lr_mult = self.rng.choice(lr_multiplication_options, size=1, p=lr_probs)[0]
            if individual.algo in ["DDPG", "TD3", "MADDPG", "MATD3"]:
                lr_choice = self.rng.choice(
                    ["lr_actor", "lr_critic"], size=1, p=lr_probs
                )[0]
            else:
                lr_choice = "lr"
            setattr(
                individual,
                lr_choice,
                min(
                    self.max_lr,
                    max(self.min_lr, getattr(individual, lr_choice) * lr_mult),
                ),
            )
            self.reinit_opt(individual)  # Reinitialise optimizer if new learning rate
            individual.mut = lr_choice

        elif mutate_param == "learn_step":
            if individual.algo in ["PPO"]:  # Needs to stay constant for on-policy
                individual.mut = "None"
                return individual
            ls_multiplication_options = [1.5, 0.75]  # Grow or shrink
            ls_probs = [0.5, 0.5]  # Equal probability
            ls_mult = self.rng.choice(ls_multiplication_options, size=1, p=ls_probs)[0]
            individual.learn_step = min(
                self.max_learn_step,
                max(self.min_learn_step, int(individual.learn_step * ls_mult)),
            )
            individual.mut = "ls"

        return individual

    def activation_mutation(self, individual):
        """Returns individual from population with activation layer mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        if individual.algo in ["PPO", "DDPG", "TD3"]:  # Needs to stay constant
            individual.mut = "None"
            return individual

        if self.multi_agent:
            if individual.algo == "MADDPG" or individual.algo == "MATD3":
                individual.mut = "None"
                return individual
            else:
                offspring_actors = getattr(individual, self.algo["actor"]["eval"])
                offspring_actors = [
                    self._permutate_activation(offspring_actor)
                    for offspring_actor in offspring_actors
                ]
                if self.accelerator is None:
                    offspring_actors = [
                        offspring_actor.to(self.device)
                        for offspring_actor in offspring_actors
                    ]
                setattr(individual, self.algo["actor"]["eval"], offspring_actors)

                # If algorithm has critics, mutate their activations too
                for critics in self.algo["critics"]:
                    offspring_critics = getattr(individual, critics["eval"])
                    offspring_critics = [
                        self._permutate_activation(offspring_critic)
                        for offspring_critic in offspring_critics
                    ]
                    if self.accelerator is None:
                        offspring_critics = [
                            offspring_critic.to(self.device)
                            for offspring_critic in offspring_critics
                        ]
                    setattr(individual, critics["eval"], offspring_critics)
        else:
            # Mutate network activation layer
            offspring_actor = getattr(individual, self.algo["actor"]["eval"])
            offspring_actor = self._permutate_activation(
                offspring_actor
            )  # Mutate activation function
            if self.accelerator is not None:
                setattr(individual, self.algo["actor"]["eval"], offspring_actor)
            else:
                setattr(
                    individual,
                    self.algo["actor"]["eval"],
                    offspring_actor.to(self.device),
                )

            # If algorithm has critics, mutate their activations too
            for critic in self.algo["critics"]:
                offspring_critic = getattr(individual, critic["eval"])
                offspring_critic = self._permutate_activation(offspring_critic)
                if self.accelerator is not None:
                    setattr(individual, critic["eval"], offspring_critic)
                else:
                    setattr(
                        individual, critic["eval"], offspring_critic.to(self.device)
                    )

            if individual.algo in ["NeuralUCB", "NeuralTS"]:
                if self.arch == "mlp":
                    if isinstance(individual.actor, EvolvableMLP):
                        individual.exp_layer = (
                            individual.actor.feature_net.linear_layer_output
                        )
                    else:
                        individual.exp_layer = (
                            individual.actor.feature_net.feature_linear_layer_output
                        )
                else:
                    individual.exp_layer = (
                        individual.actor.value_net.value_linear_layer_output
                    )

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "act"
        return individual

    def _permutate_activation(self, network):
        # Function to change network activation layer
        possible_activations = copy.deepcopy(self.activation_selection)
        current_activation = network.mlp_activation
        # Remove current activation from options to ensure different new
        # activation layer
        if len(possible_activations) > 1 and current_activation in possible_activations:
            possible_activations.remove(current_activation)
        new_activation = self.rng.choice(possible_activations, size=1)[
            0
        ]  # Select new activation
        net_dict = network.init_dict
        if self.arch == "cnn":
            net_dict["mlp_activation"] = new_activation
            net_dict["cnn_activation"] = new_activation
        else:  # mlp, gpt or bert
            net_dict["mlp_activation"] = new_activation
        new_network = type(network)(**net_dict)
        new_network.load_state_dict(network.state_dict())
        network = new_network

        if self.accelerator is None:
            network = network.to(self.device)

        return network

    def parameter_mutation(self, individual):
        """Returns individual from population with network parameters mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        # Mutate network parameters
        if self.multi_agent:
            offspring_actors = getattr(individual, self.algo["actor"]["eval"])
            offspring_actors = [
                self.classic_parameter_mutation(offspring_actor)
                for offspring_actor in offspring_actors
            ]
            if self.accelerator is None:
                offspring_actors = [
                    offspring_actor.to(self.device)
                    for offspring_actor in offspring_actors
                ]
            setattr(individual, self.algo["actor"]["eval"], offspring_actors)
        else:
            offspring_actor = getattr(individual, self.algo["actor"]["eval"])
            offspring_actor = self.classic_parameter_mutation(
                offspring_actor
            )  # Network parameter mutation function
            if self.accelerator is not None:
                setattr(individual, self.algo["actor"]["eval"], offspring_actor)
            else:
                setattr(
                    individual,
                    self.algo["actor"]["eval"],
                    offspring_actor.to(self.device),
                )
        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "param"
        return individual

    def regularize_weight(self, weight, mag):
        if weight > mag:
            weight = mag
        if weight < -mag:
            weight = -mag
        return weight

    def classic_parameter_mutation(self, network):
        """Returns network with mutated weights.

        :param network: Neural network to mutate
        :type individual: torch.nn.Module
        """
        # Function to mutate network weights with Gaussian noise
        mut_strength = self.mutation_sd
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        model_params = network.state_dict()

        potential_keys = []
        for i, key in enumerate(model_params):  # Mutate each param
            if "norm" not in key:
                W = model_params[key]
                if len(W.shape) == 2:  # Weights, no bias
                    potential_keys.append(key)

        how_many = np.random.randint(1, len(potential_keys) + 1, 1)[0]
        chosen_keys = np.random.choice(potential_keys, how_many, replace=False)

        for key in chosen_keys:
            # References to the variable keys
            W = model_params[key]
            num_weights = W.shape[0] * W.shape[1]
            # Number of mutation instances
            num_mutations = fastrand.pcg32bounded(
                int(np.ceil(num_mutation_frac * num_weights))
            )
            for _ in range(num_mutations):
                ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                random_num = self.rng.uniform(0, 1)

                if random_num < super_mut_prob:  # Super Mutation probability
                    W[ind_dim1, ind_dim2] += self.rng.normal(
                        0, np.abs(super_mut_strength * W[ind_dim1, ind_dim2].item())
                    )
                elif random_num < reset_prob:  # Reset probability
                    W[ind_dim1, ind_dim2] = self.rng.normal(0, 1)
                else:  # mutauion even normal
                    W[ind_dim1, ind_dim2] += self.rng.normal(
                        0, np.abs(mut_strength * W[ind_dim1, ind_dim2].item())
                    )

                # Regularization hard limit
                W[ind_dim1, ind_dim2] = self.regularize_weight(
                    W[ind_dim1, ind_dim2].item(), 1000000
                )

        if self.accelerator is None:
            network = network.to(self.device)

        return network

    def architecture_mutate(self, individual):
        """Returns individual from population with network architecture mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        # Mutate network architecture by adding layers or nodes
        if self.multi_agent:
            offspring_actors = [
                actor.clone()
                for actor in getattr(individual, self.algo["actor"]["eval"])
            ]  # List of actors
            offspring_critics_list = [
                [critic.clone() for critic in getattr(individual, critics["eval"])]
                for critics in self.algo["critics"]
            ]

            if self.arch == "cnn":
                mutation_methods = [
                    "add_mlp_layer",
                    "remove_mlp_layer",
                    "add_cnn_layer",
                    "remove_cnn_layer",
                    "change_cnn_kernel",
                    "add_cnn_channel",
                    "remove_cnn_channel",
                    "add_mlp_node",
                    "remove_mlp_node",
                ]
                mut_probs = [
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                ]
                mut_method = self.rng.choice(mutation_methods, size=1, p=mut_probs)[0]
                for offspring_actor in offspring_actors:
                    actor_mutation = getattr(offspring_actor, mut_method)
                    actor_mutation()
                for offspring_critics in offspring_critics_list:
                    for offspring_critic in offspring_critics:
                        critic_mutation = getattr(offspring_critic, mut_method)
                        critic_mutation()

            elif self.arch == "bert":
                mutation_methods = [
                    "add_encoder_layer",
                    "remove_encoder_layer",
                    "add_decoder_layer",
                    "remove_decoder_layer",
                    "add_node",
                    "remove_node",
                ]
                mut_probs = [
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    (1 - self.new_layer_prob) * 0.5,
                    (1 - self.new_layer_prob) * 0.5,
                ]
                mut_method = self.rng.choice(mutation_methods, size=1, p=mut_probs)[0]
                if mut_method in [
                    "add_node",
                    "remove_node",
                ]:  # Functionality to account for MATD3
                    if len(offspring_critics_list) == 1:
                        for offspring_actor, offspring_critic in zip(
                            offspring_actors, offspring_critics_list[0]
                        ):
                            actor_mutation = getattr(offspring_actor, mut_method)
                            node_dict = actor_mutation()
                            critic_mutation = getattr(offspring_critic, mut_method)
                            critic_mutation(**node_dict)
                    else:
                        for (
                            offspring_actor,
                            offspring_critic_1,
                            offspring_critic_2,
                        ) in zip(offspring_actors, *offspring_critics_list):
                            actor_mutation = getattr(offspring_actor, mut_method)
                            node_dict = actor_mutation()
                            critic_mutation_1 = getattr(offspring_critic_1, mut_method)
                            critic_mutation_2 = getattr(offspring_critic_2, mut_method)
                            critic_mutation_1(**node_dict)
                            critic_mutation_2(**node_dict)
                else:
                    for offspring_actor in offspring_actors:
                        actor_mutation = getattr(offspring_actor, mut_method)
                        actor_mutation()
                    for offspring_critics in offspring_critics_list:
                        for offspring_critic in offspring_critics:
                            critic_mutation = getattr(offspring_critic, mut_method)
                            critic_mutation()

            else:  # mlp or gpt
                mutation_methods = [
                    "add_mlp_layer",
                    "remove_mlp_layer",
                    "add_mlp_node",
                    "remove_mlp_node",
                ]
                mut_probs = [
                    self.new_layer_prob / 2,
                    self.new_layer_prob / 2,
                    (1 - self.new_layer_prob) * 0.5,
                    (1 - self.new_layer_prob) * 0.5,
                ]
                mut_method = self.rng.choice(mutation_methods, size=1, p=mut_probs)[0]
                if mut_method in [
                    "add_mlp_node",
                    "remove_mlp_node",
                ]:  # Functionality to account for MATD3
                    if len(offspring_critics_list) == 1:
                        for offspring_actor, offspring_critic in zip(
                            offspring_actors, offspring_critics_list[0]
                        ):
                            actor_mutation = getattr(offspring_actor, mut_method)
                            node_dict = actor_mutation()
                            critic_mutation = getattr(offspring_critic, mut_method)
                            critic_mutation(**node_dict)
                    else:
                        for (
                            offspring_actor,
                            offspring_critic_1,
                            offspring_critic_2,
                        ) in zip(offspring_actors, *offspring_critics_list):
                            actor_mutation = getattr(offspring_actor, mut_method)
                            node_dict = actor_mutation()
                            critic_mutation_1 = getattr(offspring_critic_1, mut_method)
                            critic_mutation_2 = getattr(offspring_critic_2, mut_method)
                            critic_mutation_1(**node_dict)
                            critic_mutation_2(**node_dict)
                else:
                    for offspring_actor in offspring_actors:
                        actor_mutation = getattr(offspring_actor, mut_method)
                        actor_mutation()
                    for offspring_critics in offspring_critics_list:
                        for offspring_critic in offspring_critics:
                            critic_mutation = getattr(offspring_critic, mut_method)
                            critic_mutation()

            if self.accelerator is None:
                offspring_actors = [
                    offspring_actor.to(self.device)
                    for offspring_actor in offspring_actors
                ]
            setattr(individual, self.algo["actor"]["eval"], offspring_actors)

            for offspring_critics, critics in zip(
                offspring_critics_list, self.algo["critics"]
            ):
                if self.accelerator is None:
                    offspring_critics = [
                        offspring_critic.to(self.device)
                        for offspring_critic in offspring_critics
                    ]
                setattr(individual, critics["eval"], offspring_critics)

        else:
            if individual.algo in ["NeuralUCB", "NeuralTS"]:
                old_actor = getattr(individual, self.algo["actor"]["eval"]).clone()
                if self.arch == "mlp":
                    if isinstance(old_actor, EvolvableMLP):
                        old_exp_layer = old_actor.feature_net.linear_layer_output
                    else:
                        old_exp_layer = (
                            old_actor.feature_net.feature_linear_layer_output
                        )
                else:
                    old_exp_layer = old_actor.value_net.value_linear_layer_output

            offspring_actor = getattr(individual, self.algo["actor"]["eval"]).clone()
            offspring_critics = [
                getattr(individual, critic["eval"]).clone()
                for critic in self.algo["critics"]
            ]

            if self.arch == "cnn":
                mutation_methods = [
                    "add_mlp_layer",
                    "remove_mlp_layer",
                    "add_cnn_layer",
                    "remove_cnn_layer",
                    "change_cnn_kernel",
                    "add_cnn_channel",
                    "remove_cnn_channel",
                    "add_mlp_node",
                    "remove_mlp_node",
                ]
                mut_probs = [
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                    (1 - self.new_layer_prob) * 0.2,
                ]
                mut_method = self.rng.choice(mutation_methods, size=1, p=mut_probs)[0]
                actor_mutation = getattr(offspring_actor, mut_method)
                actor_mutation()
                for offspring_critic in offspring_critics:
                    critic_mutation = getattr(offspring_critic, mut_method)
                    critic_mutation()

            elif self.arch == "bert":
                mutation_methods = [
                    "add_encoder_layer",
                    "remove_encoder_layer",
                    "add_decoder_layer",
                    "remove_decoder_layer",
                    "add_node",
                    "remove_node",
                ]
                mut_probs = [
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    self.new_layer_prob / 4,
                    (1 - self.new_layer_prob) * 0.5,
                    (1 - self.new_layer_prob) * 0.5,
                ]
                mut_method = self.rng.choice(mutation_methods, size=1, p=mut_probs)[0]
                if mut_method in ["add_node", "remove_node"]:
                    actor_mutation = getattr(offspring_actor, mut_method)
                    node_dict = actor_mutation()
                    for offspring_critic in offspring_critics:
                        critic_mutation = getattr(offspring_critic, mut_method)
                        critic_mutation(**node_dict)
                else:
                    actor_mutation = getattr(offspring_actor, mut_method)
                    actor_mutation()
                    for offspring_critic in offspring_critics:
                        critic_mutation = getattr(offspring_critic, mut_method)
                        critic_mutation()

            else:  # mlp or gpt
                mutation_methods = [
                    "add_mlp_layer",
                    "remove_mlp_layer",
                    "add_mlp_node",
                    "remove_mlp_node",
                ]
                mut_probs = [
                    self.new_layer_prob / 2,
                    self.new_layer_prob / 2,
                    (1 - self.new_layer_prob) * 0.5,
                    (1 - self.new_layer_prob) * 0.5,
                ]
                mut_method = self.rng.choice(mutation_methods, size=1, p=mut_probs)[0]
                if mut_method in ["add_mlp_node", "remove_mlp_node"]:
                    actor_mutation = getattr(offspring_actor, mut_method)
                    node_dict = actor_mutation()
                    for offspring_critic in offspring_critics:
                        critic_mutation = getattr(offspring_critic, mut_method)
                        critic_mutation(**node_dict)
                else:
                    actor_mutation = getattr(offspring_actor, mut_method)
                    actor_mutation()
                    for offspring_critic in offspring_critics:
                        critic_mutation = getattr(offspring_critic, mut_method)
                        critic_mutation()

            if self.accelerator is not None:
                setattr(individual, self.algo["actor"]["eval"], offspring_actor)
            else:
                setattr(
                    individual,
                    self.algo["actor"]["eval"],
                    offspring_actor.to(self.device),
                )
            for offspring_critic, critic in zip(
                offspring_critics, self.algo["critics"]
            ):
                if self.accelerator is not None:
                    setattr(individual, critic["eval"], offspring_critic)
                else:
                    setattr(
                        individual, critic["eval"], offspring_critic.to(self.device)
                    )

            if individual.algo in ["NeuralUCB", "NeuralTS"]:
                self._reinit_bandit_grads(individual, offspring_actor, old_exp_layer)

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "arch"
        return individual

    def _reinit_bandit_grads(self, individual, offspring_actor, old_exp_layer):
        if self.arch == "mlp":
            if isinstance(offspring_actor, EvolvableMLP):
                exp_layer = offspring_actor.feature_net.linear_layer_output
            else:
                exp_layer = offspring_actor.feature_net.feature_linear_layer_output
        else:
            exp_layer = offspring_actor.value_net.value_linear_layer_output

        individual.numel = sum(
            w.numel() for w in exp_layer.parameters() if w.requires_grad
        )
        individual.theta_0 = torch.cat(
            [w.flatten() for w in exp_layer.parameters() if w.requires_grad]
        )

        # create matrix that is copy of sigma inv
        # first go through old params, figure out which to remove, then remove any difference
        # then go through new params, figure out where to add, then add zeros/lambda
        new_sigma_inv = copy.deepcopy(individual.sigma_inv).cpu().numpy()
        old_params = dict(old_exp_layer.named_parameters())
        new_params = dict(exp_layer.named_parameters())

        to_remove = []
        i = 0
        for key, param in old_exp_layer.named_parameters():
            if param.requires_grad:
                old_size = param.numel()
                if key not in new_params.keys():
                    to_remove += list(range(i, i + old_size))
                else:
                    new_size = new_params[key].numel()
                    if new_size < old_size:
                        to_remove += list(range(i + new_size, i + old_size))
                i += old_size

        to_add = []
        i = 0
        for key, param in exp_layer.named_parameters():
            if param.requires_grad:
                new_size = param.numel()
                if key in old_params.keys():
                    old_size = old_params[key].numel()
                    if new_size > old_size:
                        to_add += list(range(i + old_size, i + new_size))
                else:
                    to_add += list(range(i, i + new_size))
                i += new_size

        # Adjust indixes to add after deletion
        to_remove = np.array(to_remove)
        to_add = np.array(to_add)
        to_add -= np.sum(to_add[:, np.newaxis] > to_remove, axis=1)
        to_add -= np.arange(len(to_add))

        # Remove elements corresponding to old params
        if len(to_remove) > 0:
            new_sigma_inv = np.delete(
                np.delete(new_sigma_inv, to_remove, 0), to_remove, 1
            )

        # Add new zeros corresponding to new params, make lambda down identity diagonal
        if len(to_add) > 0:
            new_sigma_inv = np.insert(
                np.insert(new_sigma_inv, to_add, 0, 0), to_add, 0, 1
            )
            for i in to_add:
                new_sigma_inv[i, i] = individual.lamb

        individual.exp_layer = exp_layer
        individual.sigma_inv = torch.from_numpy(new_sigma_inv).to(
            individual.device
            if individual.accelerator is None
            else individual.accelerator.device
        )

    def get_algo_nets(self, algo):
        """Returns dictionary with agent network names.

        :param algo: RL algorithm
        :type algo: str
        """
        # Function to return dictionary with names of agent networks to allow mutation
        if algo == "DQN":
            nets = {
                "actor": {
                    "eval": "actor",
                    "target": "actor_target",
                    "optimizer": "optimizer",
                },
                "critics": [],
            }
        elif algo == "Rainbow DQN":
            nets = {
                "actor": {
                    "eval": "actor",
                    "target": "actor_target",
                    "optimizer": "optimizer",
                },
                "critics": [],
            }
        elif algo == "DDPG":
            nets = {
                "actor": {
                    "eval": "actor",
                    "target": "actor_target",
                    "optimizer": "actor_optimizer",
                },
                "critics": [
                    {
                        "eval": "critic",
                        "target": "critic_target",
                        "optimizer": "critic_optimizer",
                    }
                ],
            }
        elif algo == "PPO":
            nets = {
                "actor": {"eval": "actor", "optimizer": "optimizer"},
                "critics": [{"eval": "critic"}],
            }
        elif algo == "CQN":
            nets = {
                "actor": {
                    "eval": "actor",
                    "target": "actor_target",
                    "optimizer": "optimizer",
                },
                "critics": [],
            }
        elif algo == "ILQL":
            nets = {
                "actor": {
                    "eval": "actor",
                    "target": "actor_target",
                    "optimizer": "optimizer",
                },
                "critics": [],
            }
        elif algo == "TD3":
            nets = {
                "actor": {
                    "eval": "actor",
                    "target": "actor_target",
                    "optimizer": "actor_optimizer",
                },
                "critics": [
                    {
                        "eval": "critic_1",
                        "target": "critic_target_1",
                        "optimizer": "critic_1_optimizer",
                    },
                    {
                        "eval": "critic_2",
                        "target": "critic_target_2",
                        "optimizer": "critic_2_optimizer",
                    },
                ],
            }

        elif algo == "MADDPG":
            nets = {
                "actor": {
                    "eval": "actors",
                    "target": "actor_targets",
                    "optimizer": "actor_optimizers",
                },
                "critics": [
                    {
                        "eval": "critics",
                        "target": "critic_targets",
                        "optimizer": "critic_optimizers",
                    }
                ],
            }

        elif algo == "MATD3":
            nets = {
                "actor": {
                    "eval": "actors",
                    "target": "actor_targets",
                    "optimizer": "actor_optimizers",
                },
                "critics": [
                    {
                        "eval": "critics_1",
                        "target": "critic_targets_1",
                        "optimizer": "critic_1_optimizers",
                    },
                    {
                        "eval": "critics_2",
                        "target": "critic_targets_2",
                        "optimizer": "critic_2_optimizers",
                    },
                ],
            }
        elif algo in ["NeuralUCB", "NeuralTS"]:
            nets = {
                "actor": {"eval": "actor", "optimizer": "optimizer"},
                "critics": [],
            }

        return nets
