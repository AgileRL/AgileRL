import numpy as np
import fastrand


class Mutations():
    """The Mutations class for evolutionary hyperparameter optimization.

    :param algo: RL algorithm used. Use str e.g. 'DQN' if using AgileRL implementation of algorithm, or provide a dict with names of agent networks
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
    :type rl_hp_selection: List[str]
    :param mutation_sd: Mutation strength
    :type mutation_sd: float
    :param arch: Network architecture type. 'mlp' or 'cnn', defaults to 'mlp'
    :type arch: str, optional
    :param rand_seed: Random seed for repeatability, defaults to None
    :type rand_seed: int, optional
    :param device: Device for accelerated computing, 'cpu' or 'cuda', defaults to 'cpu'
    :type device: str, optional
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
            rl_hp_selection,
            mutation_sd,
            arch='mlp',
            rand_seed=None,
            device='cpu'):
        # Random seed for repeatability
        self.rng = np.random.RandomState(rand_seed)

        self.arch = arch    # Network architecture type

        # Relative probabilities of mutation
        self.no_mut = no_mutation               # No mutation
        self.architecture_mut = architecture    # Architecture mutation
        # New layer mutation (type of architecture mutation)
        self.new_layer_prob = new_layer_prob
        self.parameters_mut = parameters        # Network parameters mutation
        self.activation_mut = activation        # Activation layer mutation
        self.rl_hp_mut = rl_hp                  # Learning HP mutation

        self.rl_hp_selection = rl_hp_selection  # Learning HPs to choose from
        self.mutation_sd = mutation_sd          # Mutation strength

        # Set algorithm dictionary with agent network names for mutation
        # Use custom agent dict, or pre-configured agent from API
        if isinstance(algo, dict):
            self.algo = algo
        else:
            self.algo = self.get_algo_nets(algo)

        self.device = device

    def no_mutation(self, individual):
        """Returns individual from population without mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        individual.mut = 'None'  # No mutation
        return individual

    # Generic mutation function - gather mutation options and select from these
    def mutation(self, population):
        """Returns mutated population.

        :param population: Population of agents
        :type population: List[object]
        """
        # Create lists of possible mutation functions and their respective
        # relative probabilities
        mutation_options = []
        mutation_proba = []
        if self.no_mut:
            mutation_options.append(self.no_mutation)
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

        mutation_proba = np.array(mutation_proba) / \
            np.sum(mutation_proba)  # Normalize probs

        # Raandomly choose mutation for each agent in population from options with
        # relative probabilities
        mutation_choice = self.rng.choice(
            mutation_options, len(population), p=mutation_proba)

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):

            # Call mutation function for each individual
            individual = mutation(individual)

            offspring_actor = getattr(individual, self.algo['actor']['eval'])

            # Reinitialise target network with frozen weights due to potential
            # mutation in architecture of value network
            ind_target = type(offspring_actor)(**offspring_actor.init_dict)
            ind_target.load_state_dict(offspring_actor.state_dict())
            setattr(individual, self.algo['actor']
                    ['target'], ind_target.to(self.device))

            # If algorithm has critics, reinitialize their respective target networks
            # too
            for critic in self.algo['critics']:
                offspring_critic = getattr(individual, critic['eval'])
                ind_target = type(offspring_critic)(
                    **offspring_critic.init_dict)
                ind_target.load_state_dict(offspring_critic.state_dict())
                setattr(individual, critic['target'],
                        ind_target.to(self.device))

            mutated_population.append(individual)

        return mutated_population

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
        random_num = self.rng.uniform(0, 1)
        if mutate_param == 'batch_size':
            if random_num > 0.5:
                individual.batch_size = min(
                    128, max(8, int(individual.batch_size * 1.2)))
            else:
                individual.batch_size = min(
                    128, max(8, int(individual.batch_size * 0.8)))
            individual.mut = 'bs'

        elif mutate_param == 'lr':
            if random_num > 0.5:
                individual.lr = min(0.005, max(0.00001, individual.lr * 1.2))
            else:
                individual.lr = min(0.005, max(0.00001, individual.lr * 0.8))

            # Reinitialise optimizer if new learning rate
            actor_opt = getattr(individual, self.algo['actor']['optimizer'])
            net_params = getattr(
                individual, self.algo['actor']['eval']).parameters()
            setattr(individual, self.algo['actor']['optimizer'], type(
                actor_opt)(net_params, lr=individual.lr))

            # If algorithm has critics, reinitialise their optimizers too
            for critic in self.algo['critics']:
                critic_opt = getattr(individual, critic['optimizer'])
                net_params = getattr(individual, critic['eval']).parameters()
                setattr(individual, critic['optimizer'], type(
                    critic_opt)(net_params, lr=individual.lr))
            individual.mut = 'lr'

        elif mutate_param == 'learn_step':
            if random_num > 0.5:
                individual.learn_step = min(
                    1, max(0, int(individual.learn_step * 1.5)))
            else:
                individual.learn_step = min(
                    1, max(0, int(individual.learn_step * 0.75)))

        return individual

    def activation_mutation(self, individual):
        """Returns individual from population with activation layer mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        if individual.algo == 'DDPG':   # Needs to stay tanh for DDPG continuous actions
            individual.mut = 'None'
            return individual

        # Mutate network activation layer
        offspring_actor = getattr(individual, self.algo['actor']['eval'])
        offspring_actor = self._permutate_activation(
            offspring_actor)   # Mutate activation function
        setattr(individual, self.algo['actor']
                ['eval'], offspring_actor.to(self.device))

        # If algorithm has critics, mutate their activations too
        for critic in self.algo['critics']:
            offspring_critic = getattr(individual, critic['eval'])
            offspring_critic = self._permutate_activation(offspring_critic)
            setattr(individual, critic['eval'],
                    offspring_critic.to(self.device))

        individual.mut = 'act'
        return individual

    def _permutate_activation(self, network):
        # Function to change network activation layer
        possible_activations = ['relu', 'elu', 'gelu']
        if self.arch == 'cnn':
            current_activation = network.mlp_activation
        else:   # mlp
            current_activation = network.activation
        # Remove current activation from options to ensure different new
        # activation layer
        possible_activations.remove(current_activation)
        new_activation = self.rng.choice(possible_activations, size=1)[
            0]   # Select new activation
        net_dict = network.init_dict
        if self.arch == 'cnn':
            net_dict['mlp_activation'] = new_activation
            net_dict['cnn_activation'] = new_activation
        else:   # mlp, gpt or bert
            net_dict['activation'] = new_activation
        new_network = type(network)(**net_dict)
        new_network.load_state_dict(network.state_dict())
        network = new_network

        return network.to(self.device)

    def parameter_mutation(self, individual):
        """Returns individual from population with network parameters mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        # Mutate network parameters
        offspring_actor = getattr(individual, self.algo['actor']['eval'])
        offspring_actor = self.classic_parameter_mutation(
            offspring_actor)  # Network parameter mutation function
        setattr(individual, self.algo['actor']
                ['eval'], offspring_actor.to(self.device))
        individual.mut = 'param'
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
            if 'norm' not in key:
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
                int(np.ceil(num_mutation_frac * num_weights)))
            for _ in range(num_mutations):
                ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                random_num = self.rng.uniform(0, 1)

                if random_num < super_mut_prob:  # Super Mutation probability
                    W[ind_dim1, ind_dim2] += self.rng.normal(
                        0, np.abs(super_mut_strength * W[ind_dim1, ind_dim2].item()))
                elif random_num < reset_prob:  # Reset probability
                    W[ind_dim1, ind_dim2] = self.rng.normal(0, 1)
                else:  # mutauion even normal
                    W[ind_dim1, ind_dim2] += self.rng.normal(
                        0, np.abs(mut_strength * W[ind_dim1, ind_dim2].item()))

                # Regularization hard limit
                W[ind_dim1, ind_dim2] = self.regularize_weight(
                    W[ind_dim1, ind_dim2].item(), 1000000)
        return network.to(self.device)

    def architecture_mutate(self, individual):
        """Returns individual from population with network architecture mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        # Mutate network architecture by adding layers or nodes
        offspring_actor = getattr(
            individual, self.algo['actor']['eval']).clone()
        offspring_critics = [getattr(individual, critic['eval']).clone()
                             for critic in self.algo['critics']]

        rand_numb = self.rng.uniform(0, 1)

        # Randomly select whether to add layer or node with relative probabilities
        # If algorithm has critics, apply to these too

        if self.arch == 'cnn':
            if rand_numb < self.new_layer_prob / 2:
                offspring_actor.add_mlp_layer()
                for offspring_critic in offspring_critics:
                    offspring_critic.add_mlp_layer()
            elif self.new_layer_prob / 2 <= rand_numb < self.new_layer_prob:
                offspring_actor.add_cnn_layer()
                for offspring_critic in offspring_critics:
                    offspring_critic.add_cnn_layer()
            else:
                rand_numb = self.rng.uniform(0, 1)
                if rand_numb < 0.2:
                    offspring_actor.change_cnn_kernal()
                    for offspring_critic in offspring_critics:
                        offspring_critic.change_cnn_kernal()
                elif 0.2 <= rand_numb < 0.65:
                    offspring_actor.add_cnn_channel()
                    for offspring_critic in offspring_critics:
                        offspring_critic.add_cnn_channel()
                else:
                    offspring_actor.add_mlp_node()
                    for offspring_critic in offspring_critics:
                        offspring_critic.add_mlp_node()

        elif self.arch == 'bert':
            if rand_numb < self.new_layer_prob / 2:
                if self.rng.uniform(0, 1) < 0.5:
                    offspring_actor.add_encoder_layer()
                    for offspring_critic in offspring_critics:
                        offspring_critic.add_encoder_layer()
                else:
                    offspring_actor.remove_encoder_layer()
                    for offspring_critic in offspring_critics:
                        offspring_critic.remove_encoder_layer()
            elif self.new_layer_prob / 2 <= rand_numb < self.new_layer_prob:
                if self.rng.uniform(0, 1) < 0.5:
                    offspring_actor.add_decoder_layer()
                    for offspring_critic in offspring_critics:
                        offspring_critic.add_decoder_layer()
                else:
                    offspring_actor.remove_decoder_layer()
                    for offspring_critic in offspring_critics:
                        offspring_critic.remove_decoder_layer()
            else:
                if self.rng.uniform(0, 1) < 0.5:
                    node_dict = offspring_actor.add_node()
                    for offspring_critic in offspring_critics:
                        offspring_critic.add_node(**node_dict)
                else:
                    node_dict = offspring_actor.remove_node()
                    for offspring_critic in offspring_critics:
                        offspring_critic.remove_node(**node_dict)

        else:   # mlp or gpt
            if rand_numb < self.new_layer_prob:
                if self.rng.uniform(0, 1) < 0.5:
                    offspring_actor.add_layer()
                    for offspring_critic in offspring_critics:
                        offspring_critic.add_layer()
                else:
                    offspring_actor.remove_layer()
                    for offspring_critic in offspring_critics:
                        offspring_critic.remove_layer()
            else:
                if self.rng.uniform(0, 1) < 0.5:
                    node_dict = offspring_actor.add_node()
                    for offspring_critic in offspring_critics:
                        offspring_critic.add_node(**node_dict)
                else:
                    node_dict = offspring_actor.remove_node()
                    for offspring_critic in offspring_critics:
                        offspring_critic.remove_node(**node_dict)

        setattr(individual, self.algo['actor']
                ['eval'], offspring_actor.to(self.device))
        for offspring_critic, critic in zip(offspring_critics, self.algo['critics']):
            setattr(individual, critic['eval'],
                    offspring_critic.to(self.device))

        individual.mut = 'arch'
        return individual

    def get_algo_nets(self, algo):
        """Returns dictionary with agent network names.

        :param algo: RL algorithm
        :type algo: string
        """
        # Function to return dictionary with names of agent networks to allow mutation
        if algo == 'DQN':
            nets = {
                'actor': {
                    'eval': 'actor',
                    'target': 'actor_target',
                    'optimizer': 'optimizer'
                },
                'critics': []
            }
        elif algo == 'DDPG':
            nets = {
                'actor': {
                    'eval': 'actor',
                    'target': 'actor_target',
                    'optimizer': 'actor_optimizer'
                },
                'critics': [{
                    'eval': 'critic',
                    'target': 'critic_target',
                    'optimizer': 'critic_optimizer'
                }]
            }
        elif algo == 'CQN':
            nets = {
                'actor': {
                    'eval': 'actor',
                    'target': 'actor_target',
                    'optimizer': 'optimizer'
                },
                'critics': []
            }
        elif algo == 'ILQL':
            nets = {
                'actor': {
                    'eval': 'actor',
                    'target': 'actor_target',
                    'optimizer': 'optimizer'
                },
                'critics': []
            }
        return nets
