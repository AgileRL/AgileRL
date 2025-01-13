from typing import List, Optional, Union, Dict, Callable, Any, Tuple, Type
import inspect
import copy
import numpy as np
import torch
import torch.nn as nn
import fastrand
from torch.optim import Optimizer
from accelerate import Accelerator
from torch._dynamo.eval_frame import OptimizedModule
from numpy.random import Generator

from agilerl.algorithms.neural_ts_bandit import NeuralTS
from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.algorithms.core import EvolvableAlgorithm
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.modules.mlp import EvolvableMLP
from agilerl.modules.cnn import EvolvableCNN
from agilerl.wrappers.make_evolvable import MakeEvolvable
from agilerl.modules.base import EvolvableModule
from agilerl.utils.algo_utils import remove_compile_prefix

NetworkConfig = Dict[str, str]
NetworkList = List[NetworkConfig]
MutationMethod = Callable[[EvolvableAlgorithm], EvolvableAlgorithm]
AlgoConfig = Dict[str, Union[NetworkConfig, NetworkList]]
PopulationType = List[EvolvableAlgorithm]
ModuleType = Union[OptimizedModule, EvolvableModule]
OffspringType = Union[List[EvolvableModule], EvolvableModule]
BanditAlgorithm = Union[NeuralUCB, NeuralTS]

def get_return_type(method: Callable) -> Any:
    """Get the return type of a method if annotated, otherwise return None.
    
    :param method: Method to inspect
    :type method: Callable
    :return: Return type of method
    :rtype: Any
    """
    try:
        signature = inspect.signature(method)
        return_type = signature.return_annotation
        if return_type is inspect.Signature.empty:
            return None  # No return type specified
        
        if hasattr(return_type, "__origin__"):
            return return_type.__origin__  # Return type is a type hint
        
        return return_type  # Return type is a class or type
    except ValueError as e:
        print(f"Error inspecting {method}: {e}")
        return None

def set_global_seed(seed: Optional[int]) -> None:
    """Set the global seed for random number generators.

    :param seed: Random seed for repeatability
    :type seed: int
    """
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    fastrand.pcg32_seed(seed)


def get_architecture_mut_method(
        eval: OffspringType,
        new_layer_prob: float,
        rng: Generator
        ) -> Tuple[str, Type]:
    """Get the mutation method and its return type of the individual.
    
    :param individual: The individual to inspect
    :type individual: EvolvableAlgorithm
    :param new_layer_prob: Relative probability of new layer mutation (type of architecture mutation)
    :type new_layer_prob: float
    :param rng: Random number generator
    :type rng: Generator
    :return: The mutation methods name and its return type
    :rtype: Tuple[str, Type]
    """ 
    # All of the offsprings should be the same EvolvableModule type, so we can 
    # just sample the mutation method from the first offspring
    if isinstance(eval, list):
        eval = eval[0]

    mutation_method = eval.sample_mutation_method(new_layer_prob, rng)
    mut_return_type = get_return_type(getattr(eval, mutation_method))

    return mutation_method, mut_return_type

def get_offspring_eval_modules(individual: EvolvableAlgorithm) -> Tuple[Dict[str, OffspringType], ...]:
    """Get the offsprings of all of the evaluation modules in the individual.
    
    :param individual: The individual to inspect
    :type individual: EvolvableAlgorithm
    
    :return: The offspring evaluation modules
    :rtype: Dict[str, OffspringType]
    """
    registry = individual.registry

    offspring_modules = {}
    offspring_policy = {}
    for group in registry.groups:
        eval_module: OffspringType = getattr(individual, group.eval)

        offspring = [mod.clone() for mod in eval_module] if isinstance(eval_module, list) else eval_module.clone()

        if group.policy:
            offspring_policy[group.eval] = offspring
        else:
            offspring_modules[group.eval] = offspring
    
    return offspring_policy, offspring_modules

def get_exp_layer(offspring: EvolvableModule) -> nn.Module:
    """Get the output layer of different types of offsprings for bandit algorithms. 
    Returns None if algorithm is not a bandit algorithm.
    
    :param offspring: The offspring to inspect
    :type offspring: EvolvableModule

    :return: The output layer of the offspring
    :rtype: nn.Module
    """
    if isinstance(offspring, EvolvableModule):
        exp_layer = offspring.get_output_dense()
    else:
        raise ValueError(f"Bandit algorithm architecture {type(offspring)} not supported.")
    
    return exp_layer


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
        algo: Union[str, AlgoConfig],
        no_mutation: float,
        architecture: float,
        new_layer_prob: float,
        parameters: float,
        activation: float,
        rl_hp: float,
        rl_hp_selection: List[str] = ["lr", "batch_size", "learn_step"],
        mutation_sd: float = 0.1,
        activation_selection: List[str] = ["ReLU", "ELU", "GELU"],
        min_lr: float = 0.0001,
        max_lr: float = 0.01,
        min_learn_step: int = 1,
        max_learn_step: int = 120,
        min_batch_size: int = 8,
        max_batch_size: int = 1024,
        agent_ids: Optional[List[str]] = None,
        mutate_elite: bool = True,
        rand_seed: Optional[int] = None,
        device: str = "cpu",
        accelerator: Optional[Accelerator] = None,
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
        set_global_seed(rand_seed)
        self.rng = np.random.default_rng(rand_seed)

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

        self.pretraining_mut_options, self.pretraining_mut_proba = self.get_mutations_options(pretraining=True)
        self.mut_options, self.mut_proba = self.get_mutations_options()
    
    def get_mutations_options(self, pretraining: bool = False) -> Tuple[List[Callable], List[float]]:
        """Get the mutation options and probabilities for the given mutation 
        configuration.
        
        :param pretraining: Boolean flag indicating if the mutation is before the training loop
        :type pretraining: bool
        :return: Mutation functions and their respective relative probabilities
        :rtype: Tuple[List[Callable], List[float]]
        """
        # Create lists of possible mutation functions and their 
        # respective relative probabilities
        mutation_options = [
            (self.no_mutation, self.no_mut),
            (self.architecture_mutate, self.architecture_mut),
            (self.parameter_mutation, self.parameters_mut),
            (self.activation_mutation, self.activation_mut),
            (self.rl_hyperparam_mutation, self.rl_hp_mut)
        ]

        if pretraining:
            mutation_options[0] = (self.no_mutation, 0)

        mutation_options = [(func, prob) for func, prob in mutation_options if prob > 0]

        # This will really only happen when pretraining is True and user has set 
        # all mutation probabilities to zero, hence we apply no mutation
        if len(mutation_options) == 0:
            mutation_options = [(self.no_mutation, 1)]

        mutation_funcs, mutation_proba = zip(*mutation_options)
        mutation_proba = np.array(mutation_proba) / np.sum(mutation_proba)
        return mutation_funcs, mutation_proba

    def no_mutation(self, individual: EvolvableAlgorithm):
        """Returns individual from population without mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        individual.mut = "None"  # No mutation
        return individual

    def to_device(self, offsprings: OffspringType) -> OffspringType:
        """Move offspring to device.
        
        :param offsprings: The offspring to move to device
        :type offsprings: OffspringType
        """
        if isinstance(offsprings, list):
            return [offspring.to(self.device) for offspring in offsprings]
        else:
            return offsprings.to(self.device)
    
    def reinit_module(self, module: EvolvableModule, init_dict: Dict[str, Any]) -> EvolvableModule:
        """Reinitialize the module with the given initialization dictionary.
        
        :param module: The module to reinitialize
        :type module: EvolvableModule
        
        :param init_dict: The initialization dictionary
        :type init_dict: Dict[str, Any]
        """
        if isinstance(module, torch._dynamo.eval_frame.OptimizedModule):
            module_cls = type(module._orig_mod)
        else:
            module_cls = type(module)
        
        return module_cls(**init_dict)

    def reinit_from_mutated(self, offspring: OffspringType) -> OffspringType:
        """Reinitialize the mutated offspring with their state dictionary.
        
        :param offspring: The offspring to reinitialize
        :type offspring: OffspringType
        
        :return: The reinitialized offspring
        :rtype: OffspringType
        """
        if isinstance(offspring, list):
            ind_shared = [
                self.reinit_module(offspring, offspring.init_dict) 
                for offspring in offspring
                ]

            # Load eval state dicts into shared networks                        
            state_dicts = [offspring.state_dict() for offspring in offspring]
            self.load_state_dicts(ind_shared, state_dicts)
        else:
            ind_shared = self.reinit_module(
                offspring,
                offspring.init_dict
                )
            ind_shared.load_state_dict(offspring.state_dict())
        
        return ind_shared
        

    def load_state_dicts(self, modules: List[ModuleType], state_dicts: List[Dict[str, Any]]) -> None:
        """Load the state dictionary into the module.
        
        :param module: The module to load the state dictionary into
        :type module: ModuleType
        
        :param state_dict: The state dictionary to load
        :type state_dict: Dict[str, Any]
        """
        for module, state_dict in zip(modules, state_dicts):
            if hasattr(module, "torch_compiler") and module.torch_compiler is not None:
                module.load_state_dict(remove_compile_prefix(state_dict))
            else:
                module.load_state_dict(state_dict)

    def compile_modules(self, modules: OffspringType, compiler: str) -> OffspringType:
        """Compile the modules using the given compiler.
        
        :param modules: The modules to compile
        :type modules: List[ModuleType]
        
        :param compiler: The compiler to use
        :type compiler: Optional[str]
        """
        single_offspring = not isinstance(modules, list)
        if single_offspring:
            modules = [modules]

        # Compile modules
        compiled_modules = []
        for module in modules:
            if not isinstance(module, torch._dynamo.eval_frame.OptimizedModule):
                compiled_modules.append(torch.compile(module, mode=compiler))
            else:
                compiled_modules.append(module)

        return compiled_modules if not single_offspring else compiled_modules[0]

    def mutation(self, population: PopulationType, pre_training_mut: bool=False) -> PopulationType:
        """Returns mutated population.

        :param population: Population of agents
        :type population: list[PopulationType]
        :param pre_training_mut: Boolean flag indicating if the mutation is before the training loop
        :type pre_training_mut: bool, optional
        """
        # Create lists of possible mutation functions and their respective
        # relative probabilities
        mutation_options = self.pretraining_mut_options if pre_training_mut else self.mut_options
        mutation_proba = self.pretraining_mut_proba if pre_training_mut else self.mut_proba

        # Randomly choose mutation for each agent in population from options with
        # relative probabilities
        mutation_choice: List[MutationMethod] = self.rng.choice(
            mutation_options, len(population), p=mutation_proba
        )

        # If not mutating elite member of population (first in list from tournament selection),
        # set this as the first mutation choice
        if not self.mutate_elite:
            mutation_choice[0] = self.no_mutation

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population):
            registry = individual.registry

            # Call mutation function for each individual
            individual = mutation(individual)

            # Recompile modules if applicable
            if hasattr(individual, "torch_compiler") and individual.torch_compiler:
                individual.recompile()
            
            # Reinitiliaze shared networks to mutated evaluation networks
            for net_group in registry.groups:
                if net_group.shared is not None:
                    for shared_name in net_group.shared:
                        eval_offspring: OffspringType = getattr(individual, net_group.eval)

                        # Reinitialize shared with frozen weights due to 
                        # potential mutation in architecture
                        ind_shared = self.reinit_from_mutated(eval_offspring)

                        if self.accelerator is None:
                            ind_shared = self.to_device(ind_shared)

                        # Compile modules if necessary
                        if hasattr(individual, "torch_compiler") and individual.torch_compiler:
                            ind_shared = self.compile_modules(ind_shared, individual.torch_compiler)
                        
                        setattr(individual, shared_name, ind_shared)

            # Call hooks specified by user
            individual.init_hook()

            mutated_population.append(individual)

        return mutated_population

    def reinit_opt(self, individual: EvolvableAlgorithm) -> None:
        """Reinitialize the optimizers of an individual.
        
        :param individual: The individual to reinitialize the optimizers for
        :type individual: EvolvableAlgorithm
        """
        optimizer_configs = individual.registry.optimizers
        for opt_config in optimizer_configs:
            opt: OptimizerWrapper = getattr(individual, opt_config.name)
            optimizer = opt.optimizer

            # Multiple optimizers in a single attribute (i.e. multi-agent)
            # or one module optimized by a single optimizer
            if isinstance(optimizer, list) or len(opt.network_names) == 1:
                opt_nets = getattr(individual, opt.network_names[0])

            # Multiple modules optimized by a single optimizer (e.g. PPO)
            else:
                opt_nets = [getattr(individual, net) for net in opt.network_names]
            
            # Reinitialize optimizer with mutated nets
            offspring_opt = OptimizerWrapper(
                optimizer_cls=opt_config.get_optimizer_cls(),
                networks=opt_nets,
                optimizer_kwargs=opt.optimizer_kwargs,
                network_names=opt.network_names,
            )

            setattr(individual, opt_config.name, offspring_opt)

    # TODO: Generalize this based on argument specification
    def rl_hyperparam_mutation(self, individual: EvolvableAlgorithm) -> EvolvableAlgorithm:
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

    def activation_mutation(self, individual: EvolvableAlgorithm) -> EvolvableAlgorithm:
        """Returns individual from population with activation layer mutation.

        :param individual: Individual agent from population
        :type individual: EvolvableAlgorithm
        """
        # Needs to stay constant for policy gradient methods
        # TODO: Could set up an algorithm registry to make algo checks more robust
        if individual.algo in ["PPO", "DDPG", "TD3", "MADDPG", "MATD3"]:
            individual.mut = "None"
            return individual

        # Mutate network activation layer
        registry = individual.registry
        for network_group in registry.groups:
            eval_module: OffspringType = getattr(individual, network_group.eval)
            if isinstance(eval_module, list):
                eval_module = [self._permutate_activation(mod) for mod in eval_module]
            else:
                eval_module = self._permutate_activation(eval_module)
            
            if self.accelerator is None:
                eval_module = self.to_device(eval_module)

            if isinstance(individual, (NeuralTS, NeuralUCB)):
                individual.exp_layer = get_exp_layer(eval_module)

            setattr(individual, network_group.eval, eval_module)

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "act"
        return individual

    def _permutate_activation(self, network: EvolvableModule) -> EvolvableModule:
        # Function to change network activation layer
        possible_activations = copy.deepcopy(self.activation_selection)
        current_activation = network.activation

        # Remove current activation from options to ensure different new
        # activation layer
        if len(possible_activations) > 1 and current_activation in possible_activations:
            possible_activations.remove(current_activation)

        new_activation = self.rng.choice(possible_activations, size=1)[0]  # Select new activation
        network.change_activation(new_activation, output=False)  # Change activation layer
        return network

    def parameter_mutation(self, individual: EvolvableAlgorithm) -> EvolvableAlgorithm:
        """Returns individual from population with network parameters mutation.

        :param individual: Individual agent from population
        :type individual: EvolvableAlgorithm
        """
        registry = individual.registry

        # We only apply parameter mutations to the evaluation policy network 
        # (i.e. the network used to select actions)
        offspring_policy: OffspringType = getattr(individual, registry.policy)
        if isinstance(offspring_policy, list):
            offspring_policy = [self.classic_parameter_mutation(mod) for mod in offspring_policy]
        else:
            offspring_policy = self.classic_parameter_mutation(offspring_policy)
        
        if self.accelerator is None:
            offspring_policy = self.to_device(offspring_policy)
        
        setattr(individual, registry.policy, offspring_policy)

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "param"
        return individual

    def regularize_weight(self, weight: float, mag: float) -> float:
        """Regularize the weight to be within the specified magnitude.

        :param weight: The weight to be regularized
        :type weight: float
        :param mag: The magnitude limit
        :type mag: float
        :return: The regularized weight
        :rtype: float
        """
        if weight > mag:
            weight = mag
        if weight < -mag:
            weight = -mag
        return weight

    def classic_parameter_mutation(self, network: EvolvableModule) -> EvolvableModule:
        """Returns network with mutated weights.

        :param network: Neural network to mutate
        :type network: EvolvableModule
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

        how_many = self.rng.integers(1, len(potential_keys) + 1, 1)[0]
        chosen_keys = self.rng.choice(potential_keys, how_many, replace=False)

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

    def architecture_mutate(self, individual: EvolvableAlgorithm) -> EvolvableAlgorithm:
        """Returns individual from population with network architecture mutation, which 
        adds either layers or nodes to different types of network architectures.

        :param individual: Individual agent from population
        :type individual: object
        """
        algo_cls = individual.__class__

        # Get the offspring evaluation modules
        # We first extract and apply a mutation for the algo policy and then apply 
        # the same mutation to the rest of the evaluation modules e.g. critics
        policy, offspring_evals = get_offspring_eval_modules(individual)
        policy_name, policy_offspring = list(policy.items())[0]

        # Sample mutation method from policy network
        mut_method, ret_type = get_architecture_mut_method(
            policy_offspring, self.new_layer_prob, self.rng
            )
        print(mut_method)
        mut_dict = self._apply_arch_mutation(policy_offspring, mut_method, ret_type)
        self.to_device_and_set_individual(individual, policy_name, policy_offspring)

        if algo_cls in [NeuralTS, NeuralUCB]:
            old_exp_layer = get_exp_layer(policy_offspring)
            self._reinit_bandit_grads(individual, policy_offspring, old_exp_layer)

        # Apply the same mutation to the rest of the evaluation modules
        for name, offsprings in offspring_evals.items():
            self._apply_arch_mutation(offsprings, mut_method, ret_type, mut_dict)
            self.to_device_and_set_individual(individual, name, offsprings)

            # Reinitialize bandit gradients after architecture mutation
            if algo_cls in [NeuralTS, NeuralUCB]:
                old_exp_layer = get_exp_layer(offsprings)
                self._reinit_bandit_grads(individual, offsprings, old_exp_layer)

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "arch"
        return individual

    def _apply_arch_mutation(
        self, 
        networks: OffspringType, 
        mut_method: str, 
        ret_type: Type, 
        applied_mut_dict: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Applies the mutation method to networks and returns mutation data if needed.
        
        :param networks: The networks to apply the mutation to
        :type networks: OffspringType
        :param mut_method: The mutation method to apply
        :type mut_method: str
        :param ret_type: The return type of the mutation method
        :type ret_type: Type
        :param mut_dict: The mutation dictionary, defaults to None
        :type mut_dict: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional
        :return: The mutation dictionary if ret_type is dict, otherwise None
        :rtype: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
        """
        if applied_mut_dict is None:
            applied_mut_dict = [{}] * len(networks) if isinstance(networks, list) else {}

        mut_dict = None
        if ret_type != dict:
            if isinstance(networks, list):
                for net in networks:
                    getattr(net, mut_method)()
            else:
                getattr(networks, mut_method)()
        else:
            if isinstance(networks, list):
                mut_dict = []
                for i, net in enumerate(networks):
                    mut_dict.append(getattr(net, mut_method)(**applied_mut_dict[i]))
            else:
                mut_dict = getattr(networks, mut_method)(**applied_mut_dict)
        
        return mut_dict

    def to_device_and_set_individual(
        self, 
        individual: EvolvableAlgorithm, 
        name: str, 
        networks: OffspringType
    ) -> None:
        """Moves networks to the device and assigns them back to the individual.
        
        :param individual: The individual to assign the networks to
        :type individual: EvolvableAlgorithm
        :param name: The name of the attribute to assign the networks to
        :type name: str
        :param networks: The networks to move to the device
        :type networks: OffspringType
        """
        if self.accelerator is None:
            setattr(individual, name, self.to_device(networks))
        else:
            setattr(individual, name, networks)

    def _reinit_bandit_grads(
            self,
            individual: BanditAlgorithm,
            offspring_actor: EvolvableModule,
            old_exp_layer: nn.Module) -> None:
        """Reinitialise bandit gradients after architecture mutation.
        
        :param individual: Individual agent from population
        :type individual: EvolvableAlgorithm
        :param offspring_actor: Offspring actor network
        :type offspring_actor: EvolvableModule
        :param old_exp_layer: Old linear layer
        :type old_exp_layer: nn.Module
        """
        if isinstance(offspring_actor, EvolvableModule):
            exp_layer = offspring_actor.get_output_dense()
        else:
            raise ValueError(f"Bandit algorithm architecture {type(offspring_actor)} not supported.")

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

        # Adjust indices to add after deletion
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
