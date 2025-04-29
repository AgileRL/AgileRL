import copy
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import fastrand
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper
from numpy.random import Generator
from torch._dynamo.eval_frame import OptimizedModule

from agilerl.algorithms.core import EvolvableAlgorithm, LLMAlgorithm
from agilerl.algorithms.core.wrappers import OptimizerWrapper
from agilerl.algorithms.neural_ts_bandit import NeuralTS
from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.modules.base import EvolvableModule
from agilerl.protocols import OptimizerConfig
from agilerl.utils.algo_utils import remove_compile_prefix

NetworkConfig = Dict[str, str]
NetworkList = List[NetworkConfig]
SelfEvolvableAlgorithm = TypeVar("T", bound=EvolvableAlgorithm)
MutationMethod = Callable[[SelfEvolvableAlgorithm], SelfEvolvableAlgorithm]
AlgoConfig = Dict[str, Union[NetworkConfig, NetworkList]]
PopulationType = Iterable[SelfEvolvableAlgorithm]
ModuleType = Union[OptimizedModule, EvolvableModule]
OffspringType = Union[List[EvolvableModule], EvolvableModule]
MutationReturnType = Union[Dict[str, Any], List[Dict[str, Any]]]
BanditAlgorithm = Union[NeuralUCB, NeuralTS]


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
    eval: OffspringType, new_layer_prob: float, rng: Generator
) -> str:
    """Get the mutation method and its return type of the individual.

    :param individual: The individual to inspect
    :type individual: EvolvableAlgorithm
    :param new_layer_prob: Relative probability of new layer mutation (type of architecture mutation)
    :type new_layer_prob: float
    :param rng: Random number generator
    :type rng: Generator
    :return: The mutation methods name
    :rtype: str
    """
    # All of the offsprings should be the same EvolvableModule type, so we can
    # just sample the mutation method from the first offspring
    if isinstance(eval, list):
        assert all(
            isinstance(offspring, eval[0].__class__) for offspring in eval
        ), "All offspring should be of the same type."

        # NOTE: For multi-agent settings we apply the same architecture mutations to
        # all agents. However, depending on use-case it might be beneficial to have agents
        # with different architectures; please raise an issue if you would like this!
        eval = eval[0]

    return eval.sample_mutation_method(new_layer_prob, rng)


def get_offspring_eval_modules(
    individual: SelfEvolvableAlgorithm,
) -> Tuple[Dict[str, OffspringType], ...]:
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

        # Clone the offspring prior to applying mutations
        offspring = (
            [mod.clone() for mod in eval_module]
            if isinstance(eval_module, list)
            else eval_module.clone()
        )

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
        raise ValueError(
            f"Bandit algorithm architecture {type(offspring)} not supported."
        )

    return exp_layer


class Mutations:
    """The Mutations class for evolutionary hyperparameter optimization.

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
        no_mutation: float,
        architecture: float,
        new_layer_prob: float,
        parameters: float,
        activation: float,
        rl_hp: float,
        mutation_sd: float = 0.1,
        activation_selection: List[str] = ["ReLU", "ELU", "GELU"],
        mutate_elite: bool = True,
        rand_seed: Optional[int] = None,
        device: str = "cpu",
        accelerator: Optional[Accelerator] = None,
    ):
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
        assert (
            mutation_sd >= 0
        ), "Mutation strength must be greater than or equal to zero."
        assert isinstance(
            mutation_sd, (float, int)
        ), "Mutation strength must be a float or integer."
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
        self.mutation_sd = mutation_sd  # Mutation strength
        self.mutate_elite = mutate_elite
        self.device = device
        self.accelerator = accelerator

        self.pretraining_mut_options, self.pretraining_mut_proba = (
            self.get_mutations_options(pretraining=True)
        )
        self.mut_options, self.mut_proba = self.get_mutations_options()

    def get_mutations_options(
        self, pretraining: bool = False
    ) -> Tuple[List[Callable], List[float]]:
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
            (self.rl_hyperparam_mutation, self.rl_hp_mut),
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

    def no_mutation(self, individual: SelfEvolvableAlgorithm):
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

    def to_device_and_set_individual(
        self, individual: SelfEvolvableAlgorithm, name: str, networks: OffspringType
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

    def reinit_module(
        self, module: EvolvableModule, init_dict: Dict[str, Any]
    ) -> EvolvableModule:
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

    def reinit_from_mutated(
        self, offspring: OffspringType, remove_compile_prefix: bool = False
    ) -> OffspringType:
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
            self.load_state_dicts(ind_shared, state_dicts, remove_compile_prefix)
        else:
            ind_shared = self.reinit_module(offspring, offspring.init_dict)
            ind_shared.load_state_dict(offspring.state_dict(), strict=False)

        return ind_shared

    def load_state_dicts(
        self,
        modules: List[ModuleType],
        state_dicts: List[Dict[str, Any]],
        remove_prefix: bool = False,
    ) -> None:
        """Load the state dictionary into the module.

        :param module: The module to load the state dictionary into
        :type module: ModuleType
        :param state_dict: The state dictionary to load
        :type state_dict: Dict[str, Any]
        :param remove_prefix: Boolean flag indicating if the compile prefix should be removed, defaults to False
        :type remove_prefix: bool, optional
        """
        for module, state_dict in zip(modules, state_dicts):
            state_dict = (
                remove_compile_prefix(state_dict) if remove_prefix else state_dict
            )
            module.load_state_dict(state_dict, strict=False)

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

    def mutation(
        self, population: PopulationType, pre_training_mut: bool = False
    ) -> PopulationType:
        """Returns mutated population.

        :param population: Population of agents
        :type population: list[EvolvableAlgorithm]
        :param pre_training_mut: Boolean flag indicating if the mutation is before the training loop
        :type pre_training_mut: bool, optional
        """
        # Create lists of possible mutation functions and their respective relative probabilities
        mutation_options = (
            self.pretraining_mut_options if pre_training_mut else self.mut_options
        )
        mutation_proba = (
            self.pretraining_mut_proba if pre_training_mut else self.mut_proba
        )

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
            individual: SelfEvolvableAlgorithm = individual
            registry = individual.registry

            # Call mutation function for each individual
            individual = mutation(individual)

            # Recompile modules if applicable
            compiled_individual = False
            if hasattr(individual, "torch_compiler") and individual.torch_compiler:
                compiled_individual = True
                individual.recompile()

            # Reinitiliaze shared networks to mutated evaluation networks
            for net_group in registry.groups:
                if net_group.shared is not None:
                    for shared_name in net_group.shared:
                        eval_offspring: OffspringType = getattr(
                            individual, net_group.eval
                        )

                        # Reinitialize shared with frozen weights due to
                        # potential mutation in architecture
                        ind_shared = self.reinit_from_mutated(
                            eval_offspring, remove_compile_prefix=compiled_individual
                        )

                        if self.accelerator is None:
                            ind_shared = self.to_device(ind_shared)

                        # Compile modules if necessary
                        if (
                            hasattr(individual, "torch_compiler")
                            and individual.torch_compiler
                        ):
                            ind_shared = self.compile_modules(
                                ind_shared, individual.torch_compiler
                            )

                        setattr(individual, shared_name, ind_shared)

            # Call hooks specified by user
            individual.mutation_hook()

            mutated_population.append(individual)

        return mutated_population

    def reinit_opt(
        self,
        individual: SelfEvolvableAlgorithm,
        optimizer: Optional[OptimizerConfig] = None,
    ) -> None:
        """Reinitialize the optimizers of an individual.

        :param individual: The individual to reinitialize the optimizers for
        :type individual: EvolvableAlgorithm
        """

        def _reinit_individual(config: OptimizerConfig) -> None:
            opt: Union[OptimizerWrapper, DeepSpeedOptimizerWrapper] = getattr(
                individual, config.name
            )
            optimizer = opt.optimizer

            # Multiple optimizers in a single attribute (i.e. multi-agent)
            # or one module optimized by a single optimizer
            if isinstance(opt, DeepSpeedOptimizerWrapper):
                LLMAlgorithm.update_lr(opt, individual.lr)
            else:
                if isinstance(optimizer, list) or len(opt.network_names) == 1:
                    opt_nets = getattr(individual, opt.network_names[0])

                # Multiple modules optimized by a single optimizer (e.g. PPO)
                else:
                    opt_nets = [getattr(individual, net) for net in opt.network_names]

                # Reinitialize optimizer with mutated nets
                offspring_opt = OptimizerWrapper(
                    optimizer_cls=config.get_optimizer_cls(),
                    networks=opt_nets,
                    lr=getattr(individual, opt.lr_name),
                    optimizer_kwargs=opt.optimizer_kwargs,
                    network_names=opt.network_names,
                    lr_name=opt.lr_name,
                    multiagent=opt.multiagent,
                )

                setattr(individual, config.name, offspring_opt)

        if optimizer is not None:
            _reinit_individual(optimizer)
        else:
            optimizer_configs = individual.registry.optimizers
            for opt_config in optimizer_configs:
                _reinit_individual(opt_config)

    def rl_hyperparam_mutation(
        self, individual: SelfEvolvableAlgorithm
    ) -> SelfEvolvableAlgorithm:
        """Returns individual from population with RL hyperparameter mutation.

        :param individual: Individual agent from population
        :type individual: object
        """
        # Randomly sample hyperparameter to mutate from the passed configuration
        hp_config = individual.registry.hp_config
        if not hp_config:
            individual.mut = "None"
            return individual

        mutate_attr, mutate_param = hp_config.sample()

        if mutate_param.value is None:
            mutate_param.value = getattr(individual, mutate_attr)

        # Randomly grow or shrink hyperparameters by specified factors
        new_value = mutate_param.mutate()

        setattr(individual, mutate_attr, new_value)

        # Need to reinitialize respective optimizer if mutated learning rate
        if mutate_attr in individual.get_lr_names():
            optimizer_configs = individual.registry.optimizers
            to_reinit = [
                opt_config
                for opt_config in optimizer_configs
                if mutate_attr == opt_config.lr
            ][0]
            self.reinit_opt(
                individual, optimizer=to_reinit
            )  # Reinitialise optimizer if new learning rate

        individual.mut = mutate_attr

        return individual

    # TODO: Activation mutations should really be integrated as architecture mutations
    def activation_mutation(
        self, individual: SelfEvolvableAlgorithm
    ) -> SelfEvolvableAlgorithm:
        """Returns individual from population with activation layer mutation.

        :param individual: Individual agent from population
        :type individual: EvolvableAlgorithm
        """
        # Needs to stay constant for policy gradient methods
        # NOTE: Could set up an algorithm registry to make algo checks more robust
        # OR perform activation mutations within evolvable modules directly and disable
        # on an algorithm basis
        if individual.algo in ["PPO", "DDPG", "TD3", "IPPO", "MADDPG", "MATD3", "GRPO"]:
            warnings.warn(
                f"Activation mutations are not supported for {individual.algo}."
            )
            individual.mut = "None"
            return individual

        # Mutate network activation layer
        registry = individual.registry
        no_activation = False
        for network_group in registry.groups:
            eval_module: OffspringType = getattr(individual, network_group.eval)
            if isinstance(eval_module, list):
                # TODO: Will need to modify when making multi-agent support more robust
                # to different type sof settings (i.e. different observation spaces and thus
                # network architectures for different agents)
                if eval_module[0].activation is None:
                    no_activation = True

                eval_module = [self._permutate_activation(mod) for mod in eval_module]
            else:
                if eval_module.activation is None:
                    no_activation = True

                eval_module = self._permutate_activation(eval_module)

            if no_activation:
                warnings.warn(
                    "Found no activation mutation capabilities. We advise setting the probability to "
                    "0.0 to disable activation mutations."
                )
                break
            if self.accelerator is None:
                eval_module = self.to_device(eval_module)

            if isinstance(individual, (NeuralTS, NeuralUCB)):
                individual.exp_layer = get_exp_layer(eval_module)

            setattr(individual, network_group.eval, eval_module)

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "act" if not no_activation else "None"
        return individual

    def _permutate_activation(self, network: EvolvableModule) -> EvolvableModule:
        """Permutate the activation layer of the network.

        :param network: The network to permutate the activation layer for
        :type network: EvolvableModule

        :return: The network with permutated activation layer
        :rtype: EvolvableModule
        """
        # Function to change network activation layer
        possible_activations = copy.deepcopy(self.activation_selection)
        current_activation = network.activation

        # Remove current activation from options to ensure different new
        # activation layer
        if len(possible_activations) > 1 and current_activation in possible_activations:
            possible_activations.remove(current_activation)

        new_activation = self.rng.choice(possible_activations, size=1)[
            0
        ]  # Select new activation
        network.change_activation(
            new_activation, output=False
        )  # Change activation layer
        return network

    def parameter_mutation(
        self, individual: SelfEvolvableAlgorithm
    ) -> SelfEvolvableAlgorithm:
        """Returns individual from population with network parameters mutation.

        :param individual: Individual agent from population
        :type individual: EvolvableAlgorithm
        """
        if isinstance(individual, LLMAlgorithm):
            warnings.warn("Parameter mutations are not supported for LLM algorithms.")
            individual.mut = "None"
            return individual

        registry = individual.registry

        # We only apply parameter mutations to the evaluation policy network
        # (i.e. the network used to select actions)
        offspring_policy: OffspringType = getattr(individual, registry.policy)
        if isinstance(offspring_policy, list):
            offspring_policy = [
                self.classic_parameter_mutation(mod) for mod in offspring_policy
            ]
        else:
            offspring_policy = self.classic_parameter_mutation(offspring_policy)

        if self.accelerator is None:
            offspring_policy = self.to_device(offspring_policy)

        setattr(individual, registry.policy, offspring_policy)

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "param"
        return individual

    def classic_parameter_mutation(self, network: EvolvableModule) -> EvolvableModule:
        """
        Returns network with mutated weights, with a vectorized inner loop for efficiency.

        :param network: Neural network to mutate.
        :type network: EvolvableModule
        :return: Mutated network.
        :rtype: EvolvableModule
        """
        # Parameters controlling mutation strength and probabilities
        mut_strength = self.mutation_sd
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05
        mag_limit = 1000000

        model_params = network.state_dict()

        # Collect keys corresponding to weight matrices (ignoring normalization params)
        potential_keys = [
            key
            for key in model_params
            if "norm" not in key and len(model_params[key].shape) == 2
        ]

        # Randomly choose a subset of keys to mutate
        how_many = int(self.rng.integers(1, len(potential_keys) + 1))
        chosen_keys = self.rng.choice(potential_keys, how_many, replace=False)

        for key in chosen_keys:
            W = model_params[key]
            num_weights = W.shape[0] * W.shape[1]
            num_mutations = int(np.ceil(num_mutation_frac * num_weights))
            if num_mutations < 1:
                continue

            # Vectorized generation of random indices (for rows and columns)
            rows = self.rng.integers(0, W.shape[0], size=num_mutations)
            cols = self.rng.integers(0, W.shape[1], size=num_mutations)
            rand_vals = self.rng.uniform(0, 1, size=num_mutations)

            # Convert indices and random values to torch tensors on the same device as W
            rows_tensor = torch.tensor(rows, dtype=torch.long, device=W.device)
            cols_tensor = torch.tensor(cols, dtype=torch.long, device=W.device)
            rand_vals_tensor = torch.tensor(rand_vals, dtype=W.dtype, device=W.device)

            # Get current weight values at the selected indices
            current_vals: torch.Tensor = W[rows_tensor, cols_tensor]
            new_vals = current_vals.clone()

            # Create masks for the different mutation types
            mask_super = rand_vals_tensor < super_mut_prob
            mask_reset = (rand_vals_tensor >= super_mut_prob) & (
                rand_vals_tensor < reset_prob
            )
            mask_normal = rand_vals_tensor >= reset_prob

            # Super mutation: add noise with std proportional to the absolute current value times super_mut_strength
            if mask_super.sum() > 0:
                std_super = (super_mut_strength * current_vals[mask_super]).abs()
                noise_super = torch.normal(
                    mean=torch.zeros_like(std_super), std=std_super
                )
                new_vals[mask_super] = current_vals[mask_super] + noise_super

            # Reset mutation: completely reset the weight using N(0, 1)
            if mask_reset.sum() > 0:
                noise_reset = torch.normal(
                    mean=torch.zeros(mask_reset.sum(), device=W.device),
                    std=torch.ones(mask_reset.sum(), device=W.device),
                )
                new_vals[mask_reset] = noise_reset

            # Normal mutation: add noise with std proportional to the absolute current value times mut_strength
            if mask_normal.sum() > 0:
                std_normal = (mut_strength * current_vals[mask_normal]).abs()
                noise_normal = torch.normal(
                    mean=torch.zeros_like(std_normal), std=std_normal
                )
                new_vals[mask_normal] = current_vals[mask_normal] + noise_normal

            # Integrate regularization by clamping all mutated values at once.
            # This is equivalent to your regularize_weight function.
            new_vals = new_vals.clamp(min=-mag_limit, max=mag_limit)

            # Write the mutated, clamped values back to the weight tensor
            W[rows_tensor, cols_tensor] = new_vals
            if self.accelerator is None:
                network = network.to(self.device)

        return network

    def architecture_mutate(
        self, individual: SelfEvolvableAlgorithm
    ) -> SelfEvolvableAlgorithm:
        """Returns individual from population with network architecture mutation, which
        adds either layers or nodes to different types of network architectures.

        :param individual: Individual agent from population
        :type individual: object
        """

        if isinstance(individual, LLMAlgorithm):
            warnings.warn(
                "Architecture mutations are not supported for LLM algorithms."
            )
            individual.mut = "None"
            return individual

        # Get the offspring evaluation modules
        # We first extract and apply a mutation for the algo policy and then apply
        # the same mutation to the rest of the evaluation modules e.g. critics
        policy, offspring_evals = get_offspring_eval_modules(individual)
        policy_name, policy_offspring = list(policy.items())[0]

        sample_policy = (
            policy_offspring[0]
            if isinstance(policy_offspring, list)
            else policy_offspring
        )
        if not sample_policy.mutation_methods:
            warnings.warn(
                "No mutation methods found for the policy network. Skipping architecture mutation. "
                "We advise setting the probability of architecture mutations to zero."
            )
            individual.mut = "None"
            return individual

        # Sample mutation method from policy network
        mut_method = get_architecture_mut_method(
            policy_offspring, self.new_layer_prob, self.rng
        )

        applied_mutations, mut_dict = self._apply_arch_mutation(
            policy_offspring, mut_method
        )
        self.to_device_and_set_individual(individual, policy_name, policy_offspring)

        if isinstance(individual, (NeuralTS, NeuralUCB)):
            old_exp_layer = get_exp_layer(policy_offspring)
            self._reinit_bandit_grads(individual, policy_offspring, old_exp_layer)

        # Apply the same mutation to the rest of the evaluation modules
        for name, offsprings in offspring_evals.items():
            self._apply_arch_mutation(offsprings, applied_mutations, mut_dict)
            self.to_device_and_set_individual(individual, name, offsprings)

            # Reinitialize bandit gradients after architecture mutation
            if isinstance(individual, (NeuralTS, NeuralUCB)):
                old_exp_layer = get_exp_layer(offsprings)
                self._reinit_bandit_grads(individual, offsprings, old_exp_layer)

        individual.mutation_hook()  # Apply mutation hook

        self.reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = (
            applied_mutations[0]
            if isinstance(applied_mutations, list)
            else applied_mutations
        )
        return individual

    def _apply_arch_mutation(
        self,
        networks: OffspringType,
        mut_method: Union[Optional[str], List[Optional[str]]],
        applied_mut_dict: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Tuple[Union[str, List[str]], MutationReturnType]:
        """Applies the mutation method to networks and returns mutation data if needed.

        :param networks: The networks to apply the mutation to
        :type networks: OffspringType
        :param mut_method: The mutation method to apply
        :type mut_method: str
        :param ret_type: The return type of the mutation method
        :type ret_type: Type
        :param mut_dict: The mutation dictionary, defaults to None
        :type mut_dict: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], optional

        :return: The mutation method name and the mutation dictionary
        :rtype: Tuple[Union[str, List[str]], MutationReturnType]
        """
        if applied_mut_dict is None:
            applied_mut_dict = (
                [{}] * len(networks) if isinstance(networks, list) else {}
            )

        mut_dict = None
        if isinstance(networks, list):
            if isinstance(mut_method, str) or mut_method is None:
                mut_method = [mut_method] * len(networks)

            mut_dict = []
            applied_muts = []
            for i, net in enumerate(networks):
                _to_apply = mut_method[i]
                if _to_apply is None:
                    mut_dict.append({})
                    applied_muts.append(None)
                    net.last_mutation_attr = None
                    net.last_mutation = None
                    continue

                mut_return = getattr(net, mut_method[i])(**applied_mut_dict[i])
                mut_dict.append(mut_return if mut_return is not None else {})
                applied_muts.append(net.last_mutation_attr)
        else:
            if mut_method is None:
                mut_dict = {}
                networks.last_mutation_attr = None
                networks.last_mutation = None
            else:
                mut_dict = getattr(networks, mut_method)(**applied_mut_dict)

            mut_dict = mut_dict if mut_dict is not None else {}
            applied_muts = networks.last_mutation_attr

        return applied_muts, mut_dict

    # TODO: This can be implemented as a mutation hook for the bandit algorithms
    def _reinit_bandit_grads(
        self,
        individual: BanditAlgorithm,
        offspring_actor: EvolvableModule,
        old_exp_layer: nn.Module,
    ) -> None:
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
            raise ValueError(
                f"Bandit algorithm architecture {type(offspring_actor)} not supported."
            )

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
