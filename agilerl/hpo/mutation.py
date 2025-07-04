import copy
import logging
import warnings
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import fastrand
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils.deepspeed import DeepSpeedOptimizerWrapper

from agilerl.algorithms.core import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    OptimizerWrapper,
    RLAlgorithm,
)
from agilerl.algorithms.neural_ts_bandit import NeuralTS
from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.protocols import OptimizerConfig
from agilerl.typing import (
    EvolvableNetworkType,
    MutationMethod,
    MutationReturnType,
)
from agilerl.utils.algo_utils import remove_compile_prefix
from agilerl.utils.evolvable_networks import compile_model
from agilerl.utils.llm_utils import _DummyOptimizer

IndividualType = TypeVar("IndividualType", bound=EvolvableAlgorithm)
MutationsType = TypeVar("MutationsType", bound="Mutations")
PopulationType = List[IndividualType]
BanditAlgorithm = Union[NeuralUCB, NeuralTS]

torch._dynamo.config.cache_size_limit = 64
torch._logging.set_logs(dynamo=logging.FATAL)


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


def get_offspring_eval_modules(
    individual: IndividualType,
) -> Tuple[Dict[str, EvolvableNetworkType], Dict[str, EvolvableNetworkType]]:
    """Get the offsprings of all of the evaluation modules in the individual.

    :param individual: The individual to inspect
    :type individual: EvolvableAlgorithm

    :return: Tuple of offspring policy and the rest of the evaluation modules
    :rtype: Tuple[Dict[str, NetworkType], Dict[str, NetworkType]]
    """
    registry = individual.registry

    offspring_modules = {}
    offspring_policy = {}
    for group in registry.groups:
        eval_module: EvolvableNetworkType = getattr(individual, group.eval_network)

        # Clone the offspring prior to applying mutations
        offspring = eval_module.clone()
        if group.policy:
            offspring_policy[group.eval_network] = offspring
        else:
            offspring_modules[group.eval_network] = offspring

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


def reinit_shared_networks(mutation_func=None):
    """Decorator to reinitialize shared networks after architecture and parameter mutations.

    :param mutation_func: The mutation function to decorate
    :type mutation_func: Optional[Callable[[IndividualType], IndividualType]]
    :return: The decorated mutation function or decorator
    :rtype: Callable
    """

    def decorator(func: MutationMethod) -> Callable:
        @wraps(func)
        def wrapper(self: MutationsType, individual: IndividualType) -> IndividualType:
            # Call the original mutation function
            individual = func(self, individual)

            torch._dynamo.reset()  # NOTE: Should we do this?

            # Only proceed if mutation was actually applied
            if individual.mut == "None":
                return individual

            # Recompile individual if necessary
            compiled_model = individual.torch_compiler is not None
            if compiled_model:
                # Set dynamo config before recompilation to avoid guard failures
                torch._dynamo.config.force_parameter_static_shapes = False
                individual.recompile()

            # Reinitialize shared networks to mutated evaluation networks
            for net_group in individual.registry.groups:
                if net_group.shared_networks is not None:
                    for shared_name in net_group.shared_networks:
                        eval_offspring: EvolvableNetworkType = getattr(
                            individual, net_group.eval_network
                        )
                        # Reinitialize shared with frozen weights due to
                        # potential mutation in architecture
                        ind_shared = self._reinit_from_mutated(
                            eval_offspring,
                            remove_prefix=compiled_model,
                        )
                        if self.accelerator is None:
                            ind_shared = ind_shared.to(self.device)

                        if compiled_model:
                            torch._dynamo.config.force_parameter_static_shapes = False
                            ind_shared = compile_model(
                                ind_shared, individual.torch_compiler
                            )

                        setattr(individual, shared_name, ind_shared)

            return individual

        return wrapper

    return decorator(mutation_func)


class Mutations:
    """Allows performing mutations on a population of :class:`EvolvableAlgorithm <agilerl.algorithms.core.EvolvableAlgorithm>` agents. Calling
    :func:`Mutations.mutation() <agilerl.hpo.mutation.Mutations.mutation>` on a population of agents will return a mutated population of agents.
    The type of mutation applied to each agent is sampled randomly from the probabilities given by the user. The supported types of mutations that
    can be applied to an agent are:

    * No mutation
    * Network architecture mutation - adding layers or nodes. Trained weights are reused and new weights are initialized randomly.
    * Network parameters mutation - mutating weights with Gaussian noise.
    * Network activation layer mutation - change of activation layer.
    * RL algorithm mutation - mutation of learning hyperparameter, (e.g. learning rate or batch size).

    See :ref:`evo_hyperparam_opt` for more details.

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
            self._get_mutations_options(pretraining=True)
        )
        self.mut_options, self.mut_proba = self._get_mutations_options()

    def mutation(
        self, population: PopulationType, pre_training_mut: bool = False
    ) -> PopulationType:
        """Returns a mutated population of agents. See :ref:`evo_hyperparam_opt` for more details.

        :param population: Population of agents
        :type population: list[EvolvableAlgorithm]
        :param pre_training_mut: Boolean flag indicating if the mutation is before the training loop
        :type pre_training_mut: bool, optional

        :return: Mutated population
        :rtype: list[EvolvableAlgorithm]
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
            individual = mutation(individual)  # Call sampled mutation for individual
            individual.mutation_hook()  # Call hooks specified by user

            mutated_population.append(individual)

        return mutated_population

    def no_mutation(self, individual: IndividualType):
        """Returns individual from population without mutation.

        :param individual: Individual agent from population
        :type individual:
        """
        individual.mut = "None"  # No mutation
        return individual

    @reinit_shared_networks
    def architecture_mutate(self, individual: IndividualType) -> IndividualType:
        """Performs a random mutation to the architecture of the policy network of an agent. The way in
        which we apply an architecture mutation to single and multi-agent RL algorithms inherently differs
        given the nested nature of the networks in the latter.

        * **Single-agent:** A mutation method is sampled from the policy network and then applied to the rest of the evaluation
        modules (e.g. critics). This can be done generally because all of the networks in a single-agent algorithm share the same
        architecture (given there is only one observation space).

        * **Multi-agent:** A sub-agent is sampled to perform the mutation on for the policy. We then iterate over the rest of the
        sub-agent policies and perform the same mutation if they share the same observation space. For the rest of the evaluation
        networks (e.g. critics) there is a possibility they are centralized, in which case their underlying architecture
        will differ from the policy and therefore the mutation methods won't exactly match. In such cases, we try to find an analogous
        mutation method to apply.

        .. note::
            This is currently not supported for :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param individual: Individual agent from population
        :type individual: RLAlgorithm or MultiAgentRLAlgorithm

        :return: Individual from population with network architecture mutation
        :rtype: RLAlgorithm or MultiAgentRLAlgorithm
        """
        if isinstance(individual, RLAlgorithm):
            individual = self._architecture_mutate_single(individual)
        elif isinstance(individual, MultiAgentRLAlgorithm):
            individual = self._architecture_mutate_multi(individual)
        else:
            raise MutationError(
                f"Architecture mutations are not supported for {individual.__class__.__name__}. "
                "Please make sure your algorithm inherits from 'RLAlgorithm' or 'MultiAgentRLAlgorithm'."
            )

        return individual

    def rl_hyperparam_mutation(self, individual: IndividualType) -> IndividualType:
        """Performs a random mutation of a learning hyperparameter of an agent. To do this, sample a hyperparameter from those
        specified through the :class:`HyperparameterConfig <agilerl.algorithms.core.registry.HyperparameterConfig>`
        passed during initialization of the agent. The hyperparameter is then mutated and the optimizer is reinitialized if the
        learning rate has been mutated.

        :param individual: Individual agent from population
        :type individual: EvolvableAlgorithm

        :return: Individual from population with RL hyperparameter mutation
        :rtype: EvolvableAlgorithm
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
            self._reinit_opt(
                individual, optimizer=to_reinit
            )  # Reinitialise optimizer if new learning rate

        individual.mut = mutate_attr
        return individual

    # TODO: Activation mutations should really be integrated as architecture mutations
    @reinit_shared_networks
    def activation_mutation(self, individual: IndividualType) -> IndividualType:
        """Performs a random mutation of the activation layer of the evaluation networks of an agent.

        .. note::
            This is currently not supported for :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param individual: Individual agent from population
        :type individual: RLAlgorithm or MultiAgentRLAlgorithm

        :return: Individual from population with activation layer mutation
        :rtype: RLAlgorithm or MultiAgentRLAlgorithm
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
            eval_module: EvolvableNetworkType = getattr(
                individual, network_group.eval_network
            )

            if eval_module.activation is None:
                no_activation = True
            else:
                eval_module = self._permutate_activation(eval_module)

            if no_activation:
                warnings.warn(
                    "Found no activation mutation capabilities. We advise setting the probability to "
                    "0.0 to disable activation mutations."
                )
                break

            if self.accelerator is None:
                eval_module = eval_module.to(self.device)

            if isinstance(individual, (NeuralTS, NeuralUCB)):
                individual.exp_layer = get_exp_layer(eval_module)

            setattr(individual, network_group.eval_network, eval_module)

        self._reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "act" if not no_activation else "None"
        return individual

    def parameter_mutation(self, individual: IndividualType) -> IndividualType:
        """Performs a random mutation to the weights of the policy network of an agent through
        the addition of Gaussian noise.

        .. note::
            This is currently not supported for :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param individual: Individual agent from population
        :type individual: RLAlgorithm or MultiAgentRLAlgorithm

        :return: Individual from population with network parameters mutation
        :rtype: RLAlgorithm or MultiAgentRLAlgorithm
        """
        if isinstance(individual, LLMAlgorithm):
            warnings.warn("Parameter mutations are not supported for LLM algorithms.")
            individual.mut = "None"
            return individual

        registry = individual.registry

        # We only apply parameter mutations to the evaluation policy network
        # (i.e. the network used to select actions)
        policy_group = registry.policy(return_group=True)
        offspring_policy: EvolvableNetworkType = getattr(
            individual, policy_group.eval_network
        )
        if isinstance(offspring_policy, ModuleDict):
            for agent_id, module in offspring_policy.items():
                offspring_policy[agent_id] = self._gaussian_parameter_mutation(module)
        else:
            offspring_policy = self._gaussian_parameter_mutation(offspring_policy)

        self._to_device_and_set_individual(
            individual, policy_group.eval_network, offspring_policy
        )

        # Load state dicts for shared networks
        if policy_group.shared_networks is not None:
            for shared in policy_group.shared_networks:
                offspring_shared: EvolvableNetworkType = getattr(individual, shared)
                offspring_shared.load_state_dict(
                    offspring_policy.state_dict(), strict=False
                )
                self._to_device_and_set_individual(individual, shared, offspring_shared)

        self._reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = "param"
        return individual

    def _get_mutations_options(
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

    def _reinit_opt(
        self,
        individual: IndividualType,
        optimizer: Optional[OptimizerConfig] = None,
    ) -> None:
        """Reinitialize the optimizers of an individual.

        :param individual: The individual to reinitialize the optimizers for
        :type individual: EvolvableAlgorithm
        :param optimizer: The optimizer to reinitialize, defaults to None, in which case
            all optimizers are reinitialized.
        :type optimizer: Optional[OptimizerConfig], optional
        """

        def _reinit_individual(config: OptimizerConfig) -> None:
            opt: Optional[Union[OptimizerWrapper, DeepSpeedOptimizerWrapper]] = getattr(
                individual, config.name
            )
            optimizer = opt.optimizer if hasattr(opt, "optimizer") else None

            # Multiple optimizers in a single attribute (i.e. multi-agent)
            # or one module optimized by a single optimizer
            if isinstance(opt, DeepSpeedOptimizerWrapper):
                if isinstance(opt.optimizer, _DummyOptimizer):
                    opt = getattr(
                        getattr(individual, "actor"), "optimizer"
                    )  # If the optimizer is defined in the deepspeed config, we do this
                individual.accelerator, individual.lr_scheduler = (
                    LLMAlgorithm.update_lr(
                        opt,
                        individual.lr,
                        individual.accelerator,
                        individual.cosine_lr_schedule_config,
                    )
                )
            else:
                if isinstance(optimizer, dict) or len(opt.network_names) == 1:
                    opt_nets = getattr(individual, opt.network_names[0])

                # Multiple modules optimized by a single optimizer (e.g. PPO)
                else:
                    opt_nets = [getattr(individual, net) for net in opt.network_names]

                # Reinitialize optimizer with mutated nets
                # NOTE: We need to do this since there is a chance the network parameters have changed
                # due to architecture mutations
                offspring_opt = OptimizerWrapper(
                    optimizer_cls=config.get_optimizer_cls(),
                    networks=opt_nets,
                    lr=getattr(individual, opt.lr_name),
                    optimizer_kwargs=opt.optimizer_kwargs,
                    network_names=opt.network_names,
                    lr_name=opt.lr_name,
                )

                setattr(individual, config.name, offspring_opt)

        if optimizer is not None:
            _reinit_individual(optimizer)
        else:
            optimizer_configs = individual.registry.optimizers
            for opt_config in optimizer_configs:
                _reinit_individual(opt_config)

    def _to_device_and_set_individual(
        self, individual: IndividualType, name: str, networks: EvolvableNetworkType
    ) -> None:
        """Moves networks to the device and assigns them back to the individual.

        :param individual: The individual to assign the networks to
        :type individual: EvolvableAlgorithm
        :param name: The name of the attribute to assign the networks to
        :type name: str
        :param networks: The networks to move to the device
        :type networks: EvolvableNetworkType
        """
        if self.accelerator is None:
            networks = networks.to(self.device)

        setattr(individual, name, networks)

    def _reinit_module(
        self, module: EvolvableModule, init_dict: Dict[str, Any]
    ) -> EvolvableModule:
        """Reinitialize the module with the given initialization dictionary.

        :param module: The module to reinitialize
        :type module: EvolvableModule
        :param init_dict: The initialization dictionary
        :type init_dict: Dict[str, Any]

        :return: The reinitialized module
        :rtype: EvolvableModule
        """
        module_orig = (
            module._orig_mod
            if isinstance(module, torch._dynamo.eval_frame.OptimizedModule)
            else module
        )
        return type(module_orig)(**init_dict)

    def _reinit_from_mutated(
        self, offspring: EvolvableNetworkType, remove_prefix: bool = False
    ) -> EvolvableNetworkType:
        """Reinitialize the mutated offspring with their state dictionary.

        :param offspring: The offspring to reinitialize
        :type offspring: NetworkType
        :param remove_prefix: Whether to remove the prefix from the offspring
        :type remove_prefix: bool

        :return: The reinitialized offspring
        :rtype: EvolvableNetworkType
        """
        if isinstance(offspring, ModuleDict):
            reinit_modules: Dict[str, EvolvableModule] = OrderedDict()
            for agent_id in offspring:
                nested_offspring: EvolvableModule = offspring[agent_id]
                reinit_modules[agent_id] = self._reinit_module(
                    nested_offspring, nested_offspring.init_dict
                )

            state_dicts = {
                agent_id: nested_offspring.state_dict()
                for agent_id, nested_offspring in offspring.items()
            }
            self._load_state_dicts(reinit_modules, state_dicts, remove_prefix)

            ind_shared = ModuleDict(reinit_modules)
        else:
            ind_shared = self._reinit_module(offspring, offspring.init_dict)
            ind_shared.load_state_dict(offspring.state_dict(), strict=False)

        return ind_shared

    def _load_state_dicts(
        self,
        modules: ModuleDict[EvolvableModule],
        state_dicts: Dict[str, Dict[str, Any]],
        remove_prefix: bool = False,
    ) -> None:
        """Load the state dictionaries for a multi-agent ModuleDict.

        :param modules: The modules to load the state dictionary into
        :type modules: ModuleDict[EvolvableModule]
        :param state_dicts: The state dictionary to load
        :type state_dicts: Dict[str, Dict[str, Any]]
        :param remove_prefix: Whether to remove the prefix from the state dictionary
        :type remove_prefix: bool
        """
        for agent_id, module in modules.items():
            state_dict = (
                remove_compile_prefix(state_dicts[agent_id])
                if remove_prefix
                else state_dicts[agent_id]
            )
            module.load_state_dict(state_dict, strict=False)

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

        # Remove current activation from options to ensure different new activation layer
        if len(possible_activations) > 1 and current_activation in possible_activations:
            possible_activations.remove(current_activation)

        # Select new activation and modify network
        new_activation = self.rng.choice(possible_activations, size=1)[0]
        network.change_activation(new_activation, output=False)

        return network

    def _gaussian_parameter_mutation(self, network: EvolvableModule) -> EvolvableModule:
        """
        Returns network with mutated weights using a Gaussian distribution.

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

        model_params: Dict[str, torch.Tensor] = network.state_dict()

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
            W: torch.Tensor = model_params[key]
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

    def _architecture_mutate_single(self, individual: RLAlgorithm) -> RLAlgorithm:
        """
        Apply an architecture mutation to a single-agent RL algorithm. Since all of the
        networks in a single-agent algorithm share the same architecture (given there is
        only one observation space), we first sample a mutation method from the policy network
        and then apply the same mutation to the rest of the evaluation modules (e.g. critics).
        This is preferred since it reduces variance attributed to evolutionary HPO during training
        and different evaluation networks usually solve tasks of similar complexity and should
        therefore share a similar architecture.

        :param individual: Individual agent from population
        :type individual: RLAlgorithm

        :return: Individual from population with network architecture mutation
        :rtype: RLAlgorithm
        """
        # Get the offspring evaluation modules
        # We first extract and apply a mutation to the policy and then apply
        # the same mutation to the rest of the evaluation modules e.g. critics
        policy, offspring_evals = get_offspring_eval_modules(individual)
        policy_name, policy_offspring = list(policy.items())[0]

        if not policy_offspring.mutation_methods:
            warnings.warn(
                "No mutation methods found for the policy network. Skipping architecture mutation. "
                "We advise setting the probability of architecture mutations to zero when using non-evolvable networks."
            )
            individual.mut = "None"
            return individual

        # Sample mutation method from policy network
        mut_method = policy_offspring.sample_mutation_method(
            self.new_layer_prob, self.rng
        )

        applied_mutation, mut_dict = self._apply_arch_mutation(
            policy_offspring, mut_method
        )
        self._to_device_and_set_individual(individual, policy_name, policy_offspring)

        if isinstance(individual, (NeuralTS, NeuralUCB)):
            old_exp_layer = get_exp_layer(policy_offspring)
            self._reinit_bandit_grads(individual, policy_offspring, old_exp_layer)

        # Apply the same mutation to the rest of the evaluation modules
        for name, offspring in offspring_evals.items():
            if applied_mutation in offspring.mutation_methods:
                self._apply_arch_mutation(offspring, applied_mutation, mut_dict)
                self._to_device_and_set_individual(individual, name, offspring)

        individual.mutation_hook()  # Apply mutation hook

        self._reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = applied_mutation or "None"
        return individual

    def _architecture_mutate_multi(
        self, individual: MultiAgentRLAlgorithm
    ) -> MultiAgentRLAlgorithm:
        """
        Apply an architecture mutation to a multi-agent RL algorithm. Since each agent has its own
        observation space, we can't generally apply the same architecture mutation to all sub-agents.
        Instead, we sample a sub-agent to perform the mutation on for the policy. We then iterate over
        the rest of the sub-agent policies and perform the same mutation if they share the same observation
        space. For the rest of the evaluation networks (e.g. critics) there is a possibility they are
        shared by multiple agents, in which case their underlying architecture will differ from the policy and
        therefore the mutation methods won't exactly match. In this case, we try to find an analogous
        mutation method to apply.

        .. note::
            Since we use `agilerl.modules.ModuleDict` to store multi-agent networks, the available mutation
            methods will have the form ``<agent_id>.<mutation_method>``.

        :param individual: Individual agent from population
        :type individual: MultiAgentRLAlgorithm

        :return: Individual from population with network architecture mutation
        :rtype: MultiAgentRLAlgorithm
        """
        # Get the offspring evaluation modules
        # We first extract and apply a mutation to the policy and then apply
        # the same mutation to the rest of the evaluation modules e.g. critics
        policy, offspring_evals = get_offspring_eval_modules(individual)
        policy_name, policy_offspring = list(policy.items())[0]

        if not policy_offspring.mutation_methods:
            warnings.warn(
                "No mutation methods found for the policy network. Skipping architecture mutation. "
                "We advise setting the probability of architecture mutations to zero when using non-evolvable networks."
            )
            individual.mut = "None"
            return individual

        # Sample mutation method from policy network
        mut_method = policy_offspring.sample_mutation_method(
            self.new_layer_prob, self.rng
        )

        # Apply the sampled method to the policy network (will only apply to one sub-agent)
        applied_mutation, mut_dict = self._apply_arch_mutation(
            policy_offspring, mut_method
        )

        applied_mutations = []
        if applied_mutation is not None:
            split_mutation = applied_mutation.split(".")
            sampled_agent_id = split_mutation[0]
            sampled_mutation = ".".join(split_mutation[1:])
            applied_mutations.append(sampled_agent_id)
        else:
            sampled_agent_id = mut_method.split(".")[0]
            sampled_mutation = None

        # Apply the sampled method to the sub-agents that share the same observation space
        for agent_id, policy in policy_offspring.items():
            if agent_id == sampled_agent_id:
                continue

            # Apply the sampled mutation only if it is available for the current sub-agent
            applied_agent = None
            if sampled_mutation in policy.mutation_methods:
                applied_agent, _ = self._apply_arch_mutation(
                    policy, sampled_mutation, mut_dict
                )

            if applied_agent is not None:
                applied_mutations.append(agent_id)

        self._to_device_and_set_individual(individual, policy_name, policy_offspring)

        # Try to apply an analogous mutation to the rest of the evaluation modules
        for name, offspring_eval in offspring_evals.items():
            # Iterate over the agents in the offspring evaluation module
            for agent_id, agent_eval in offspring_eval.items():
                # Iterate over the the agents whose policies were mutated
                analogous_method = False
                for mutated_agent in applied_mutations:
                    # Don't want to reapply the same method redundantly
                    if (
                        analogous_method
                        and agent_eval.last_mutation_attr == analogous_method
                    ):
                        continue

                    available_methods = agent_eval.mutation_methods

                    # Try to find an analogous mutation method
                    analogous_method = self._find_analogous_mutation(
                        sampled_mutation, available_methods, mutated_agent
                    )

                    if analogous_method is not None:
                        self._apply_arch_mutation(
                            agent_eval, analogous_method, mut_dict
                        )
                    else:
                        raise MutationError(
                            f"Mutation method '{sampled_mutation}' not found in '{agent_eval.__class__.__name__}'. "
                            f"No analogous method found for agent '{agent_id}'. "
                            f"Available methods: {agent_eval.mutation_methods}."
                        )

            self._to_device_and_set_individual(individual, name, offspring_eval)

        individual.mutation_hook()  # Apply mutation hook

        self._reinit_opt(individual)  # Reinitialise optimizer
        individual.mut = sampled_mutation or "None"
        return individual

    def _apply_arch_mutation(
        self,
        network: EvolvableNetworkType,
        mut_method: Optional[str],
        applied_mut_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], MutationReturnType]:
        """Applies the mutation method to networks and returns mutation data if needed.

        :param networks: The networks to apply the mutation to
        :type networks: EvolvableNetworkType
        :param mut_method: The mutation method to apply
        :type mut_method: Optional[str]
        :param applied_mut_dict: The mutation dictionary, defaults to None
        :type applied_mut_dict: Optional[Dict[str, Any]], optional

        :return: The mutation method name and the mutation dictionary
        :rtype: Tuple[Optional[str], MutationReturnType]
        """
        if not isinstance(network, EvolvableModule):
            raise MutationError(
                f"Can't apply architecture mutation to {network.__class__.__name__} network."
                "Please make sure your network inherits from 'EvolvableModule'."
            )

        applied_mut_dict = applied_mut_dict or {}
        mut_dict = None
        if mut_method is None:
            mut_dict = {}
            network.last_mutation_attr = None
            network.last_mutation = None
        else:
            if mut_method not in network.mutation_methods:
                raise MutationError(
                    f"Mutation method '{mut_method}' not found in '{network.__class__.__name__}'; "
                    f"available methods: \n {network.mutation_methods}."
                )

            mut_dict = getattr(network, mut_method)(**applied_mut_dict)

        mut_dict = mut_dict or {}
        applied_mut = network.last_mutation_attr

        return applied_mut, mut_dict

    # TODO: Can this be implemented as a mutation hook for the bandit algorithms?
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

    def _find_analogous_mutation(
        self, sampled_mutation: str, available_methods: List[str], policy_agent: str
    ) -> Optional[str]:
        """Find an analogous mutation method when exact match is not found.

        Tries to match based on bottom-level method and agent ID.

        :param sampled_mutation: The mutation method that was sampled (e.g., 'encoder.add_channel')
        :type sampled_mutation: str
        :param available_methods: List of available mutation methods
        :type available_methods: List[str]
        :param policy_agent: The agent ID to match (e.g., 'agent_0')
        :type policy_agent: str

        :return: Analogous mutation method if found, None otherwise
        :rtype: Optional[str]
        """
        if not sampled_mutation:
            return None

        elif sampled_mutation in available_methods:
            return sampled_mutation

        sampled_parts = sampled_mutation.split(".")
        bottom_level_method = sampled_parts[-1]

        # Look for methods that:
        # 1. End with the same bottom-level method
        # 2. Contain the policy_agent or 'vector_mlp' as one of the parts
        for method in available_methods:
            method_parts = method.split(".")

            # Check if bottom-level method matches
            if method_parts[-1] == bottom_level_method:
                if policy_agent in method_parts or "vector_mlp" in method_parts:
                    return method

        return None


class MutationError(Exception):
    """Custom exception for mutation errors."""

    pass
