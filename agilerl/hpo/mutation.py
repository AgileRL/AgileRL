import copy
import logging
import warnings
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import fastrand
import numpy as np
import torch
from accelerate import Accelerator
from torch import nn

from agilerl.algorithms import NeuralTS, NeuralUCB
from agilerl.algorithms.core import (
    EvolvableAlgorithm,
    LLMAlgorithm,
    MultiAgentRLAlgorithm,
    RLAlgorithm,
)
from agilerl.hpo import function_preserving as fp
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.typing import (
    EvolvableNetworkType,
    MutationMethod,
    MutationReturnType,
)
from agilerl.utils.algo_utils import remove_compile_prefix
from agilerl.utils.evolvable_networks import compile_model
from agilerl.wrappers.agent import AgentWrapper

IndividualType = TypeVar("IndividualType", bound=EvolvableAlgorithm)
MutationsType = TypeVar("MutationsType", bound="Mutations")
PopulationType = list[IndividualType]
BanditAlgorithm = NeuralUCB | NeuralTS

torch._dynamo.config.cache_size_limit = 64
torch._logging.set_logs(dynamo=logging.FATAL)

logger = logging.getLogger(__name__)


def set_global_seed(seed: int | None) -> None:
    """Set the global seed for random number generators.

    :param seed: Random seed for repeatability
    :type seed: int
    """
    if seed is None:
        return

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    fastrand.pcg32_seed(seed)


def get_offspring_eval_modules(
    individual: IndividualType,
) -> tuple[dict[str, EvolvableNetworkType], dict[str, EvolvableNetworkType]]:
    """Get the offsprings of all of the evaluation modules in the individual.

    :param individual: The individual to inspect
    :type individual: EvolvableAlgorithm

    :return: Tuple of offspring policy and the rest of the evaluation modules
    :rtype: tuple[dict[str, NetworkType], dict[str, NetworkType]]
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
        msg = f"Bandit algorithm architecture {type(offspring)} not supported."
        raise TypeError(msg)

    return exp_layer


def reinit_shared_networks(
    mutation_func: Any = None,
) -> Callable[..., Any]:
    """Reinitialize shared networks after architecture and parameter mutations (decorator).

    :param mutation_func: The mutation function to decorate
    :type mutation_func: Callable[[IndividualType], IndividualType] | None
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
                            individual,
                            net_group.eval_network,
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
                                ind_shared,
                                individual.torch_compiler,
                            )

                        setattr(individual, shared_name, ind_shared)

            return individual

        return wrapper

    return decorator(mutation_func)


class Mutations:
    """Allow performing mutations on a population of :class:`EvolvableAlgorithm <agilerl.algorithms.core.EvolvableAlgorithm>` agents.
    Calling :func:`Mutations.mutation() <agilerl.hpo.mutation.Mutations.mutation>` on a population of agents will return a mutated population of agents.
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
    :param arch_mut_type: Architecture-mutation strategy, ``"original"`` or
        ``"func_preserving"`` (Net2Net-style), defaults to ``"original"``
    :type arch_mut_type: str, optional
    :param arch_fp_noise: Symmetry-breaking noise scale for function-preserving
        additions, relative to the consuming layer's existing outgoing-weight std;
        ``0.0`` keeps the exact-zero fan-out, defaults to 0.1
    :type arch_fp_noise: float, optional
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
        activation_selection: list[str] | None = None,
        mutate_elite: bool = True,
        rand_seed: int | None = None,
        device: str = "cpu",
        accelerator: Accelerator | None = None,
        arch_mut_type: str = "original",
        arch_fp_noise: float = 0.1,
    ) -> None:
        if activation_selection is None:
            activation_selection = ["ReLU", "ELU", "GELU"]
        assert isinstance(
            no_mutation,
            (float, int),
        ), "Probability of no mutation must be a float or integer."
        assert no_mutation >= 0, (
            "Probability of no mutation must be greater than or equal to zero."
        )
        assert isinstance(
            architecture,
            (float, int),
        ), "Probability of architecture mutation must be a float or integer."
        assert architecture >= 0, (
            "Probability of architecture mutation must be greater than or equal to zero."
        )
        assert isinstance(
            new_layer_prob,
            (float, int),
        ), "Probability of new layer architecture mutation must be a float or integer."
        assert 1 >= new_layer_prob >= 0, (
            "Probability of new layer architecture mutation must be between zero and one (inclusive)."
        )
        assert isinstance(
            parameters,
            (float, int),
        ), "Probability of parameters mutation must be a float or integer."
        assert parameters >= 0, (
            "Probability of parameters mutation must be greater than or equal to zero."
        )
        assert isinstance(
            activation,
            (float, int),
        ), "Probability of activation mutation must be a float or integer."
        assert activation >= 0, (
            "Probability of activation mutation must be greater than or equal to zero."
        )
        assert isinstance(
            rl_hp,
            (float, int),
        ), (
            "Probability of reinforcement learning hyperparameter mutation must be a float or integer."
        )
        assert rl_hp >= 0, (
            "Probability of reinforcement learning hyperparameter mutation must be greater than or equal to zero."
        )
        assert mutation_sd >= 0, (
            "Mutation strength must be greater than or equal to zero."
        )
        assert isinstance(
            mutation_sd,
            (float, int),
        ), "Mutation strength must be a float or integer."
        assert isinstance(
            mutate_elite,
            bool,
        ), "Mutate elite must be boolean value True or False."
        assert isinstance(rand_seed, int) or rand_seed is None, (
            "Random seed must be an integer or None."
        )
        if isinstance(rand_seed, int):
            assert rand_seed >= 0, "Random seed must be greater than or equal to zero."
        assert arch_mut_type in ("original", "func_preserving"), (
            "arch_mut_type must be either 'original' or 'func_preserving'."
        )
        assert isinstance(arch_fp_noise, (float, int)), (
            "arch_fp_noise must be a float or integer."
        )
        assert arch_fp_noise >= 0, (
            "arch_fp_noise must be greater than or equal to zero."
        )

        # Random seed for repeatability
        set_global_seed(rand_seed)
        self.rng = np.random.default_rng(rand_seed)

        # Relative probabilities of mutation
        self.no_mut = no_mutation  # No mutation
        self.architecture_mut = architecture  # Architecture mutation
        self.new_layer_prob = (
            new_layer_prob  # New layer mutation (type of architecture mutation)
        )
        self.parameters_mut = parameters  # Network parameters mutation
        self.activation_mut = activation  # Activation layer mutation
        self.rl_hp_mut = rl_hp  # Learning HP mutation
        self.activation_selection = activation_selection  # Activation functions
        self.mutation_sd = mutation_sd  # Mutation strength
        self.mutate_elite = mutate_elite
        # Function-preserving architecture-mutation configuration (Net2Net; Chen et
        # al. / Fehring et al.). When ``arch_mut_type == "func_preserving"`` the
        # *addition* operators (add_node/add_channel/add_layer, and their latent
        # counterparts) are modified so a mutated child starts out computing the
        # parent's function. Removals are deliberately left to AgileRL's original
        # random-count positional operator, so the two regimes differ only in how
        # capacity is *added*.
        self.arch_mut_type = arch_mut_type
        # Symmetry-breaking noise scale (relative to the existing outgoing-weight
        # std) for function-preserving additions; 0.0 keeps the exact-zero fan-out.
        self.arch_fp_noise = arch_fp_noise
        # One-time warning guards (function preservation caveats / fallbacks).
        self._fp_warned_layernorm = False
        self._fp_warned_activation = False
        self._fp_warned_kernel = False
        self.device = device
        self.accelerator = accelerator

        self.pretraining_mut_options, self.pretraining_mut_proba = (
            self._get_mutations_options(pretraining=True)
        )
        self.mut_options, self.mut_proba = self._get_mutations_options()

    def mutation(
        self,
        population: PopulationType,
        pre_training_mut: bool = False,
    ) -> PopulationType:
        """Return a mutated population of agents. See :ref:`evo_hyperparam_opt` for more details.

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
        mutation_choice: list[MutationMethod] = self.rng.choice(
            mutation_options,
            len(population),
            p=mutation_proba,
        )

        # If not mutating elite member of population (first in list from tournament selection),
        # set this as the first mutation choice
        if not self.mutate_elite:
            mutation_choice[0] = self.no_mutation

        mutated_population = []
        for mutation, individual in zip(mutation_choice, population, strict=False):
            wrapped_ind = isinstance(individual, AgentWrapper)
            agent = individual.agent if wrapped_ind else individual

            agent = mutation(agent)  # Call sampled mutation for individual
            agent.mutation_hook()  # Call hooks specified by user

            if wrapped_ind:
                individual.agent = agent
            else:
                individual = agent

            mutated_population.append(individual)

        return mutated_population

    def no_mutation(self, individual: IndividualType) -> IndividualType:
        """Return individual from population without mutation.

        :param individual: Individual agent from population
        :type individual:
        """
        individual.mut = "None"  # No mutation
        individual.mut_details = {"category": "no mutation", "name": "none"}
        return individual

    @reinit_shared_networks
    def architecture_mutate(self, individual: IndividualType) -> IndividualType:
        """Perform a random mutation to the architecture of the policy network of an agent. The way in
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
            msg = (
                f"Architecture mutations are not supported for {individual.__class__.__name__}. "
                "Please make sure your algorithm inherits from 'RLAlgorithm' or 'MultiAgentRLAlgorithm'."
            )
            raise MutationError(
                msg,
            )

        return individual

    def rl_hyperparam_mutation(self, individual: IndividualType) -> IndividualType:
        """Perform a random mutation of a learning hyperparameter of an agent. To do this, sample a hyperparameter from those
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
            individual.mut_details = {"category": "no mutation", "name": "none"}
            return individual

        mutate_attr, mutate_param = hp_config.sample()

        if mutate_param.value is None:
            mutate_param.value = getattr(individual, mutate_attr)

        # Capture the hyperparameter value before mutating it
        hp_before = getattr(individual, mutate_attr)

        # Randomly grow or shrink hyperparameters by specified factors
        new_value = mutate_param.mutate()

        setattr(individual, mutate_attr, new_value)

        # Reinitialize optimizer if mutated learning rate
        if mutate_attr in individual.get_lr_names():
            optimizer_configs = individual.registry.optimizers
            to_reinit = next(
                opt_config
                for opt_config in optimizer_configs
                if mutate_attr == opt_config.lr
            )

            individual.reinit_optimizers(optimizer=to_reinit)

        individual.mut = mutate_attr
        individual.mut_details = {
            "category": "hyperparameter",
            "name": mutate_attr,
            "hp_before": hp_before,
            "hp_after": new_value,
        }
        return individual

    # TODO: Activation mutations should really be integrated as architecture mutations
    @reinit_shared_networks
    def activation_mutation(self, individual: IndividualType) -> IndividualType:
        """Perform a random mutation of the activation layer of the evaluation networks of an agent.

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
                f"Activation mutations are not supported for {individual.algo}.",
                stacklevel=2,
            )
            individual.mut = "None"
            individual.mut_details = {"category": "no mutation", "name": "none"}
            return individual

        # Mutate network activation layer
        registry = individual.registry
        no_activation = False
        act_before = None
        act_after = None
        for network_group in registry.groups:
            eval_module: EvolvableNetworkType = getattr(
                individual,
                network_group.eval_network,
            )

            if eval_module.activation is None:
                no_activation = True
            else:
                if act_before is None:
                    act_before = eval_module.activation
                eval_module = self._permutate_activation(eval_module)
                if act_after is None:
                    act_after = eval_module.activation

            if no_activation:
                warnings.warn(
                    "Found no activation mutation capabilities. We advise setting the probability to "
                    "0.0 to disable activation mutations.",
                    stacklevel=2,
                )
                break

            if self.accelerator is None:
                eval_module = eval_module.to(self.device)

            if isinstance(individual, (NeuralTS, NeuralUCB)):
                individual.exp_layer = get_exp_layer(eval_module)

            setattr(individual, network_group.eval_network, eval_module)

        individual.reinit_optimizers()  # Reinitialize optimizer
        individual.mut = "act" if not no_activation else "None"
        if no_activation:
            individual.mut_details = {"category": "no mutation", "name": "none"}
        else:
            individual.mut_details = {
                "category": "activation",
                "name": "activation",
                "act_before": act_before,
                "act_after": act_after,
            }
        return individual

    def parameter_mutation(self, individual: IndividualType) -> IndividualType:
        """Perform a random mutation to the weights of the policy network of an agent through
        the addition of Gaussian noise.

        .. note::
            This is currently not supported for :class:`LLMAlgorithm <agilerl.algorithms.core.LLMAlgorithm>` agents.

        :param individual: Individual agent from population
        :type individual: RLAlgorithm or MultiAgentRLAlgorithm

        :return: Individual from population with network parameters mutation
        :rtype: RLAlgorithm or MultiAgentRLAlgorithm
        """
        if isinstance(individual, LLMAlgorithm):
            warnings.warn(
                "Parameter mutations are not supported for LLM algorithms. Skipping mutation.",
                stacklevel=2,
            )
            individual.mut = "None"
            individual.mut_details = {"category": "no mutation", "name": "none"}
            return individual

        registry = individual.registry

        # We only apply parameter mutations to the evaluation policy network
        # (i.e. the network used to select actions)
        policy_group = registry.policy(return_group=True)
        offspring_policy: EvolvableNetworkType = getattr(
            individual,
            policy_group.eval_network,
        )
        # Accumulate per-category weight counts across all mutated networks
        counts = {"reset": 0, "ordinary": 0, "amplified": 0}
        if isinstance(offspring_policy, ModuleDict):
            for agent_id, module in offspring_policy.items():
                offspring_policy[agent_id] = self._gaussian_parameter_mutation(
                    module, counts=counts
                )
        else:
            offspring_policy = self._gaussian_parameter_mutation(
                offspring_policy, counts=counts
            )

        self._to_device_and_set_individual(
            individual,
            policy_group.eval_network,
            offspring_policy,
        )

        # Load state dicts for shared networks
        if policy_group.shared_networks is not None:
            for shared in policy_group.shared_networks:
                offspring_shared: EvolvableNetworkType = getattr(individual, shared)
                offspring_shared.load_state_dict(
                    offspring_policy.state_dict(),
                    strict=False,
                )
                self._to_device_and_set_individual(individual, shared, offspring_shared)

        individual.reinit_optimizers()  # Reinitialize optimizer
        individual.mut = "param"
        individual.mut_details = {
            "category": "parameter",
            "name": "param",
            "weights_reset": counts["reset"],
            "weights_ordinary_noise": counts["ordinary"],
            "weights_amplified_noise": counts["amplified"],
        }

        return individual

    def _get_mutations_options(
        self,
        pretraining: bool = False,
    ) -> tuple[list[Callable], list[float]]:
        """Get the mutation options and probabilities for the given mutation
        configuration.

        :param pretraining: Boolean flag indicating if the mutation is before the training loop
        :type pretraining: bool
        :return: Mutation functions and their respective relative probabilities
        :rtype: tuple[list[Callable], list[float]]
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

        mutation_funcs, mutation_proba = zip(*mutation_options, strict=False)
        mutation_proba = np.array(mutation_proba) / np.sum(mutation_proba)
        return mutation_funcs, mutation_proba

    def _to_device_and_set_individual(
        self,
        individual: IndividualType,
        name: str,
        networks: EvolvableNetworkType,
    ) -> None:
        """Move networks to the device and assigns them back to the individual.

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
        self,
        module: EvolvableModule,
        init_dict: dict[str, Any],
    ) -> EvolvableModule:
        """Reinitialize the module with the given initialization dictionary.

        :param module: The module to reinitialize
        :type module: EvolvableModule
        :param init_dict: The initialization dictionary
        :type init_dict: dict[str, Any]

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
        self,
        offspring: EvolvableNetworkType,
        remove_prefix: bool = False,
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
            reinit_modules: dict[str, EvolvableModule] = OrderedDict()
            for agent_id in offspring:
                nested_offspring: EvolvableModule = offspring[agent_id]
                reinit_modules[agent_id] = self._reinit_module(
                    nested_offspring,
                    nested_offspring.init_dict,
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
        state_dicts: dict[str, dict[str, Any]],
        remove_prefix: bool = False,
    ) -> None:
        """Load the state dictionaries for a multi-agent ModuleDict.

        :param modules: The modules to load the state dictionary into
        :type modules: ModuleDict[EvolvableModule]
        :param state_dicts: The state dictionary to load
        :type state_dicts: dict[str, dict[str, Any]]
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
        new_activation = str(self.rng.choice(possible_activations, size=1)[0])
        network.change_activation(new_activation, output=False)

        return network

    def _gaussian_parameter_mutation(
        self,
        network: EvolvableModule,
        counts: dict[str, int] | None = None,
    ) -> EvolvableModule:
        """Return network with mutated weights using a Gaussian distribution.

        :param network: Neural network to mutate.
        :type network: EvolvableModule
        :param counts: Optional dict accumulating the number of weights mutated by
            category (``"reset"``, ``"ordinary"``, ``"amplified"``). Updated in place.
        :type counts: dict[str, int] | None
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

        model_params: dict[str, torch.Tensor] = network.state_dict()

        # Collect keys corresponding to weight matrices (ignoring normalization / lstm params)
        exclude_keys = ["lstm", "norm"]
        potential_keys = [
            key
            for key in model_params
            if all(exclude_key not in key for exclude_key in exclude_keys)
            and len(model_params[key].shape) == 2
        ]

        # Randomly choose a subset of keys to mutate
        how_many = int(self.rng.integers(1, len(potential_keys) + 1))
        chosen_keys = self.rng.choice(potential_keys, how_many, replace=False)

        for key in chosen_keys:
            W: torch.Tensor = model_params[key]
            # A diverged agent can reach this operator carrying non-finite
            # weights (NaN/inf) while still scoring a finite fitness, so it is
            # not caught by the fitness-level tournament guard. ``abs(NaN)`` is
            # ``NaN``, which fails ``torch.normal``'s ``std >= 0`` check and
            # aborts the run. Scrub in place before sampling so the mutation
            # repairs the weights instead of crashing; this is a no-op on the
            # finite path, so seeded runs stay reproducible.
            W.nan_to_num_(nan=0.0, posinf=mag_limit, neginf=-mag_limit)
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
                    mean=torch.zeros_like(std_super),
                    std=std_super,
                )
                new_vals[mask_super] = current_vals[mask_super] + noise_super
                if counts is not None:
                    counts["amplified"] += int(mask_super.sum().item())

            # Reset mutation: completely reset the weight using N(0, 1)
            if mask_reset.sum() > 0:
                noise_reset = torch.normal(
                    mean=torch.zeros(mask_reset.sum(), device=W.device),
                    std=torch.ones(mask_reset.sum(), device=W.device),
                )
                new_vals[mask_reset] = noise_reset
                if counts is not None:
                    counts["reset"] += int(mask_reset.sum().item())

            # Normal mutation: add noise with std proportional to the absolute current value times mut_strength
            if mask_normal.sum() > 0:
                std_normal = (mut_strength * current_vals[mask_normal]).abs()
                noise_normal = torch.normal(
                    mean=torch.zeros_like(std_normal),
                    std=std_normal,
                )
                new_vals[mask_normal] = current_vals[mask_normal] + noise_normal
                if counts is not None:
                    counts["ordinary"] += int(mask_normal.sum().item())

            # Integrate regularization by clamping all mutated values at once.
            # This is equivalent to your regularize_weight function.
            new_vals = new_vals.clamp(min=-mag_limit, max=mag_limit)

            # Write the mutated, clamped values back to the weight tensor
            W[rows_tensor, cols_tensor] = new_vals
            if self.accelerator is None:
                network = network.to(self.device)

        return network

    def _architecture_mutate_single(self, individual: RLAlgorithm) -> RLAlgorithm:
        """Apply an architecture mutation to a single-agent RL algorithm. Since all of the
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
        policy_name, policy_offspring = next(iter(policy.items()))

        if not policy_offspring.mutation_methods:
            warnings.warn(
                "No mutation methods found for the policy network. Skipping architecture mutation. "
                "We advise setting the probability of architecture mutations to zero when using non-evolvable networks.",
                stacklevel=2,
            )
            individual.mut = "None"
            individual.mut_details = {"category": "no mutation", "name": "none"}
            return individual

        # Sample mutation method from policy network
        mut_method = policy_offspring.sample_mutation_method(
            self.new_layer_prob,
            self.rng,
        )

        sizes_before = self._arch_signature(policy_offspring)
        applied_mutation, mut_dict = self._apply_arch_mutation(
            policy_offspring,
            mut_method,
        )
        sizes_after = self._arch_signature(policy_offspring)
        self._to_device_and_set_individual(individual, policy_name, policy_offspring)

        if isinstance(individual, (NeuralTS, NeuralUCB)):
            old_exp_layer = get_exp_layer(policy_offspring)
            self._reinit_bandit_grads(individual, policy_offspring, old_exp_layer)

        # Apply the same mutation to the rest of the evaluation modules
        for name, offspring in offspring_evals.items():
            if applied_mutation in offspring.mutation_methods:
                self._apply_arch_mutation(
                    offspring,
                    applied_mutation,
                    mut_dict,
                )
                self._to_device_and_set_individual(individual, name, offspring)

        individual.mutation_hook()  # Apply mutation hook
        individual.reinit_optimizers()  # Reinitialize optimizer
        individual.mut = applied_mutation or "None"
        individual.mut_details = self._build_arch_details(
            applied_mutation, mut_dict, sizes_before, sizes_after
        )
        individual.mut_details["arch_func_preserving"] = (
            self.arch_mut_type == "func_preserving"
        )

        return individual

    def _architecture_mutate_multi(
        self,
        individual: MultiAgentRLAlgorithm,
    ) -> MultiAgentRLAlgorithm:
        """Apply an architecture mutation to a multi-agent RL algorithm. Since each agent has its own
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
        policy_name, policy_offspring = next(iter(policy.items()))

        if not policy_offspring.mutation_methods:
            warnings.warn(
                "No mutation methods found for the policy network. Skipping architecture mutation. "
                "We advise setting the probability of architecture mutations to zero when using non-evolvable networks.",
                stacklevel=2,
            )
            individual.mut = "None"
            individual.mut_details = {"category": "no mutation", "name": "none"}
            return individual

        # Sample mutation method from policy network
        mut_method = policy_offspring.sample_mutation_method(
            self.new_layer_prob,
            self.rng,
        )

        # Apply the sampled method to the policy network (will only apply to one sub-agent)
        sizes_before = self._arch_signature(policy_offspring)
        applied_mutation, mut_dict = self._apply_arch_mutation(
            policy_offspring,
            mut_method,
        )
        sizes_after = self._arch_signature(policy_offspring)

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
                    policy,
                    sampled_mutation,
                    mut_dict,
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
                        sampled_mutation,
                        available_methods,
                        mutated_agent,
                    )

                    if analogous_method is not None:
                        self._apply_arch_mutation(
                            agent_eval,
                            analogous_method,
                            mut_dict,
                        )
                    else:
                        msg = (
                            f"Mutation method '{sampled_mutation}' not found in '{agent_eval.__class__.__name__}'. "
                            f"No analogous method found for agent '{agent_id}'. "
                            f"Available methods: {agent_eval.mutation_methods}."
                        )
                        raise MutationError(
                            msg,
                        )

            self._to_device_and_set_individual(individual, name, offspring_eval)

        individual.mutation_hook()  # Apply mutation hook
        individual.reinit_optimizers()  # Reinitialize optimizer
        individual.mut = sampled_mutation or "None"
        individual.mut_details = self._build_arch_details(
            sampled_mutation, mut_dict, sizes_before, sizes_after
        )
        individual.mut_details["arch_func_preserving"] = (
            self.arch_mut_type == "func_preserving"
        )

        return individual

    @staticmethod
    def _arch_signature(network: EvolvableModule) -> list[int]:
        """Return a per-layer width signature of a network's weight tensors.

        The signature is the output dimension (``shape[0]``) of every weight
        tensor with two or more dimensions (linear weights ``[out, in]`` and
        conv weights ``[out_ch, in_ch, ...]``). Diffing the signature before and
        after a mutation reveals which layer changed width or whether a layer was
        added/removed, generically across MLP and CNN policies (and ``ModuleDict``
        multi-agent policies, whose ``state_dict`` flattens all sub-agents).

        :param network: The network to summarize.
        :type network: EvolvableModule
        :return: List of per-layer output widths.
        :rtype: list[int]
        """
        try:
            return [
                int(tensor.shape[0])
                for tensor in network.state_dict().values()
                if hasattr(tensor, "shape") and len(tensor.shape) >= 2
            ]
        except (AttributeError, RuntimeError):
            return []

    @staticmethod
    def _build_arch_details(
        applied_mutation: str | None,
        mut_dict: dict[str, Any] | None,
        sizes_before: list[int],
        sizes_after: list[int],
    ) -> dict[str, Any]:
        """Build the ``mut_details`` dict for an architecture mutation.

        Node mutations (``add_node``/``remove_node``/``add_channel``/``remove_channel``)
        carry the changed layer and the number of neurons/channels in ``mut_dict``.
        Layer mutations (``add_layer``/``remove_layer``) are derived from the
        before/after width signatures.

        :param applied_mutation: The mutation method name (may include an
            ``<agent_id>.`` prefix for multi-agent algorithms).
        :type applied_mutation: str | None
        :param mut_dict: The dict returned by the mutation method.
        :type mut_dict: dict[str, Any] | None
        :param sizes_before: Width signature before the mutation.
        :type sizes_before: list[int]
        :param sizes_after: Width signature after the mutation.
        :type sizes_after: list[int]
        :return: The architecture mutation details.
        :rtype: dict[str, Any]
        """
        if applied_mutation is None:
            return {"category": "no mutation", "name": "none"}

        mut_dict = mut_dict or {}
        # Strip any "<agent_id>." prefix used by multi-agent mutation names
        base_name = applied_mutation.split(".")[-1]
        details: dict[str, Any] = {
            "category": "architecture",
            "name": applied_mutation,
            "layer_changed": "",
            "neurons_delta": "",
            "new_layer_position": "",
            "new_layer_size": "",
        }

        node_neurons = mut_dict.get("numb_new_nodes", mut_dict.get("numb_new_channels"))
        if "node" in base_name or "channel" in base_name:
            details["layer_changed"] = mut_dict.get("hidden_layer", "")
            if node_neurons is not None:
                sign = -1 if base_name.startswith("remove") else 1
                details["neurons_delta"] = sign * int(node_neurons)
        elif "layer" in base_name:
            if base_name.startswith("add") and len(sizes_after) > len(sizes_before):
                pos = next(
                    (
                        i
                        for i in range(len(sizes_before))
                        if sizes_before[i] != sizes_after[i]
                    ),
                    len(sizes_before),
                )
                details["new_layer_position"] = pos
                details["new_layer_size"] = sizes_after[pos]
                details["neurons_delta"] = sizes_after[pos]
            elif base_name.startswith("remove") and len(sizes_after) < len(
                sizes_before
            ):
                pos = next(
                    (
                        i
                        for i in range(len(sizes_after))
                        if sizes_before[i] != sizes_after[i]
                    ),
                    len(sizes_after),
                )
                details["new_layer_position"] = pos
                details["neurons_delta"] = -sizes_before[pos]

        return details

    def _apply_arch_mutation(
        self,
        network: EvolvableNetworkType,
        mut_method: str | None,
        applied_mut_dict: dict[str, Any] | None = None,
    ) -> tuple[str | None, MutationReturnType]:
        """Apply the mutation method to networks and returns mutation data if needed.

        :param networks: The networks to apply the mutation to
        :type networks: EvolvableNetworkType
        :param mut_method: The mutation method to apply
        :type mut_method: str | None
        :param applied_mut_dict: The mutation dictionary, defaults to None. Empty on
            the *policy* call and populated on the mirrored calls that replay the
            policy's mutation onto the agent's other evaluation networks.
        :type applied_mut_dict: dict[str, Any] | None, optional

        :return: The mutation method name and the mutation dictionary
        :rtype: tuple[str | None, MutationReturnType]
        """
        if not isinstance(network, EvolvableModule):
            msg = (
                f"Can't apply architecture mutation to {network.__class__.__name__} network."
                "Please make sure your network inherits from 'EvolvableModule'."
            )
            raise MutationError(
                msg,
            )

        applied_mut_dict = applied_mut_dict or {}
        mut_dict = None

        # Function-preserving pre-mutation step: warn where preservation is not
        # guaranteed and snapshot hidden widths so an add fixup can size itself.
        # A no-op when arch_mut_type == "original".
        func_preserving = (
            self.arch_mut_type == "func_preserving" and mut_method is not None
        )
        fp_before_widths: list[int] = []
        if func_preserving:
            fp_before_widths = self._fp_pre_mutation(network, mut_method)

        if mut_method is None:
            mut_dict = {}
            network.last_mutation_attr = None
            network.last_mutation = None
        else:
            if mut_method not in network.mutation_methods:
                msg = (
                    f"Mutation method '{mut_method}' not found in '{network.__class__.__name__}'; "
                    f"available methods: \n {network.mutation_methods}."
                )
                raise MutationError(
                    msg,
                )

            # The mirrored calls replay the policy's mut_dict, so every evaluation
            # network of the agent keeps a consistent architecture.
            mut_dict = getattr(network, mut_method)(**applied_mut_dict)

        mut_dict = mut_dict or {}
        applied_mut = network.last_mutation_attr

        # Function-preserving post-mutation fixups for the (resolved) addition
        # operators: zero the new units' outgoing weights / identity-init a new
        # layer. Keyed on ``applied_mut`` so add_layer -> add_node fallbacks are
        # handled correctly.
        if func_preserving and applied_mut is not None:
            self._fp_post_mutation(network, applied_mut, mut_dict, fp_before_widths)

        return applied_mut, mut_dict

    # ------------------------------------------------------------------ #
    # Function-preserving architecture-mutation helpers
    # ------------------------------------------------------------------ #
    def _fp_pre_mutation(
        self,
        network: EvolvableNetworkType,
        mut_method: str,
    ) -> list[int]:
        """Warn where preservation is not guaranteed and snapshot hidden widths.

        Only the *additions* need anything done before the mutation runs: the widths
        recorded here tell :meth:`_fp_post_mutation` which fan-out columns are new.
        Removals are left entirely to AgileRL's original positional operator, so they
        need no pre-step at all.

        :param network: The network being mutated.
        :param mut_method: The mutation method about to be applied.
        :return: The target sub-module's hidden-layer widths before the mutation
            (used to size the outgoing-weight zeroing of an addition), or, for a
            latent-dimension mutation, a single-element list holding the latent dim.
        """
        # Latent-dimension mutations cross the encoder->head boundary and are named
        # without an ``encoder``/``head_net`` segment, so handle them separately.
        if fp.is_latent_mutation(mut_method.split(".")[-1]):
            return self._fp_pre_latent_mutation(network, mut_method)

        agent_id, submodule_name, base = fp.parse_mut_target(mut_method)
        if submodule_name is None:
            return []
        try:
            _fwd_net, submodule = fp.resolve_target(network, agent_id, submodule_name)
        # TypeError covers a nested sub-encoder method
        # (``encoder.feature_net.<key>.remove_channel`` on EvolvableMultiInput), whose
        # name parses to an agent_id that is not a ModuleDict key -- the resolve then
        # subscripts a plain module. Leave it to the original operator.
        except (KeyError, AttributeError, TypeError):
            return []

        if base in fp.ADD_NODE_MUTATIONS:
            # add_node/add_channel zero the new units' fan-out, so they stay
            # function-preserving under ANY activation; only a norm layer (which
            # re-normalises over the changed unit set) breaks preservation.
            if fp.has_norm_layer(_fwd_net):
                self._fp_warn_layernorm()
        elif base in fp.ADD_LAYER_MUTATIONS:
            # add_layer's Net2DeeperNet identity init additionally requires a
            # ReLU/Identity base activation, so warn on a norm layer OR a
            # non-ReLU/Identity activation.
            if fp.has_norm_layer(_fwd_net):
                self._fp_warn_layernorm()
            elif fp.has_nonpreserving_activation(_fwd_net):
                self._fp_warn_add_layer_activation()
        if base == "change_kernel":
            self._fp_warn_kernel()

        return fp.hidden_widths(submodule)

    def _fp_post_mutation(
        self,
        network: EvolvableNetworkType,
        applied_mut: str,
        mut_dict: dict[str, Any],
        before_widths: list[int],
    ) -> None:
        """Zero/noise new units' outgoing weights / identity-init a new head layer."""
        # Latent-dimension adds are fixed up across the encoder->head boundary.
        if fp.is_latent_mutation(applied_mut.split(".")[-1]):
            self._fp_post_latent_mutation(network, applied_mut, before_widths)
            return

        agent_id, submodule_name, base = fp.parse_mut_target(applied_mut)
        if submodule_name is None:
            return
        try:
            _fwd_net, submodule = fp.resolve_target(network, agent_id, submodule_name)
        except (KeyError, AttributeError, TypeError):
            return

        if base in fp.ADD_NODE_MUTATIONS:
            hidden_layer = mut_dict.get("hidden_layer")
            if hidden_layer is None:
                return
            hidden_layer = int(hidden_layer)
            old_width = (
                before_widths[hidden_layer]
                if 0 <= hidden_layer < len(before_widths)
                else None
            )
            fp.init_new_outgoing(submodule, hidden_layer, old_width, self.arch_fp_noise)
        elif base in fp.ADD_LAYER_MUTATIONS:
            fp.identity_new_layer(submodule)

    def _fp_pre_latent_mutation(
        self,
        network: EvolvableNetworkType,
        mut_method: str,
    ) -> list[int]:
        """Snapshot the latent dim before a latent-dimension mutation.

        :return: ``[latent_dim_before]`` -- the latent dim before the mutation, used
            to size the head-input-column fixup of a latent addition. Latent
            removals need nothing here; they run the original positional operator.
        """
        agent_id, _base = fp.parse_latent_target(mut_method)
        try:
            fwd_net = fp.resolve_latent_network(network, agent_id)
        except (KeyError, AttributeError, TypeError):
            return []
        return [int(getattr(fwd_net, "latent_dim", 0))]

    def _fp_post_latent_mutation(
        self,
        network: EvolvableNetworkType,
        applied_mut: str,
        before_widths: list[int],
    ) -> None:
        """Zero/noise the head's new input columns after a latent-dimension add."""
        agent_id, base = fp.parse_latent_target(applied_mut)
        if base not in fp.LATENT_ADD_MUTATIONS:
            return  # latent removals are handled entirely pre-mutation
        try:
            fwd_net = fp.resolve_latent_network(network, agent_id)
        except (KeyError, AttributeError, TypeError):
            return
        old_latent = before_widths[0] if before_widths else None
        fp.init_new_latent_outgoing(fwd_net, old_latent, self.arch_fp_noise)

    def _fp_warn_layernorm(self) -> None:
        if not self._fp_warned_layernorm:
            self._fp_warned_layernorm = True
            warnings.warn(
                "arch_mut_type='func_preserving': the network uses a norm layer "
                "(LayerNorm/BatchNorm/GroupNorm), which re-normalises over the "
                "changed unit set, so function preservation cannot be guaranteed "
                "for add_node/add_channel/add_layer mutations.",
                stacklevel=2,
            )

    def _fp_warn_add_layer_activation(self) -> None:
        if not self._fp_warned_activation:
            self._fp_warned_activation = True
            warnings.warn(
                "arch_mut_type='func_preserving': add_layer uses Net2DeeperNet "
                "identity initialisation, which is only function-preserving for "
                "ReLU/Identity activations; the base activation is different, so "
                "function preservation cannot be guaranteed for add_layer "
                "mutations. (add_node/add_channel remain preserving.)",
                stacklevel=2,
            )

    def _fp_warn_kernel(self) -> None:
        if not self._fp_warned_kernel:
            self._fp_warned_kernel = True
            warnings.warn(
                "arch_mut_type='func_preserving': change_kernel cannot be made "
                "function-preserving; falling back to the original kernel-change "
                "behaviour for this mutation.",
                stacklevel=2,
            )

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
            msg = (
                f"Bandit algorithm architecture {type(offspring_actor)} not supported."
            )
            raise ValueError(
                msg,
            )

        individual.numel = sum(
            w.numel() for w in exp_layer.parameters() if w.requires_grad
        )
        individual.theta_0 = torch.cat(
            [w.flatten() for w in exp_layer.parameters() if w.requires_grad],
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
                if key not in new_params:
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
                if key in old_params:
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
                np.delete(new_sigma_inv, to_remove, 0),
                to_remove,
                1,
            )

        # Add new zeros corresponding to new params, make lambda down identity diagonal
        if len(to_add) > 0:
            new_sigma_inv = np.insert(
                np.insert(new_sigma_inv, to_add, 0, 0),
                to_add,
                0,
                1,
            )
            for i in to_add:
                new_sigma_inv[i, i] = individual.lamb

        individual.exp_layer = exp_layer
        individual.sigma_inv = torch.from_numpy(new_sigma_inv).to(
            (
                individual.device
                if individual.accelerator is None
                else individual.accelerator.device
            ),
        )

    def _find_analogous_mutation(
        self,
        sampled_mutation: str,
        available_methods: list[str],
        policy_agent: str,
    ) -> str | None:
        """Find an analogous mutation method when exact match is not found.

        Tries to match based on bottom-level method and agent ID.

        :param sampled_mutation: The mutation method that was sampled (e.g., 'encoder.add_channel')
        :type sampled_mutation: str
        :param available_methods: List of available mutation methods
        :type available_methods: list[str]
        :param policy_agent: The agent ID to match (e.g., 'agent_0')
        :type policy_agent: str

        :return: Analogous mutation method if found, None otherwise
        :rtype: str | None
        """
        if not sampled_mutation:
            return None

        if sampled_mutation in available_methods:
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
