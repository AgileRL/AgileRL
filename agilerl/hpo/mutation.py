# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, TypeGuard, TypeVar

import fastrand  # ty: ignore[unresolved-import] — C extension without type stubs
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
from agilerl.hpo import function_preserving
from agilerl.modules import EvolvableModule, ModuleDict
from agilerl.protocols import EvolvableAlgorithmProtocol
from agilerl.typing import MutationReturn
from agilerl.utils.algo_utils import remove_compile_prefix
from agilerl.utils.evolvable_networks import compile_model
from agilerl.wrappers.agent import AgentWrapper

AgentT = TypeVar("AgentT", bound=EvolvableAlgorithmProtocol)
IndividualT = TypeVar("IndividualT", bound=EvolvableAlgorithm)
SingleAgentT = TypeVar("SingleAgentT", bound=RLAlgorithm)
MultiAgentT = TypeVar("MultiAgentT", bound=MultiAgentRLAlgorithm)
BanditAlgorithm = NeuralUCB | NeuralTS

# A bound mutation method of `Mutations`: maps an individual to a mutated
# individual of the same type.
MutationFunc = Callable[[IndividualT], IndividualT]

torch._dynamo.config.cache_size_limit = 64
torch._logging.set_logs(dynamo=logging.FATAL)

logger = logging.getLogger(__name__)

_UNSUPPORTED_ACTIVATION_MUTATION_ALGOS = frozenset(
    {"PPO", "DDPG", "TD3", "IPPO", "MADDPG", "MATD3"},
)


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
    individual: EvolvableAlgorithm,
) -> tuple[dict[str, EvolvableModule], dict[str, EvolvableModule]]:
    """Get the offsprings of all of the evaluation modules in the individual.

    :param individual: The individual to inspect
    :type individual: EvolvableAlgorithm

    :return: Tuple of offspring policy and the rest of the evaluation modules
    :rtype: tuple[dict[str, EvolvableModule], dict[str, EvolvableModule]]
    """
    registry = individual.registry

    offspring_modules: dict[str, EvolvableModule] = {}
    offspring_policy: dict[str, EvolvableModule] = {}
    for group in registry.groups:
        eval_name = group.eval_network_name()
        eval_module: EvolvableModule = getattr(individual, eval_name)

        # Clone the offspring prior to applying mutations
        offspring = eval_module.clone()
        if group.policy:
            offspring_policy[eval_name] = offspring
        else:
            offspring_modules[eval_name] = offspring

    return offspring_policy, offspring_modules


def _is_module_dict(
    module: EvolvableModule,
) -> TypeGuard["ModuleDict[EvolvableModule]"]:
    """Narrow an evaluation module to its per-agent ``ModuleDict`` mapping.

    :param module: The evaluation module to check
    :type module: EvolvableModule
    :return: Whether the module is a per-agent ``ModuleDict``
    :rtype: TypeGuard[ModuleDict[EvolvableModule]]
    """
    return isinstance(module, ModuleDict)


def _as_module_dict(module: EvolvableModule) -> "ModuleDict[EvolvableModule]":
    """Narrow a multi-agent evaluation module to its per-agent mapping.

    :param module: The evaluation module to reinterpret
    :type module: EvolvableModule
    :return: The module as a mapping of per-agent modules
    :rtype: ModuleDict[EvolvableModule]
    """
    assert _is_module_dict(module), (
        "Multi-agent mutation requires a per-agent ModuleDict container."
    )
    return module


def get_exp_layer(offspring: EvolvableModule) -> nn.Linear:
    """Get the output layer of different types of offsprings for bandit algorithms.

    :param offspring: The offspring to inspect
    :type offspring: EvolvableModule

    :return: The output layer of the offspring
    :rtype: nn.Linear
    """
    if not isinstance(offspring, EvolvableModule):
        msg = f"Bandit algorithm architecture {type(offspring)} not supported."
        raise TypeError(msg)

    exp_layer = offspring.get_output_dense()
    if not isinstance(exp_layer, nn.Linear):
        msg = (
            f"Bandit algorithm architecture {type(offspring)} not supported: expected "
            f"a linear output layer, found {type(exp_layer)}."
        )
        raise TypeError(msg)

    return exp_layer


def reinit_shared_networks(
    mutation_func: Callable[["Mutations", IndividualT], IndividualT],
) -> Callable[["Mutations", IndividualT], IndividualT]:
    """Reinitialize shared networks after architecture and parameter mutations (decorator).

    :param mutation_func: The mutation function to decorate
    :type mutation_func: Callable[[Mutations, IndividualT], IndividualT]
    :return: The decorated mutation function
    :rtype: Callable[[Mutations, IndividualT], IndividualT]
    """

    @wraps(mutation_func)
    def wrapper(self: "Mutations", individual: IndividualT) -> IndividualT:
        # Call the original mutation function
        individual = mutation_func(self, individual)

        torch._dynamo.reset()  # NOTE: Should we do this?

        # Only proceed if mutation was actually applied
        if individual.mut == "None":
            return individual

        # Recompile individual if necessary
        compiled_model = individual.torch_compiler is not None
        if compiled_model:
            # Set dynamo config before recompilation to avoid guard failures.
            # The ignores below are needed because dynamo's config module infers
            # its attribute types from the default values, so they read as literals.
            torch._dynamo.config.force_parameter_static_shapes = False  # ty: ignore[invalid-assignment]
            individual.recompile()

        # Reinitialize shared networks to mutated evaluation networks
        for net_group in individual.registry.groups:
            for shared_name in net_group.shared_network_names():
                eval_offspring: EvolvableModule = getattr(
                    individual,
                    net_group.eval_network_name(),
                )
                # Reinitialize shared with frozen weights due to
                # potential mutation in architecture
                ind_shared: nn.Module = self._reinit_from_mutated(
                    eval_offspring,
                    remove_prefix=compiled_model,
                )
                if self.accelerator is None:
                    ind_shared = ind_shared.to(self.device)

                if compiled_model:
                    torch._dynamo.config.force_parameter_static_shapes = False  # ty: ignore[invalid-assignment]
                    ind_shared = compile_model(
                        ind_shared,
                        individual.torch_compiler,
                    )

                setattr(individual, shared_name, ind_shared)

        return individual

    return wrapper


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
        self.device = device
        self.accelerator = accelerator

        # Kept apart from self.rng so drawing symmetry-breaking noise never
        # shifts the stream that samples mutations for later agents.
        self._fp_rng = np.random.default_rng(rand_seed)
        self._warned_fp: set[str] = set()
        self._pre_training_mut = False

        self.pretraining_mut_options, self.pretraining_mut_proba = (
            self._get_mutations_options(pretraining=True)
        )
        self.mut_options, self.mut_proba = self._get_mutations_options()

    def mutation(
        self,
        population: list[AgentT],
        pre_training_mut: bool = False,
        indices: list[int] | None = None,
    ) -> list[AgentT]:
        """Return a mutated population of agents. See :ref:`evo_hyperparam_opt` for more details.

        :param population: Population of agents
        :type population: list[EvolvableAlgorithm]
        :param pre_training_mut: Boolean flag indicating if the mutation is before the training loop
        :type pre_training_mut: bool, optional
        :param indices: When given, mutate only the agents whose index appears in this list.
            Defaults to None
        :type indices: list[int], optional

        :return: Mutated population
        :rtype: list[EvolvableAlgorithm]
        """
        self._pre_training_mut = pre_training_mut

        # Create lists of possible mutation functions and their respective relative probabilities
        mutation_options = (
            self.pretraining_mut_options if pre_training_mut else self.mut_options
        )
        mutation_proba = (
            self.pretraining_mut_proba if pre_training_mut else self.mut_proba
        )

        if indices is not None:
            return self._mutate_selected(
                population, mutation_options, mutation_proba, indices
            )

        # Randomly choose mutation for each agent in population from options with
        # relative probabilities
        sampled_indices = self.rng.choice(
            len(mutation_options),
            len(population),
            p=mutation_proba,
        )
        mutation_choice: list[MutationFunc[Any]] = [
            mutation_options[int(index)] for index in sampled_indices
        ]

        # If not mutating elite member of population (first in list from tournament selection),
        # set this as the first mutation choice
        if not self.mutate_elite:
            mutation_choice[0] = self.no_mutation

        return [
            self._apply_mutation(individual, mutation)
            for mutation, individual in zip(mutation_choice, population, strict=False)
        ]

    def _mutate_selected(
        self,
        population: list[AgentT],
        mutation_options: list[MutationFunc[Any]],
        mutation_proba: list[float],
        indices: list[int],
    ) -> list[AgentT]:
        """Mutate only the agents whose globally-unique index is in indices.

        :param population: The whole population.
        :type population: list[EvolvableAlgorithm]
        :param mutation_options: Candidate mutation methods.
        :type mutation_options: list[MutationFunc]
        :param mutation_proba: Relative probabilities of ``mutation_options``.
        :type mutation_proba: list[float]
        :param indices: Indices of the agents to mutate.
        :type indices: list[int]
        :return: The population with the selected agents mutated.
        :rtype: list[EvolvableAlgorithm]
        """
        target_ids = set(indices)
        targets = [
            individual for individual in population if individual.index in target_ids
        ]
        sampled_indices = self.rng.choice(
            len(mutation_options),
            len(targets),
            p=mutation_proba,
        )
        mutation_choice: list[MutationFunc[Any]] = [
            mutation_options[int(index)] for index in sampled_indices
        ]
        chosen = {
            id(individual): mutation
            for individual, mutation in zip(targets, mutation_choice, strict=False)
        }

        mutated_population: list[AgentT] = []
        for individual in population:
            mutation = chosen.get(id(individual))
            if mutation is None:  # a non-selected agent passes through untouched
                mutated_population.append(individual)
                continue
            mutated_population.append(self._apply_mutation(individual, mutation))

        return mutated_population

    def _apply_mutation(
        self,
        individual: AgentT,
        mutation: MutationFunc[Any],
    ) -> AgentT:
        """Apply a single sampled mutation to one individual.

        :param individual: Individual to mutate, optionally wrapped.
        :type individual: EvolvableAlgorithm
        :param mutation: Sampled mutation method to apply to the underlying agent.
        :type mutation: MutationFunc

        :return: The mutated individual, wrapped exactly as it came in.
        :rtype: EvolvableAlgorithm
        """
        wrapped_ind = isinstance(individual, AgentWrapper)
        agent = individual.agent if wrapped_ind else individual

        agent = mutation(agent)
        agent.mutation_hook()

        if wrapped_ind:
            individual.agent = agent
            return individual

        return agent

    def no_mutation(self, individual: IndividualT) -> IndividualT:
        """Return individual from population without mutation.

        :param individual: Individual agent from population
        :type individual:
        """
        individual.mut = "None"  # No mutation
        return individual

    @reinit_shared_networks
    def architecture_mutate(self, individual: IndividualT) -> IndividualT:
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

    def rl_hyperparam_mutation(self, individual: IndividualT) -> IndividualT:
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
            return individual

        mutate_attr, mutate_param = hp_config.sample()

        if mutate_param.value is None:
            mutate_param.value = getattr(individual, mutate_attr)

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
        return individual

    # TODO: Activation mutations should really be integrated as architecture mutations
    @reinit_shared_networks
    def activation_mutation(self, individual: IndividualT) -> IndividualT:
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
        if isinstance(individual, LLMAlgorithm) or (
            individual.algo in _UNSUPPORTED_ACTIVATION_MUTATION_ALGOS
        ):
            label = (
                "LLM algorithms"
                if isinstance(individual, LLMAlgorithm)
                else individual.algo
            )
            warnings.warn(
                f"Activation mutations are not supported for {label}. Skipping mutation.",
                stacklevel=2,
            )
            individual.mut = "None"
            return individual

        # Mutate network activation layer
        registry = individual.registry
        no_activation = False
        for network_group in registry.groups:
            eval_name = network_group.eval_network_name()
            eval_module: EvolvableModule = getattr(individual, eval_name)

            if eval_module.activation is None:
                no_activation = True
            else:
                eval_module = self._permutate_activation(eval_module)

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

            setattr(individual, eval_name, eval_module)

        individual.reinit_optimizers()  # Reinitialize optimizer
        individual.mut = "act" if not no_activation else "None"
        return individual

    def parameter_mutation(self, individual: IndividualT) -> IndividualT:
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
            return individual

        registry = individual.registry

        # We only apply parameter mutations to the evaluation policy network
        # (i.e. the network used to select actions)
        policy_group = registry.policy(return_group=True)
        if policy_group is None:
            msg = (
                f"No policy network group registered for {individual.__class__.__name__}. "
                "Please register one of the network groups with 'policy=True'."
            )
            raise MutationError(msg)

        policy_name = policy_group.eval_network_name()
        offspring_policy: EvolvableModule = getattr(individual, policy_name)
        if _is_module_dict(offspring_policy):
            for agent_id, module in offspring_policy.items():
                offspring_policy[agent_id] = self._gaussian_parameter_mutation(module)
        else:
            offspring_policy = self._gaussian_parameter_mutation(offspring_policy)

        self._to_device_and_set_individual(
            individual,
            policy_name,
            offspring_policy,
        )

        # Load state dicts for shared networks
        for shared in policy_group.shared_network_names():
            offspring_shared: EvolvableModule = getattr(individual, shared)
            offspring_shared.load_state_dict(
                offspring_policy.state_dict(),
                strict=False,
            )
            self._to_device_and_set_individual(individual, shared, offspring_shared)

        individual.reinit_optimizers()  # Reinitialize optimizer
        individual.mut = "param"

        return individual

    def _fp_snapshot(
        self,
        network: EvolvableModule,
        mut_method: str | None,
    ) -> list[int] | None:
        """Record the widths an architecture mutation is about to change.

        Taken before the mutation runs, because
        :meth:`EvolvableModule.preserve_parameters` appends new units at the
        tail and the fixup needs to know where the old ones ended.

        :param network: Network about to be mutated.
        :type network: EvolvableModule
        :param mut_method: The mutation method about to be applied.
        :type mut_method: str | None

        :return: The affected widths, or None when there is nothing to record.
        :rtype: list[int] | None
        """
        target = function_preserving.resolve_target(network, mut_method)
        if target is None:
            return None

        if function_preserving.is_latent_mutation(
            function_preserving.base_mutation(mut_method),
        ):
            return [int(getattr(target, "latent_dim", 0))]

        return function_preserving.hidden_widths(target)

    def _fp_preserve(
        self,
        network: EvolvableModule,
        applied_mut: str | None,
        mut_dict: MutationReturn,
        before: list[int],
    ) -> None:
        """Initialise an addition's new capacity so the network is unchanged.

        Keyed on the mutation that was *applied* rather than the one that was
        sampled, since ``add_layer`` and ``remove_layer`` fall back to widening
        when they hit their depth limits.

        A fixup that raises leaves the network exactly as the original operator
        built it, so a failure costs the preservation and nothing else.

        :param network: Network that was mutated.
        :type network: EvolvableModule
        :param applied_mut: The mutation that was actually applied.
        :type applied_mut: str | None
        :param mut_dict: The mutation's own report of what it changed.
        :type mut_dict: MutationReturn
        :param before: The widths recorded before the mutation.
        :type before: list[int]

        :return: None.
        :rtype: None
        """
        base = function_preserving.base_mutation(applied_mut)
        if base not in function_preserving.PRESERVED_MUTATIONS:
            return

        target = function_preserving.resolve_target(network, applied_mut)
        if target is None:
            return

        try:
            reason = self._fp_apply(target, base, mut_dict, before)
        except Exception as exc:  # leave the original operator's result in place
            logger.warning("Function-preserving fixup skipped for a network: %s", exc)
            return

        # Warned outside the guard above: escalating warnings to errors is a
        # deliberate caller choice, and swallowing that here would report a
        # declined mutation as a failed fixup.
        self._fp_warn_declined(reason)

    def _fp_apply(
        self,
        target: nn.Module,
        base: str,
        mut_dict: MutationReturn,
        before: list[int],
    ) -> str | None:
        """Run the fixup that matches an addition, unless something blocks it.

        A blocker only rules on structure, so a fixup it waves through can still
        find nothing to write. Reporting that as a decline keeps a silently
        unpreserved addition from being indistinguishable from a preserved one.

        :param target: The module the mutation acted on.
        :type target: nn.Module
        :param base: The applied mutation's trailing method name.
        :type base: str
        :param mut_dict: The mutation's own report of what it changed.
        :type mut_dict: MutationReturn
        :param before: The widths recorded before the mutation.
        :type before: list[int]

        :return: The reason preservation was declined, or None when it applied.
        :rtype: str | None
        """
        if base in function_preserving.LATENT_ADDITIONS:
            reason = function_preserving.latent_addition_blocker(target)
            if reason is not None:
                return reason

            written = function_preserving.preserve_added_latent(
                target,
                before[0],
                self._fp_rng,
                function_preserving.FP_NOISE_SCALE,
            )
            return None if written else "not_written"

        if base in function_preserving.LAYER_ADDITIONS:
            reason = function_preserving.layer_addition_blocker(target)
            if reason is not None:
                return reason

            written = function_preserving.preserve_added_layer(target)
            return None if written else "not_written"

        # A layer mutation reports no index at all, and the declared
        # ``MutationReturn`` union also admits a per-sub-agent mapping, so
        # narrow to the plain integer the node mutations document rather than
        # trusting the key to be present and scalar.
        reported = mut_dict.get("hidden_layer")
        hidden_layer = reported if isinstance(reported, int) else None

        # A structural exclusion outranks a missing index, so the blocker runs
        # first and gets to name the reason.
        reason = function_preserving.node_addition_blocker(target, hidden_layer)
        if reason is not None:
            return reason
        if hidden_layer is None or not 0 <= hidden_layer < len(before):
            return "no_consumer"

        written = function_preserving.preserve_added_nodes(
            target,
            hidden_layer,
            before[hidden_layer],
            self._fp_rng,
            function_preserving.FP_NOISE_SCALE,
        )
        return None if written else "not_written"

    def _fp_warn_declined(self, reason: str | None) -> None:
        """Report once per instance that an addition could not be preserved.

        Repeating the warning per agent per generation would drown the training
        log, while staying silent would leave a configuration that never gets
        preservation indistinguishable from one that always does.

        :param reason: Why preservation was declined, or None.
        :type reason: str | None

        :return: None.
        :rtype: None
        """
        if reason is None or self._pre_training_mut or reason in self._warned_fp:
            return

        # Recorded only once the warning has actually been delivered: a caller
        # running with warnings escalated to errors never sees this one, and
        # marking it reported would silence the reason for the rest of the run.
        warnings.warn(
            "Architecture mutation fell back from function-preserving "
            f"initialisation: {function_preserving.DECLINE_REASONS[reason]}. The "
            "new capacity is initialised randomly instead.",
            stacklevel=4,
        )
        self._warned_fp.add(reason)

    def _get_mutations_options(
        self,
        pretraining: bool = False,
    ) -> tuple[list[MutationFunc[Any]], list[float]]:
        """Get the mutation options and probabilities for the given mutation
        configuration.

        :param pretraining: Boolean flag indicating if the mutation is before the training loop
        :type pretraining: bool
        :return: Mutation functions and their respective relative probabilities
        :rtype: tuple[list[MutationFunc], list[float]]
        """
        # Create lists of possible mutation functions and their
        # respective relative probabilities. No mutation is never sampled
        # during pre-training mutations.
        weighted_options = (
            (self.no_mutation, 0.0 if pretraining else self.no_mut),
            (self.architecture_mutate, self.architecture_mut),
            (self.parameter_mutation, self.parameters_mut),
            (self.activation_mutation, self.activation_mut),
            (self.rl_hyperparam_mutation, self.rl_hp_mut),
        )

        mutation_funcs: list[MutationFunc[Any]] = []
        weights: list[float] = []
        for func, prob in weighted_options:
            if prob > 0:
                mutation_funcs.append(func)
                weights.append(prob)

        # This will really only happen when pretraining is True and user has set
        # all mutation probabilities to zero, hence we apply no mutation
        if not mutation_funcs:
            mutation_funcs.append(self.no_mutation)
            weights.append(1.0)

        total = sum(weights)
        mutation_proba = [weight / total for weight in weights]
        return mutation_funcs, mutation_proba

    def _to_device_and_set_individual(
        self,
        individual: EvolvableAlgorithm,
        name: str,
        networks: EvolvableModule,
    ) -> None:
        """Move networks to the device and assigns them back to the individual.

        :param individual: The individual to assign the networks to
        :type individual: EvolvableAlgorithm
        :param name: The name of the attribute to assign the networks to
        :type name: str
        :param networks: The networks to move to the device
        :type networks: EvolvableModule
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
        offspring: EvolvableModule,
        remove_prefix: bool = False,
    ) -> EvolvableModule:
        """Reinitialize the mutated offspring with their state dictionary.

        :param offspring: The offspring to reinitialize
        :type offspring: EvolvableModule
        :param remove_prefix: Whether to remove the prefix from the offspring
        :type remove_prefix: bool

        :return: The reinitialized offspring
        :rtype: EvolvableModule
        """
        ind_shared: EvolvableModule
        if _is_module_dict(offspring):
            reinit_modules: dict[str, EvolvableModule] = OrderedDict()
            for agent_id, nested_offspring in offspring.items():
                reinit_modules[agent_id] = self._reinit_module(
                    nested_offspring,
                    nested_offspring.init_dict,
                )

            state_dicts = {
                agent_id: nested.state_dict() for agent_id, nested in offspring.items()
            }
            self._load_state_dicts(reinit_modules, state_dicts, remove_prefix)

            ind_shared = ModuleDict(reinit_modules)
        else:
            ind_shared = self._reinit_module(offspring, offspring.init_dict)
            ind_shared.load_state_dict(offspring.state_dict(), strict=False)

        return ind_shared

    def _load_state_dicts(
        self,
        modules: Mapping[str, EvolvableModule],
        state_dicts: Mapping[str, dict[str, Any]],
        remove_prefix: bool = False,
    ) -> None:
        """Load the state dictionaries for a multi-agent ModuleDict.

        :param modules: The modules to load the state dictionary into
        :type modules: Mapping[str, EvolvableModule]
        :param state_dicts: The state dictionary to load
        :type state_dicts: Mapping[str, dict[str, Any]]
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

    def _gaussian_parameter_mutation(self, network: EvolvableModule) -> EvolvableModule:
        """Return network with mutated weights using a Gaussian distribution.

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

            # Reset mutation: completely reset the weight using N(0, 1)
            num_reset = int(mask_reset.sum())
            if num_reset > 0:
                noise_reset = torch.normal(
                    mean=torch.zeros(num_reset, device=W.device),
                    std=torch.ones(num_reset, device=W.device),
                )
                new_vals[mask_reset] = noise_reset

            # Normal mutation: add noise with std proportional to the absolute current value times mut_strength
            if mask_normal.sum() > 0:
                std_normal = (mut_strength * current_vals[mask_normal]).abs()
                noise_normal = torch.normal(
                    mean=torch.zeros_like(std_normal),
                    std=std_normal,
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

    def _architecture_mutate_single(
        self,
        individual: SingleAgentT,
    ) -> SingleAgentT:
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
            return individual

        # Sample mutation method from policy network. The sampled method is the
        # attribute name of the mutation, drawn as a numpy string.
        mut_method = str(
            policy_offspring.sample_mutation_method(self.new_layer_prob, self.rng),
        )

        applied_mutation, mut_dict = self._apply_arch_mutation(
            policy_offspring,
            mut_method,
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
        individual.reinit_optimizers()  # Reinitialize optimizer
        individual.mut = applied_mutation or "None"

        return individual

    def _architecture_mutate_multi(
        self,
        individual: MultiAgentT,
    ) -> MultiAgentT:
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
        policy_name, policy_module = next(iter(policy.items()))

        if not policy_module.mutation_methods:
            warnings.warn(
                "No mutation methods found for the policy network. Skipping architecture mutation. "
                "We advise setting the probability of architecture mutations to zero when using non-evolvable networks.",
                stacklevel=2,
            )
            individual.mut = "None"
            return individual

        # Sample mutation method from policy network.
        mut_method = str(
            policy_module.sample_mutation_method(self.new_layer_prob, self.rng),
        )

        # Apply the sampled method to the policy network (will only apply to one sub-agent)
        applied_mutation, mut_dict = self._apply_arch_mutation(
            policy_module,
            mut_method,
        )

        applied_mutations: list[str] = []
        if applied_mutation is not None:
            split_mutation = applied_mutation.split(".")
            sampled_agent_id = split_mutation[0]
            sampled_mutation = ".".join(split_mutation[1:])
            applied_mutations.append(sampled_agent_id)
        else:
            sampled_agent_id = mut_method.split(".")[0]
            sampled_mutation = None

        # Applying to the remaining sub-agents needs the per-agent mapping.
        policy_offspring = _as_module_dict(policy_module)
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
        for name, eval_module in offspring_evals.items():
            offspring_eval = _as_module_dict(eval_module)

            # Iterate over the agents in the offspring evaluation module
            for agent_id, agent_eval in offspring_eval.items():
                # Iterate over the the agents whose policies were mutated
                analogous_method: str | None = None
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

        return individual

    def _apply_arch_mutation(
        self,
        network: EvolvableModule,
        mut_method: str | None,
        applied_mut_dict: MutationReturn | None = None,
    ) -> tuple[str | None, MutationReturn]:
        """Apply the mutation method to networks and returns mutation data if needed.

        :param networks: The networks to apply the mutation to
        :type networks: EvolvableModule
        :param mut_method: The mutation method to apply
        :type mut_method: str | None
        :param applied_mut_dict: The mutation dictionary, defaults to None
        :type applied_mut_dict: MutationReturn | None, optional

        :return: The mutation method name and the mutation dictionary
        :rtype: tuple[str | None, MutationReturn]
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
        before = None
        mut_dict = None
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

            before = self._fp_snapshot(network, mut_method)
            mut_dict = getattr(network, mut_method)(**applied_mut_dict)

        mut_dict = mut_dict or {}
        applied_mut = network.last_mutation_attr
        if before is not None:
            self._fp_preserve(network, applied_mut, mut_dict, before)

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
            exp_layer = get_exp_layer(offspring_actor)
        else:
            msg = (
                f"Bandit algorithm architecture {type(offspring_actor)} not supported."
            )
            raise ValueError(msg)

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

        # Bandit actors always expose a linear output layer (asserted where
        # the algorithm first resolves it).
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
        sampled_mutation: str | None,
        available_methods: list[str],
        policy_agent: str,
    ) -> str | None:
        """Find an analogous mutation method when exact match is not found.

        Tries to match based on bottom-level method and agent ID.

        :param sampled_mutation: The mutation method that was sampled (e.g., 'encoder.add_channel')
        :type sampled_mutation: str | None
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
