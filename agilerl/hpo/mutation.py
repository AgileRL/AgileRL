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
from agilerl.utils.dormant_neurons import (
    _eval_networks,
    capture_per_neuron_scores,
)
from agilerl.utils.evolvable_networks import compile_model
from agilerl.wrappers.agent import AgentWrapper

IndividualType = TypeVar("IndividualType", bound=EvolvableAlgorithm)
MutationsType = TypeVar("MutationsType", bound="Mutations")
PopulationType = list[IndividualType]
BanditAlgorithm = NeuralUCB | NeuralTS

torch._dynamo.config.cache_size_limit = 64
torch._logging.set_logs(dynamo=logging.FATAL)

logger = logging.getLogger(__name__)

# Normalisation layers the dormant-neuron surgery has to account for: they preserve neuron
# identity but hold per-neuron state of their own, so a recycled neuron's entry has
# to travel with it. Listed explicitly rather than via ``nn.modules.batchnorm``'s
# private base class, and filtered so the tuple stays valid on torch versions
# predating ``RMSNorm``.
_NORM_LAYER_TYPES: tuple[type[nn.Module], ...] = tuple(
    layer
    for layer in (
        nn.LayerNorm,
        nn.GroupNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        getattr(nn, "RMSNorm", None),
    )
    if layer is not None
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
    :param random_reset_param_mut: Whether the Gaussian parameter mutation keeps its
        random-reset band (a selected weight is *replaced* by a fresh ``N(0, 1)`` draw
        rather than perturbed), defaults to True
    :type random_reset_param_mut: bool, optional
    :param amplified_gauss_param_mut: Whether the Gaussian parameter mutation keeps its
        amplified ("super") noise band, defaults to True
    :type amplified_gauss_param_mut: bool, optional
    :param regrama_param_mut: Whether the parameter mutation resets dormant neurons
        (ReGraMa) before the Gaussian pass, defaults to False
    :type regrama_param_mut: bool, optional
    :param dormant_tau: ReGraMa dormancy threshold on the normalised per-neuron gradient,
        defaults to 0.1
    :type dormant_tau: float, optional
    :param overact_beta: Over-activity threshold on the normalised per-neuron gradient,
        above which a neuron is *split* across the dormant neurons it is reborn into
        (ReBorn, Qin et al.). Not exposed on :class:`~agilerl.models.hpo.MutationSpec`:
        the default ``inf`` disables splitting outright, leaving the pure dormant-neuron
        reset (ReGraMa) as the manifest-reachable behaviour. Must exceed ``dormant_tau``,
        defaults to ``float("inf")``
    :type overact_beta: float, optional
    :param regrama_out_scale: ReGraMa revival strength -- the outgoing weights of a
        Xavier-reset neuron are re-seeded at this fraction of the consumer layer's live
        column scale. ``0.0`` restores the original zero-outgoing behaviour, defaults to 0.02
    :type regrama_out_scale: float, optional
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
        random_reset_param_mut: bool = True,
        amplified_gauss_param_mut: bool = True,
        regrama_param_mut: bool = False,
        dormant_tau: float = 0.1,
        overact_beta: float = float("inf"),
        regrama_out_scale: float = 0.02,
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
        assert isinstance(random_reset_param_mut, bool), (
            "random_reset_param_mut must be boolean value True or False."
        )
        assert isinstance(amplified_gauss_param_mut, bool), (
            "amplified_gauss_param_mut must be boolean value True or False."
        )
        assert isinstance(regrama_param_mut, bool), (
            "regrama_param_mut must be boolean value True or False."
        )
        assert dormant_tau > 0, "dormant_tau must be greater than zero."
        assert overact_beta >= 0, "overact_beta must be non-negative."
        assert overact_beta > dormant_tau, (
            "overact_beta must be greater than dormant_tau."
        )
        assert regrama_out_scale >= 0, "regrama_out_scale must be non-negative."
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
        # Which of the Gaussian parameter mutation's three bands stay live. Each flag
        # only zeroes its own band's mask, so the dropped mass is *not* redistributed:
        # those weights simply keep their trained values, and the ordinary band still
        # covers 90% of the sampled entries, so a parameter mutation is never a no-op.
        self.random_reset_param_mut = random_reset_param_mut
        self.amplified_gauss_param_mut = amplified_gauss_param_mut
        # ReGraMa parameter-mutation configuration ("Measure gradients, not
        # activations!"). When ``regrama_param_mut`` is set, the parameter mutation
        # resets dormant neurons before the Gaussian noise pass; ``mutation_sd`` still
        # scales that pass's ordinary noise. Detection reads the per-neuron
        # pre-activation gradient snapshot captured during training (GraMa), threaded
        # per-parent through ``self._grama_side_table`` (set by :meth:`mutation`)
        # during the main loop. ``overact_beta`` is inf unless a caller overrides it in
        # Python, so the ReBorn neuron-split half (Qin et al.) never fires by default.
        self.regrama_param_mut = regrama_param_mut
        self.dormant_tau = dormant_tau
        self.overact_beta = overact_beta
        self.regrama_out_scale = regrama_out_scale
        self._grama_side_table: dict[int, Any] | None = None
        self._warned_recurrent = False
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
        env: Any | None = None,
        grama_scores: dict[int, Any] | None = None,
    ) -> PopulationType:
        """Return a mutated population of agents. See :ref:`evo_hyperparam_opt` for more details.

        :param population: Population of agents
        :type population: list[EvolvableAlgorithm]
        :param pre_training_mut: Boolean flag indicating if the mutation is before the training loop
        :type pre_training_mut: bool, optional
        :param env: Retained for API compatibility; unused by the gradient-based
            ReGraMa parameter mutation (which reads the pre-computed gradient
            snapshot rather than collecting observations).
        :type env: Any | None, optional
        :param grama_scores: Per-parent map ``{agent.index: _grama_scores}`` of the
            gradient snapshots captured during the last training block (see
            :class:`agilerl.utils.dormant_neurons.GraMaCapture`). Used by ReGraMa to
            score neurons; looked up per child via its ``_parent_index``. When
            ``None`` (e.g. the pre-training mutation), a ReGraMa-configured operator
            falls back to the Gaussian parameter mutation alone.
        :type grama_scores: dict[int, Any] | None, optional

        :return: Mutated population
        :rtype: list[EvolvableAlgorithm]
        """
        # Make the gradient snapshot side-table available to the (ReGraMa) parameter
        # mutation for the duration of this call only; reset afterwards so a later
        # snapshot-less call (e.g. pre-training) cannot reuse a stale table.
        self._grama_side_table = grama_scores

        # A ReGraMa regime needs the per-parent gradient snapshot to score neurons.
        # When configured for ReGraMa but called without one on a regular (non
        # pre-training) mutation step -- e.g. a trainer that does not thread the
        # snapshots, or the accelerator path -- the parameter mutation silently
        # falls back to the Gaussian operator, which would misattribute results.
        # The pre-training step is expected to run snapshot-less, so it is exempt.
        if self.regrama_param_mut and grama_scores is None and not pre_training_mut:
            warnings.warn(
                "regrama_param_mut is set but no gradient snapshot was provided to "
                "mutation(); falling back to the Gaussian parameter mutation for "
                "this step. ReGraMa is only wired into the on-policy, off-policy and "
                "multi-agent on-policy trainers running without an accelerator.",
                stacklevel=2,
            )

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

        # Drop the snapshot table so it is not held beyond this call.
        self._grama_side_table = None

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

        # ReGraMa parameter mutation: reset dormant neurons before the Gaussian
        # noise pass. It scores neurons from the per-neuron pre-activation gradient
        # snapshot captured during the parent's last training block (GraMa), looked
        # up via the child's ``_parent_index``. If no snapshot is available (e.g. the
        # pre-training mutation step, or an untrained agent), we fall back to the
        # Gaussian mutation below on its own.
        if self.regrama_param_mut:
            grama_scores = None
            side_table = self._grama_side_table
            parent_index = getattr(individual, "_parent_index", None)
            if side_table is not None and parent_index is not None:
                grama_scores = side_table.get(parent_index)
            if grama_scores:
                return self.regrama_parameter_mutation(individual, grama_scores)
            if side_table is not None:
                # Snapshots *were* supplied for this step, so this is not the
                # expected env-less fallback the call-level guard in
                # :meth:`mutation` already warns about: this particular agent
                # degrades to the Gaussian operator and is then recorded under the
                # "parameter" category, indistinguishable from a configured
                # Gaussian run. Name both ways the lookup fails so the cause is
                # actionable rather than invisible.
                warnings.warn(
                    "regrama_param_mut is set but no gradient snapshot was found "
                    f"for the agent cloned from parent index {parent_index!r}; "
                    "falling back to the Gaussian parameter mutation for this "
                    "agent. Either the agent carries no '_parent_index' (it must "
                    "be set on the unwrapped algorithm -- assigning it to an "
                    "AgentWrapper leaves it on the wrapper) or its parent's "
                    "snapshot is missing from the table passed to "
                    "mutation(grama_scores=...).",
                    stacklevel=2,
                )

        registry = individual.registry

        # We only apply parameter mutations to the evaluation policy network
        # (i.e. the network used to select actions)
        policy_group = registry.policy(return_group=True)
        offspring_policy: EvolvableNetworkType = getattr(
            individual,
            policy_group.eval_network,
        )
        # Accumulate per-category weight counts across all mutated networks. The two
        # band switches are the single authority over which bands run, so a ReGraMa
        # regime that reaches this branch as the snapshot-less fallback keeps exactly
        # the bands it is configured with.
        include_reset = self.random_reset_param_mut
        include_amplified = self.amplified_gauss_param_mut
        counts = {"reset": 0, "ordinary": 0, "amplified": 0}
        if isinstance(offspring_policy, ModuleDict):
            for agent_id, module in offspring_policy.items():
                offspring_policy[agent_id] = self._gaussian_parameter_mutation(
                    module,
                    counts=counts,
                    include_reset=include_reset,
                    include_amplified=include_amplified,
                )
        else:
            offspring_policy = self._gaussian_parameter_mutation(
                offspring_policy,
                counts=counts,
                include_reset=include_reset,
                include_amplified=include_amplified,
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

    # ------------------------------------------------------------------ #
    # ReGraMa parameter mutation ("Measure gradients, not activations!";  #
    # the neuron-split half is ReBorn, Qin et al., "The Dormant Neuron    #
    # Phenomenon in Multi-Agent RL Value Factorization")                  #
    # ------------------------------------------------------------------ #
    def regrama_parameter_mutation(
        self, individual: IndividualType, grama_scores: list[list[Any]]
    ) -> IndividualType:
        """Reset dormant neurons, then add a gentle Gaussian pass.

        For every measured layer of every evaluation network (actors, critics and
        each multi-agent sub-policy), dormant neurons are Xavier-reset with their
        outgoing weights re-seeded small (at ``regrama_out_scale`` of the consumer's
        live column scale -- *not* zeroed; see :meth:`_revived_out_block`). Detection
        uses the same per-neuron gradient scoring as the dormant-neuron diagnostic,
        read from the parent's captured gradient snapshot *grama_scores* (no
        forward/backward pass here).

        ``overact_beta`` defaults to ``inf``, so no neuron is ever over-active and
        every dormant neuron falls through to that reset. A caller that lowers it in
        Python re-enables ReBorn's neuron *split*, in which an over-active neuron is
        reborn into the dormant neurons it claims. **That split is not
        function-preserving in general** -- see :meth:`_apply_reborn_to_layer` for the
        conditions it needs, how it adapts to a normalisation, and what it preserves
        unconditionally. It is a mutation regardless of whether those conditions hold:
        a perturbation the tournament can select against, not a guaranteed-safe
        rewrite. Callers should not rely on the child's outputs matching its parent's.

        After that surgery, the policy evaluation network additionally receives the
        Gaussian parameter mutation, restricted to whichever bands
        ``random_reset_param_mut`` / ``amplified_gauss_param_mut`` leave live and
        scaled by ``self.mutation_sd``. This breaks the symmetry of the reset units so
        they can specialise. It runs before the shared/target sync so those copies
        stay consistent with the fully mutated policy.

        :param individual: Individual agent from population.
        :type individual: RLAlgorithm or MultiAgentRLAlgorithm
        :param grama_scores: The parent's captured per-neuron pre-activation
            gradient snapshot (``_grama_scores``): one list per evaluation network
            in :func:`_eval_networks` order, each a per-layer list aligned to
            :func:`_target_activations` order.
        :type grama_scores: list[list[Any]]
        :return: The individual with ReGraMa-reset parameters.
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

        counts = {"reborn": 0, "xavier": 0, "overactive": 0, "dormant": 0}

        # Route each evaluation network's captured per-layer gradient scores
        # positionally (the child's architecture matches its parent's, since an
        # agent receiving ReGraMa did not receive an architecture mutation this
        # cycle). ``capture_per_neuron_scores`` guards any length mismatch.
        for idx, (_network_id, network) in enumerate(_eval_networks(individual)):
            per_neuron_list = grama_scores[idx] if idx < len(grama_scores) else None
            try:
                self._regrama_network_surgery(network, per_neuron_list, counts)
            except Exception as exc:  # keep this network untouched on failure
                logger.warning("ReGraMa surgery skipped for a network: %s", exc)

        # After resetting neurons, apply a gentle exploration pass to the policy
        # network, keeping whichever Gaussian bands are configured live. Running this
        # after the ReGraMa surgery breaks the symmetry of the freshly reset units so
        # they can specialise; running it before the shared/target sync below keeps
        # those copies consistent with the fully mutated policy.
        gauss_counts = {"reset": 0, "ordinary": 0, "amplified": 0}
        policy_group = individual.registry.policy(return_group=True)
        offspring_policy: EvolvableNetworkType = getattr(
            individual, policy_group.eval_network
        )
        if isinstance(offspring_policy, ModuleDict):
            for agent_id, module in offspring_policy.items():
                offspring_policy[agent_id] = self._gaussian_parameter_mutation(
                    module,
                    counts=gauss_counts,
                    include_reset=self.random_reset_param_mut,
                    include_amplified=self.amplified_gauss_param_mut,
                )
        else:
            offspring_policy = self._gaussian_parameter_mutation(
                offspring_policy,
                counts=gauss_counts,
                include_reset=self.random_reset_param_mut,
                include_amplified=self.amplified_gauss_param_mut,
            )
        self._to_device_and_set_individual(
            individual, policy_group.eval_network, offspring_policy
        )

        # Sync shared / target networks from the mutated eval networks, per group.
        for group in individual.registry.groups:
            if group.shared_networks:
                eval_net = getattr(individual, group.eval_network)
                for shared in group.shared_networks:
                    shared_net = getattr(individual, shared)
                    shared_net.load_state_dict(eval_net.state_dict(), strict=False)
                    self._to_device_and_set_individual(individual, shared, shared_net)

        individual.reinit_optimizers()
        individual.mut = "param_reborn"
        individual.mut_details = self._regrama_mut_details(counts, gauss_counts)
        return individual

    @staticmethod
    def _regrama_mut_details(
        counts: dict[str, int], gauss_counts: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Build the ``mut_details`` record for a ReGraMa mutation.

        Includes both the neuron-recycling counts and the trailing Gaussian pass's
        per-category weight counts (a band switched off contributes ``0``).
        ``gauss_counts`` defaults to all-zero for the bad-obs early return, where no
        surgery and no noise pass ran.

        The ``"reborn"`` category and ``reborn_*`` detail keys are the on-disk schema
        of ``mutation_history.csv`` (see :class:`agilerl.logger.MutationHistoryLogger`)
        and are deliberately left at their original spelling so runs recorded before
        the ReGraMa rename stay directly comparable.
        """
        gauss_counts = gauss_counts or {"reset": 0, "ordinary": 0, "amplified": 0}
        return {
            "category": "reborn",
            "name": "param_reborn",
            "neurons_reborn": counts["reborn"],
            "neurons_xavier_reset": counts["xavier"],
            "overactive_count": counts["overactive"],
            "dormant_count": counts["dormant"],
            "weights_reset": gauss_counts["reset"],
            "weights_ordinary_noise": gauss_counts["ordinary"],
            "weights_amplified_noise": gauss_counts["amplified"],
        }

    def _regrama_network_surgery(
        self,
        network: nn.Module,
        per_neuron_list: list[Any] | None,
        counts: dict[str, int],
    ) -> None:
        """Apply the dormant-neuron surgery to every measured layer of a network."""
        self._warn_if_recurrent(network)

        scores = capture_per_neuron_scores(network, per_neuron_list)
        if not scores:
            return

        encoder = getattr(network, "encoder", None)
        head = getattr(network, "head_net", None)
        cnn_dims = self._cnn_dims_by_module(encoder)

        for act_module, per_neuron in scores:
            producer, norm, next_layers, _is_encoder = self._resolve_producer_and_next(
                act_module, encoder, head
            )
            if (
                producer is None
                or not next_layers
                or not self._owns_trainable_weight(producer)
            ):
                continue

            # Keyed on the producer, not the network: a nested sub-encoder's conv
            # stack has its own flattened layout (see :meth:`_cnn_dims_by_module`).
            cnn_channels, cnn_spatial = cnn_dims.get(id(producer), (None, None))
            self._apply_reborn_to_layer(
                producer,
                next_layers,
                cnn_channels,
                cnn_spatial,
                per_neuron,
                counts,
                norm=norm,
            )

    def _apply_reborn_to_layer(
        self,
        producer: nn.Module,
        next_layers: list[nn.Module],
        cnn_channels: int | None,
        cnn_spatial: int | None,
        per_neuron: torch.Tensor,
        counts: dict[str, int],
        norm: nn.Module | None = None,
    ) -> None:
        """Perform the dormant-neuron surgery on the neurons of one producing layer.

        Keeps the ReBorn name because the *split* it performs when a neuron scores
        above ``overact_beta`` is Qin et al.'s operator. That half is unreachable
        from a manifest -- ``overact_beta`` defaults to ``inf``, leaving every
        dormant neuron to the Xavier reset that is ReGraMa.

        *next_layers* holds every layer that consumes the producer's neurons --
        more than one when the neurons feed parallel streams, as a duelling
        Q-network's latent feeds both the value and advantage heads. Each
        consumer's columns are rescaled by the same factor, so whatever the split
        preserves, it preserves for all of them simultaneously.

        **What the split does and does not preserve.** The scaling below always
        redistributes the *over-active* neuron's own contribution exactly across its
        copies (the alpha softmax sums to one). Whether the *network's* output is
        unchanged additionally requires a positively homogeneous activation,
        ``f(beta * z) == beta * f(z)`` -- true for ReLU, false for Tanh/ELU/Sigmoid.
        That is not checked and not true in general, so this is a *mutation* rather
        than a guaranteed-safe rewrite: outside that condition the child's outputs
        differ from its parent's, and the tournament is what selects against a bad
        draw. A second condition is inherent to gradient-based detection: the
        recycled neurons' own prior contributions are discarded, which is free only
        when they were also inactive -- guaranteed under the activation-based
        dormancy the split was designed for (Sokar et al.), not under the GraMa
        gradient scoring used here, where a neuron can be frozen yet highly active.

        **Under a normalisation** (*norm*, applied to these neurons between the
        producer and the activation -- the default for every evolvable MLP) the
        incoming ``beta`` is divided straight back out, which would leave the
        outgoing ``alpha / beta`` over-correcting. So the split instead pins
        ``beta = 1`` and copies the over-active neuron's per-neuron affine (and any
        running statistics) onto its partners: the copies then reproduce the
        parent's activation exactly and ``alpha`` alone carries the split. Exactness
        is still out of reach for a LayerNorm, whose mean/variance are pooled over
        the whole feature dimension, so replacing a dormant neuron's pre-activation
        perturbs every other neuron by ``O(m / H)``; a BatchNorm normalises
        per-channel and is unaffected.

        **Unclaimed dormant neurons are deliberately not function-preserving.**
        They are Xavier-reset and given a fresh outgoing column of norm
        ``regrama_out_scale * (the consumer's live column scale)`` rather than the
        zero column ReDo prescribes -- under gradient scoring a zero column leaves
        the neuron both unscorable and unlearnable. See :meth:`_revived_out_block`
        for why, and set ``regrama_out_scale=0.0`` to recover the zeroed behaviour.
        Their normalisation state is reset to the identity ``(1, 0)`` for the same
        reason the outgoing column is re-seeded rather than zeroed: a revived neuron
        that inherits the decayed gain of the unit it replaced is re-suppressed
        before it can learn anything.

        :param norm: The normalisation layer applied to *producer*'s outputs before
            the activation, or ``None`` when nothing normalises between them.
        :type norm: torch.nn.Module | None
        """
        prod_w = self._weight_param(producer).data
        prod_b = self._bias_param(producer)
        prod_b = prod_b.data if prod_b is not None else None

        # A NoisyLinear's realised weight is ``mu + sigma * epsilon``, so its noise
        # scale is a second, parallel set of per-neuron rows and columns. Every
        # *scaling* below applies to it identically -- rewriting mu alone would
        # leave each copy of a split neuron injecting a full-magnitude independent
        # perturbation instead of sharing the parent's, inflating the noise on that
        # path by sqrt(m). Only the *reset* of a revived neuron differs, since a
        # fresh unit wants the layer's initial noise scale rather than the one it
        # inherited (see below).
        prod_sigma_w, prod_sigma_b = self._noise_params(producer)
        prod_rows = [t for t in (prod_w, prod_sigma_w) if t is not None]
        prod_biases = [t for t in (prod_b, prod_sigma_b) if t is not None]

        # Pair each consumer's weight tensor with its column stride: a conv ->
        # flatten -> dense boundary spends ``cnn_spatial`` adjacent columns per
        # feature map, every other boundary exactly one.
        prod_neurons = prod_w.shape[0]
        consumers: list[tuple[torch.Tensor, int | None]] = []
        for next_layer in next_layers:
            kind = self._boundary_kind(producer, next_layer)
            if kind is None:
                continue
            next_w = self._weight_param(next_layer).data
            if kind == "conv_dense":
                if cnn_spatial is None or cnn_channels is None:
                    # Every conv stack the evolvable encoders build reports its
                    # pre-flatten shape, so this is an unrecognised architecture
                    # rather than an expected skip -- log it instead of silently
                    # leaving the layer unrecycled.
                    logger.debug(
                        "ReGraMa: no flattened column layout for %s -> %s; "
                        "leaving the layer unrecycled.",
                        type(producer).__name__,
                        type(next_layer).__name__,
                    )
                    continue
                stride = cnn_spatial
            else:
                stride = 1

            # A consumer must spend its columns on *these* neurons and nothing
            # else: one column block each, no others interleaved. Anything failing
            # that is not this producer's consumer -- a nested sub-encoder's
            # features, say, are only a slice of ``EvolvableMultiInput``'s fusion
            # input -- and recycling against its columns would rewrite weights
            # belonging to other neurons. Skip it rather than corrupt it.
            if next_w.shape[1] != prod_neurons * stride:
                continue

            consumers.append((next_w, cnn_spatial if kind == "conv_dense" else None))

            # The consumer's own noise columns ride along under the same stride.
            next_sigma_w, _next_sigma_b = self._noise_params(next_layer)
            if next_sigma_w is not None and next_sigma_w.shape == next_w.shape:
                consumers.append(
                    (next_sigma_w, cnn_spatial if kind == "conv_dense" else None)
                )

        if not consumers:
            return

        # Normalised per-neuron scores (guarding NaN and a dead layer), mirroring
        # ``_count_dormant`` in the dormant-neuron diagnostic.
        scores = torch.nan_to_num(per_neuron.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        mean = float(scores.mean()) if scores.numel() else 0.0
        if mean <= 0.0:
            dormant_idx = list(range(scores.numel()))
            overactive_idx: list[int] = []
        else:
            normalised = scores / mean
            dormant_idx = (
                torch.nonzero(normalised <= self.dormant_tau).flatten().tolist()
            )
            overactive_idx = (
                torch.nonzero(normalised >= self.overact_beta).flatten().tolist()
            )

        counts["dormant"] += len(dormant_idx)
        counts["overactive"] += len(overactive_idx)
        if not dormant_idx:
            return

        def get_out(n: int) -> list[torch.Tensor]:
            return [
                w[:, n * s : (n + 1) * s].clone() if s else w[:, n].clone()
                for w, s in consumers
            ]

        def set_out(n: int, values: list[torch.Tensor]) -> None:
            for (w, s), value in zip(consumers, values, strict=True):
                if s:
                    w[:, n * s : (n + 1) * s] = value
                else:
                    w[:, n] = value

        mag_limit = 1_000_000

        # Reproducibly shuffle the dormant pool and claim without replacement.
        pool = list(self.rng.permutation(np.array(dormant_idx, dtype=np.int64)))
        ptr = 0
        claimed: set[int] = set()

        for x in sorted(overactive_idx):
            m_target = int(self.rng.integers(2, 6))
            take = min(m_target, len(pool) - ptr)
            if take <= 0:
                break
            partners = [int(p) for p in pool[ptr : ptr + take]]
            ptr += take
            claimed.update(partners)

            # A normalisation between the producer and the activation divides the
            # incoming beta back out, so varying it would only break the outgoing
            # alpha / beta compensation. Pin it to 1 and let alpha carry the split.
            betas = (
                [1.0] * (take + 1)
                if norm is not None
                else [float(b) for b in self.rng.uniform(0.5, 1.5, size=take + 1)]
            )
            alpha = self._softmax(self.rng.standard_normal(take + 1))

            w_in_x = [t[x].clone() for t in prod_rows]
            b_x = [t[x].clone() for t in prod_biases]
            w_out_x = get_out(x)

            # Neuron split (net2net widening). Each copy j scales its incoming
            # weights/bias by beta_j, so for a positively homogeneous activation
            # (ReLU) -- and only then -- its activation becomes beta_j * h_x.
            # Scaling its outgoing weights by (alpha_j / beta_j) makes copy j's
            # contribution alpha_j * w_out_x * h_x; since alpha is a softmax
            # (sum_j alpha_j == 1), the copies together reproduce w_out_x * h_x.
            # Over-active neuron x keeps a scaled copy of itself.
            for t, row in zip(prod_rows, w_in_x, strict=True):
                t[x] = betas[0] * row
            for t, bias in zip(prod_biases, b_x, strict=True):
                t[x] = betas[0] * bias
            set_out(x, [(alpha[0] / betas[0]) * w for w in w_out_x])

            # Each claimed dormant neuron is reborn as a scaled copy of x.
            for k, i in enumerate(partners):
                beta_i = betas[k + 1]
                alpha_i = alpha[k + 1]
                for t, row in zip(prod_rows, w_in_x, strict=True):
                    t[i] = beta_i * row
                for t, bias in zip(prod_biases, b_x, strict=True):
                    t[i] = beta_i * bias
                set_out(i, [(alpha_i / beta_i) * w for w in w_out_x])
                # Under a norm the copy is only equivalent to its parent if it is
                # normalised identically, so its affine (and running statistics)
                # come from x rather than from the unit it replaced.
                self._copy_norm_state(norm, dst=i, src=x, neurons=prod_neurons)

            counts["reborn"] += take

        # Unclaimed dormant neurons: Xavier-reset incoming, re-seed outgoing at
        # ``regrama_out_scale`` times each consumer's live column scale. The
        # reference pool excludes every neuron this pass rewrites, so a layer that
        # has been recycled repeatedly measures itself against units at their
        # trained scale rather than against its own shrinking leftovers.
        rewritten = set(dormant_idx) | set(overactive_idx)
        keep = [n for n in range(prod_neurons) if n not in rewritten]
        out_scales = [
            self.regrama_out_scale * self._live_column_scale(w, s or 1, keep)
            for w, s in consumers
        ]

        # A revived neuron is a fresh unit, so its noise scale is the one the layer
        # would have been initialised with -- not the collapsed (or inflated) one it
        # inherits from the unit it replaces.
        noise_init = self._noise_init_scales(producer)

        for i in dormant_idx:
            if i in claimed:
                continue
            self._xavier_reset_row(prod_w, i)
            if prod_b is not None:
                prod_b[i] = 0.0
            if noise_init is not None:
                weight_fill, bias_fill = noise_init
                if prod_sigma_w is not None:
                    prod_sigma_w[i] = weight_fill
                if prod_sigma_b is not None:
                    prod_sigma_b[i] = bias_fill
            set_out(
                i,
                [
                    self._revived_out_block(block, scale)
                    for block, scale in zip(get_out(i), out_scales, strict=True)
                ],
            )
            self._reset_norm_state(norm, index=i, neurons=prod_neurons)
            counts["xavier"] += 1

        # Defensive clamp + NaN scrub so the surgery never introduces / propagates NaN.
        for t in (*prod_rows, *prod_biases):
            t.clamp_(-mag_limit, mag_limit).nan_to_num_()
        for next_w, _stride in consumers:
            next_w.clamp_(-mag_limit, mag_limit).nan_to_num_()

    # ---------------------- Neuron-surgery helpers ------------------------ #
    @staticmethod
    def _noise_params(
        module: nn.Module,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return a noisy layer's ``(weight_sigma, bias_sigma)`` data tensors.

        ``(None, None)`` for an ordinary layer, so callers can treat the noise
        scale as an optional second set of weights rather than branching on type.

        :param module: The layer to inspect.
        :type module: torch.nn.Module
        :return: The noise-scale tensors, each ``None`` when absent.
        :rtype: tuple[torch.Tensor | None, torch.Tensor | None]
        """
        weight_sigma = getattr(module, "weight_sigma", None)
        bias_sigma = getattr(module, "bias_sigma", None)
        return (
            weight_sigma.data if weight_sigma is not None else None,
            bias_sigma.data if bias_sigma is not None else None,
        )

    @staticmethod
    def _noise_init_scales(module: nn.Module) -> tuple[float, float] | None:
        """Return the ``(weight, bias)`` noise scales a fresh unit starts from.

        Mirrors :meth:`NoisyLinear.reset_parameters
        <agilerl.modules.custom_components.NoisyLinear.reset_parameters>`, which
        fills each sigma with ``std_init`` divided by the square root of the
        corresponding fan.

        :param module: The producing layer whose neuron is being revived.
        :type module: torch.nn.Module
        :return: The two fill values, or ``None`` for a layer that carries no noise.
        :rtype: tuple[float, float] | None
        """
        weight_sigma = getattr(module, "weight_sigma", None)
        if weight_sigma is None or weight_sigma.dim() < 2:
            return None
        std_init = float(getattr(module, "std_init", 0.5))
        fan_out, fan_in = weight_sigma.shape[0], weight_sigma.shape[1]
        return (
            std_init / float(np.sqrt(fan_in)),
            std_init / float(np.sqrt(fan_out)),
        )

    @staticmethod
    def _norm_state_tensors(
        norm: nn.Module | None, neurons: int
    ) -> list[tuple[torch.Tensor, float]]:
        """Return a normalisation's per-neuron state as ``(tensor, identity)`` pairs.

        Covers the affine gain/shift and, for the batch norms, the running
        statistics -- every tensor holding one entry per neuron, paired with the
        value that makes that entry a no-op. Tensors whose length does not match
        the producing layer are dropped: a norm over a different axis does not
        index by neuron, and writing into it would corrupt unrelated state.

        :param norm: The normalisation layer, or ``None``.
        :type norm: torch.nn.Module | None
        :param neurons: Number of neurons the producing layer emits.
        :type neurons: int
        :return: The per-neuron tensors and their neutral values.
        :rtype: list[tuple[torch.Tensor, float]]
        """
        if norm is None:
            return []
        candidates = (
            (getattr(norm, "weight", None), 1.0),
            (getattr(norm, "bias", None), 0.0),
            (getattr(norm, "running_mean", None), 0.0),
            (getattr(norm, "running_var", None), 1.0),
        )
        return [
            (tensor.data, identity)
            for tensor, identity in candidates
            if tensor is not None and tensor.dim() == 1 and tensor.shape[0] == neurons
        ]

    @staticmethod
    def _copy_norm_state(
        norm: nn.Module | None, dst: int, src: int, neurons: int
    ) -> None:
        """Copy one neuron's normalisation state onto another's."""
        for tensor, _identity in Mutations._norm_state_tensors(norm, neurons):
            tensor[dst] = tensor[src]

    @staticmethod
    def _reset_norm_state(norm: nn.Module | None, index: int, neurons: int) -> None:
        """Reset one neuron's normalisation state to the identity transform."""
        for tensor, identity in Mutations._norm_state_tensors(norm, neurons):
            tensor[index] = identity

    def _warn_if_recurrent(self, network: nn.Module) -> None:
        """Warn once that a recurrent core lies outside what the surgery can reach.

        The surgery needs two things per measured layer: an activation sub-module
        whose gradient scores its neurons, and a weight matrix whose rows are one
        neuron's incoming weights. A fused recurrent module offers neither -- its
        gate non-linearities are internal to the kernel, and one hidden unit owns
        four interleaved row blocks across ``weight_ih`` *and* ``weight_hh`` plus a
        column of the recurrent self-connection. So a recurrent network is recycled
        only from its output projection onward, and the counts the operator reports
        say nothing about the encoder.

        Warning is not cosmetic: without it the mutation is recorded as an ordinary
        ``param_reborn`` and is indistinguishable from one that recycled the whole
        network.

        :param network: The evaluation network about to undergo surgery.
        :type network: torch.nn.Module
        """
        if self._warned_recurrent:
            return
        if not any(isinstance(m, nn.RNNBase) for _name, m in network.named_modules()):
            return

        self._warned_recurrent = True
        warnings.warn(
            "ReGraMa does not recycle the recurrent core of a recurrent encoder "
            "(nn.RNN / nn.LSTM / nn.GRU): its gate non-linearities are fused, so "
            "no per-neuron gradient is captured for them, and its hidden units do "
            "not own contiguous weight rows. Only the layers after it are "
            "recycled, so the reported recycling counts cover the projection and "
            "head alone.",
            stacklevel=2,
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax over a 1-D numpy array."""
        e = np.exp(x - np.max(x))
        return e / e.sum()

    @staticmethod
    def _live_column_scale(weight: torch.Tensor, stride: int, keep: list[int]) -> float:
        """Median outgoing-column norm of a consumer over the neurons in *keep*.

        One "column" is everything a single producer neuron owns in the consumer,
        which is a different slice per boundary kind: a whole ``(out_c, *kernel)``
        filter when the consumer is a convolution, the ``stride`` adjacent columns
        of a conv -> flatten -> dense boundary, and a single column otherwise. The
        neuron axis is therefore dimension 1 of a conv weight but a *strided* view
        of dimension 1 of a dense one -- a single reshape cannot serve both, and
        reshaping a 4-D weight as if its columns were 1-wide folds the kernel into
        the neuron axis, so ``keep`` indexes kernel positions of the first few
        channels rather than channels. That axis is never too short to index, so
        the miscount is silent: it under-reported by the kernel's linear size (4x
        at the Nature CNN's 4x4 boundary, 3x at its 3x3 one) *and* measured the
        wrong neurons.

        The median is taken over *keep* -- the neurons this recycling pass leaves
        alone -- so the reference tracks the layer's trained scale. Measuring the
        whole layer instead would let a repeatedly-recycled layer shrink without
        bound, each generation sizing its revivals against the previous one's.

        Falls back to the median over every neuron when *keep* is empty or has
        collapsed to zero (a layer with no gradient anywhere is reported entirely
        dormant, leaving no live reference), and finally to the norm a
        Xavier-initialised block would have, so a revival is never silently
        zero-scaled.

        :param weight: The consumer layer's weight tensor: ``(out, neurons * stride)``
            when dense, ``(out_c, neurons, *kernel)`` when convolutional.
        :param stride: Columns per producer neuron; ignored for a conv consumer,
            whose neuron axis is already dimension 1.
        :param keep: Producer-neuron indices to measure.
        :return: A strictly positive column-norm reference.
        """
        if weight.dim() > 2:  # conv consumer: one filter per producer neuron
            blocks = weight.reshape(weight.shape[0], weight.shape[1], -1)
            # Conv fans count the receptive field on both sides.
            fan_out = blocks.shape[0] * blocks.shape[2]
        else:
            blocks = weight.reshape(weight.shape[0], -1, stride)
            fan_out = blocks.shape[0]

        def median_norm(idx: list[int] | None) -> float:
            selected = blocks if idx is None else blocks[:, idx, :]
            if selected.shape[1] == 0:
                return 0.0
            norms = selected.pow(2).sum(dim=(0, 2)).sqrt()
            norms = norms[norms.isfinite()]
            return float(norms.median()) if norms.numel() else 0.0

        for candidate in (median_norm(keep) if keep else 0.0, median_norm(None)):
            if candidate > 0.0:
                return candidate

        # Xavier bound over the consumer's own fan-in/fan-out, and the number of
        # entries one producer neuron's block holds -- both read off ``blocks`` so
        # the conv and dense layouts stay in step with the branch above.
        fan_in = blocks.shape[1] * blocks.shape[2]
        block_entries = blocks.shape[0] * blocks.shape[2]
        bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
        # RMS of U(-bound, bound) is bound / sqrt(3).
        return bound / np.sqrt(3.0) * float(np.sqrt(block_entries))

    def _revived_out_block(self, template: torch.Tensor, scale: float) -> torch.Tensor:
        """Draw the outgoing weights a Xavier-reset neuron is revived with.

        Returns a random direction scaled so the block's norm is *scale*.

        The original recipe (Sokar et al.'s ReDo, which ReGraMa inherits) zeroes
        these weights: it costs nothing under an *activation*-based dormancy
        metric, and it makes the reset exactly function-preserving. Under the
        GraMa *gradient* scoring used here it is self-defeating, because
        ``grad(z_i) = (sum_j W_out[j,i] * delta_j) * act'(z_i)`` is identically zero
        when the outgoing column is -- so the revived neuron is scored as maximally
        dormant, and, worse, ``grad(W_in[i,:]) = grad(z_i) * x`` is zero too, which
        freezes its *incoming* weights until the outgoing ones bootstrap
        themselves off zero. A small non-zero column restores both gradients at
        the cost of exact function preservation; ``regrama_out_scale`` trades the
        two off and ``0.0`` recovers the original behaviour exactly (no RNG is
        consumed on that path, so seeded ablations stay comparable).

        :param template: The neuron's current outgoing block, for shape/dtype/device.
        :param scale: Target L2 norm of the returned block.
        :return: The replacement outgoing block.
        """
        if scale <= 0.0:
            return torch.zeros_like(template)

        # ``self.rng`` rather than torch's global RNG, so the draw is reproducible
        # and thread-count invariant like ``_xavier_reset_row``.
        sampled = self.rng.standard_normal(size=tuple(template.shape))
        block = torch.as_tensor(sampled, dtype=template.dtype, device=template.device)
        norm = float(block.norm())
        if norm <= 0.0:  # astronomically unlikely; fall back to the zero column
            return torch.zeros_like(template)
        return block * (scale / norm)

    @staticmethod
    def _is_weight_layer(module: nn.Module) -> bool:
        """Whether *module* carries recyclable weights (Linear / Conv / Noisy)."""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            return True
        return hasattr(module, "weight_mu") and hasattr(module, "bias_mu")

    @staticmethod
    def _weight_param(module: nn.Module) -> nn.Parameter:
        """Return the (mean) weight parameter of a weight layer."""
        if hasattr(module, "weight_mu"):
            return module.weight_mu
        return module.weight

    @staticmethod
    def _bias_param(module: nn.Module) -> nn.Parameter | None:
        """Return the (mean) bias parameter of a weight layer, if any."""
        if hasattr(module, "bias_mu"):
            return module.bias_mu
        return getattr(module, "bias", None)

    @staticmethod
    def _owns_trainable_weight(module: nn.Module) -> bool:
        """Whether *module* owns the weights the surgery would rewrite.

        :func:`~agilerl.utils.algo_utils.share_encoder_parameters` pins a non-policy
        network's encoder to detached, non-leaf clones of the policy encoder's
        parameters, and PPO re-runs that pinning from a mutation hook -- so writing a
        recycled neuron's incoming weights there is discarded moments later, while
        the matching outgoing rewrite in *that* network's head survives, leaving the
        head compensating a neuron split that no longer exists. Skipping a borrowed
        producer keeps the two sides consistent.

        A real training run never reaches that state: the clones carry no gradient,
        so their layers are captured as ``None`` and filtered out before the surgery
        sees them. The check makes an invariant that currently holds by accident
        explicit, and is what stops an injected snapshot from breaking it.

        :param module: The producing layer whose neurons would be recycled.
        :type module: torch.nn.Module
        :return: ``True`` if the layer's weights are its own, trainable parameters.
        :rtype: bool
        """
        weight = Mutations._weight_param(module)
        return isinstance(weight, nn.Parameter) and weight.requires_grad

    @staticmethod
    def _unwrap_module(module: nn.Module | None) -> nn.Module | None:
        """Strip :class:`EvolvableWrapper`/:class:`EvolvableDistribution` layers.

        PPO/IPPO expose ``head_net`` as an ``EvolvableDistribution`` whose real MLP
        sits behind ``.wrapped``; the surgery has to reach that inner module.
        """
        seen: set[int] = set()
        while module is not None and id(module) not in seen:
            seen.add(id(module))
            wrapped = getattr(module, "wrapped", None)
            if wrapped is None:
                break
            module = wrapped
        return module

    @staticmethod
    def _first_weight_layer(module: nn.Module | None) -> nn.Module | None:
        """Return the first weight-bearing layer inside *module*, in forward order."""
        if module is None:
            return None
        for _name, child in module.named_modules():
            if Mutations._is_weight_layer(child):
                return child
        return None

    @staticmethod
    def _head_entry_layers(head: nn.Module | None) -> list[nn.Module]:
        """Return the first weight layer of every parallel stream in *head*.

        A latent neuron's outgoing weights live in whatever the head feeds it to --
        one layer for a plain MLP head, but *two* for a duelling Q-network, whose
        value and advantage streams are sibling sub-networks that both consume the
        full latent.
        """
        head = Mutations._unwrap_module(head)
        if head is None:
            return []
        children = list(head.children())
        # A head whose own children are layers is a single flat stream.
        if any(Mutations._is_weight_layer(child) for child in children):
            first = Mutations._first_weight_layer(head)
            return [first] if first is not None else []
        entries = [Mutations._first_weight_layer(child) for child in children]
        return [entry for entry in entries if entry is not None]

    def _resolve_producer_and_next(
        self,
        act_module: nn.Module,
        encoder: nn.Module | None,
        head: nn.Module | None,
    ) -> tuple[nn.Module | None, nn.Module | None, list[nn.Module], bool]:
        """Find the layer that produced *act_module*'s neurons and its consumers.

        The producing layer's weight *rows* (and bias) are the neurons' incoming
        weights; each consuming layer's *columns* are their outgoing weights.

        The search walks ``named_modules()`` rather than one flat ``nn.Sequential``,
        and restricts the look-behind/look-ahead to the activation's own parent
        container. Both parts are load-bearing: ``EvolvableMultiInput`` holds its
        sub-networks in a ``ModuleDict`` with a bare ``final_dense``/``output`` tail
        and so has no single sequential to unwrap, while a duelling head's two
        streams are siblings -- scanning past the parent would pair a value-stream
        activation with an advantage-stream layer.

        An activation with no consumer inside its own container is resolved outward
        rather than assumed to be the latent: an encoder can end several streams
        before its real output (again ``EvolvableMultiInput``, whose sub-encoders
        all feed one fusion layer), and only the stream that nothing else in the
        encoder follows is the one the head consumes.

        Any normalisation applied between the producer and the activation is
        returned alongside them: it holds per-neuron state of its own, so the
        surgery has to carry it across a split and reset it on a revival. It is
        tracked as "the last norm seen since the running producer" and cleared
        whenever a later weight layer takes over as producer, which is what
        distinguishes an evolvable MLP's ``linear -> layer_norm -> activation`` from
        a SimBa block's ``layer_norm -> linear -> activation`` -- the latter
        normalises the block's *input* and leaves these neurons alone.

        :param act_module: The measured activation whose neurons are being recycled.
        :param encoder: The network's encoder, if any.
        :param head: The network's head, if any.
        :return: ``(producer, norm, consumers, is_encoder)``; ``(None, None, [],
            False)`` when the activation cannot be located, so the caller skips it.
        """
        for root, is_encoder in ((encoder, True), (head, False)):
            search_root = self._unwrap_module(root)
            if search_root is None:
                continue

            ordered = list(search_root.named_modules())
            name = next((n for n, m in ordered if m is act_module), None)
            if name is None:
                continue

            parent = name.rpartition(".")[0]
            prefix = f"{parent}." if parent else ""

            # The activation's outermost container: layers inside it are either its
            # own stream or a sibling stream, never something it feeds into.
            container = name.split(".")[0]

            producer: nn.Module | None = None
            norm: nn.Module | None = None
            consumers: list[nn.Module] = []
            enclosing: nn.Module | None = None
            passed = False
            for other_name, other in ordered:
                if other is act_module:
                    passed = True
                    continue
                in_stream = other_name.startswith(prefix)
                if not passed and in_stream and self._is_norm_layer(other):
                    norm = other  # applies to the running producer's outputs
                    continue
                if not self._is_weight_layer(other):
                    continue
                if not passed:
                    if in_stream:
                        producer = other  # keep the nearest one behind
                        norm = None  # anything seen so far normalised its input
                elif in_stream:
                    if not consumers:
                        consumers = [other]  # the nearest one ahead ends the search
                elif enclosing is None and not other_name.startswith(f"{container}."):
                    enclosing = other

            if not consumers and is_encoder:
                # Nothing consumes it inside its own container, so it is either the
                # encoder's terminal activation -- the latent, consumed by every
                # stream of the head -- or the tail of a *nested* sub-encoder, whose
                # neurons the encoder's own fusion layer consumes further on
                # (``EvolvableMultiInput``'s ``final_dense``). Treating the second
                # case as the first would recycle against the head's unrelated
                # columns, so a later layer inside the encoder always wins.
                consumers = (
                    [enclosing]
                    if enclosing is not None
                    else self._head_entry_layers(head)
                )

            return producer, norm, consumers, is_encoder

        return None, None, [], False

    @staticmethod
    def _is_norm_layer(module: nn.Module) -> bool:
        """Whether *module* normalises its input without remapping its neurons.

        Recognised by type rather than by the presence of a ``weight``, which would
        also match every projecting layer.

        :param module: The sub-module to classify.
        :type module: torch.nn.Module
        :return: ``True`` for a normalisation layer.
        :rtype: bool
        """
        return isinstance(module, _NORM_LAYER_TYPES)

    @staticmethod
    def _boundary_kind(producer: nn.Module, next_layer: nn.Module) -> str | None:
        """Classify the (producer, next_layer) pair for outgoing-weight indexing."""
        prod_conv = isinstance(producer, (nn.Conv2d, nn.Conv3d))
        next_conv = isinstance(next_layer, (nn.Conv2d, nn.Conv3d))
        if prod_conv and next_conv:
            return "conv_conv"
        if prod_conv and not next_conv:
            return "conv_dense"
        if not prod_conv and not next_conv:
            return "dense_dense"
        return None  # a dense layer feeding a conv layer never occurs here

    @staticmethod
    def _cnn_dims_by_module(
        encoder: nn.Module | None,
    ) -> dict[int, tuple[int, int]]:
        """Map each sub-module to the ``(channels, spatial)`` dims of its owning CNN.

        A conv -> flatten -> dense consumer spends ``spatial`` adjacent columns per
        feature map, and that layout is a property of the ``EvolvableCNN`` owning the
        conv stack -- which is the encoder itself for an image observation, but a
        ``feature_net`` entry under ``EvolvableMultiInput``. Reading ``cnn_output_size``
        off the encoder alone therefore resolves nothing for a dict/tuple observation
        and every nested CNN's last conv layer is dropped from the surgery, silently.
        Indexing by producer instead keeps both cases on the same path.

        ``named_modules`` is outer-first, so a nested CNN's entry overwrites the one
        its parent would have contributed.

        :param encoder: The network's encoder, if any.
        :type encoder: torch.nn.Module | None
        :return: ``{id(sub_module): (channels, spatial)}`` for every CNN descendant.
        :rtype: dict[int, tuple[int, int]]
        """
        dims: dict[int, tuple[int, int]] = {}
        if encoder is None:
            return dims

        for _name, sub in encoder.named_modules():
            shape = getattr(sub, "cnn_output_size", None)
            if shape is None or len(shape) < 3:
                continue
            spatial = 1
            for dim in shape[2:]:
                spatial *= int(dim)
            for _child_name, child in sub.named_modules():
                dims[id(child)] = (int(shape[1]), spatial)

        return dims

    def _xavier_reset_row(self, weight: torch.Tensor, index: int) -> None:
        """Xavier-uniform reset of one output neuron's incoming weights in place.

        Uses ``self.rng`` so the reset is reproducible and thread-count invariant.
        Handles both a Linear weight row and a conv filter slice.
        """
        row = weight[index]
        fan_out = weight.shape[0]
        if weight.dim() == 2:  # Linear: (out_features, in_features)
            fan_in = weight.shape[1]
        else:  # Conv: (out_channels, in_channels, *kernel)
            receptive = 1
            for dim in weight.shape[2:]:
                receptive *= int(dim)
            fan_in = int(weight.shape[1]) * receptive
            fan_out = fan_out * receptive
        bound = float(np.sqrt(6.0 / (fan_in + fan_out)))
        sampled = self.rng.uniform(-bound, bound, size=tuple(row.shape))
        weight[index] = torch.as_tensor(
            sampled, dtype=weight.dtype, device=weight.device
        )

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
        include_reset: bool = True,
        include_amplified: bool = True,
    ) -> EvolvableModule:
        """Return network with mutated weights using a Gaussian distribution.

        Each selected weight falls into one of three bands: an amplified ("super")
        perturbation, a random reset, or ordinary noise. ``include_reset`` and
        ``include_amplified`` switch the first two off independently.

        A switched-off band is *skipped*, not reassigned: its weights keep their
        trained values and the remaining bands do not grow to absorb its probability
        mass. The *selection* is unaffected either way -- which keys, indices and
        bands are drawn comes from ``self.rng``, so every setting picks exactly the
        same weights and assigns them to the same three bands. The surviving bands'
        **values** do shift, though: they come from :func:`torch.normal`, which draws
        on the global torch RNG, so skipping a band's draw advances that stream
        differently for everything after it. Two runs sharing a seed therefore agree
        on *where* the noise lands but not on its magnitude -- this is a band being
        dropped, not a strict subset of the same perturbation.

        :param network: Neural network to mutate.
        :type network: EvolvableModule
        :param counts: Optional dict accumulating the number of weights mutated by
            category (``"reset"``, ``"ordinary"``, ``"amplified"``). Updated in place.
        :type counts: dict[str, int] | None
        :param include_reset: When ``False``, the random-reset band is skipped, so no
            weight is replaced by a fresh ``N(0, 1)`` draw.
        :type include_reset: bool
        :param include_amplified: When ``False``, the amplified ("super") noise band
            is skipped.
        :type include_amplified: bool
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
            if include_amplified and mask_super.sum() > 0:
                std_super = (super_mut_strength * current_vals[mask_super]).abs()
                noise_super = torch.normal(
                    mean=torch.zeros_like(std_super),
                    std=std_super,
                )
                new_vals[mask_super] = current_vals[mask_super] + noise_super
                if counts is not None:
                    counts["amplified"] += int(mask_super.sum().item())

            # Reset mutation: completely reset the weight using N(0, 1)
            if include_reset and mask_reset.sum() > 0:
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
