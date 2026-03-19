"""Helper functions for :mod:`agilerl.trainer`.

Each function converts a Pydantic spec into the corresponding runtime
object, keeping the Trainer classes lean and testable.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agilerl.components.replay_buffer import MultiStepReplayBuffer, ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.models.algo import ALGO_REGISTRY, AlgorithmMeta, RLAlgorithmSpec
from agilerl.models.hpo import MutationSpec, TournamentSelectionSpec
from agilerl.models.training import ReplayBufferSpec, TrainingSpec

if TYPE_CHECKING:
    import gymnasium as gym

    from agilerl.algorithms.core import EvolvableAlgorithm
    from agilerl.typing import PopulationType


def resolve_algo_name(algorithm: EvolvableAlgorithm | RLAlgorithmSpec | str) -> str:
    """Extract the canonical algorithm name from *algorithm*.

    :param algorithm: An :class:`EvolvableAlgorithm` instance, an
        :class:`RLAlgorithmSpec` (or subclass), or a plain string name.
    :type algorithm: EvolvableAlgorithm | RLAlgorithmSpec | str
    :returns: The algorithm name as registered in :data:`ALGO_REGISTRY`.
    :rtype: str
    :raises ValueError: If a spec subclass has no matching registry entry.
    """
    if isinstance(algorithm, str):
        return algorithm
    if isinstance(algorithm, RLAlgorithmSpec):
        for name, meta in ALGO_REGISTRY.items():
            if isinstance(algorithm, meta.spec_cls):
                return name
        msg = f"No registry entry for spec type {type(algorithm).__name__}"
        raise ValueError(msg)
    return algorithm.algo


def get_algo_meta(name: str) -> AlgorithmMeta:
    """Look up :class:`AlgorithmMeta` from the registry.

    :param name: Canonical algorithm name (e.g. ``"PPO"``).
    :type name: str
    :returns: The corresponding registry entry.
    :rtype: AlgorithmMeta
    :raises ValueError: If *name* is not in :data:`ALGO_REGISTRY`.
    """
    meta = ALGO_REGISTRY.get(name)
    if meta is None:
        supported = ", ".join(sorted(ALGO_REGISTRY))
        msg = f"No registry entry for algorithm {name!r}. Supported: {supported}"
        raise ValueError(msg)
    return meta


def create_population_from_spec(
    spec: RLAlgorithmSpec,
    algo_meta: AlgorithmMeta,
    env: gym.Env,
    pop_size: int,
    device: str,
) -> PopulationType:
    """Instantiate a population of agents from an :class:`RLAlgorithmSpec`.

    Uses ``spec.model_dump()`` to extract constructor kwargs and lazily
    imports the algorithm class via *algo_meta.algo_path*.

    :param spec: Algorithm hyperparameters.
    :type spec: RLAlgorithmSpec
    :param algo_meta: Registry metadata for the algorithm.
    :type algo_meta: AlgorithmMeta
    :param env: Gymnasium environment (used for observation/action spaces).
    :type env: gym.Env
    :param pop_size: Number of agents to create.
    :type pop_size: int
    :param device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    :type device: str
    :returns: A list of algorithm instances.
    :rtype: PopulationType
    """
    module_path, class_name = algo_meta.algo_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    algo_cls = getattr(mod, class_name)

    kwargs = spec.model_dump(exclude={"hp_config"})
    net_config = kwargs.pop("net_config", None)
    if net_config is not None:
        kwargs["net_config"] = net_config

    kwargs["observation_space"] = env.single_observation_space
    kwargs["action_space"] = env.single_action_space
    kwargs["device"] = device

    if spec.hp_config is not None:
        kwargs["hp_config"] = spec.hp_config

    return [algo_cls(**kwargs, index=i) for i in range(pop_size)]


def build_mutations(spec: MutationSpec | None, device: str) -> Mutations | None:
    """Convert a :class:`MutationSpec` into a :class:`Mutations` instance.

    :param spec: Mutation configuration, or ``None`` to skip mutations.
    :type spec: MutationSpec | None
    :param device: Torch device string.
    :type device: str
    :returns: A configured :class:`Mutations` object, or ``None``.
    :rtype: Mutations | None
    """
    if spec is None:
        return None
    p = spec.probabilities
    return Mutations(
        no_mutation=p.no_mut,
        architecture=p.arch_mut,
        new_layer_prob=p.new_layer,
        parameters=p.params_mut,
        activation=p.act_mut,
        rl_hp=p.rl_hp_mut,
        mutation_sd=spec.mutation_sd,
        rand_seed=spec.rand_seed,
        device=device,
    )


def build_tournament(
    spec: TournamentSelectionSpec | None,
    training: TrainingSpec,
) -> TournamentSelection | None:
    """Convert a :class:`TournamentSelectionSpec` into a :class:`TournamentSelection`.

    :param spec: Tournament configuration, or ``None`` to skip.
    :type spec: TournamentSelectionSpec | None
    :param training: Training spec (provides ``pop_size`` and ``eval_loop``).
    :type training: TrainingSpec
    :returns: A configured :class:`TournamentSelection`, or ``None``.
    :rtype: TournamentSelection | None
    """
    if spec is None:
        return None
    return TournamentSelection(
        tournament_size=spec.tournament_size,
        elitism=spec.elitism,
        population_size=training.pop_size,
        eval_loop=training.eval_loop,
    )


def build_replay_buffer(
    spec: ReplayBufferSpec | None,
    algo_meta: AlgorithmMeta,
    device: str,
) -> ReplayBuffer | None:
    """Convert a :class:`ReplayBufferSpec` into a :class:`ReplayBuffer`.

    If *spec* is ``None`` but the algorithm requires a buffer, a default
    ``ReplayBuffer(max_size=100_000)`` is created automatically.

    :param spec: Buffer configuration, or ``None``.
    :type spec: ReplayBufferSpec | None
    :param algo_meta: Registry metadata (used for ``requires_buffer``).
    :type algo_meta: AlgorithmMeta
    :param device: Torch device string.
    :type device: str
    :returns: A :class:`ReplayBuffer` (or :class:`MultiStepReplayBuffer`),
        or ``None`` for on-policy algorithms.
    :rtype: ReplayBuffer | None
    """
    if spec is None:
        if algo_meta.requires_buffer:
            return ReplayBuffer(max_size=100_000, device=device)
        return None

    if spec.n_step_buffer:
        return MultiStepReplayBuffer(
            max_size=spec.memory_size,
            n_step=spec.n_step_buffer_args.n_step,
            gamma=0.99,
            device=device,
        )
    return ReplayBuffer(max_size=spec.memory_size, device=device)


_TRAINING_FN_CACHE: dict[str, Callable[..., Any]] = {}


def get_training_fn(algo_meta: AlgorithmMeta) -> Callable[..., Any]:
    """Lazily import and return the training function for *algo_meta*.

    Results are cached so repeated calls avoid re-importing.

    :param algo_meta: Registry metadata for the algorithm.
    :type algo_meta: AlgorithmMeta
    :returns: The training loop function (e.g. ``train_on_policy``).
    :rtype: Callable[..., Any]
    """
    fn_name = algo_meta.train_fn_name
    if fn_name in _TRAINING_FN_CACHE:
        return _TRAINING_FN_CACHE[fn_name]

    module_path = f"agilerl.training.{fn_name}"
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    _TRAINING_FN_CACHE[fn_name] = fn
    return fn


def build_train_kwargs(
    *,
    algo_meta: AlgorithmMeta,
    algo_name: str,
    env: gym.Env,
    env_name: str,
    pop: PopulationType,
    training: TrainingSpec,
    tournament: TournamentSelection | None,
    mutations: Mutations | None,
    memory: ReplayBuffer | None,
    swap_channels: bool = False,
    checkpoint: int | None = None,
    checkpoint_path: str | None = None,
    save_elite: bool = False,
    elite_path: str | None = None,
    wb: bool = False,
    verbose: bool = True,
    accelerator: Any | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the keyword-argument dict for a training function call.

    Conditionally includes ``memory``, ``learning_delay``, and
    ``wandb_kwargs`` based on the algorithm's training loop requirements.

    :param algo_meta: Registry metadata for the algorithm.
    :type algo_meta: AlgorithmMeta
    :param algo_name: Canonical algorithm name.
    :type algo_name: str
    :param env: Gymnasium environment instance.
    :type env: gym.Env
    :param env_name: Human-readable environment name for logging.
    :type env_name: str
    :param pop: Population of algorithm instances.
    :type pop: PopulationType
    :param training: Training loop parameters.
    :type training: TrainingSpec
    :param tournament: Tournament selection instance, or ``None``.
    :type tournament: TournamentSelection | None
    :param mutations: Mutations instance, or ``None``.
    :type mutations: Mutations | None
    :param memory: Replay buffer instance, or ``None``.
    :type memory: ReplayBuffer | None
    :param swap_channels: Whether to swap observation channels.
    :type swap_channels: bool
    :param checkpoint: Save a checkpoint every *n* steps (``None`` to disable).
    :type checkpoint: int | None
    :param checkpoint_path: Directory for checkpoint files.
    :type checkpoint_path: str | None
    :param save_elite: Whether to save the elite agent after training.
    :type save_elite: bool
    :param elite_path: Directory for elite agent files.
    :type elite_path: str | None
    :param wb: Enable Weights & Biases logging.
    :type wb: bool
    :param verbose: Print progress during training.
    :type verbose: bool
    :param accelerator: Optional HuggingFace ``Accelerator``.
    :type accelerator: Any | None
    :param wandb_api_key: Weights & Biases API key.
    :type wandb_api_key: str | None
    :param wandb_kwargs: Extra kwargs forwarded to ``wandb.init``.
    :type wandb_kwargs: dict[str, Any] | None
    :returns: Keyword arguments ready to be unpacked into the training function.
    :rtype: dict[str, Any]
    """
    kwargs: dict[str, Any] = {
        "env": env,
        "env_name": env_name,
        "algo": algo_name,
        "pop": pop,
        "swap_channels": swap_channels,
        "max_steps": training.max_steps,
        "evo_steps": training.evo_steps,
        "eval_loop": training.eval_loop,
        "target": training.target_score,
        "tournament": tournament,
        "mutation": mutations,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "save_elite": save_elite,
        "elite_path": elite_path,
        "wb": wb,
        "verbose": verbose,
        "accelerator": accelerator,
        "wandb_api_key": wandb_api_key,
    }

    if algo_meta.requires_buffer:
        kwargs["memory"] = memory
        if algo_meta.train_fn_name in (
            "train_off_policy",
            "train_multi_agent_off_policy",
        ):
            kwargs["learning_delay"] = training.learning_delay

    if algo_meta.train_fn_name in ("train_on_policy", "train_off_policy"):
        kwargs["wandb_kwargs"] = wandb_kwargs

    return kwargs
