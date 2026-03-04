from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agilerl.algorithms.core import (
    EvolvableAlgorithm,
)
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    ReplayBuffer,
)
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_bandits import train_bandits
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy
from agilerl.training.train_off_policy import train_off_policy
from agilerl.training.train_offline import train_offline
from agilerl.training.train_on_policy import train_on_policy

if TYPE_CHECKING:
    import gymnasium as gym

    from agilerl.typing import PopulationType

logger = logging.getLogger(__name__)

_TRAINING_FN_REGISTRY: dict[str, Callable[..., Any]] = {
    "PPO": train_on_policy,
    "DQN": train_off_policy,
    "DDPG": train_off_policy,
    "TD3": train_off_policy,
    "RainbowDQN": train_off_policy,
    "CQN": train_offline,
    "NeuralUCB": train_bandits,
    "NeuralTS": train_bandits,
    "IPPO": train_multi_agent_on_policy,
    "MADDPG": train_multi_agent_off_policy,
    "MATD3": train_multi_agent_off_policy,
}

_OFF_POLICY_ALGOS = frozenset({"DQN", "DDPG", "TD3", "RainbowDQN"})
_BANDIT_ALGOS = frozenset({"NeuralUCB", "NeuralTS"})
_OFFLINE_ALGOS = frozenset({"CQN"})

_TRAINING_LOOP_NAMES: dict[str, str] = {
    "PPO": "train_on_policy",
    "DQN": "train_off_policy",
    "DDPG": "train_off_policy",
    "TD3": "train_off_policy",
    "RainbowDQN": "train_off_policy",
    "CQN": "train_offline",
    "NeuralUCB": "train_bandits",
    "NeuralTS": "train_bandits",
    "IPPO": "train_multi_agent_on_policy",
    "MADDPG": "train_multi_agent_off_policy",
    "MATD3": "train_multi_agent_off_policy",
}


def _resolve_algo_name(algorithm: EvolvableAlgorithm | dict[str, Any] | str) -> str:
    """Extract the algorithm name from an instance, dict, or string."""
    if isinstance(algorithm, str):
        return algorithm
    if isinstance(algorithm, dict):
        name = algorithm.get("name") or algorithm.get("algo")
        if name is None:
            msg = "Algorithm dict must contain a 'name' key."
            raise ValueError(msg)
        return name
    return algorithm.algo


class Trainer(ABC):
    """Abstract base trainer for AgileRL evolutionary training.

    Accepts either instantiated AgileRL objects or plain configuration
    dicts for each component.  Subclasses resolve these into the form
    they need (live objects for :class:`LocalTrainer`, a manifest for
    :class:`ArenaTrainer`).

    :param algorithm: An algorithm instance or a dict describing the
        algorithm (must include ``"name"`` and hyperparameters).
    :param environment: A ``gym.Env`` instance, or an env-name string
        (for Arena).
    :param env_name: Human-readable environment name forwarded to the
        training loop.
    :param mutations: A :class:`Mutations` instance or a config dict.
    :param tournament: A :class:`TournamentSelection` instance or a
        config dict.
    :param replay_buffer: A :class:`ReplayBuffer` (or subclass)
        instance, or a config dict.
    :param hp_config: A :class:`HyperparameterConfig` for RL-HP
        mutation ranges, or a dict mapping param names to range dicts.
    :param network_config: Network architecture config dict.
    :param pop_size: Population size.
    :param max_steps: Total environment steps.
    :param evo_steps: Steps between evolutionary events.
    :param eval_loop: Evaluation episodes per agent per evo step.
    :param learning_delay: Steps before learning begins (off-policy).
    :param target_score: Optional early-stopping fitness target.
    :param swap_channels: Whether to swap observation channels.
    :param device: Torch device string.
    """

    def __init__(
        self,
        algorithm: EvolvableAlgorithm | dict[str, Any] | str,
        environment: gym.Env | str | Any,
        *,
        env_name: str | None = None,
        mutations: Mutations | dict[str, Any] | None = None,
        tournament: TournamentSelection | dict[str, Any] | None = None,
        replay_buffer: ReplayBuffer | dict[str, Any] | None = None,
        hp_config: HyperparameterConfig | dict[str, Any] | None = None,
        network_config: dict[str, Any] | None = None,
        pop_size: int = 4,
        max_steps: int = 1_000_000,
        evo_steps: int = 10_000,
        eval_loop: int = 1,
        learning_delay: int = 0,
        target_score: float | None = None,
        swap_channels: bool = False,
        device: str = "cpu",
    ) -> None:
        self.algorithm = algorithm
        self.environment = environment
        self.env_name = env_name
        self.mutations = mutations
        self.tournament = tournament
        self.replay_buffer = replay_buffer
        self.hp_config = hp_config
        self.network_config = network_config
        self.pop_size = pop_size
        self.max_steps = max_steps
        self.evo_steps = evo_steps
        self.eval_loop = eval_loop
        self.learning_delay = learning_delay
        self.target_score = target_score
        self.swap_channels = swap_channels
        self.device = device

        self._algo_name = _resolve_algo_name(algorithm)

    @abstractmethod
    def train(self) -> Any:
        """Run the training loop.  Return type varies by subclass."""
        ...

    # -- Helpers for normalizing hp_config -----------------------------------

    @staticmethod
    def _build_hp_config(
        raw: dict[str, Any],
    ) -> HyperparameterConfig:
        """Convert a plain dict into a :class:`HyperparameterConfig`.

        Each value should be a dict with ``min``, ``max``, and
        optionally ``grow_factor``, ``shrink_factor``, ``dtype``.
        """
        params: dict[str, RLParameter] = {}
        for name, spec in raw.items():
            if isinstance(spec, RLParameter):
                params[name] = spec
            elif isinstance(spec, dict):
                dtype_val = spec.get("dtype", float)
                if dtype_val == "int" or dtype_val is int:
                    dtype_val = int
                else:
                    dtype_val = float
                params[name] = RLParameter(
                    min=spec["min"],
                    max=spec["max"],
                    grow_factor=spec.get("grow_factor", 1.2),
                    shrink_factor=spec.get("shrink_factor", 0.8),
                    dtype=dtype_val,
                )
            else:
                msg = (
                    f"Expected dict or RLParameter for hp_config[{name!r}], "
                    f"got {type(spec).__name__}"
                )
                raise TypeError(msg)
        return HyperparameterConfig(**params)


class LocalTrainer(Trainer):
    """Trains a population of agents locally using AgileRL training loops.

    Resolves the appropriate training function based on the algorithm
    name and delegates to it.

    :param algorithm: An algorithm instance (one member of the
        population) **or** a config dict for population creation.
    :param population: Pre-built population list.  When provided,
        *algorithm* is used only for name resolution.
    :param environment: A ``gym.Env`` instance.
    :param verbose: Print progress during training.
    :param accelerator: Optional HuggingFace ``Accelerator``.
    :param checkpoint: Save a checkpoint every *n* steps (``None`` to
        disable).
    :param checkpoint_path: Directory for checkpoint files.
    :param wb: Enable Weights & Biases logging.
    :param wandb_api_key: W&B API key.
    :param wandb_kwargs: Extra kwargs forwarded to ``wandb.init``.

    All other keyword arguments are forwarded to :class:`Trainer`.
    """

    def __init__(
        self,
        algorithm: EvolvableAlgorithm | dict[str, Any] | str,
        environment: gym.Env | Any,
        *,
        population: PopulationType | None = None,
        verbose: bool = True,
        accelerator: Any | None = None,
        checkpoint: int | None = None,
        checkpoint_path: str | None = None,
        save_elite: bool = False,
        elite_path: str | None = None,
        wb: bool = False,
        wandb_api_key: str | None = None,
        wandb_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(algorithm, environment, **kwargs)
        self.population = population
        self.verbose = verbose
        self.accelerator = accelerator
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path
        self.save_elite = save_elite
        self.elite_path = elite_path
        self.wb = wb
        self.wandb_api_key = wandb_api_key
        self.wandb_kwargs = wandb_kwargs

    def train(self) -> tuple[PopulationType, list[list[float]]]:
        """Run local evolutionary training and return the final population
        and fitness history.
        """
        pop = self._resolve_population()
        mutations = self._resolve_mutations()
        tournament = self._resolve_tournament()
        memory = self._resolve_replay_buffer()
        train_fn = self._resolve_training_fn()

        env_name = self.env_name or str(self.environment)

        common_kwargs: dict[str, Any] = {
            "env": self.environment,
            "env_name": env_name,
            "algo": self._algo_name,
            "pop": pop,
            "swap_channels": self.swap_channels,
            "max_steps": self.max_steps,
            "evo_steps": self.evo_steps,
            "eval_loop": self.eval_loop,
            "target": self.target_score,
            "tournament": tournament,
            "mutation": mutations,
            "checkpoint": self.checkpoint,
            "checkpoint_path": self.checkpoint_path,
            "save_elite": self.save_elite,
            "elite_path": self.elite_path,
            "wb": self.wb,
            "verbose": self.verbose,
            "accelerator": self.accelerator,
            "wandb_api_key": self.wandb_api_key,
        }

        if self._algo_name in _OFF_POLICY_ALGOS:
            if memory is None:
                msg = f"A replay buffer is required for off-policy algorithm {self._algo_name!r}."
                raise ValueError(msg)
            common_kwargs["memory"] = memory
            common_kwargs["learning_delay"] = self.learning_delay
            common_kwargs["wandb_kwargs"] = self.wandb_kwargs

        elif self._algo_name in _BANDIT_ALGOS:
            if memory is None:
                msg = f"A replay buffer is required for bandit algorithm {self._algo_name!r}."
                raise ValueError(msg)
            common_kwargs["memory"] = memory

        elif self._algo_name in _OFFLINE_ALGOS:
            if memory is None:
                msg = f"A replay buffer is required for offline algorithm {self._algo_name!r}."
                raise ValueError(msg)
            common_kwargs["memory"] = memory

        elif self._algo_name == "PPO":
            common_kwargs["wandb_kwargs"] = self.wandb_kwargs

        elif self._algo_name in {"MADDPG", "MATD3"}:
            if memory is None:
                msg = f"A replay buffer is required for multi-agent off-policy algorithm {self._algo_name!r}."
                raise ValueError(msg)
            common_kwargs["memory"] = memory
            common_kwargs["learning_delay"] = self.learning_delay

        return train_fn(**common_kwargs)

    # -- Resolution helpers --------------------------------------------------

    def _resolve_population(self) -> PopulationType:
        if self.population is not None:
            return self.population

        if isinstance(self.algorithm, EvolvableAlgorithm):
            return [self.algorithm.clone(index=i) for i in range(self.pop_size)]

        msg = (
            "Pass either a pre-built population or an algorithm instance "
            "to LocalTrainer.  Config-dict population creation is not "
            "yet supported; use agilerl.utils.utils.create_population "
            "to build the population, then pass it via the 'population' parameter."
        )
        raise TypeError(msg)

    def _resolve_mutations(self) -> Mutations | None:
        if self.mutations is None:
            return None
        if isinstance(self.mutations, Mutations):
            return self.mutations
        if isinstance(self.mutations, dict):
            d = self.mutations
            return Mutations(
                no_mutation=d.get("no_mutation", d.get("no_mut", 0.4)),
                architecture=d.get("architecture", d.get("arch_mut", 0.2)),
                new_layer_prob=d.get("new_layer_prob", d.get("new_layer", 0.2)),
                parameters=d.get("parameters", d.get("params_mut", 0.2)),
                activation=d.get("activation", d.get("act_mut", 0.0)),
                rl_hp=d.get("rl_hp", d.get("rl_hp_mut", 0.2)),
                mutation_sd=d.get("mutation_sd", 0.1),
                rand_seed=d.get("rand_seed"),
                device=self.device,
            )
        msg = f"Expected Mutations or dict, got {type(self.mutations).__name__}"
        raise TypeError(msg)

    def _resolve_tournament(self) -> TournamentSelection | None:
        if self.tournament is None:
            return None
        if isinstance(self.tournament, TournamentSelection):
            return self.tournament
        if isinstance(self.tournament, dict):
            d = self.tournament
            return TournamentSelection(
                tournament_size=d.get("tournament_size", 2),
                elitism=d.get("elitism", True),
                population_size=self.pop_size,
                eval_loop=self.eval_loop,
            )
        msg = f"Expected TournamentSelection or dict, got {type(self.tournament).__name__}"
        raise TypeError(msg)

    def _resolve_replay_buffer(self) -> ReplayBuffer | None:
        if self.replay_buffer is None:
            return None
        if isinstance(self.replay_buffer, ReplayBuffer):
            return self.replay_buffer
        if isinstance(self.replay_buffer, dict):
            d = self.replay_buffer
            max_size = d.get("memory_size", d.get("max_size", 100_000))
            if d.get("n_step_buffer"):
                n_step = d.get("n_step", 3)
                gamma = d.get("gamma", 0.99)
                return MultiStepReplayBuffer(
                    max_size=max_size, n_step=n_step, gamma=gamma, device=self.device
                )
            return ReplayBuffer(max_size=max_size, device=self.device)
        msg = f"Expected ReplayBuffer or dict, got {type(self.replay_buffer).__name__}"
        raise TypeError(msg)

    def _resolve_training_fn(self) -> Callable[..., Any]:
        fn = _TRAINING_FN_REGISTRY.get(self._algo_name)
        if fn is None:
            supported = ", ".join(sorted(_TRAINING_FN_REGISTRY))
            msg = (
                f"No training loop registered for algorithm {self._algo_name!r}. "
                f"Supported: {supported}"
            )
            raise ValueError(msg)
        return fn


class ArenaTrainer(Trainer):
    """Submits evolutionary training jobs to the Arena RLOps platform.

    Mirrors the :class:`LocalTrainer` interface but, instead of running
    training locally, builds an Arena manifest from the provided objects
    and submits it via :class:`~agilerl.arena.client.ArenaClient`.

    :param algorithm: An algorithm instance or config dict.
    :param env_name: Registered Arena environment name.
    :param env_version: Environment version.
    :param client: An authenticated :class:`ArenaClient`.  One is
        created automatically if not provided.
    :param num_envs: Number of parallel environments.
    :param channels_last: Whether observations use channels-last layout.
    :param custom_env: Whether this is a custom (user-uploaded)
        environment.
    :param env_config: Optional environment configuration dict.
    :param env_entrypoint: Entrypoint for custom environments.
    :param reporting_interval: Steps between progress reports.
    :param experience_sharing: Share experience across the population.

    All other keyword arguments are forwarded to :class:`Trainer`.
    """

    def __init__(
        self,
        algorithm: EvolvableAlgorithm | dict[str, Any] | str,
        env_name: str,
        env_version: int | str,
        *,
        client: Any | None = None,
        num_envs: int = 16,
        channels_last: bool = False,
        custom_env: bool = False,
        env_config: dict[str, Any] | None = None,
        env_entrypoint: str | None = None,
        reporting_interval: int = 4096,
        experience_sharing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            algorithm,
            env_name,
            env_name=env_name,
            **kwargs,
        )
        self.env_version = env_version
        self.num_envs = num_envs
        self.channels_last = channels_last
        self.custom_env = custom_env
        self.env_config = env_config
        self.env_entrypoint = env_entrypoint
        self.reporting_interval = reporting_interval
        self.experience_sharing = experience_sharing

        if client is not None:
            self._client = client
        else:
            from agilerl.arena.client import ArenaClient

            self._client = ArenaClient()

    def train(self, *, stream: bool = False) -> dict[str, Any]:
        """Build the manifest and submit the training job to Arena.

        :param stream: If ``True``, stream logs and block until
            completion.
        :returns: Arena API response (includes ``job_id``).
        """
        manifest = self.to_manifest()
        return self._client.submit_training_job(manifest, stream=stream)

    def to_manifest(self) -> Any:
        """Build a :class:`TrainingJobManifest` from the trainer state."""
        from agilerl.arena.models import (
            AlgorithmManifest,
            EnvironmentManifest,
            MutationManifest,
            MutationProbabilities,
            NetworkManifest,
            ReplayBufferManifest,
            RLHPRange,
            TournamentManifest,
            TrainingJobManifest,
            TrainingManifest,
        )

        algo_manifest = self._build_algorithm_manifest(AlgorithmManifest)
        env_manifest = self._build_environment_manifest(EnvironmentManifest)
        mutation_manifest = self._build_mutation_manifest(
            MutationManifest, MutationProbabilities, RLHPRange
        )
        tournament_manifest = self._build_tournament_manifest(TournamentManifest)
        network_manifest = self._build_network_manifest(NetworkManifest)
        buffer_manifest = self._build_replay_buffer_manifest(ReplayBufferManifest)
        training_manifest = self._build_training_manifest(TrainingManifest)

        return TrainingJobManifest(
            algorithm=algo_manifest,
            environment=env_manifest,
            mutation=mutation_manifest,
            network=network_manifest,
            replay_buffer=buffer_manifest,
            tournament_selection=tournament_manifest,
            training=training_manifest,
        )

    # -- Manifest section builders -------------------------------------------

    def _build_algorithm_manifest(self, cls: type) -> Any:
        if isinstance(self.algorithm, dict):
            return cls.from_flat_dict(self.algorithm)

        if isinstance(self.algorithm, str):
            return cls(name=self.algorithm)

        attrs = EvolvableAlgorithm.inspect_attributes(
            self.algorithm, input_args_only=True
        )
        attrs["name"] = self._algo_name
        attrs.pop("index", None)
        attrs.pop("hp_config", None)
        attrs.pop("device", None)
        attrs.pop("accelerator", None)
        attrs.pop("wrap", None)
        attrs.pop("observation_space", None)
        attrs.pop("action_space", None)
        attrs.pop("normalize_images", None)
        attrs.pop("net_config", None)
        attrs.pop("mut", None)
        attrs.pop("actor_network", None)
        attrs.pop("critic_network", None)
        attrs.pop("torch_compiler", None)
        return cls.from_flat_dict(attrs)

    def _build_environment_manifest(self, cls: type) -> Any:
        return cls(
            name=self.env_name,
            version=self.env_version,
            num_envs=self.num_envs,
            channels_last=self.channels_last,
            custom=self.custom_env,
            config=self.env_config,
            entrypoint=self.env_entrypoint,
        )

    def _build_mutation_manifest(
        self, manifest_cls: type, prob_cls: type, range_cls: type
    ) -> Any:
        if self.mutations is None:
            return manifest_cls()

        if isinstance(self.mutations, dict):
            d = self.mutations
            probs = prob_cls(
                no_mut=d.get("no_mutation", d.get("no_mut", 0.4)),
                arch_mut=d.get("architecture", d.get("arch_mut", 0.2)),
                new_layer=d.get("new_layer_prob", d.get("new_layer", 0.2)),
                params_mut=d.get("parameters", d.get("params_mut", 0.2)),
                act_mut=d.get("activation", d.get("act_mut", 0.0)),
                rl_hp_mut=d.get("rl_hp", d.get("rl_hp_mut", 0.2)),
            )
            rl_hp_selection = self._extract_rl_hp_selection(range_cls)
            return manifest_cls(
                probabilities=probs,
                rl_hp_selection=rl_hp_selection,
                mutation_sd=d.get("mutation_sd", 0.1),
                rand_seed=d.get("rand_seed"),
            )

        m: Mutations = self.mutations
        probs = prob_cls(
            no_mut=m.no_mutation,
            arch_mut=m.architecture,
            new_layer=m.new_layer_prob,
            params_mut=m.parameters,
            act_mut=m.activation,
            rl_hp_mut=m.rl_hp,
        )
        rl_hp_selection = self._extract_rl_hp_selection(range_cls)
        return manifest_cls(
            probabilities=probs,
            rl_hp_selection=rl_hp_selection,
            mutation_sd=m.mutation_sd,
            rand_seed=getattr(m, "rand_seed", None),
        )

    def _extract_rl_hp_selection(self, range_cls: type) -> dict[str, Any]:
        """Extract RL hyperparameter mutation ranges from hp_config."""
        selection: dict[str, Any] = {}
        source = self.hp_config

        if source is None and isinstance(self.algorithm, EvolvableAlgorithm):
            source = self.algorithm.registry.hp_config

        if source is None:
            return selection

        if isinstance(source, dict):
            for name, spec in source.items():
                if isinstance(spec, dict):
                    selection[name] = range_cls(
                        min=spec["min"],
                        max=spec["max"],
                        grow_factor=spec.get("grow_factor", 1.2),
                        shrink_factor=spec.get("shrink_factor", 0.8),
                    )
                elif isinstance(spec, RLParameter):
                    selection[name] = range_cls(
                        min=spec.min,
                        max=spec.max,
                        grow_factor=spec.grow_factor,
                        shrink_factor=spec.shrink_factor,
                    )
            return selection

        if isinstance(source, HyperparameterConfig):
            for name, param in source.config.items():
                selection[name] = range_cls(
                    min=param.min,
                    max=param.max,
                    grow_factor=param.grow_factor,
                    shrink_factor=param.shrink_factor,
                )
            return selection

        return selection

    def _build_tournament_manifest(self, cls: type) -> Any:
        if self.tournament is None:
            return cls()

        if isinstance(self.tournament, dict):
            return cls(
                tournament_size=self.tournament.get("tournament_size", 2),
                elitism=self.tournament.get("elitism", True),
            )

        t: TournamentSelection = self.tournament
        return cls(
            tournament_size=t.tournament_size,
            elitism=t.elitism,
        )

    def _build_network_manifest(self, cls: type) -> Any:
        if self.network_config is not None:
            return cls.model_validate(self.network_config)
        return cls()

    def _build_replay_buffer_manifest(self, cls: type) -> Any:
        if self.replay_buffer is None:
            return cls()

        if isinstance(self.replay_buffer, dict):
            return cls.model_validate(self.replay_buffer)

        buf = self.replay_buffer
        is_n_step = isinstance(buf, MultiStepReplayBuffer)
        name = type(buf).__name__
        return cls(
            name=name,
            memory_size=buf.max_size,
            standard_buffer=not is_n_step,
            n_step_buffer=is_n_step,
        )

    def _build_training_manifest(self, cls: type) -> Any:
        loop_name = _TRAINING_LOOP_NAMES.get(self._algo_name, "train_off_policy")
        return cls(
            name=loop_name,
            max_steps=self.max_steps,
            pop_size=self.pop_size,
            evo_steps=self.evo_steps,
            eval_loop=self.eval_loop,
            learning_delay=self.learning_delay,
            reporting_interval=self.reporting_interval,
            experience_sharing=self.experience_sharing,
            target_score=self.target_score,
        )
