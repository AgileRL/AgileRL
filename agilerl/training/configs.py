# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Thematic configs for AgileRL training loops and logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.protocols import SelectionStrategyProtocol
from agilerl.typing import InitHyperparams


@dataclass
class TrainLoopConfig:
    """Step budget, evolution cadence, and evaluation."""

    max_steps: int = 1_000_000
    evo_steps: int = 10_000
    eval_steps: int | None = None
    eval_loop: int = 1
    target: float | None = None


@dataclass
class TrainEvolutionConfig:
    """Tournament selection and mutation for a training run."""

    mut_p: InitHyperparams = None
    selection_strategy: SelectionStrategyProtocol | None = None
    tournament: TournamentSelection | None = None
    mutation: Mutations | None = None


@dataclass
class TrainCheckpointConfig:
    """Population checkpoints and elite export."""

    checkpoint: int | None = None
    checkpoint_path: str | None = None
    overwrite_checkpoints: bool = False
    save_elite: bool = False
    elite_path: str | None = None


@dataclass
class TrainLoggingConfig:
    """W&B, TensorBoard, CSV, and stdout progress."""

    wb: bool = False
    tensorboard: bool = False
    tensorboard_log_dir: str | None = None
    csv: bool = False
    csv_log_dir: str | None = None
    verbose: bool = True
    wandb_api_key: str | None = None
    wandb_kwargs: dict[str, Any] | None = None


@dataclass
class LoggerExperiment:
    """Algorithm and environment identity for training loggers."""

    algo: str
    env_name: str
    init_hyperparams: dict[str, Any] | None = None
    mutation_hyperparams: dict[str, Any] | None = None


@dataclass
class TrainRunConfig:
    """Loop, evolution, checkpoint, and logging nested for a train_* entrypoint."""

    loop: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    evolution: TrainEvolutionConfig = field(default_factory=TrainEvolutionConfig)
    checkpoint: TrainCheckpointConfig = field(default_factory=TrainCheckpointConfig)
    logging: TrainLoggingConfig = field(default_factory=TrainLoggingConfig)


@dataclass
class OffPolicyExploreConfig:
    """Epsilon-greedy schedule and learning delay for off-policy train loops."""

    learning_delay: int = 0
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay: float = 0.995


@dataclass
class OfflineDatasetConfig:
    """Offline dataset source for ``train_offline``."""

    dataset: Any | None = None
    minari_dataset_id: str | None = None
    remote: bool = False


@dataclass
class MultiAgentTrainLoopConfig(TrainLoopConfig):
    """Multi-agent train loops use a shorter default evolution cadence."""

    max_steps: int = 50_000
    evo_steps: int = 25


@dataclass
class MultiAgentTrainRunConfig:
    """TrainRunConfig with multi-agent loop defaults."""

    loop: MultiAgentTrainLoopConfig = field(default_factory=MultiAgentTrainLoopConfig)
    evolution: TrainEvolutionConfig = field(default_factory=TrainEvolutionConfig)
    checkpoint: TrainCheckpointConfig = field(default_factory=TrainCheckpointConfig)
    logging: TrainLoggingConfig = field(default_factory=TrainLoggingConfig)


@dataclass
class MultiAgentScoreConfig:
    """Whether to sum sub-agent rewards into one episode score."""

    sum_scores: bool = True


@dataclass
class MultiAgentOffPolicyExploreConfig:
    """Learning delay and score aggregation for multi-agent off-policy training."""

    learning_delay: int = 0
    sum_scores: bool = True


@dataclass
class BanditTrainLoopConfig:
    """Bandit train loops use episode steps instead of vectorized env steps."""

    max_steps: int = 20_000
    episode_steps: int = 500
    evo_steps: int = 2500
    eval_steps: int = 500
    eval_loop: int = 1
    target: float | None = None


@dataclass
class BanditTrainRunConfig:
    """TrainRunConfig with bandit loop defaults."""

    loop: BanditTrainLoopConfig = field(default_factory=BanditTrainLoopConfig)
    evolution: TrainEvolutionConfig = field(default_factory=TrainEvolutionConfig)
    checkpoint: TrainCheckpointConfig = field(default_factory=TrainCheckpointConfig)
    logging: TrainLoggingConfig = field(default_factory=TrainLoggingConfig)


@dataclass
class LLMTrainCheckpointConfig:
    """LLM finetune checkpoints use step counts rather than env-step intervals."""

    checkpoint_steps: int | None = None
    checkpoint_path: str | None = None
    save_elite: bool | None = None
    elite_path: str | None = None


@dataclass
class LLMRolloutLoopConfig:
    """Generative rollout loop budget and evaluation cadence."""

    max_steps: int = 32768
    evo_steps: int | None = None
    eval_loop: int = 1
    evaluation_interval: int = 50
    max_reward: float | None = None
    max_wall_seconds: float | None = None
    io_timeout_s: float | None = 600.0


@dataclass
class LLMDatasetLoopConfig:
    """Teacher-forced dataset loop budget."""

    max_steps: int | None = None
    num_epochs: int | None = None
    evo_steps: int | None = None
    evaluation_interval: int = 10


@dataclass
class LLMRolloutRunConfig:
    """Train run config for ``train_llm_rollout``."""

    loop: LLMRolloutLoopConfig = field(default_factory=LLMRolloutLoopConfig)
    evolution: TrainEvolutionConfig = field(default_factory=TrainEvolutionConfig)
    checkpoint: LLMTrainCheckpointConfig = field(
        default_factory=LLMTrainCheckpointConfig
    )
    logging: TrainLoggingConfig = field(default_factory=TrainLoggingConfig)


@dataclass
class LLMDatasetRunConfig:
    """Train run config for ``train_llm_dataset``."""

    loop: LLMDatasetLoopConfig = field(default_factory=LLMDatasetLoopConfig)
    evolution: TrainEvolutionConfig = field(default_factory=TrainEvolutionConfig)
    checkpoint: LLMTrainCheckpointConfig = field(
        default_factory=LLMTrainCheckpointConfig
    )
    logging: TrainLoggingConfig = field(default_factory=TrainLoggingConfig)
