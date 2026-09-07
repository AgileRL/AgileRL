# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared setup, evolution, and checkpoint helpers for ``train_*`` entry points."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from accelerate import Accelerator
from tqdm import tqdm

from agilerl.population import Population
from agilerl.training.configs import (
    LoggerExperiment,
    TrainCheckpointConfig,
    TrainEvolutionConfig,
    TrainLoggingConfig,
)
from agilerl.typing import InitHyperparams
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
    save_population_checkpoint,
)

logger = logging.getLogger(__name__)


class TrainRunLike(Protocol):
    """Loop, evolution, checkpoint, and logging nested on a train_* run config."""

    loop: object
    evolution: TrainEvolutionConfig
    checkpoint: TrainCheckpointConfig
    logging: TrainLoggingConfig


@dataclass
class TrainSession:
    """Population, progress bar, and run config for one ``train_*`` loop."""

    population: Population
    pbar: tqdm.tqdm
    run: TrainRunLike
    env_name: str
    algo: str
    accelerator: Accelerator | None
    save_path: str
    checkpoint_count: int = 0
    capture_grama: bool = False


def validate_train_run(run: TrainRunLike, *, algo: str) -> None:
    """Check types on the fields classic RL train loops require."""
    loop = run.loop
    logging_cfg = run.logging
    checkpoint = run.checkpoint
    assert isinstance(algo, str), (
        "'algo' must be the name of the algorithm as a string."
    )
    assert isinstance(loop.max_steps, int), "Number of steps must be an integer."
    assert isinstance(loop.evo_steps, int), "Evolution frequency must be an integer."
    if loop.target is not None:
        assert isinstance(loop.target, (float, int)), (
            "Target score must be a float or an integer."
        )
    if checkpoint.checkpoint is not None:
        assert isinstance(checkpoint.checkpoint, int), "Checkpoint must be an integer."
    assert isinstance(logging_cfg.wb, bool), (
        "'wb' must be a boolean flag, indicating whether to record run with W&B"
    )
    assert isinstance(logging_cfg.verbose, bool), "Verbose must be a boolean."
    if checkpoint.save_elite is False and checkpoint.elite_path is not None:
        warnings.warn(
            "'save_elite' set to False but 'elite_path' has been defined, elite will not\
                      be saved unless 'save_elite' is set to True.",
            stacklevel=2,
        )
    if checkpoint.checkpoint is None and checkpoint.checkpoint_path is not None:
        warnings.warn(
            "'checkpoint' set to None but 'checkpoint_path' has been defined, checkpoint will not\
                      be saved unless 'checkpoint' is defined.",
            stacklevel=2,
        )


def resolve_train_evolution(run: TrainRunLike) -> TrainRunLike:
    """Resolve the deprecated ``tournament`` alias into ``selection_strategy``."""
    evolution = run.evolution
    selection = resolve_selection_strategy(
        evolution.selection_strategy,
        evolution.tournament,
    )
    return replace(run, evolution=replace(evolution, selection_strategy=selection))


def checkpoint_save_path(
    env_name: str,
    algo: str,
    checkpoint_path: str | None,
) -> str:
    """Build the checkpoint stem from an explicit path or env/algo/timestamp."""
    if checkpoint_path is not None:
        return checkpoint_path.split(".pt")[0]
    stamp = datetime.now().strftime("%m%d%Y%H%M%S")
    return f"{env_name}-EvoHPO-{algo}-{stamp}"


def start_train_session(
    agents: list[object],
    run: TrainRunLike,
    *,
    algo: str,
    env_name: str,
    accelerator: Accelerator | None,
    init_hp: InitHyperparams,
    wandb_kwargs_overlay: dict[str, object] | None = None,
) -> TrainSession:
    """Build loggers, wrap the population, and apply pre-training mutation."""
    logging_cfg = run.logging
    wandb_kwargs = logging_cfg.wandb_kwargs
    if wandb_kwargs_overlay is not None:
        wandb_kwargs = {**wandb_kwargs_overlay, **(wandb_kwargs or {})}
    logging_cfg = replace(logging_cfg, wandb_kwargs=wandb_kwargs)
    run = replace(run, logging=logging_cfg)
    mutation = run.evolution.mutation
    pbar = default_progress_bar(run.loop.max_steps, accelerator)
    loggers = init_loggers(
        experiment=LoggerExperiment(
            algo=algo,
            env_name=env_name,
            init_hyperparams=init_hp,
            mutation_hyperparams=run.evolution.mut_p,
        ),
        pbar=pbar,
        logging=logging_cfg,
        accelerator=accelerator,
    )
    population = Population(agents=agents, accelerator=accelerator, loggers=loggers)
    if accelerator is None and mutation is not None:
        population.update(mutation.mutation(population.agents, pre_training_mut=True))
    return TrainSession(
        population=population,
        pbar=pbar,
        run=run,
        env_name=env_name,
        algo=algo,
        accelerator=accelerator,
        save_path=checkpoint_save_path(
            env_name,
            algo,
            run.checkpoint.checkpoint_path,
        ),
        capture_grama=mutation is not None and mutation.parameters_mut > 0,
    )


def report_after_eval(session: TrainSession) -> None:
    """Increment the evo counter then publish metrics (off-policy / bandit / MA)."""
    session.population.increment_evo_step()
    session.population.report_metrics(clear=True)


def report_before_evo(session: TrainSession) -> None:
    """Publish metrics then increment the evo counter (on-policy)."""
    session.population.report_metrics(clear=True)
    session.population.increment_evo_step()


def stop_if_target(session: TrainSession) -> bool:
    """Finish the run when the target score is met. Returns True if training stops."""
    if not session.population.should_stop(session.run.loop.target):
        return False
    logger.info("Target score has been reached. Stopping training.")
    close_training(session)
    return True


def maybe_evolve(session: TrainSession) -> None:
    """Run tournament selection and mutation when both are configured."""
    evolution = session.run.evolution
    checkpoint = session.run.checkpoint
    if evolution.selection_strategy is None or evolution.mutation is None:
        return
    session.population.update(
        run_selection_and_mutation(
            evolution.selection_strategy,
            population=session.population.agents,
            mutation=evolution.mutation,
            env_name=session.env_name,
            algo=session.algo,
            elite_path=checkpoint.elite_path,
            save_elite=checkpoint.save_elite,
            accelerator=session.accelerator,
        ),
    )


def maybe_checkpoint(session: TrainSession, steps: int | None = None) -> None:
    """Write a population checkpoint when the step cadence elapses."""
    checkpoint = session.run.checkpoint
    if checkpoint.checkpoint is None:
        return
    count_steps = session.population.local_step if steps is None else steps
    if count_steps // checkpoint.checkpoint <= session.checkpoint_count:
        return
    save_population_checkpoint(
        population=session.population.agents,
        save_path=session.save_path,
        overwrite_checkpoints=checkpoint.overwrite_checkpoints,
        accelerator=session.accelerator,
    )
    session.checkpoint_count += 1


def evolve_and_checkpoint(session: TrainSession) -> None:
    """Run tournament mutation and write a periodic population checkpoint."""
    maybe_evolve(session)
    maybe_checkpoint(session)


def close_training(session: TrainSession) -> None:
    """Close loggers after a finished or early-stopped run."""
    session.population.finish()
    session.pbar.close()
