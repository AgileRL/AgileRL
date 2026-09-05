# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced dataset training for LLM finetuning.

:func:`train_llm_dataset` runs the offline dataloader loop over a
:class:`~agilerl.llm_envs.DatasetEnv` for the preference (DPO) and SFT
objectives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import DPO
from agilerl.algorithms.sft import SFT
from agilerl.population import Population
from agilerl.training.configs import LLMDatasetRunConfig, LoggerExperiment
from agilerl.training.llm.common import (
    LLMCheckpointProgress,
    _compute_training_steps,
    _resolve_training_envs,
    _validate_finetune_args,
    maybe_save_llm_step_checkpoint,
    save_llm_elite_if_requested,
)
from agilerl.utils.constructor_kwargs import accept_flat_kwargs
from agilerl.utils.llm_utils import is_preference_prompts, is_sft_prompts
from agilerl.utils.utils import (
    _distributed_world_size,
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.llm_envs import DatasetEnv

if TYPE_CHECKING:
    from tqdm import tqdm

SupportedDataset = DPO | SFT


@dataclass
class LLMDatasetSession:
    """Population, envs, and run config for one ``train_llm_dataset`` loop."""

    population: Population
    run: LLMDatasetRunConfig
    envs: list[DatasetEnv]
    uses_env_fn: bool
    accelerator: Accelerator | None
    pbar: tqdm
    env_name: str
    max_steps: int
    training_steps: int
    effective_data_batch_size: int
    checkpoint_progress: LLMCheckpointProgress
    total_steps: int = 0
    displayed_steps: int = 0


def _resolve_dataset_run(run: LLMDatasetRunConfig | None) -> LLMDatasetRunConfig:
    run = run or LLMDatasetRunConfig()
    evolution = run.evolution
    selection = resolve_selection_strategy(
        evolution.selection_strategy,
        evolution.tournament,
    )
    return replace(run, evolution=replace(evolution, selection_strategy=selection))


def _validate_dataset_run(
    pop: list[SupportedDataset],
    run: LLMDatasetRunConfig,
) -> None:
    loop = run.loop
    evolution = run.evolution
    _validate_finetune_args(
        loop.evo_steps,
        evolution.selection_strategy,
        evolution.mutation,
        loop.num_epochs,
        loop.max_steps,
        pop,
        (DPO, SFT),
        (
            "The algorithm must be DPO (preference) or SFT (supervised) for "
            f"dataset finetuning. Got {type(pop[0])} instead."
        ),
        checkpoint_steps=run.checkpoint.checkpoint_steps,
    )


def _start_llm_dataset(
    pop: list[SupportedDataset],
    env: DatasetEnv | None,
    env_fn: Callable[[], DatasetEnv] | None,
    init_hp: dict[str, Any] | None,
    run: LLMDatasetRunConfig,
    accelerator: Accelerator | None,
) -> LLMDatasetSession:
    """Validate args, resolve envs, and build loggers."""
    envs, uses_env_fn = _resolve_training_envs(pop=pop, env=env, env_fn=env_fn)
    env_name = envs[0].name
    _validate_dataset_run(pop, run)
    init_hp = (
        {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }
        if init_hp is None
        else init_hp
    )
    data_increment = _distributed_world_size(accelerator)
    effective_data_batch_size = data_increment * envs[0].data_batch_size_per_gpu
    if envs[0].world_size != data_increment:
        msg = (
            f"DatasetEnv was built with world_size={envs[0].world_size} but the "
            f"run has {data_increment} data-parallel ranks; every rank would "
            "draw the same batches. Build the env with the runtime rank/world_size."
        )
        raise ValueError(msg)
    if run.logging.wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
    # ``len(envs[0])`` is this rank's shard; scale back to the global row count
    # so epoch accounting matches the whole dataset.
    max_steps, training_steps = _compute_training_steps(
        run.loop.max_steps,
        run.loop.num_epochs,
        len(envs[0]) * data_increment,
        effective_data_batch_size,
        len(pop),
    )
    pbar = default_progress_bar(max_steps, accelerator)
    loggers = init_loggers(
        experiment=LoggerExperiment(
            algo=init_hp.get("ALGO", pop[0].algo),
            env_name=env_name,
            init_hyperparams=init_hp,
        ),
        pbar=pbar,
        logging=run.logging,
        accelerator=accelerator,
    )
    return LLMDatasetSession(
        population=Population(agents=pop, accelerator=accelerator, loggers=loggers),
        run=run,
        envs=envs,
        uses_env_fn=uses_env_fn,
        accelerator=accelerator,
        pbar=pbar,
        env_name=env_name,
        max_steps=max_steps,
        training_steps=training_steps,
        effective_data_batch_size=effective_data_batch_size,
        checkpoint_progress=LLMCheckpointProgress(run.checkpoint.checkpoint_steps),
    )


def _learn_dataset_batch(
    agent: SupportedDataset,
    batch: dict[str, Any],
) -> float:
    if isinstance(agent, DPO):
        if not is_preference_prompts(batch):
            msg = (
                "DPO needs an objective='preference' DatasetEnv batch; "
                f"got keys {sorted(batch)}."
            )
            raise ValueError(msg)
        learn_result = agent.learn(batch)
        return float(learn_result["chosen_reward"] - learn_result["rejected_reward"])
    if not is_sft_prompts(batch):
        msg = (
            "SFT needs an objective='sft' DatasetEnv batch; "
            f"got keys {sorted(batch)}."
        )
        raise ValueError(msg)
    learn_result = agent.learn(batch)
    return -float(learn_result["loss"])


def _train_one_dataset_agent(
    session: LLMDatasetSession,
    agent: SupportedDataset,
    agent_idx: int,
    iteration: int,
) -> None:
    training_env = session.envs[agent_idx] if session.uses_env_fn else session.envs[0]
    agent.set_reference_policy(training_env.num_epochs)
    agent.init_training_step()
    # ``reset`` returns the next collated batch; ``step`` does nothing.
    # Rewind a reused env at run start so it does not begin mid-epoch.
    # Shared env: once (first agent). Per-agent envs: each env once.
    reset_dataloaders = iteration == 0 and (session.uses_env_fn or agent_idx == 0)
    score = _learn_dataset_batch(
        agent, training_env.reset(reset_dataloaders=reset_dataloaders)
    )
    agent.add_scores([score])
    agent.finalize_training_step(training_env.data_batch_size_per_gpu)
    session.total_steps += session.effective_data_batch_size


def _maybe_eval_dataset(session: LLMDatasetSession, iteration: int) -> None:
    if (iteration + 1) % session.run.loop.evaluation_interval != 0:
        return
    for agent_idx, agent in enumerate(session.population.agents):
        agent.test(session.envs[agent_idx] if session.uses_env_fn else session.envs[0])
    if session.accelerator is not None:
        session.accelerator.wait_for_everyone()


def _report_dataset_metrics(session: LLMDatasetSession) -> None:
    if session.accelerator is None or session.accelerator.is_main_process:
        increment = min(
            session.effective_data_batch_size,
            session.max_steps - session.displayed_steps,
        )
        if increment > 0:
            session.pbar.update(increment)
            session.displayed_steps += increment
        session.population.report_metrics(clear=True)
        return
    # Metrics accumulate on every rank; only main reports, so the
    # others must still clear or their stores grow for the whole run.
    session.population.clear_agent_metrics()


def _evolve_or_checkpoint_dataset(session: LLMDatasetSession, iteration: int) -> None:
    evolution = session.run.evolution
    if evolution.selection_strategy is not None and evolution.mutation is not None:
        # evo_steps is guaranteed set here: it is validated as set on entry
        # when a selection strategy and mutation are enabled.
        assert session.run.loop.evo_steps is not None
        if (iteration + 1) % session.run.loop.evo_steps != 0:
            return
        if session.accelerator is not None:
            session.accelerator.wait_for_everyone()
        session.population.update(
            run_selection_and_mutation(
                evolution.selection_strategy,
                population=session.population.agents,
                mutation=evolution.mutation,
                env_name=session.env_name,
                accelerator=session.accelerator,
                language_model=True,
                elite_path=session.run.checkpoint.elite_path,
                save_elite=bool(session.run.checkpoint.save_elite),
            ),
        )
        if session.accelerator is not None:
            session.accelerator.wait_for_everyone()
        session.population.increment_evo_step()
        return
    maybe_save_llm_step_checkpoint(
        session.population.agents,
        session.checkpoint_progress,
        session.run.checkpoint,
        session.total_steps,
        session.max_steps,
    )


def _run_llm_dataset_loop(session: LLMDatasetSession) -> None:
    for iteration in range(session.training_steps):
        if session.accelerator is not None:
            session.accelerator.wait_for_everyone()
        for agent_idx, agent in enumerate(session.population.agents):
            if session.total_steps >= session.max_steps:
                break
            _train_one_dataset_agent(session, agent, agent_idx, iteration)
        _maybe_eval_dataset(session, iteration)
        _report_dataset_metrics(session)
        _evolve_or_checkpoint_dataset(session, iteration)


@accept_flat_kwargs
def train_llm_dataset(
    pop: list[SupportedDataset],
    env: DatasetEnv | None = None,
    env_fn: Callable[[], DatasetEnv] | None = None,
    init_hp: dict[str, Any] | None = None,
    run: LLMDatasetRunConfig | None = None,
    accelerator: Accelerator | None = None,
) -> tuple[list[SupportedDataset], Any]:
    """Train a population of DPO or SFT agents over a ``DatasetEnv`` dataloader.

    Each training step draws a labelled batch from the dataset environment. The
    algorithm of ``pop[0]`` selects the regime: DPO minimises a pairwise preference
    loss over chosen/rejected pairs, while SFT minimises the response cross-entropy.
    Both share evolution, checkpointing, and metrics logging.

    :param pop: Population of DPO or SFT agents to finetune.
    :type pop: list[SupportedDataset]
    :param env: Shared dataset environment that yields labelled batches.
    :type env: DatasetEnv | None
    :param env_fn: Optional factory that creates one dataset environment per agent.
    :type env_fn: Callable[[], DatasetEnv] | None
    :param init_hp: Initial hyperparameters for logging and defaults.
    :type init_hp: dict[str, Any] | None
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: LLMDatasetRunConfig, optional
    :param accelerator: Optional accelerator for distributed training.
    :type accelerator: Accelerator | None
    :return: The finetuned population and its last recorded fitnesses.
    :rtype: tuple[list[SupportedDataset], Any]
    """
    run = _resolve_dataset_run(run)
    session = _start_llm_dataset(pop, env, env_fn, init_hp, run, accelerator)
    _run_llm_dataset_loop(session)
    save_llm_elite_if_requested(session.population.agents, session.run.checkpoint)
    session.population.finish()
    session.pbar.close()
    return session.population.agents, session.population.last_scalar_fitnesses
