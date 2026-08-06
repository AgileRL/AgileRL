# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced dataset training for LLM finetuning.

:func:`train_llm_dataset` runs the offline dataloader loop over a
:class:`~agilerl.llm_envs.DatasetEnv` for the preference (DPO) and SFT
objectives.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import DPO
from agilerl.algorithms.sft import SFT
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.protocols import SelectionStrategyProtocol
from agilerl.training.llm.common import (
    _compute_training_steps,
    _resolve_training_envs,
    _validate_finetune_args,
)
from agilerl.utils.utils import (
    _distributed_world_size,
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
    save_llm_checkpoint,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.llm_envs import DatasetEnv

if TYPE_CHECKING:
    SupportedDataset = DPO | SFT


def train_llm_dataset(
    pop: "list[SupportedDataset]",
    env: "DatasetEnv | None" = None,
    env_fn: "Callable[[], DatasetEnv] | None" = None,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    csv: bool = False,
    csv_log_dir: str | None = None,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    checkpoint_path: str | None = None,
    selection_strategy: SelectionStrategyProtocol | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    evaluation_interval: int = 10,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> "tuple[list[SupportedDataset], Any]":
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
    :param save_elite: Whether to save the elite checkpoint during evolution.
    :type save_elite: bool | None
    :param elite_path: Path used for checkpoint saving.
    :type elite_path: str | None
    :param wb: Whether to log metrics to Weights and Biases.
    :type wb: bool
    :param tensorboard: Whether to log to TensorBoard.
    :type tensorboard: bool
    :param tensorboard_log_dir: Directory for TensorBoard event files.
    :type tensorboard_log_dir: str | None
    :param csv: Whether to log aggregate metrics to CSV.
    :type csv: bool
    :param csv_log_dir: Path for the CSV file.
    :type csv_log_dir: str | None
    :param evo_steps: Number of outer iterations between evolution steps.
    :type evo_steps: int | None
    :param checkpoint_steps: Number of iterations between checkpoint saves when
        evolution is disabled.
    :type checkpoint_steps: int | None
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path.
    :type checkpoint_path: str | None
    :param selection_strategy: Selection strategy driving evolution.
    :type selection_strategy: SelectionStrategyProtocol | None
    :param tournament: Deprecated alias for selection_strategy.
    :type tournament: TournamentSelection | None
    :param mutation: Mutation operator used during evolution.
    :type mutation: Mutations | None
    :param wandb_api_key: Optional W&B API key.
    :type wandb_api_key: str | None
    :param wandb_kwargs: Additional kwargs forwarded to ``wandb.init()``.
    :type wandb_kwargs: dict[str, Any] | None
    :param evaluation_interval: Frequency (iterations) for evaluation.
    :type evaluation_interval: int
    :param verbose: Whether to print periodic training summaries.
    :type verbose: bool
    :param accelerator: Optional accelerator for distributed training.
    :type accelerator: Accelerator | None
    :param max_steps: Maximum step budget; defaults to dataset-driven length.
    :type max_steps: int | None
    :param num_epochs: Number of epochs to run; takes precedence over max_steps.
    :type num_epochs: int | None
    :return: The finetuned population and its last recorded fitnesses.
    :rtype: tuple[list[SupportedDataset], Any]
    """
    envs, uses_env_fn = _resolve_training_envs(pop=pop, env=env, env_fn=env_fn)
    env_name = envs[0].name

    selection_strategy = resolve_selection_strategy(selection_strategy, tournament)

    is_preference = isinstance(pop[0], DPO)
    _validate_finetune_args(
        evo_steps,
        selection_strategy,
        mutation,
        num_epochs,
        max_steps,
        pop,
        (DPO, SFT),
        (
            "The algorithm must be DPO (preference) or SFT (supervised) for "
            f"dataset finetuning. Got {type(pop[0])} instead."
        ),
        checkpoint_steps=checkpoint_steps,
    )

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

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path

    # ``len(envs[0])`` is this rank's shard; scale back to the global row count
    # so epoch accounting matches the whole dataset.
    max_steps, training_steps = _compute_training_steps(
        max_steps,
        num_epochs,
        len(envs[0]) * data_increment,
        effective_data_batch_size,
        len(pop),
    )

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=init_hp.get("ALGO", pop[0].algo),
        env_name=env_name,
        pbar=pbar,
        verbose=verbose,
        wb=wb,
        tensorboard=tensorboard,
        csv=csv,
        tensorboard_log_dir=tensorboard_log_dir,
        csv_log_dir=csv_log_dir,
        accelerator=accelerator,
        wandb_api_key=wandb_api_key,
        wandb_kwargs=wandb_kwargs,
        init_hyperparams=init_hp,
    )

    population = Population(agents=pop, accelerator=accelerator, loggers=loggers)

    total_steps = 0
    displayed_steps = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False

    for i in range(training_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent_idx, agent in enumerate(population.agents):
            if total_steps >= max_steps:
                break
            training_env = envs[agent_idx] if uses_env_fn else envs[0]

            agent.set_reference_policy(training_env.num_epochs)
            agent.init_training_step()

            # ``DatasetEnv.reset`` is the data-advancing call: each invocation
            # returns the next collated batch (``step`` is a no-op). The first
            # training step rewinds to the dataset start so a reused env
            # doesn't begin mid-epoch.
            # The spec pairs the env's objective with the population's algorithm,
            # so the batch always matches this agent's ``learn`` shape.
            batch = cast("Any", training_env.reset(reset_dataloaders=i == 0))
            learn_result = agent.learn(batch)
            score = (
                float(learn_result["chosen_reward"] - learn_result["rejected_reward"])
                if is_preference
                else -float(learn_result["loss"])
            )

            agent.add_scores([score])
            agent.finalize_training_step(training_env.data_batch_size_per_gpu)
            total_steps += effective_data_batch_size

        if (i + 1) % evaluation_interval == 0:
            for agent_idx, agent in enumerate(population.agents):
                agent.test(envs[agent_idx] if uses_env_fn else envs[0])
            if accelerator is not None:
                accelerator.wait_for_everyone()

        if accelerator is None or accelerator.is_main_process:
            increment = min(effective_data_batch_size, max_steps - displayed_steps)
            if increment > 0:
                pbar.update(increment)
                displayed_steps += increment

            population.report_metrics(clear=True)
        else:
            # Metrics accumulate on every rank; only main reports, so the
            # others must still clear or their stores grow for the whole run.
            population.clear_agent_metrics()

        if selection_strategy is not None and mutation is not None:
            # evo_steps is guaranteed set here: it is validated as set on entry
            # when a selection strategy and mutation are enabled.
            assert evo_steps is not None
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                population.update(
                    run_selection_and_mutation(
                        selection_strategy,
                        population=population.agents,
                        mutation=mutation,
                        env_name=env_name,
                        accelerator=accelerator,
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=bool(save_elite),
                    ),
                )
                if accelerator is not None:
                    accelerator.wait_for_everyone()

                population.increment_evo_step()
        else:
            checkpoint_due = False
            if checkpoint_steps is not None:
                while (
                    next_checkpoint_step is not None
                    and total_steps >= next_checkpoint_step
                ):
                    checkpoint_due = True
                    next_checkpoint_step += checkpoint_steps
            if (
                total_steps >= max_steps
                and not max_steps_checkpoint_saved
                and (
                    checkpoint_steps is not None
                    or checkpoint_path is not None
                    or elite_path is not None
                )
            ):
                checkpoint_due = True
                max_steps_checkpoint_saved = True
            if checkpoint_due:
                save_llm_checkpoint(
                    population.agents[-1],
                    checkpoint_path if checkpoint_path is not None else elite_path,
                )

    if save_elite and elite_path is not None:
        elite = max(
            population.agents,
            key=lambda a: a.fitness[-1] if a.fitness else float("-inf"),
        )
        save_llm_checkpoint(elite, elite_path)

    population.finish()
    pbar.close()
    # LLM fitnesses are scalar mean rewards; `Population` types them as the wider
    # scalar-or-per-agent-dict row shared with multi-agent training.
    return population.agents, population.last_scalar_fitnesses
