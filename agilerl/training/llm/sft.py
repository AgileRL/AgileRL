# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms.sft import SFT
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.training.llm.common import (
    _compute_training_steps,
    _validate_finetune_args,
)
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.llm_envs import SFTGym


def finetune_llm_sft(
    pop: list[SFT],
    env: SFTGym,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    checkpoint_path: str | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    evaluation_interval: int = 10,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> tuple[list[SFT], list[float]]:
    """Finetune a population of SFT agents on (prompt, response) pairs.

    Each training step draws a batch from ``env`` and minimises the cross-entropy
    loss over the *response* tokens only (prompt and padding positions are masked
    with ``ignore_index=-100``).

    :param pop: Population of SFT agents.
    :type pop: list[SFT]
    :param env: SFTGym environment wrapping the dataset.
    :type env: SFTGym
    :param init_hp: Initial hyperparameters for the population, defaults to None
    :type init_hp: dict, optional
    :param save_elite: Whether to save the elite model, defaults to None
    :type save_elite: bool, optional
    :param elite_path: Directory for checkpoints, defaults to None
    :type elite_path: str, optional
    :param wb: Whether to use Weights and Biases, defaults to False
    :type wb: bool, optional
    :param tensorboard: TensorBoard tracking, defaults to False
    :type tensorboard: bool, optional
    :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to None
    :type tensorboard_log_dir: str, optional
    :param evo_steps: Steps between HPO evolution rounds, defaults to None
    :type evo_steps: int, optional
    :param checkpoint_steps: Steps between non-HPO saves, defaults to None
    :type checkpoint_steps: int, optional
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path, defaults to None
    :type checkpoint_path: str | None, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: Mutations, optional
    :param wandb_api_key: W&B API key, defaults to None
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wandb_kwargs: dict, optional
    :param evaluation_interval: Steps between eval passes, defaults to 10
    :type evaluation_interval: int, optional
    :param verbose: Whether to print verbose output, defaults to True
    :type verbose: bool, optional
    :param accelerator: Distributed training handle, defaults to None
    :type accelerator: Accelerator, optional
    :param max_steps: Total samples to process (one epoch if None), defaults to None
    :type max_steps: int, optional
    :param num_epochs: Dataset passes; overrides max_steps when set, defaults to None
    :type num_epochs: int, optional
    :return: The finetuned population.
    :rtype: PopulationType
    """
    _validate_finetune_args(
        evo_steps,
        tournament,
        mutation,
        num_epochs,
        max_steps,
        pop,
        SFT,
        f"Population must contain SFT agents. Got {type(pop[0])}.",
        checkpoint_steps=checkpoint_steps,
        algo="sft",
    )
    init_hp = (
        {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }
        if init_hp is None
        else init_hp
    )

    data_increment = accelerator.num_processes if accelerator is not None else 1
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path

    max_steps, training_steps = _compute_training_steps(
        max_steps, num_epochs, len(env), effective_data_batch_size, len(pop)
    )

    pbar = default_progress_bar(max_steps, accelerator)

    # Initialize loggers and Population wrapper
    loggers = init_loggers(
        algo=init_hp.get("ALGO", "SFT"),
        env_name=env.name,
        pbar=pbar,
        verbose=verbose,
        wb=wb,
        tensorboard=tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        accelerator=accelerator,
        wandb_api_key=wandb_api_key,
        wandb_kwargs=wandb_kwargs,
        init_hyperparams=init_hp,
    )
    population = Population(
        agents=pop,
        accelerator=accelerator,
        loggers=loggers,
    )

    total_steps = 0
    displayed_steps = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False
    prompts = env.reset(reset_dataloaders=True)
    for i in range(training_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        # Learn from dataset
        for agent in population.agents:
            agent.set_reference_policy(env.num_epochs)
            agent.init_training_step()

            learn_result = agent.learn(prompts)
            agg_loss = learn_result["loss"]
            next_prompts = env.step()

            agent.add_scores([-agg_loss])
            agent.finalize_training_step(env.data_batch_size_per_gpu)
            total_steps += effective_data_batch_size

        prompts = next_prompts

        # Evaluate performance
        if (i + 1) % evaluation_interval == 0:
            for agent in population.agents:
                agent.test(env)

        # Report progress
        if accelerator is None or accelerator.is_main_process:
            increment = min(effective_data_batch_size, max_steps - displayed_steps)
            if increment > 0:
                pbar.update(increment)
                displayed_steps += increment

        population.report_metrics(clear=True)

        # Tournament selection and mutation
        if tournament and mutation is not None:
            # evo_steps is guaranteed set here: it is validated as set on entry
            # when tournament and mutation are enabled.
            assert evo_steps is not None
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                population.update(
                    tournament_selection_and_mutation(
                        population=population.agents,
                        tournament=tournament,
                        mutation=mutation,
                        env_name=env.name,
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
                    agent,
                    checkpoint_path if checkpoint_path is not None else elite_path,
                )

        if env.num_epochs == num_epochs:
            break

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
