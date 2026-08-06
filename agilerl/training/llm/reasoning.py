# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import GRPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.protocols import SelectionStrategyProtocol
from agilerl.training.llm.common import (
    _compute_training_steps,
    _num_epochs_reached,
    _resolve_training_envs,
    _validate_finetune_args,
)
from agilerl.utils.distributed import barrier, get_world_size, is_distributed, is_main_process
from agilerl.utils.llm_utils import (
    align_completion_batch_shapes_across_ranks,
    needs_cross_rank_seq_padding,
    aggregate_metrics_across_gpus,
)
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    resolve_selection_strategy,
    run_selection_and_mutation,
    save_llm_checkpoint,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import ReasoningGym

if TYPE_CHECKING:
    SupportedReasoning = GRPO | LLMPPO | LLMREINFORCE


def finetune_llm_reasoning(
    pop: "list[SupportedReasoning]",
    env: ReasoningGym | None = None,
    env_fn: Callable[[], ReasoningGym] | None = None,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    checkpoint_path: str | None = None,
    selection_strategy: SelectionStrategyProtocol | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    evaluation_interval: int = 10,
    max_reward: int | None = None,
    verbose: bool = True,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> "tuple[list[SupportedReasoning], list[float]]":
    """Finetunes a population of GRPO/LLMPPO/LLMREINFORCE agents on a ReasoningGym.

    :param pop: Population of reasoning RL agents to finetune.
    :type pop: list[GRPO | LLMPPO | LLMREINFORCE]
    :param env: Shared ReasoningGym environment. Mutually exclusive with ``env_fn``.
    :type env: ReasoningGym | None
    :param env_fn: Factory that creates one ReasoningGym per agent. Mutually exclusive
        with ``env``.
    :type env_fn: Callable[[], ReasoningGym] | None
    :param init_hp: Initial hyperparameters for the population
    :type init_hp: dict, optional
    :param save_elite: Whether to save the elite model, defaults to None
    :type save_elite: bool, optional
    :param elite_path: Path to save the elite model, defaults to None
    :type elite_path: str, optional
    :param wb: Whether to use Weights and Biases, defaults to False
    :type wb: bool, optional
    :param tensorboard: TensorBoard tracking, defaults to False
    :type tensorboard: bool, optional
    :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to None
    :type tensorboard_log_dir: str, optional
    :param evo_steps: Number of steps between evolution, defaults to None
    :type evo_steps: int, optional
    :param checkpoint_steps: Number of steps between checkpoints, defaults to None
    :type checkpoint_steps: int, optional
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path, defaults to None
    :type checkpoint_path: str | None, optional
    :param selection_strategy: Selection strategy driving evolution, defaults to None
    :type selection_strategy: SelectionStrategyProtocol | None, optional
    :param tournament: Deprecated alias for selection_strategy, defaults to None
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: Mutations, optional
    :param wandb_api_key: Wandb API key, defaults to None
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wandb_kwargs: dict, optional
    :param evaluation_interval: Number of steps between evaluation, defaults to 10
    :type evaluation_interval: int, optional
    :param max_reward: Maximum reward to aim for, defaults to None
    :type max_reward: int, optional
    :param verbose: Whether to print verbose output, defaults to True
    :type verbose: bool, optional
    :param max_steps: Maximum number of steps to run, defaults to None
    :type max_steps: int, optional
    :param num_epochs: Number of epochs to run, if set, takes precedence over max_steps,
        defaults to None
    :type num_epochs: int, optional
    :return: The finetuned population.
    :rtype: PopulationType
    """
    envs, uses_env_fn = _resolve_training_envs(pop=pop, env=env, env_fn=env_fn)

    selection_strategy = resolve_selection_strategy(selection_strategy, tournament)

    _validate_finetune_args(
        evo_steps,
        selection_strategy,
        mutation,
        num_epochs,
        max_steps,
        pop,
        (GRPO, LLMPPO, LLMREINFORCE),
        (
            "The algorithm must be GRPO, LLMPPO, or LLMREINFORCE for reasoning-based "
            f"reinforcement learning. Got {type(pop[0])} instead."
        ),
        checkpoint_steps=checkpoint_steps,
        algo="grpo",
    )
    init_hp = (
        {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }
        if init_hp is None
        else init_hp
    )
    data_increment = get_world_size()
    effective_data_batch_size = data_increment * envs[0].data_batch_size_per_gpu

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = is_distributed()
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path

    max_steps, training_steps = _compute_training_steps(
        max_steps, num_epochs, len(envs[0]), effective_data_batch_size, len(pop)
    )

    pbar = default_progress_bar(max_steps)

    # Initialize loggers and Population wrapper
    loggers = init_loggers(
        algo=init_hp.get("ALGO", pop[0].algo),
        env_name=envs[0].name,
        pbar=pbar,
        verbose=verbose,
        wb=wb,
        tensorboard=tensorboard,
        tensorboard_log_dir=tensorboard_log_dir,
        wandb_api_key=wandb_api_key,
        wandb_kwargs=wandb_kwargs,
        init_hyperparams=init_hp,
    )
    population = Population(
        agents=pop,
        loggers=loggers,
    )

    total_steps = 0
    displayed_steps = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False

    # Reset the environments
    if uses_env_fn:
        prompts_by_agent = [e.reset(reset_dataloaders=True) for e in envs]
    else:
        prompts = envs[0].reset(reset_dataloaders=True)

    for i in range(training_steps):
        barrier()

        for agent_idx, agent in enumerate(population.agents):
            training_env = envs[agent_idx] if uses_env_fn else envs[0]
            current_prompts = prompts_by_agent[agent_idx] if uses_env_fn else prompts

            agent.set_reference_policy(training_env.num_epochs)
            agent.init_training_step()

            action_result = agent.get_action(current_prompts)
            completion_ids = action_result.completion_ids
            action_masks = action_result.action_masks
            # Per-row vLLM sampling logprobs captured during get_action (only
            # when the GRPO-family mismatch correction is enabled); ``None``
            # otherwise.
            sampling_logps = action_result.sampling_logps
            next_prompts_i, rewards = training_env.step(completion_ids)

            experiences = (completion_ids, action_masks, rewards)
            if needs_cross_rank_seq_padding(
                agent,
                world_size=get_world_size(),
            ):
                completion_ids, action_masks, rewards = (
                    align_completion_batch_shapes_across_ranks(
                        completion_ids,
                        action_masks,
                        rewards,
                        pad_token_id=agent.pad_token_id,
                    )
                )
                experiences = (completion_ids, action_masks, rewards)
            learn_kwargs = (
                {"sampling_logps": sampling_logps}
                if sampling_logps is not None
                and isinstance(agent, (GRPO, LLMPPO, LLMREINFORCE))
                else {}
            )
            agent.learn(experiences, **learn_kwargs)

            if max_reward is not None:
                if "accuracy" not in agent.metrics.additional_metrics:
                    agent.metrics.register("accuracy")

                accuracy = (rewards == max_reward).sum() / len(rewards.flatten())
                agg_accuracy = aggregate_metrics_across_gpus(accuracy)
                if is_main_process():
                    agent.metrics.log("accuracy", agg_accuracy)

            agg_rewards = aggregate_metrics_across_gpus(rewards)
            agent.add_scores([agg_rewards])
            agent.finalize_training_step(training_env.data_batch_size_per_gpu)
            total_steps += effective_data_batch_size

            if uses_env_fn:
                prompts_by_agent[agent_idx] = next_prompts_i
            else:
                prompts = next_prompts_i

        # Evaluate performance
        if (i + 1) % evaluation_interval == 0:
            for idx, agent in enumerate(population.agents):
                agent.test(envs[idx] if uses_env_fn else envs[0])

        # Report progress
        if is_main_process():
            increment = min(effective_data_batch_size, max_steps - displayed_steps)
            if increment > 0:
                pbar.update(increment)
                displayed_steps += increment

        population.report_metrics(clear=True)

        # Selection and mutation
        if selection_strategy is not None and mutation is not None:
            # evo_steps is guaranteed set here: it is validated as set on entry
            # when tournament and mutation are enabled.
            assert evo_steps is not None
            if (i + 1) % evo_steps == 0:
                barrier()
                population.update(
                    run_selection_and_mutation(
                        selection_strategy,
                        population=population.agents,
                        mutation=mutation,
                        env_name=envs[0].name,
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=bool(save_elite),
                    ),
                )
                barrier()

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

        if _num_epochs_reached(envs, num_epochs):
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
