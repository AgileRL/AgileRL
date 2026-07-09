import time
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from accelerate import Accelerator

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import DPO, GRPO
from agilerl.algorithms.sft import SFT
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.utils.llm_utils import safe_aggregate_metrics
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
)

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import (
        PreferenceGym,
        ReasoningGym,
        SFTGym,
        SyncMultiTurnVecEnv,
    )
    from agilerl.protocols import MultiTurnEnv
    from agilerl.rollouts.on_policy import collect_rollouts_llm
    from agilerl.utils.algo_utils import stack_and_pad_experiences

if TYPE_CHECKING:
    SupportedReasoning = GRPO | LLMPPO | LLMREINFORCE
    SupportedMultiturn = LLMPPO | LLMREINFORCE | GRPO

InitDictType = dict[str, Any] | None


def _validate_finetune_args(
    evo_steps: int | None,
    tournament: TournamentSelection | None,
    mutation: Mutations | None,
    num_epochs: int | None,
    max_steps: int | None,
    pop: list[Any],
    expected_type: type | tuple[type, ...],
    algorithm_type_error: str,
    *,
    checkpoint_steps: int | None = None,
    algo: Literal["grpo", "dpo", "sft", "multiturn"],
) -> None:
    if evo_steps is not None and (tournament is None or mutation is None):
        warnings.warn(
            "'evo_steps' is set but at least one of 'tournament' or 'mutation' "
            "is set to None. Evolution will not take place.",
            stacklevel=2,
        )
    if (tournament is not None and mutation is not None) and evo_steps is None:
        msg = "'evo_steps' must be set if 'tournament' and 'mutation' are not None."
        raise ValueError(msg)
    if num_epochs is not None and max_steps is not None:
        warnings.warn(
            "'num_epochs' is set but 'max_steps' is also set. "
            "'num_epochs' will take precedence over 'max_steps'.",
            stacklevel=2,
        )

    evo_active = (
        evo_steps is not None and tournament is not None and mutation is not None
    )
    if checkpoint_steps is not None and evo_active:
        warnings.warn(
            "'checkpoint_steps' is set, but evolution is active ('evo_steps', "
            "'tournament', and 'mutation'). Periodic step-based checkpoints "
            "are skipped while evolution is enabled.",
            stacklevel=2,
        )

    if mutation is not None:
        assert mutation.architecture_mut == 0, (
            "Probability of architecture mutation must be 0 for LLM finetuning."
        )
        assert mutation.new_layer_prob == 0, (
            "Probability of new layer mutation must be 0 for LLM finetuning."
        )
        assert mutation.parameters_mut == 0, (
            "Probability of network parameters mutation must be 0 for LLM finetuning."
        )
        assert mutation.activation_mut == 0, (
            "Probability of activation mutation must be 0 for LLM finetuning."
        )

    if not isinstance(pop[0], expected_type):
        raise ValueError(algorithm_type_error)


def _compute_training_steps(
    max_steps: int | None,
    num_epochs: int | None,
    env_len: int,
    effective_data_batch_size: int,
    pop_size: int = 1,
) -> tuple[int, int]:
    """Compute the number of training steps."""
    if max_steps is None and num_epochs is None:
        max_steps = env_len
    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * env_len
    assert max_steps is not None
    steps_per_iteration = effective_data_batch_size * pop_size
    training_steps = -(max_steps // -steps_per_iteration)
    return max_steps, training_steps


def _resolve_training_envs(
    pop: list[Any],
    env: ReasoningGym | PreferenceGym | None,
    env_fn: Callable[[], ReasoningGym | PreferenceGym] | None,
) -> tuple[list[ReasoningGym | PreferenceGym], bool]:
    """Resolve shared or per-agent training environments.

    :param pop: Population of agents being trained.
    :type pop: PopulationType
    :param env: Shared environment instance.
    :type env: ReasoningGym | PreferenceGym | None
    :param env_fn: Factory that creates one environment per agent.
    :type env_fn: Callable[[], ReasoningGym | PreferenceGym] | None
    :return: Environment list (aligned with population) and whether env_fn mode is active.
    :rtype: tuple[list, bool]
    """
    if env is not None and env_fn is not None:
        msg = "Provide exactly one of 'env' or 'env_fn', not both."
        raise ValueError(msg)
    if env is None and env_fn is None:
        msg = "Either 'env' or 'env_fn' must be provided."
        raise ValueError(msg)

    if env_fn is not None:
        return [env_fn() for _ in pop], True

    if len(pop) > 1:
        warnings.warn(
            "A shared 'env' is being used with multiple agents. This can introduce "
            "fairness bias; prefer 'env_fn' for per-agent environments.",
            stacklevel=2,
        )
    assert env is not None
    return [env], False


def _num_epochs_reached(
    envs: list[ReasoningGym | PreferenceGym], num_epochs: int | None
) -> bool:
    """Check whether all active environments have reached the epoch budget."""
    if num_epochs is None:
        return False
    epoch_counts = [getattr(e, "num_epochs", None) for e in envs]
    if not all(isinstance(c, int) for c in epoch_counts):
        return False
    return all(c >= num_epochs for c in epoch_counts)


# ---------------------------------------------------------------------------
# Public training entry points
# ---------------------------------------------------------------------------


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
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    evaluation_interval: int = 10,
    max_reward: int | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> "list[SupportedReasoning]":
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
    :param tournament: Tournament selection object, defaults to None
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
    :param accelerator: Accelerator object, defaults to None
    :type accelerator: Accelerator, optional
    :param max_steps: Maximum number of steps to run, defaults to None
    :type max_steps: int, optional
    :param num_epochs: Number of epochs to run, if set, takes precedence over max_steps,
        defaults to None
    :type num_epochs: int, optional
    :return: The finetuned population.
    :rtype: PopulationType
    """
    envs, uses_env_fn = _resolve_training_envs(pop=pop, env=env, env_fn=env_fn)

    _validate_finetune_args(
        evo_steps,
        tournament,
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
    data_increment = accelerator.num_processes if accelerator is not None else 1
    effective_data_batch_size = data_increment * envs[0].data_batch_size_per_gpu

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path

    max_steps, training_steps = _compute_training_steps(
        max_steps, num_epochs, len(envs[0]), effective_data_batch_size, len(pop)
    )

    pbar = default_progress_bar(max_steps, accelerator)

    # Initialize loggers and Population wrapper
    loggers = init_loggers(
        algo=init_hp.get("ALGO", "GRPO"),
        env_name=envs[0].name,
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

    # Reset the environments
    if uses_env_fn:
        prompts_by_agent = [e.reset(reset_dataloaders=True) for e in envs]
    else:
        prompts = envs[0].reset(reset_dataloaders=True)

    for i in range(training_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

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
                agg_accuracy = safe_aggregate_metrics(accelerator, accuracy)
                if accelerator is None or accelerator.is_main_process:
                    agent.metrics.log("accuracy", agg_accuracy)

            agg_rewards = safe_aggregate_metrics(accelerator, rewards)
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
            if accelerator is not None:
                accelerator.wait_for_everyone()

        # Report progress
        if accelerator is None or accelerator.is_main_process:
            increment = min(effective_data_batch_size, max_steps - displayed_steps)
            if increment > 0:
                pbar.update(increment)
                displayed_steps += increment

            population.report_metrics(clear=True)

        # Tournament selection and mutation
        if tournament and mutation is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                population.update(
                    tournament_selection_and_mutation(
                        population=population.agents,
                        tournament=tournament,
                        mutation=mutation,
                        env_name=envs[0].name,
                        accelerator=accelerator,
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=save_elite,
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
            if total_steps >= max_steps and not max_steps_checkpoint_saved:
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
    return population.agents, population.last_fitnesses


def finetune_llm_preference(
    pop: list[DPO],
    env: PreferenceGym | None = None,
    env_fn: Callable[[], PreferenceGym] | None = None,
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
) -> list[DPO]:
    """Finetune a population of DPO agents on pairwise preference data.

    :param pop: Population of DPO agents to finetune.
    :type pop: list[DPO]
    :param env: Shared PreferenceGym environment. Mutually exclusive with ``env_fn``.
    :type env: PreferenceGym | None
    :param env_fn: Factory that creates one PreferenceGym per agent. Mutually exclusive
        with ``env``.
    :type env_fn: Callable[[], PreferenceGym] | None
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
    :param evo_steps: Number of steps between evolution, defaults to None
    :type evo_steps: int, optional
    :param checkpoint_steps: Number of steps between checkpoints, defaults to None
    :type checkpoint_steps: int, optional
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path, defaults to None
    :type checkpoint_path: str | None, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: Mutations, optional
    :param wandb_api_key: Wandb API key, defaults to None
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wandb_kwargs: dict, optional
    :param evaluation_interval: Number of steps between evaluation, defaults to 10
    :type evaluation_interval: int, optional
    :param verbose: Whether to print verbose output, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator object, defaults to None
    :type accelerator: Accelerator, optional
    :param max_steps: Maximum number of steps to run, defaults to None
    :type max_steps: int, optional
    :param num_epochs: Number of epochs to run, if set, takes precedence over max_steps,
        defaults to None
    :type num_epochs: int, optional
    :return: The finetuned population.
    :rtype: PopulationType
    """
    envs, uses_env_fn = _resolve_training_envs(pop=pop, env=env, env_fn=env_fn)

    _validate_finetune_args(
        evo_steps,
        tournament,
        mutation,
        num_epochs,
        max_steps,
        pop,
        DPO,
        (
            "The algorithm must be DPO for preference-based reinforcement learning. "
            f"Got {type(pop[0])} instead."
        ),
        checkpoint_steps=checkpoint_steps,
        algo="dpo",
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
    effective_data_batch_size = data_increment * envs[0].data_batch_size_per_gpu

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path

    max_steps, training_steps = _compute_training_steps(
        max_steps, num_epochs, len(envs[0]), effective_data_batch_size, len(pop)
    )

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=init_hp.get("ALGO", "DPO"),
        env_name=envs[0].name,
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

    if uses_env_fn:
        prompts_by_agent = [e.reset(reset_dataloaders=True) for e in envs]
    else:
        prompts = envs[0].reset(reset_dataloaders=True)

    for i in range(training_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent_idx, agent in enumerate(population.agents):
            training_env = envs[agent_idx] if uses_env_fn else envs[0]
            current_prompts = prompts_by_agent[agent_idx] if uses_env_fn else prompts

            agent.set_reference_policy(training_env.num_epochs)
            agent.init_training_step()

            learn_result = agent.learn(current_prompts)
            chosen_reward = learn_result["chosen_reward"]
            rejected_reward = learn_result["rejected_reward"]
            next_prompts_i = training_env.step()

            agent.add_scores([float(chosen_reward - rejected_reward)])
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

            if accelerator is not None:
                accelerator.wait_for_everyone()

        # Report progress
        if accelerator is None or accelerator.is_main_process:
            increment = min(effective_data_batch_size, max_steps - displayed_steps)
            if increment > 0:
                pbar.update(increment)
                displayed_steps += increment

            population.report_metrics(clear=True)

        # Tournament selection and mutation
        if tournament and mutation is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                population.update(
                    tournament_selection_and_mutation(
                        population=population.agents,
                        tournament=tournament,
                        mutation=mutation,
                        env_name=envs[0].name,
                        accelerator=accelerator,
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=save_elite,
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
            if total_steps >= max_steps and not max_steps_checkpoint_saved:
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
    return population.agents, population.last_fitnesses


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
) -> list[SFT]:
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

            if accelerator is not None:
                accelerator.wait_for_everyone()

        # Report progress
        if accelerator is None or accelerator.is_main_process:
            increment = min(effective_data_batch_size, max_steps - displayed_steps)
            if increment > 0:
                pbar.update(increment)
                displayed_steps += increment

            population.report_metrics(clear=True)

        # Tournament selection and mutation
        if tournament and mutation is not None:
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
                        save_elite=save_elite,
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
            if total_steps >= max_steps and not max_steps_checkpoint_saved:
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
    return population.agents, population.last_fitnesses


def finetune_llm_multiturn(
    pop: "list[SupportedMultiturn]",
    max_turns: int,
    env_factory: "Callable[[], MultiTurnEnv]",
    env_config: dict[str, Any] | None = None,
    init_hp: dict[str, Any] | None = None,
    max_steps: int = 32768,
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
    evaluation_interval: int = 50,
    max_wall_seconds: float | None = None,
    max_reward: float | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
) -> "list[SupportedMultiturn]":
    """Finetune a population of agents on a multi-turn environment.

    Collects token-level episodes via ``SyncMultiTurnVecEnv`` and
    ``collect_rollouts_llm``, then runs turn-level updates (PPO/REINFORCE with
    ``turn_ids``, or GRPO without).

    :param pop: Population of LLMPPO, LLMREINFORCE, or GRPO agents.
    :type pop: PopulationType
    :param max_turns: Maximum interaction turns per episode.
    :type max_turns: int
    :param env_factory: Factory returning a fresh multi-turn env for each rollout.
    :type env_factory: Callable[[], MultiTurnEnv]
    :param env_config: Configuration for the environment factory.
    :type env_config: dict[str, Any], optional
    :param init_hp: Initial hyperparameters.
    :type init_hp: dict, optional
    :param max_steps: Progress-bar budget in sample steps, defaults to 32768.
    :type max_steps: int
    :param save_elite: Whether to save the elite checkpoint, defaults to None.
    :type save_elite: bool, optional
    :param elite_path: Directory for checkpoints, defaults to None.
    :type elite_path: str, optional
    :param wb: Whether to log to Weights and Biases, defaults to False.
    :type wb: bool
    :param tensorboard: TensorBoard tracking, defaults to False.
    :type tensorboard: bool
    :param tensorboard_log_dir: Directory for TensorBoard logs, defaults to None.
    :type tensorboard_log_dir: str, optional
    :param evo_steps: Steps between evolution (requires tournament and mutation).
    :type evo_steps: int, optional
    :param checkpoint_steps: Save checkpoint every N outer iterations when no evolution.
    :type checkpoint_steps: int, optional
    :param checkpoint_path: Directory for periodic checkpoints; falls back to elite_path, defaults to None
    :type checkpoint_path: str | None, optional
    :param tournament: Tournament selection for evolution, defaults to None.
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation operator for evolution, defaults to None.
    :type mutation: Mutations, optional
    :param wandb_api_key: W&B API key, defaults to None.
    :type wandb_api_key: str, optional
    :param wandb_kwargs: Additional kwargs to pass to wandb.init()
    :type wandb_kwargs: dict, optional
    :param evaluation_interval: Evaluate every N outer iterations.
    :type evaluation_interval: int
    :param max_wall_seconds: Stop after this wall-clock duration (seconds); ``None`` disables.
    :type max_wall_seconds: float | None
    :param max_reward: If set, adds accuracy metric vs this threshold.
    :type max_reward: float, optional
    :param verbose: Progress bar and periodic train summaries, defaults to True.
    :type verbose: bool
    :param accelerator: Hugging Face Accelerate instance, defaults to None.
    :type accelerator: Accelerator, optional
    :return: The finetuned population (same list object, possibly mutated in place).
    :rtype: PopulationType
    """
    _validate_finetune_args(
        evo_steps,
        tournament,
        mutation,
        None,
        max_steps,
        pop,
        (LLMPPO, LLMREINFORCE, GRPO),
        (
            "The algorithm must be LLMPPO, LLMREINFORCE, or GRPO for multi-turn "
            f"finetuning. Got {type(pop[0])} instead."
        ),
        checkpoint_steps=checkpoint_steps,
        algo="multiturn",
    )

    if init_hp is None:
        init_hp = {
            "BATCH_SIZE_PER_GPU": pop[0].batch_size_per_process,
            "ALGO": pop[0].algo,
        }

    batch_size = init_hp.get("BATCH_SIZE", pop[0].batch_size)

    data_increment = accelerator.num_processes if accelerator is not None else 1
    effective_data_batch_size = data_increment * batch_size
    env_name = init_hp.get("env_name", "multiturn")

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = batch_size
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_hp["max_turns"] = max_turns

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=init_hp.get("ALGO", "LLMPPO"),
        env_name=env_name,
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
    i = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False
    group_size = getattr(pop[0], "group_size", 1)
    rollout_env = SyncMultiTurnVecEnv(env_factory, batch_size, group_size, env_config)
    # ``agent.test`` expects a single ``MultiTurnEnv``; ``rollout_env`` is a
    # ``SyncMultiTurnVecEnv`` wrapping N inner envs whose state is mid-rollout
    # during training. Build a separate test env so evaluation is isolated.
    # NOTE: this means one extra env is held for the run's lifetime. Future
    # refactor could share a subset of the rollout envs (e.g. lease one of the
    # vec env's inner ``MultiTurnEnv`` instances when no trajectory is active)
    # to avoid the duplication for heavy env setups.
    test_env = env_factory(**(env_config or {}))
    group_seed = np.random.randint(0, 1_000_000)
    wall_deadline = (
        time.monotonic() + max_wall_seconds
        if max_wall_seconds is not None and max_wall_seconds > 0
        else None
    )
    while total_steps < max_steps:
        if wall_deadline is not None and time.monotonic() >= wall_deadline:
            if accelerator is None or accelerator.is_main_process:
                print(
                    f"\nStopping multiturn training: wall time limit ({max_wall_seconds}s) reached.",
                )
            break

        # Collect rollouts and learn
        iteration_steps = 0
        for agent in population.agents:
            agent.init_training_step()
            (
                completion_ids_list,
                action_masks_list,
                all_turn_ids,
                all_rewards,
                batch_steps,
                group_seed,
                all_sampling_logps,
            ) = collect_rollouts_llm(
                agent=agent,
                env=rollout_env,
                n_steps=max_turns,
                batch_size=batch_size,
                group_size=group_size,
                group_seed=group_seed,
            )

            normalized_rewards = [
                reward.unsqueeze(0) if reward.dim() == 1 else reward
                for reward in all_rewards
            ]
            (turn_ids_padded,) = stack_and_pad_experiences(
                all_turn_ids, padding_values=[-1]
            )
            (rewards_2d,) = stack_and_pad_experiences(
                normalized_rewards, padding_values=[0.0]
            )
            rewards_2d = rewards_2d.float()

            episode_scores = (
                rewards_2d.sum(dim=1) if rewards_2d.dim() > 1 else rewards_2d
            )
            mean_score = episode_scores.mean().to(agent.device)

            experiences = (
                completion_ids_list,
                action_masks_list,
                rewards_2d,
            )

            # Pass turn_ids to every multi-turn RL agent that accepts it
            # (LLMPPO/LLMREINFORCE, and the GRPO family — GRPO/CISPO/GSPO —
            # which use it for turn-level importance sampling + per-turn
            # group-relative advantages). Agents that don't need it (e.g.
            # token/sequence levels, GSPO) simply ignore it.
            learn_kwargs = (
                {"turn_ids": turn_ids_padded}
                if isinstance(agent, (LLMREINFORCE, LLMPPO, GRPO))
                else {}
            )
            if all_sampling_logps is not None and isinstance(
                agent, (GRPO, LLMPPO, LLMREINFORCE)
            ):
                learn_kwargs["sampling_logps"] = all_sampling_logps
            agent.learn(experiences, **learn_kwargs)

            agg_score = safe_aggregate_metrics(accelerator, mean_score)

            if max_reward is not None:
                if "accuracy" not in agent.metrics.additional_metrics:
                    agent.metrics.register("accuracy")

                accuracy = (
                    (episode_scores >= max_reward).float().mean().to(agent.device)
                )
                agg_accuracy = safe_aggregate_metrics(accelerator, accuracy)

                if accelerator is None or accelerator.is_main_process:
                    agent.metrics.log("accuracy", agg_accuracy)

            effective_batch_steps = batch_steps * data_increment
            agent.finalize_training_step(batch_steps)
            total_steps += effective_batch_steps
            iteration_steps += effective_batch_steps

            if accelerator is None or accelerator.is_main_process:
                agent.add_scores([float(agg_score)])

            # Evaluate performance
            if (i + 1) % evaluation_interval == 0:
                agent.test(test_env)

        # Report training metrics
        if accelerator is None or accelerator.is_main_process:
            pbar.update(iteration_steps // len(population.agents))
            population.report_metrics(clear=True)

        if accelerator is not None:
            accelerator.wait_for_everyone()

        # Tournament selection and mutation
        if tournament and mutation is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                population.update(
                    tournament_selection_and_mutation(
                        population=population.agents,
                        tournament=tournament,
                        mutation=mutation,
                        env_name=env_name,
                        accelerator=accelerator,
                        language_model=True,
                        elite_path=elite_path,
                        save_elite=save_elite,
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
            if total_steps >= max_steps and not max_steps_checkpoint_saved:
                checkpoint_due = True
                max_steps_checkpoint_saved = True
            if checkpoint_due:
                save_llm_checkpoint(
                    agent,
                    checkpoint_path if checkpoint_path is not None else elite_path,
                )

        i += 1

    if save_elite and elite_path is not None:
        elite = max(
            population.agents,
            key=lambda a: a.fitness[-1] if a.fitness else float("-inf"),
        )
        save_llm_checkpoint(elite, elite_path)

    population.finish()
    pbar.close()
    return population.agents, population.last_fitnesses
