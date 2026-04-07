import warnings
from typing import Any

import torch.distributed as dist
from accelerate import Accelerator

from agilerl.algorithms import DPO, GRPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.utils.llm_utils import (
    PreferenceGym,
    ReasoningGym,
    aggregate_metrics_across_gpus,
)
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = dict[str, Any] | None
SupportedReasoning = GRPO


def finetune_llm_reasoning(
    pop: list[SupportedReasoning],
    env: ReasoningGym,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    evo_steps: int | None = 20,
    checkpoint_steps: int | None = None,
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
) -> None:
    """Finetunes a population of GRPOs on a ReasoningGym environment.

    :param pop: Population of GRPOs to finetune
    :type pop: list[GRPO]
    :param env: ReasoningGym environment to finetune on
    :type env: ReasoningGym
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
    :param num_epochs: Number of epochs to run, if set, takes precedence over max_steps, defaults to None
    :type num_epochs: int, optional
    """
    if evo_steps is not None and (tournament is None or mutation is None):
        warnings.warn(
            "'evo_steps' is set but at least one of 'tournament' or 'mutation' is set to None. Evolution will not take place.",
            stacklevel=2,
        )

    if (tournament is not None and mutation is not None) and evo_steps is None:
        msg = "'evo_steps' must be set if 'tournament' and 'mutation' are not None."
        raise ValueError(
            msg,
        )

    if num_epochs is not None and max_steps is not None:
        warnings.warn(
            "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'.",
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

    if not isinstance(pop[0], GRPO):
        msg = (
            "The algorithm must be GRPO for reasoning-based reinforcement learning. "
            f"Got {type(pop[0])} instead."
        )
        raise ValueError(
            msg,
        )

    if init_hp is None:
        init_hp = {}
        init_hp["BATCH_SIZE_PER_GPU"] = pop[0].batch_size_per_process
        init_hp["ALGO"] = pop[0].algo

    data_increment = (
        getattr(dist, "get_world_size", lambda: 1)() if dist.is_initialized() else 1
    )
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path

    if max_steps is None and num_epochs is None:
        max_steps = len(env)
    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * len(env)

    training_steps = -(max_steps // -effective_data_batch_size)

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=init_hp.get("ALGO", "GRPO"),
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

    # calling env.reset() supplies the first batch of training data
    prompts = env.reset(reset_dataloaders=True)
    for i in range(training_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent in population.agents:
            agent.set_reference_policy(env.num_epochs)
            agent.init_evo_step()

            completion_ids, action_masks = agent.get_action(prompts)

            # Use the reward function stored in env.step to calculate reward
            next_prompts, rewards = env.step(completion_ids)

            experiences = (completion_ids, action_masks, rewards)

            _agg_loss, _agg_kl = agent.learn(experiences)

            agg_rewards = aggregate_metrics_across_gpus(accelerator, rewards)

            if max_reward is not None:
                accuracy = (rewards == max_reward).sum() / len(rewards.flatten())
                agg_accuracy = aggregate_metrics_across_gpus(accelerator, accuracy)
                agent.metrics.log("accuracy", agg_accuracy)

            agent.add_scores([float(agg_rewards)])
            agent.finalize_evo_step(env.data_batch_size_per_gpu)

        prompts = next_prompts
        pbar.update(effective_data_batch_size)
        population.increment_evo_step()

        # Evaluate periodically
        if (i + 1) % evaluation_interval == 0:
            for agent in population.agents:
                agent.test(env)
            if accelerator is not None:
                accelerator.wait_for_everyone()

            # Report metrics and clear accumulators -> clear metrics after reporting
            population.report_metrics(clear=True)

        # Tournament selection and population mutation
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

        elif (i + 1) * effective_data_batch_size % max_steps == 0 or (
            checkpoint_steps is not None
            and (i + 1) * effective_data_batch_size % checkpoint_steps == 0
        ):
            save_llm_checkpoint(agent, elite_path)

        if env.num_epochs == num_epochs:
            break

    population.finish()
    pbar.close()


SupportedPreference = DPO


def finetune_llm_preference(
    pop: list[SupportedPreference],
    env: PreferenceGym,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    tensorboard: bool = False,
    tensorboard_log_dir: str | None = None,
    evo_steps: int | None = 20,
    checkpoint_steps: int | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
    evaluation_interval: int = 10,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> None:
    """Finetunes a population of DPOs on a PreferenceGym environment.

    :param pop: Population of DPOs to finetune
    :type pop: list[SupportedPreference]
    :param env: PreferenceGym environment to finetune on
    :type env: PreferenceGym
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
    :param num_epochs: Number of epochs to run, if set, takes precedence over max_steps, defaults to None
    :type num_epochs: int, optional
    """
    if evo_steps is not None and (tournament is None or mutation is None):
        warnings.warn(
            "'evo_steps' is set but at least one of 'tournament' or 'mutation' is set to None. Evolution will not take place.",
            stacklevel=2,
        )

    if (tournament is not None and mutation is not None) and evo_steps is None:
        msg = "'evo_steps' must be set if 'tournament' and 'mutation' are not None."
        raise ValueError(
            msg,
        )
    if num_epochs is not None and max_steps is not None:
        warnings.warn(
            "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'.",
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

    if not isinstance(pop[0], DPO):
        msg = (
            "The algorithm must be DPO for preference-based reinforcement learning. "
            f"Got {type(pop[0])} instead."
        )
        raise ValueError(
            msg,
        )

    if init_hp is None:
        init_hp = {}
        init_hp["BATCH_SIZE_PER_GPU"] = pop[0].batch_size_per_process
        init_hp["ALGO"] = pop[0].algo

    data_increment = accelerator.num_processes if accelerator is not None else 1
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu

    if wb:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path

    if max_steps is None and num_epochs is None:
        max_steps = len(env)
    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * len(env)

    training_steps = -(max_steps // -effective_data_batch_size)

    pbar = default_progress_bar(max_steps, accelerator)

    loggers = init_loggers(
        algo=init_hp.get("ALGO", "DPO"),
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

    prompts = env.reset(reset_dataloaders=True)
    for i in range(training_steps):
        if accelerator is not None:
            accelerator.wait_for_everyone()

        for agent in population.agents:
            agent.set_reference_policy(env.num_epochs)
            agent.init_evo_step()

            # learn() aggregates and logs loss/chosen/rejected/margin to metrics
            _, chosen_reward, rejected_reward = agent.learn(prompts)
            next_prompts = env.step()

            agent.add_scores([float(chosen_reward - rejected_reward)])
            agent.finalize_evo_step(env.data_batch_size_per_gpu)

        prompts = next_prompts
        pbar.update(effective_data_batch_size)
        population.increment_evo_step()

        # Evaluate periodically
        if (i + 1) % evaluation_interval == 0:
            for agent in population.agents:
                agent.test(env)

            if accelerator is not None:
                accelerator.wait_for_everyone()

            # Report metrics and clear accumulators -> clear metrics after reporting
            population.report_metrics(clear=True)

        # Tournament selection and population mutation
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

        elif (i + 1) * effective_data_batch_size % max_steps == 0 or (
            checkpoint_steps is not None
            and (i + 1) * effective_data_batch_size % checkpoint_steps == 0
        ):
            save_llm_checkpoint(agent, elite_path)

        if env.num_epochs == num_epochs:
            break

    population.finish()
    pbar.close()
