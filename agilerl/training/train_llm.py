import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import trange

import wandb
from agilerl.algorithms import DPO, GRPO, LLMPPO, LLMReinforce
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.rollouts.on_policy import collect_rollouts_llm
from agilerl.typing import MultiTurnEnvType, PopulationType
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import ReasoningGym
from agilerl.utils.utils import (
    _distributed_world_size,
    aggregate_metrics_across_gpus,
    init_wandb,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
)
from agilerl.wrappers.multiturn_wrappers import SyncMultiTurnVecEnv

InitDictType = dict[str, Any] | None


def _validate_llm_evolution_args(
    evo_steps: int | None,
    tournament: TournamentSelection | None,
    mutation: Mutations | None,
) -> None:
    """Validate that evolution arguments are provided in compatible combinations."""
    if evo_steps is not None and (tournament is None or mutation is None):
        warnings.warn(
            "'evo_steps' is set but at least one of 'tournament' or 'mutation' is set to None. Evolution will not take place.",
            stacklevel=2,
        )
    if (tournament is not None and mutation is not None) and evo_steps is None:
        msg = "'evo_steps' must be set if 'tournament' and 'mutation' are not None."
        raise ValueError(msg)


def _validate_llm_mutation_probs(mutation: Mutations | None) -> None:
    """Ensure unsupported mutation operators are disabled for LLM finetuning."""
    if mutation is None:
        return
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


def _init_llm_wandb(
    init_hp: dict[str, Any],
    pop: PopulationType,
    env_name: str,
    effective_data_batch_size: int,
    wb: bool,
    wandb_api_key: str | None,
    accelerator: Accelerator | None,
    additional_fields: dict[str, Any] | None = None,
) -> None:
    """Initialize W&B run metadata for the current LLM finetuning session."""
    if not wb or (accelerator is not None and not accelerator.is_main_process):
        return
    init_hp["effective_data_batch_size"] = effective_data_batch_size
    init_hp["batch_size"] = init_hp.get("BATCH_SIZE", pop[0].batch_size)
    init_hp["distributed_training"] = accelerator is not None
    init_hp["model_name"] = pop[0].pretrained_model_name_or_path
    if additional_fields is not None:
        init_hp.update(additional_fields)
    init_wandb(
        algo=init_hp["ALGO"],
        env_name=env_name,
        wandb_api_key=wandb_api_key,
        init_hyperparams=init_hp,
    )


def _format_metric_name(metric_name: str) -> str:
    """Format snake_case metric names into display-friendly title case."""
    acronym_overrides = {"kl": "KL", "pg": "PG", "vf": "VF"}
    words = metric_name.split("_")
    return " ".join(
        acronym_overrides.get(word.lower(), word.capitalize()) for word in words
    )


def _format_prefixed_metrics(
    metrics: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Prefix and normalize metric keys for train/eval logging outputs."""
    return {
        f"{prefix}/{_format_metric_name(metric_name)}": metric
        for metric_name, metric in metrics.items()
    }


def _collect_metric_values(
    agent_metrics_dict: dict[str, dict[str, Any]],
    pop: PopulationType,
    split_name: str,
    metric_name: str,
) -> list[Any]:
    """Collect a single metric across all agents for one split."""
    values = []
    for agent_idx, _ in enumerate(pop):
        metric_bucket = agent_metrics_dict.get(f"agent_{agent_idx}/{split_name}", {})
        value = metric_bucket.get(metric_name)
        if value is not None:
            values.append(value)
    return values


def _collect_hpo_wandb_fields(pop: PopulationType) -> dict[str, Any]:
    """Build per-agent hyperparameter fields for W&B logging."""
    if len(pop[0].registry.hp_config.config.keys()) == 0:
        return {}
    return {
        f"HPO_agent_{agent_idx}/{key}": getattr(agent, key)
        for agent_idx, agent in enumerate(pop)
        for key in agent.registry.hp_config.config
    }


def _normalize_learn_metrics(
    agent: GRPO | LLMPPO | LLMReinforce | DPO,
    learn_output: dict[str, Any] | tuple[Any, ...],
    mode: str,
) -> dict[str, Any]:
    """Normalize algorithm-specific learn outputs into a common metric dict."""
    if isinstance(learn_output, dict):
        return dict(learn_output)
    if not isinstance(learn_output, tuple):
        msg = f"Expected learn() to return dict or tuple, got {type(learn_output)}."
        raise TypeError(msg)

    if mode == "preference":
        if len(learn_output) != 3:
            msg = "Preference learn() tuple output must have 3 values."
            raise ValueError(msg)
        return {
            "loss": learn_output[0],
            "mean_chosen_reward": learn_output[1],
            "mean_rejected_reward": learn_output[2],
        }

    if len(learn_output) == 2:
        return {"mean_loss": learn_output[0], "mean_kl": learn_output[1]}
    if len(learn_output) == 4:
        return {
            "mean_loss": learn_output[0],
            "mean_kl": learn_output[1],
            "pg_loss": learn_output[2],
            "entropy": learn_output[3],
        }
    if len(learn_output) == 5:
        return {
            "mean_loss": learn_output[0],
            "mean_kl": learn_output[1],
            "mean_pg_loss": learn_output[2],
            "mean_vf_loss": learn_output[3],
            "mean_entropy": learn_output[4],
        }
    msg = "Reasoning/multi-turn learn() tuple output has an unsupported shape."
    raise ValueError(msg)


def build_train_wandb_dict(
    agent_metrics_dict: dict[str, dict[str, Any]],
    pop: PopulationType,
    agent: GRPO | LLMPPO | LLMReinforce | DPO,
    max_reward: float | None = None,
    mode: str = "reasoning",
) -> dict[str, Any]:
    """Aggregate train metrics into a W&B-ready dictionary."""
    wandb_dict: dict[str, Any] = {}
    if mode == "preference":
        reward_margin = _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", "Train/Mean Reward Margin"
        )
        if reward_margin:
            wandb_dict |= {
                "Train/Best Reward Margin": np.max(reward_margin),
                "Train/Mean Population Reward Margin": np.mean(reward_margin),
            }

        for metric_name, wandb_name in (
            ("Train/Loss", "Train/Mean Population Loss"),
            ("Train/Mean Chosen Reward", "Train/Mean Population Chosen Reward"),
            ("Train/Mean Rejected Reward", "Train/Mean Population Rejected Reward"),
        ):
            values = _collect_metric_values(
                agent_metrics_dict, pop, "train_metrics", metric_name
            )
            if values:
                wandb_dict[wandb_name] = np.mean(values)
        return wandb_dict

    reward_metric = "Train/Rewards" if mode == "reasoning" else "Train/Mean Score"
    reward_values = _collect_metric_values(
        agent_metrics_dict, pop, "train_metrics", reward_metric
    )
    if reward_values:
        wandb_dict |= {
            "Train/Best Reward": np.max(reward_values),
            "Train/Mean Population Reward": np.mean(reward_values),
        }

    for metric_name, wandb_name in (
        ("Train/Mean Loss", "Train/Mean Population Loss"),
        ("Train/Mean KL", "Train/Mean Population KL Divergence"),
        ("Train/Completion Length", "Train/Mean Population Completion Length"),
    ):
        values = _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", metric_name
        )
        if values:
            wandb_dict[wandb_name] = np.mean(values)

    if isinstance(agent, (LLMPPO, LLMReinforce)):
        pg_values = _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", "Train/Mean PG Loss"
        ) or _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", "Train/PG Loss"
        )
        entropy_values = _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", "Train/Mean Entropy"
        ) or _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", "Train/Entropy"
        )
        if pg_values:
            wandb_dict["Train/Mean Population PG Loss"] = np.mean(pg_values)
        if entropy_values:
            wandb_dict["Train/Mean Population Entropy"] = np.mean(entropy_values)

    if isinstance(agent, LLMPPO):
        vf_values = _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", "Train/Mean VF Loss"
        )
        if vf_values:
            wandb_dict["Train/Mean Population Critic Loss"] = np.mean(vf_values)

    if max_reward is not None:
        accuracy_values = _collect_metric_values(
            agent_metrics_dict, pop, "train_metrics", "Train/Accuracy"
        )
        if accuracy_values:
            wandb_dict |= {
                "Train/Mean Population Accuracy": np.mean(accuracy_values),
                "Train/Best Accuracy": np.max(accuracy_values),
            }

    wandb_dict |= _collect_hpo_wandb_fields(pop)
    return wandb_dict


def build_eval_wandb_dict(
    agent_metrics_dict: dict[str, dict[str, Any]],
    pop: PopulationType,
    max_reward: float | None = None,
    mode: str = "reasoning",
    eval_score_mode: bool = False,
) -> dict[str, Any]:
    """Aggregate evaluation metrics into a W&B-ready dictionary."""
    eval_dict: dict[str, Any] = {}
    if mode == "preference":
        reward_margin = _collect_metric_values(
            agent_metrics_dict, pop, "test_metrics", "Eval/Mean Reward Margin"
        )
        if reward_margin:
            eval_dict |= {
                "Eval/Best Reward Margin": np.max(reward_margin),
                "Eval/Mean Population Reward Margin": np.mean(reward_margin),
            }
        return eval_dict

    if mode == "multiturn" or eval_score_mode:
        eval_scores = _collect_metric_values(
            agent_metrics_dict, pop, "test_metrics", "Eval/Score"
        )
        if eval_scores:
            eval_dict |= {
                "Eval/Best Score": np.max(eval_scores),
                "Eval/Mean Population Score": np.mean(eval_scores),
            }
        return eval_dict

    eval_rewards = _collect_metric_values(
        agent_metrics_dict, pop, "test_metrics", "Eval/Reward"
    )
    if eval_rewards:
        eval_dict |= {
            "Eval/Best Reward": np.max(eval_rewards),
            "Eval/Mean Population Reward": np.mean(eval_rewards),
        }
    if max_reward is not None:
        eval_accuracy = _collect_metric_values(
            agent_metrics_dict, pop, "test_metrics", "Eval/Accuracy"
        )
        if eval_accuracy:
            eval_dict |= {
                "Eval/Mean Population Accuracy": np.mean(eval_accuracy),
                "Eval/Best Accuracy": np.max(eval_accuracy),
            }
    return eval_dict


def _resolve_training_envs(
    pop: PopulationType,
    env: ReasoningGym | None,
    env_fn: Callable[[], ReasoningGym] | None,
) -> tuple[list[ReasoningGym], bool]:
    """Resolve shared or per-agent training environments.

    :param pop: Population of agents being trained.
    :type pop: PopulationType
    :param env: Shared environment instance.
    :type env: ReasoningGym | None
    :param env_fn: Factory for creating one environment per agent.
    :type env_fn: Callable[[], ReasoningGym] | None
    :return: Environment list (aligned with population) and whether env_fn mode is active.
    :rtype: tuple[list[ReasoningGym], bool]
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
            "A shared 'env' is being used with multiple agents. This can introduce fairness bias; prefer 'env_fn' for per-agent environments.",
            stacklevel=2,
        )
    assert env is not None
    return [env], False


def _num_epochs_reached(envs: list[ReasoningGym], num_epochs: int | None) -> bool:
    """Check whether all active environments have reached the epoch budget."""
    if num_epochs is None:
        return False
    epoch_counts = [getattr(training_env, "num_epochs", None) for training_env in envs]
    if not all(isinstance(epoch_count, int) for epoch_count in epoch_counts):
        return False
    return all(epoch_count >= num_epochs for epoch_count in epoch_counts)


def finetune_llm_reasoning(
    pop: PopulationType,
    env: ReasoningGym | None = None,
    env_fn: Callable[[], ReasoningGym] | None = None,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    evaluation_interval: int = 10,
    max_reward: int | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> None:
    """Finetunes a population of GRPO/LLMPPO/LLMReinforce agents on a ReasoningGym environment.

    :param pop: Population of GRPO/LLMPPO/LLMReinforce agents to finetune
    :type pop: list[GRPO | LLMPPO | LLMReinforce]
    :param env: Shared ReasoningGym environment to finetune on.
    :type env: ReasoningGym | None
    :param env_fn: Optional factory that creates one ReasoningGym environment
        per agent.
    :type env_fn: Callable[[], ReasoningGym] | None
    :param init_hp: Initial hyperparameters for the population
    :type init_hp: dict, optional
    :param save_elite: Whether to save the elite model, defaults to None
    :type save_elite: bool, optional
    :param elite_path: Path to save the elite model, defaults to None
    :type elite_path: str, optional
    :param wb: Whether to use Weights and Biases, defaults to False
    :type wb: bool, optional
    :param evo_steps: Number of steps between evolution, defaults to None
    :type evo_steps: int, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: Mutations, optional
    :param wandb_api_key: Wandb API key, defaults to None
    :type wandb_api_key: str, optional
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
    _validate_llm_evolution_args(evo_steps, tournament, mutation)
    envs, uses_env_fn = _resolve_training_envs(pop=pop, env=env, env_fn=env_fn)
    env_name = envs[0].name

    if num_epochs is not None and max_steps is not None:
        warnings.warn(
            "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'.",
            stacklevel=2,
        )

    _validate_llm_mutation_probs(mutation)

    if not isinstance(pop[0], (GRPO, LLMPPO, LLMReinforce)):
        msg = (
            "The algorithm must be GRPO, LLMPPO, or LLMReinforce for reasoning-based reinforcement learning. "
            f"Got {type(pop[0])} instead."
        )
        raise ValueError(
            msg,
        )

    if init_hp is None:
        init_hp = {}
        init_hp["BATCH_SIZE_PER_GPU"] = pop[0].batch_size_per_process
        init_hp["ALGO"] = pop[0].algo
    data_increment = _distributed_world_size(accelerator)
    effective_data_batch_size = data_increment * envs[0].data_batch_size_per_gpu

    _init_llm_wandb(
        init_hp=init_hp,
        pop=pop,
        env_name=env_name,
        effective_data_batch_size=effective_data_batch_size,
        wb=wb,
        wandb_api_key=wandb_api_key,
        accelerator=accelerator,
    )

    if accelerator is None or accelerator.is_main_process:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if max_steps is None and num_epochs is None:
        max_steps = len(envs[0])

    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * len(envs[0])

    steps_per_population_iteration = effective_data_batch_size * len(pop)
    training_steps = -(max_steps // -steps_per_population_iteration)
    if accelerator is None or accelerator.is_main_process:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    total_steps = 0
    displayed_steps = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False

    # calling env.reset() supplies the first batch of training data
    if uses_env_fn:
        prompts_by_agent = [
            training_env.reset(reset_dataloaders=True) for training_env in envs
        ]
    else:
        prompts = envs[0].reset(reset_dataloaders=True)
    for i in range(training_steps):
        agent_metrics_dict = {}
        for agent_idx, agent in enumerate(pop):
            if total_steps >= max_steps:
                break

            training_env = envs[agent_idx] if uses_env_fn else envs[0]
            current_prompts = prompts_by_agent[agent_idx] if uses_env_fn else prompts

            agent.set_reference_policy(training_env.num_epochs)
            completion_ids, action_masks = agent.get_action(current_prompts)
            completion_lengths = np.mean([x.shape[1] for x in completion_ids])

            # Use the reward function stored in env.step to calculate reward of the each answer from the group
            next_prompts, rewards = training_env.step(completion_ids)

            experiences = (
                completion_ids,
                action_masks,
                rewards,
            )

            learn_output = agent.learn(experiences)
            metrics = _normalize_learn_metrics(
                agent=agent,
                learn_output=learn_output,
                mode="reasoning",
            )
            metrics["rewards"] = rewards
            metrics["completion_length"] = completion_lengths

            if max_reward is not None:
                accuracy = (rewards == max_reward).sum() / len(rewards.flatten())
                metrics["accuracy"] = accuracy
            agg_metrics = {
                metric_name: aggregate_metrics_across_gpus(accelerator, metric)
                for metric_name, metric in metrics.items()
            }
            if uses_env_fn:
                prompts_by_agent[agent_idx] = next_prompts
            else:
                prompts = next_prompts
            agent.steps[-1] += effective_data_batch_size
            total_steps += effective_data_batch_size
            agg_test_metrics = None

            if (i + 1) % evaluation_interval == 0:
                test_reward = agent.test(training_env)
                test_metrics = {"reward": test_reward}
                if max_reward is not None:
                    test_accuracy = (test_reward == max_reward).sum() / len(
                        test_reward.flatten(),
                    )
                    test_metrics["accuracy"] = test_accuracy
                agg_test_metrics = {
                    metric_name: aggregate_metrics_across_gpus(accelerator, metric)
                    for metric_name, metric in test_metrics.items()
                }

                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if accelerator is None or accelerator.is_main_process:
                # metrics order: rewards, completion_lengths, loss, kl,
                # then (LLMPPO/Reinforce) pg, critic, entropy; optional accuracy last.
                metrics_dict = _format_prefixed_metrics(agg_metrics, "Train")
                metrics_dict["global_step"] = total_steps

                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_test_metrics is not None:
                    test_metrics_dict = _format_prefixed_metrics(
                        agg_test_metrics, "Eval"
                    )
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = (
                        test_metrics_dict
                    )
                increment = min(effective_data_batch_size, max_steps - displayed_steps)
                if increment > 0:
                    pbar.update(increment)
                    displayed_steps += increment
                agent.scores.append(agg_metrics["rewards"])

        if accelerator is not None:
            accelerator.wait_for_everyone()
        if tournament and mutation is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                pop = tournament_selection_and_mutation(
                    population=pop,
                    tournament=tournament,
                    mutation=mutation,
                    env_name=env_name,
                    accelerator=accelerator,
                    language_model=True,
                    elite_path=elite_path,
                    save_elite=save_elite,
                )
                if accelerator is not None:
                    accelerator.wait_for_everyone()
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
                save_llm_checkpoint(agent, elite_path)

        if wb and (accelerator is None or accelerator.is_main_process):
            wandb_dict = build_train_wandb_dict(
                agent_metrics_dict=agent_metrics_dict,
                pop=pop,
                agent=agent,
                max_reward=max_reward,
                mode="reasoning",
            )
            if agg_test_metrics is not None:
                wandb_dict |= build_eval_wandb_dict(
                    agent_metrics_dict=agent_metrics_dict,
                    pop=pop,
                    max_reward=max_reward,
                    mode="reasoning",
                )
            wandb.log(wandb_dict)

        if _num_epochs_reached(envs, num_epochs) or total_steps >= max_steps:
            break

    if (
        verbose
        and total_steps > evaluation_interval
        and (accelerator is None or accelerator.is_main_process)
    ):
        fitness_calculated = len(agent.fitness) > 0
        fitness = (
            [str(round(agent.fitness[-1], 2)) for agent in pop]
            if fitness_calculated
            else [None] * len(pop)
        )
        avg_fitness = (
            [f"{np.mean(agent.fitness[-5:]):.2f}" for agent in pop]
            if fitness_calculated
            else [None] * len(pop)
        )
        avg_score = [f"{np.mean(agent.scores[-10:]):.2f}" for agent in pop]
        agents = [agent.index for agent in pop]
        num_steps = [agent.steps[-1] for agent in pop]
        muts = [agent.mut for agent in pop]

        banner_text = f"Global Steps {total_steps}"
        banner_width = max(len(banner_text) + 8, 35)
        border = "=" * banner_width
        centered_text = f"{banner_text}".center(banner_width)
        pbar.write(
            f"{border}\n"
            f"{centered_text}\n"
            f"{border}\n"
            f"Fitness:\t\t{fitness}\n"
            f"Score:\t\t{agg_metrics['rewards']}\n"
            f"5 fitness avgs:\t{avg_fitness}\n"
            f"10 score avgs:\t{avg_score}\n"
            f"Agents:\t\t{agents}\n"
            f"Steps:\t\t{num_steps}\n"
            f"Mutations:\t\t{muts}",
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    if accelerator is None or accelerator.is_main_process:
        pbar.close()
        if wb:
            wandb.finish()


def finetune_llm_preference(
    pop: PopulationType,
    env: ReasoningGym | None = None,
    env_fn: Callable[[], ReasoningGym] | None = None,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    evaluation_interval: int = 10,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> None:
    """Finetune a population of DPO agents on pairwise preference data.

    Runs iterative preference updates, optional periodic evaluation, and optional
    evolutionary selection/mutation while tracking metrics for console and W&B.

    :param pop: Population of DPO agents to finetune.
    :type pop: PopulationType
    :param env: Shared preference environment that yields pairwise prompts/batches.
    :type env: ReasoningGym | None
    :param env_fn: Optional factory that creates one preference environment per agent.
    :type env_fn: Callable[[], ReasoningGym] | None
    :param init_hp: Initial hyperparameters for logging and defaults.
    :type init_hp: dict[str, Any] | None
    :param save_elite: Whether to save the elite checkpoint during evolution.
    :type save_elite: bool | None
    :param elite_path: Path used for checkpoint saving.
    :type elite_path: str | None
    :param wb: Whether to log metrics to Weights and Biases.
    :type wb: bool
    :param evo_steps: Number of outer iterations between evolution steps.
    :type evo_steps: int | None
    :param checkpoint_steps: Number of iterations between checkpoint saves when
        evolution is disabled.
    :type checkpoint_steps: int | None
    :param tournament: Tournament selection strategy for evolution.
    :type tournament: TournamentSelection | None
    :param mutation: Mutation operator used during evolution.
    :type mutation: Mutations | None
    :param wandb_api_key: Optional W&B API key.
    :type wandb_api_key: str | None
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
    """
    _validate_llm_evolution_args(evo_steps, tournament, mutation)
    envs, uses_env_fn = _resolve_training_envs(pop=pop, env=env, env_fn=env_fn)
    env_name = envs[0].name
    if num_epochs is not None and max_steps is not None:
        warnings.warn(
            "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'.",
            stacklevel=2,
        )
    _validate_llm_mutation_probs(mutation)

    if not isinstance(pop[0], DPO):
        msg = (
            "The algorithm must be DPO for preference-based reinforcement learning."
            f"Got {type(pop[0])} instead."
        )
        raise ValueError(
            msg,
        )

    if init_hp is None:
        init_hp = {}
        init_hp["BATCH_SIZE_PER_GPU"] = pop[0].batch_size_per_process
        init_hp["ALGO"] = pop[0].algo

    data_increment = _distributed_world_size(accelerator)
    effective_data_batch_size = data_increment * envs[0].data_batch_size_per_gpu

    _init_llm_wandb(
        init_hp=init_hp,
        pop=pop,
        env_name=env_name,
        effective_data_batch_size=effective_data_batch_size,
        wb=wb,
        wandb_api_key=wandb_api_key,
        accelerator=accelerator,
    )

    if accelerator is None or accelerator.is_main_process:
        pass

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if max_steps is None and num_epochs is None:
        max_steps = len(envs[0])

    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * len(envs[0])

    steps_per_population_iteration = effective_data_batch_size * len(pop)
    training_steps = -(max_steps // -steps_per_population_iteration)
    if accelerator is None or accelerator.is_main_process:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    total_steps = 0
    displayed_steps = 0
    next_checkpoint_step = checkpoint_steps
    max_steps_checkpoint_saved = False
    mean_reward_margin = 0.0

    if uses_env_fn:
        prompts_by_agent = [
            training_env.reset(reset_dataloaders=True) for training_env in envs
        ]
    else:
        prompts = envs[0].reset(reset_dataloaders=True)
    for i in range(training_steps):
        agent_metrics_dict = {}
        for agent_idx, agent in enumerate(pop):
            if total_steps >= max_steps:
                break
            training_env = envs[agent_idx] if uses_env_fn else envs[0]
            current_prompts = prompts_by_agent[agent_idx] if uses_env_fn else prompts
            agent.set_reference_policy(training_env.num_epochs)
            learn_output = agent.learn(current_prompts)
            metrics = _normalize_learn_metrics(
                agent=agent,
                learn_output=learn_output,
                mode="preference",
            )
            next_prompts = training_env.step()
            agg_metrics = {
                metric_name: aggregate_metrics_across_gpus(accelerator, metric)
                for metric_name, metric in metrics.items()
            }
            mean_reward_margin = (
                agg_metrics["mean_chosen_reward"] - agg_metrics["mean_rejected_reward"]
            )
            if uses_env_fn:
                prompts_by_agent[agent_idx] = next_prompts
            else:
                prompts = next_prompts
            agent.steps[-1] += effective_data_batch_size
            total_steps += effective_data_batch_size
            agg_test_metrics = None

            if (i + 1) % evaluation_interval == 0:
                test_reward = agent.test(training_env)
                test_metrics = {"mean_reward_margin": test_reward}
                agg_test_metrics = {
                    metric_name: aggregate_metrics_across_gpus(accelerator, metric)
                    for metric_name, metric in test_metrics.items()
                }

                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if accelerator is None or accelerator.is_main_process:
                metrics_dict = _format_prefixed_metrics(agg_metrics, "Train")
                metrics_dict["global_step"] = total_steps
                metrics_dict["Train/Mean Reward Margin"] = mean_reward_margin
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_test_metrics is not None:
                    test_metrics_dict = _format_prefixed_metrics(
                        agg_test_metrics, "Eval"
                    )
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = (
                        test_metrics_dict
                    )
                increment = min(effective_data_batch_size, max_steps - displayed_steps)
                if increment > 0:
                    pbar.update(increment)
                    displayed_steps += increment
                agent.scores.append(mean_reward_margin)
        if accelerator is not None:
            accelerator.wait_for_everyone()

        if tournament and mutation is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                pop = tournament_selection_and_mutation(
                    population=pop,
                    tournament=tournament,
                    mutation=mutation,
                    env_name=env_name,
                    accelerator=accelerator,
                    language_model=True,
                    elite_path=elite_path,
                    save_elite=save_elite,
                )
                if accelerator is not None:
                    accelerator.wait_for_everyone()
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
                save_llm_checkpoint(agent, elite_path)

        if wb and (accelerator is None or accelerator.is_main_process):
            wandb_dict = build_train_wandb_dict(
                agent_metrics_dict=agent_metrics_dict,
                pop=pop,
                agent=agent,
                mode="preference",
            )
            if agg_test_metrics is not None:
                wandb_dict |= build_eval_wandb_dict(
                    agent_metrics_dict=agent_metrics_dict,
                    pop=pop,
                    mode="preference",
                )
            wandb.log(wandb_dict)
        if _num_epochs_reached(envs, num_epochs) or total_steps >= max_steps:
            break
    if (
        verbose
        and total_steps > evaluation_interval
        and (accelerator is None or accelerator.is_main_process)
    ):
        fitness_calculated = len(agent.fitness) > 0
        fitness = (
            [str(round(agent.fitness[-1], 2)) for agent in pop]
            if fitness_calculated
            else [None] * len(pop)
        )
        avg_fitness = (
            [f"{np.mean(agent.fitness[-5:]):.2f}" for agent in pop]
            if fitness_calculated
            else [None] * len(pop)
        )
        avg_score = [f"{np.mean(agent.scores[-10:]):.2f}" for agent in pop]
        agents = [agent.index for agent in pop]
        num_steps = [agent.steps[-1] for agent in pop]
        muts = [agent.mut for agent in pop]

        banner_text = f"Global Steps {total_steps}"
        banner_width = max(len(banner_text) + 8, 35)
        border = "=" * banner_width
        centered_text = f"{banner_text}".center(banner_width)
        pbar.write(
            f"{border}\n"
            f"{centered_text}\n"
            f"{border}\n"
            f"Fitness:\t\t{fitness}\n"
            f"Score:\t\t{mean_reward_margin}\n"
            f"5 fitness avgs:\t{avg_fitness}\n"
            f"10 score avgs:\t{avg_score}\n"
            f"Agents:\t\t{agents}\n"
            f"Steps:\t\t{num_steps}\n"
            f"Mutations:\t\t{muts}",
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    if accelerator is None or accelerator.is_main_process:
        pbar.close()
        if wb:
            wandb.finish()


def finetune_llm_multiturn(
    pop: PopulationType,
    max_turns: int,
    env_factory: Callable[[], MultiTurnEnvType],
    env_config: dict[str, Any] | None = None,
    init_hp: dict[str, Any] | None = None,
    max_steps: int = 32768,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    eval_fn: Callable[[LLMPPO | LLMReinforce], float] | None = None,
    evaluation_interval: int = 50,
    max_reward: float | None = None,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
) -> PopulationType:
    """Finetune a population of LLMPPO agents on a multi-turn GEM environment.

    Collects token-level episodes (``reset`` returns ``(obs, info)``,
    repeated ``get_action`` / ``step`` (full completion tensor), then
    ``get_episode_data``), then runs turn-level PPO updates. For
    ``TokenObservationWrapper`` with ``max_model_len`` set, sliding-window
    prompt fields are included in each observation before generation.

    :param pop: Population of LLMPPO agents to finetune.
    :type pop: PopulationType
    :param max_turns: Maximum interaction turns per episode.
    :type max_turns: int
    :param env_factory: Factory that returns a fresh multi-turn env for each
        trajectory rollout. Required to ensure trajectory state isolation.
    :type env_factory: Callable[[], GemEnv]
    :param env_config: Configuration for the environment factory.
    :type env_config: dict[str, Any], optional
    :param init_hp: Initial hyperparameters (e.g. ``BATCH_SIZE``, ``ALGO``).
    :type init_hp: dict, optional
    :param max_steps: Progress-bar / outer-loop budget in sample steps, defaults to 32768.
    :type max_steps: int, optional
    :param save_elite: Whether to save the elite checkpoint, defaults to None.
    :type save_elite: bool, optional
    :param elite_path: Directory or path prefix for checkpoints, defaults to None.
    :type elite_path: str, optional
    :param wb: Whether to log to Weights and Biases, defaults to False.
    :type wb: bool, optional
    :param evo_steps: Steps between evolution (requires tournament and mutation).
    :type evo_steps: int, optional
    :param checkpoint_steps: Save checkpoint every N outer iterations when no evolution.
    :type checkpoint_steps: int, optional
    :param tournament: Tournament selection for evolution, defaults to None.
    :type tournament: TournamentSelection, optional
    :param mutation: Mutation operator for evolution, defaults to None.
    :type mutation: Mutations, optional
    :param wandb_api_key: W&B API key, defaults to None.
    :type wandb_api_key: str, optional
    :param eval_fn: Optional ``(agent) -> float`` evaluated on the main process.
    :type eval_fn: Callable[[LLMPPO], float], optional
    :param evaluation_interval: How often to run ``eval_fn`` and verbose banners.
    :type evaluation_interval: int, optional
    :param max_reward: If set, adds accuracy metric vs this threshold.
    :type max_reward: float, optional
    :param verbose: Progress bar and periodic train summaries, defaults to True.
    :type verbose: bool, optional
    :param accelerator: Hugging Face Accelerate instance, defaults to None.
    :type accelerator: Accelerator, optional
    :return: The finetuned population (same list object, possibly mutated in place).
    :rtype: PopulationType
    """
    _validate_llm_evolution_args(evo_steps, tournament, mutation)
    _validate_llm_mutation_probs(mutation)

    if not isinstance(pop[0], (LLMPPO, LLMReinforce, GRPO)):
        msg = (
            "The algorithm must be LLMPPO, LLMReinforce, or GRPO for multi-turn GEM finetuning. "
            f"Got {type(pop[0])} instead."
        )
        raise ValueError(
            msg,
        )

    if init_hp is None:
        init_hp = {}
        init_hp["BATCH_SIZE_PER_GPU"] = pop[0].batch_size_per_process
        init_hp["ALGO"] = pop[0].algo

    batch_size = init_hp.get("BATCH_SIZE", pop[0].batch_size)
    for agent in pop:
        effective_group_size = getattr(agent, "group_size", 1)
        if isinstance(agent, GRPO):
            if (
                batch_size > effective_group_size
                and batch_size % effective_group_size != 0
            ):
                msg = (
                    f"Batch size ({batch_size}) must be divisible by "
                    f"group_size ({effective_group_size}) for GRPO when group size is greater than batch size."
                )
                raise ValueError(msg)

            if (
                batch_size < effective_group_size
                and effective_group_size % batch_size != 0
            ):
                msg = (
                    f"Group size ({effective_group_size}) must be divisible by "
                    f"batch size ({batch_size}) for GRPO when batch size is less than group size."
                )
                raise ValueError(msg)

    env_name = init_hp.get("env_name", "gem_multiturn")
    data_increment = _distributed_world_size(accelerator)
    effective_data_batch_size = data_increment * batch_size

    _init_llm_wandb(
        init_hp=init_hp,
        pop=pop,
        env_name=env_name,
        effective_data_batch_size=effective_data_batch_size,
        wb=wb,
        wandb_api_key=wandb_api_key,
        accelerator=accelerator,
        additional_fields={"max_turns": max_turns, "batch_size": batch_size},
    )

    if accelerator is None or accelerator.is_main_process:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is None or accelerator.is_main_process:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    total_steps = 0
    agg_metrics: dict[str, Any] = {}
    agg_eval_score: float | None = None
    group_seed = np.random.randint(0, 1000000)
    i = 0
    group_size = getattr(pop[0], "group_size", 1)
    rollout_env = SyncMultiTurnVecEnv(env_factory, batch_size, group_size, env_config)
    while total_steps < max_steps:
        agent_metrics_dict = {}
        iteration_steps = 0
        for agent_idx, agent in enumerate(pop):
            (
                completion_ids_list,
                action_masks_list,
                all_turn_ids,
                all_rewards,
                batch_steps,
                group_seed,
            ) = collect_rollouts_llm(
                agent=agent,
                env=rollout_env,
                n_steps=max_turns,
                batch_size=batch_size,
                group_size=group_size,
                group_seed=group_seed,
            )

            # Normalize rewards to 2D [1, n_turns] per trajectory so padding
            # stacks into [batch, max_turns] rather than flattening to 1D.
            normalized_rewards = [
                reward.unsqueeze(0) if reward.dim() == 1 else reward
                for reward in all_rewards
            ]
            (turn_ids_padded,) = stack_and_pad_experiences(
                all_turn_ids,
                padding_values=[-1],
            )
            (rewards_2d,) = stack_and_pad_experiences(
                normalized_rewards,
                padding_values=[0.0],
            )
            rewards_2d = rewards_2d.float()
            completion_lengths = np.mean([x.shape[1] for x in completion_ids_list])
            episode_scores = (
                rewards_2d.sum(dim=1) if rewards_2d.dim() > 1 else rewards_2d
            )
            mean_score = episode_scores.mean().to(agent.device)

            experiences = (
                completion_ids_list,
                action_masks_list,
                rewards_2d,
            )

            learn_kwargs = (
                {"turn_ids": turn_ids_padded}
                if isinstance(agent, (LLMReinforce, LLMPPO))
                else {}
            )
            learn_output = agent.learn(experiences, **learn_kwargs)
            metrics = _normalize_learn_metrics(
                agent=agent,
                learn_output=learn_output,
                mode="multiturn",
            )
            metrics["mean_score"] = mean_score
            metrics["completion_length"] = torch.tensor(
                completion_lengths, dtype=torch.float32, device=agent.device
            )
            if max_reward is not None:
                accuracy = (
                    (episode_scores >= max_reward).float().mean().to(agent.device)
                )
                metrics["accuracy"] = accuracy
            agg_metrics = {
                metric_name: aggregate_metrics_across_gpus(accelerator, metric)
                for metric_name, metric in metrics.items()
            }

            effective_batch_steps = batch_steps * data_increment
            agent.steps[-1] += effective_batch_steps
            total_steps += effective_batch_steps
            iteration_steps += effective_batch_steps
            agg_eval_score = None

            if (i + 1) % evaluation_interval == 0 and eval_fn is not None:
                eval_score = eval_fn(agent)
                eval_tensor = torch.tensor(
                    eval_score, dtype=torch.float32, device=agent.device
                )
                agg_eval_score = aggregate_metrics_across_gpus(accelerator, eval_tensor)
                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if accelerator is None or accelerator.is_main_process:
                metrics_dict = _format_prefixed_metrics(agg_metrics, "Train")
                metrics_dict["global_step"] = total_steps
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_eval_score is not None:
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = {
                        "Eval/Score": agg_eval_score,
                    }
                agent.scores.append(agg_metrics["mean_score"])

        if (
            verbose
            and (i + 1) % evaluation_interval == 0
            and (accelerator is None or accelerator.is_main_process)
        ):
            banner_text = f"Step {i + 1}  ({total_steps} samples)"
            banner_width = max(len(banner_text) + 8, 40)
            border = "=" * banner_width
            lines = [
                f"\n{border}",
                banner_text.center(banner_width),
                border,
                f"Train score:\t\t{agg_metrics['mean_score']:.3f}",
                f"Loss:\t\t\t{agg_metrics['mean_loss']:.4f}",
                f"KL-divergence:\t\t{agg_metrics['mean_kl']:.4f}",
            ]
            if "mean_pg_loss" in agg_metrics:
                lines.extend(
                    [
                        f"PG loss:\t\t{agg_metrics['mean_pg_loss']:.4f}",
                        f"VF loss:\t\t{agg_metrics.get('mean_vf_loss', 0.0):.4f}",
                        f"Entropy:\t\t{agg_metrics['mean_entropy']:.4f}",
                    ]
                )
            if max_reward is not None:
                lines.insert(4, f"Train accuracy:\t\t{agg_metrics['accuracy']:.3f}")
            if agg_eval_score is not None:
                lines.append(f"Eval score:\t\t{agg_eval_score:.3f}")
            lines.append(border)
            pbar.write("\n".join(lines))

        if accelerator is None or accelerator.is_main_process:
            postfix = {
                "loss": f"{np.mean([agent_metrics_dict[f'agent_{j}/train_metrics']['Train/Mean Loss'] for j in range(len(pop))]):.4f}",
                "kl": f"{np.mean([agent_metrics_dict[f'agent_{j}/train_metrics']['Train/Mean KL'] for j in range(len(pop))]):.4f}",
                "score": f"{np.mean([agent_metrics_dict[f'agent_{j}/train_metrics']['Train/Mean Score'] for j in range(len(pop))]):.3f}",
            }
            if max_reward is not None:
                postfix["acc"] = (
                    f"{np.mean([agent_metrics_dict[f'agent_{j}/train_metrics']['Train/Accuracy'] for j in range(len(pop))]):.3f}"
                )
            pbar.set_postfix(**postfix)
            pbar.update(iteration_steps // len(pop))

        if accelerator is not None:
            accelerator.wait_for_everyone()

        if tournament is not None and mutation is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                pop = tournament_selection_and_mutation(
                    population=pop,
                    tournament=tournament,
                    mutation=mutation,
                    env_name=env_name,
                    accelerator=accelerator,
                    language_model=True,
                    elite_path=elite_path,
                    save_elite=save_elite,
                )
                if accelerator is not None:
                    accelerator.wait_for_everyone()
        elif total_steps >= max_steps or (
            checkpoint_steps is not None and (i + 1) % checkpoint_steps == 0
        ):
            save_llm_checkpoint(agent, elite_path)

        if wb and (accelerator is None or accelerator.is_main_process):
            wandb_dict = build_train_wandb_dict(
                agent_metrics_dict=agent_metrics_dict,
                pop=pop,
                agent=agent,
                max_reward=max_reward,
                mode="multiturn",
            )
            eval_wandb_dict = build_eval_wandb_dict(
                agent_metrics_dict=agent_metrics_dict,
                pop=pop,
                mode="multiturn",
                eval_score_mode=True,
            )
            wandb_dict |= eval_wandb_dict

            wandb.log(wandb_dict)

        i += 1

    if (
        verbose
        and total_steps > evaluation_interval
        and (accelerator is None or accelerator.is_main_process)
    ):
        fitness_calculated = len(agent.fitness) > 0
        fitness = (
            [str(round(agent.fitness[-1], 2)) for agent in pop]
            if fitness_calculated
            else [None] * len(pop)
        )
        avg_fitness = (
            [f"{np.mean(agent.fitness[-5:]):.2f}" for agent in pop]
            if fitness_calculated
            else [None] * len(pop)
        )
        avg_score = [f"{np.mean(agent.scores[-10:]):.2f}" for agent in pop]
        agents = [agent.index for agent in pop]
        num_steps = [agent.steps[-1] for agent in pop]
        muts = [agent.mut for agent in pop]

        banner_text = f"Global Steps {total_steps}"
        banner_width = max(len(banner_text) + 8, 35)
        border = "=" * banner_width
        centered_text = f"{banner_text}".center(banner_width)
        pbar.write(
            f"{border}\n"
            f"{centered_text}\n"
            f"{border}\n"
            f"Fitness:\t\t{fitness}\n"
            f"Score:\t\t{agg_metrics['mean_score']}\n"
            f"5 fitness avgs:\t{avg_fitness}\n"
            f"10 score avgs:\t{avg_score}\n"
            f"Agents:\t\t{agents}\n"
            f"Steps:\t\t{num_steps}\n"
            f"Mutations:\t\t{muts}",
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    if accelerator is None or accelerator.is_main_process:
        pbar.close()
        if wb:
            wandb.finish()

    return pop
