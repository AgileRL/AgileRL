import csv
import os
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from tqdm import trange

from agilerl.algorithms import DPO, GRPO
from agilerl.algorithms.sft import SFT
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.typing import PopulationType
from agilerl.utils.llm_utils import ReasoningGym, SFTGym
from agilerl.utils.utils import (
    aggregate_metrics_across_gpus,
    init_wandb,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = dict[str, Any] | None


def safe_aggregate_metrics(
    accelerator: Accelerator | None, metrics: list[float]
) -> float:
    if accelerator is None:
        return float(metrics) if not isinstance(metrics, float) else metrics
    return aggregate_metrics_across_gpus(accelerator, metrics)


def _is_main_process(accelerator: Accelerator | None) -> bool:
    return accelerator is None or accelerator.is_main_process


def _validate_finetune_args(
    evo_steps: int | None,
    tournament: TournamentSelection | None,
    mutation: Mutations | None,
    num_epochs: int | None,
    max_steps: int | None,
    pop: PopulationType,
    expected_type: type,
    algorithm_type_error: str,
    *,
    algo: Literal["grpo", "dpo", "sft"],
) -> None:
    if algo in ["grpo", "dpo"]:
        if evo_steps is not None and (tournament is None or mutation is None):
            warnings.warn(
                "'evo_steps' is set but at least one of 'tournament' or 'mutation' is set to None. Evolution will not take place.",
                stacklevel=2,
            )
        if (tournament is not None and mutation is not None) and evo_steps is None:
            msg = "'evo_steps' must be set if 'tournament' and 'mutation' are not None."
            raise ValueError(msg)
        if num_epochs is not None and max_steps is not None:
            warnings.warn(
                "'num_epochs' is set but 'max_steps' is also set. 'num_epochs' will take precedence over 'max_steps'.",
                stacklevel=2,
            )
    else:
        if evo_steps is not None and (tournament is None or mutation is None):
            warnings.warn(
                "'evo_steps' is set but 'tournament' or 'mutation' is None. "
                "Evolution will not take place.",
                stacklevel=2,
            )
        if (tournament is not None and mutation is not None) and evo_steps is None:
            msg = (
                "'evo_steps' must be set when 'tournament' and 'mutation' are not None."
            )
            raise ValueError(msg)
        if num_epochs is not None and max_steps is not None:
            warnings.warn(
                "'num_epochs' overrides 'max_steps'.",
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


def _setup_wandb(
    accelerator: Accelerator | None,
    use_wandb: bool,
    init_hp: dict[str, Any],
    env_name: str,
    wandb_api_key: str | None,
    pop0: Any,
    effective_data_batch_size: int,
    *,
    project: str = "AgileRL",
    addl_args: dict[str, Any] | None = None,
) -> None:
    if not use_wandb or not _is_main_process(accelerator):
        return
    init_hp["effective_data_batch_size"] = effective_data_batch_size
    init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
    init_hp["distributed_training"] = accelerator is not None
    init_hp["model_name"] = getattr(pop0, "pretrained_model_name_or_path", "")
    init_wandb(
        algo=init_hp["ALGO"],
        env_name=env_name,
        wandb_api_key=wandb_api_key,
        init_hyperparams=init_hp,
        project=project,
        addl_args=addl_args,
    )


def _compute_training_steps(
    max_steps: int | None,
    num_epochs: int | None,
    env_len: int,
    effective_data_batch_size: int,
) -> tuple[int, int]:
    if max_steps is None and num_epochs is None:
        max_steps = env_len
    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * env_len
    assert max_steps is not None
    training_steps = -(max_steps // -effective_data_batch_size)
    return max_steps, training_steps


def _create_pbar(
    accelerator: Accelerator | None,
    max_steps: int,
    *,
    bar_format: str | None = None,
) -> Any:
    default_fmt = (
        "{l_bar}{bar:10}| {n:4}/{total_fmt} "
        "[{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    )
    fmt = bar_format if bar_format is not None else default_fmt
    if not _is_main_process(accelerator):
        return None
    return trange(
        max_steps,
        unit="step",
        bar_format=fmt,
        ascii=True,
        dynamic_ncols=True,
    )


def _per_agent_metrics(
    agent_metrics_dict: dict[str, Any], n_agents: int
) -> Callable[[str, str], list[Any]]:
    """Return ``metric_values(group, key)`` -> list of that metric across agents."""

    def metric_values(group: str, key: str) -> list[Any]:
        return [agent_metrics_dict[f"agent_{i}/{group}"][key] for i in range(n_agents)]

    return metric_values


def _wandb_extend_hpo_hyperparams(
    wandb_dict: dict[str, Any], pop: PopulationType
) -> None:
    if not pop[0].registry.hp_config.config.keys():
        return
    wandb_dict |= {
        f"HPO_agent_{agent_idx}/{key}": getattr(agent, key)
        for agent_idx, agent in enumerate(pop)
        for key in agent.registry.hp_config.config
    }


def _wandb_log(metrics: dict[str, Any]) -> None:
    """Log to W&B; call only from ``if wb and _is_main_process(accelerator)`` blocks."""
    wandb.log(metrics)


def _handle_evolution_or_checkpoint(
    i: int,
    pop: PopulationType,
    evo_steps: int | None,
    tournament: TournamentSelection | None,
    mutation: Mutations | None,
    env_name: str,
    accelerator: Accelerator | None,
    elite_path: str | None,
    save_elite: bool | None,
    effective_data_batch_size: int,
    max_steps: int,
    checkpoint_steps: int | None,
    agent: Any,
    *,
    require_evo_steps_for_hpo: bool,
) -> PopulationType:
    if require_evo_steps_for_hpo:
        if (
            tournament is not None
            and mutation is not None
            and evo_steps is not None
            and (i + 1) % evo_steps == 0
        ):
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
        elif (i + 1) * effective_data_batch_size % max_steps == 0 or (
            checkpoint_steps is not None
            and (i + 1) * effective_data_batch_size % checkpoint_steps == 0
        ):
            save_llm_checkpoint(agent, elite_path)
    elif tournament and mutation is not None:
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
    elif (i + 1) * effective_data_batch_size % max_steps == 0 or (
        checkpoint_steps is not None
        and (i + 1) * effective_data_batch_size % checkpoint_steps == 0
    ):
        save_llm_checkpoint(agent, elite_path)
    return pop


def _save_elite_checkpoint(
    pop: PopulationType,
    save_elite: bool | None,
    elite_path: str | None,
    accelerator: Accelerator | None,
) -> None:
    if save_elite and elite_path is not None:
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if _is_main_process(accelerator):
            elite = max(
                pop, key=lambda a: a.fitness[-1] if a.fitness else float("-inf")
            )
            save_llm_checkpoint(elite, elite_path)


def _finalize_training_pbar_wandb(
    accelerator: Accelerator | None,
    pbar: Any,
    use_wandb: bool,
) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if _is_main_process(accelerator):
        pbar.close()
        if use_wandb:
            wandb.finish()


def _open_csv_log(
    elite_path: str | None,
    fieldnames: list[str],
    accelerator: Accelerator | None,
) -> tuple[Any, Any]:
    if elite_path is None or not _is_main_process(accelerator):
        return None, None
    os.makedirs(elite_path, exist_ok=True)
    csv_path = os.path.join(elite_path, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    return csv_file, writer


def _log_csv_row(writer: Any, csv_file: Any, row_dict: dict[str, Any]) -> None:
    if writer is not None:
        writer.writerow(row_dict)
        csv_file.flush()


def finetune_llm_reasoning(
    pop: PopulationType,
    env: ReasoningGym,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    evo_steps: int | None = 20,
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
    _validate_finetune_args(
        evo_steps,
        tournament,
        mutation,
        num_epochs,
        max_steps,
        pop,
        GRPO,
        (
            "The algorithm must be GRPO for reasoning-based reinforcement learning."
            f"Got {type(pop[0])} instead."
        ),
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
    data_increment = (
        getattr(dist, "get_world_size", lambda: 1)() if dist.is_initialized() else 1
    )
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu

    _setup_wandb(
        accelerator,
        wb,
        init_hp,
        env.name,
        wandb_api_key,
        pop[0],
        effective_data_batch_size,
    )

    if _is_main_process(accelerator):
        print("\nTraining...")

    max_steps, training_steps = _compute_training_steps(
        max_steps, num_epochs, len(env), effective_data_batch_size
    )
    pbar = _create_pbar(accelerator, max_steps)

    total_steps = 0

    # calling env.reset() supplies the first batch of training data
    prompts = env.reset(reset_dataloaders=True)
    for i in range(training_steps):
        agent_metrics_dict = {}
        for agent_idx, agent in enumerate(pop):
            agent.set_reference_policy(env.num_epochs)
            completion_ids, action_masks = agent.get_action(prompts)
            completion_lengths = np.mean([x.shape[1] for x in completion_ids])

            # Use the reward function stored in env.step to calculate reward of the each answer from the group
            next_prompts, rewards = env.step(completion_ids)

            experiences = (
                completion_ids,
                action_masks,
                rewards,
            )
            loss, kl = agent.learn(experiences)
            metrics = [loss, kl, rewards, completion_lengths]
            if max_reward is not None:
                accuracy = (rewards == max_reward).sum() / len(rewards.flatten())
                metrics.append(accuracy)
            agg_metrics = [
                aggregate_metrics_across_gpus(accelerator, metric) for metric in metrics
            ]
            prompts = next_prompts
            agent.steps[-1] += effective_data_batch_size
            total_steps += effective_data_batch_size
            agg_test_metrics = None

            if (i + 1) % evaluation_interval == 0:
                test_reward = agent.test(env)
                test_metrics = [test_reward]
                if max_reward is not None:
                    test_accuracy = (test_reward == max_reward).sum() / len(
                        test_reward.flatten(),
                    )
                    test_metrics.append(test_accuracy)
                agg_test_metrics = [
                    aggregate_metrics_across_gpus(accelerator, metric)
                    for metric in test_metrics
                ]

                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if _is_main_process(accelerator):
                metrics_dict = {
                    "global_step": total_steps,
                    "Train/Loss": agg_metrics[0],
                    "Train/KL-divergence": agg_metrics[1],
                    "Train/Mean reward": agg_metrics[2],
                    "Train/Average completion length": int(agg_metrics[3]),
                }
                if max_reward is not None:
                    metrics_dict |= {"Train/Accuracy": agg_metrics[4]}
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_test_metrics is not None:
                    test_metrics_dict = {"Eval/Mean reward": agg_test_metrics[0]}
                    if max_reward is not None:
                        test_metrics_dict |= {"Eval/Accuracy": agg_test_metrics[1]}
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = (
                        test_metrics_dict
                    )
                pbar.update(effective_data_batch_size)
                agent.scores.append(agg_metrics[2])

        if accelerator is not None:
            accelerator.wait_for_everyone()
        pop = _handle_evolution_or_checkpoint(
            i,
            pop,
            evo_steps,
            tournament,
            mutation,
            env.name,
            accelerator,
            elite_path,
            save_elite,
            effective_data_batch_size,
            max_steps,
            checkpoint_steps,
            agent,
            require_evo_steps_for_hpo=False,
        )

        if wb and _is_main_process(accelerator):
            metric_values = _per_agent_metrics(agent_metrics_dict, len(pop))
            wandb_dict = {
                "Train/Best reward": np.max(
                    metric_values("train_metrics", "Train/Mean reward")
                ),
                "Train/Mean population reward": np.mean(
                    metric_values("train_metrics", "Train/Mean reward")
                ),
                "Train/Mean population loss": np.mean(
                    metric_values("train_metrics", "Train/Loss")
                ),
                "Train/Mean population KL divergence": np.mean(
                    metric_values("train_metrics", "Train/KL-divergence")
                ),
                "Train/Mean population completion length": np.mean(
                    metric_values("train_metrics", "Train/Average completion length")
                ),
            }
            if max_reward is not None:
                wandb_dict |= {
                    "Train/Mean population accuracy": np.mean(
                        metric_values("train_metrics", "Train/Accuracy")
                    ),
                    "Train/Best accuracy": np.max(
                        metric_values("train_metrics", "Train/Accuracy")
                    ),
                }
            _wandb_extend_hpo_hyperparams(wandb_dict, pop)

            if agg_test_metrics is not None:
                test_dict = {
                    "Eval/Best reward": np.max(
                        metric_values("test_metrics", "Eval/Mean reward")
                    ),
                    "Eval/Mean population reward": np.mean(
                        metric_values("test_metrics", "Eval/Mean reward")
                    ),
                }
                if max_reward is not None:
                    test_dict |= {
                        "Eval/Mean population accuracy": np.mean(
                            metric_values("test_metrics", "Eval/Accuracy")
                        ),
                    }
                    wandb_dict |= {
                        "Eval/Best accuracy": np.max(
                            metric_values("test_metrics", "Eval/Accuracy")
                        ),
                    }
                wandb_dict |= test_dict
            _wandb_log(wandb_dict)

        if env.num_epochs == num_epochs:
            break

    if verbose and total_steps > evaluation_interval and _is_main_process(accelerator):
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
            f"Score:\t\t{agg_metrics[2]}\n"
            f"5 fitness avgs:\t{avg_fitness}\n"
            f"10 score avgs:\t{avg_score}\n"
            f"Agents:\t\t{agents}\n"
            f"Steps:\t\t{num_steps}\n"
            f"Mutations:\t\t{muts}",
        )

    _save_elite_checkpoint(pop, save_elite, elite_path, accelerator)
    _finalize_training_pbar_wandb(accelerator, pbar, wb)


def finetune_llm_preference(
    pop: PopulationType,
    env: ReasoningGym,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    evo_steps: int | None = 20,
    checkpoint_steps: int | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_project: str = "AgileRL",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    evaluation_interval: int = 10,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> None:
    _validate_finetune_args(
        evo_steps,
        tournament,
        mutation,
        num_epochs,
        max_steps,
        pop,
        DPO,
        (
            "The algorithm must be DPO for preference-based reinforcement learning."
            f"Got {type(pop[0])} instead."
        ),
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
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu

    _setup_wandb(
        accelerator,
        wb,
        init_hp,
        env.name,
        wandb_api_key,
        pop[0],
        effective_data_batch_size,
        project=wandb_project,
        addl_args={
            **({"name": wandb_run_name} if wandb_run_name is not None else {}),
            **({"entity": wandb_entity} if wandb_entity is not None else {}),
        }
        or None,
    )

    max_steps, training_steps = _compute_training_steps(
        max_steps, num_epochs, len(env), effective_data_batch_size
    )
    pbar = _create_pbar(accelerator, max_steps)

    total_steps = 0

    dpo_csv_file, dpo_csv_writer = _open_csv_log(
        elite_path,
        [
            "step",
            "train_loss",
            "train_chosen_reward",
            "train_rejected_reward",
            "train_reward_margin",
            "eval_reward_margin",
        ],
        accelerator,
    )

    prompts = env.reset(reset_dataloaders=True)
    for i in range(training_steps):
        agent_metrics_dict = {}
        for agent_idx, agent in enumerate(pop):
            agent.set_reference_policy(env.num_epochs)
            loss, chosen_reward, rejected_reward = agent.learn(prompts)
            next_prompts = env.step()
            agg_metrics = [
                safe_aggregate_metrics(accelerator, loss),
                safe_aggregate_metrics(accelerator, chosen_reward),
                safe_aggregate_metrics(accelerator, rejected_reward),
            ]
            prompts = next_prompts
            agent.steps[-1] += effective_data_batch_size
            total_steps += effective_data_batch_size
            agg_test_metrics = None

            if (i + 1) % evaluation_interval == 0:
                test_reward = agent.test(env)
                agg_test_metrics = [safe_aggregate_metrics(accelerator, test_reward)]

                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if _is_main_process(accelerator):
                metrics_dict = {
                    "global_step": total_steps,
                    "Train/Loss": agg_metrics[0],
                    "Train/Mean chosen reward": agg_metrics[1],
                    "Train/Mean rejected reward": agg_metrics[2],
                    "Train/Mean reward margin": agg_metrics[1] - agg_metrics[2],
                }
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_test_metrics is not None:
                    test_metrics_dict = {
                        "Eval/Mean reward margin": agg_test_metrics[0],
                    }
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = (
                        test_metrics_dict
                    )
                pbar.update(effective_data_batch_size)
                pbar.set_postfix(
                    loss=f"{agg_metrics[0]:.4f}",
                    chosen=f"{agg_metrics[1]:.4f}",
                    rejected=f"{agg_metrics[2]:.4f}",
                    margin=f"{agg_metrics[1] - agg_metrics[2]:.4f}",
                    **(
                        {"eval_margin": f"{agg_test_metrics[0]:.4f}"}
                        if agg_test_metrics is not None
                        else {}
                    ),
                )
                agent.scores.append(agg_metrics[1] - agg_metrics[2])

        if dpo_csv_writer is not None and agent_metrics_dict:
            metric_values = _per_agent_metrics(agent_metrics_dict, len(pop))
            eval_margin = (
                np.mean(metric_values("test_metrics", "Eval/Mean reward margin"))
                if agg_test_metrics is not None
                else ""
            )
            _log_csv_row(
                dpo_csv_writer,
                dpo_csv_file,
                {
                    "step": total_steps,
                    "train_loss": np.mean(metric_values("train_metrics", "Train/Loss")),
                    "train_chosen_reward": np.mean(
                        metric_values("train_metrics", "Train/Mean chosen reward")
                    ),
                    "train_rejected_reward": np.mean(
                        metric_values("train_metrics", "Train/Mean rejected reward")
                    ),
                    "train_reward_margin": np.mean(
                        metric_values("train_metrics", "Train/Mean reward margin")
                    ),
                    "eval_reward_margin": eval_margin,
                },
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

        pop = _handle_evolution_or_checkpoint(
            i,
            pop,
            evo_steps,
            tournament,
            mutation,
            env.name,
            accelerator,
            elite_path,
            save_elite,
            effective_data_batch_size,
            max_steps,
            checkpoint_steps,
            agent,
            require_evo_steps_for_hpo=False,
        )

        if wb and _is_main_process(accelerator):
            metric_values = _per_agent_metrics(agent_metrics_dict, len(pop))
            wandb_dict = {
                "Train/Best reward margin": np.max(
                    metric_values("train_metrics", "Train/Mean reward margin")
                ),
                "Train/Mean population reward margin": np.mean(
                    metric_values("train_metrics", "Train/Mean reward margin")
                ),
                "Train/Mean population loss": np.mean(
                    metric_values("train_metrics", "Train/Loss")
                ),
                "Train/Mean population chosen reward": np.mean(
                    metric_values("train_metrics", "Train/Mean chosen reward")
                ),
                "Train/Mean population rejected reward": np.mean(
                    metric_values("train_metrics", "Train/Mean rejected reward")
                ),
            }
            if agg_test_metrics is not None:
                wandb_dict |= {
                    "Eval/Best reward margin": np.max(
                        metric_values("test_metrics", "Eval/Mean reward margin")
                    ),
                    "Eval/Mean population reward margin": np.mean(
                        metric_values("test_metrics", "Eval/Mean reward margin")
                    ),
                }
            _wandb_log(wandb_dict)
        if env.num_epochs == num_epochs:
            break
    if verbose and _is_main_process(accelerator):
        agent = pop[0]
        scores = agent.scores  # reward margin per step
        fitness = agent.fitness  # eval reward margin per eval step

        banner_text = f"Training complete — {total_steps} steps"
        banner_width = max(len(banner_text) + 4, 40)
        border = "=" * banner_width
        lines = [border, banner_text.center(banner_width), border]

        if scores:
            lines += [
                f"  Reward margin — initial: {scores[0]:.4f}  "
                f"final: {scores[-1]:.4f}  "
                f"best: {max(scores):.4f}  "
                f"mean: {np.mean(scores):.4f}",
            ]
        if fitness:
            lines += [
                f"  Eval margin  — final: {fitness[-1]:.4f}  best: {max(fitness):.4f}",
            ]
        if evo_steps is not None:
            muts = [a.mut for a in pop]
            agents = [a.index for a in pop]
            lines += [f"  Agents: {agents}  Mutations: {muts}"]

        pbar.write("\n".join(lines))

    _save_elite_checkpoint(pop, save_elite, elite_path, accelerator)

    if dpo_csv_file is not None:
        dpo_csv_file.close()
        print(f"Training metrics saved to {os.path.join(elite_path, 'metrics.csv')}")

    _finalize_training_pbar_wandb(accelerator, pbar, wb)


def finetune_llm_sft(
    pop: PopulationType,
    env: SFTGym,
    init_hp: dict[str, Any] | None = None,
    save_elite: bool | None = None,
    elite_path: str | None = None,
    wb: bool = False,
    evo_steps: int | None = None,
    checkpoint_steps: int | None = None,
    tournament: TournamentSelection | None = None,
    mutation: Mutations | None = None,
    wandb_api_key: str | None = None,
    wandb_project: str = "AgileRL",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    evaluation_interval: int = 10,
    verbose: bool = True,
    accelerator: Accelerator | None = None,
    max_steps: int | None = None,
    num_epochs: int | None = None,
) -> None:
    """Finetune a population of SFT agents on (prompt, response) pairs.

    Each training step draws a batch from ``env`` and minimises the cross-entropy
    loss over the *response* tokens only (prompt and padding positions are masked
    with ``ignore_index=-100``).

    :param pop: Population of SFT agents.
    :param env: SFTGym environment wrapping the dataset.
    :param init_hp: Hyperparameter dict forwarded to wandb, defaults to None.
    :param save_elite: Save best agent to disk, defaults to None.
    :param elite_path: Directory for checkpoints, defaults to None.
    :param wb: Weights & Biases logging, defaults to False.
    :param evo_steps: Steps between HPO evolution rounds, defaults to None.
    :param checkpoint_steps: Steps between non-HPO saves, defaults to None.
    :param tournament: Tournament selection object, defaults to None.
    :param mutation: Mutation object, defaults to None.
    :param wandb_api_key: W&B API key, defaults to None.
    :param evaluation_interval: Steps between eval passes, defaults to 10.
    :param verbose: Print summary at end, defaults to True.
    :param accelerator: Distributed training handle, defaults to None.
    :param max_steps: Total samples to process (one epoch if None), defaults to None.
    :param num_epochs: Dataset passes; overrides max_steps when set, defaults to None.
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

    _setup_wandb(
        accelerator,
        wb,
        init_hp,
        env.name,
        wandb_api_key,
        pop[0],
        effective_data_batch_size,
        project=wandb_project,
        addl_args={
            **({"name": wandb_run_name} if wandb_run_name is not None else {}),
            **({"entity": wandb_entity} if wandb_entity is not None else {}),
        }
        or None,
    )

    max_steps, training_steps = _compute_training_steps(
        max_steps, num_epochs, len(env), effective_data_batch_size
    )
    pbar = _create_pbar(accelerator, max_steps)

    total_steps = 0
    prompts = env.reset(reset_dataloaders=True)

    sft_csv_file, sft_csv_writer = _open_csv_log(
        elite_path,
        ["step", "train_loss", "train_perplexity", "eval_loss"],
        accelerator,
    )

    for i in range(training_steps):
        agent_metrics_dict = {}
        for agent_idx, agent in enumerate(pop):
            loss, perplexity = agent.learn(prompts)
            next_prompts = env.step()
            agg_metrics = [
                safe_aggregate_metrics(accelerator, loss),
                safe_aggregate_metrics(accelerator, perplexity),
            ]
            prompts = next_prompts
            agent.steps[-1] += effective_data_batch_size
            total_steps += effective_data_batch_size
            agg_test_metrics = None

            if (i + 1) % evaluation_interval == 0:
                test_score = agent.test(env)
                agg_test_metrics = [safe_aggregate_metrics(accelerator, test_score)]
                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if _is_main_process(accelerator):
                metrics_dict = {
                    "global_step": total_steps,
                    "Train/Loss": agg_metrics[0],
                    "Train/Perplexity": agg_metrics[1],
                }
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_test_metrics is not None:
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = {
                        "Eval/Negative loss (fitness)": agg_test_metrics[0],
                    }
                pbar.update(effective_data_batch_size)
                pbar.set_postfix(
                    loss=f"{agg_metrics[0]:.4f}",
                    ppl=f"{agg_metrics[1]:.2f}",
                    **(
                        {"eval_loss": f"{-agg_test_metrics[0]:.4f}"}
                        if agg_test_metrics is not None
                        else {}
                    ),
                )
                agent.scores.append(-agg_metrics[0])  # higher = better

        if sft_csv_writer is not None and agent_metrics_dict:
            metric_values = _per_agent_metrics(agent_metrics_dict, len(pop))
            eval_loss = (
                -np.mean(metric_values("test_metrics", "Eval/Negative loss (fitness)"))
                if agg_test_metrics is not None
                else ""
            )
            _log_csv_row(
                sft_csv_writer,
                sft_csv_file,
                {
                    "step": total_steps,
                    "train_loss": np.mean(metric_values("train_metrics", "Train/Loss")),
                    "train_perplexity": np.mean(
                        metric_values("train_metrics", "Train/Perplexity")
                    ),
                    "eval_loss": eval_loss,
                },
            )

        if accelerator is not None:
            accelerator.wait_for_everyone()

        pop = _handle_evolution_or_checkpoint(
            i,
            pop,
            evo_steps,
            tournament,
            mutation,
            env.name,
            accelerator,
            elite_path,
            save_elite,
            effective_data_batch_size,
            max_steps,
            checkpoint_steps,
            agent,
            require_evo_steps_for_hpo=True,
        )

        if wb and _is_main_process(accelerator):
            metric_values = _per_agent_metrics(agent_metrics_dict, len(pop))
            wandb_dict = {
                "Train/Mean population loss": np.mean(
                    metric_values("train_metrics", "Train/Loss")
                ),
                "Train/Mean population perplexity": np.mean(
                    metric_values("train_metrics", "Train/Perplexity")
                ),
                "Train/Best loss": np.min(metric_values("train_metrics", "Train/Loss")),
            }
            if agg_test_metrics is not None:
                wandb_dict["Eval/Best fitness"] = np.max(
                    metric_values("test_metrics", "Eval/Negative loss (fitness)")
                )
            _wandb_log(wandb_dict)

        if env.num_epochs == num_epochs:
            break

    if verbose and _is_main_process(accelerator):
        agent = pop[0]
        scores = agent.scores  # list of -loss per step
        fitness = agent.fitness  # list of -eval_loss per eval step

        banner_text = f"Training complete — {total_steps} steps"
        banner_width = max(len(banner_text) + 4, 40)
        border = "=" * banner_width
        lines = [border, banner_text.center(banner_width), border]

        if scores:
            losses = [-s for s in scores]
            lines += [
                f"  Train loss  — initial: {losses[0]:.4f}  "
                f"final: {losses[-1]:.4f}  "
                f"best: {min(losses):.4f}  "
                f"mean: {np.mean(losses):.4f}",
            ]
        if fitness:
            eval_losses = [-f for f in fitness]
            lines += [
                f"  Eval  loss  — final: {eval_losses[-1]:.4f}  "
                f"best: {min(eval_losses):.4f}",
            ]
        if evo_steps is not None:
            muts = [a.mut for a in pop]
            agents = [a.index for a in pop]
            lines += [
                f"  Agents: {agents}  Mutations: {muts}",
            ]

        pbar.write("\n".join(lines))

    _save_elite_checkpoint(pop, save_elite, elite_path, accelerator)

    if sft_csv_file is not None:
        sft_csv_file.close()
        print(f"Training metrics saved to {os.path.join(elite_path, 'metrics.csv')}")

    _finalize_training_pbar_wandb(accelerator, pbar, wb)
