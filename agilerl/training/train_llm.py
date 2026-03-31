import warnings
from typing import Any

import numpy as np
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from tqdm import trange

from agilerl.algorithms import DPO, GRPO, SFT
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


def _plot_sft_loss_curves(
    pop: list,
    effective_data_batch_size: int,
    evaluation_interval: int,
    plot_path: str,
) -> None:
    """Save a loss-curve plot for a completed SFT training run.

    :param pop: Trained agent population.
    :param effective_data_batch_size: Samples processed per training step.
    :param evaluation_interval: Steps between evaluation passes.
    :param plot_path: File path for the saved figure (e.g. ``"plots/sft_loss.png"``).
    """
    try:
        import os

        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib is not installed — skipping loss-curve plot.", stacklevel=2
        )
        return

    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

    has_eval = any(len(a.fitness) > 0 for a in pop)
    n_plots = 2 if has_eval else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 4), squeeze=False)
    fig.suptitle("SFT Training", fontweight="bold")

    for agent_idx, agent in enumerate(pop):
        label = f"agent {agent_idx}" if len(pop) > 1 else None
        scores = agent.scores  # stored as -loss
        if not scores:
            continue

        # Training loss
        train_losses = [-s for s in scores]
        train_steps = [
            (i + 1) * effective_data_batch_size for i in range(len(train_losses))
        ]
        ax = axes[0][0]
        ax.plot(train_steps, train_losses, linewidth=1.2, alpha=0.85, label=label)

        # Smoothed trend (rolling mean over 10%)
        window = max(1, len(train_losses) // 10)
        smoothed = np.convolve(
            train_losses, np.ones(window) / window, mode="valid"
        )
        smooth_steps = train_steps[window - 1 :]
        ax.plot(smooth_steps, smoothed, linewidth=2, linestyle="--", alpha=0.6)

        # Eval loss
        if has_eval and agent.fitness:
            eval_losses = [-f for f in agent.fitness]
            eval_steps = [
                (j + 1) * evaluation_interval * effective_data_batch_size
                for j in range(len(eval_losses))
            ]
            axes[0][1].plot(
                eval_steps, eval_losses, marker="o", linewidth=1.5, label=label
            )

    axes[0][0].set_title("Training loss")
    axes[0][0].set_xlabel("Samples")
    axes[0][0].set_ylabel("Loss")
    axes[0][0].grid(True, alpha=0.3)
    if len(pop) > 1:
        axes[0][0].legend()

    if has_eval:
        axes[0][1].set_title("Eval loss")
        axes[0][1].set_xlabel("Samples")
        axes[0][1].set_ylabel("Loss")
        axes[0][1].grid(True, alpha=0.3)
        if len(pop) > 1:
            axes[0][1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Loss curves saved to {plot_path}")


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
            "The algorithm must be GRPO for reasoning-based reinforcement learning."
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

    if wb and (accelerator is None or accelerator.is_main_process):
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_wandb(
            algo=init_hp["ALGO"],
            env_name=env.name,
            wandb_api_key=wandb_api_key,
            init_hyperparams=init_hp,
        )

    if accelerator is None or accelerator.is_main_process:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if max_steps is None and num_epochs is None:
        max_steps = len(env)

    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * len(env)

    training_steps = -(max_steps // -effective_data_batch_size)
    if accelerator is None or accelerator.is_main_process:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

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

            if accelerator is None or accelerator.is_main_process:
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
        if tournament and mutation is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                pop = tournament_selection_and_mutation(
                    population=pop,
                    tournament=tournament,
                    mutation=mutation,
                    env_name=env.name,
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

        if wb and (accelerator is None or accelerator.is_main_process):
            wandb_dict = {
                "Train/Best reward": np.max(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean reward"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population reward": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean reward"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population loss": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Loss"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population KL divergence": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/KL-divergence"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population completion length": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Average completion length"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
            }
            if max_reward is not None:
                wandb_dict |= {
                    "Train/Mean population accuracy": np.mean(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                                "Train/Accuracy"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                }
                wandb_dict |= {
                    "Train/Best accuracy": np.max(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                                "Train/Accuracy"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                }
            if len(pop[0].registry.hp_config.config.keys()) > 0:
                wandb_dict |= {
                    f"HPO_agent_{agent_idx}/{key}": getattr(agent, key)
                    for agent_idx, agent in enumerate(pop)
                    for key in agent.registry.hp_config.config
                }

            if agg_test_metrics is not None:
                test_dict = {
                    "Eval/Best reward": np.max(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                "Eval/Mean reward"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                    "Eval/Mean population reward": np.mean(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                "Eval/Mean reward"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                }
                if max_reward is not None:
                    test_dict |= {
                        "Eval/Mean population accuracy": np.mean(
                            [
                                agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                    "Eval/Accuracy"
                                ]
                                for agent_idx, _ in enumerate(pop)
                            ],
                        ),
                    }
                    wandb_dict |= {
                        "Eval/Best accuracy": np.max(
                            [
                                agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                    "Eval/Accuracy"
                                ]
                                for agent_idx, _ in enumerate(pop)
                            ],
                        ),
                    }
                wandb_dict |= test_dict
            wandb.log(wandb_dict)

        if env.num_epochs == num_epochs:
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
            f"Score:\t\t{agg_metrics[2]}\n"
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

    data_increment = accelerator.num_processes if accelerator is not None else 1
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu

    if wb and (accelerator is None or accelerator.is_main_process):
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_wandb(
            algo=init_hp["ALGO"],
            env_name=env.name,
            wandb_api_key=wandb_api_key,
            init_hyperparams=init_hp,
            project=wandb_project,
            addl_args={
                **({"name": wandb_run_name} if wandb_run_name is not None else {}),
                **({"entity": wandb_entity} if wandb_entity is not None else {}),
            } or None,
        )

    if accelerator is None or accelerator.is_main_process:
        pass

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if max_steps is None and num_epochs is None:
        max_steps = len(env)

    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * len(env)

    training_steps = -(max_steps // -effective_data_batch_size)
    if accelerator is None or accelerator.is_main_process:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    total_steps = 0

    prompts = env.reset(reset_dataloaders=True)
    for i in range(training_steps):
        agent_metrics_dict = {}
        for agent_idx, agent in enumerate(pop):
            agent.set_reference_policy(env.num_epochs)
            loss, chosen_reward, rejected_reward = agent.learn(prompts)
            next_prompts = env.step()
            metrics = [loss, chosen_reward, rejected_reward]
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
                agg_test_metrics = [
                    aggregate_metrics_across_gpus(accelerator, metric)
                    for metric in test_metrics
                ]

                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if accelerator is None or accelerator.is_main_process:
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
                agent.scores.append(agg_metrics[1] - agg_metrics[2])
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
                    env_name=env.name,
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

        if wb and (accelerator is None or accelerator.is_main_process):
            wandb_dict = {
                "Train/Best reward margin": np.max(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean reward margin"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population reward margin": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean reward margin"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population loss": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Loss"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population chosen reward": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean chosen reward"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population rejected reward": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean rejected reward"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
            }
            if agg_test_metrics is not None:
                test_dict = {
                    "Eval/Best reward margin": np.max(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                "Eval/Mean reward margin"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                    "Eval/Mean population reward margin": np.mean(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                "Eval/Mean reward margin"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                }
                wandb_dict |= test_dict
            wandb.log(wandb_dict)
        if env.num_epochs == num_epochs:
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
            f"Score:\t\t{agg_metrics[2]}\n"
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
    plot_path: str | None = None,
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
    import warnings

    if evo_steps is not None and (tournament is None or mutation is None):
        warnings.warn(
            "'evo_steps' is set but 'tournament' or 'mutation' is None. "
            "Evolution will not take place.",
            stacklevel=2,
        )

    if (tournament is not None and mutation is not None) and evo_steps is None:
        raise ValueError(
            "'evo_steps' must be set when 'tournament' and 'mutation' are not None."
        )

    if num_epochs is not None and max_steps is not None:
        warnings.warn(
            "'num_epochs' overrides 'max_steps'.",
            stacklevel=2,
        )

    if mutation is not None:
        assert mutation.architecture_mut == 0, (
            "Architecture mutation must be 0 for LLM finetuning."
        )
        assert mutation.new_layer_prob == 0, (
            "New-layer mutation must be 0 for LLM finetuning."
        )
        assert mutation.parameters_mut == 0, (
            "Parameters mutation must be 0 for LLM finetuning."
        )
        assert mutation.activation_mut == 0, (
            "Activation mutation must be 0 for LLM finetuning."
        )

    from agilerl.algorithms.sft import SFT as _SFT

    if not isinstance(pop[0], _SFT):
        raise ValueError(
            f"Population must contain SFT agents. Got {type(pop[0])}."
        )

    if init_hp is None:
        init_hp = {}
        init_hp["BATCH_SIZE_PER_GPU"] = pop[0].batch_size_per_process
        init_hp["ALGO"] = pop[0].algo

    data_increment = accelerator.num_processes if accelerator is not None else 1
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu

    if wb and (accelerator is None or accelerator.is_main_process):
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = init_hp.get("BATCH_SIZE", 1)
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_wandb(
            algo=init_hp["ALGO"],
            env_name=env.name,
            wandb_api_key=wandb_api_key,
            init_hyperparams=init_hp,
            project=wandb_project,
            addl_args={
                **({"name": wandb_run_name} if wandb_run_name is not None else {}),
                **({"entity": wandb_entity} if wandb_entity is not None else {}),
            } or None,
        )

    bar_format = (
        "{l_bar}{bar:10}| {n:4}/{total_fmt} "
        "[{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    )
    if max_steps is None and num_epochs is None:
        max_steps = len(env)
    elif max_steps is None and num_epochs is not None:
        max_steps = num_epochs * len(env)

    training_steps = -(max_steps // -effective_data_batch_size)

    if accelerator is None or accelerator.is_main_process:
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    def _agg(m):
        if accelerator is None:
            return float(m) if not isinstance(m, float) else m
        return aggregate_metrics_across_gpus(accelerator, m)

    total_steps = 0
    prompts = env.reset(reset_dataloaders=True)

    for i in range(training_steps):
        agent_metrics_dict = {}
        for agent_idx, agent in enumerate(pop):
            loss, perplexity = agent.learn(prompts)
            next_prompts = env.step()
            agg_metrics = [_agg(loss), _agg(perplexity)]
            prompts = next_prompts
            agent.steps[-1] += effective_data_batch_size
            total_steps += effective_data_batch_size
            agg_test_metrics = None

            if (i + 1) % evaluation_interval == 0:
                test_score = agent.test(env)
                agg_test_metrics = [_agg(test_score)]
                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if accelerator is None or accelerator.is_main_process:
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

        if accelerator is not None:
            accelerator.wait_for_everyone()

        if tournament is not None and mutation is not None and evo_steps is not None:
            if (i + 1) % evo_steps == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                pop = tournament_selection_and_mutation(
                    population=pop,
                    tournament=tournament,
                    mutation=mutation,
                    env_name=env.name,
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

        if wb and (accelerator is None or accelerator.is_main_process):
            wandb_dict = {
                "Train/Mean population loss": np.mean(
                    [
                        agent_metrics_dict[f"agent_{j}/train_metrics"]["Train/Loss"]
                        for j in range(len(pop))
                    ]
                ),
                "Train/Mean population perplexity": np.mean(
                    [
                        agent_metrics_dict[f"agent_{j}/train_metrics"][
                            "Train/Perplexity"
                        ]
                        for j in range(len(pop))
                    ]
                ),
                "Train/Best loss": np.min(
                    [
                        agent_metrics_dict[f"agent_{j}/train_metrics"]["Train/Loss"]
                        for j in range(len(pop))
                    ]
                ),
            }
            if agg_test_metrics is not None:
                wandb_dict["Eval/Best fitness"] = np.max(
                    [
                        agent_metrics_dict[f"agent_{j}/test_metrics"][
                            "Eval/Negative loss (fitness)"
                        ]
                        for j in range(len(pop))
                    ]
                )
            wandb.log(wandb_dict)

        if env.num_epochs == num_epochs:
            break

    if verbose and (accelerator is None or accelerator.is_main_process):
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

    if plot_path is not None and (accelerator is None or accelerator.is_main_process):
        _plot_sft_loss_curves(
            pop=pop,
            effective_data_batch_size=effective_data_batch_size,
            evaluation_interval=evaluation_interval,
            plot_path=plot_path,
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    if accelerator is None or accelerator.is_main_process:
        pbar.close()
        if wb:
            wandb.finish()
