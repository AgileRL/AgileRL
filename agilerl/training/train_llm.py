import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import trange

import wandb
from agilerl.algorithms import DPO, GRPO, LLMPPO, LLMReinforce
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.typing import PopulationType
from agilerl.utils.algo_utils import stack_and_pad_experiences
from agilerl.utils.llm_utils import ReasoningGym
from agilerl.utils.utils import (
    aggregate_metrics_across_gpus,
    init_wandb,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
    _distributed_world_size,
)

if TYPE_CHECKING:
    from gem.core import Env as GemEnv

InitDictType = dict[str, Any] | None


def finetune_llm_reasoning(
    pop: PopulationType,
    env: ReasoningGym,
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
            metrics = [rewards, completion_lengths]
            if isinstance(agent, (LLMPPO, LLMReinforce)):
                loss, kl, pg_loss, critic_loss, entropy = agent.learn(experiences)
                metrics.extend([loss, kl, pg_loss, critic_loss, entropy])

            else:
                loss, kl = agent.learn(experiences)
                metrics.extend([loss, kl])

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
                # metrics order: rewards, completion_lengths, loss, kl,
                # then (LLMPPO/Reinforce) pg, critic, entropy; optional accuracy last.
                metrics_dict = {
                    "global_step": total_steps,
                    "Train/Mean reward": agg_metrics[0],
                    "Train/Average completion length": int(agg_metrics[1]),
                    "Train/Loss": agg_metrics[2],
                    "Train/KL-divergence": agg_metrics[3],
                }
                if isinstance(agent, (LLMPPO, LLMReinforce)):
                    metrics_dict |= {
                        "Train/PG loss": agg_metrics[4],
                        "Train/Critic loss": agg_metrics[5],
                        "Train/Entropy": agg_metrics[6],
                    }
                if max_reward is not None:
                    metrics_dict["Train/Accuracy"] = agg_metrics[-1]
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_test_metrics is not None:
                    test_metrics_dict = {"Eval/Mean reward": agg_test_metrics[0]}
                    if max_reward is not None:
                        test_metrics_dict |= {"Eval/Accuracy": agg_test_metrics[1]}
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = (
                        test_metrics_dict
                    )
                pbar.update(effective_data_batch_size)
                agent.scores.append(agg_metrics[0])

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
            if isinstance(agent, (LLMPPO, LLMReinforce)):
                wandb_dict |= {
                    "Train/Mean population PG loss": np.mean(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                                "Train/PG loss"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                    "Train/Mean population critic loss": np.mean(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                                "Train/Critic loss"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ],
                    ),
                    "Train/Mean population entropy": np.mean(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                                "Train/Entropy"
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
            f"Score:\t\t{agg_metrics[0]}\n"
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

    data_increment = _distributed_world_size(accelerator)
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


def finetune_llm_multiturn(
    pop: PopulationType,
    env: "GemEnv",
    max_turns: int,
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
    :param env: Multi-turn environment (often a ``TokenObservationWrapper``).
    :type env: GemEnv
    :param max_turns: Maximum interaction turns per episode.
    :type max_turns: int
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

    if not isinstance(pop[0], (LLMPPO, LLMReinforce)):
        msg = (
            "The algorithm must be LLMPPO or LLMReinforce for multi-turn GEM finetuning. "
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
    env_name = init_hp.get("env_name", "gem_multiturn")
    data_increment = _distributed_world_size(accelerator)
    effective_data_batch_size = data_increment * batch_size

    if wb and (accelerator is None or accelerator.is_main_process):
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["batch_size"] = batch_size
        init_hp["distributed_training"] = accelerator is not None
        init_hp["model_name"] = pop[0].pretrained_model_name_or_path
        init_hp["max_turns"] = max_turns
        init_wandb(
            algo=init_hp["ALGO"],
            env_name=env_name,
            wandb_api_key=wandb_api_key,
            init_hyperparams=init_hp,
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
    agg_metrics: list[float] = []
    agg_eval_score: float | None = None

    i = 0
    while total_steps < max_steps:
        agent_metrics_dict = {}
        iteration_steps = 0
        for agent_idx, agent in enumerate(pop):
            agent.set_reference_policy(i + 1)

            completion_ids_list: list[torch.Tensor] = []
            action_masks_list: list[torch.Tensor] = []
            all_turn_ids: list[torch.Tensor] = []
            all_rewards: list[torch.Tensor] = []
            batch_steps = 0

            for _ in range(batch_size):
                prompt_dict, _info = env.reset()
                sw_ml = getattr(env, "_sw_max_model_len", None)
                if sw_ml is not None:
                    assert sw_ml == agent.max_model_len, (
                        f"env max_model_len ({sw_ml}) != agent.max_model_len "
                        f"({agent.max_model_len})"
                    )

                for _turn_idx in range(max_turns):
                    completion_ids, _ = agent.get_action([prompt_dict], training=True)
                    prompt_dict, _reward, terminated, truncated, _info = env.step(
                        completion_ids[0],
                    )

                    if terminated or truncated:
                        break

                ep_ids, action_mask, turn_ids, turn_rewards_t = env.get_episode_data()

                completion_ids_list.append(ep_ids)
                action_masks_list.append(action_mask)
                all_turn_ids.append(turn_ids)
                all_rewards.append(turn_rewards_t)
                batch_steps += len(env.turn_boundaries)

            (turn_ids_padded,) = stack_and_pad_experiences(
                all_turn_ids,
                padding_values=[-1],
            )
            rewards_2d = torch.stack(all_rewards)

            completion_lengths = np.mean([x.shape[1] for x in completion_ids_list])
            episode_scores = rewards_2d.sum(dim=1)
            mean_score = episode_scores.mean().to(agent.device)

            experiences = (
                completion_ids_list,
                action_masks_list,
                rewards_2d,
            )
            loss, kl, pg_loss, critic_loss, entropy = agent.learn(
                experiences,
                turn_ids=turn_ids_padded,
            )

            metrics = [
                torch.tensor(loss, dtype=torch.float32, device=agent.device),
                torch.tensor(kl, dtype=torch.float32, device=agent.device),
                mean_score,
                torch.tensor(
                    completion_lengths, dtype=torch.float32, device=agent.device
                ),
            ]
            if max_reward is not None:
                accuracy = (
                    (episode_scores >= max_reward).float().mean().to(agent.device)
                )
                metrics.append(accuracy)
            agg_metrics = [
                aggregate_metrics_across_gpus(accelerator, metric) for metric in metrics
            ]

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
                metrics_dict = {
                    "global_step": total_steps,
                    "Train/Loss": agg_metrics[0],
                    "Train/KL-divergence": agg_metrics[1],
                    "Train/Mean reward": agg_metrics[2],
                    "Train/Average completion length": int(agg_metrics[3]),
                    "Train/PG loss": pg_loss,
                    "Train/Critic loss": critic_loss,
                    "Train/Entropy": entropy,
                }
                if max_reward is not None:
                    metrics_dict["Train/Accuracy"] = agg_metrics[4]
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict
                if agg_eval_score is not None:
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = {
                        "Eval/Score": agg_eval_score,
                    }
                agent.scores.append(agg_metrics[2])

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
                f"Train score:\t\t{agg_metrics[2]:.3f}",
                f"Loss:\t\t\t{agg_metrics[0]:.4f}",
                f"KL-divergence:\t\t{agg_metrics[1]:.4f}",
                f"PG loss:\t\t{pg_loss:.4f}",
                f"VF loss:\t\t{critic_loss:.4f}",
                f"Entropy:\t\t{entropy:.4f}",
            ]
            if max_reward is not None:
                lines.insert(4, f"Train accuracy:\t\t{agg_metrics[4]:.3f}")
            if agg_eval_score is not None:
                lines.append(f"Eval score:\t\t{agg_eval_score:.3f}")
            lines.append(border)
            pbar.write("\n".join(lines))

        if accelerator is None or accelerator.is_main_process:
            postfix = {
                "loss": f"{np.mean([agent_metrics_dict[f'agent_{j}/train_metrics']['Train/Loss'] for j in range(len(pop))]):.4f}",
                "kl": f"{np.mean([agent_metrics_dict[f'agent_{j}/train_metrics']['Train/KL-divergence'] for j in range(len(pop))]):.4f}",
                "score": f"{np.mean([agent_metrics_dict[f'agent_{j}/train_metrics']['Train/Mean reward'] for j in range(len(pop))]):.3f}",
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
            wandb_dict = {
                "Train/Mean population PG loss": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/PG loss"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
                "Train/Mean population critic loss": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Critic loss"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ],
                ),
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
                "Train/Mean population entropy": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Entropy"
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

            last_eval_scores = [
                agent_metrics_dict.get(f"agent_{agent_idx}/test_metrics", {}).get(
                    "Eval/Score",
                )
                for agent_idx, _ in enumerate(pop)
            ]
            if any(s is not None for s in last_eval_scores):
                wandb_dict |= {
                    "Eval/Best score": np.max(
                        [s for s in last_eval_scores if s is not None]
                    ),
                    "Eval/Mean population score": np.mean(
                        [s for s in last_eval_scores if s is not None],
                    ),
                }

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

    return pop
