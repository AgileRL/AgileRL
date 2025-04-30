import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from tqdm import trange

from agilerl.algorithms import GRPO
from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.llm_utils import HuggingFaceGym
from agilerl.utils.utils import (
    aggregate_metrics_across_gpus,
    init_wandb,
    save_llm_checkpoint,
    tournament_selection_and_mutation,
)

InitDictType = Optional[Dict[str, Any]]
PopulationType = List[RLAlgorithm]


def finetune_llm(
    pop: List[GRPO],
    env: HuggingFaceGym,
    init_hp: Optional[Dict[str, Any]] = None,
    save_elite: Optional[bool] = None,
    elite_path: Optional[str] = None,
    wb: bool = False,
    evo_steps: Optional[int] = 20,
    tournament: Optional[TournamentSelection] = None,
    mutation: Optional[Mutations] = None,
    wandb_api_key: Optional[str] = None,
    evaluation_interval: int = 10,
    max_reward: Optional[int] = None,
    verbose: bool = True,
    accelerator: Optional[Accelerator] = None,
    max_steps: Optional[int] = None,
):
    """
    Finetunes a population of GRPOs on a HuggingFaceGym environment.

    :param pop: Population of GRPOs to finetune
    :type pop: list[GRPO]
    :param env: HuggingFaceGym environment to finetune on
    :type env: HuggingFaceGym
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
    """

    if evo_steps is not None and (tournament is None or mutation is None):
        warnings.warn(
            "'evo_steps' is set but at least one of 'tournament' or 'mutation' is set to None. Evolution will not take place."
        )

    if (tournament is not None and mutation is not None) and evo_steps is None:
        raise ValueError(
            "'evo_steps' must be set if 'tournament' and 'mutation' are not None."
        )

    if mutation is not None:
        assert (
            mutation.architecture_mut == 0
        ), "Architecture mutation is not allowed for LLM finetuning."
        assert (
            mutation.new_layer_prob == 0
        ), "New layer mutation is not allowed for LLM finetuning."
        assert (
            mutation.parameters_mut == 0
        ), "Network parameters mutation is not allowed for LLM finetuning."
        assert (
            mutation.activation_mut == 0
        ), "Activation mutation is not allowed for LLM finetuning."

    if init_hp is None:
        init_hp = {}
        init_hp["BATCH_SIZE_PER_GPU"] = pop[0].batch_size
        init_hp["ALGO"] = pop[0].algo
    data_increment = (
        getattr(dist, "get_world_size", lambda: 1)() if dist.is_initialized() else 1
    )
    grad_accum = getattr(pop[0].actor, "gradient_accumulation_steps", lambda: 1)()
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu
    effective_learning_batch_size = (
        data_increment * init_hp["BATCH_SIZE_PER_GPU"] * grad_accum
    )
    if accelerator is None or accelerator.is_main_process:
        print(
            f"""
=========================================================================
Commencing RL finetuning

Data batch size per gpu: {env.data_batch_size_per_gpu}
Number of GPUs: {data_increment}
Gradient accumulation: {grad_accum}
Effective data batch size: {data_increment} * {env.data_batch_size_per_gpu} = {effective_data_batch_size}
Effective learning batch_size: {data_increment} * {init_hp["BATCH_SIZE_PER_GPU"]} * {grad_accum} = {effective_learning_batch_size}
=========================================================================
        """
        )
    if wb and (accelerator is None or accelerator.is_main_process):
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["effective_learning_batch_size"] = effective_learning_batch_size
        init_hp["distributed_training"] = True if accelerator is not None else False
        init_wandb(
            algo=init_hp["ALGO"],
            env_name=env.name,
            wandb_api_key=wandb_api_key,
            init_hyperparams=init_hp,
        )
    if accelerator is None or accelerator.is_main_process:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if max_steps is None:
        max_steps = len(env)
    training_steps = max_steps // effective_data_batch_size
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
                        test_reward.flatten()
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
                    accelerator=accelerator,  # Set as None for LLM finetuning as it does not require the same accelerator handling as standard RL models
                    language_model=True,
                    elite_path=elite_path,
                    save_elite=save_elite,
                )
                if accelerator is not None:
                    accelerator.wait_for_everyone()
        else:
            if (i + 1) % max_steps == 0:
                save_llm_checkpoint(agent, elite_path, i + 1)

        if wb and (accelerator is None or accelerator.is_main_process):
            wandb_dict = {
                "Train/Best reward": np.max(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean reward"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ]
                ),
                "Train/Mean population reward": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Mean reward"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ]
                ),
                "Train/Mean population loss": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Loss"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ]
                ),
                "Train/Mean population KL divergence": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/KL-divergence"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ]
                ),
                "Train/Mean population completion length": np.mean(
                    [
                        agent_metrics_dict[f"agent_{agent_idx}/train_metrics"][
                            "Train/Average completion length"
                        ]
                        for agent_idx, _ in enumerate(pop)
                    ]
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
                        ]
                    )
                }
            if len(pop[0].registry.hp_config.config.keys()) > 0:
                wandb_dict |= {
                    f"HPO_agent_{agent_idx}/{key}": getattr(agent, key)
                    for agent_idx, agent in enumerate(pop)
                    for key in agent.registry.hp_config.config.keys()
                }

            if agg_test_metrics is not None:
                test_dict = {
                    "Eval/Best reward": np.max(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                "Eval/Mean reward"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ]
                    ),
                    "Eval/Mean population reward": np.mean(
                        [
                            agent_metrics_dict[f"agent_{agent_idx}/test_metrics"][
                                "Eval/Mean reward"
                            ]
                            for agent_idx, _ in enumerate(pop)
                        ]
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
                            ]
                        )
                    }
                wandb_dict |= test_dict
            wandb.log(wandb_dict)

    if (
        verbose
        and total_steps > evaluation_interval
        and (accelerator is None or accelerator.is_main_process)
    ):
        fitness = [str(round(agent.fitness[-1], 2)) for agent in pop]
        avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
        avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
        agents = [agent.index for agent in pop]
        num_steps = [agent.steps[-1] for agent in pop]
        muts = [agent.mut for agent in pop]
        print(
            f"""
            --- Global Steps {total_steps} ---
            Fitness:\t\t{fitness}
            Score:\t\t{agg_metrics[2]}
            5 fitness avgs:\t{avg_fitness}
            10 score avgs:\t{avg_score}
            Agents:\t\t{agents}
            Steps:\t\t{num_steps}
            Mutations:\t\t{muts}
            """,
            end="\r",
        )

    if accelerator is not None:
        accelerator.wait_for_everyone()
    if accelerator is None or accelerator.is_main_process:
        pbar.close()
        if wb:
            wandb.finish()
