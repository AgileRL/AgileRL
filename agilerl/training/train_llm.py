import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import trange
from accelerate import Accelerator

import wandb
from agilerl.algorithms import GRPO
from agilerl.algorithms.core.base import RLAlgorithm
from agilerl.utils.llm_utils import HuggingFaceGym
from agilerl.utils.utils import init_wandb, tournament_selection_and_mutation

InitDictType = Optional[Dict[str, Any]]
PopulationType = List[RLAlgorithm]


def finetune_llm(
    agent: GRPO,
    env: HuggingFaceGym,
    checkpoint_interval: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    wb: bool = False,
    wandb_api_key: Optional[str] = None,
    evaluation_interval: Optional[int] = 10,
    max_reward: Optional[int] = None,
) -> None:
    data_increment = (
        agent.accelerator.num_processes if agent.accelerator is not None else 1
    )
    grad_accum = getattr(agent.actor, "gradient_accumulation_steps", lambda: 1)()
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu
    effective_learning_batch_size = (
        data_increment * env.data_batch_size_per_gpu * grad_accum
    )
    if agent.accelerator.is_main_process:
        print(  
            f"""
=========================================================================
Commencing RL finetuning

Data batch size per gpu: {env.data_batch_size_per_gpu}
Number of GPUs: {data_increment}
Gradient accumulation: {grad_accum}
Effective data batch size: {data_increment} * {env.data_batch_size_per_gpu} = {effective_data_batch_size}
Effective learning batch_size: {data_increment} * {agent.batch_size} * {grad_accum} = {effective_learning_batch_size}
=========================================================================
        """
        )
    if wb and agent.accelerator.is_main_process:  
        init_wandb(
            algo=agent.algo,
            env_name=env.name,
            wandb_api_key=wandb_api_key,
            init_hyperparams={
                "effective_data_batch_size": effective_data_batch_size,
                "effective_learning_batch_size": effective_learning_batch_size,
                "group_size": agent.group_size,
                "temperature": agent.generation_config.temperature,
                "max_new_tokens": agent.generation_config.max_new_tokens,
                "min_new_tokens": agent.generation_config.min_new_tokens,
                "pad_token_id": agent.generation_config.pad_token_id,
                "beta": agent.beta,
                "clip_coefficient": agent.clip_coef,
                "update_epochs": agent.update_epochs,
                "reduce_memory_peak": agent.reduce_memory_peak,
                "cosine_lr_scheduler": (
                    True if agent.cosine_lr_schedule_config is not None else False
                ),
                "distributed_training": (
                    True if agent.accelerator is not None else False
                ),
                "learning_rate": agent.lr,
            },
        )

    if agent.accelerator.is_main_process: 
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    max_steps = len(env) // effective_data_batch_size
    if agent.accelerator.is_main_process: 
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    # calling env.reset() supplies the first batch of training data
    prompts = env.reset(reset_dataloaders=True)
    for i in range(max_steps):
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
            aggregate_metrics_across_gpus(agent, metric) for metric in metrics
        ]
        prompts = next_prompts
        agg_test_metrics = None
        if (i + 1) % evaluation_interval == 0:
            test_reward = agent.test(env)
            test_metrics = [test_reward]
            if max_reward is not None:
                test_accuracy = (test_reward == max_reward).sum() / len(
                    rewards.flatten()
                )
                test_metrics.append(test_accuracy)
            agg_test_metrics = [
                aggregate_metrics_across_gpus(agent, metric) for metric in test_metrics
            ]
        if agent.accelerator.is_main_process: 
            metrics_dict = {
                "Train/Loss": agg_metrics[0],
                "Train/KL-divergence": agg_metrics[1],
                "Train/Mean reward": agg_metrics[2],
                "Train/Average completion length": int(agg_metrics[3]),
            }
            if max_reward is not None:
                metrics_dict |= {"Train/Accuracy": agg_metrics[4]}
            print(metrics_dict)
            if agg_test_metrics is not None:
                test_metrics_dict = {"Eval/Mean reward": agg_test_metrics[0]}
                if max_reward is not None:
                    test_metrics_dict |= {"Eval/Accuracy": agg_test_metrics[1]}
                print(test_metrics_dict)
                if wb:
                    wandb.log(test_metrics_dict)
            if wb:
                wandb.log(metrics_dict)
            if (
                checkpoint_path is not None
                and checkpoint_interval is not None
                and (i + 1) % checkpoint_interval == 0
            ):
                save_llm_checkpoint(agent, checkpoint_path, i)
            pbar.update(effective_data_batch_size)


def finetune_evolvable_llm(
    pop: GRPO,
    env: HuggingFaceGym,
    init_hp: Optional[Dict[str, Any]] = None,
    checkpoint_interval: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    wb: bool = False,
    evo_steps: int = 500,
    tournament = None,
    mutation = None,
    wandb_api_key: Optional[str] = None,
    evaluation_interval: Optional[int] = 10,
    max_reward: Optional[int] = None,
    verbose: bool = True,
    accelerator: Optional[Accelerator] = None
):
    if init_hp is None: 
        init_hp = {}
        init_hp["BATCH_SIZE"] = pop[0].batch_size
    data_increment = (
        getattr(dist, "get_world_size", lambda: 1)() if dist.is_initialized() else 1
    )
    grad_accum = getattr(pop[0].actor, "gradient_accumulation_steps", lambda: 1)()
    effective_data_batch_size = data_increment * env.data_batch_size_per_gpu
    effective_learning_batch_size = (
        data_increment * env.data_batch_size_per_gpu * grad_accum
    )
    if accelerator.is_main_process:
        print(
            f"""
=========================================================================
Commencing RL finetuning

Data batch size per gpu: {env.data_batch_size_per_gpu}
Number of GPUs: {data_increment}
Gradient accumulation: {grad_accum}
Effective data batch size: {data_increment} * {env.data_batch_size_per_gpu} = {effective_data_batch_size}
Effective learning batch_size: {data_increment} * {init_hp["BATCH_SIZE"]} * {grad_accum} = {effective_learning_batch_size}
=========================================================================
        """
        )
    if wb and accelerator.is_main_process:
        init_hp["effective_data_batch_size"] = effective_data_batch_size
        init_hp["effective_learning_batch_size"] = effective_learning_batch_size
        init_hp["distributed_training"] = True if accelerator is not None else False
        init_wandb(
            algo=init_hp["ALGO"],
            env_name=env.name,
            wandb_api_key=wandb_api_key,
            init_hyperparams=init_hp
        )
    if accelerator.is_main_process:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    max_steps = len(env) // effective_data_batch_size
    if accelerator.is_main_process:
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
    for i in range(max_steps):
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
                aggregate_metrics_across_gpus(agent, metric) for metric in metrics
            ]
            prompts = next_prompts
            agg_test_metrics = None
            if (i + 1) % evaluation_interval == 0:
                test_reward = agent.test(env)
                test_metrics = [test_reward]
                if max_reward is not None:
                    test_accuracy = (test_reward == max_reward).sum() / len(
                        rewards.flatten()
                    )
                    test_metrics.append(test_accuracy)
                agg_test_metrics = [
                    aggregate_metrics_across_gpus(agent, metric)
                    for metric in test_metrics
                ]
            if accelerator.is_main_process:
                metrics_dict = {
                    "Train/Loss": agg_metrics[0],
                    "Train/KL-divergence": agg_metrics[1],
                    "Train/Mean reward": (mean_scores := agg_metrics[2]),
                    "Train/Average completion length": int(agg_metrics[3]),
                }
                if max_reward is not None:
                    metrics_dict |= {"Train/Accuracy": agg_metrics[4]}
                agent_metrics_dict[f"agent_{agent_idx}/train_metrics"] = metrics_dict 
                if agg_test_metrics is not None:
                    test_metrics_dict = {"Eval/Mean reward": agg_test_metrics[0]}
                    if max_reward is not None:
                        test_metrics_dict |= {"Eval/Accuracy": agg_test_metrics[1]}
                    agent_metrics_dict[f"agent_{agent_idx}/test_metrics"] = test_metrics_dict
                #     if wb:
                #         wandb.log(test_metrics_dict)
                # if wb:
                #     wandb.log(metrics_dict)
                if (
                    checkpoint_path is not None
                    and checkpoint_interval is not None
                    and (i + 1) % checkpoint_interval == 0
                ):
                    save_llm_checkpoint(agent, checkpoint_path, i)
                pbar.update(effective_data_batch_size)
                agent.steps.append(effective_data_batch_size)
                agent.scores.append(mean_scores)
                total_steps += effective_data_batch_size
        
                # if verbose:
                #     fitness = [str(round(agent.fitness[-1], 2)) for agent in pop]
                #     avg_fitness = ["%.2f" % np.mean(agent.fitness[-5:]) for agent in pop]
                #     avg_score = ["%.2f" % np.mean(agent.scores[-10:]) for agent in pop]
                #     agents = [agent.index for agent in pop]
                #     num_steps = [agent.steps[-1] for agent in pop]
                #     muts = [agent.mut for agent in pop]
                #     print(
                #         f"""
                #         --- Global Steps {total_steps} ---
                #         Fitness:\t\t{fitness}
                #         Score:\t\t{mean_scores}
                #         5 fitness avgs:\t{avg_fitness}
                #         10 score avgs:\t{avg_score}
                #         Agents:\t\t{agents}
                #         Steps:\t\t{num_steps}
                #         Mutations:\t\t{muts}
                #         """,
                #         end="\r",
                #     )
        accelerator.wait_for_everyone()
        if (i + 1) % evo_steps == 0:
            if tournament and mutation is not None:
                pop = tournament_selection_and_mutation(
                population=pop,
                tournament=tournament,
                mutation=mutation,
                env_name=env.name,
                accelerator=None, # Set as None for LLM finetuning as it does not require the same accelerator handling as standard RL models
            ) 


        if wb and accelerator.is_main_process:
            wandb_dict = {
                "Train/Best reward": np.max([agent_metrics_dict[f"agent_{agent_idx}/train_metrics"]["Train/Mean reward"] for agent_idx,_ in enumerate(pop)]),
                "Train/Mean population reward": np.mean([agent_metrics_dict[f"agent_{agent_idx}/train_metrics"]["Train/Mean reward"] for agent_idx,_ in enumerate(pop)]),
                "Train/Mean population loss": np.mean([agent_metrics_dict[f"agent_{agent_idx}/train_metrics"]["Train/Loss"] for agent_idx,_ in enumerate(pop)]),
                "Train/Mean population KL divergence": np.mean([agent_metrics_dict[f"agent_{agent_idx}/train_metrics"]["Train/KL-divergence"] for agent_idx,_ in enumerate(pop)]),
                "Train/Mean population completion length": np.mean([agent_metrics_dict[f"agent_{agent_idx}/train_metrics"]["Train/Average completion length"] for agent_idx,_ in enumerate(pop)]),
                "Train/Mean population accuracy": np.mean([agent_metrics_dict[f"agent_{agent_idx}/train_metrics"]["Train/Accuracy"] for agent_idx,_ in enumerate(pop)]),
                "HPO_agent_0/beta": pop[0].beta,
                "HPO_agent_1/beta": pop[1].beta,
                "HPO_agent_0/lr": pop[0].lr,
                "HPO_agent_1/lr": pop[1].lr,
                "HPO_agent_0/group_size": pop[0].group_size,
                "HPO_agent_1/group_size": pop[1].group_size,
            }
            try:
                test_dict = {
                    "Eval/Best reward": np.max([agent_metrics_dict.get(f"agent_{agent_idx}/test_metrics", None).get("Eval/Mean reward", None) for agent_idx,_ in enumerate(pop)]),
                    "Eval/Mean population reward": np.mean([agent_metrics_dict.get(f"agent_{agent_idx}/test_metrics", None).get("Eval/Mean reward", None) for agent_idx,_ in enumerate(pop)]),
                    "Eval/Mean population accuracy": np.mean([agent_metrics_dict[f"agent_{agent_idx}/test_metrics"]["Eval/Accuracy"] for agent_idx,_ in enumerate(pop)]),
                }
            except:
                test_dict = {"key": None} #FIXME sort this out tomorrow
            if all(val is not None for val in test_dict.values()):
                wandb_dict |= test_dict
            wandb.log(wandb_dict)   


def gather_tensor(tensor: torch.Tensor, agent: GRPO) -> torch.Tensor:
    """Gather tensors from gpus

    :param tensor: Tensor to gather
    :type tensor: torch.Tensor
    :param agent: GRPO agent object
    :type agent: GRPO
    :return: Stacked tensors
    :rtype: torch.Tensor
    """
    # Convert to tensor if it's a scalar
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=f"cuda:{agent.local_rank}")

    if tensor.device != agent.device:
        tensor = tensor.to(agent.device)
    # Ensure tensor is on correct device
    tensor = tensor.detach().clone()
    # Create a list to store tensors from all processes
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]

    # Gather the tensor from all processes
    dist.all_gather(gathered_tensors, tensor)
    return torch.stack(gathered_tensors)


def aggregate_metrics_across_gpus(agent: GRPO, metric_tensor: torch.Tensor) -> float:
    """Aggregate gathered tensors

    :param agent: GRPO agent
    :type agent: GRPO
    :param metric_tensor: Metrics
    :type metric_tensor: torch.Tensor
    :return: Mean metric
    :rtype: float
    """
    all_metrics = gather_tensor(metric_tensor, agent)
    avg_metrics = all_metrics.mean().item()
    return avg_metrics


def save_llm_checkpoint(agent: GRPO, checkpoint_path: str | None, step: int) -> None:
    """Checkpoint the LLM

    :param agent: GRPO agent
    :type agent: GRPO
    :param checkpoint_path: Checkpoint path
    :type checkpoint_path: str
    :param step: Training step
    :type step: int
    """
    base_path = "./saved_llms" if checkpoint_path is None else checkpoint_path
    path = base_path + f"/step_{step}"
    os.makedirs(path, exist_ok=True)
    if agent.accelerator is not None:
        unwrapped_model = agent.accelerator.unwrap_model(agent.actor)
        unwrapped_model.save_pretrained(path)
    else:
        agent.actor.save_pretrained(path)
