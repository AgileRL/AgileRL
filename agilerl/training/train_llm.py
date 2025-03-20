from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import trange

import wandb
from agilerl.algorithms import GRPO
from agilerl.utils.llm_utils import HuggingFaceGym
from agilerl.utils.utils import init_wandb

InitDictType = Optional[Dict[str, Any]]


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
        getattr(dist, "get_world_size", lambda: 1)() if dist.is_initialized() else 1
    )
    grad_accum = getattr(agent.actor, "gradient_accumulation_steps", lambda: 1)()
    if agent.local_rank == "0":
        print(
            f"""
=========================================================================
Commencing RL finetuning

Data batch size per gpu: {env.data_batch_size_per_gpu}
Number of GPUs: {data_increment}
Gradient accumulation: {grad_accum}
Effective data batch size: {data_increment} * {env.data_batch_size_per_gpu} * {grad_accum} = {data_increment * env.data_batch_size_per_gpu * grad_accum}
=========================================================================
        """
        )

    if wb and agent.local_rank == "0":
        init_wandb(
            algo=agent.algo,
            env_name=env.name,
            wandb_api_key=wandb_api_key,
        )

    if agent.local_rank == "0":
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    max_steps = len(env) // env.data_batch_size_per_gpu
    if agent.local_rank == "0":
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
            accuracy = (rewards == max_reward).sum() / len(rewards.squeeze())
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
                test_accuracy = (test_reward == max_reward).sum() / test_reward.shape[0]
                test_metrics.append(test_accuracy)
            agg_test_metrics = [
                aggregate_metrics_across_gpus(agent, metric) for metric in test_metrics
            ]
        if agent.local_rank == "0":
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
            pbar.update(data_increment)


def gather_tensor(tensor: torch.Tensor, agent: GRPO) -> torch.Tensor:
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


def aggregate_metrics_across_gpus(agent: GRPO, metrics: torch.Tensor):
    all_metrics = gather_tensor(metrics, agent)
    avg_metrics = all_metrics.mean().item()
    return avg_metrics


def save_llm_checkpoint(agent, checkpoint_path, step):
    checkpoint_path = f"step_{step}" if checkpoint_path is None else checkpoint_path
    if agent.accelerator is not None:
        unwrapped_model = agent.accelerator.unwrap_model(agent.actor)
        unwrapped_model.save_pretrained(checkpoint_path)
    else:
        agent.actor.save_pretrained(checkpoint_path)
