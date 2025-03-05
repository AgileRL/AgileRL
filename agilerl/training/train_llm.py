from typing import Any, Dict, Optional

import torch
from tqdm import trange
import torch.distributed as dist
import wandb
from agilerl.algorithms import GRPO
from agilerl.utils.llm_utils import HuggingFaceGym
from agilerl.utils.utils import init_wandb

InitDictType = Optional[Dict[str, Any]]


def finetune_llm(
    agent: GRPO,
    env: HuggingFaceGym,
    INIT_HP: InitDictType,
    checkpoint_interval: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    MUT_P: Optional[InitDictType] = None,
    wb: bool = False,
    wandb_api_key: Optional[str] = None,
    evaluation_interval: Optional[int] = 10,
) -> None:
    if wb and agent.local_rank == '0':
        init_wandb(
            algo=agent.algo,
            env_name=env.name,
            init_hyperparams=INIT_HP,
            mutation_hyperparams=MUT_P,
            wandb_api_key=wandb_api_key,
        )

    if agent.local_rank == '0':
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    max_steps = len(env) // env.data_batch_size
    if agent.local_rank == '0':
        pbar = trange(
            max_steps,
            unit="step",
            bar_format=bar_format,
            ascii=True,
            dynamic_ncols=True,
        )

    prompts = env.reset(
        reset_dataloaders=True
    )  # calling env.reset() supplies the first batch of training data
    for i in range(max_steps):
        completion_ids, action_masks = agent.get_action(prompts)
        next_prompts, rewards = env.step(
            completion_ids
        )  # Use the reward function stored in env.step to calculate reward of the each answer from the group
        experiences = (
            completion_ids,
            action_masks,
            rewards,
        )
        loss, kl, grad_norm = agent.learn(experiences)
        avg_loss, avg_kl, avg_grad_norm, avg_reward = aggregate_metrics_across_gpus(agent, loss, kl, grad_norm, rewards)
        prompts = next_prompts
        if agent.local_rank == '0':
            print(
                f"Step: {i + 1}",
                i,
                f"| Loss: {avg_loss}",
                f"| KL-divergence: {avg_kl}",
                f"| Grad-norm: {avg_grad_norm}",
            )

            pbar.update(1)
            if wb:
                wandb.log(
                    {
                        "Loss": avg_loss,
                        "KL-divergence": avg_kl,
                        "Grad-norm": avg_grad_norm,
                        "Mean training reward": avg_reward,
                    }   
                )
            if (i + 1) % evaluation_interval == 0:
                test_reward = agent.test(env)
                print(f"Test reward: {test_reward}")
                if wb:
                    wandb.log({"Test reward": test_reward})
            if (
                checkpoint_path is not None
                and checkpoint_interval is not None
                and (i + 1) % checkpoint_interval == 0
            ):
                agent.save_checkpoint(save_path := f"step_{i}.pt")
                print(f"Saved checkpoint {save_path}")


def gather_tensor(tensor, agent):
    # Convert to tensor if it's a scalar
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=f"cuda:{agent.local_rank}")
    
    # Ensure tensor is on correct device
    tensor = tensor.detach().clone()
    
    # Create a list to store tensors from all processes
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather the tensor from all processes
    dist.all_gather(gathered_tensors, tensor)
    
    return torch.stack(gathered_tensors)


def aggregate_metrics_across_gpus(agent, loss, kl, grad_norm, rewards):
    rewards = rewards.to(agent.device)
    all_losses = gather_tensor(loss, agent)
    all_kls = gather_tensor(kl, agent)
    all_grad_norms = gather_tensor(grad_norm, agent)
    all_rewards = gather_tensor(torch.mean(rewards), agent)
    
    # Compute aggregated metrics
    avg_loss = all_losses.mean().item()
    avg_kl = all_kls.mean().item()
    avg_grad_norm = all_grad_norms.mean().item()
    avg_reward = all_rewards.mean().item()
    return avg_loss, avg_kl, avg_grad_norm, avg_reward