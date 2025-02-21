from typing import Any, Dict, Optional

from tqdm import trange

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
) -> None:
    if wb:
        init_wandb(
            algo=agent.algo,
            env_name=env.name,
            init_hyperparams=INIT_HP,
            mutation_hyperparams=MUT_P,
            wandb_api_key=wandb_api_key,
        )

    print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"

    pbar = trange(
        (max_steps := env.length),
        unit="step",
        bar_format=bar_format,
        ascii=True,
        dynamic_ncols=True,
    )

    prompts, info = (
        env.reset()
    )  # calling env.reset() supplies the first batch of training data
    for i in range(max_steps):

        # Obtain answers for the batch of prompts (prompt_batch x group_size), the log probs here are the reference log probs
        # ATM its a list of length prompt_batch, each with a batch of sequence ids of size=group_size
        completions, log_probs = agent.get_action(prompts)
        prompts, rewards = env.step(
            completions
        )  # Use the reward function stored in env.step to calculate reward of the each answer from the group
        log_probs.append(log_probs)
        experiences = (
            prompts,
            rewards,
            log_probs,
        )
        loss, kl = agent.learn(experiences)
        if wb:
            wandb.log({"Loss": loss, "KL-divergence": kl})
        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (i + 1) % checkpoint_interval == 0
        ):
            agent.save_checkpoint(save_path := checkpoint_path / f"step_{i}.pt")
            print(f"Saved checkpoint {save_path}")
        pbar.update(i)
