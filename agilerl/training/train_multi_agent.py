import os
from datetime import datetime

import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

import wandb
from agilerl.components.replay_data import ReplayDataset
from agilerl.components.sampler import Sampler


def train_multi_agent(
    env,
    env_name,
    algo,
    pop,
    memory,
    INIT_HP,
    MUT_P,
    net_config,
    swap_channels=False,
    n_episodes=2000,
    max_steps=25,
    evo_epochs=5,
    evo_loop=5,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay=0.995,
    target=200.0,
    tournament=None,
    mutation=None,
    checkpoint=None,
    checkpoint_path=None,
    wb=False,
    verbose=True,
    accelerator=None,
):
    """The general online multi-agent RL training function. Returns trained population of agents
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: List[object]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param swap_channels: Swap image channels dimension from last to first
    [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
    :param n_episodes: Maximum number of training episodes, defaults to 2000
    :type n_episodes: int, optional
    :param max_steps: Maximum number of steps in environment per episode, defaults to
    500
    :type max_steps: int, optional
    :param evo_epochs: Evolution frequency (episodes), defaults to 5
    :type evo_epochs: int, optional
    :param evo_loop: Number of evaluation episodes, defaults to 1
    :type evo_loop: int, optional
    :param eps_start: Maximum exploration - initial epsilon value, defaults to 1.0
    :type eps_start: float, optional
    :param eps_end: Minimum exploration - final epsilon value, defaults to 0.1
    :type eps_end: float, optional
    :param eps_decay: Epsilon decay per episode, defaults to 0.995
    :type eps_decay: float, optional
    :param target: Target score for early stopping, defaults to 200.
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (episodes), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param verbose: Display training stats, defaults to True
    :type verbose: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Hugging Face accelerate.Accelerator(), optional
    """
    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="EvoMADDPGTesting",
                    name="{}-MultiAgentEvoHPO-{}-{}".format(
                        env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
                    ),
                    # track hyperparameters and run metadata
                    config={
                        "algo": f"Evo HPO {algo}",
                        "env": env_name,
                        "net_config": net_config,
                        "batch_size": INIT_HP["BATCH_SIZE"],
                        "lr": INIT_HP["LR"],
                        "gamma": INIT_HP["GAMMA"],
                        "memory_size": INIT_HP["MEMORY_SIZE"],
                        "learn_step": INIT_HP["LEARN_STEP"],
                        "tau": INIT_HP["TAU"],
                        "pop_size": INIT_HP["POP_SIZE"],
                        "no_mut": MUT_P["NO_MUT"],
                        "arch_mut": MUT_P["ARCH_MUT"],
                        "params_mut": MUT_P["PARAMS_MUT"],
                        "act_mut": MUT_P["ACT_MUT"],
                        "rl_hp_mut": MUT_P["RL_HP_MUT"],
                    },
                )
            accelerator.wait_for_everyone()
        else:
            wandb.init(
                # set the wandb project where this run will be logged
                entity="agilerl",
                project="MADDPG Benchmarking",
                name="{}-{}-{}-RandomActions".format(
                    env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
                ),
                # track hyperparameters and run metadata
                config={
                    "algo": f"Evo HPO {algo}",
                    "env": env_name,
                    "net_config": net_config,
                    "batch_size": INIT_HP["BATCH_SIZE"],
                    "lr": INIT_HP["LR"],
                    "gamma": INIT_HP["GAMMA"],
                    "memory_size": INIT_HP["MEMORY_SIZE"],
                    "learn_step": INIT_HP["LEARN_STEP"],
                    "tau": INIT_HP["TAU"],
                    "pop_size": INIT_HP["POP_SIZE"],
                    "no_mut": MUT_P["NO_MUT"],
                    "arch_mut": MUT_P["ARCH_MUT"],
                    "params_mut": MUT_P["PARAMS_MUT"],
                    "act_mut": MUT_P["ACT_MUT"],
                    "rl_hp_mut": MUT_P["RL_HP_MUT"],
                },
            )

    if accelerator is not None:
        accel_temp_models_path = f"models/{env_name}"
        if accelerator.is_main_process:
            if not os.path.exists(accel_temp_models_path):
                os.makedirs(accel_temp_models_path)

    save_path = (
        checkpoint_path.split(".pt")[0]
        if checkpoint_path is not None
        else "{}-EvoHPO-{}-{}".format(
            env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S")
        )
    )

    if accelerator is not None:
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(
            distributed=True, dataset=replay_dataset, dataloader=replay_dataloader
        )
    else:
        sampler = Sampler(distributed=False, memory=memory)

    epsilon = eps_start

    if accelerator is not None:
        print(f"\nDistributed training on {accelerator.device}...")
    else:
        print("\nTraining...")

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    if accelerator is not None:
        pbar = trange(
            n_episodes,
            unit="ep",
            bar_format=bar_format,
            ascii=True,
            disable=not accelerator.is_local_main_process,
        )
    else:
        pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    pop_fitnesses = []
    total_steps = 0

    # Pre-training mutation
    if accelerator is None:
        if mutation is not None:
            pop = mutation.mutation(pop, pre_training_mut=True)

    # RL training loop
    for idx_epi in pbar:
        if accelerator is not None:
            accelerator.wait_for_everyone()
        for agent in pop:  # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            agent_reward = {agent_id: 0 for agent_id in env.agents}
            if swap_channels:
                state = {
                    agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                    for agent_id, s in state.items()
                }

            for _ in range(max_steps):
                total_steps += 1
                # Get next action from agent
                action = agent.getAction(state, epsilon)
                next_state, reward, done, truncation, _ = env.step(
                    action
                )  # Act in environment

                # Save experience to replay buffer
                if swap_channels:
                    state = {agent_id: np.squeeze(s) for agent_id, s in state.items()}
                    next_state = {
                        agent_id: np.moveaxis(ns, [2], [0])
                        for agent_id, ns in next_state.items()
                    }

                if any(truncation.values()) or any(done.values()):
                    break

                memory.save2memory(state, action, reward, next_state, done)

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Learn according to learning frequency
                if (memory.counter % agent.learn_step == 0) and (
                    len(memory) >= agent.batch_size
                ):
                    # Sample replay buffer
                    experiences = sampler.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

                # Update the state
                if swap_channels:
                    next_state = {
                        agent_id: np.expand_dims(ns, 0)
                        for agent_id, ns in next_state.items()
                    }
                state = next_state

            score = sum(agent_reward.values())
            agent.scores.append(score)

            agent.steps[-1] += total_steps

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env, swap_channels=swap_channels, max_steps=max_steps, loop=evo_loop
                )
                for agent in pop
            ]
            pop_fitnesses.append(fitnesses)

            mean_scores = np.mean([agent.scores[-20:] for agent in pop], axis=1)

            if wb:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        wandb.log(
                            {
                                "global_step": total_steps
                                * accelerator.state.num_processes,
                                "train/mean_score": np.mean(mean_scores),
                                "train/best_score": np.max(
                                    [agent.scores[-1] for agent in pop]
                                ),
                                "eval/mean_fitness": np.mean(fitnesses),
                                "eval/best_fitness": np.max(fitnesses),
                            }
                        )
                    accelerator.wait_for_everyone()
                else:
                    wandb.log(
                        {
                            "global_step": total_steps,
                            "train/mean_score": np.mean(mean_scores),
                            "train/best_score": np.max(
                                [agent.scores[-1] for agent in pop]
                            ),
                            "eval/mean_fitness": np.mean(fitnesses),
                            "eval/best_fitness": np.max(fitnesses),
                        }
                    )

                for idx, agent in enumerate(pop):
                    wandb.log(
                        {
                            f"learn_step_agent_{idx}": agent.learn_step,
                            f"learning_rate_agent_{idx}": agent.lr,
                            f"batch_size_agent_{idx}": agent.batch_size,
                            f"indi_fitness_agent_{idx}": agent.fitness[-1],
                        }
                    )

            # Update step counter
            for agent in pop:
                agent.steps.append(agent.steps[-1])

            # Early stop if consistently reaches target
            if (
                np.all(
                    np.greater([np.mean(agent.fitness[-100:]) for agent in pop], target)
                )
                and idx_epi >= 100
            ):
                if wb:
                    wandb.finish()
                return pop, pop_fitnesses

            # Tournament selection and population mutation
            if tournament and mutation is not None:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.unwrap_models()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        elite, pop = tournament.select(pop)
                        pop = mutation.mutation(pop)
                        for pop_i, model in enumerate(pop):
                            model.saveCheckpoint(
                                f"{accel_temp_models_path}/{algo}_{pop_i}.pt"
                            )
                    accelerator.wait_for_everyone()
                    if not accelerator.is_main_process:
                        for pop_i, model in enumerate(pop):
                            model.loadCheckpoint(
                                f"{accel_temp_models_path}/{algo}_{pop_i}.pt"
                            )
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.wrap_models()
                else:
                    elite, pop = tournament.select(pop)
                    pop = mutation.mutation(pop)

            if verbose:
                fitness = ["%.2f" % fitness for fitness in fitnesses]
                avg_fitness = ["%.2f" % np.mean(agent.fitness[-100:]) for agent in pop]
                avg_score = ["%.2f" % np.mean(agent.scores[-100:]) for agent in pop]
                agents = [agent.index for agent in pop]
                num_steps = [agent.steps[-1] for agent in pop]
                muts = [agent.mut for agent in pop]
                pbar.update(0)

                print(
                    f"""
                    --- Epoch {idx_epi + 1} ---
                    Fitness:\t\t{fitness}
                    100 fitness avgs:\t{avg_fitness}
                    100 score avgs:\t{avg_score}
                    Agents:\t\t{agents}
                    Steps:\t\t{num_steps}
                    Mutations:\t\t{muts}
                    """,
                    end="\r",
                )

        # Save model checkpoint
        if checkpoint is not None:
            if (idx_epi + 1) % checkpoint == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    if not accelerator.is_main_process:
                        for i, agent in enumerate(pop):
                            agent.saveCheckpoint(f"{save_path}_{i}_{idx_epi+1}.pt")
                    accelerator.wait_for_everyone()
                else:
                    for i, agent in enumerate(pop):
                        agent.saveCheckpoint(f"{save_path}_{i}_{idx_epi+1}.pt")

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()

    return pop, pop_fitnesses
