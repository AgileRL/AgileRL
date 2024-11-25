import os

import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import trange

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.replay_data import ReplayDataset
from agilerl.components.sampler import Sampler
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import (
    create_population,
    make_vect_envs,
    observation_space_channels_to_first
)

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


if __name__ == "__main__":
    accelerator = Accelerator()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("===== AgileRL Online Distributed Demo =====")
    accelerator.wait_for_everyone()

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [32, 32],  # Actor hidden size
    }

    INIT_HP = {
        "DOUBLE": True,  # Use double Q-learning in DQN or CQN
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 1e-3,  # For soft update of target network parameters
        # Swap image channels dimension last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "POP_SIZE": 4,  # Population size
    }

    num_envs = 8
    env = make_vect_envs("LunarLander-v2", num_envs=num_envs)  # Create environment
    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if INIT_HP["CHANNELS_LAST"]:
        observation_space = observation_space_channels_to_first(observation_space)

    pop = create_population(
        algo="DQN",  # Algorithm
        observation_space=observation_space,  # Observation space
        action_space=action_space,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # No. vectorized envs
        accelerator=accelerator,  # Accelerator
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,
    )  # Field names to store in memory
    replay_dataset = ReplayDataset(memory, INIT_HP["BATCH_SIZE"])
    replay_dataloader = DataLoader(replay_dataset, batch_size=None)
    replay_dataloader = accelerator.prepare(replay_dataloader)
    sampler = Sampler(
        distributed=True, dataset=replay_dataset, dataloader=replay_dataloader
    )

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo="DQN",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
        mutation_sd=0.1,  # Mutation strength
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=1,  # Random seed
        accelerator=accelerator,
    )  # Accelerator)

    max_steps = 200000  # Max steps
    learning_delay = 1000  # Steps before starting learning

    # Exploration params
    eps_start = 1.0  # Max exploration
    eps_end = 0.1  # Min exploration
    eps_decay = 0.995  # Decay per episode
    epsilon = eps_start

    evo_steps = 10000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    total_steps = 0

    accel_temp_models_path = "models/{}".format("LunarLander-v2")
    if accelerator.is_main_process:
        if not os.path.exists(accel_temp_models_path):
            os.makedirs(accel_temp_models_path)

    print(f"\nDistributed training on {accelerator.device}...")

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step", disable=not accelerator.is_local_main_process)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        accelerator.wait_for_everyone()
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores, losses = [], []
            steps = 0
            epsilon = eps_start

            for idx_step in range(evo_steps):
                # Get next action from agent
                action = agent.get_action(state, epsilon)
                epsilon = max(
                    eps_end, epsilon * eps_decay
                )  # Decay epsilon for exploration

                # Act in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                scores += np.array(reward)
                steps += num_envs
                total_steps += num_envs

                # Collect scores for completed episodes
                for idx, (d, t) in enumerate(zip(terminated, truncated)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0

                # Save experience to replay buffer
                memory.save_to_memory_vect_envs(
                    state, action, reward, next_state, terminated
                )

                # Learn according to learning frequency
                if memory.counter > learning_delay and len(memory) >= agent.batch_size:
                    for _ in range(num_envs // agent.learn_step):
                        # Sample dataloader
                        experiences = sampler.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                state = next_state

            pbar.update(evo_steps // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Reset epsilon start to latest decayed value for next round of population training
        eps_start = epsilon

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        if accelerator.is_main_process:
            print(f"--- Global steps {total_steps} ---")
            print(f"Steps {[agent.steps[-1] for agent in pop]}")
            print(f"Scores: {mean_scores}")
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(
                f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
            )

        # Tournament selection and population mutation
        accelerator.wait_for_everyone()
        for model in pop:
            model.unwrap_models()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
            for pop_i, model in enumerate(pop):
                model.save_checkpoint(f"{accel_temp_models_path}/DQN_{pop_i}.pt")
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            for pop_i, model in enumerate(pop):
                model.load_checkpoint(f"{accel_temp_models_path}/DQN_{pop_i}.pt")
        accelerator.wait_for_everyone()
        for model in pop:
            model.wrap_models()

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()
