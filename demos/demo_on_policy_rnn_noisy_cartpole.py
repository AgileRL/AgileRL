import gymnasium as gym
import numpy as np
import torch
from tqdm import trange
import os
from gymnasium.wrappers import RecordVideo

from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population, make_vect_envs

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


# Define the Noisy Observation Wrapper
class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, real_every=2):
        super().__init__(env)
        self.real_every = real_every
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        self.last_real_obs = obs  # Store initial real obs
        return obs, info

    def observation(self, obs):
        self.step_count += 1
        if self.step_count % self.real_every == 0:
            self.last_real_obs = obs
            return obs
        else:
            return np.array(
                [0.0, 0.0, 0.0, 0.0]
            )  # Noise observation that needs to be filtered out by the LSTM


if __name__ == "__main__":
    print("===== AgileRL On-policy RNN Noisy CartPole Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "encoder_config": {
            "hidden_state_size": 64,  # LSTM hidden state size
            # "hidden_size": [64],  # MLP hidden size if not recurrent
        },
    }

    INIT_HP = {
        "POP_SIZE": 4,  # Population size
        "BATCH_SIZE": 256,  # Batch size (bigger often better for RNNs)
        "LR": 1e-4,  # Learning rate (potentially smaller for RNNs)
        "LEARN_STEP": 256,  # Learning frequency
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
        "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient
        "VF_COEF": 0.5,  # Value function coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
        "UPDATE_EPOCHS": 4,  # Number of policy update epochs
        "HIDDEN_STATE_SIZE": 64,  # Matching NET_CONFIG
        # PPO Specific
        "DISCRETE_ACTIONS": True,
        "ACTION_STD_INIT": 0.6,  # Only used for continuous actions
        "TARGET_KL": None,
        "CHANNELS_LAST": False,  # CartPole obs are 1D
    }

    num_envs = 1  # Number of parallel environments
    env = gym.vector.SyncVectorEnv(
        [lambda: NoisyObsWrapper(gym.make("CartPole-v1"), real_every=1)] * num_envs
    )

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    pop = create_population(
        algo="PPO",
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
        algo_kwargs={
            "use_rollout_buffer": True,
            "recurrent": True,
            "hidden_state_size": 64,
        },
    )

    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=INIT_HP["POP_SIZE"],
        eval_loop=1,
    )

    mutations = Mutations(
        no_mutation=0.4,
        architecture=0,  # RNN architecture mutations not standard yet
        new_layer_prob=0.0,
        parameters=0.2,
        activation=0,
        rl_hp=0.2,
        mutation_sd=0.1,
        activation_selection=["ReLU", "ELU", "GELU"],
        mutate_elite=True,
        rand_seed=1,
        device=device,
    )

    max_steps = 100000  # Max steps
    evo_steps = 10000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 3  # Number of evaluation episodes

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps * num_envs, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            agent.collect_rollouts(env)  # Collect rollouts using internal handling
            agent.learn()  # Learn from the rollout buffer

            total_steps += agent.learn_step
            agent.steps[-1] += agent.learn_step

            # printing is pretty slow...
            if total_steps % (agent.learn_step * 5) == 0:
                pbar.update(agent.learn_step * num_envs * 5 // len(pop))

            # Store scores from completed episodes during rollouts
            # Access buffer info if needed (optional)
            # completed_episode_scores = agent.rollout_buffer.get_completed_scores()
            # if completed_episode_scores:
            #     agent.scores.extend(completed_episode_scores)
            #     pop_episode_scores.append(completed_episode_scores)

        # --- Evaluate population --- #
        if total_steps // evo_steps > (total_steps - agent.learn_step) // evo_steps:
            fitnesses = [
                agent.test(
                    env,  # Use the wrapped noisy env for testing too
                    swap_channels=False,
                    max_steps=eval_steps,
                    loop=eval_loop,
                )
                for agent in pop
            ]
            # Retrieve scores for printing (average of last N evaluations)
            mean_scores = [np.mean(agent.fitness[-eval_loop:]) for agent in pop]

            print(f"--- Global steps {total_steps} ---")
            print(f"Steps {[agent.steps[-1] for agent in pop]}")
            print(f"Scores: {mean_scores}")  # Print evaluation scores
            print(f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}")
            print(
                f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}"
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

            # Update step counter
            for agent in pop:
                agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()

    # Export render after training
    print("Recording video of best agent...")

    # Create videos directory if it doesn't exist
    videos_dir = "videos"
    os.makedirs(videos_dir, exist_ok=True)

    # Find the best agent based on fitness
    best_agent_idx = np.argmax([np.mean(agent.fitness[-5:]) for agent in pop])
    best_agent = pop[best_agent_idx]

    # Create a new environment with video recording
    render_env = RecordVideo(
        NoisyObsWrapper(gym.make("CartPole-v1", render_mode="rgb_array"), real_every=5),
        videos_dir,
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="rnn_noisy_cartpole",
    )

    # Run the best agent for a few episodes
    for episode in range(3):
        obs, _ = render_env.reset()
        done = False
        episode_reward = 0
        hidden_state = np.zeros((1, 1, INIT_HP["HIDDEN_STATE_SIZE"]))

        while not done:
            # Get action from the agent with RNN state
            action, hidden_state = best_agent.get_action(
                obs, deterministic=True, hidden_state=hidden_state
            )

            # Take step in environment
            obs, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            episode_reward += reward

        print(f"Recorded Episode {episode + 1} Reward: {episode_reward}")

    render_env.close()
    print(f"Videos saved to {os.path.abspath(videos_dir)}")
