# AgileRL On-policy (RNN/MLP) MiniGrid Memory Demo
#
# This script demonstrates how to use recurrent neural networks (RNNs) or MLPs with PPO to solve the MiniGrid-DoorKey-8x8-v0 environment.
# The observation wrapper flattens the image and concatenates a one-hot encoding of the agent's direction.
# This version follows the training structure of performance_flamegraph_cartpole.py and performance_flamegraph_lunar_lander.py,
# using a population and a simple evolutionary loop.

import minigrid  # lgtm[py/unused-import] noqa: F401 pylint: disable=unused-import
import gymnasium as gym
import numpy as np
import torch
from tqdm import trange
import os
from gymnasium.wrappers import RecordVideo

from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population


# --- Define the MiniGrid Observation Wrapper ---
class MiniGridObsWrapper(gym.ObservationWrapper):
    """
    Extracts and flattens the 'image' observation from MiniGrid, normalizes to [0, 1],
    and concatenates a one-hot encoding of the 'direction' field.
    """

    def __init__(self, env):
        super().__init__(env)
        img_shape = self.observation_space["image"].shape
        flat_img_dim = np.prod(img_shape)
        # The new observation space is flattened image + 4 for direction one-hot
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(flat_img_dim + 4,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        img = obs["image"].astype(np.float32)
        flat_img = img.flatten()
        direction = obs["direction"]
        direction_onehot = np.zeros(4, dtype=np.float32)
        direction_onehot[direction] = 1.0
        concat_obs = np.concatenate([flat_img, direction_onehot], axis=0)
        return concat_obs


# --- Setup Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Toggle this to True for RNN (LSTM), False for MLP
recurrent = True  # <--- CHANGE THIS TO ENABLE/DISABLE RECURRENT

if recurrent:
    NET_CONFIG = {
        "encoder_config": {
            "hidden_state_size": 128,  # LSTM hidden state size
        },
    }
else:
    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [128],
        },
    }

# Hyperparameters
INIT_HP = {
    "POP_SIZE": 1,  # Population size
    "BATCH_SIZE": 128,
    "LEARN_STEP": 128,
    "LR": 1e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "UPDATE_EPOCHS": 4,
    "HIDDEN_STATE_SIZE": 128,
    "SHARE_ENCODERS": True,
    "DISCRETE_ACTIONS": True,
    "ACTION_STD_INIT": 0.6,
    "TARGET_KL": None,
    "CHANNELS_LAST": False,
}

# --- Create Environment and Population ---
num_envs = 64  # Fewer envs for MiniGrid due to slowness


def make_env(render_mode=None):
    def thunk():
        env = gym.make("MiniGrid-DoorKey-8x8-v0", render_mode=render_mode)
        env = MiniGridObsWrapper(env)
        return env

    return thunk


env = gym.vector.SyncVectorEnv([make_env() for _ in range(num_envs)])
single_test_env = gym.vector.SyncVectorEnv([make_env()])

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
        **({"recurrent": True, "hidden_state_size": 128} if recurrent else {}),
        "max_seq_len": 128,
    },
)

# --- Setup Evolution Components ---
tournament = TournamentSelection(
    tournament_size=2,
    elitism=True,
    population_size=INIT_HP["POP_SIZE"],
    eval_loop=1,
)

mutations = Mutations(
    no_mutation=0.4,
    architecture=0,
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

# --- Training Loop (Performance-Flamegraph Style) ---
max_steps = 5_000_000 // num_envs
required_score = 0.9
evo_steps = num_envs * INIT_HP["LEARN_STEP"] * 50
eval_steps = None
eval_loop = 5

total_steps = 0
training_complete = False

print("Training...")
pbar = trange(max_steps * num_envs, unit="step")
while (
    np.less([agent.steps[-1] for agent in pop], max_steps).all()
    and not training_complete
):
    for agent in pop:
        agent.collect_rollouts(env)
        agent.learn()
        total_steps += agent.learn_step * num_envs
        agent.steps[-1] += agent.learn_step
        pbar.update(agent.learn_step * num_envs // len(pop))

    # Evaluate and evolve
    if total_steps % evo_steps == 0:
        fitnesses = [
            agent.test(
                single_test_env,
                swap_channels=False,
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [np.mean(agent.fitness[-eval_loop:]) for agent in pop]
        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}")
        print(
            f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}"
        )

        if any(score >= required_score for score in mean_scores):
            print(
                f"\nAgent achieved required score {required_score}. Stopping training."
            )
            elite, _ = tournament.select(pop)
            training_complete = True
            break

        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)
        for agent in pop:
            agent.steps.append(agent.steps[-1])

pbar.close()
env.close()

# --- Record GIF of Best Agent ---
print("Recording GIF of best agent...")

gifs_dir = "gifs"
os.makedirs(gifs_dir, exist_ok=True)
env_to_wrap = make_env(render_mode="rgb_array")()

if not training_complete:
    fitnesses = [
        agent.test(
            env_to_wrap,
            swap_channels=False,
            max_steps=eval_steps,
            loop=eval_loop,
            vectorized=False,
        )
        for agent in pop
    ]
    elite, _ = tournament.select(pop)

render_env = gym.wrappers.RecordVideo(
    env_to_wrap, video_folder="temp_video", disable_logger=True
)

import imageio

frames = []
total_steps = 0
episode_rewards = []

for episode in range(3):
    obs, _ = render_env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    if recurrent:
        hidden_state = elite.get_initial_hidden_state(1)
    episode_frames = []

    while not done:
        frame = render_env.render()
        episode_frames.append(frame)
        if recurrent:
            action, _, _, _, hidden_state = elite.get_action(
                obs, hidden_state=hidden_state
            )
        else:
            action, _, _, _, _ = elite.get_action(obs)
        obs, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        episode_reward += reward
        episode_steps += 1

    if episode_frames:
        gif_path = os.path.join(
            gifs_dir,
            f"minigrid_memory_{'rnn' if recurrent else 'mlp'}_episode_{episode + 1}.gif",
        )
        imageio.mimsave(gif_path, episode_frames, fps=15)
        print(f"Saved GIF for episode {episode + 1} to {gif_path}")

    total_steps += episode_steps
    episode_rewards.append(episode_reward)
    print(
        f"Recorded Episode {episode + 1} Reward: {episode_reward}, Steps: {episode_steps}"
    )

avg_reward = sum(episode_rewards) / len(episode_rewards)
avg_steps = total_steps / len(episode_rewards)
print(f"Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")

render_env.close()
import shutil

if os.path.exists("temp_video"):
    shutil.rmtree("temp_video")

print(f"GIFs saved to {os.path.abspath(gifs_dir)}")
