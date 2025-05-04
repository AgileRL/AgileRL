# Sequence Parity RNN-Only Memory Game Demo
#
# This environment requires the agent to remember a sequence of random binary digits (0 or 1)
# shown one at a time. At the end of the sequence, the agent must output whether the total
# number of 1s in the sequence was even or odd (i.e., the parity). The agent receives a reward
# only if it correctly outputs the parity. Only an RNN can solve this; an MLP cannot, as the
# relevant information is distributed across time.

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population

# --- Define the Sequence Parity RNN-Only Memory Game Environment ---
class SequenceParityMemoryEnv(gym.Env):
    """
    Observation: At each step t < seq_len, agent sees a single bit (0 or 1) as a one-hot vector [1,0] or [0,1].
    At the final step, agent receives a special query observation [0.5, 0.5] and must output the parity (even=0, odd=1).
    Action: discrete, 2 (0=even, 1=odd).
    Reward: +1 if action matches the parity at the query step, else 0.
    Episode: seq_len + 1 steps (seq_len bits, then query).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, seq_len=5, render_mode=None):
        super().__init__()
        self.seq_len = seq_len
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)  # 0=even, 1=odd
        self.render_mode = render_mode
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.sequence = np.random.randint(0, 2, size=self.seq_len)
        self.done = False
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        obs = np.zeros(2, dtype=np.float32)
        obs[self.sequence[0]] = 1.0  # Show first bit as one-hot
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        if self.current_step < self.seq_len:
            # Show next bit in sequence
            obs = np.zeros(2, dtype=np.float32)
            obs[self.sequence[self.current_step]] = 1.0
            reward = 0.0
            terminated = False
        elif self.current_step == self.seq_len:
            # Query step: ask for parity
            obs = np.array([0.5, 0.5], dtype=np.float32)  # Query signal
            self.last_action = action
            parity = np.sum(self.sequence) % 2
            reward = 1.0 if action == parity else 0.0
            terminated = True
            self.done = True
        else:
            # Episode is over
            obs = np.zeros(2, dtype=np.float32)
            reward = 0.0
            terminated = True
            self.done = True

        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.current_step < self.seq_len:
            print(
                f"Step {self.current_step}: Show bit {self.sequence[self.current_step]}"
            )
        elif self.current_step == self.seq_len:
            parity = np.sum(self.sequence) % 2
            print(
                f"Query: Agent answered {self.last_action} (0=even, 1=odd), correct was {parity} (sequence: {self.sequence})"
            )

    def close(self):
        pass


# --- Setup Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Toggle this to True for RNN (LSTM), False for MLP
recurrent = True  # <--- CHANGE THIS TO ENABLE/DISABLE RECURRENT

if recurrent:
    NET_CONFIG = {
        "encoder_config": {
            "hidden_state_size": 32,  # LSTM hidden state size (small, task is simple)
        },
    }
else:
    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [32],
        },
    }

# --- Create Environment and Population ---
seq_len = 2
num_envs = 32  # Fewer envs, since task is simple

# Hyperparameters
INIT_HP = {
    "POP_SIZE": 8,  # Population size
    "BATCH_SIZE": 256,
    "LEARN_STEP": seq_len + 1,  # Steps per episode
    "HIDDEN_STATE_SIZE": 32,
    "LR": 1e-3,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.001,
    "VF_COEF": 1.0,
    "MAX_GRAD_NORM": 0.5,
    "UPDATE_EPOCHS": 4,
    "SHARE_ENCODERS": True,
    "DISCRETE_ACTIONS": True,
    "ACTION_STD_INIT": 0.6,
    "TARGET_KL": None,
    "CHANNELS_LAST": False,
}


def make_env():
    def thunk():
        return SequenceParityMemoryEnv(seq_len=seq_len)

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
        **(
            {"recurrent": True, "hidden_state_size": INIT_HP["HIDDEN_STATE_SIZE"]}
            if recurrent
            else {}
        ),
        "max_seq_len": seq_len + 1,
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
    no_mutation=0.5,
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

# --- Training Loop ---
max_steps = 500_000 // num_envs
required_score = 1
evo_steps = num_envs * INIT_HP["LEARN_STEP"] * 100
eval_steps = None
eval_loop = 10

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
    
    # check if Evaluate and evolve
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
        mean_scores = [round(np.mean(agent.fitness[-eval_loop:]), 2) for agent in pop]
        print(f"--- Global steps {total_steps} ---")
        print(f"Scores: {mean_scores}")

        if any(score >= required_score for score in mean_scores):
            print(
                f"\nAgent achieved required score {required_score}. Stopping training."
            )
            elite, _ = tournament.select(pop)
            training_complete = True
            break

        elite, pop = tournament.select(pop)
        for agent in pop:
            agent.steps.append(agent.steps[-1])

pbar.close()
env.close()

# --- Evaluate Best Agent ---
print("Evaluating best agent...")

if not training_complete:
    fitnesses = [
        agent.test(
            single_test_env,
            swap_channels=False,
            max_steps=eval_steps,
            loop=eval_loop,
            vectorized=True,
        )
        for agent in pop
    ]
    elite, _ = tournament.select(pop)

# --- Run a few episodes and print results ---
print("Running a few episodes with the best agent:")
total_steps = 0
episode_rewards = []

for episode in range(10):
    obs, _ = single_test_env.reset()
    done = np.array([False])
    episode_reward = 0
    episode_steps = 0
    if recurrent:
        hidden_state = elite.get_initial_hidden_state(1)
    while not done[0]:
        if recurrent:
            action, _, _, _, hidden_state = elite.get_action(
                obs, hidden_state=hidden_state
            )
        else:
            action, _, _, _, _ = elite.get_action(obs)
        obs, reward, terminated, truncated, _ = single_test_env.step(action)
        done = np.logical_or(terminated, truncated)
        episode_reward += reward[0]
        episode_steps += 1
    print(f"Episode {episode + 1}: Reward: {episode_reward}, Steps: {episode_steps}")
    total_steps += episode_steps
    episode_rewards.append(episode_reward)

avg_reward = sum(episode_rewards) / len(episode_rewards)
avg_steps = total_steps / len(episode_rewards)
print(f"Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")

print("Demo complete.")
