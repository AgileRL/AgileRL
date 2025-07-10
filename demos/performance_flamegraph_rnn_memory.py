# AgileRL Performance Profiling with Flamegraphs
# This script demonstrates profiling AgileRL training using flamegraphs to identify bottlenecks
# for the Memory Game RNN/MLP demo (see demo_on_policy_rnn_memory.py).

# Profiling tools
import cProfile
import io
import os
import pstats
import time
import webbrowser

import gymnasium as gym

# --- Memory Game Environment (from demo_on_policy_rnn_memory.py) ---
import numpy as np
import pyinstrument
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import trange

from agilerl.rollouts.on_policy import collect_rollouts_recurrent
from agilerl.utils.utils import create_population


class MemoryGameEnv(gym.Env):
    """
    Observation: one-hot vector of length n_symbols, or all zeros during delay.
    Action: discrete, n_symbols.
    Reward: +1 if action matches the original symbol at the query step, else 0.
    Episode: Each episode is delay_steps+2 steps (show, delay, query).
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, n_symbols=5, delay_steps=2, render_mode=None):
        super().__init__()
        self.n_symbols = n_symbols
        self.delay_steps = delay_steps
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_symbols,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(n_symbols)
        self.render_mode = render_mode
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.symbol = np.random.randint(self.n_symbols)
        self.done = False
        self.last_action = None
        self.reward_given = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        obs = np.zeros(self.n_symbols, dtype=np.float32)
        obs[self.symbol] = 1.0  # Show symbol at first step
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        obs = np.zeros(self.n_symbols, dtype=np.float32)  # Default to zeros
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Note: Step 0 (symbol display) is handled by reset()

        if self.current_step <= self.delay_steps:
            # Delay steps: obs remains zeros
            pass
        elif self.current_step == self.delay_steps + 1:
            # Query step: Set a distinct observation
            obs = np.ones(
                self.n_symbols, dtype=np.float32
            )  # <-- Change observation here
            self.last_action = action
            if action == self.symbol:
                reward = 1.0
            terminated = True
            self.done = True
        else:
            # Should not happen
            terminated = True
            self.done = True

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.current_step == 0:
            print(f"Show symbol: {self.symbol}")
        elif self.current_step <= self.delay_steps:
            print(f"Delay step {self.current_step}")
        elif self.current_step == self.delay_steps + 1:
            print(
                f"Query: Agent answered {self.last_action}, correct was {self.symbol}"
            )

    def close(self):
        pass


# --- Setup Configuration (from demo_on_policy_rnn_memory.py) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Toggle this to True for RNN (LSTM), False for MLP
recurrent = True  # <--- CHANGE THIS TO ENABLE/DISABLE RECURRENT

# --- Create Environment and Population ---
n_symbols = 5
delay_steps = 4
num_envs = 512  # Can be higher for faster training/profiling


if recurrent:
    NET_CONFIG = {
        "encoder_config": {
            "hidden_state_size": 128,  # LSTM hidden state size
            "max_seq_len": delay_steps + 2,
        },
    }
else:
    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [64],
        },
    }


# Hyperparameters
INIT_HP = {
    "POP_SIZE": 1,  # Single agent for profiling
    "BATCH_SIZE": 512,
    "LEARN_STEP": (delay_steps + 2),  # Match episode length (delay_steps + 2)
    "HIDDEN_STATE_SIZE": 128,
    "LR": 1e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 1.0,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.001,
    "VF_COEF": 1.0,
    "MAX_GRAD_NORM": 0.5,
    "UPDATE_EPOCHS": 2,
    "SHARE_ENCODERS": True,
    "USE_ROLLOUT_BUFFER": True,
    "RECURRENT": recurrent,
    "ACTION_STD_INIT": 0.6,
    "TARGET_KL": None,
    "CHANNELS_LAST": False,
}


def make_env():
    def thunk():
        return MemoryGameEnv(n_symbols=n_symbols, delay_steps=delay_steps)

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
)

agent = pop[0]

# =====================================================================
# EXAMPLE PROFILING WITH CPROFILE
# =====================================================================
print("\n--- Profiling with cProfile ---")
pr = cProfile.Profile()
pr.enable()
collect_rollouts_recurrent(agent, env)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(12)
print(s.getvalue())

# =====================================================================
# EXAMPLE PROFILING WITH PYINSTRUMENT
# =====================================================================
print("\n--- Profiling with pyinstrument ---")
profiler = pyinstrument.Profiler()
profiler.start()
agent.learn()
profiler.stop()
# You can print or save the profiler output if desired:
# print(profiler.output_text(unicode=True, color=True))

# =====================================================================
# PROFILING A COMPLETE TRAINING LOOP
# =====================================================================
use_profiler = True  # Set to True to enable flamegraph profiling for the full loop

max_steps = 10_000_000 // num_envs  # Reduced for profiling
total_steps = 0
start_time = time.time()

if use_profiler:
    full_profiler = pyinstrument.Profiler()
    full_profiler.start()

print("\n--- Running Training Loop ---")
pbar = trange(max_steps * num_envs, unit="step")
while total_steps < max_steps:
    collect_rollouts_recurrent(agent, env)
    agent.learn()
    total_steps += INIT_HP["LEARN_STEP"]
    pbar.update(INIT_HP["LEARN_STEP"] * num_envs)
pbar.close()
env.close()

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

if use_profiler:
    full_profiler.stop()
    flamegraph_file = os.path.abspath("flamegraph_training.html")
    with open(flamegraph_file, "w") as f:
        f.write(full_profiler.output_html())
    print(f"Saving flamegraph to {flamegraph_file}")
    webbrowser.open(flamegraph_file)

# =====================================================================
# TEST AGENT AND RECORD VIDEO (OPTIONAL)
# =====================================================================
print("\n--- Testing Agent and Recording Video ---")
video_folder = "videos_rnn_memory_test"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# For MemoryGameEnv, we use a single SyncVectorEnv for testing
testing_env_single = gym.vector.SyncVectorEnv([make_env()])

# Optionally, you could wrap with RecordVideo if you implement a render_mode='rgb_array' for MemoryGameEnv
# For now, we just run a test and print results
print("Testing agent...")
mean_reward = agent.test(testing_env_single, loop=5, max_steps=None, vectorized=True)
print(f"Achieved mean reward of: {mean_reward}")

# =====================================================================
# ADDITIONAL PROFILING WITH PYTORCH PROFILER
# =====================================================================
print("\n--- Profiling with PyTorch Profiler ---")
with profile(
    activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True
) as prof:
    with record_function("training_step"):
        collect_rollouts_recurrent(agent, env)
        agent.learn()

prof.export_chrome_trace("pytorch_trace.json")
print("PyTorch profiler trace saved to pytorch_trace.json")
