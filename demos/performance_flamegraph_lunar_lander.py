# AgileRL Performance Profiling with Flamegraphs
# This script demonstrates profiling AgileRL training using flamegraphs to identify bottlenecks.

import webbrowser
import torch
from tqdm import trange
import time
import os
import glob
import io
import gymnasium as gym

# Profiling tools
import cProfile
import pstats
import pyinstrument
from torch.profiler import profile, record_function, ProfilerActivity
from IPython.display import display, HTML, Video
import tempfile
from agilerl.utils.utils import create_population, make_vect_envs

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:
# import sys
# sys.path.append('../')

# =====================================================================
# SETUP CONFIGURATION
# =====================================================================
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Network configuration
NET_CONFIG = {
    "encoder_config": {
        "hidden_size": [256, 256],  # Larger MLP hidden size for LunarLander
    },
}

# Hyperparameters
INIT_HP = {
    "POP_SIZE": 1,  # Single agent for profiling
    "BATCH_SIZE": 128,
    "LEARN_STEP": 128,  # Smaller learn step for profiling
    "LR": 3e-4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_COEF": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "UPDATE_EPOCHS": 4,
    "SHARE_ENCODERS": True,
    # PPO Specific
    "DISCRETE_ACTIONS": True,  # LunarLander-v3 has discrete actions
    "ACTION_STD_INIT": 0.6,  # Only used for continuous actions
    "TARGET_KL": None,
    "CHANNELS_LAST": False,  # LunarLander obs are 1D
}

# =====================================================================
# CREATE ENVIRONMENT AND AGENT
# =====================================================================
# Create vectorized environment
num_envs = 64  # Number of parallel environments for profiling
env = make_vect_envs("LunarLander-v3", num_envs=num_envs, should_async_vector=False)

observation_space = env.single_observation_space
action_space = env.single_action_space

# Create population of agents (just one for profiling)
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
    },
)

# Get the agent from the population
agent = pop[0]

# =====================================================================
# EXAMPLE PROFILING WITH CPROFILE
# =====================================================================
print("\n--- Profiling with cProfile ---")
# Profile the collect_rollouts method
pr = cProfile.Profile()
pr.enable()

# Run the function to profile
agent.collect_rollouts(env)

pr.disable()

# Print the stats
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(12)  # Print top 12 functions by cumulative time
print(s.getvalue())

# =====================================================================
# EXAMPLE PROFILING WITH PYINSTRUMENT
# =====================================================================
print("\n--- Profiling with pyinstrument ---")
# Profile the learn method with pyinstrument
profiler = pyinstrument.Profiler()
profiler.start()

# Run the function to profile
agent.learn()

profiler.stop()

# =====================================================================
# PROFILING A COMPLETE TRAINING LOOP
# =====================================================================
# Choose whether to profile the full training loop
use_profiler = False  # Set to True to enable flamegraph profiling for the full loop

# Training parameters
max_steps = 400000 // num_envs  # Reduced for profiling
total_steps = 0
start_time = time.time()

# Start profiler if enabled
if use_profiler:
    full_profiler = pyinstrument.Profiler()
    full_profiler.start()

# TRAINING LOOP
print("\n--- Running Training Loop ---")
pbar = trange(max_steps * num_envs, unit="step")
while total_steps < max_steps:
    # Collect rollouts and learn
    agent.collect_rollouts(env)  # Collect rollouts for each environment
    agent.learn()
    # Update counters
    total_steps += INIT_HP["LEARN_STEP"]
    pbar.update(INIT_HP["LEARN_STEP"] * num_envs)

pbar.close()
env.close()

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

# Stop profiler and save results if enabled
if use_profiler:
    full_profiler.stop()
    # Save the flamegraph to a file
    flamegraph_file = os.path.abspath("flamegraph_training.html")
    with open(flamegraph_file, "w") as f:
        f.write(full_profiler.output_html())
    print(f"Saving flamegraph to {flamegraph_file}")
    webbrowser.open(flamegraph_file)

# =====================================================================
# TEST AGENT AND RECORD VIDEO
# =====================================================================
print("\n--- Testing Agent and Recording Video ---")
# Define video folder
video_folder = "videos_ppo_test"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# Create a single synchronous environment for testing and recording
# We use gym.make directly as RecordVideo works best with a base environment
# render_mode='rgb_array' is required for RecordVideo
testing_env_single = gym.make("LunarLander-v3", render_mode="rgb_array")

# Wrap the environment for recording
# Record only the first episode (episode_trigger=lambda x: x == 0)
recorded_env = gym.wrappers.RecordVideo(
    testing_env_single, video_folder=video_folder, episode_trigger=lambda x: x == 0
)

# Test the agent using the recorded environment
print("Testing agent and recording video...")
# Run test for 1 loop (episode)
mean_reward = agent.test(recorded_env, loop=1, max_steps=1000, vectorized=False)
print(f"Achieved mean reward of: {mean_reward}")

# Close the environment wrapper (this also closes the base environment)
recorded_env.close()

# Find and display the recorded video
video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
if video_files:
    # Display the latest video
    video_path = sorted(video_files)[-1]
    print(f"Displaying video: {video_path}")
    display(Video(video_path, embed=True, html_attributes="loop autoplay"))
else:
    print(f"No video found in {video_folder}. Recording might have failed.")
    print(
        "Ensure ffmpeg is installed (`conda install ffmpeg` or `apt-get install ffmpeg`)."
    )
    print("If running headless, ensure necessary libraries (e.g., xvfb) are installed.")

# =====================================================================
# ADDITIONAL PROFILING WITH PYTORCH PROFILER
# =====================================================================
print("\n--- Profiling with PyTorch Profiler ---")
# PyTorch profiler example
with profile(
    activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True
) as prof:
    with record_function("training_step"):
        agent.collect_rollouts(env)
        agent.learn()

# Export trace that can be loaded in chrome://tracing
prof.export_chrome_trace("pytorch_trace.json")
print("PyTorch profiler trace saved to pytorch_trace.json")
