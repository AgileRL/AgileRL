import os

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image

from agilerl.algorithms.ppo import PPO


# Resizes frames to make file size smaller
def resize_frames(frames, fraction):
    resized_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        new_width = int(img.width * fraction)
        new_height = int(img.height * fraction)
        img_resized = img.resize((new_width, new_height))
        resized_frames.append(np.array(img_resized))

    return resized_frames


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    skills = ["stabilize", "center", "landing"]

    for skill in skills:
        # Load the saved algorithm into the PPO object
        path = f"./models/PPO/PPO_trained_agent_{skill}.pt"  # Path to saved agent checkpoint
        agent = PPO.load(path)

        # Define test loop parameters
        episodes = 3  # Number of episodes to test agent on
        max_steps = (
            500  # Max number of steps to take in the environment in each episode
        )

        rewards = []  # List to collect total episodic reward
        frames = []  # List to collect frames

        print("============================================")
        print(f"Skill: {skill}")

        # Test loop for inference
        for ep in range(episodes):
            state, _ = env.reset()  # Reset environment at start of episode
            frames.append(env.render())
            score = 0
            for idx_step in range(max_steps):
                # Get next action from agent
                if state[6] or state[7]:
                    action = [0]
                else:
                    action, log_prob, _, value = agent.get_action(state)
                next_state, reward, termination, truncation, _ = env.step(
                    action[0]
                )  # Act in environment

                # Save the frame for this step and append to frames list
                frames.append(env.render())

                score += reward

                # Stop episode if any agents have terminated
                if termination or truncation:
                    break

                state = next_state

            print("-" * 15, f"Episode: {ep+1}", "-" * 15)
            print(f"Episode length: {idx_step}")
            print(f"Score: {score}")

        print("============================================")

        frames = frames[::2]

        # Save the gif to specified path
        gif_path = "./videos/"
        os.makedirs(gif_path, exist_ok=True)
        imageio.mimwrite(
            os.path.join("./videos/", f"LunarLander-v2_{skill}.gif"),
            frames,
            duration=40,
            loop=0,
        )

    env.close()
