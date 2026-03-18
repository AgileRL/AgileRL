import os

import gymnasium as gym
import imageio
import numpy as np
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
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

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

        frames = []  # List to collect frames

        print("============================================")
        print(f"Skill: {skill}")

        # Test loop for inference
        for ep in range(episodes):
            state, _ = env.reset()  # Reset environment at start of episode
            frames.append(env.render())
            score = 0
            for step in range(max_steps):
                # Get next action from agent
                if state[6] or state[7]:
                    action = [0]
                else:
                    action, _, _, _ = agent.get_action(state)
                next_state, reward, termination, truncation, _ = env.step(
                    action[0],
                )  # Act in environment

                # Save the frame for this step and append to frames list
                frames.append(env.render())

                score += reward

                # Stop episode if any agents have terminated
                if termination or truncation:
                    break

                state = next_state

            print("-" * 15, f"Episode: {ep + 1}", "-" * 15)
            print(f"Episode length: {step}")
            print(f"Score: {score}")

        print("============================================")

        frames = frames[::2]

        # Save the gif to specified path
        gif_path = "./videos/"
        os.makedirs(gif_path, exist_ok=True)
        imageio.mimwrite(
            os.path.join("./videos/", f"LunarLander-v3_{skill}.gif"),
            frames,
            duration=40,
            loop=0,
        )

    env.close()
