import os

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


def add_text_to_image(
    image_array, text, position, font_size=30, font_color=(153, 255, 255)
):
    """Add text to an image represented as a numpy array.

    :param image_array: numpy array of the image.
    :param text: string of text to add.
    :param position: tuple (x, y) for the position of the text.
    :param font_size: size of the font. Default is 20.
    :param font_color: color of the font in BGR (not RGB). Default is yellow (153, 255, 255).
    :return: Modified image as numpy array.
    """
    image = Image.fromarray(image_array)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=font_color)
    modified_image_array = np.array(image)

    return modified_image_array


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    stabilize_agent = PPO.load("./models/PPO/PPO_trained_agent_stabilize.pt")
    center_agent = PPO.load("./models/PPO/PPO_trained_agent_center.pt")
    landing_agent = PPO.load("./models/PPO/PPO_trained_agent_landing.pt")

    trained_skills = {
        0: {"skill": "stabilize", "agent": stabilize_agent, "skill_duration": 40},
        1: {"skill": "center", "agent": center_agent, "skill_duration": 40},
        2: {"skill": "landing", "agent": landing_agent, "skill_duration": 40},
    }

    # Load the saved algorithm into the PPO object
    selector_path = (
        "./models/PPO/PPO_trained_agent_selector.pt"  # Path to saved agent checkpoint
    )
    agent = PPO.load(selector_path)

    # Define test loop parameters
    episodes = 3  # Number of episodes to test agent on
    max_steps = 100  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames

    print("============================================")
    print("Skill selector")

    # Test loop for inference
    for ep in range(episodes):
        state, _ = env.reset()  # Reset environment at start of episode
        frames.append(env.render())
        score = 0
        steps = 0
        for idx_step in range(max_steps):
            # Get next action from agent
            action, log_prob, _, value = agent.get_action(state)

            # Internal loop to execute trained skill
            skill_name = trained_skills[action[0]]["skill"]
            skill_agent = trained_skills[action[0]]["agent"]
            skill_duration = trained_skills[action[0]]["skill_duration"]
            reward = 0
            for skill_step in range(skill_duration):
                if state[6] or state[7]:
                    next_state, skill_reward, termination, truncation, _ = env.step(0)
                else:
                    skill_action, _, _, _ = skill_agent.get_action(state)
                    next_state, skill_reward, termination, truncation, _ = env.step(
                        skill_action[0]
                    )  # Act in environment

                # Save the frame for this step and append to frames list
                frame = env.render()
                frame = add_text_to_image(frame, skill_name, (450, 35))
                frames.append(frame)

                reward += skill_reward
                steps += 1
                if termination or truncation:
                    break
                state = next_state
            score += reward

            # Stop episode if any agents have terminated
            if termination or truncation:
                break

            state = next_state

        print("-" * 15, f"Episode: {ep+1}", "-" * 15)
        print(f"Episode length: {steps}")
        print(f"Score: {score}")

    print("============================================")

    # frames = resize_frames(frames, 0.5)
    frames = frames[::2]

    # Save the gif to specified path
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(
        os.path.join("./videos/", "LunarLander-v2_selector.gif"),
        frames,
        duration=40,
        loop=0,
    )

    env.close()
