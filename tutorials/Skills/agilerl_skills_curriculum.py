import os
from datetime import datetime

import numpy as np
import torch
import wandb
from tqdm import trange

from agilerl.algorithms.ppo import PPO
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import (
    create_population,
    make_skill_vect_envs,
    make_vect_envs,
    observation_space_channels_to_first,
)
from agilerl.wrappers.learning import Skill


class StabilizeSkill(Skill):
    def __init__(self, env):
        super().__init__(env)

        self.theta_level = 0
        self.history = {"x": [], "y": [], "theta": []}

    def skill_reward(self, observation, reward, terminated, truncated, info):
        if terminated or truncated:
            reward = -100.0
            self.history = {"x": [], "y": [], "theta": []}
            return observation, reward, terminated, truncated, info

        reward, terminated, truncated = 1.0, 0, 0
        x, y, theta = observation[0], observation[1], observation[4]

        # Ensure there are previous observations to compare with
        if len(self.history["x"]) == 0:
            self.history["x"].append(x)
            self.history["y"].append(y)
            self.history["theta"].append(theta)
            return observation, reward, terminated, truncated, info

        # Minimise x movement
        reward -= (abs(self.history["x"][-1] - x) * 10) ** 2
        # Minimise y movement
        reward -= (abs(self.history["y"][-1] - y) * 10) ** 2
        # Minimise tilt angle
        reward -= (abs(self.history["theta"][-1] - theta) * 10) ** 2

        self.history["x"].append(x)
        self.history["y"].append(y)
        self.history["theta"].append(theta)

        # Reset episode if longer than 300 steps
        if len(self.history["x"]) > 300:
            reward = 10.0
            terminated = True
            self.history = {"x": [], "y": [], "theta": []}
            self.env.reset()

        return observation, reward, terminated, truncated, info


class CenterSkill(Skill):
    def __init__(self, env):
        super().__init__(env)

        self.x_center = 0
        self.history = {"y": [], "theta": []}

    def skill_reward(self, observation, reward, terminated, truncated, info):
        if terminated or truncated:
            reward = -1000.0
            self.history = {"y": [], "theta": []}
            return observation, reward, terminated, truncated, info

        reward, terminated, truncated = 1.0, 0, 0
        x, y, theta = observation[0], observation[1], observation[4]

        # Ensure there are previous observations to compare with
        if len(self.history["y"]) == 0:
            self.history["y"].append(y)
            self.history["theta"].append(theta)
            return observation, reward, terminated, truncated, info

        # Minimise x distance to center
        reward -= abs((self.x_center - x) * 2) ** 2
        # Minimise y movement
        reward -= (abs(self.history["y"][-1] - y) * 10) ** 2
        # Minimise tilt angle
        reward -= (abs(self.history["theta"][-1] - theta) * 10) ** 2

        self.history["y"].append(y)
        self.history["theta"].append(theta)

        # Reset episode if longer than 300 steps
        if len(self.history["y"]) > 300:
            reward = 10.0
            terminated = True
            self.history = {"y": [], "theta": []}
            self.env.reset()

        return observation, reward, terminated, truncated, info


class LandingSkill(Skill):
    def __init__(self, env):
        super().__init__(env)

        self.x_landing = 0
        self.y_landing = 0
        self.theta_level = 0

    def skill_reward(self, observation, reward, terminated, truncated, info):
        if terminated or truncated:
            return observation, reward, terminated, truncated, info

        x, y, theta = observation[0], observation[1], observation[4]
        reward, terminated, truncated = 1.0, 0, 0

        # Minimise x distance to landing zone
        reward -= (abs(self.x_landing - x)) ** 2
        # Minimise y distance to landing zone
        reward -= (abs(self.y_landing - y)) ** 2
        # Minimise tilt angle
        reward -= abs(self.theta_level - theta)

        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    NET_CONFIG = {
        "encoder_config": {"hidden_size": [64]},  # Encoder hidden size
        "head_config": {"hidden_size": [64]},  # Actor head hidden size
    }

    INIT_HP = {
        "ENV_NAME": "LunarLander-v2",
        "ALGO": "PPO",
        "POPULATION_SIZE": 1,  # Population size
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "LEARN_STEP": 128,  # Learning frequency
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
        "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
        "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient
        "VF_COEF": 0.5,  # Value function coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
        "TARGET_KL": None,  # Target KL divergence threshold
        "TARGET_SCORE": 2000,
        "MAX_STEPS": 10_000_000,
        "EVO_STEPS": 10_000,
        "UPDATE_EPOCHS": 4,  # Number of policy update epochs
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "WANDB": True,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory to save trained agents and skills
    save_dir = "./models/PPO"
    os.makedirs(save_dir, exist_ok=True)

    skills = {
        "stabilize": StabilizeSkill,
        "center": CenterSkill,
        "landing": LandingSkill,
    }

    for skill in skills.keys():
        env = make_skill_vect_envs(
            INIT_HP["ENV_NAME"], skills[skill], num_envs=1
        )  # Create environment

        observation_space = env.single_observation_space
        action_space = env.single_action_space

        pop = create_population(
            algo="PPO",  # Algorithm
            observation_space=observation_space,  # Observation space
            action_space=action_space,  # Action space
            net_config=NET_CONFIG,  # Network configuration
            INIT_HP=INIT_HP,  # Initial hyperparameters
            population_size=INIT_HP["POPULATION_SIZE"],  # Population size
            device=device,
        )

        trained_pop, pop_fitnesses = train_on_policy(
            env=env,  # Gym-style environment
            env_name=f"{INIT_HP['ENV_NAME']}-{skill}",  # Environment name
            algo=INIT_HP["ALGO"],  # Algorithm
            pop=pop,  # Population of agents
            swap_channels=INIT_HP[
                "CHANNELS_LAST"
            ],  # Swap image channel from last to first
            max_steps=INIT_HP["MAX_STEPS"],  # Max number of training episodes
            evo_steps=INIT_HP["EVO_STEPS"],  # Evolution frequency
            target=INIT_HP["TARGET_SCORE"],  # Target score for early stopping
            tournament=None,  # Tournament selection object
            mutation=None,  # Mutations object
            wb=INIT_HP["WANDB"],  # Weights and Biases tracking
        )

        # Save the trained algorithm
        filename = f"PPO_trained_agent_{skill}.pt"
        save_path = os.path.join(save_dir, filename)
        trained_pop[0].save_checkpoint(save_path)

        env.close()

    # Now train the skill selector, which will choose which of the learned skills to use
    # First load the learned skill agents
    stabilize_agent = PPO.load(os.path.join(save_dir, "PPO_trained_agent_stabilize.pt"))
    center_agent = PPO.load(os.path.join(save_dir, "PPO_trained_agent_center.pt"))
    landing_agent = PPO.load(os.path.join(save_dir, "PPO_trained_agent_landing.pt"))

    trained_skills = {
        0: {"skill": "stabilize", "agent": stabilize_agent, "skill_duration": 40},
        1: {"skill": "center", "agent": center_agent, "skill_duration": 40},
        2: {"skill": "landing", "agent": landing_agent, "skill_duration": 40},
    }

    env = make_vect_envs(INIT_HP["ENV_NAME"], num_envs=1)  # Create environment

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    action_dim = len(
        trained_skills
    )  # Selector will be trained to choose which trained skill to use

    if INIT_HP["CHANNELS_LAST"]:
        observation_space = observation_space_channels_to_first(observation_space)

    pop = create_population(
        algo="PPO",  # Algorithm
        observation_space=observation_space,  # Observation space
        action_dim=action_dim,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device,
    )

    if INIT_HP["WANDB"]:
        wandb.init(
            # set the wandb project where this run will be logged
            project="EvoWrappers",
            name="{}-EvoHPO-{}-{}".format(
                INIT_HP["ENV_NAME"],
                INIT_HP["ALGO"],
                datetime.now().strftime("%m%d%Y%H%M%S"),
            ),
            # track hyperparameters and run metadata
            config={
                "algo": f"Evo HPO {INIT_HP['ALGO']}",
                "env": INIT_HP["ENV_NAME"],
                "INIT_HP": INIT_HP,
            },
        )

    bar_format = "{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]"
    pbar = trange(INIT_HP["MAX_STEPS"], unit="step", bar_format=bar_format, ascii=True)

    total_steps = 0

    # RL training loop
    while np.less([agent.steps[-1] for agent in pop], INIT_HP["MAX_STEPS"]).all():
        for agent in pop:  # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            score = 0

            states = []
            actions = []
            log_probs = []
            rewards = []
            dones = []
            values = []

            done = np.zeros(1)

            for idx_step in range(500):
                # Get next action from agent
                action, log_prob, _, value = agent.get_action(state)

                # Internal loop to execute trained skill
                skill_agent = trained_skills[action[0]]["agent"]
                skill_duration = trained_skills[action[0]]["skill_duration"]
                reward = 0
                for skill_step in range(skill_duration):
                    # If landed, do nothing
                    if state[0][6] or state[0][7]:
                        next_state, skill_reward, termination, truncation, _ = env.step(
                            [0]
                        )
                    else:
                        skill_action, _, _, _ = skill_agent.get_action(state)
                        next_state, skill_reward, termination, truncation, _ = env.step(
                            skill_action
                        )  # Act in environment
                    next_done = np.logical_or(termination, truncation).astype(np.int8)
                    reward += skill_reward
                    if np.any(termination) or np.any(truncation):
                        break
                    state = next_state
                    done = next_done
                score += reward

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

            agent.scores.append(score)

            # Learn according to agent's RL algorithm
            agent.learn(
                (
                    states,
                    actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                    next_state,
                    next_done,
                )
            )

            agent.steps[-1] += idx_step + 1
            total_steps += idx_step + 1

        if (agent.steps[-1]) % INIT_HP["EVO_STEPS"] == 0:
            mean_scores = np.mean([agent.scores[-20:] for agent in pop], axis=1)
            if INIT_HP["WANDB"]:
                wandb.log(
                    {
                        "global_step": total_steps,
                        "train/mean_score": np.mean(mean_scores),
                    }
                )
            print(
                f"""
                --- Global Steps {total_steps} ---
                Score:\t\t{mean_scores}
                """,
                end="\r",
            )

    if INIT_HP["WANDB"]:
        wandb.finish()
    env.close()

    # Save the trained selector
    filename = "PPO_trained_agent_selector.pt"
    save_path = os.path.join(save_dir, filename)
    pop[0].save_checkpoint(save_path)
