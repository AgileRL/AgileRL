# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime, timezone

import numpy as np
import torch
import wandb
from gymnasium.spaces import Discrete

from agilerl.algorithms.ppo import PPO
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import (
    make_skill_vect_envs,
    make_vect_envs,
)
from agilerl.wrappers.learning import Skill


class StabilizeSkill(Skill):
    def __init__(self, env):
        super().__init__(env)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "LunarLander-v3"

    # Network configuration
    net_config = {
        "encoder_config": {"hidden_size": [64]},
        "head_config": {"hidden_size": [64]},
    }

    # Algorithm hyperparameters
    init_hp = {
        "batch_size": 128,
        "lr": 1e-3,
        "learn_step": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "action_std_init": 0.6,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "update_epochs": 4,
    }

    # Training parameters
    population_size = 1
    max_steps = 10_000_000
    evo_steps = 10_000
    target_score = 2000
    use_wandb = True

    # Directory to save trained agents and skills
    save_dir = "./models/PPO"
    os.makedirs(save_dir, exist_ok=True)

    skills = {
        "stabilize": StabilizeSkill,
        "center": CenterSkill,
        "landing": LandingSkill,
    }

    for skill in skills:
        env = make_skill_vect_envs(
            env_name,
            skills[skill],
            num_envs=1,
        )  # Create environment

        observation_space = env.single_observation_space
        action_space = env.single_action_space

        pop = PPO.population(
            size=population_size,
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            device=device,
            **init_hp,
        )

        trained_pop, pop_fitnesses = train_on_policy(
            env=env,  # Gym-style environment
            env_name=f"{env_name}-{skill}",  # Environment name
            algo="PPO",  # Algorithm
            pop=pop,  # Population of agents
            max_steps=max_steps,  # Max number of training episodes
            evo_steps=evo_steps,  # Evolution frequency
            target=target_score,  # Target score for early stopping
            selection_strategy=None,  # Selection strategy object
            mutation=None,  # Mutations object
            wb=True,  # Weights and Biases tracking
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

    env = make_vect_envs(env_name, num_envs=1)  # Create environment

    observation_space = env.single_observation_space

    action_dim = len(
        trained_skills,
    )  # Selector will be trained to choose which trained skill to use

    pop = PPO.population(
        size=population_size,
        observation_space=observation_space,
        action_space=Discrete(action_dim),
        net_config=net_config,
        device=device,
        **init_hp,
    )

    if use_wandb:
        wandb.init(
            project="EvoWrappers",
            name="{}-EvoHPO-{}-{}".format(
                env_name,
                "PPO",
                datetime.now(tz=timezone.utc).strftime("%m%d%Y%H%M%S"),
            ),
            config={
                "algo": "Evo HPO PPO",
                "env": env_name,
                "init_hp": init_hp,
            },
        )

    # RL training loop
    total_steps = 0
    while np.less([agent.steps for agent in pop], max_steps).all():
        for agent in pop:  # Loop through population
            agent.set_training_mode(True)

            for _ in range(-(evo_steps // -agent.learn_step)):
                state = env.reset()[0]  # Reset environment at start of episode
                score = 0
                done = np.zeros(1)

                agent.rollout_buffer.reset()
                for _ in range(agent.learn_step):
                    obs = state  # Observation used to select the skill

                    # Get next action from agent
                    action, log_prob, _, value = agent.get_action(obs)

                    # Internal loop to execute trained skill
                    skill_agent = trained_skills[action[0]]["agent"]
                    skill_duration = trained_skills[action[0]]["skill_duration"]
                    reward = 0
                    next_state, next_done = state, done
                    for _ in range(skill_duration):
                        # If landed, do nothing
                        if state[0][6] or state[0][7]:
                            next_state, skill_reward, termination, truncation, _ = (
                                env.step(
                                    [0],
                                )
                            )
                        else:
                            skill_action, _, _, _ = skill_agent.get_action(state)
                            next_state, skill_reward, termination, truncation, _ = (
                                env.step(
                                    skill_action,
                                )
                            )  # Act in environment
                        next_done = np.logical_or(termination, truncation).astype(
                            np.int8
                        )
                        reward += skill_reward
                        if np.any(termination) or np.any(truncation):
                            break
                        state = next_state
                    score += reward

                    # Save experience in the agent's rollout buffer
                    agent.rollout_buffer.add(
                        obs=obs,
                        action=action,
                        reward=np.atleast_1d(reward),
                        done=np.atleast_1d(next_done),
                        value=np.atleast_1d(value),
                        log_prob=np.atleast_1d(log_prob),
                        next_obs=next_state,
                    )

                    state = next_state
                    done = next_done

                agent.scores.append(score)

                # Bootstrap the final state value and learn from the rollout buffer
                _, _, _, last_value = agent.get_action(state)
                agent.rollout_buffer.compute_returns_and_advantages(
                    last_value=np.atleast_1d(last_value),
                    last_done=np.atleast_1d(done),
                )
                agent.learn()

                agent.steps += agent.learn_step
                total_steps += agent.learn_step

        mean_scores = np.mean([agent.scores[-20:] for agent in pop], axis=1)
        if use_wandb:
            wandb.log(
                {
                    "global_step": total_steps,
                    "train/mean_score": np.mean(mean_scores),
                },
            )
        print(
            f"""
            --- Global Steps {total_steps} ---
            Score:\t\t{mean_scores}
            """,
            end="\r",
        )

    wandb.finish()
    env.close()

    # Save the trained selector
    filename = "PPO_trained_agent_selector.pt"
    save_path = os.path.join(save_dir, filename)
    pop[0].save_checkpoint(save_path)
