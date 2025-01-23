import torch
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict

from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.modules.multi_input import EvolvableMultiInput
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.algo_utils import observation_space_channels_to_first
from agilerl.utils.utils import (
    create_population,
    print_hyperparams
)

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')

import sys
sys.path.append('../racecar_gym')

import racecar_gym

class FlattenActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(FlattenActionWrapper, self).__init__(env)

        # Get the original action space (Dict)
        original_action_space = self.env.action_space

        # Check if it's a Dict space
        if isinstance(original_action_space, Dict):
            # Flatten the Dict space into a single Box space
            low = np.concatenate([original_action_space['motor'].low, original_action_space['steering'].low])
            high = np.concatenate([original_action_space['motor'].high, original_action_space['steering'].high])

            self.action_space = Box(low=low, high=high, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"Expected Dict action space, but got {type(original_action_space)}.")

    def reset(self, **kwargs):
        # Reset the environment and return the initial observation
        obs, info = self.env.reset(**kwargs)
        obs['rgb_camera'] = obs['rgb_camera'].astype(np.uint8)
        return obs, info

    def step(self, action):
        # Convert the flattened action back into the original Dict format
        motor_action = action[0]
        steering_action = action[1]
        
        # Create the Dict action
        action_dict = {
            'motor': np.array([motor_action], dtype=np.float32),
            'steering': np.array([steering_action], dtype=np.float32)
        }

        # Pass the action to the environment
        obs, reward, done, truncated, info = self.env.step(action_dict)

        # change dtype of rgb_camera to uint8
        obs['rgb_camera'] = obs['rgb_camera'].astype(np.uint8)
        
        return obs, reward, done, truncated, info

def make_vect_envs(env_name, num_envs):
    return gym.vector.AsyncVectorEnv(
        [lambda: FlattenActionWrapper(gym.make(env_name, render_mode="rgb_array_birds_eye")) for i in range(num_envs)]
    )

def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============ AgileRL ============")
    print(f"DEVICE: {device}")

    env = make_vect_envs(INIT_HP["ENV_NAME"], num_envs=INIT_HP["NUM_ENVS"])

    observation_space = env.single_observation_space
    action_space = env.single_action_space
    if INIT_HP["CHANNELS_LAST"]:
        observation_space = observation_space_channels_to_first(observation_space)

    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )
    mutations = Mutations(
        no_mutation=MUTATION_PARAMS["NO_MUT"],
        architecture=MUTATION_PARAMS["ARCH_MUT"],
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],
        parameters=MUTATION_PARAMS["PARAMS_MUT"],
        activation=MUTATION_PARAMS["ACT_MUT"],
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],
        mutation_sd=MUTATION_PARAMS["MUT_SD"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    if use_net:
        actor = EvolvableMultiInput(
            observation_space=observation_space,
            channel_size=[16, 16],
            kernel_size=[3, 3],
            stride_size=[1, 1],
            hidden_size=[64, 64],
            latent_dim=16,
            num_outputs=action_space.shape[0],
            device=device
        )

        critic = EvolvableMultiInput(
            observation_space=observation_space,
            channel_size=[16, 16],
            kernel_size=[3, 3],
            stride_size=[1, 1],
            hidden_size=[64, 64],
            latent_dim=16,
            num_outputs=1,
            device=device,
        )
    else:
        actor = None
        critic = None

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        actor_network=actor,
        critic_network=critic,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=INIT_HP["NUM_ENVS"],
        device=device,
    )

    trained_pop, pop_fitnesses = train_on_policy(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
    )

    print_hyperparams(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    with open("configs/training/multi_input.yaml") as file:
        ppo_config = yaml.safe_load(file)
    INIT_HP = ppo_config["INIT_HP"]
    MUTATION_PARAMS = ppo_config["MUTATION_PARAMS"]
    NET_CONFIG = ppo_config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=False)
