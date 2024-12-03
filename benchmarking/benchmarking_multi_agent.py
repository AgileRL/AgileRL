import importlib

import supersuit as ss
import torch
import yaml
from accelerate import Accelerator

from agilerl.algorithms.core.base import MultiAgentAlgorithm
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.modules.mlp import EvolvableMLP
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.utils.algo_utils import observation_space_channels_to_first
from agilerl.utils.utils import (
    create_population,
    make_multi_agent_vect_envs
)

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, DISTRIBUTED_TRAINING, use_net=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============ AgileRL Multi-agent benchmarking ============")

    if DISTRIBUTED_TRAINING:
        accelerator = Accelerator()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("===== Distributed Training =====")
        accelerator.wait_for_everyone()
    else:
        accelerator = None
    print(f"DEVICE: {device}")

    env = importlib.import_module(f"{INIT_HP['ENV_NAME']}").parallel_env
    env_kwargs = dict(max_cycles=25, continuous_actions=True)
    env = make_multi_agent_vect_envs(env, num_envs=INIT_HP["NUM_ENVS"], **env_kwargs)

    if INIT_HP["CHANNELS_LAST"]:
        # Environment processing for image based observations
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)

    env.reset()

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    if INIT_HP["CHANNELS_LAST"]:
        observation_spaces = [observation_space_channels_to_first(obs) for obs in observation_spaces]

    INIT_HP["AGENT_IDS"] = [agent_id for agent_id in env.agents]

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )

    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=MUTATION_PARAMS["NO_MUT"],
        architecture=MUTATION_PARAMS["ARCH_MUT"],
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],
        parameters=MUTATION_PARAMS["PARAMS_MUT"],
        activation=MUTATION_PARAMS["ACT_MUT"],
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],
        rl_hp_selection=MUTATION_PARAMS["RL_HP_SELECTION"],
        mutation_sd=MUTATION_PARAMS["MUT_SD"],
        min_lr=MUTATION_PARAMS["MIN_LR"],
        max_lr=MUTATION_PARAMS["MAX_LR"],
        min_learn_step=MUTATION_PARAMS["MIN_LEARN_STEP"],
        max_learn_step=MUTATION_PARAMS["MAX_LEARN_STEP"],
        min_batch_size=MUTATION_PARAMS["MIN_BATCH_SIZE"],
        max_batch_size=MUTATION_PARAMS["MAX_BATCH_SIZE"],
        agent_ids=INIT_HP["AGENT_IDS"],
        arch=NET_CONFIG["arch"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
        accelerator=accelerator,
    )

    state_dims = MultiAgentAlgorithm.get_state_dim(observation_spaces)
    action_dims = MultiAgentAlgorithm.get_action_dim(action_spaces)
    total_state_dims = sum(state_dim[0] for state_dim in state_dims)
    total_action_dims = sum(action_dims)
    if use_net:
        ## Critic nets currently set-up for MATD3
        actor = [
            EvolvableMLP(
                num_inputs=state_dim[0],
                num_outputs=action_dim,
                hidden_size=[64, 64],
                mlp_activation="ReLU",
                mlp_output_activation="Sigmoid",
                device=device,
            )
            for state_dim, action_dim in zip(state_dims, action_dims)
        ]
        NET_CONFIG = None
        critic = [
            [
                EvolvableMLP(
                    num_inputs=total_state_dims + total_action_dims,
                    num_outputs=1,
                    device=device,
                    hidden_size=[64, 64],
                    mlp_activation="ReLU",
                    mlp_output_activation=None,
                )
                for _ in range(INIT_HP["N_AGENTS"])
            ]
            for _ in range(2)
        ]
    else:
        actor = None
        critic = None

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=observation_spaces,
        action_space=action_spaces,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        actor_network=actor,
        critic_network=critic,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=INIT_HP["NUM_ENVS"],
        device=device,
        accelerator=accelerator,
        torch_compiler=INIT_HP["TORCH_COMPILE"],
    )

    train_multi_agent(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        sum_scores=True,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
        accelerator=accelerator,
    )

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with open("configs/training/maddpg.yaml") as file:
        config = yaml.safe_load(file)
    INIT_HP = config["INIT_HP"]
    MUTATION_PARAMS = config["MUTATION_PARAMS"]
    NET_CONFIG = config["NET_CONFIG"]
    DISTRIBUTED_TRAINING = config["DISTRIBUTED_TRAINING"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, DISTRIBUTED_TRAINING, use_net=False)
