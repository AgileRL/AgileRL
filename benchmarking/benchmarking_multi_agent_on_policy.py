import importlib

import supersuit as ss
import torch
import yaml
from accelerate import Accelerator
from pettingzoo.utils import env_logger

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_multi_agent_on_policy import train_multi_agent_on_policy
from agilerl.utils.utils import (
    create_population,
    make_multi_agent_vect_envs,
    observation_space_channels_to_first,
)

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, DISTRIBUTED_TRAINING):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_logger.EnvLogger.suppress_output()

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

    def create_env(**kwargs):
        env = importlib.import_module(f"{INIT_HP['ENV_NAME']}").parallel_env(**kwargs)

        if INIT_HP["CHANNELS_LAST"]:
            # Environment processing for image based observations
            env = ss.frame_skip_v0(env, 4)
            env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
            env = ss.color_reduction_v0(env, mode="B")
            env = ss.resize_v1(env, x_size=84, y_size=84)
            env = ss.frame_stack_v1(env, 4)

        return env

    env_kwargs = dict(max_cycles=25, continuous_actions=False)
    env = make_multi_agent_vect_envs(
        create_env, num_envs=INIT_HP["NUM_ENVS"], **env_kwargs
    )

    env.reset(seed=42)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    if INIT_HP["CHANNELS_LAST"]:
        observation_spaces = [
            observation_space_channels_to_first(obs) for obs in observation_spaces
        ]

    INIT_HP["AGENT_IDS"] = [agent_id for agent_id in env.agents]

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
        accelerator=accelerator,
    )

    hp_config = HyperparameterConfig(
        lr=RLParameter(min=MUTATION_PARAMS["MIN_LR"], max=MUTATION_PARAMS["MAX_LR"]),
        batch_size=RLParameter(
            min=MUTATION_PARAMS["MIN_BATCH_SIZE"],
            max=MUTATION_PARAMS["MAX_BATCH_SIZE"],
            dtype=int,
        ),
        learn_step=RLParameter(
            min=MUTATION_PARAMS["MIN_LEARN_STEP"],
            max=MUTATION_PARAMS["MAX_LEARN_STEP"],
            dtype=int,
            grow_factor=1.5,
            shrink_factor=0.75,
        ),
        ent_coef=RLParameter(
            min=MUTATION_PARAMS["MIN_ENT_COEF"],
            max=MUTATION_PARAMS["MAX_ENT_COEF"],
        ),
    )

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=observation_spaces,
        action_space=action_spaces,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        hp_config=hp_config,
        actor_network=None,
        critic_network=None,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=INIT_HP["NUM_ENVS"],
        device=device,
        accelerator=accelerator,
        torch_compiler=INIT_HP["TORCH_COMPILE"],
    )

    try:
        train_multi_agent_on_policy(
            env,
            INIT_HP["ENV_NAME"],
            INIT_HP["ALGO"],
            agent_pop,
            sum_scores=True,
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
            accelerator=accelerator,
        )
    except Exception as e:
        env.close()
        raise e

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    config = "configs/training/multi_agent/ippo.yaml"
    with open(config) as file:
        config = yaml.safe_load(file)
    INIT_HP = config["INIT_HP"]
    MUTATION_PARAMS = config["MUTATION_PARAMS"]
    NET_CONFIG = config["NET_CONFIG"]
    DISTRIBUTED_TRAINING = config["DISTRIBUTED_TRAINING"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, DISTRIBUTED_TRAINING)
