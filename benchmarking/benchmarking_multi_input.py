import copy
import sys

import torch
import yaml

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.networks.actors import StochasticActor
from agilerl.networks.value_functions import ValueFunction
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import create_population, make_vect_envs, print_hyperparams

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


sys.path.append("../racecar_gym")


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============ AgileRL ============")
    print(f"DEVICE: {device}")

    env = make_vect_envs(INIT_HP["ENV_NAME"], num_envs=INIT_HP["NUM_ENVS"])

    observation_space = env.single_observation_space
    action_space = env.single_action_space

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
        net_config = NET_CONFIG
        critic_net_config = copy.deepcopy(net_config)

        head_config = net_config.get("head_config", None)
        critic_head_config = copy.deepcopy(head_config)
        critic_head_config["output_activation"] = (
            None  # Value function has no output activation
        )
        critic_net_config["head_config"] = critic_head_config

        actor = StochasticActor(
            observation_space,
            action_space,
            log_std_init=INIT_HP["ACTION_STD_INIT"],
            device=device,
            **net_config,
        )
        critic = ValueFunction(observation_space, device=device, **critic_net_config)
        actor.filter_mutation_methods("kernel")
        critic.filter_mutation_methods("kernel")
    else:
        actor = None
        critic = None

    hp_config = HyperparameterConfig(
        lr=RLParameter(min=MUTATION_PARAMS["MIN_LR"], max=MUTATION_PARAMS["MAX_LR"]),
        batch_size=RLParameter(
            min=MUTATION_PARAMS["MIN_BATCH_SIZE"],
            max=MUTATION_PARAMS["MAX_BATCH_SIZE"],
            dtype=int,
        ),
    )

    agent_pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        hp_config=hp_config,
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
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, use_net=True)
