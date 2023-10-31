import importlib

import supersuit as ss
import torch
import yaml
from accelerate import Accelerator

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.utils.utils import initialPopulation, printHyperparams

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, DISTRIBUTED_TRAINING):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============ AgileRL ============")

    if DISTRIBUTED_TRAINING:
        accelerator = Accelerator()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("===== Distributed Training =====")
        accelerator.wait_for_everyone()
    else:
        accelerator = None
        print(device)

    print("Multi-agent benchmarking")

    env = importlib.import_module(f"{INIT_HP['ENV_NAME']}").parallel_env(
        max_cycles=25, continuous_actions=True
    )

    if INIT_HP["CHANNELS_LAST"]:
        # Environment processing for image based observations
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)

    env.reset()

    # Configure the multi-agent algo input arguments
    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [env.action_space(agent).high for agent in env.agents]
        INIT_HP["MIN_ACTION"] = [env.action_space(agent).low for agent in env.agents]

    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    INIT_HP["N_AGENTS"] = env.num_agents
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
        INIT_HP["EVO_EPOCHS"],
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

    agent_pop = initialPopulation(
        algo=INIT_HP["ALGO"],
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        device=device,
        accelerator=accelerator,
    )

    trained_pop, pop_fitnesses = train_multi_agent(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        net_config=NET_CONFIG,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        n_episodes=INIT_HP["EPISODES"],
        evo_epochs=INIT_HP["EVO_EPOCHS"],
        evo_loop=1,
        max_steps=900,
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
        accelerator=accelerator,
    )

    printHyperparams(trained_pop)
    # plotPopulationScore(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with open("../configs/training/matd3.yaml") as file:
        config = yaml.safe_load(file)
    INIT_HP = config["INIT_HP"]
    MUTATION_PARAMS = config["MUTATION_PARAMS"]
    NET_CONFIG = config["NET_CONFIG"]
    DISTRIBUTED_TRAINING = config["DISTRIBUTED_TRAINING"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG, DISTRIBUTED_TRAINING)
