import sys

import torch
import yaml

sys.path.append("../")

from agilerl.training.train_off_policy import train_off_policy
from agilerl.components.replay_buffer import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============ AgileRL ============")
    print(f"DEVICE: {device}")

    env = makeVectEnvs(INIT_HP["ENV_NAME"], num_envs=INIT_HP["NUM_ENVS"])

    try:
        state_dim = env.single_observation_space.n
        one_hot = True
    except Exception:
        state_dim = env.single_observation_space.shape
        one_hot = False
    try:
        action_dim = env.single_action_space.n
    except Exception:
        action_dim = env.single_action_space.shape[0]

    if INIT_HP["CHANNELS_LAST"]:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    field_names = ["state", "action", "reward", "next_state", "done"]
    n_step_memory = None
    per = INIT_HP["PER"]
    n_step = True if INIT_HP["N_STEP"] > 1 else False
    if per:
        memory = PrioritizedReplayBuffer(
            action_dim,
            memory_size=INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            num_envs=INIT_HP["NUM_ENVS"],
            alpha=INIT_HP["ALPHA"],
            gamma=INIT_HP["GAMMA"],
            device=device,
        )
        if n_step:
            n_step_memory = MultiStepReplayBuffer(
                action_dim,
                memory_size=INIT_HP["MEMORY_SIZE"],
                field_names=field_names,
                num_envs=INIT_HP["NUM_ENVS"],
                n_step=INIT_HP["N_STEP"],
                gamma=INIT_HP["GAMMA"],
                device=device,
            )
    elif n_step:
        memory = MultiStepReplayBuffer(
            action_dim,
            memory_size=INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
            num_envs=INIT_HP["NUM_ENVS"],
            n_step=INIT_HP["N_STEP"],
            gamma=INIT_HP["GAMMA"],
            device=device,
        )
    else:
        memory = ReplayBuffer(
            action_dim,
            memory_size=INIT_HP["MEMORY_SIZE"],
            field_names=field_names,
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
        arch=NET_CONFIG["arch"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
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
    )

    trained_pop, pop_fitnesses = train_off_policy(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        n_step_memory=n_step_memory,
        n_step=n_step,
        per=per,
        noisy=False,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        n_episodes=INIT_HP["EPISODES"],
        evo_epochs=INIT_HP["EVO_EPOCHS"],
        evo_loop=INIT_HP["EVO_LOOP"],
        target=INIT_HP["TARGET_SCORE"],
        tournament=None,
        mutation=None,
        wb=INIT_HP["WANDB"],
    )

    printHyperparams(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    with open("../configs/training/dqn_rainbow.yaml") as file:
        rainbow_dqn_config = yaml.safe_load(file)
    INIT_HP = rainbow_dqn_config["INIT_HP"]
    MUTATION_PARAMS = rainbow_dqn_config["MUTATION_PARAMS"]
    NET_CONFIG = rainbow_dqn_config["NET_CONFIG"]

    # Run number 1 = use only the normal buffer
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)

    # Run number 2 = use per and n step
    INIT_HP["N_STEP"] = 3
    INIT_HP["PER"] = True
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)

    # Run number 3 = use just per
    INIT_HP["N_STEP"] = 1
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)

    # Run number 4 = use just n step
    INIT_HP["N_STEP"] = 3
    INIT_HP["PER"] = False
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)

    # Run normal DQN
    with open("../configs/training/dqn.yaml") as file:
        dqn_config = yaml.safe_load(file)
    INIT_HP = dqn_config["INIT_HP"]
    INIT_HP["PER"] = False
    INIT_HP["N_STEP"] = 1
    INIT_HP["NUM_ENVS"] = 16
    INIT_HP["EVO_LOOP"] = 3
    MUTATION_PARAMS = dqn_config["MUTATION_PARAMS"]
    NET_CONFIG = dqn_config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)
