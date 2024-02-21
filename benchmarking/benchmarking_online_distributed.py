from accelerate import Accelerator


import sys
sys.path.append('../')
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_off_policy import train_off_policy
from agilerl.utils.utils import initialPopulation, makeVectEnvs, printHyperparams

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    accelerator = Accelerator()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("============ AgileRL Distributed ============")
    accelerator.wait_for_everyone()

    env = makeVectEnvs(INIT_HP["ENV_NAME"], num_envs=16)
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
    memory = ReplayBuffer(action_dim, INIT_HP["MEMORY_SIZE"], field_names=field_names)
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
        accelerator=accelerator,
        min_lr = 0.000000000000001,
        max_lr = 1000.0
    )

    agent_pop = initialPopulation(
        algo=INIT_HP["ALGO"],
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        accelerator=accelerator,
    )

    trained_pop, pop_fitnesses = train_off_policy(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        n_episodes=INIT_HP["EPISODES"],
        evo_epochs=INIT_HP["EVO_EPOCHS"],
        evo_loop=1,
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
        accelerator=accelerator,
    )

    printHyperparams(trained_pop)
    # plotPopulationScore(trained_pop)

    env.close()


if __name__ == "__main__":
    INIT_HP = {
        "ENV_NAME": "LunarLanderContinuous-v2",  # Gym environment name
        "ALGO": "DDPG",  # Algorithm
        #"DOUBLE": True,  # Use double Q-learning
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 128,  # Batch size
        "LR_ACTOR": 0.0001,           # Actor learning rate
        "LR_CRITIC": 0.001,       
        "EPISODES": 1000,  # Max no. episodes
        "TARGET_SCORE": 200.0,  # Early training stop at avg score of last 100 episodes
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 0.005,  # For soft update of target parameters
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
        "POP_SIZE": 2,  # Population size
        "EVO_EPOCHS": 2,  # Evolution frequency
        "POLICY_FREQ": 2,  # Policy network update frequency
        "WANDB": False,  # Log with Weights and Biases
        'MAX_ACTION': 1,
        'MIN_ACTION': -1
    }

    MUTATION_PARAMS = {  # Relative probabilities
        "NO_MUT": 0,  # No mutation
        "ARCH_MUT": 0,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0,  # Network parameters mutation
        "ACT_MUT": 0,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        # Learning HPs to choose from
        "RL_HP_SELECTION": ["lr"],
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": 42,  # Random seed
        "MIN_LR": 0.0001,  # Define max and min limits for mutating RL hyperparams
        "MAX_LR": 0.01,
        "MIN_BATCH_SIZE": 8,
        "MAX_BATCH_SIZE": 1024
    }

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [64, 64],  # Actor hidden size
    }

    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)
