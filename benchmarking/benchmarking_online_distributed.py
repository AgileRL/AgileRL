from accelerate import Accelerator

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
    memory = ReplayBuffer(INIT_HP["MEMORY_SIZE"], field_names=field_names)
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
        arch=NET_CONFIG["arch"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
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
        num_envs=INIT_HP["NUM_ENVS"],
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
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        learning_delay=INIT_HP["LEARNING_DELAY"],
        eps_start=INIT_HP["EPS_START"] if "EPS_START" in INIT_HP else 1.0,
        eps_end=INIT_HP["EPS_END"] if "EPS_END" in INIT_HP else 0.01,
        eps_decay=INIT_HP["EPS_DECAY"] if "EPS_DECAY" in INIT_HP else 0.999,
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
        "ENV_NAME": "LunarLander-v2",  # Gym environment name
        "ALGO": "DQN",  # Algorithm
        "DOUBLE": True,  # Use double Q-learning
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "NUM_ENVS": 16,
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "MAX_STEPS": 1_000_000,  # Max no. steps
        "TARGET_SCORE": 200.0,  # Early training stop at avg score of last 100 episodes
        "GAMMA": 0.99,  # Discount factor
        "MEMORY_SIZE": 10000,  # Max memory buffer size
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 1e-3,  # For soft update of target parameters
        "TOURN_SIZE": 2,  # Tournament size
        "ELITISM": True,  # Elitism in tournament selection
        "POP_SIZE": 6,  # Population size
        "EVO_STEPS": 10_000,  # Evolution frequency
        "EVAL_STEPS": None,  # Evaluation steps
        "EVAL_LOOP": 1,  # Evaluation episodes
        "LEARNING_DELAY": 1000,  # Steps before learning
        "WANDB": False,  # Log with Weights and Biases
    }

    MUTATION_PARAMS = {  # Relative probabilities
        "NO_MUT": 0.4,  # No mutation
        "ARCH_MUT": 0.2,  # Architecture mutation
        "NEW_LAYER": 0.2,  # New layer mutation
        "PARAMS_MUT": 0.2,  # Network parameters mutation
        "ACT_MUT": 0,  # Activation layer mutation
        "RL_HP_MUT": 0.2,  # Learning HP mutation
        # Learning HPs to choose from
        "RL_HP_SELECTION": ["lr", "batch_size", "learn_step"],
        "MUT_SD": 0.1,  # Mutation strength
        "RAND_SEED": 1,  # Random seed
    }

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [32, 32],  # Actor hidden size
    }

    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)
