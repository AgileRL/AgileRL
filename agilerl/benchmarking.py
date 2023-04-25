import torch

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.utils.utils import makeVectEnvs, initialPopulation, printHyperparams
from agilerl.training.train import train


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('============ AgileRL ============')
    print(f'DEVICE: {device}')

    env = makeVectEnvs(INIT_HP['ENV_NAME'], num_envs=16)
    try:
        state_dim = env.single_observation_space.n
        one_hot = True
    except BaseException:
        state_dim = env.single_observation_space.shape
        one_hot = False
    try:
        action_dim = env.single_action_space.n
    except BaseException:
        action_dim = env.single_action_space.shape[0]

    if INIT_HP['CHANNELS_LAST']:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        action_dim, INIT_HP['MEMORY_SIZE'], field_names=field_names, device=device)
    tournament = TournamentSelection(
        INIT_HP['TOURN_SIZE'],
        INIT_HP['ELITISM'],
        INIT_HP['POP_SIZE'],
        INIT_HP['EVO_EPOCHS'])
    mutations = Mutations(algo=INIT_HP['ALGO'],
                          no_mutation=MUTATION_PARAMS['NO_MUT'],
                          architecture=MUTATION_PARAMS['ARCH_MUT'],
                          new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],
                          parameters=MUTATION_PARAMS['PARAMS_MUT'],
                          activation=MUTATION_PARAMS['ACT_MUT'],
                          rl_hp=MUTATION_PARAMS['RL_HP_MUT'],
                          rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'],
                          mutation_sd=MUTATION_PARAMS['MUT_SD'],
                          arch=NET_CONFIG['arch'],
                          rand_seed=MUTATION_PARAMS['RAND_SEED'],
                          device=device)

    agent_pop = initialPopulation(
        INIT_HP['ALGO'],
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        INIT_HP['POP_SIZE'],
        device=device)

    trained_pop, pop_fitnesses = train(env,
                                       INIT_HP['ENV_NAME'],
                                       INIT_HP['ALGO'],
                                       agent_pop,
                                       memory=memory,
                                       swap_channels=INIT_HP['CHANNELS_LAST'],
                                       n_episodes=INIT_HP['EPISODES'],
                                       evo_epochs=INIT_HP['EVO_EPOCHS'],
                                       evo_loop=1,
                                       target=INIT_HP['TARGET_SCORE'],
                                       tournament=tournament,
                                       mutation=mutations,
                                       wb=INIT_HP['WANDB'],
                                       device=device)

    printHyperparams(trained_pop)
    # plotPopulationScore(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == '__main__':
    INIT_HP = {
        'ENV_NAME': 'LunarLander-v2',   # Gym environment name
        'ALGO': 'DQN',                  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False,
        'BATCH_SIZE': 256,              # Batch size
        'LR': 1e-3,                     # Learning rate
        'EPISODES': 2000,               # Max no. episodes
        'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.99,                  # Discount factor
        'MEMORY_SIZE': 10000,           # Max memory buffer size
        'LEARN_STEP': 1,                # Learning frequency
        'TAU': 1e-3,                    # For soft update of target parameters
        'TOURN_SIZE': 2,                # Tournament size
        'ELITISM': True,                # Elitism in tournament selection
        'POP_SIZE': 6,                  # Population size
        'EVO_EPOCHS': 20,               # Evolution frequency
        'POLICY_FREQ': 2,               # Policy network update frequency
        'WANDB': True                   # Log with Weights and Biases
    }

    MUTATION_PARAMS = {  # Relative probabilities
        'NO_MUT': 0.4,                              # No mutation
        'ARCH_MUT': 0.2,                            # Architecture mutation
        'NEW_LAYER': 0.2,                           # New layer mutation
        'PARAMS_MUT': 0.2,                          # Network parameters mutation
        'ACT_MUT': 0,                               # Activation layer mutation
        'RL_HP_MUT': 0.2,                           # Learning HP mutation
        # Learning HPs to choose from
        'RL_HP_SELECTION': ['lr', 'batch_size'],
        'MUT_SD': 0.1,                              # Mutation strength
        'RAND_SEED': 1,                             # Random seed
    }

    NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'h_size': [32, 32],    # Actor hidden size
    }

    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)
