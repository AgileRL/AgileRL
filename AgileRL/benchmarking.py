import torch
import gymnasium as gym

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.utils import makeVectEnvs, initialPopulation, printHyperparams, plotPopulationScore
from agilerl.training.train import train

def main(INIT_HP, MUTATION_PARAMS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('============ AgileRL ============')
    print(f'DEVICE: {device}')

    env = makeVectEnvs(env_name=INIT_HP['ENV_NAME'], num_envs=16)
    num_states = env.single_observation_space.shape[0]
    try:
        num_actions = env.single_action_space.n
    except:
        num_actions = env.single_action_space.shape[0]

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(num_actions, INIT_HP['MEMORY_SIZE'], field_names=field_names, device=device)
    tournament = TournamentSelection(INIT_HP['TOURN_SIZE'], INIT_HP['ELITISM'], INIT_HP['POP_SIZE'], INIT_HP['EVO_EPOCHS'])
    mutations = Mutations(algo=INIT_HP['ALGO'],
                        no_mutation=MUTATION_PARAMS['NO_MUT'], 
                        architecture=MUTATION_PARAMS['ARCH_MUT'], 
                        new_layer_prob=MUTATION_PARAMS['NEW_LAYER'], 
                        parameters=MUTATION_PARAMS['PARAMS_MUT'], 
                        activation=MUTATION_PARAMS['ACT_MUT'], 
                        rl_hp=MUTATION_PARAMS['RL_HP_MUT'], 
                        rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'], 
                        mutation_sd=MUTATION_PARAMS['MUT_SD'], 
                        rand_seed=MUTATION_PARAMS['RAND_SEED'],
                        device=device)

    agent_pop = initialPopulation(INIT_HP['ALGO'], num_states, num_actions, INIT_HP, INIT_HP['POP_SIZE'], device=device)

    trained_pop, pop_fitnesses = train(env,
        INIT_HP['ENV_NAME'],
        INIT_HP['ALGO'],
        agent_pop,
        memory=memory,
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
        # 'ENV_NAME': 'LunarLander-v2',   # Gym environment name
        # 'ALGO': 'DQN',                  # Algorithm
        'ENV_NAME': 'LunarLanderContinuous-v2',   # Gym environment name
        'ALGO': 'DDPG',                  # Algorithm
        'HIDDEN_SIZE': [64,64],         # Actor network hidden size
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
        'WANDB': False                   # Log with Weights and Biases
    }

    MUTATION_PARAMS = {  # Relative probabilities
        'NO_MUT': 0.4,                              # No mutation
        'ARCH_MUT': 0.2,                            # Architecture mutation
        'NEW_LAYER': 0.2,                           # New layer mutation
        'PARAMS_MUT': 0.2,                          # Network parameters mutation
        'ACT_MUT': 0,                               # Activation layer mutation
        'RL_HP_MUT': 0.2,                           # Learning HP mutation
        'RL_HP_SELECTION': ['lr', 'batch_size'],    # Learning HPs to choose from
        'MUT_SD': 0.1,                              # Mutation strength
        'RAND_SEED': 1,                             # Random seed
    }

    main(INIT_HP, MUTATION_PARAMS)