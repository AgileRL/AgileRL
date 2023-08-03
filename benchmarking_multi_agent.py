import torch
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.utils.utils import makeVectEnvs, initialPopulation, printHyperparams
from agilerl.training.train_multi_agent import train_multi_agent
from pettingzoo.mpe import simple_v3

# def make_env(env_name, env_params):
#     return env_name.parallel_env(env_params['max_cycles'], env_params['continuous_actions'])


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('============ AgileRL ============')
    print('Multi-agent benchmarking')
    print(f'DEVICE: {device}')

    env = simple_v3.parallel_env(max_cycles=25, continuous_actions=True)
    env.reset()

    # Configure the maddpg input arguments
    INIT_HP['N_AGENTS'] = env.num_agents
    action_dim = [env.action_space(agent).shape[0] for agent in env.agents] # [action_agent_1, action_agent_2, ..., action_agent_n]
    state_dim = [env.observation_space(agent).shape for agent in env.agents] # [state_agent_1, state_agent_2, ..., state_agent_n]
    INIT_HP['AGENT_IDS'] = [agent_id for agent_id in env.agents]
    INIT_HP['MAX_ACTION'] = [env.action_space(agent).high for agent in env.agents]
    INIT_HP['MIN_ACTION'] = [env.action_space(agent).low for agent in env.agents]
   
    one_hot = False
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(INIT_HP['MEMORY_SIZE'], 
                                    field_names=field_names, 
                                    agent_ids=INIT_HP['AGENT_IDS'],
                                    device=device)
    
    tournament = TournamentSelection(INIT_HP['TOURN_SIZE'],
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
                          agent_ids=INIT_HP['AGENT_IDS'],
                          arch=NET_CONFIG['arch'],
                          rand_seed=MUTATION_PARAMS['RAND_SEED'],
                          device=device)

    agent_pop = initialPopulation(INIT_HP['ALGO'],
                                  state_dim,
                                  action_dim,
                                  one_hot,
                                  NET_CONFIG,
                                  INIT_HP,
                                  INIT_HP['POP_SIZE'],
                                  device=device)

    trained_pop, pop_fitnesses = train_multi_agent(env,
                                            INIT_HP['ENV_NAME'],
                                            INIT_HP['ALGO'],
                                            agent_pop,
                                            memory=memory,
                                            init_hp=INIT_HP,
                                            mut_p=MUTATION_PARAMS,
                                            net_config=NET_CONFIG,
                                            swap_channels=INIT_HP['CHANNELS_LAST'],
                                            n_episodes=INIT_HP['EPISODES'],
                                            evo_epochs=INIT_HP['EVO_EPOCHS'],
                                            evo_loop=1,
                                            target=INIT_HP['TARGET_SCORE'],
                                            tournament=tournament,
                                            mutation=mutations,
                                            wb=INIT_HP['WANDB'])

    printHyperparams(trained_pop)
    # plotPopulationScore(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == '__main__':
    INIT_HP = {
        'ENV_NAME': 'simple_v3',   # Gym environment name
        'ENV_PARAMS' : {
            'max_cycles': 25,
            'continuous_actions' : False
        },
        'ALGO': 'MADDPG',                  # Algorithm
        'DOUBLE': False,                 # Use double Q-learning
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False,
        'BATCH_SIZE': 256,             # Batch size
        'LR': 0.005,               # Learning rate
        'EPISODES': 20_000,             # Max no. episodes
        'TARGET_SCORE': 100,            # Early training stop at avg score of last 100 episodes
        'GAMMA': 0.95,                  # Discount factor
        'MEMORY_SIZE': 1_000_000,       # Max memory buffer size
        'LEARN_STEP': 100,              # Learning frequency
        'TAU': 0.01,                    # For soft update of target parameters
        'TOURN_SIZE': 2,                # Tournament size
        'ELITISM': True,                # Elitism in tournament selection
        'POP_SIZE': 6,                  # Population size
        'EVO_EPOCHS': 20,               # Evolution frequency
        'POLICY_FREQ': 2,               # Policy network update frequency
        'WANDB': True                  # Log with Weights and Biases
    }

    MUTATION_PARAMS = {  # Relative probabilities
        'NO_MUT': 0.4,                              # No mutation
        'ARCH_MUT': 0,                            # Architecture mutation
        'NEW_LAYER': 0.2,                           # New layer mutation
        'PARAMS_MUT': 0,                          # Network parameters mutation
        'ACT_MUT': 0,                               # Activation layer mutation
        'RL_HP_MUT': 0.2,                           # Learning HP mutation
        # Learning HPs to choose from
        'RL_HP_SELECTION': ["lr"],
        'MUT_SD': 0.1,                              # Mutation strength
        'RAND_SEED': 1,                             # Random seed
    }

    NET_CONFIG = {
        'arch': 'mlp',      # Network architecture
        'h_size': [32, 32]    # Actor hidden size
    }

    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)
