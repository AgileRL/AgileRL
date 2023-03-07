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

    agent_pop = initialPopulation(num_states, num_actions, INIT_HP, INIT_HP['POP_SIZE'], device=device)

    trained_pop, pop_fitnesses = train(env,
        INIT_HP['ENV_NAME'],
        INIT_HP['ALGO'],
        agent_pop,
        memory=memory,
        n_episodes=INIT_HP['EPISODES'],
        max_steps=INIT_HP['MAX_STEPS'],
        evo_epochs=INIT_HP['EVO_EPOCHS'],
        evo_loop=1,
        target=INIT_HP['TARGET_SCORE'],
        tournament=tournament,
        mutation=mutations,
        device=device)

    printHyperparams(trained_pop)
    # plotPopulationScore(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == '__main__':
    env_names = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
    target_scores = [450., -150., -120.]
    max_steps = [500, 200, 500]

    for env_name, target_score, max_step in zip(env_names, target_scores, max_steps):
        INIT_HP = {
            'ENV_NAME': env_name,
            'ALGO': 'DQN',
            'MAX_STEPS': max_step,
            'HIDDEN_SIZE': [64,64],
            'BATCH_SIZE': 256,
            'LR': 1e-3,
            'EPISODES': 2000,
            'TARGET_SCORE': target_score,     # early training stop at avg score of last 100 episodes
            'GAMMA': 0.99,            # discount factor
            'MEMORY_SIZE': 10000,     # max memory buffer size
            'LEARN_STEP': 1,          # how often to learn
            'TAU': 1e-3,              # for soft update of target parameters
            'TOURN_SIZE': 2,
            'ELITISM': True,
            'POP_SIZE': 6,
            'EVO_EPOCHS': 20
        }

        MUTATION_PARAMS = {
            'NO_MUT': 0.4, #0.2,
            'ARCH_MUT': 0.2, #0.2,
            'NEW_LAYER': 0.2,
            'PARAMS_MUT': 0.2, # 0.2,
            'ACT_MUT': 0, #0.2,
            'RL_HP_MUT': 0.2,
            'RL_HP_SELECTION': ['lr', 'batch_size'],
            'MUT_SD': 0.1,
            'RAND_SEED': 1,
        }

        for i in range(5):
            main(INIT_HP, MUTATION_PARAMS)