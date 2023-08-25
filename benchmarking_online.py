import torch
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.utils.utils import makeVectEnvs, initialPopulation, printHyperparams
from agilerl.training.train import train
import yaml


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('============ AgileRL ============')
    print(f'DEVICE: {device}')

    env = makeVectEnvs(INIT_HP['ENV_NAME'], num_envs=16)

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

    if INIT_HP['CHANNELS_LAST']:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    if INIT_HP['ALGO'] == 'TD3':
        max_action = float(env.single_action_space.high[0])
        INIT_HP["MAX_ACTION"] = max_action

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
                                       INIT_HP=INIT_HP,
                                       MUT_P=MUTATION_PARAMS,
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
    with open('configs/training/dqn.yaml', 'r') as file:
        dqn_config = yaml.safe_load(file)
    INIT_HP = dqn_config['INIT_HP']
    MUTATION_PARAMS = dqn_config['MUTATION_PARAMS']
    NET_CONFIG = dqn_config['NET_CONFIG']
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)

   
