import numpy as np
import os
from tqdm import trange
import torch
import wandb
from datetime import datetime
from torch.utils.data import DataLoader
from agilerl.components.replay_data import ReplayDataset
from agilerl.components.sampler import Sampler
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.utils.utils import makeVectEnvs, initialPopulation, printHyperparams
from agilerl.training.train_multi_agent import train_multi_agent
from agilerl.components.replay_buffer import ReplayBuffer
from pettingzoo.mpe import simple_v3


def train_ddpg_multi_agent(env, env_name, algo, pop, memory, init_hp, mut_p, net_config, swap_channels=False, n_episodes=2000, 
          max_steps=500, evo_epochs=5, evo_loop=1, eps_start=1.0, eps_end=0.1, 
          eps_decay=0.995, target=200., tournament=None, mutation=None, checkpoint=None, 
          checkpoint_path=None, wb=False, accelerator=None):
    """The general online RL training function. Returns trained population of agents 
    and their fitnesses.

    :param env: The environment to train in. Can be vectorized.
    :type env: Gym-style environment
    :param env_name: Environment name
    :type env_name: str
    :param algo: RL algorithm name
    :type algo: str
    :param pop: Population of agents
    :type pop: List[object]
    :param memory: Experience Replay Buffer
    :type memory: object
    :param swap_channels: Swap image channels dimension from last to first 
    [H, W, C] -> [C, H, W], defaults to False
    :type swap_channels: bool, optional
    :param n_episodes: Maximum number of training episodes, defaults to 2000
    :type n_episodes: int, optional
    :param max_steps: Maximum number of steps in environment per episode, defaults to 
    500
    :type max_steps: int, optional
    :param evo_epochs: Evolution frequency (episodes), defaults to 5
    :type evo_epochs: int, optional
    :param evo_loop: Number of evaluation episodes, defaults to 1
    :type evo_loop: int, optional
    :param eps_start: Maximum exploration - initial epsilon value, defaults to 1.0
    :type eps_start: float, optional
    :param eps_end: Minimum exploration - final epsilon value, defaults to 0.1
    :type eps_end: float, optional
    :param eps_decay: Epsilon decay per episode, defaults to 0.995
    :type eps_decay: float, optional
    :param target: Target score for early stopping, defaults to 200.
    :type target: float, optional
    :param tournament: Tournament selection object, defaults to None
    :type tournament: object, optional
    :param mutation: Mutation object, defaults to None
    :type mutation: object, optional
    :param checkpoint: Checkpoint frequency (episodes), defaults to None
    :type checkpoint: int, optional
    :param checkpoint_path: Location to save checkpoint, defaults to None
    :type checkpoint_path: str, optional
    :param wb: Weights & Biases tracking, defaults to False
    :type wb: bool, optional
    :param accelerator: Accelerator for distributed computing, defaults to None
    :type accelerator: Hugging Face accelerate.Accelerator(), optional
    """
    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="EvoMADDPGTesting",
                    name="{}-MultiAgentEvoHPO-{}-{}".format(env_name, algo,
                                                datetime.now().strftime("%m%d%Y%H%M%S")),
                    # track hyperparameters and run metadata
                    config={
                        "algo": "Evo HPO {}".format(algo),
                        "env": env_name,
                    }
                )
            accelerator.wait_for_everyone()
        else:
            wandb.init(
                    # set the wandb project where this run will be logged
                    project="EvoMADDPGTesting",
                    name="{}-DDPGBenchmarkNoMutTournRLHP-FixedBS-{}-{}".format(env_name, algo,
                                                datetime.now().strftime("%m%d%Y%H%M%S")),
                    # track hyperparameters and run metadata
                    config={
                        "algo": "Evo HPO {}".format(algo),
                        "env": env_name,
                        "net_config": net_config,
                        "details": "DDPG benchmark with rl hp mut."
                    }
                )
            
        wandb.config.batch_size = init_hp['BATCH_SIZE']
        wandb.config.critic_lr = init_hp['LR']
        wandb.config.gamma = init_hp['GAMMA']
        wandb.config.memory_size = init_hp['MEMORY_SIZE']
        wandb.config.learn_step = init_hp['LEARN_STEP']
        wandb.config.tau = init_hp['TAU']
        wandb.config.pop_size = init_hp['TOURN_SIZE']
        wandb.config.no_mut = mut_p['NO_MUT']
        wandb.config.arch_mut = mut_p['ARCH_MUT']
        wandb.config.params_mut = mut_p['PARAMS_MUT']
        wandb.config.act_mut = mut_p['ACT_MUT']
        wandb.config.rl_hp_mut = mut_p['RL_HP_MUT']


    if accelerator is not None:
        accel_temp_models_path = 'models/{}'.format(env_name)
        if accelerator.is_main_process:
            if not os.path.exists(accel_temp_models_path):
                os.makedirs(accel_temp_models_path)

    save_path = checkpoint_path.split('.pt')[0] if checkpoint_path is not None else "{}-EvoHPO-{}-{}".format(
        env_name, algo, datetime.now().strftime("%m%d%Y%H%M%S"))
    
    if accelerator is not None:
        # Create dataloader from replay buffer
        replay_dataset = ReplayDataset(memory, pop[0].batch_size)
        replay_dataloader = DataLoader(replay_dataset, batch_size=None)
        replay_dataloader = accelerator.prepare(replay_dataloader)
        sampler = Sampler(distributed=True, 
                          dataset=replay_dataset, 
                          dataloader=replay_dataloader)
    else:
        sampler = Sampler(distributed=False, memory=memory)

    epsilon = eps_start

    if accelerator is not None:
        print(f'\nDistributed training on {accelerator.device}...')
    else:
        print('\nTraining...')

    bar_format = '{l_bar}{bar:10}| {n:4}/{total_fmt} [{elapsed:>7}<{remaining:>7}, {rate_fmt}{postfix}]'
    if accelerator is not None:
        pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True, 
                      disable=not accelerator.is_local_main_process)
    else:
        pbar = trange(n_episodes, unit="ep", bar_format=bar_format, ascii=True)

    pop_fitnesses = []
    total_steps = 0

    # RL training loop
    for idx_epi in pbar:
        if accelerator is not None:
            accelerator.wait_for_everyone()   
        for agent in pop:   # Loop through population 
            state = env.reset()[0]  # Reset environment at start of episode
            state = state['agent_0']
            agent_reward = {agent_id: 0 for agent_id in env.agents}

            while env.agents:
                total_steps += 1
                if swap_channels:
                    state = np.moveaxis(state, [3], [1])
                # Get next action from agent
                action = agent.getAction(state, epsilon)
                action = {"agent_0":action.reshape(5)}
                next_state, reward, done, _, _ = env.step(action)   # Act in environment
                next_state = next_state["agent_0"]
                reward = reward["agent_0"]
                done = done["agent_0"]

                # Save experience to replay buffer
                if swap_channels:
                    memory.save2memory(
                        state, action, reward, np.moveaxis(next_state, [3], [1]), done)
                else:
                    memory.save2memory(
                        state, action["agent_0"], reward, next_state, done)
                
                agent_reward["agent_0"] += reward 

                #Learn according to learning frequency
                if memory.counter % agent.learn_step == 0 and len(
                        memory) >= agent.batch_size:
                    # Sample replay buffer
                    experiences = memory.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

                state = next_state
            
            score = sum(agent_reward.values())
            agent.scores.append(score)

            agent.steps[-1] += total_steps

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

        # Now evolve if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=swap_channels,
                    max_steps=max_steps,
                    loop=evo_loop) for agent in pop]
            pop_fitnesses.append(fitnesses)

            mean_scores = np.mean([agent.scores[-20:] for agent in pop], axis=1)

            if wb:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        wandb.log({"global_step": total_steps*accelerator.state.num_processes,
                                "eval/mean_score": np.mean(mean_scores),
                                "eval/mean_reward": np.mean(fitnesses),
                                "eval/best_fitness": np.max(fitnesses)})
                    accelerator.wait_for_everyone()
                else:
                    wandb.log({"global_step": total_steps,
                                "eval/mean_score": np.mean(mean_scores),
                                "eval/mean_reward": np.mean(fitnesses),
                                "eval/best_fitness": np.max(fitnesses)})

            # Update step counter
            for agent in pop:
                agent.steps.append(agent.steps[-1])

            fitness = ["%.2f"%fitness for fitness in fitnesses]
            avg_fitness = ["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]
            avg_score = ["%.2f"%np.mean(agent.scores[-100:]) for agent in pop]
            agents = [agent.index for agent in pop]
            num_steps = [agent.steps[-1] for agent in pop]
            muts = [agent.mut for agent in pop]
            perf_info = f'Fitness: {fitness}, 100 fitness avgs: {avg_fitness}, 100 score avgs: {avg_score}'
            pop_info = f'Agents: {agents}, Steps: {num_steps}, Mutations: {muts}' 
            pbar_string = perf_info + ', ' + pop_info
            pbar.set_postfix_str(pbar_string)
            pbar.update(0)

            # Early stop if consistently reaches target
            if np.all(np.greater([np.mean(agent.fitness[-100:])
                      for agent in pop], target)) and idx_epi >= 100:
                if wb:
                    wandb.finish()
                return pop, pop_fitnesses

            # Tournament selection and population mutation
            if tournament and mutation is not None:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.unwrap_models()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        elite, pop = tournament.select(pop)
                        pop = mutation.mutation(pop)
                        for pop_i, model in enumerate(pop):
                            model.saveCheckpoint(f'{accel_temp_models_path}/{algo}_{pop_i}.pt')
                    accelerator.wait_for_everyone()
                    if not accelerator.is_main_process:
                        for pop_i, model in enumerate(pop):
                            model.loadCheckpoint(f'{accel_temp_models_path}/{algo}_{pop_i}.pt')
                    accelerator.wait_for_everyone()
                    for model in pop:
                        model.wrap_models()
                else:
                    elite, pop = tournament.select(pop)
                    #pop_ = mutation.mutation(pop)

        # Save model checkpoint
        if checkpoint is not None:
            if (idx_epi + 1) % checkpoint == 0:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    if not accelerator.is_main_process:
                        for i, agent in enumerate(pop):
                            agent.saveCheckpoint(f'{save_path}_{i}_{idx_epi+1}.pt')
                    accelerator.wait_for_everyone()
                else:
                    for i, agent in enumerate(pop):
                        agent.saveCheckpoint(f'{save_path}_{i}_{idx_epi+1}.pt')

    if wb:
        if accelerator is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                wandb.finish()
            accelerator.wait_for_everyone()
        else:
            wandb.finish()

    return pop, pop_fitnesses

def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('============ AgileRL ============')
    print('Multi-agent DDPG benchmarking')
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
    memory = ReplayBuffer(action_dim,
                          INIT_HP['MEMORY_SIZE'], 
                          field_names=field_names, 
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
                          arch=NET_CONFIG['arch'],
                          rand_seed=MUTATION_PARAMS['RAND_SEED'],
                          device=device)

    agent_pop = initialPopulation(INIT_HP['ALGO'],
                                  state_dim[0],
                                  action_dim[0],
                                  one_hot,
                                  NET_CONFIG,
                                  INIT_HP,
                                  INIT_HP['POP_SIZE'],
                                  device=device)

    trained_pop, pop_fitnesses = train_ddpg_multi_agent(env,
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
            'continuous_actions' : True
        },
        'ALGO': 'DDPG',                 # Algorithm
        'DOUBLE': False,                # Use double Q-learning
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False,
        'BATCH_SIZE': 256,              # Batch size
        'LR': 0.0001,                    # Learning rate
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
        'WANDB': True                # Log with Weights and Biases
    }

    MUTATION_PARAMS = {  # Relative probabilities
        'NO_MUT': 0.4,                              # No mutation
        'ARCH_MUT': 0,                            # Architecture mutation
        'NEW_LAYER': 0,                           # New layer mutation
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
