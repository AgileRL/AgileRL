from agilerl.utils import makeVectEnvs, initialPopulation
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
import gymnasium as gym
import numpy as np
import torch

if __name__ == "__main__":

    INIT_HP = {
                'HIDDEN_SIZE': [64,64], # Actor network hidden size
                'BATCH_SIZE': 128,      # Batch size
                'LR': 1e-3,             # Learning rate
                'GAMMA': 0.99,          # Discount factor
                'LEARN_STEP': 1,        # Learning frequency
                'TAU': 1e-3             # For soft update of target network parameters
              }

    pop = initialPopulation(algo='DQN',             # Algorithm
                            num_states=8,           # State dimension
                            num_actions=4,          # Action dimension
                            INIT_HP=INIT_HP,        # Initial hyperparameters
                            population_size=6,      # Population size
                            device=torch.device("cuda"))

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(n_actions=4,              # Number of agent actions
                          memory_size=10000,        # Max replay buffer size
                          field_names=field_names,  # Field names to store in memory
                          device=torch.device("cuda"))

    tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                     elitism=True,      # Elitism in tournament selection
                                     population_size=6, # Population size
                                     evo_step=1)        # Evaluate using last N fitness scores

    mutations = Mutations(algo='DQN',                           # Algorithm
                          no_mutation=0.4,                      # No mutation
                          architecture=0.2,                     # Architecture mutation
                          new_layer_prob=0.2,                   # New layer mutation
                          parameters=0.2,                       # Network parameters mutation
                          activation=0,                         # Activation layer mutation
                          rl_hp=0.2,                            # Learning HP mutation
                          rl_hp_selection=['lr', 'batch_size'], # Learning HPs to choose from
                          mutation_sd=0.1,                      # Mutation strength
                          rand_seed=1,                          # Random seed
                          device=torch.device("cuda"))
    
    max_episodes = 1000 # Max training episodes
    max_steps = 500     # Max steps per episode
    
    # Exploration params
    eps_start = 1.0     # Max exploration
    eps_end = 0.1       # Min exploration
    eps_decay = 0.995   # Decay per episode
    epsilon = eps_start

    evo_epochs = 5      # Evolution frequency
    evo_loop = 1        # Number of evaluation episodes

    env = makeVectEnvs('LunarLander-v2', num_envs=16)   # Create environment

    # TRAINING LOOP
    for idx_epi in range(max_episodes):
        for agent in pop:   # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            score = 0
            for idx_step in range(max_steps):
                action = agent.getAction(state, epsilon)    # Get next action from agent
                next_state, reward, done, _, _ = env.step(action)   # Act in environment
                
                # Save experience to replay buffer
                memory.save2memoryVectEnvs(state, action, reward, next_state, done)

                # Learn according to learning frequency
                if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
                    experiences = memory.sample(agent.batch_size) # Sample replay buffer
                    agent.learn(experiences)    # Learn according to agent's RL algorithm
                
                state = next_state
                score += reward

        epsilon = max(eps_end, epsilon*eps_decay) # Update epsilon for exploration

        # Now evolve population if necessary
        if (idx_epi+1) % evo_epochs == 0:
            
            # Evaluate population
            fitnesses = [agent.test(env, max_steps=max_steps, loop=evo_loop) for agent in pop]

            print(f'Episode {idx_epi+1}/{max_episodes}')
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)