import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from tqdm import trange

from agilerl.components.data import Transition
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import create_population, make_vect_envs
from agilerl.wrappers.make_evolvable import MakeEvolvable


class MLPActor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())  # Activation function

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())  # Activation function

        # Add output layer with a sigmoid activation
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("===== AgileRL Off-policy Custom Network Demo =====")

    # Device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    INIT_HP = {
        "DOUBLE": True,  # Use double Q-learning
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 1e-3,  # For soft update of target network parameters
        "CHANNELS_LAST": False,  # Swap image channels dimension last to first [H, W, C] -> [C, H, W]
        "POP_SIZE": 4,  # Population size
    }

    num_envs = 16
    env = make_vect_envs("LunarLander-v3", num_envs=num_envs)  # Create environment

    # Instantiate mlp and then make it evolvable
    observation_space = env.single_observation_space
    action_space: spaces.Discrete = env.single_action_space
    mlp = MLPActor(observation_space.shape[0], [32, 32], action_space.n)
    evolvable_mlp = MakeEvolvable(
        mlp,
        input_tensor=torch.ones(
            observation_space.shape[0]
        ),  # Example input tensor to the network
        device=device,
    )

    # Create a population of DQN agents
    pop = create_population(
        algo="DQN",  # Algorithm
        observation_space=env.observation_space,  # Observation space
        action_space=env.action_space,  # Action space
        net_config=None,  # Network configuration set as None
        actor_network=evolvable_mlp,  # Custom evolvable actor
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized envs
        device=device,
    )

    # Create the replay buffer
    memory = ReplayBuffer(INIT_HP["MEMORY_SIZE"], device=device)

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo="DQN",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 200000  # Max steps
    learning_delay = 1000  # Steps before starting learning

    # Exploration params
    eps_start = 1.0  # Max exploration
    eps_end = 0.1  # Min exploration
    eps_decay = 0.995  # Decay per episode
    epsilon = eps_start

    evo_steps = 10000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores, losses = [], []
            steps = 0
            epsilon = eps_start

            for idx_step in range(evo_steps // num_envs):
                if INIT_HP["CHANNELS_LAST"]:
                    state = obs_channels_to_first(state)

                action = agent.get_action(state, epsilon)  # Get next action from agent
                epsilon = max(
                    eps_end, epsilon * eps_decay
                )  # Decay epsilon for exploration

                # Act in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                scores += np.array(reward)
                steps += num_envs
                total_steps += num_envs

                # Collect scores for completed episodes
                for idx, (d, t) in enumerate(zip(terminated, truncated)):
                    if d or t:
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0

                # Save experience to replay buffer
                next_state = (
                    obs_channels_to_first(next_state)
                    if INIT_HP["CHANNELS_LAST"]
                    else next_state
                )
                transition = Transition(
                    obs=state,
                    action=action,
                    reward=reward,
                    next_obs=next_state,
                    done=terminated,
                    batch_size=[num_envs],
                )
                memory.add(transition.to_tensordict())

                # Learn according to learning frequency
                if memory.counter > learning_delay and len(memory) >= agent.batch_size:
                    for _ in range(num_envs // agent.learn_step):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                state = next_state

            pbar.update(evo_steps // len(pop))
            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Reset epsilon start to latest decayed value for next round of population training
        eps_start = epsilon

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()
