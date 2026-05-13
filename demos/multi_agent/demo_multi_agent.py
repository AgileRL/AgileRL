import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tensordict import TensorDictBase

from agilerl.algorithms import MADDPG
from agilerl.components.data import MultiAgentTransition
from agilerl.components.replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    init_loggers,
    tournament_selection_and_mutation,
)
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo =====")

    # Define the network configuration
    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [32, 32],  # Actor hidden size
        },
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "BATCH_SIZE": 32,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 100,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
        "POP_SIZE": 4,  # Population size
    }

    num_envs = 8

    # Define the simple speaker listener environment as a parallel environment
    env = AsyncPettingZooVecEnv(
        [
            lambda: simple_speaker_listener_v4.parallel_env(continuous_actions=True)
            for _ in range(num_envs)
        ],
    )
    env.reset()

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop: list[MADDPG] = create_population(
        "MADDPG",
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_steps = 1_000_000  # Max steps
    learning_delay = 0  # Steps before starting learning

    evo_steps = 10_000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    pbar = default_progress_bar(max_steps)

    # Initialize loggers and population wrapper
    loggers = init_loggers(
        algo="MADDPG",
        env_name="simple_speaker_listener_v4",
        pbar=pbar,
        verbose=True,
    )

    population = Population(
        agents=pop,
        loggers=loggers,
    )

    # Pre-training mutation
    population.update(mutations.mutation(population.agents, pre_training_mut=True))

    # TRAINING LOOP
    print("Training...")
    while population.all_below(max_steps):
        for agent in population.agents:  # Loop through population
            agent.set_training_mode(True)
            agent.init_evo_step()

            obs, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            for idx_step in range(evo_steps // num_envs):
                # Get next action from agent
                action, raw_action = agent.get_action(obs=obs, infos=info)

                # Act in environment
                next_obs, reward, termination, truncation, info = env.step(action)

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                steps += num_envs

                # Save experiences to replay buffer
                transition: TensorDictBase = MultiAgentTransition(
                    obs=obs,
                    action=raw_action,
                    reward=reward,
                    next_obs=next_obs,
                    done=termination,
                )
                transition = transition.to_tensordict()
                transition.batch_size = [num_envs]
                memory.add(transition)

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        agent.learn(
                            experiences
                        )  # Learn according to agent's RL algorithm

                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                obs = next_obs

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(
                    zip(term_array, trunc_array, strict=False),
                ):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)
                agent.reset_action_noise(reset_noise_indices)

            agent.add_scores(completed_episode_scores)
            agent.finalize_evo_step(steps)
            pbar.update(evo_steps // population.size)

        population.increment_evo_step()

        # Evaluate population
        for agent in population.agents:
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
            )

        population.report_metrics(clear=True)

        # Tournament selection and population mutation
        population.update(
            tournament_selection_and_mutation(
                population=population.agents,
                tournament=tournament,
                mutation=mutations,
                env_name="simple_speaker_listener_v4",
                algo="MADDPG",
            ),
        )

    population.finish()
    pbar.close()
    env.close()
