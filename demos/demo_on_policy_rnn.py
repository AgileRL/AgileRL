import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

# pip install minigrid
# If you get an error like "cannot import name 'MINIGridEnv'", try:
# pip install gymnasium[classic-control,box2d,mujoco]
# pip install 'minigrid<=2.2.0'
# pip install gymnasium[minigrid]
try:
    import minigrid  # noqa
except ImportError:
    print("-" * 10)
    print("Warning: MiniGrid not installed.")
    print("Please install MiniGrid to run this demo:")
    print("pip install minigrid")
    print(
        "If you get an error like \"cannot import name 'MINIGridEnv'\", follow the install instructions printed above."
    )
    print("-" * 10)
    import sys

    sys.exit(1)


from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


if __name__ == "__main__":
    print("===== AgileRL Recurrent On-policy Demo =====")
    print("Using MiniGrid-Memory-S7-v0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        # Network configuration for the agent's networks (encoder, head)
        # For RNNs, encoder_config is handled internally by setting recurrent=True
        # And hidden_state_size in INIT_HP. Head config can be added if needed.
        # e.g., "head_config": {"hidden_size": [64], "activation": "ReLU"}
    }

    INIT_HP = {
        "ENV_NAME": "MiniGrid-Memory-S7-v0",
        "POP_SIZE": 4,  # Population size - reduced for faster demo
        "RECURRENT": True,  # Use recurrent networks
        "HIDDEN_STATE_SIZE": 128,  # Hidden state size for LSTM
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-4,  # Learning rate - often lower for RNNs
        "LEARN_STEP": 128,  # Learning frequency - shorter sequences can be better
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
        "ACTION_STD_INIT": 0.6,  # Initial action standard deviation (for continuous actions)
        "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient
        "VF_COEF": 0.5,  # Value function coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
        "TARGET_KL": None,  # Target KL divergence threshold
        "UPDATE_EPOCHS": 4,  # Number of policy update epochs
        "USE_ROLLOUT_BUFFER": True,  # Necessary for clean RNN implementation
        # MiniGrid observation spaces are usually dictionaries containing 'image'
        # which defaults to channels last. CHANNELS_LAST=True is assumed by PPO
        # when dealing with dictionary spaces containing image keys.
        "CHANNELS_LAST": True,
    }

    # Since make_vect_envs doesn't easily handle MiniGrid's specific wrappers,
    # we'll use a single environment for simplicity in this demo.
    # For multi-environment training, custom wrapper setup would be needed.
    num_envs = 1
    try:
        env = gym.make(INIT_HP["ENV_NAME"])
        env.reset()
    except Exception as e:
        print(f"Failed to create environment {INIT_HP['ENV_NAME']}")
        print(e)
        print("Ensure you have minigrid installed: pip install minigrid")
        print(
            "If you get an error like \"cannot import name 'MINIGridEnv'\", follow the install instructions printed above."
        )
        raise

    observation_space = env.observation_space
    try:
        action_space = env.action_space
        INIT_HP["DISCRETE_ACTIONS"] = isinstance(action_space, gym.spaces.Discrete)
    except Exception:
        INIT_HP["DISCRETE_ACTIONS"] = True

    # Define algo_kwargs dictionary for additional PPO parameters
    algo_kwargs = {
        "use_rollout_buffer": INIT_HP["USE_ROLLOUT_BUFFER"],
        "recurrent": INIT_HP["RECURRENT"],
        "hidden_state_size": INIT_HP["HIDDEN_STATE_SIZE"],
    }

    # Create population
    pop = create_population(
        algo="PPO",  # Algorithm
        observation_space=observation_space,  # Observation space
        action_space=action_space,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of environments (1 for this demo)
        device=device,
        algo_kwargs=algo_kwargs,  # Pass additional PPO args here
    )

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo="PPO",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size", "learn_step"],  # RL HPs to choose from
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 500000  # Max steps - May need more for MiniGrid
    evo_steps = 20000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - run until done
    eval_loop = 3  # Number of evaluation episodes

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps // num_envs, unit="step")

    # Keep track of episode scores for printing
    episode_rewards = {agent.index: [] for agent in pop}
    agent_episode_scores = {agent.index: [] for agent in pop}

    for agent in pop:
        state, info = env.reset(seed=42)
        score = 0
        steps = 0
        # Initialize hidden state for the single environment
        # Note: PPO's collect_rollouts handles hidden state internally for its buffer

    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            agent.actor.eval()  # Set to evaluation mode for rollout collection
            experiences_collected = 0
            while experiences_collected < evo_steps:
                # Collect rollouts using the agent's method
                agent.collect_rollouts(env, n_steps=agent.learn_step)
                experiences_collected += agent.learn_step * num_envs
                total_steps += agent.learn_step * num_envs
                steps += (
                    agent.learn_step * num_envs
                )  # Add to local step counter for the agent

                # Learn from the collected rollouts in the buffer
                agent.actor.train()  # Set back to train mode for learning
                agent.learn()
                agent.actor.eval()  # Set back to eval mode

            # Update agent steps
            agent.steps[-1] += steps
            pbar.update(steps // len(pop))  # Update progress bar

            # Store the scores collected during rollouts by the buffer for logging
            # (RolloutBuffer doesn't directly track episode scores, so this part is removed)
            # We rely on the separate testing loop for proper evaluation scores.
            pop_episode_scores.append([])  # Placeholder

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

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        # print(f"Scores: {mean_scores}") # Removed as rollout buffer doesn't track scores easily
        print(f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}")
        print(
            f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop if len(agent.fitness) > 0]}"
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()
