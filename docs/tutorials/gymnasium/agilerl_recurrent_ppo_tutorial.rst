.. _agilerl_recurrent_ppo_tutorial:

Partially Observable Pendulum-v1 with Recurrent PPO
======================================================

In this tutorial, we will be training and optimising the hyperparameters of a population of PPO agents
to beat a partially observable variant of the Gymnasium ``Pendulum-v1`` environment where angular velocity observations are masked.

The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with
its center of gravity right above the fixed point.

.. figure:: ../gymnasium/agilerl_ppo_pendulum.gif
  :width: 400
  :alt: agent-environment-diagram
  :align: center

  Figure 1: Completed Pendulum-v1 environment using an AgileRL PPO agent

Partially Observable Markov Decision Processes (POMDPs)
-------------------------------------------------------

The Markov assumption states that the future depends only on the current state, not on the history of how we got there. However, in
many real-world applications, agents need information about the past to predict the future effectively. For example:

    - A robot navigating a maze needs to remember which paths it has already tried, not just its current position
    - A trading agent must track price trends over time, not just the current price point
    - A self-driving car has to remember the recent trajectory of other vehicles to predict where they might go next

These scenarios are examples of Partially Observable Markov Decision Processes (POMDPs), where the agent only receives incomplete or noisy
observations of the true environment state. This partial observability makes the learning task significantly more challenging than fully
observable MDPs, since the agent needs to:

    1. Remember important information from past observations
    2. Infer hidden state information from incomplete observations
    3. Deal with uncertainty about the true state of the environment

This is where recurrent neural networks (RNNs) become particularly valuable. Unlike standard feedforward networks, RNNs maintain an internal
memory state that can help the agent:

    - Track important features over time
    - Identify temporal patterns in the observation sequence
    - Make better decisions with incomplete information

The Pendulum-v1 environment we'll be using demonstrates this concept by masking velocity information - forcing the agent to infer angular velocity
from position changes over time rather than receiving it directly. This creates a POMDP that requires temporal reasoning to solve effectively.

Dependencies
------------

.. code-block:: python

    # Author: Jaime Sabal
    import gymnasium as gym
    import numpy as np
    import torch
    from typing import List
    from tqdm import trange

    from agilerl.algorithms import PPO
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.utils.utils import create_population, make_vect_envs
    from agilerl.rollouts.on_policy import collect_rollouts_recurrent

Defining Hyperparameters
------------------------
Before we commence training, it's easiest to define all of our hyperparameters in one dictionary. Below is an example of
such for the PPO algorithm. Additionally, we also define a mutations parameters dictionary, in which we determine what
mutations we want to happen, to what extent we want these mutations to occur, and what RL hyperparameters we want to tune.
Additionally, we also define our upper and lower limits for these hyperparameters to define search spaces.

.. collapse:: Hyperparameter Configuration
    :open:

    .. code-block:: python

        # Initial hyperparameters
        INIT_HP = {
            "POP_SIZE": 4,  # Population size
            "BATCH_SIZE": 256,  # Batch size
            "LR": 0.001,  # Learning rate
            "LEARN_STEP": 1024,  # Learning frequency
            "GAMMA": 0.9,  # Discount factor
            "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
            "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
            "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
            "ENT_COEF": 0.0,  # Entropy coefficient
            "VF_COEF": 0.5,  # Value function coefficient
            "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
            "RECURRENT": True # Flag to signal that we want a recurrent policy
            "USE_ROLLOUT_BUFFER ": True # Use a rollout buffer for data collection
            "TARGET_KL": None,  # Target KL divergence threshold
            "UPDATE_EPOCHS": 4,  # Number of policy update epochs
            "TARGET_SCORE": 200.0,  # Target score that will beat the environment
            "MAX_STEPS": 150000,  # Maximum number of steps an agent takes in an environment
            "EVO_STEPS": 10000,  # Evolution frequency
            "EVAL_STEPS": None,  # Number of evaluation steps per episode
            "EVAL_LOOP": 3,  # Number of evaluation episodes
            "TOURN_SIZE": 2,  # Tournament size
            "ELITISM": True,  # Elitism in tournament selection
        }

        # Mutation parameters
        MUT_P = {
            # Mutation probabilities
            "NO_MUT": 0.4,  # No mutation
            "ARCH_MUT": 0.2,  # Architecture mutation
            "NEW_LAYER": 0.2,  # New layer mutation
            "PARAMS_MUT": 0.2,  # Network parameters mutation
            "ACT_MUT": 0.2,  # Activation layer mutation
            "RL_HP_MUT": 0.2,  # Learning HP mutation
            "MUT_SD": 0.1,  # Mutation strength
            "RAND_SEED": 42,  # Random seed
        }

        # RL hyperparameters configuration for mutation during training
        hp_config = HyperparameterConfig(
            lr = RLParameter(min=1e-4, max=1e-2),
            batch_size = RLParameter(
                min=8, max=1024, dtype=int
                )
        )

Create the Environment
----------------------
In this particular tutorial, we will be focusing on the ``Pendulum-v1`` environment with masked angular velocities. To do the
latter, we can define a wrapper to modify the observations after they have been collected.

.. code-block:: python

    class MaskVelocityWrapper(gym.ObservationWrapper):
        """
        Gym environment observation wrapper used to mask velocity terms in
        observations. The intention is the make the MDP partially observable.
        Adapted from https://github.com/LiuWenlin595/FinalProject.

        Taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/wrappers.py#L299.

        :param env: Gym environment
        """

        # Supported envs
        velocity_indices: ClassVar[dict[str, np.ndarray]] = {
            "CartPole-v1": np.array([1, 3]),
            "MountainCar-v0": np.array([1]),
            "MountainCarContinuous-v0": np.array([1]),
            "Pendulum-v1": np.array([2]),
            "LunarLander-v3": np.array([2, 3, 5]),
            "LunarLanderContinuous-v3": np.array([2, 3, 5]),
        }

        def __init__(self, env: gym.Env):
            super().__init__(env)

            assert env.unwrapped.spec is not None
            env_id: str = env.unwrapped.spec.id
            # By default no masking
            self.mask = np.ones_like(env.observation_space.sample())
            try:
                # Mask velocity
                self.mask[self.velocity_indices[env_id]] = 0.0
            except KeyError as e:
                raise NotImplementedError(f"Velocity masking not implemented for {env_id}") from e

        def observation(self, observation: np.ndarray) -> np.ndarray:
            observation = np.squeeze(observation)
            return observation * self.mask


.. code-block:: python

    def make_env():
        return MaskVelocityWrapper(gym.make("Pendulum-v1"))

    num_envs = 8
    env = make_vect_envs(make_env=make_env, num_envs=num_envs, should_async_vector=False)

    observation_space = env.single_observation_space
    action_space = env.single_action_space

Create a Population of Agents
-----------------------------
To perform evolutionary HPO, we require a population of agents. Since PPO is an on-policy algorithm, there is no
experience replay and so members in the population will not share experiences like they do with off-policy algorithms.
That being said, tournament selection and mutation still prove to be highly effective in determining the efficacy of
certain hyperparameters. Individuals that learn best are more likely to survive until the next generation, and so their
hyperparameters are more likely to remain present in the population. The sequence of evolution (tournament selection
followed by mutations) is detailed further below.

.. code-block:: python

    # Set-up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
    net_config = {
        "encoder_config": {
            "hidden_state_size": 64,  # LSTM hidden state size
            "num_layers": 1,
            "max_seq_len": 1024,
        },
    }

    # Define a population
    pop = create_population(
        algo="PPO",  # RL algorithm
        observation_space=observation_space,  # State dimension
        action_space=action_space,  # Action dimension
        net_config=net_config,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameter
        hp_config=hp_config,  # RL hyperparameter configuration
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,
        device=device,
    )

Creating Mutations and Tournament Objects
-----------------------------------------
Tournament selection is used to select the agents from a population which will make up the next generation of agents. If
elitism is used, the best agent from a population is automatically preserved and becomes a member of the next generation.
Then, for each tournament, k individuals are randomly chosen, and the agent with the best evaluation fitness is preserved.
This is repeated until the population for the next generation is full.

The class ``TournamentSelection()`` defines the functions required for tournament selection. TournamentSelection.select()
returns the best agent, and the new generation of agents.

.. code-block:: python

    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVAL_LOOP"],
    )

Mutation is periodically used to explore the hyperparameter space, allowing different hyperparameter combinations to be
trialled during training. If certain hyperparameters prove relatively beneficial to training, then that agent is more
likely to be preserved in the next generation, and so those characteristics are more likely to remain in the population.

The ``Mutations()`` class is used to mutate agents with pre-set probabilities. The available mutations currently implemented are:

* No mutation
* Network architecture mutation - adding layers or nodes. Trained weights are reused and new weights are initialized randomly.
* Network parameters mutation - mutating weights with Gaussian noise.
* Network activation layer mutation - change of activation layer.
* RL algorithm mutation - mutation of learning hyperparameter, such as learning rate or batch size.

``Mutations.mutation(population)`` returns a mutated population.

Tournament selection and mutation should be applied sequentially to fully evolve a population between evaluation and learning cycles.

.. code-block:: python

    mutations = Mutations(
        no_mutation=MUT_P["NO_MUT"],
        architecture=MUT_P["ARCH_MUT"],
        new_layer_prob=MUT_P["NEW_LAYER"],
        parameters=MUT_P["PARAMS_MUT"],
        activation=MUT_P["ACT_MUT"],
        rl_hp=MUT_P["RL_HP_MUT"],
        mutation_sd=MUT_P["MUT_SD"],
        rand_seed=MUT_P["RAND_SEED"],
        device=device,
    )

Training and Saving an Agent
----------------------------

Using AgileRL ``train_on_policy`` function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The simplest way to train an AgileRL agent is to use one of the implemented AgileRL train functions.
Given that PPO is an on-policy algorithm, we can make use of the ``train_on_policy`` function. This
training function will orchestrate the training and hyperparameter optimisation process, removing the
the need to implement a training loop. It will return a trained population, as well as the associated
fitnesses (fitness is each agents test scores on the environment).

.. code-block:: python

    # Define a save path for our trained agent
    save_path = "PPO_trained_agent.pt"

    trained_pop, pop_fitnesses = train_on_policy(
        env=env,
        env_name="PendulumPO-v1",
        algo="PPO",
        pop=pop,
        INIT_HP=INIT_HP,
        MUT_P=MUT_P,
        max_steps=INIT_HP["MAX_STEPS"],
        evo_steps=INIT_HP["EVO_STEPS"],
        eval_steps=INIT_HP["EVAL_STEPS"],
        eval_loop=INIT_HP["EVAL_LOOP"],
        tournament=tournament,
        mutation=mutations,
        wb=False,  # Boolean flag to record run with Weights & Biases
        save_elite=True,  # Boolean flag to save the elite agent in the population
        elite_path=save_path,
    )

.. note::

   Known `Gymnasium issue <https://github.com/Farama-Foundation/Gymnasium/issues/722>`_ - running vectorize environments as top-level code (without ``if __name__ == "__main__":``) may cause multiprocessing errors. To fix, run the above as a method under ``main``, e.g.

   .. code-block:: python

      def train_agent():
          # ... training code

      if __name__ == "__main__":
          train_agent()


Using a custom training loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If we wanted to have more control over the training process, it is also possible to write our own custom
training loops to train our agents. The training loop below can be used alternatively to the above ``train_on_policy``
function and is an example of how we might choose to make use of a population of AgileRL agents in our own training loop.

.. collapse:: Custom Training Loop

    .. code-block:: python


        # --- Training Loop (Performance-Flamegraph Style) ---
        max_steps = 1_000_000 // len(pop)
        required_score = 0.95
        evo_steps = num_envs * INIT_HP["LEARN_STEP"] * 1
        eval_steps = None

        total_steps = 0
        training_complete = False

        print("Training...")
        pbar = trange(max_steps * len(pop), unit="step")
        while (
            np.less([agent.steps[-1] for agent in pop], max_steps).all()
            and not training_complete
        ):
            for agent in pop:
                collect_rollouts_recurrent(agent, env)
                agent.learn()
                total_steps += agent.learn_step * num_envs
                agent.steps[-1] += agent.learn_step * num_envs
                pbar.update(agent.learn_step * num_envs // len(pop))

            # Evaluate and evolve
            if total_steps % evo_steps == 0:
                fitnesses = [
                    agent.test(
                        single_test_env,
                        max_steps=eval_steps,
                        loop=eval_loop,
                    )
                    for agent in pop
                ]
                mean_scores = [
                    round(np.mean(agent.fitness[-eval_loop:]), 1) for agent in pop
                ]
                print(f"--- Global steps {total_steps} ---")
                print(f"Steps {[agent.steps[-1] for agent in pop]}")
                print(f"Scores: {mean_scores}")
                print(f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}")
                print(
                    f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}"
                )

                if any(score >= required_score for score in mean_scores):
                    print(
                        f"\nAgent achieved required score {required_score}. Stopping training."
                    )
                    elite, _ = tournament.select(pop)
                    training_complete = True
                    break

                elite, pop = tournament.select(pop)
                pop = mutations.mutation(pop)
                for agent in pop:
                    agent.steps.append(agent.steps[-1])

        pbar.close()
        env.close()


Loading an Agent for Inference and Rendering your Solved Environment
--------------------------------------------------------------------
Once we have trained and saved an agent, we may want to then use our trained agent for inference. Below outlines
how we would load a saved agent and how it can then be used in a testing loop.

Load agent
~~~~~~~~~~
.. code-block:: python

    ppo = PPO.load(save_path, device=device)

Test loop for inference
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    single_test_env = gym.vector.SyncVectorEnv([make_env])
    total_steps = 0
    episode_rewards = []

    for episode in range(20):
        obs, _ = single_test_env.reset()
        done = np.array([False])
        episode_reward = 0
        episode_steps = 0
        hidden_state = ppo.get_initial_hidden_state(1)

        while not done[0]:
            action, _, _, _, hidden_state = ppo.get_action(
                obs, hidden_state=hidden_state
            )
            obs, reward, terminated, truncated, _ = single_test_env.step(action)
            done = np.logical_or(terminated, truncated)
            episode_reward += reward[0]
            episode_steps += 1
        print(
            f"Episode {episode + 1}: Reward: {episode_reward}, Steps: {episode_steps}"
        )
        total_steps += episode_steps
        episode_rewards.append(episode_reward)

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = total_steps / len(episode_rewards)
    print(f"Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")
