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
    from typing import ClassVar
    from tqdm import trange

    from agilerl.typing import BPTTSequenceType
    from agilerl.algorithms import PPO
    from agilerl.wrappers.agent import RSNorm
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.utils.utils import make_vect_envs
    from agilerl.rollouts.on_policy import collect_rollouts_recurrent
    from agilerl.training.train_on_policy import train_on_policy

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
            return observation * self.mask


.. code-block:: python

    def make_env():
        return MaskVelocityWrapper(gym.make("Pendulum-v1"))

    num_envs = 16
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
        "latent_dim": 64,
        "encoder_config": {
            "hidden_state_size": 64,  # LSTM hidden state size
            "num_layers": 1,
        },
        "head_config": {
            "hidden_size": [128],
        },
    }

    # RL hyperparameters configuration for mutation during training
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=128, max=1024),
        ent_coef=RLParameter(min=0.0001, max=0.001, grow_factor=1.0, shrink_factor=0.9),
    )

    # Algorithm hyperparameters
    init_hp = {
        "batch_size": 256,
        "lr": 0.001,
        "learn_step": 1024 * num_envs,
        "gamma": 0.9,
        "gae_lambda": 0.95,
        "action_std_init": 0.6,
        "clip_coef": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "share_encoders": True,
        "recurrent": True,
        "target_kl": None,
        "update_epochs": 4,
        "bptt_sequence_type": BPTTSequenceType.CHUNKED,
        "max_seq_len": 16,
        "num_envs": num_envs,
    }

    # Define a population
    population_size = 4
    pop = PPO.population(
        size=population_size,
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config,
        hp_config=hp_config,
        device=device,
        wrapper_cls=RSNorm,
        **init_hp,
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
        tournament_size=2,
        elitism=True,
        population_size=population_size,
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
        no_mutation=0.4,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0.0,
        rl_hp=0.2,
        mutation_sd=0.1,
        rand_seed=42,
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

    # Training parameters
    max_steps = 1000000
    evo_steps = 10000
    eval_steps = None
    eval_loop = 3

    trained_pop, pop_fitnesses = train_on_policy(
        env=env,
        env_name="PendulumNoVel-v1",
        algo="PPO",
        pop=pop,
        init_hp=init_hp,
        max_steps=max_steps,
        evo_steps=evo_steps,
        eval_steps=eval_steps,
        eval_loop=eval_loop,
        selection_strategy=tournament,
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


Using a Custom Training Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If we wanted to have more control over the training process, it is also possible to write our own custom
training loops to train our agents. The training loop below can be used alternatively to the above ``train_on_policy``
function and is an example of how we might choose to make use of a population of AgileRL agents in our own training loop.

.. collapse:: Custom Training Loop

    .. code-block:: python

        from agilerl.rollouts import collect_rollouts_recurrent
        from agilerl.utils.utils import default_progress_bar

        # --- Training Loop ---
        max_steps = 1_000_000
        required_score = -200
        evo_steps = init_hp["learn_step"] * 5
        eval_steps = None

        total_steps = 0
        training_complete = False

        pbar = default_progress_bar(max_steps)
        while (
            np.less([agent.steps for agent in pop], max_steps).all()
            and not training_complete
        ):
            pop_episode_scores = []
            for agent in pop:
                steps = 0
                completed_episodes = []
                last_obs, last_done, last_scores, last_info = None, None, None, None
                for _ in range(-(evo_steps // -agent.learn_step)):
                    # Collect rollouts and save in buffer
                    episode_scores, last_obs, last_done, last_scores, last_info = (
                        collect_rollouts_recurrent(
                            agent,
                            env,
                            last_obs=last_obs,
                            last_done=last_done,
                            last_scores=last_scores,
                            last_info=last_info,
                        )
                    )

                    agent.learn() # Learn from rollout buffer

                    total_steps += agent.learn_step
                    steps += agent.learn_step
                    agent.steps += agent.learn_step
                    completed_episodes += episode_scores

                # Update step counter and scores
                pop_episode_scores.append(
                    round(np.mean(completed_episodes), 2)
                    if len(completed_episodes) > 0
                    else "0 completed episodes"
                )

                pbar.update(steps // len(pop))

            # Evaluate and evolve
            fitnesses = [
                agent.test(
                    single_test_env,
                    max_steps=eval_steps,
                    loop=eval_loop,
                )
                for agent in pop
            ]

            pbar.write(
                f"--- Global steps {total_steps} ---\n"
                f"Steps: {[agent.steps for agent in pop]}\n"
                f"Scores: {pop_episode_scores}\n"
                f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}\n"
                f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}\n"
                f"Mutations: {[agent.mut for agent in pop]}\n"
            )

            if any(score >= required_score for score in pop_episode_scores):
                print(
                    f"\nAgent achieved required score {required_score}. Stopping training."
                )
                elite, _, _ = tournament.select(pop)
                training_complete = True
                break

            elite, pop, _ = tournament.select(pop)
            pop = mutations.mutation(pop)

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
