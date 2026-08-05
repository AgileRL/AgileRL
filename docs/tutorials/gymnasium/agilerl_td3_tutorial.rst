.. _td3_tutorial:


Lunar Lander with TD3
=====================

In this tutorial, we will be training and optimising the hyperparameters of a population of TD3 agents
to beat the Gymnasium continuous lunar lander environment. AgileRL is a deep reinforcement learning
library, focussed on improving the RL training process through evolutionary hyperparameter
optimisation (HPO), which has resulted in up to 10x faster HPO compared to other popular deep RL
libraries. Check out the AgileRL github `repository <https://github.com/AgileRL/AgileRL/>`__ for
more information about the library.

To complete the lunar lander environment, the agent must learn to fire the engine left, right, up,
or not at all to safely navigate the lander to the landing pad without crashing.

.. figure:: agilerl_td3_lunar_lander.gif
  :width: 400
  :alt: agent-environment-diagram
  :align: center

  Figure 1: Completed Lunar Lander environment using an AgileRL TD3 agent


TD3 Overview
------------

TD3 (twin-delayed deep deterministic policy gradient) is an off-policy actor-critic algorithm used
to estimate the optimal policy function, which determines what actions an agent should take given the
observed state of the environment. The agent does this by using a policy network (actor) to determine actions
given a particular state and then a value network (critic) to estimate the Q-value of the state-action pairs
determined by the policy network (actor). TD3 improves upon DDPG (deep deterministic policy gradient) to reduce
overestimation bias by doing the following:

* Using two Q networks (critics) and selecting the minimum Q-value
* Updating the policy network less frequently than the Q network
* Adding noise to actions used to estimate the target Q value


Dependencies
------------

.. code-block:: python

    # Author: Michael Pratt
    import os

    import imageio
    import gymnasium as gym
    import numpy as np
    import torch
    from tqdm import trange

    from agilerl.algorithms import TD3
    from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
    from agilerl.components.replay_buffer import ReplayBuffer
    from agilerl.hpo.mutation import Mutations
    from agilerl.hpo.tournament import TournamentSelection
    from agilerl.training.train_off_policy import train_off_policy
    from agilerl.utils.utils import make_vect_envs

Create the Environment
----------------------
In this particular tutorial, we will be focussing on the continuous lunar lander environment as TD3 can only be
used with continuous action environments.

.. code-block:: python

    # Create vectorized environment
    num_envs = 8
    env = make_vect_envs("LunarLanderContinuous-v3", num_envs=num_envs)  # Create environment

    observation_space = env.single_observation_space
    action_space = env.single_action_space


Create a Population of Agents
-----------------------------
To perform evolutionary HPO, we require a population of agents. Individuals in this population will share experiences but
learn individually, allowing us to determine the efficacy of certain hyperparameters. Individuals that learn best
are more likely to survive until the next generation, and so their hyperparameters are more likely to remain present in the
population. The sequence of evolution (tournament selection followed by mutation) is detailed further below.

.. code-block:: python

    # Set-up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the network configuration of a simple mlp with two hidden layers, each with 64 nodes
    net_config = {
        "encoder_config": {"hidden_size": [64, 64]},  # Encoder hidden size
        "head_config": {"hidden_size": [64, 64]},  # Head hidden size
    }

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor = RLParameter(min=1e-4, max=1e-2),
        lr_critic = RLParameter(min=1e-4, max=1e-2),
        learn_step = RLParameter(min=1, max=16),
        batch_size = RLParameter(min=8, max=512),
    )

    # Algorithm hyperparameters
    init_hp = {
        "batch_size": 128,
        "lr_actor": 0.0001,
        "lr_critic": 0.001,
        "O_U_noise": True,
        "expl_noise": 0.1,
        "mean_noise": 0.0,
        "theta": 0.15,
        "dt": 0.01,
        "gamma": 0.99,
        "policy_freq": 2,
        "learn_step": 1,
        "tau": 0.005,
        "num_envs": num_envs,
    }

    # Define a population
    pop = TD3.population(
        size=4,
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config,
        hp_config=hp_config,
        device=device,
        **init_hp,
    )


Experience Replay
-----------------

In order to efficiently train a population of RL agents, off-policy algorithms are able to share memory within populations.
This reduces the exploration needed by an individual agent because it allows faster learning from the behaviour of other agents.
For example, if you were able to watch a bunch of people attempt to solve a maze, you could learn from their mistakes and successes
without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by
the class ``ReplayBuffer()``. During training we use the ``ReplayBuffer.add()`` function to add experiences to the buffer as ``TensorDict``
objects. Specifically, we wrap transitions through the ``Transition`` tensorclass that wraps the ``obs``, ``action``, ``reward``,
``next_obs``, and ``done`` fields as ``torch.Tensor`` objects. To sample from the replay buffer, call ``ReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer

    memory = ReplayBuffer(
        max_size=10000,  # Max replay buffer size
        device=device,
    )


Creating Mutations and Tournament objects
-----------------------------------------

Tournament selection is used to select the agents from a population which will make up the next generation of agents. If
elitism is used, the best agent from a population is automatically preserved and becomes a member of the next generation.
Then, for each tournament, k individuals are randomly chosen, and the agent with the best evaluation fitness is preserved.
This is repeated until the population for the next generation is full.

The class ``TournamentSelection()`` defines the functions required for tournament selection. ``TournamentSelection.select()``
returns the best agent, and the new generation of agents.

.. code-block:: python

    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=4,
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

``Mutations.mutation()`` returns a mutated population.
Tournament selection and mutation should be applied sequentially to fully evolve a population between evaluation and learning cycles.

.. code-block:: python

    mutations = Mutations(
        no_mutation=0.4,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0.2,
        rl_hp=0.2,
        mutation_sd=0.1,
        rand_seed=42,
        device=device,
    )


Training and Saving an Agent
----------------------------
The simplest way to train an AgileRL agent is to use one of the implemented AgileRL train functions.
Given that TD3 is an off-policy algorithm, we can make use of the ``train_off_policy`` function. This
training function will orchestrate the training and hyperparameter optimisation process, removing the
the need to implement a custom training loop. It will return a trained population, as well as the associated
fitnesses (fitness is each agents test scores on the environment).

.. code-block:: python

    # Training parameters
    max_steps = 200000
    evo_steps = 10000
    eval_steps = None
    eval_loop = 1
    learning_delay = 1000
    target_score = 200.0

    trained_pop, pop_fitnesses = train_off_policy(
        env=env,
        env_name="LunarLanderContinuous-v3",
        algo="TD3",
        pop=pop,
        memory=memory,
        init_hp=init_hp,
        max_steps=max_steps,
        evo_steps=evo_steps,
        eval_steps=eval_steps,
        eval_loop=eval_loop,
        learning_delay=learning_delay,
        target=target_score,
        tournament=tournament,
        mutation=mutations,
        wb=False,  # Boolean flag to record run with Weights & Biases
        save_elite=True,  # Boolean flag to save the elite agent in the population
        elite_path="TD3_trained_agent.pt",
    )

.. note::

   Known `Gymnasium issue <https://github.com/Farama-Foundation/Gymnasium/issues/722>`_ - running vectorize environments as top-level code (without ``if __name__ == "__main__":``)
   may cause multiprocessing errors. To fix, run the above as a method under ``main``, e.g.

   .. code-block:: python

      def train_agent():
          # ... training code

      if __name__ == "__main__":
          train_agent()

Using a custom training loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If we wanted to have more control over the training process, it is also possible to write our own custom
training loops to train our agents. The training loop below can be used alternatively to the above ``train_off_policy``
function and is an example of how we might choose to make use of a population of AgileRL agents in our own training loop.

.. collapse:: Custom Training Loop

    .. code-block:: python

        total_steps = 0

        # TRAINING LOOP
        pbar = trange(max_steps, unit="step")
        while np.less([agent.steps for agent in pop], max_steps).all():
            pop_episode_scores = []
            for agent in pop:  # Loop through population
                obs, info = env.reset()  # Reset environment at start of episode
                scores = np.zeros(num_envs)
                completed_episode_scores = []
                steps = 0

                for idx_step in range(evo_steps // num_envs):
                    action = agent.get_action(obs)  # Get next action from agent

                    # Act in environment
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    scores += np.array(reward)
                    steps += num_envs
                    total_steps += num_envs

                    # Collect scores for completed episodes
                    reset_noise_indices = []
                    for idx, (d, t) in enumerate(zip(terminated, truncated)):
                        if d or t:
                            completed_episode_scores.append(scores[idx])
                            agent.scores.append(scores[idx])
                            scores[idx] = 0
                            reset_noise_indices.append(idx)

                    # Reset action noise
                    agent.reset_action_noise(reset_noise_indices)

                    # Save experience to replay buffer
                    done = terminated or truncated
                    transition = Transition(
                        obs=obs,
                        action=action,
                        reward=reward,
                        next_obs=next_obs,
                        done=done,
                        batch_size=[num_envs]
                    )
                    transition = transition.to_tensordict()
                    memory.add(transition)

                    # Learn according to learning frequency
                    if memory.size > learning_delay and len(memory) >= agent.batch_size:
                        for _ in range(num_envs // agent.learn_step):
                            # Sample replay buffer
                            experiences = memory.sample(agent.batch_size)
                            # Learn according to agent's RL algorithm
                            agent.learn(experiences)

                    obs = next_obs

                pbar.update(evo_steps // len(pop))
                agent.steps += steps
                pop_episode_scores.append(completed_episode_scores)

            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
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
            print(f"Steps {[agent.steps for agent in pop]}")
            print(f"Scores: {mean_scores}")
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(
                f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

        # Save the trained algorithm
        save_path = "TD3_trained_agent.pt"
        elite.save_checkpoint(save_path)

        pbar.close()
        env.close()


Loading an Agent for Inference and Rendering your Solved Environment
--------------------------------------------------------------------
Once we have trained and saved an agent, we may want to then use our trained agent for inference. Below outlines
how we would load a saved agent and how it can then be used in a testing loop.


Load agent
~~~~~~~~~~
.. code-block:: python

    td3 = TD3.load_checkpoint(save_path, device=device)


Test loop for inference
~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    test_env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    rewards = []
    frames = []
    testing_eps = 7
    max_testing_steps = 1000
    td3.set_training_mode(False)
    with torch.no_grad():
        for ep in range(testing_eps):
            obs = test_env.reset()[0]  # Reset environment at start of episode
            score = 0

            for step in range(max_testing_steps):
                # Get next action from agent
                action, *_ = td3.get_action(obs)

                # Save the frame for this step and append to frames list
                frame = test_env.render()
                frames.append(frame)

                # Take the action in the environment
                obs, reward, terminated, truncated, _ = test_env.step(action)

                # Collect the score
                score += reward

                # Break if environment 0 is done or truncated
                if terminated or truncated:
                    print("terminated")
                    break

            # Collect and print episodic reward
            rewards.append(score)
            print("-" * 15, f"Episode: {ep}", "-" * 15)
            print("Episodic Reward: ", rewards[-1])

        print(rewards)

        test_env.close()



Save test episosdes as a gif
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    frames = frames[::3]
    gif_path = "./videos/"
    os.makedirs(gif_path, exist_ok=True)
    imageio.mimwrite(
        os.path.join("./videos/", "td3_lunar_lander.gif"), frames, duration=50, loop=0
    )
    mean_fitness = np.mean(rewards)
