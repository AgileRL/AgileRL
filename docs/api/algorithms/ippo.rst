.. _ippo:

Independent Proximal Policy Optimization (IPPO)
===============================================

IPPO (Independent Proximal Policy Optimization) extends the PPO algorithm for multi-agent settings,
enabling cooperative or competitive training of multiple agents in complex environments.
The algorithm employs independent learning, in which each agent simply estimates its local value function,
and is well-suited to problems with many homogeneous agents.

* IPPO paper: https://arxiv.org/pdf/2011.09533

Can I use it?
-------------

.. list-table::
   :widths: 20 20 20
   :header-rows: 1

   * -
     - Action
     - Observation
   * - Discrete
     - ✔️
     - ✔️
   * - Continuous
     - ✔️
     - ✔️


Homogeneous Agents
------------------

IPPO can efficiently solve environments with large numbers of homogeneous (identical) agents because they share actor and critic networks.
This is useful for problems where we want multiple agents to learn the same behaviour, and can avoid training them all individually. Allowing
all homogeneous agents to learn from the experiences collected by each other can be a very fast way to explore an environment.

Labelling agents as homogeneous (or not) is as simple as choosing the names of agents in an environment. The agent_ids will be
read from the environment, and split on the final ``"_"``. Any agent_ids with matching prefixes will be assumed to be homogeneous.

For example, if an environment contains agents named ``"bob_0"``, ``"bob_1"`` and ``"fred_0"``, then ``"bob_0"`` and ``"bob_1"`` will be assumed to be homogeneous,
and the same actor and critic networks will be used for them. ``"fred_0"`` will receive its own actor and network, since it has a different prefix.

.. code-block:: python

  env.agent_ids = ["bob_0", "bob_1", "fred_0"]

  agent = IPPO(
    observation_spaces=env.observation_spaces,
    action_spaces=env.action_spaces,
    agent_ids=env.agent_ids
  )

Agents must have the same observation and action spaces to be homogeneous. In the above example, all ``bob_`` agents must have the same observation
and action spaces, but these can be different to the observation and action spaces of ``fred_`` agents.


Agent Masking
-------------

If you need to take actions from agents at different timesteps, you can use agent masking to only retrieve new actions for certain agents whilst
providing 'environment defined actions' for other agents, which act as a nominal action for such "masked" agents to take. These nominal actions
should be returned as part of the ``info`` dictionary. Following the PettingZoo API we recommend the ``info`` dictionary to be keyed by the
agents, with ``env_defined_actions`` defined as follows:

.. code-block:: python

    info = {'speaker_0': {'env_defined_actions':  None},
            'listener_0': {'env_defined_actions': np.array([0,0,0,0,0])}

For agents that you wish not to be masked, the ``env_defined_actions`` should be set to ``None``. If your environment has discrete action spaces
then provide 'env_defined_actions' as a numpy array with a single value. For example, an action space of type ``Discrete(5)`` may have an
``env_defined_action`` of ``np.array([4])``. For an environment with continuous actions spaces (e.g. ``Box(0, 1, (5,))``) then the shape of the
array should be the size of the action space (``np.array([0.5, 0.5, 0.5, 0.5, 0.5])``). Agent masking is handled automatically by the AgileRL
multi-agent training function by passing the info dictionary into the agents get_action method:

.. code-block:: python

    state, info = env.reset()  # or: next_state, reward, done, truncation, info = env.step(action)
    cont_actions, discrete_action = agent.get_action(state, infos=info)
    if agent.discrete_actions:
        action = discrete_action
    else:
        action = cont_actions


Example Training Loop
---------------------

.. code-block:: python

    import numpy as np
    import torch
    from pettingzoo.mpe import simple_speaker_listener_v4
    from tqdm import trange

    from agilerl.algorithms import IPPO
    from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
    from agilerl.utils.algo_utils import obs_channels_to_first

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = 8
    env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
    env.reset()

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    agent_ids = [agent_id for agent_id in env.agents]

    channels_last = False  # Flag to swap image channels dimension from last to first [H, W, C] -> [C, H, W]

    agent = IPPO(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
    )

    # Define training loop parameters
    max_steps = 100000  # Max steps
    while agent.steps[-1] < max_steps:
        state, info  = env.reset() # Reset environment at start of episode
        scores = np.zeros((num_envs, len(agent.shared_agent_ids)))
        completed_episode_scores = []
        steps = 0

        if channels_last:
            state = {
                agent_id: obs_channels_to_first(s)
                for agent_id, s in state.items()
            }

        for _ in range(agent.learn_step):

            states = {agent_id: [] for agent_id in agent.agent_ids}
            actions = {agent_id: [] for agent_id in agent.agent_ids}
            log_probs = {agent_id: [] for agent_id in agent.agent_ids}
            rewards = {agent_id: [] for agent_id in agent.agent_ids}
            dones = {agent_id: [] for agent_id in agent.agent_ids}
            values = {agent_id: [] for agent_id in agent.agent_ids}

            done = {agent_id: np.zeros(num_envs) for agent_id in agent.agent_ids}

            for idx_step in range(-(agent.learn_step // -num_envs)):

                # Get next action from agent
                action, log_prob, _, value = agent.get_action(obs=state, infos=info)

                # Act in environment
                next_state, reward, termination, truncation, info = env.step(action)
                scores += np.array(list(reward.values())).transpose()

                steps += num_envs

                next_done = {}
                for agent_id in agent.agent_ids:
                    states[agent_id].append(state[agent_id])
                    actions[agent_id].append(action[agent_id])
                    log_probs[agent_id].append(log_prob[agent_id])
                    rewards[agent_id].append(reward[agent_id])
                    dones[agent_id].append(done[agent_id])
                    values[agent_id].append(value[agent_id])
                    next_done[agent_id] = np.logical_or(termination[agent_id], truncation[agent_id]).astype(np.int8)

                if channels_last:
                    next_state = {
                        agent_id: obs_channels_to_first(s)
                        for agent_id, s in next_state.items()
                    }

                # Find which agents are "done" - i.e. terminated or truncated
                dones = {
                    agent_id: termination[agent_id] | truncation[agent_id]
                    for agent_id in agent.agent_ids
                }

                # Calculate scores for completed episodes
                for idx, agent_dones in enumerate(zip(*dones.values())):
                    if all(agent_dones):
                        completed_score = list(scores[idx])
                        completed_episode_scores.append(completed_score)
                        agent.scores.append(completed_score)
                        scores[idx].fill(0)

                state = next_state
                done = next_done

            experiences = (
                states,
                actions,
                log_probs,
                rewards,
                dones,
                values,
                next_state,
                next_done,
            )

            # Learn according to agent's RL algorithm
            loss = agent.learn(experiences)

        agent.steps[-1] += steps


Neural Network Configuration
----------------------------

To configure the architecture of the network's encoder / head, pass a kwargs dict to the IPPO ``net_config`` field.
Full arguments can be found in the documentation of :ref:`EvolvableMLP<mlp>`, :ref:`EvolvableCNN<cnn>`, and
:ref:`EvolvableMultiInput<multi_input>`.

For discrete / vector observations:

.. code-block:: python

  NET_CONFIG = {
        "encoder_config": {'hidden_size': [32, 32]},  # Network head hidden size
        "head_config": {'hidden_size': [32]}      # Network head hidden size
    }

For image observations:

.. code-block:: python

  NET_CONFIG = {
      "encoder_config": {
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
      },
      "head_config": {'hidden_size': [32]}  # Network head hidden size
    }

For dictionary / tuple observations containing any combination of image, discrete, and vector observations:

.. code-block:: python

  NET_CONFIG = {
      "encoder_config": {
        'hidden_size': [32, 32],  # Network head hidden size
        'channel_size': [32, 32], # CNN channel size
        'kernel_size': [8, 4],   # CNN kernel size
        'stride_size': [4, 2],   # CNN stride size
      },
      "head_config": {'hidden_size': [32]}  # Network head hidden size
    }

.. code-block:: python

  # Create IPPO agent
  agent = IPPO(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    net_config=NET_CONFIG,
    device=device,
  )

Evolutionary Hyperparameter Optimization
----------------------------------------

AgileRL allows for efficient hyperparameter optimization during training to provide state-of-the-art results in a fraction of the time.
For more information on how this is done, please refer to the :ref:`Evolutionary Hyperparameter Optimization <evo_hyperparam_opt>` documentation.

Saving and loading agents
-------------------------

To save an agent, use the ``save_checkpoint`` method:

.. code-block:: python

  from agilerl.algorithms import IPPO

  # Create IPPO agent
  agent = IPPO(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    net_config=NET_CONFIG,
    device=device,
  )

  checkpoint_path = "path/to/checkpoint"
  agent.save_checkpoint(checkpoint_path)

To load a saved agent, use the ``load`` method:

.. code-block:: python

  from agilerl.algorithms import IPPO

  checkpoint_path = "path/to/checkpoint"
  agent = IPPO.load(checkpoint_path)

Parameters
------------

.. autoclass:: agilerl.algorithms.ippo.IPPO
  :members:
  :inherited-members:
