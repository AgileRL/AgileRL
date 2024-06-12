.. _debugging_rl:

Debugging Reinforcement Learning
================================

Reinforcement learning is notoriously tricky to get working. There are a large number of things that can go wrong, and it is often difficult to appreciate what is causing issues.

Andy Jones provides a great `blog post <https://andyljones.com/posts/rl-debugging.html>`_ on debugging RL that offers good advice on how to figure out what your problem is, and how to solve it.

You can also join the AgileRL `Discord server <https://discord.com/invite/eB8HyTA2ux>`_ to ask questions, get help, and learn more about reinforcement learning.

Probe environments
------------------

Probe environments can be used to localise errors and confirm that algorithms are learning correctly. We provide various single- and multi-agent probe environments, for vector and image
observation spaces, and discrete and continuous action spaces, that can be used to debug reinforcement learning implementations. These are detailed in the tables below.

How to use Probe Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide functions that can be used directly or edited to quickly check your algorithm's ability to solve a probe environment. Each environment also contains inputs and corresponding outputs that a
correctly functioning agent should be able to learn, and can be used to diagnose issues.

.. collapse:: Single-agent - Check Q-learning

    .. code-block:: python

        import torch
        from agilerl.algorithms.dqn import DQN
        from agilerl.components.replay_buffer import ReplayBuffer
        from agilerl.utils.probe_envs import check_q_learning_with_probe_env

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vector_envs = [
            (ConstantRewardEnv(), 1000),
            (ObsDependentRewardEnv(), 1000),
            (DiscountedRewardEnv(), 3000),
            (FixedObsPolicyEnv(), 1000),
            (PolicyEnv(), 1000),
        ]

        for env, learn_steps in vector_envs:
            algo_args = {
                "state_dim": (env.observation_space.n,),
                "action_dim": env.action_space.n,
                "one_hot": True if env.observation_space.n > 1 else False,
                "lr": 1e-2,
            }

            field_names = ["state", "action", "reward", "next_state", "done"]
            memory = ReplayBuffer(
                memory_size=1000,  # Max replay buffer size
                field_names=field_names,  # Field names to store in memory
                device=device,
            )

            check_q_learning_with_probe_env(env, DQN, algo_args, memory, learn_steps, device)

    See function docs: :ref:`agilerl.utils.probe_envs.check_q_learning_with_probe_env<single_check_q_learning_with_probe_env>`

.. collapse:: Single-agent - Check Policy and Q-learning

    .. code-block:: python

        import torch
        from agilerl.algorithms.ddpg import DDPG
        from agilerl.components.replay_buffer import ReplayBuffer
        from agilerl.utils.probe_envs import check_policy_q_learning_with_probe_env

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cont_vector_envs = [
            (ConstantRewardContActionsEnv(), 1000),
            (ObsDependentRewardContActionsEnv(), 1000),
            (DiscountedRewardContActionsEnv(), 5000),
            (FixedObsPolicyContActionsEnv(), 3000),
            (PolicyContActionsEnv(), 3000),
        ]

        for env, learn_steps in cont_vector_envs:
            algo_args = {
                "state_dim": (env.observation_space.n,),
                "action_dim": env.action_space.shape[0],
                "one_hot": True if env.observation_space.n > 1 else False,
                "max_action": 1.0,
                "min_action": 0.0,
                "lr_actor": 1e-2,
                "lr_critic": 1e-2,
            }

            field_names = ["state", "action", "reward", "next_state", "done"]
            memory = ReplayBuffer(
                memory_size=1000,  # Max replay buffer size
                field_names=field_names,  # Field names to store in memory
                device=device,
            )

            check_policy_q_learning_with_probe_env(
                env, DDPG, algo_args, memory, learn_steps, device
            )

    See function docs: :ref:`agilerl.utils.probe_envs.check_policy_q_learning_with_probe_env<single_check_policy_q_learning_with_probe_env>`

.. collapse:: Single-agent - Check Policy and Value (On-Policy)

    .. code-block:: python

          import torch
          from agilerl.algorithms.ppo import PPO
          from agilerl.utils.probe_envs import check_policy_on_policy_with_probe_env

          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          cont_vector_envs = [
              (ConstantRewardContActionsEnv(), 1000),
              (ObsDependentRewardContActionsEnv(), 1000),
              (DiscountedRewardContActionsEnv(), 5000),
              (FixedObsPolicyContActionsEnv(), 3000),
              (PolicyContActionsEnv(), 3000),
          ]

          for env, learn_steps in cont_vector_envs:
              algo_args = {
                  "state_dim": (env.observation_space.n,),
                  "action_dim": env.action_space.shape[0],
                  "one_hot": True if env.observation_space.n > 1 else False,
                  "discrete_actions": False,
                  "lr": 0.001
              }

              check_policy_on_policy_with_probe_env(
                  env, PPO, algo_args, memory, learn_steps, device
        )

See function docs: :ref:`agilerl.utils.probe_envs.check_policy_on_policy_with_probe_env<single_check_policy_on_policy_with_probe_env>`


.. collapse:: Multi-agent - Check Policy and Q-learning

    .. code-block:: python

        import torch
        from agilerl.algorithms.maddpg import MADDPG
        from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
        from agilerl.utils.probe_envs_ma import check_policy_q_learning_with_probe_env

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vector_envs = [
            (ConstantRewardEnv(), 1000),
            (ObsDependentRewardEnv(), 1000),
            (DiscountedRewardEnv(), 3000),
            (FixedObsPolicyEnv(), 1000),
            (PolicyEnv(), 4000),
            (MultiPolicyEnv(), 8000),
        ]

        for env, learn_steps in vector_envs:
            algo_args = {
                "state_dims": [(env.observation_space[agent].n,) for agent in env.agents],
                "action_dims": [env.action_space[agent].n for agent in env.agents],
                "one_hot": True,
                "n_agents": env.num_agents,
                "agent_ids": env.possible_agents,
                "max_action": [(1.0,), (1.0,)],
                "min_action": [(0.0,), (0.0,)],
                "discrete_actions": True,
                "net_config": {"arch": "mlp", "hidden_size": [32, 32]},
                "batch_size": 256,
            }
            field_names = ["state", "action", "reward", "next_state", "done"]
            memory = MultiAgentReplayBuffer(
                memory_size=10000,  # Max replay buffer size
                field_names=field_names,  # Field names to store in memory
                agent_ids=algo_args["agent_ids"],
                device=device,
            )

            check_policy_q_learning_with_probe_env(env, MADDPG, algo_args, memory, learn_steps, device)

    See function docs: :ref:`agilerl.utils.probe_envs.check_policy_q_learning_with_probe_env<single_check_policy_q_learning_with_probe_env>`

Single and multi-agent probe environments are detailed in the tables below, with links to further documentation.

Single-agent Probe Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50, 50
   :header-rows: 1

   * - **Probe Environment**
     - **Details**
   * - :ref:`ConstantRewardEnv<single_ConstantRewardEnv>`

       :ref:`ConstantRewardImageEnv<single_ConstantRewardImageEnv>`

       :ref:`ConstantRewardContActionsEnv<single_ConstantRewardContActionsEnv>`

       :ref:`ConstantRewardContActionsImageEnv<single_ConstantRewardContActionsImageEnv>`

     - **Constant Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the value of the only observation they see is 1. Failure indicates a problem with the loss of this network.
   * - :ref:`ObsDependentRewardEnv<single_ObsDependentRewardEnv>`

       :ref:`ObsDependentRewardImageEnv<single_ObsDependentRewardImageEnv>`

       :ref:`ObsDependentRewardContActionsEnv<single_ObsDependentRewardContActionsEnv>`

       :ref:`ObsDependentRewardContActionsImageEnv<single_ObsDependentRewardContActionsImageEnv>`

     - **Observation-dependent Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the reward depends on the simple observation. Failure indicates a problem with the learning of this network.
   * - :ref:`DiscountedRewardEnv<single_DiscountedRewardEnv>`

       :ref:`DiscountedRewardImageEnv<single_DiscountedRewardImageEnv>`

       :ref:`DiscountedRewardContActionsEnv<single_DiscountedRewardContActionsEnv>`

       :ref:`DiscountedRewardContActionsImageEnv<single_DiscountedRewardContActionsImageEnv>`

     - **Discounted Reward Environment**

       Agents should be able to learn that the reward depends on the simple observation, and also apply discounting. Failure indicates a problem with reward discounting.
   * - :ref:`FixedObsPolicyEnv<single_FixedObsPolicyEnv>`

       :ref:`FixedObsPolicyImageEnv<single_FixedObsPolicyImageEnv>`

       :ref:`FixedObsPolicyContActionsEnv<single_FixedObsPolicyContActionsEnv>`

       :ref:`FixedObsPolicyContActionsImageEnv<single_FixedObsPolicyContActionsImageEnv>`

     - **Fixed-observation Policy Environment**

       Isolates the policy/actor network. Agents should be able to learn the reward depends on action taken under the same observation. Failure indicates a problem with policy loss or updates.
   * - :ref:`PolicyEnv<single_PolicyEnv>`

       :ref:`PolicyImageEnv<single_PolicyImageEnv>`

       :ref:`PolicyContActionsEnv<single_PolicyContActionsEnv>`

       :ref:`PolicyContActionsImageEnvSimple<single_PolicyContActionsImageEnvSimple>`

       :ref:`PolicyContActionsImageEnv<single_PolicyContActionsImageEnv>`

     - **Observation-dependent Policy Environment**

       Agents should be able to learn the reward depends on different actions taken under different observations. The value/critic and policy/actor networks work together to learn to solve the environment. The policy network should learn the correct actions to output and the value network should learn the value. With Q-learning, the actor is doing both. Failure indicates a problem with the overall algorithm, batching, or even hyperparameters.


Multi-agent Probe Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50, 50
   :header-rows: 1

   * - **Probe Environment**
     - **Details**
   * - :ref:`ConstantRewardEnv<multi_ConstantRewardEnv>`

       :ref:`ConstantRewardImageEnv<multi_ConstantRewardImageEnv>`

       :ref:`ConstantRewardContActionsEnv<multi_ConstantRewardContActionsEnv>`

       :ref:`ConstantRewardContActionsImageEnv<multi_ConstantRewardContActionsImageEnv>`

     - **Constant Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the value of the only observation they see is 1. Failure indicates a problem with the loss of this network.
   * - :ref:`ObsDependentRewardEnv<multi_ObsDependentRewardEnv>`

       :ref:`ObsDependentRewardImageEnv<multi_ObsDependentRewardImageEnv>`

       :ref:`ObsDependentRewardContActionsEnv<multi_ObsDependentRewardContActionsEnv>`

       :ref:`ObsDependentRewardContActionsImageEnv<multi_ObsDependentRewardContActionsImageEnv>`

     - **Observation-dependent Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the reward depends on the simple observation. Failure indicates a problem with the learning of this network.
   * - :ref:`DiscountedRewardEnv<multi_DiscountedRewardEnv>`

       :ref:`DiscountedRewardImageEnv<multi_DiscountedRewardImageEnv>`

       :ref:`DiscountedRewardContActionsEnv<multi_DiscountedRewardContActionsEnv>`

       :ref:`DiscountedRewardContActionsImageEnv<multi_DiscountedRewardContActionsImageEnv>`

     - **Discounted Reward Environment**

       Agents should be able to learn that the reward depends on the simple observation, and also apply discounting. Failure indicates a problem with reward discounting.
   * - :ref:`FixedObsPolicyEnv<multi_FixedObsPolicyEnv>`

       :ref:`FixedObsPolicyImageEnv<multi_FixedObsPolicyImageEnv>`

       :ref:`FixedObsPolicyContActionsEnv<multi_FixedObsPolicyContActionsEnv>`

       :ref:`FixedObsPolicyContActionsImageEnv<multi_FixedObsPolicyContActionsImageEnv>`

     - **Fixed-observation Policy Environment**

       Isolates the policy/actor network. Agents should be able to learn the reward depends on action taken under the same observation. Failure indicates a problem with policy loss or updates.
   * - :ref:`PolicyEnv<multi_PolicyEnv>`

       :ref:`PolicyImageEnv<multi_PolicyImageEnv>`

       :ref:`PolicyContActionsEnv<multi_PolicyContActionsEnv>`

       :ref:`PolicyContActionsImageEnv<multi_PolicyContActionsImageEnv>`

     - **Observation-dependent Policy Environment**

       Agents should be able to learn the reward depends on different actions taken under different observations. The value/critic and policy/actor networks work together to learn to solve the environment. The policy network should learn the correct actions to output and the value network should learn the value. With Q-learning, the actor is doing both. Failure indicates a problem with the overall algorithm, batching, or even hyperparameters.
   * - :ref:`MultiPolicyEnv<multi_MultiPolicyEnv>`

       :ref:`MultiPolicyImageEnv<multi_MultiPolicyImageEnv>`

     - **Observation-dependent Multi-agent Policy Environment**

       Harder version of Observation-dependent Policy Environment. Critic networks should be able to evaluate a reward dependent on actions taken by all agents, while actors should still learn to take the correct action. Failure indicates a problem with the mutli-agent algorithm, or may have other minor causes such as incorrect hyperparameters.
