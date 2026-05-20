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
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "lr": 1e-2,
            }

            memory = ReplayBuffer(
                max_size=1000,  # Max replay buffer size
                device=device,
            )

            check_q_learning_with_probe_env(env, DQN, algo_args, memory, learn_steps, device)

    See function docs: :func:`~agilerl.utils.probe_envs.check_q_learning_with_probe_env`

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
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "lr_actor": 1e-2,
                "lr_critic": 1e-2,
            }

            memory = ReplayBuffer(
                max_size=1000,  # Max replay buffer size
                device=device,
            )

            check_policy_q_learning_with_probe_env(
                env, DDPG, algo_args, memory, learn_steps, device
            )

    See function docs: :func:`~agilerl.utils.probe_envs.check_policy_q_learning_with_probe_env`

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
                  "observation_space": env.observation_space,
                  "action_space": env.action_space,
                  "lr": 0.001
              }

              check_policy_on_policy_with_probe_env(
                  env, PPO, algo_args, memory, learn_steps, device
        )

See function docs: :func:`~agilerl.utils.probe_envs.check_policy_on_policy_with_probe_env`


.. collapse:: Multi-agent - Check Policy and Q-learning

    .. code-block:: python

        import torch
        from agilerl.algorithms.maddpg import MADDPG
        from agilerl.components.replay_buffer import MultiAgentReplayBuffer
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
                "observation_spaces": [env.observation_spaces[agent] for agent in env.agents],
                "action_spaces": [env.action_space[agent] for agent in env.agents],
                "agent_ids": env.possible_agents,
                "net_config": {"head_config": {"hidden_size": [32, 32]}},
                "batch_size": 256,
            }
            memory = MultiAgentReplayBuffer(
                max_size=10_000,
                device=device,
            )

            check_policy_q_learning_with_probe_env(env, MADDPG, algo_args, memory, learn_steps, device)

    See function docs: :func:`~agilerl.utils.probe_envs.check_policy_q_learning_with_probe_env`

Debugging with LLM Probe Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For LLM-based agents, use ``agilerl.utils.probe_envs_llm`` to isolate failures in token generation, reward wiring, and multi-turn behavior before running larger tasks.

All probe environments use string observations and string actions. The first three probes are single-turn (``max_turns=1``) and return terminal rewards of ``+1`` or ``-1``. In these probes, reward is positive when the expected target digit appears anywhere in the generated text.

Use the probes in this order:

1. :class:`~agilerl.utils.probe_envs_llm.ConstantTargetEnv`: fixed prompt and fixed target digit. This is a basic end-to-end sanity check.
2. :class:`~agilerl.utils.probe_envs_llm.ConditionalTargetEnv`: one input digit, target ``(digit % 3) + 1``. This checks whether the policy conditions on the observation.
3. :class:`~agilerl.utils.probe_envs_llm.MultiInputConditionalEnv`: two input digits, target ``((d1 + d2) % 3) + 1``. This checks multi-input composition.
4. :class:`~agilerl.utils.probe_envs_llm.GridNavigationEnv`: multi-turn 1D navigation with step costs. The reset observation is ``<start><target>``, later observations are only the current position, and actions are parsed from the first valid digit in generated text (``"1"=left, "2"=stay, "3"=right``).

The ready-to-run staged scripts in ``demos/llm/debugging/`` follow the same progression:

- ``debugging_llm.py`` -> :class:`~agilerl.utils.probe_envs_llm.ConstantTargetEnv`
- ``debugging_llm_stage_1.py`` -> :class:`~agilerl.utils.probe_envs_llm.ConditionalTargetEnv`
- ``debugging_llm_stage_2.py`` -> :class:`~agilerl.utils.probe_envs_llm.MultiInputConditionalEnv`
- ``debugging_llm_stage_3.py`` -> :class:`~agilerl.utils.probe_envs_llm.GridNavigationEnv`

These scripts use slightly different evaluation metrics by stage. Some stages use exact-match accuracy for generated output, which is stricter than the substring-based reward used by the environment.

Minimal environment sanity check:

.. code-block:: python

    from agilerl.utils.probe_envs_llm import ConditionalTargetEnv

    env = ConditionalTargetEnv(seed=0)
    obs, info = env.reset()
    print("obs:", obs, "target:", info["target"])

    # In this env the expected mapping is target = (digit % 3) + 1.
    # Reward is positive if the generated action contains that target digit.
    action = info["target"]
    next_obs, reward, terminated, truncated, _ = env.step(action)
    print(next_obs, reward, terminated, truncated)

When debugging training runs, start with short ``max_steps`` and frequent evaluation, and only move to the next probe after the current one reaches stable, high accuracy.

Single, multi-agent, and LLM probe environments are detailed in the tables below, with links to further documentation.

Single-agent Probe Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50, 50
   :header-rows: 1

   * - **Probe Environment**
     - **Details**
   * - :class:`~agilerl.utils.probe_envs.ConstantRewardEnv`

       :class:`~agilerl.utils.probe_envs.ConstantRewardImageEnv`

       :class:`~agilerl.utils.probe_envs.ConstantRewardContActionsEnv`

       :class:`~agilerl.utils.probe_envs.ConstantRewardContActionsImageEnv`

     - **Constant Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the value of the only observation they see is 1. Failure indicates a problem with the loss of this network.
   * - :class:`~agilerl.utils.probe_envs.ObsDependentRewardEnv`

       :class:`~agilerl.utils.probe_envs.ObsDependentRewardImageEnv`

       :class:`~agilerl.utils.probe_envs.ObsDependentRewardContActionsEnv`

       :class:`~agilerl.utils.probe_envs.ObsDependentRewardContActionsImageEnv`

     - **Observation-dependent Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the reward depends on the simple observation. Failure indicates a problem with the learning of this network.
   * - :class:`~agilerl.utils.probe_envs.DiscountedRewardEnv`

       :class:`~agilerl.utils.probe_envs.DiscountedRewardImageEnv`

       :class:`~agilerl.utils.probe_envs.DiscountedRewardContActionsEnv`

       :class:`~agilerl.utils.probe_envs.DiscountedRewardContActionsImageEnv`

     - **Discounted Reward Environment**

       Agents should be able to learn that the reward depends on the simple observation, and also apply discounting. Failure indicates a problem with reward discounting.
   * - :class:`~agilerl.utils.probe_envs.FixedObsPolicyEnv`

       :class:`~agilerl.utils.probe_envs.FixedObsPolicyImageEnv`

       :class:`~agilerl.utils.probe_envs.FixedObsPolicyContActionsEnv`

       :class:`~agilerl.utils.probe_envs.FixedObsPolicyContActionsImageEnv`

     - **Fixed-observation Policy Environment**

       Isolates the policy/actor network. Agents should be able to learn the reward depends on action taken under the same observation. Failure indicates a problem with policy loss or updates.
   * - :class:`~agilerl.utils.probe_envs.PolicyEnv`

       :class:`~agilerl.utils.probe_envs.PolicyImageEnv`

       :class:`~agilerl.utils.probe_envs.PolicyContActionsEnv`

       :class:`~agilerl.utils.probe_envs.PolicyContActionsImageEnvSimple`

       :class:`~agilerl.utils.probe_envs.PolicyContActionsImageEnv`

     - **Observation-dependent Policy Environment**

       Agents should be able to learn the reward depends on different actions taken under different observations. The value/critic and policy/actor networks work together to learn to solve the environment. The policy network should learn the correct actions to output and the value network should learn the value. With Q-learning, the actor is doing both. Failure indicates a problem with the overall algorithm, batching, or even hyperparameters.


Multi-agent Probe Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 50, 50
   :header-rows: 1

   * - **Probe Environment**
     - **Details**
   * - :class:`~agilerl.utils.probe_envs_ma.ConstantRewardEnv`

       :class:`~agilerl.utils.probe_envs_ma.ConstantRewardImageEnv`

       :class:`~agilerl.utils.probe_envs_ma.ConstantRewardContActionsEnv`

       :class:`~agilerl.utils.probe_envs_ma.ConstantRewardContActionsImageEnv`

     - **Constant Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the value of the only observation they see is 1. Failure indicates a problem with the loss of this network.
   * - :class:`~agilerl.utils.probe_envs_ma.ObsDependentRewardEnv`

       :class:`~agilerl.utils.probe_envs_ma.ObsDependentRewardImageEnv`

       :class:`~agilerl.utils.probe_envs_ma.ObsDependentRewardContActionsEnv`

       :class:`~agilerl.utils.probe_envs_ma.ObsDependentRewardContActionsImageEnv`

     - **Observation-dependent Reward Environment**

       Isolates the value/critic network. Agents should be able to learn that the reward depends on the simple observation. Failure indicates a problem with the learning of this network.
   * - :class:`~agilerl.utils.probe_envs_ma.DiscountedRewardEnv`

       :class:`~agilerl.utils.probe_envs_ma.DiscountedRewardImageEnv`

       :class:`~agilerl.utils.probe_envs_ma.DiscountedRewardContActionsEnv`

       :class:`~agilerl.utils.probe_envs_ma.DiscountedRewardContActionsImageEnv`

     - **Discounted Reward Environment**

       Agents should be able to learn that the reward depends on the simple observation, and also apply discounting. Failure indicates a problem with reward discounting.
   * - :class:`~agilerl.utils.probe_envs_ma.FixedObsPolicyEnv`

       :class:`~agilerl.utils.probe_envs_ma.FixedObsPolicyImageEnv`

       :class:`~agilerl.utils.probe_envs_ma.FixedObsPolicyContActionsEnv`

       :class:`~agilerl.utils.probe_envs_ma.FixedObsPolicyContActionsImageEnv`

     - **Fixed-observation Policy Environment**

       Isolates the policy/actor network. Agents should be able to learn the reward depends on action taken under the same observation. Failure indicates a problem with policy loss or updates.
   * - :class:`~agilerl.utils.probe_envs_ma.PolicyEnv`

       :class:`~agilerl.utils.probe_envs_ma.PolicyImageEnv`

       :class:`~agilerl.utils.probe_envs_ma.PolicyContActionsEnv`

       :class:`~agilerl.utils.probe_envs_ma.PolicyContActionsImageEnv`

     - **Observation-dependent Policy Environment**

       Agents should be able to learn the reward depends on different actions taken under different observations. The value/critic and policy/actor networks work together to learn to solve the environment. The policy network should learn the correct actions to output and the value network should learn the value. With Q-learning, the actor is doing both. Failure indicates a problem with the overall algorithm, batching, or even hyperparameters.
   * - :class:`~agilerl.utils.probe_envs_ma.MultiPolicyEnv`

       :class:`~agilerl.utils.probe_envs_ma.MultiPolicyImageEnv`

     - **Observation-dependent Multi-agent Policy Environment**

       Harder version of Observation-dependent Policy Environment. Critic networks should be able to evaluate a reward dependent on actions taken by all agents, while actors should still learn to take the correct action. Failure indicates a problem with the mutli-agent algorithm, or may have other minor causes such as incorrect hyperparameters.


LLM Probe Environments
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 45, 55
   :header-rows: 1

   * - **Probe Environment**
     - **Details**
   * - :class:`~agilerl.utils.probe_envs_llm.ConstantTargetEnv`
     - **Constant target (single turn).** Fixed prompt and fixed target digit. Reward is ``+1`` when the generated text contains the target digit, otherwise ``-1``. Useful for basic end-to-end wiring checks.
   * - :class:`~agilerl.utils.probe_envs_llm.ConditionalTargetEnv`
     - **Observation-conditioned target (single turn).** Observation is one digit and target is ``(digit % 3) + 1``. Reward is substring-based (target digit appears in generated text). Useful for checking observation conditioning.
   * - :class:`~agilerl.utils.probe_envs_llm.MultiInputConditionalEnv`
     - **Two-input conditional target (single turn).** Observation is two digits and target is ``((d1 + d2) % 3) + 1``. Reward is substring-based. Useful for checking simple compositional reasoning over multiple inputs.
   * - :class:`~agilerl.utils.probe_envs_llm.GridNavigationEnv`
     - **Multi-turn navigation.** Initial observation encodes ``<start><target>``; later observations provide current position only. Actions are parsed from the first valid generated digit (``"1"`` left, ``"2"`` stay, ``"3"`` right). Rewards are ``+1`` on success, ``step_cost`` on intermediate steps, and ``-1`` when max turns are reached without success.
