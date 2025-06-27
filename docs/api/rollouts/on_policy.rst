On-Policy Rollout Functions
===========================

These helpers gather transitions from an environment using an on-policy agent.
They fill the agent's rollout buffer so that calling ``agent.learn()`` will
update the policy from the collected data.

.. autofunction:: agilerl.rollouts.collect_rollouts

.. autofunction:: agilerl.rollouts.collect_rollouts_recurrent

Example
-------

Using a non-recurrent PPO agent::

   import gymnasium as gym
   from agilerl.algorithms import PPO
   from agilerl.rollouts import collect_rollouts

   env = gym.make("CartPole-v1")
   agent = PPO(env.observation_space, env.action_space, use_rollout_buffer=True)

   collect_rollouts(agent, env, n_steps=agent.learn_step)
   agent.learn()

For recurrent policies, use ``collect_rollouts_recurrent``::

   num_envs = 4
   env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")] * num_envs)
   agent = PPO(
       env.single_observation_space,
       env.single_action_space,
       use_rollout_buffer=True,
       recurrent=True,
       num_envs=num_envs,
   )

   collect_rollouts_recurrent(agent, env, n_steps=5)
   agent.learn()
