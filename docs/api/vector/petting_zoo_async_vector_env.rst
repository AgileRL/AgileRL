Petting Zoo Async Vector Environment
====================================

Class for vectorizing pettingzoo parallel environments, for both custom and default pettingzoo parallel environments.

.. code-block:: python

  # Default pettingzoo environment
  from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
  from pettingzoo.mpe import simple_speaker_listener_v4

  num_envs = 4
  env = simple_speaker_listener_v4.parallel_env()
  vec_env = AsyncPettingZooVecEnv([lambda : env for _ in range(num_envs)])
  observations, infos = vec_env.reset()
  for step in range(25):
      actions = {
          agent: [vec_env.single_action_space(agent).sample() for n in range(num_envs)]
          for agent in vec_env.agents
      }
      observations, rewards, terminations, truncations, infos = vec_env.step(actions)



.. code-block:: python

    # Custom pettingzoo environment
    from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
    env = CustomEnv()
    num_envs = 4
    AsyncPettingZooVecEnv([lambda : env for _ in range(num_envs)])
    observations, infos = vec_env.reset()
    for step in range(25):
        actions = {
            agent: [vec_env.single_action_space(agent).sample() for n in range(num_envs)]
            for agent in vec_env.agents
        }
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)


Parameters
----------

.. autoclass:: agilerl.vector.pz_async_vec_env.AsyncPettingZooVecEnv
  :members:

.. autoclass:: agilerl.vector.pz_async_vec_env.AsyncState
  :members:

.. autoclass:: agilerl.vector.pz_async_vec_env.PettingZooExperienceSpec
  :members:

.. autoclass:: agilerl.vector.pz_async_vec_env.Observations
  :members:

.. autoclass:: agilerl.vector.pz_async_vec_env.SharedMemory
  :members:

.. autofunction:: agilerl.vector.pz_async_vec_env._async_worker
