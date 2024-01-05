PettingZoo Vectorization Parallel Wrapper
==========================================
The `PettingZooVectorizationParallelWrapper` class is a wrapper that vectorizes the environment,
allowing multiple instances of the environment to be run in parallel.


.. code-block:: python

  from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
  from pettingzoo.atari import space_invaders_v2

  env = space_invaders_v2.parallel_env()
  n_envs = 4
  vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
  observations, infos = vec_env.reset()
  for step in range(25):
      actions = {
          agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
          for agent in vec_env.agents
      }
      observations, rewards, terminations, truncations, infos = vec_env.step(actions)


Parameters
------------

.. autoclass:: agilerl.wrappers.pettingzoo_wrappers.PettingZooVectorizationParallelWrapper
  :members:
  :inherited-members:
