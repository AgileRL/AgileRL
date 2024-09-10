PettingZoo Vectorization Parallel Wrappers
==========================================
AgileRL has two PettingZoo vectorization wrappers, allowing multiple instances of the environments to run in parallel. The first,
`PettingZooVectorizationParallelWrapper` is to be used with default parallel environments from the PettingZoo library. To vectorize
any of the default PettingZoo envs, the wrapper can be used as follows:

.. code-block:: python
  rom agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
  from pettingzoo.atari import space_invaders_v2

  env = space_invaders_v2
  env_args = {"continuous_actions": True, "max_cycles": 25}
  n_envs = 4
  vec_env = PettingZooVectorizationParallelWrapper(env, n_envs=n_envs, enable_autoreset=True, env_args=env_args)
  observations, infos = vec_env.reset()
  for step in range(25):
      actions = {
          agent: [vec_env.action_space(agent).sample() for n in range(n_envs)]
          for agent in vec_env.agents
      }
      observations, rewards, terminations, truncations, infos = vec_env.step(actions)

The second is for custom Parallel API PettingZoo environments, `CustomPettingZooVectorizationParallelWrapper`, which
should be used for environment classes that subclass `ParallelEnv` and can be used as follows:


.. code-block:: python

  from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper
  from pettingzoo.atari import space_invaders_v2

  env = CustomEnv(custom_arg="custom_arg")
  n_envs = 4
  vec_env = CustomPettingZooVectorizationParallelWrapper(env, n_envs=n_envs)
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

.. autoclass:: agilerl.wrappers.pettingzoo_wrappers.CustomPettingZooVectorizationParallelWrapper
  :members:
  :inherited-members:
