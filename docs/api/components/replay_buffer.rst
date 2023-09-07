Experience Replay Buffer
========================

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``ReplayBuffer()``.
During training it can be added to using the ``ReplayBuffer.save2memory()`` function, or ``ReplayBuffer.save2memoryVectEnvs()`` for vectorized environments (recommended).
To sample from the replay buffer, call ``ReplayBuffer.sample()``.

.. code-block:: python

  from agilerl.components.replay_buffer import ReplayBuffer
  import torch

  field_names = ["state", "action", "reward", "next_state", "done"]
  memory = ReplayBuffer(action_dim=action_dim,    # Number of agent actions
                        memory_size=10000,        # Max replay buffer size
                        field_names=field_names,  # Field names to store in memory
                        device=torch.device("cuda"))


Parameters
------------

.. autoclass:: agilerl.components.replay_buffer.ReplayBuffer
  :members:
  :inherited-members:
