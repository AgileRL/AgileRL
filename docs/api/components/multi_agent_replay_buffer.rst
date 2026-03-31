Multi-Agent Experience Replay Buffer
====================================

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``MultiAgentReplayBuffer()`` for
multi-agent environments. It extends ``ReplayBuffer`` with support for nested per-agent TensorDicts. Transitions are built using the ``MultiAgentTransition`` tensorclass
and added via ``memory.add()``, and sampled using ``memory.sample()``.

.. code-block:: python

    from agilerl.components.replay_buffer import MultiAgentReplayBuffer
    import torch

    memory = MultiAgentReplayBuffer(max_size=1_000_000,           # Max replay buffer size
                                    device=torch.device("cuda"))

Parameters
------------

.. autoclass:: agilerl.components.replay_buffer.MultiAgentReplayBuffer
  :members:
  :inherited-members:
