Multi-Agent Experience Replay Buffer
========================

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``MultiAgentReplayBuffer()`` for
multi-agent environments. During training it can be added to using the ``MultiAgentReplayBuffer.save2memory()`` function and sampled using the  ``MultiAgentReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
    import torch

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(memory_size=1_000_000,          # Max replay buffer size
                                    field_names=field_names,        # Field names to store in memory
                                    agent_ids=INIT_HP['AGENT_IDS'], # ID for each agent
                                    device=torch.device("cuda"))

Parameters
------------

.. autoclass:: agilerl.components.multi_agent_replay_buffer.MultiAgentReplayBuffer
  :members:
  :inherited-members:
