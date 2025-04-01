Experience Replay Buffer
========================

In order to efficiently train a population of RL agents, off-policy algorithms must be used to share memory within populations. This reduces the exploration needed
by an individual agent because it allows faster learning from the behaviour of other agents. For example, if you were able to watch a bunch of people attempt to solve
a maze, you could learn from their mistakes and successes without necessarily having to explore the entire maze yourself.

The object used to store experiences collected by agents in the environment is called the Experience Replay Buffer, and is defined by the class ``ReplayBuffer()``.
During training we use the ``ReplayBuffer.add()`` function to add experiences to the buffer as ``TensorDict`` objects. Specifically, we wrap transitions through the
``Transition`` tensorclass that wraps the ``obs``, ``action``, ``reward``, ``next_obs``, and ``done`` fields as ``torch.Tensor`` objects. To sample from the replay
buffer, call ``ReplayBuffer.sample()``.

.. code-block:: python

    from agilerl.components.replay_buffer import ReplayBuffer

    memory = ReplayBuffer(
        max_size=10000,  # Max replay buffer size
        device=device,
    )

Parameters
------------

.. autoclass:: agilerl.components.replay_buffer.ReplayBuffer
  :members:
  :inherited-members:

.. autoclass:: agilerl.components.replay_buffer.MultiStepReplayBuffer
  :members:
  :inherited-members:

.. autoclass:: agilerl.components.replay_buffer.PrioritizedReplayBuffer
  :members:
  :inherited-members:
