Data Structures and Utilities
=============================

This module provides essential data structures and utility functions for handling experiences and datasets in reinforcement learning.
The main components include the ``Transition`` tensorclass for representing environment transitions, the ``ReplayDataset`` for
creating iterable datasets from replay buffers, and utility functions for converting between different data formats.

The ``Transition`` class wraps observations, actions, rewards, next observations, and done flags as a structured data container,
automatically handling conversions between different data types and formats. The ``ReplayDataset`` enables integration with
PyTorch's DataLoader for distributed training scenarios.

.. code-block:: python

    from agilerl.components.data import Transition, ReplayDataset, to_tensordict
    from agilerl.components.replay_buffer import ReplayBuffer

    # Create a transition
    transition = Transition(
        obs=obs,
        action=action,
        reward=reward,
        next_obs=next_obs,
        done=done
    )

    # Create a dataset from a replay buffer
    buffer = ReplayBuffer(max_size=10000, device=device)
    dataset = ReplayDataset(buffer, batch_size=32)

Functions
---------

.. autofunction:: agilerl.components.data.to_tensordict

.. autofunction:: agilerl.components.data.to_torch_tensor

Classes
-------

.. autoclass:: agilerl.components.data.Transition
  :members:
  :inherited-members:

.. autoclass:: agilerl.components.data.ReplayDataset
  :members:
  :inherited-members:
