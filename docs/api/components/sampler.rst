Experience Sampler
==================

The ``Sampler`` class provides a unified interface for sampling experiences from replay buffers in both standard and distributed
training scenarios. It can work with various types of replay buffers including standard replay buffers, multi-agent buffers,
prioritized buffers, and multi-step buffers.

For distributed training, the sampler can work with PyTorch DataLoaders and provides custom collate functions to properly handle
TensorDict objects. This enables efficient batch processing of experiences across multiple workers.

The sampler automatically detects whether it's being used in a standard training setup (with a replay buffer) or distributed
setup (with a dataset and dataloader) and provides the appropriate sampling interface.

.. code-block:: python

    from agilerl.components.sampler import Sampler
    from agilerl.components.replay_buffer import ReplayBuffer

    # Standard training setup
    buffer = ReplayBuffer(max_size=10000, device=device)
    sampler = Sampler(memory=buffer)

    # Sample experiences
    batch = sampler.sample(batch_size=32)

    # Distributed training setup
    from agilerl.components.data import ReplayDataset
    dataset = ReplayDataset(buffer, batch_size=32)
    dataloader = Sampler.create_dataloader(dataset, batch_size=32)
    distributed_sampler = Sampler(dataset=dataset, dataloader=dataloader)

Classes
-------

.. autoclass:: agilerl.components.sampler.Sampler
  :members:
  :inherited-members:
