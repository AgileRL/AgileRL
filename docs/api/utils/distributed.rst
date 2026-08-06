Distributed
===========

Torch-native distributed helpers. AgileRL is single-device by default;
multi-GPU LLM training initialises ``torch.distributed`` from the standard
launcher environment variables set by ``torchrun`` (or an orchestration layer
such as Ray), and these helpers no-op on a single device.

.. autofunction:: agilerl.utils.distributed.init_distributed
.. autofunction:: agilerl.utils.distributed.is_distributed
.. autofunction:: agilerl.utils.distributed.get_rank
.. autofunction:: agilerl.utils.distributed.get_local_rank
.. autofunction:: agilerl.utils.distributed.get_world_size
.. autofunction:: agilerl.utils.distributed.is_main_process
.. autofunction:: agilerl.utils.distributed.barrier
.. autofunction:: agilerl.utils.distributed.broadcast_object_list
.. autofunction:: agilerl.utils.distributed.all_reduce_mean
.. autofunction:: agilerl.utils.distributed.sync_grads
.. autofunction:: agilerl.utils.distributed.resolve_device

.. autoclass:: agilerl.utils.distributed.FSDPConfig
   :members:

.. autofunction:: agilerl.utils.distributed.apply_fsdp2
.. autofunction:: agilerl.utils.distributed.shard_dataloader_kwargs
