Distributed
===========

Torch-native distributed helpers. AgileRL is single-device by default;
multi-GPU LLM training initialises ``torch.distributed`` from the standard
launcher environment variables set by ``torchrun`` (or an orchestration layer
such as Ray), and these helpers no-op on a single device.

How to launch, when to use data parallel vs FSDP2, and every
:class:`~agilerl.distributed.FSDPConfig` field: :ref:`llm_distributed`.

.. autofunction:: agilerl.distributed.init_distributed
.. autofunction:: agilerl.distributed.is_distributed
.. autofunction:: agilerl.distributed.is_fsdp_sharded
.. autofunction:: agilerl.distributed.get_rank
.. autofunction:: agilerl.distributed.get_local_rank
.. autofunction:: agilerl.distributed.get_world_size
.. autofunction:: agilerl.distributed.is_main_process
.. autofunction:: agilerl.distributed.barrier
.. autofunction:: agilerl.distributed.broadcast_object_list
.. autofunction:: agilerl.distributed.all_reduce_mean
.. autofunction:: agilerl.distributed.gather_tensor
.. autofunction:: agilerl.distributed.gather_objects
.. autofunction:: agilerl.distributed.allreduce_minmax_int
.. autofunction:: agilerl.distributed.aggregate_metrics_across_gpus
.. autofunction:: agilerl.distributed.aggregate_metrics_dict
.. autofunction:: agilerl.distributed.sync_grads
.. autofunction:: agilerl.distributed.materialize_dtensors
.. autofunction:: agilerl.distributed.gather_params
.. autofunction:: agilerl.distributed.full_shape_views
.. autofunction:: agilerl.distributed.resolve_device

.. autoclass:: agilerl.distributed.FSDPConfig
   :members:

.. autofunction:: agilerl.distributed.apply_fsdp2
.. autofunction:: agilerl.distributed.shard_dataloader_kwargs
