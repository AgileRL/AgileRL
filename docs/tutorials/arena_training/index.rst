Training on Arena
=================

`Arena <https://arena.agilerl.com>`_ is AgileRL's managed **RLOps platform** for running
reinforcement learning on cloud infrastructure. It is built for the same evolutionary
training pipeline you use locally with :class:`~agilerl.training.trainer.LocalTrainer`, but runs
jobs on Arena's managed compute so you can train and scale your workloads efficiently without
the headache of setting up a pipeline on a cloud provider or managing a cluster. Arena is
provider-agnostic and gives users the flexibility to train on a variety of compute resources.

The **Arena client** is how you interact with the platform from your machine: upload and validate
custom environments, submit training jobs on scalable compute, and deploy trained agents for inference.

.. seealso::

   :ref:`arena_client` for the full CLI and Python SDK reference (commands, manifests, and API details).

   :ref:`trainers` for how :class:`~agilerl.training.trainer.LocalTrainer` and
   :class:`~agilerl.training.trainer.ArenaTrainer` relate to local and cloud training.

Tutorials
---------

- :doc:`GRPO on GSM8K <grpo_gsm8k>`: Register the **GSM8K** math dataset from Hugging Face, write a
  reward function, fine-tune ``Qwen2.5-0.5B-Instruct`` with GRPO, and deploy the model for inference.

.. toctree::
   :maxdepth: 1
   :hidden:

   GRPO on GSM8K <grpo_gsm8k>
