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

.. note::

   Arena trains LLMs. Classic RL algorithms such as PPO, DQN and TD3 are no longer available on
   Arena; train them with open-source AgileRL, starting from the :ref:`Gymnasium tutorials <Gymnasium_tutorials>`.

.. seealso::

   :ref:`arena_client` for the full CLI and Python SDK reference (commands, manifests, and API details).

   :ref:`trainers` for how :class:`~agilerl.training.trainer.LocalTrainer` and
   :class:`~agilerl.training.trainer.ArenaTrainer` relate to local and cloud training.

Tutorials
---------

- :doc:`GRPO on GSM8K <grpo_gsm8k>`: Register the **GSM8K** math dataset from Hugging Face, write a
  reward function, fine-tune ``Qwen/Qwen3.5-0.8B`` with GRPO, and deploy the model for inference.
- :doc:`GRPO with an LLM judge <grpo_llm_judge>`: Store a judge's API key as an organization
  secret, combine a correctness check with an LLM judge in a rubric, watch the judge's metrics
  and notes, and try RULER, which scores a whole group in one judge call.

.. toctree::
   :maxdepth: 1
   :hidden:

   GRPO on GSM8K <grpo_gsm8k>
   GRPO with an LLM judge <grpo_llm_judge>
