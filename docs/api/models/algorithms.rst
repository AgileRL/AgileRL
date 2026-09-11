Algorithm Specifications
========================

A **training manifest** is a YAML (or JSON) file that describes a full AgileRL
run: which algorithm to train, on which environment, with which network and
hyperparameters. You write it once. The trainer validates it before anything
starts, so a typo or an unknown field fails immediately instead of an hour
into training. The same file can run locally or on Arena. See
:ref:`training_manifests` for the file layout.

The ``algorithm`` section of that file is checked against a **spec**: a
Pydantic model that lists the fields PPO, DQN, GRPO, and the others accept.
Those models live in ``agilerl-arena``. The framework uses them as-is:
:class:`agilerl.models.PPOSpec` *is*
:class:`agilerl.arena.models.algorithms.PPOSpec`. A manifest that validates on
your laptop therefore validates when you submit it to Arena.

A spec is data. It does not construct a torch module or pick a training loop.
:mod:`agilerl.builders` turn a spec into a live algorithm;
:mod:`agilerl.strategies` pick the loop that trains it.

Base Specs
----------

.. autoclass:: agilerl.arena.models.algorithms.AlgorithmSpec
   :members:

.. autoclass:: agilerl.arena.models.algorithms.SingleAgentAlgorithmSpec
   :members:

.. autoclass:: agilerl.arena.models.algorithms.MultiAgentAlgorithmSpec
   :members:

.. autoclass:: agilerl.arena.models.algorithms.LLMAlgorithmSpec
   :members:

Registry
--------

One registry maps names (for example ``"DQN"``) to spec classes. The local
trainer, the manifest, and Arena all read it, so ``algorithm.name`` means the
same thing everywhere.

.. autoclass:: agilerl.arena.models.registry.AlgorithmRegistry
   :members:

.. autodata:: agilerl.arena.models.registry.MANIFEST_REGISTRY

Builders and training strategies
--------------------------------

A **builder** turns a spec into an algorithm object, including values the YAML
cannot hold (a peft ``LoraConfig``, the vLLM dataclass). A **strategy**
chooses which training loop that paradigm uses and the keyword arguments the
loop takes. Runtime-only inputs (a pre-built network, a resolved
hyperparameter config) are arguments to
:meth:`~agilerl.builders.AlgorithmBuilder.build`, passed through from
:class:`~agilerl.training.trainer.LocalTrainer`. Both are looked up from the
spec's paradigm rather than declared on it, so the schema does not depend on
framework classes.

Builders
~~~~~~~~

.. autofunction:: agilerl.builders.select_builder

.. autoclass:: agilerl.builders.AlgorithmBuilder
   :members:

.. autoclass:: agilerl.builders.SingleAgentBuilder
   :members: build

.. autoclass:: agilerl.builders.MultiAgentBuilder
   :members: build

.. autoclass:: agilerl.builders.LLMBuilder
   :members: build

Strategies
~~~~~~~~~~

.. autofunction:: agilerl.strategies.select_strategy

.. autoclass:: agilerl.strategies.TrainingStrategy
   :members:

.. autoclass:: agilerl.strategies.SingleAgentOnPolicyStrategy
.. autoclass:: agilerl.strategies.SingleAgentOffPolicyStrategy
.. autoclass:: agilerl.strategies.OfflineStrategy
.. autoclass:: agilerl.strategies.BanditStrategy
.. autoclass:: agilerl.strategies.MultiAgentOnPolicyStrategy
.. autoclass:: agilerl.strategies.MultiAgentOffPolicyStrategy
.. autoclass:: agilerl.strategies.LLMStrategy
.. autoclass:: agilerl.strategies.LLMRolloutStrategy
.. autoclass:: agilerl.strategies.LLMDatasetStrategy
