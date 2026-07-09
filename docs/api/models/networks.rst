Network Specifications
======================

Pydantic models that configure the encoder and head architecture for
evolvable networks. The ``encoder_config`` field uses a discriminated
union on the ``arch`` literal so YAML manifests can select the encoder
type with a single key.

Encoder Specs
-------------

.. autoclass:: agilerl.models.networks.MlpSpec
   :members:

.. autoclass:: agilerl.models.networks.CnnSpec
   :members:

.. autoclass:: agilerl.models.networks.LstmSpec
   :members:

.. autoclass:: agilerl.models.networks.SimbaSpec
   :members:

.. autoclass:: agilerl.models.networks.MultiInputSpec
   :members:

Network Specs
-------------

.. autoclass:: agilerl.models.networks.NetworkSpec
   :members:

.. autoclass:: agilerl.models.networks.QNetworkSpec
   :members:

.. autoclass:: agilerl.models.networks.ContinuousQNetworkSpec
   :members:

.. autoclass:: agilerl.models.networks.DeterministicActorSpec
   :members:

.. autoclass:: agilerl.models.networks.StochasticActorSpec
   :members:

.. autoclass:: agilerl.models.networks.ValueNetworkSpec
   :members:

.. autoclass:: agilerl.models.networks.RainbowQNetworkSpec
   :members:

LLM Finetuning Specs
--------------------

Used by the ``network`` section of manifests for LLM algorithms (GRPO, DPO,
etc.) instead of :class:`~agilerl.models.networks.NetworkSpec`.

.. autoclass:: agilerl.models.networks.LoraConfigDict
   :members:

.. autoclass:: agilerl.models.networks.FinetuningNetworkSpec
   :members:
