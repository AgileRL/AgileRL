Inference
=========

HTTP client for deployed Arena inference endpoints. Deployment metadata is
fetched on construction (unless ``probe_on_init=False``) and exposed as
:attr:`~agilerl.arena.inference.Agent.metadata` (:class:`~agilerl.arena.inference.StatusResponse`).
Use :meth:`~agilerl.arena.inference.Agent.get_action` (RL),
:meth:`~agilerl.arena.inference.Agent.predict` (supervised), or
:meth:`~agilerl.arena.inference.Agent.generate` /
:meth:`~agilerl.arena.inference.Agent.generate_stream` (LLM). Errors raise
:class:`~agilerl.arena.exceptions.ArenaInferenceError`.

RL and supervised requests use base64 ``.npy`` serialization via
:mod:`agilerl.arena.inference.serde`. LLM requests use plain JSON with
:class:`~agilerl.arena.inference.LLMParams`.

CLI: ``arena agent run <deployment>`` then ``arena agent generate --prompt '...'``
(uses ``generate_stream``; pass a deployment name to ``generate`` to override).

.. autoclass:: agilerl.arena.inference.LLMParams
   :members:

.. autoclass:: agilerl.arena.inference.StatusResponse
   :members:

.. autoclass:: agilerl.arena.inference.PredictResult
   :members:

.. autoclass:: agilerl.arena.inference.LLMCompletionResult
   :members:

.. autoclass:: agilerl.arena.inference.LLMResults
   :members:

.. autoclass:: agilerl.arena.inference.Agent
   :members:
