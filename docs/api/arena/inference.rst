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

Every request carries your own Arena credential, a profile personal access token
(``arena_pat_...``) or an access token from
:meth:`~agilerl.arena.client.ArenaClient.login`. A deployment that keeps memory per
user works out who is calling from it, so there is nothing extra to pass.
:meth:`~agilerl.arena.client.ArenaClient.open_inference_agent` fills the credential
in from the client. Building an :class:`~agilerl.arena.inference.Agent` directly
reads ``ARENA_API_KEY`` when you pass no ``api_key``. The credential is never
logged, shown in ``repr()``, or written to disk.

Pass ``session_id`` to :meth:`~agilerl.arena.inference.Agent.generate` or
:meth:`~agilerl.arena.inference.Agent.generate_stream` to continue an earlier
chat, and use :meth:`~agilerl.arena.inference.Agent.list_sessions` and
:meth:`~agilerl.arena.inference.Agent.get_session` to find one. Sessions come
back most recently updated first. Leaving ``session_id`` out sends no session
field, so the deployment starts a new conversation; no id is minted here. The
CLI does mint one, which is how it continues a conversation across commands.

:meth:`~agilerl.arena.inference.Agent.delete_session` removes a conversation and
its messages from the deployment. It reports whether there was anything to
delete rather than raising, since a session that is already gone and one that
belongs to someone else answer the same way.

Which sessions you can see, and which you can delete, depends on the
deployment's memory scope, set with ``memory_scope`` on
:meth:`~agilerl.arena.client.ArenaClient.deploy_agent`.

CLI: ``arena agent run <deployment>`` then ``arena agent generate --prompt '...'``
(uses ``generate_stream``; pass a deployment name to ``generate`` to override).
Add ``--session-id`` to continue a chat. ``arena agent sessions list`` and
``arena agent sessions get <session-id>`` show what is stored. Prompts continue
one conversation by default, so the id need not be repeated;
``arena agent sessions resume`` switches to an older one,
``arena agent sessions clear`` starts a new one on the next prompt, and
``arena agent sessions delete <session-id>`` removes one from the deployment.

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

.. autoclass:: agilerl.arena.inference.SessionInfo
   :members:

.. autoclass:: agilerl.arena.inference.SessionMessage
   :members:

.. autoclass:: agilerl.arena.inference.SessionDetail
   :members:

.. autoclass:: agilerl.arena.inference.Agent
   :members:
