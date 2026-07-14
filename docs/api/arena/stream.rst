Stream
======

Typed event model and NDJSON stream wrapper for Arena streaming endpoints.
Used internally by the client for real-time progress during validation,
profiling, and training submission.

Events
------

.. autoclass:: agilerl.arena.stream.StatusEvent
   :members:

.. autoclass:: agilerl.arena.stream.CheckEvent
   :members:

.. autoclass:: agilerl.arena.stream.ErrorEvent
   :members:

.. autoclass:: agilerl.arena.stream.LogEvent
   :members:

Parsing
-------

.. autofunction:: agilerl.arena.stream.parse_ndjson_line

Stream Iterator
---------------

.. autoclass:: agilerl.arena.stream.NDJsonStream
   :members:
