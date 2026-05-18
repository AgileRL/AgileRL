Exceptions
==========

Custom exception hierarchy for Arena client errors. Provides structured error
messages with SDK/CLI hints and automatic internal-URL sanitization.

Base
----

.. autoclass:: agilerl.arena.exceptions.ArenaError
   :members:

Authentication Errors
---------------------

.. autoclass:: agilerl.arena.exceptions.ArenaAuthError
   :members:

.. autoclass:: agilerl.arena.exceptions.ArenaTimeoutError
   :members:

API Errors
----------

.. autoclass:: agilerl.arena.exceptions.ArenaAPIError
   :members:

.. autoclass:: agilerl.arena.exceptions.ArenaValidationError
   :members:

.. autoclass:: agilerl.arena.exceptions.ArenaTrainingError
   :members:
