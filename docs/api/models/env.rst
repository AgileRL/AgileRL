Environment Specifications
==========================

Environment sections use the same schema classes as Arena, re-exported here.
Construction lives beside them as functions: :func:`~agilerl.models.env.make_env`
dispatches to :func:`~agilerl.models.env.make_gym_env`,
:func:`~agilerl.models.env.make_pz_env`,
:func:`~agilerl.models.env.make_bandit_env`, or
:func:`~agilerl.models.env.make_llm_env`. PettingZoo runs use
:class:`~agilerl.arena.models.env.GymEnvSpec` with :func:`~agilerl.models.env.make_pz_env`.

.. autoclass:: agilerl.arena.models.env.GymEnvSpec
   :members:

.. autoclass:: agilerl.arena.models.env.OfflineEnvSpec
   :members:

.. autoclass:: agilerl.arena.models.env.LLMEnvSpec
   :members:

.. autoclass:: agilerl.arena.models.env.BanditEnvSpec
   :members:

.. autofunction:: agilerl.models.env.make_env

.. autofunction:: agilerl.models.env.make_gym_env

.. autofunction:: agilerl.models.env.make_pz_env

.. autofunction:: agilerl.models.env.make_bandit_env

.. autofunction:: agilerl.models.env.make_llm_env

.. autofunction:: agilerl.models.env.make_multiturn_env_factory

.. autofunction:: agilerl.models.env.make_single_env
