"""Back-compat re-export. ``PreferenceGym`` is now a thin :class:`DatasetEnv` subclass.

It moved to :mod:`agilerl.llm_envs.dataset_env` when the preference / SFT gyms collapsed
into one descriptor-configured ``DatasetEnv``. Import from there going forward.
"""

from __future__ import annotations

from agilerl.llm_envs.dataset_env import PreferenceGym

__all__ = ["PreferenceGym"]
