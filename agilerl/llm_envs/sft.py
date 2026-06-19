"""Back-compat re-export. ``SFTGym`` is now a thin :class:`DatasetEnv` subclass.

It moved to :mod:`agilerl.llm_envs.dataset_env` when the preference / SFT gyms collapsed
into one descriptor-configured ``DatasetEnv``. Import from there going forward.
"""

from __future__ import annotations

from agilerl.llm_envs.dataset_env import SFTGym

__all__ = ["SFTGym"]
