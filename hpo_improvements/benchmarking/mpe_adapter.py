"""Adapter exposing Farama's MPEv2 (``mpe2``) cooperative environments to the harness.

The cooperative Multi-Agent Particle Environments (Lowe et al., 2017) are
maintained by the Farama Foundation as the standalone ``mpe2`` package (the
successor to PettingZoo's bundled ``pettingzoo.mpe``). Each scenario already
exposes a standard PettingZoo :class:`~pettingzoo.utils.env.ParallelEnv` via its
``parallel_env(...)`` factory (per-agent observation/reward/done dicts), which is
exactly the contract AgileRL's ``AsyncPettingZooVecEnv`` / IPPO expect -- so no
wrapping is needed; this module is only a thin *dispatcher* that maps the benchmark's
``env_id`` onto the right ``mpe2`` scenario.

:func:`make_mpe_parallel_env` is a **module-level** factory (a closure would not
survive the ``spawn`` pickle round-trip used by the vec-env subprocesses), wired in
from YAML via ``PzEnvSpec``'s ``entrypoint: "mpe_adapter:make_mpe_parallel_env"``.
``import mpe2`` is kept local to the factory so non-MPE runs never import it.

All cooperative MPE tasks are run with ``continuous_actions=False`` (discrete action
spaces), matching the MAPPO setup of Yu et al. (2021) whose hyperparameters drive the
IPPO configs.
"""

from __future__ import annotations

from typing import Any

from pettingzoo.utils.env import ParallelEnv

# Supported cooperative MPE tasks, keyed by the ``env_id`` the benchmark passes
# through ``environment.config`` (also the registry key and the plot label). The
# value is the ``mpe2`` submodule name whose ``parallel_env`` factory is called.
_MPE_SCENARIOS: dict[str, str] = {
    "simple_spread_v3": "simple_spread_v3",
    "simple_reference_v3": "simple_reference_v3",
    "simple_speaker_listener_v4": "simple_speaker_listener_v4",
}


def make_mpe_parallel_env(
    *,
    env_id: str,
    render_mode: str | None = None,
    continuous_actions: bool = False,
    **kwargs: Any,
) -> ParallelEnv:
    """Build a cooperative MPEv2 PettingZoo parallel environment.

    :param env_id: Cooperative MPE task id, one of the keys of
        :data:`_MPE_SCENARIOS` (e.g. ``"simple_spread_v3"``).
    :type env_id: str
    :param render_mode: PettingZoo render mode (``None`` for training, ``"rgb_array"``
        for the best-agent video). Defaults to ``None``.
    :type render_mode: str or None
    :param continuous_actions: Whether to use continuous action spaces. The cooperative
        MAPPO/IPPO setup uses discrete actions, so this defaults to ``False``.
    :type continuous_actions: bool
    :param kwargs: Remaining per-task constructor kwargs forwarded verbatim to the
        scenario's ``parallel_env`` (e.g. ``N``, ``local_ratio``, ``max_cycles``).
    :type kwargs: Any
    :return: A standard PettingZoo parallel environment (per-agent dicts).
    :rtype: pettingzoo.utils.env.ParallelEnv
    :raises ValueError: If ``env_id`` is not a supported cooperative MPE task.
    """
    if env_id not in _MPE_SCENARIOS:
        msg = f"Unknown MPE env_id '{env_id}'. Supported: {sorted(_MPE_SCENARIOS)}"
        raise ValueError(msg)

    # Local import so non-MPE runs (PPO/MuJoCo, DQN/Atari) never pull in mpe2 /
    # pygame-ce. mpe2 exposes each scenario as a submodule (importing the
    # top-level package does not pull them in), so import the submodule explicitly.
    import importlib

    scenario = importlib.import_module(f"mpe2.{_MPE_SCENARIOS[env_id]}")
    return scenario.parallel_env(
        render_mode=render_mode,
        continuous_actions=continuous_actions,
        **kwargs,
    )
