"""Adapter exposing Farama's MPEv2 (``mpe2``) cooperative environments to the harness.

The cooperative Multi-Agent Particle Environments (Lowe et al., 2017) are
maintained by the Farama Foundation as the standalone ``mpe2`` package (the
successor to PettingZoo's bundled ``pettingzoo.mpe``). Each scenario already
exposes a standard PettingZoo :class:`~pettingzoo.utils.env.ParallelEnv` via its
``parallel_env(...)`` factory (per-agent observation/reward/done dicts), which is
the contract AgileRL's ``AsyncPettingZooVecEnv`` / IPPO expect.

:func:`make_mpe_parallel_env` is a **module-level** factory (a closure would not
survive the ``spawn`` pickle round-trip used by the vec-env subprocesses), wired in
from YAML via ``PzEnvSpec``'s ``entrypoint: "mpe_adapter:make_mpe_parallel_env"``.
``import mpe2`` is kept local to the factory so non-MPE runs never import it.

All cooperative MPE tasks are run with ``continuous_actions=False`` (discrete action
spaces), matching the MAPPO setup of Yu et al. (2021) whose hyperparameters drive the
IPPO configs.

Separate (independent) policies
-------------------------------
AgileRL's IPPO groups agents whose ids share a prefix (everything before the final
``"_"``) into a *single parameter-shared* actor/critic. Empirically, that shared-policy
path does not learn the cooperative MPE tasks: training leaves the policy at its
uniform initialisation and fitness stuck at the random baseline, while the *same* tasks
trained with one policy per agent (and single-agent MPE) learn normally. So this
adapter, by default (``separate_policies=True``), renames same-prefix agents
(``agent_0`` -> ``agent_0_0``) to give every agent a unique prefix and hence its **own**
policy -- i.e. fully-independent IPPO (the original "Independent" PPO formulation),
trading parameter-sharing's sample efficiency for a learning signal that actually
works. The fix lives entirely in this adapter; no AgileRL change is required. Agents
that already have a unique prefix (e.g. ``speaker_0`` / ``listener_0`` in
Simple-Speaker-Listener) are left untouched.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

# Supported cooperative MPE tasks, keyed by the ``env_id`` the benchmark passes
# through ``environment.config`` (also the registry key and the plot label). The
# value is the ``mpe2`` submodule name whose ``parallel_env`` factory is called.
_MPE_SCENARIOS: dict[str, str] = {
    "simple_spread_v3": "simple_spread_v3",
    "simple_reference_v3": "simple_reference_v3",
    "simple_speaker_listener_v4": "simple_speaker_listener_v4",
}


class _SeparatePoliciesWrapper(BaseParallelWrapper):
    """Rename same-prefix (homogeneous) agents so each gets its own IPPO policy.

    AgileRL groups agents sharing a name prefix (the part before the final ``"_"``)
    into one parameter-shared policy. To force a *separate* policy per agent we append
    a suffix to every agent that shares its prefix with another -- e.g. ``agent_0`` ->
    ``agent_0_0`` (new prefix ``agent_0``), ``agent_1`` -> ``agent_1_0`` -- so each
    ends up with a unique prefix. Agents already alone in their prefix (e.g.
    ``speaker_0``, ``listener_0``) are left unchanged. See the module docstring for why
    separate policies are used.

    Defined at module level (not a closure) so it survives the ``spawn`` pickle
    round-trip used by the vectorized PettingZoo subprocesses.
    """

    def __init__(self, env: ParallelEnv) -> None:
        super().__init__(env)
        counts = Counter(a.rsplit("_", 1)[0] for a in env.possible_agents)
        # Only rename agents whose prefix is shared (would otherwise be grouped).
        self._to_new = {
            a: (f"{a}_0" if counts[a.rsplit("_", 1)[0]] > 1 else a)
            for a in env.possible_agents
        }
        self._to_old = {new: old for old, new in self._to_new.items()}

    @property
    def possible_agents(self) -> list:
        return [self._to_new[a] for a in self.env.possible_agents]

    @property
    def agents(self) -> list:
        return [self._to_new[a] for a in self.env.agents]

    def observation_space(self, agent: str):
        return self.env.observation_space(self._to_old[agent])

    def action_space(self, agent: str):
        return self.env.action_space(self._to_old[agent])

    def reset(self, seed=None, options=None) -> tuple:
        obs, info = self.env.reset(seed=seed, options=options)
        return (
            {self._to_new[k]: v for k, v in obs.items()},
            {self._to_new[k]: v for k, v in info.items()},
        )

    def step(self, actions: dict) -> tuple:
        obs, rew, term, trunc, info = self.env.step(
            {self._to_old[k]: v for k, v in actions.items()}
        )
        rename = lambda d: {self._to_new[k]: v for k, v in d.items()}  # noqa: E731
        return rename(obs), rename(rew), rename(term), rename(trunc), rename(info)


def make_mpe_parallel_env(
    *,
    env_id: str,
    render_mode: str | None = None,
    continuous_actions: bool = False,
    separate_policies: bool = True,
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
    :param separate_policies: If ``True`` (the default), homogeneous (same-prefix)
        agents are renamed via :class:`_SeparatePoliciesWrapper` so each trains its own
        IPPO policy instead of a single shared one (AgileRL's shared-policy path does
        not learn these tasks; see the module docstring). Set ``False`` to keep native
        agent ids and AgileRL's parameter-shared grouping.
    :type separate_policies: bool
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
    env = scenario.parallel_env(
        render_mode=render_mode,
        continuous_actions=continuous_actions,
        **kwargs,
    )
    if separate_policies:
        env = _SeparatePoliciesWrapper(env)
    return env
