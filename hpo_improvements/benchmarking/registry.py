"""Per-algorithm environment registries and normalization scores.

The benchmarking script uses this module to (a) tell the user which
environments are valid for the algorithm declared in their YAML config, and
(b) normalize fitness against the random/expert baselines reported in the
literature.

Sources of the random/expert (human) baselines, per algorithm:

* **PPO / MuJoCo** — *"Jack of All Trades, Master of Some, a Multi-Purpose
  Transformer Agent"* (Gallouedec et al., 2024), Appendix A, Table 4.
* **DQN / Atari** — *"Agent57: Outperforming the Atari Human Benchmark"*
  (Badia et al., 2020), Appendix H, Table H.4 ("Atari 57 Table of Scores").
  The "random" score is the ``Random`` column and "expert" is the
  ``Average Human`` column. That table covers the full 57-game "Atari-57" suite.
* **IPPO / cooperative MPE (MPEv2)** — the active IPPO suite. The cooperative MPE
  reward is the (negated) sum of agent-to-landmark distances plus collision/communication
  penalties, so episodic return is always negative and ``0`` is the unreachable
  theoretical optimum (every agent exactly on its target, no collisions, perfect
  communication). We therefore normalize with:
  - ``random`` = **measured**: mean episodic return of a uniform-random policy, summed
    across agents, over 1000 episodes (seed 42, mpe2 1.1.0) on the exact ``mpe2`` config
    in :data:`MPE_ENV_CONFIGS`. This is self-consistent with IPPO's fitness, which is
    also the sum-across-agents return (``test(sum_scores=True)``).
  - ``expert`` = **0.0** (the fixed theoretical optimum above) for every task.
  This was a deliberate choice over literature expert scores: published MAPPO/IPPO MPE
  numbers (e.g. Papoudakis et al. 2021; Yu et al. 2021) were measured on the *legacy*
  MPE (OpenAI / bundled ``pettingzoo.mpe``), whose reward scale differs from Farama's
  ``mpe2`` — so much so that the legacy "expert" for ``simple_spread`` is *more*
  negative than our measured ``mpe2`` random, which would invert the normalization.
  Mixing a measured-``mpe2`` random with a legacy-MPE expert is invalid, so we anchor
  on the scale-stable theoretical optimum instead. Consequence: 0 is unreachable, so
  MPE normalized scores stay below 1.0 and are not directly comparable in *level* to
  the human-normalized DQN/PPO suites — but they remain monotone, cross-env comparable,
  and correctly ordered (0 = random, 1 = perfect).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizationScores:
    """Random and expert baseline scores used to normalize fitness.

    Normalized score = ``(fitness - random) / (expert - random)``.

    :param random: Mean episodic return of a random policy.
    :type random: float
    :param expert: Mean episodic return of the expert policy.
    :type expert: float
    """

    random: float
    expert: float

    def normalize(self, fitness: float) -> float:
        """Return the expert-normalized fitness.

        :param fitness: Raw episodic return.
        :type fitness: float
        :return: Normalized fitness where 0 is random and 1 is expert.
        :rtype: float
        """
        denom = self.expert - self.random
        return (fitness - self.random) / denom if denom != 0 else float("nan")


# JAT paper, Appendix A, Table 4 (mean values; std omitted). Keyed by the
# Gymnasium environment id used in the benchmark configs. The JAT random/expert
# baselines were measured on the **v4** MuJoCo environments, so the ids below
# are pinned to ``-v4`` to keep the normalization consistent with the source.
#
# Caveat: ``Pusher-v4`` is only supported on ``mujoco<3`` (the block to push is
# lighter than air on newer MuJoCo and Gymnasium disabled the model — see
# https://github.com/Farama-Foundation/Gymnasium/issues/950). We train it via
# EnvPool (which ships its own MuJoCo) and save its elite checkpoint, but the
# Gymnasium render env cannot be built under the project's mujoco>=3, so
# rendering is skipped for Pusher (see ``benchmark.render_best_agent``).
_MUJOCO_SCORES: dict[str, NormalizationScores] = {
    "Ant-v4": NormalizationScores(random=-59.9, expert=5846.4),
    "HalfCheetah-v4": NormalizationScores(random=-285.0, expert=7437.8),
    "Hopper-v4": NormalizationScores(random=18.4, expert=1858.7),
    "Humanoid-v4": NormalizationScores(random=122.0, expert=6281.0),
    "HumanoidStandup-v4": NormalizationScores(random=33135.8, expert=273574.2),
    "InvertedDoublePendulum-v4": NormalizationScores(random=57.5, expert=9338.7),
    "InvertedPendulum-v4": NormalizationScores(random=6.1, expert=475.4),
    "Pusher-v4": NormalizationScores(random=-149.7, expert=-25.2),
    "Reacher-v4": NormalizationScores(random=-43.0, expert=-5.7),
    "Swimmer-v4": NormalizationScores(random=0.8, expert=92.2),
    "Walker2d-v4": NormalizationScores(random=2.7, expert=4631.2),
}


# Agent57 paper, Appendix H, Table H.4 ("Atari 57 Table of Scores"): the full
# Atari-57 suite. ``random`` is the ``Random`` column and ``expert`` is the
# ``Average Human`` column. Keyed by the EnvPool environment id (e.g.
# ``"Pong-v5"``); EnvPool's Atari ids are also the names written into the
# manifest's ``environment.name`` and handed to ``envpool.make_gymnasium`` at
# run time (note these differ from Gymnasium's ``ALE/Pong-v5`` namespace ids).
_ATARI_SCORES: dict[str, NormalizationScores] = {
    "Alien-v5": NormalizationScores(random=227.8, expert=7127.7),
    "Amidar-v5": NormalizationScores(random=5.8, expert=1719.5),
    "Assault-v5": NormalizationScores(random=222.4, expert=742.0),
    "Asterix-v5": NormalizationScores(random=210.0, expert=8503.3),
    "Asteroids-v5": NormalizationScores(random=719.1, expert=47388.7),
    "Atlantis-v5": NormalizationScores(random=12850.0, expert=29028.1),
    "BankHeist-v5": NormalizationScores(random=14.2, expert=753.1),
    "BattleZone-v5": NormalizationScores(random=2360.0, expert=37187.5),
    "BeamRider-v5": NormalizationScores(random=363.9, expert=16926.5),
    "Berzerk-v5": NormalizationScores(random=123.7, expert=2630.4),
    "Bowling-v5": NormalizationScores(random=23.1, expert=160.7),
    "Boxing-v5": NormalizationScores(random=0.1, expert=12.1),
    "Breakout-v5": NormalizationScores(random=1.7, expert=30.5),
    "Centipede-v5": NormalizationScores(random=2090.9, expert=12017.0),
    "ChopperCommand-v5": NormalizationScores(random=811.0, expert=7387.8),
    "CrazyClimber-v5": NormalizationScores(random=10780.5, expert=35829.4),
    "Defender-v5": NormalizationScores(random=2874.5, expert=18688.9),
    "DemonAttack-v5": NormalizationScores(random=152.1, expert=1971.0),
    "DoubleDunk-v5": NormalizationScores(random=-18.6, expert=-16.4),
    "Enduro-v5": NormalizationScores(random=0.0, expert=860.5),
    "FishingDerby-v5": NormalizationScores(random=-91.7, expert=-38.7),
    "Freeway-v5": NormalizationScores(random=0.0, expert=29.6),
    "Frostbite-v5": NormalizationScores(random=65.2, expert=4334.7),
    "Gopher-v5": NormalizationScores(random=257.6, expert=2412.5),
    "Gravitar-v5": NormalizationScores(random=173.0, expert=3351.4),
    "Hero-v5": NormalizationScores(random=1027.0, expert=30826.4),
    "IceHockey-v5": NormalizationScores(random=-11.2, expert=0.9),
    "Jamesbond-v5": NormalizationScores(random=29.0, expert=302.8),
    "Kangaroo-v5": NormalizationScores(random=52.0, expert=3035.0),
    "Krull-v5": NormalizationScores(random=1598.0, expert=2665.5),
    "KungFuMaster-v5": NormalizationScores(random=258.5, expert=22736.3),
    "MontezumaRevenge-v5": NormalizationScores(random=0.0, expert=4753.3),
    "MsPacman-v5": NormalizationScores(random=307.3, expert=6951.6),
    "NameThisGame-v5": NormalizationScores(random=2292.3, expert=8049.0),
    "Phoenix-v5": NormalizationScores(random=761.4, expert=7242.6),
    "Pitfall-v5": NormalizationScores(random=-229.4, expert=6463.7),
    "Pong-v5": NormalizationScores(random=-20.7, expert=14.6),
    "PrivateEye-v5": NormalizationScores(random=24.9, expert=69571.3),
    "Qbert-v5": NormalizationScores(random=163.9, expert=13455.0),
    "Riverraid-v5": NormalizationScores(random=1338.5, expert=17118.0),
    "RoadRunner-v5": NormalizationScores(random=11.5, expert=7845.0),
    "Robotank-v5": NormalizationScores(random=2.2, expert=11.9),
    "Seaquest-v5": NormalizationScores(random=68.4, expert=42054.7),
    "Skiing-v5": NormalizationScores(random=-17098.1, expert=-4336.9),
    "Solaris-v5": NormalizationScores(random=1236.3, expert=12326.7),
    "SpaceInvaders-v5": NormalizationScores(random=148.0, expert=1668.7),
    "StarGunner-v5": NormalizationScores(random=664.0, expert=10250.0),
    "Surround-v5": NormalizationScores(random=-10.0, expert=6.5),
    "Tennis-v5": NormalizationScores(random=-23.8, expert=-8.3),
    "TimePilot-v5": NormalizationScores(random=3568.0, expert=5229.2),
    "Tutankham-v5": NormalizationScores(random=11.4, expert=167.6),
    "UpNDown-v5": NormalizationScores(random=533.4, expert=11693.2),
    "Venture-v5": NormalizationScores(random=0.0, expert=1187.5),
    "VideoPinball-v5": NormalizationScores(random=0.0, expert=17667.9),
    "WizardOfWor-v5": NormalizationScores(random=563.5, expert=4756.5),
    "YarsRevenge-v5": NormalizationScores(random=3092.9, expert=54576.9),
    "Zaxxon-v5": NormalizationScores(random=32.5, expert=9173.3),
}


# Cooperative MPE (MPEv2) suite for IPPO. Keyed by the ``mpe2`` task id (also the
# ``env_name`` the benchmark passes to ``run_training`` / ``fetch_and_plot`` and the
# display label on plots). ``expert`` is the fixed theoretical optimum ``0.0`` (see the
# module docstring for the rationale vs literature scores). ``random`` is MEASURED: mean
# episodic return of a uniform-random policy, summed across agents, over 1000 episodes
# (seed 42, mpe2 1.1.0), on the exact MPE_ENV_CONFIGS task. Re-measure and update these
# if MPE_ENV_CONFIGS changes.
_MPE_SCORES: dict[str, NormalizationScores] = {
    "simple_spread_v3": NormalizationScores(random=-79.22, expert=0.0),
    "simple_reference_v3": NormalizationScores(random=-57.25, expert=0.0),
    "simple_speaker_listener_v4": NormalizationScores(random=-80.45, expert=0.0),
}


# Per-MPE-env keyword arguments forwarded to ``mpe_adapter.make_mpe_parallel_env``.
# The benchmark's multi-agent branch merges the selected env's entry into the
# manifest's ``environment.config``, so one config file drives every MPE task.
# ``env_id`` selects the scenario; the rest are the cooperative-task defaults
# (discrete actions are the adapter default, matching the MAPPO setup). ``max_cycles``
# is the per-episode horizon; ``N`` (spread) is the agent/landmark count; ``local_ratio``
# (spread, reference) mixes global vs per-agent reward.
MPE_ENV_CONFIGS: dict[str, dict[str, object]] = {
    "simple_spread_v3": {
        "env_id": "simple_spread_v3",
        "N": 3,
        "local_ratio": 0.5,
        "max_cycles": 25,
    },
    "simple_reference_v3": {
        "env_id": "simple_reference_v3",
        "local_ratio": 0.5,
        "max_cycles": 25,
    },
    "simple_speaker_listener_v4": {
        "env_id": "simple_speaker_listener_v4",
        "max_cycles": 25,
    },
}


# Allowed environment suites per algorithm. PPO (MuJoCo), DQN (Atari) and IPPO
# (cooperative MPE / MPEv2) are wired up.
ENV_SUITES: dict[str, dict[str, NormalizationScores]] = {
    "PPO": _MUJOCO_SCORES,
    "DQN": _ATARI_SCORES,
    "IPPO": _MPE_SCORES,
}


# Human-readable environment-suite names per algorithm, used in plot titles
# (e.g. the aggregate and performance-profile figures).
ENV_SUITE_NAMES: dict[str, str] = {
    "PPO": "MuJoCo",
    "DQN": "Atari",
    "IPPO": "MPE",
}


# Per-algorithm map from ``env_name`` to the custom env-factory kwargs merged into the
# manifest's ``environment.config`` on the multi-agent path. Keyed by algorithm so a
# future suite swap only touches this table, not ``benchmark.py``.
_ENV_CONFIGS: dict[str, dict[str, dict[str, object]]] = {
    "IPPO": MPE_ENV_CONFIGS,
}


def env_suite_name(algo: str) -> str:
    """Return the human-readable environment-suite name for an algorithm.

    :param algo: Algorithm name as declared in the YAML (e.g. ``"PPO"``).
    :type algo: str
    :return: Suite name (e.g. ``"MuJoCo"`` for PPO, ``"Atari"`` for DQN),
        falling back to the algorithm name when no suite name is registered.
    :rtype: str
    """
    return ENV_SUITE_NAMES.get(algo, algo)


def allowed_envs(algo: str) -> list[str]:
    """Return the sorted list of valid environment names for an algorithm.

    :param algo: Algorithm name as declared in the YAML (e.g. ``"PPO"``).
    :type algo: str
    :return: Permitted environment ids.
    :rtype: list[str]
    :raises KeyError: If the algorithm has no registered suite.
    """
    if algo not in ENV_SUITES:
        msg = (
            f"No environment suite registered for algorithm '{algo}'. "
            f"Available: {sorted(ENV_SUITES)}"
        )
        raise KeyError(msg)
    return sorted(ENV_SUITES[algo])


def env_config(algo: str, env_name: str) -> dict[str, object]:
    """Return the custom env-factory kwargs for an algorithm/environment pair.

    Used by the benchmark's multi-agent branch to fill ``environment.config`` for the
    PettingZoo ``entrypoint`` factory (e.g. :func:`mpe_adapter.make_mpe_parallel_env`).
    Falls back to ``{"env_id": env_name}`` when the algorithm has no registered
    per-env config table.

    :param algo: Algorithm name (e.g. ``"IPPO"``).
    :type algo: str
    :param env_name: Environment id.
    :type env_name: str
    :return: Factory kwargs (always includes ``env_id``).
    :rtype: dict[str, object]
    """
    return _ENV_CONFIGS.get(algo, {}).get(env_name, {"env_id": env_name})


def normalization_scores(algo: str, env_name: str) -> NormalizationScores:
    """Return the normalization scores for an algorithm/environment pair.

    :param algo: Algorithm name.
    :type algo: str
    :param env_name: Environment id.
    :type env_name: str
    :return: Random/expert baseline scores.
    :rtype: NormalizationScores
    """
    return ENV_SUITES[algo][env_name]


def resolve_env_selection(algo: str, selection: list[str]) -> list[str]:
    """Resolve a user selection (possibly ``["all"]``) into concrete env names.

    :param algo: Algorithm name.
    :type algo: str
    :param selection: User-entered env names, or ``["all"]``.
    :type selection: list[str]
    :return: Concrete, validated env names in suite order.
    :rtype: list[str]
    :raises ValueError: If any requested env is not in the suite.
    """
    valid = allowed_envs(algo)
    if len(selection) == 1 and selection[0].strip().lower() == "all":
        return valid

    requested = [s.strip() for s in selection if s.strip()]
    unknown = [s for s in requested if s not in valid]
    if unknown:
        msg = f"Unknown environment(s) for {algo}: {unknown}. Valid options: {valid}"
        raise ValueError(msg)
    # Preserve suite ordering for deterministic output.
    return [e for e in valid if e in requested]
