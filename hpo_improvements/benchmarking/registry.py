"""Per-algorithm environment registries and normalization scores.

The benchmarking script uses this module to (a) tell the user which
environments are valid for the algorithm declared in their YAML config, and
(b) normalize fitness against the random/expert baselines reported in the
literature.

Sources of the random/expert (human) baselines, per algorithm:

* **PPO / MuJoCo** — *"Jack of All Trades, Master of Some, a Multi-Purpose
  Transformer Agent"* (Gallouedec et al., 2024), Appendix A, Table 4.
* **DQN / Atari** — *"Human-level control through deep reinforcement
  learning"* (Mnih et al., 2015), Extended Data Table 2. The "expert" score is
  the professional human games tester column; "random" is the random-play
  column. That table reports the 49 Atari 2600 games on which DQN was evaluated
  (the larger 57-game "Atari-57" suite comes from later work and is not scored
  in this paper).
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
# Gymnasium environment id used in the benchmark configs.
_MUJOCO_SCORES: dict[str, NormalizationScores] = {
    "Ant-v5": NormalizationScores(random=-59.9, expert=5846.4),
    "HalfCheetah-v5": NormalizationScores(random=-285.0, expert=7437.8),
    "Hopper-v5": NormalizationScores(random=18.4, expert=1858.7),
    "Humanoid-v5": NormalizationScores(random=122.0, expert=6281.0),
    "HumanoidStandup-v5": NormalizationScores(random=33135.8, expert=273574.2),
    "InvertedDoublePendulum-v5": NormalizationScores(random=57.5, expert=9338.7),
    "InvertedPendulum-v5": NormalizationScores(random=6.1, expert=475.4),
    "Pusher-v5": NormalizationScores(random=-149.7, expert=-25.2),
    "Reacher-v5": NormalizationScores(random=-43.0, expert=-5.7),
    "Swimmer-v5": NormalizationScores(random=0.8, expert=92.2),
    "Walker2d-v5": NormalizationScores(random=2.7, expert=4631.2),
}


# Mnih et al. (2015), Extended Data Table 2: the 49 Atari 2600 games DQN was
# evaluated on. ``random`` is the random-play column and ``expert`` is the
# professional human games tester column. Keyed by the EnvPool environment id
# (e.g. ``"Pong-v5"``); EnvPool's Atari ids are also the names written into the
# manifest's ``environment.name`` and handed to ``envpool.make_gymnasium`` at
# run time (note these differ from Gymnasium's ``ALE/Pong-v5`` namespace ids).
_ATARI_SCORES: dict[str, NormalizationScores] = {
    "Alien-v5": NormalizationScores(random=227.8, expert=6875.0),
    "Amidar-v5": NormalizationScores(random=5.8, expert=1676.0),
    "Assault-v5": NormalizationScores(random=222.4, expert=1496.0),
    "Asterix-v5": NormalizationScores(random=210.0, expert=8503.0),
    "Asteroids-v5": NormalizationScores(random=719.1, expert=13157.0),
    "Atlantis-v5": NormalizationScores(random=12850.0, expert=29028.0),
    "BankHeist-v5": NormalizationScores(random=14.2, expert=734.4),
    "BattleZone-v5": NormalizationScores(random=2360.0, expert=37800.0),
    "BeamRider-v5": NormalizationScores(random=363.9, expert=5775.0),
    "Bowling-v5": NormalizationScores(random=23.1, expert=154.8),
    "Boxing-v5": NormalizationScores(random=0.1, expert=4.3),
    "Breakout-v5": NormalizationScores(random=1.7, expert=31.8),
    "Centipede-v5": NormalizationScores(random=2091.0, expert=11963.0),
    "ChopperCommand-v5": NormalizationScores(random=811.0, expert=9882.0),
    "CrazyClimber-v5": NormalizationScores(random=10781.0, expert=35411.0),
    "DemonAttack-v5": NormalizationScores(random=152.1, expert=3401.0),
    "DoubleDunk-v5": NormalizationScores(random=-18.6, expert=-15.5),
    "Enduro-v5": NormalizationScores(random=0.0, expert=309.6),
    "FishingDerby-v5": NormalizationScores(random=-91.7, expert=5.5),
    "Freeway-v5": NormalizationScores(random=0.0, expert=29.6),
    "Frostbite-v5": NormalizationScores(random=65.2, expert=4335.0),
    "Gopher-v5": NormalizationScores(random=257.6, expert=2321.0),
    "Gravitar-v5": NormalizationScores(random=173.0, expert=2672.0),
    "Hero-v5": NormalizationScores(random=1027.0, expert=25763.0),
    "IceHockey-v5": NormalizationScores(random=-11.2, expert=0.9),
    "Jamesbond-v5": NormalizationScores(random=29.0, expert=406.7),
    "Kangaroo-v5": NormalizationScores(random=52.0, expert=3035.0),
    "Krull-v5": NormalizationScores(random=1598.0, expert=2395.0),
    "KungFuMaster-v5": NormalizationScores(random=258.5, expert=22736.0),
    "MontezumaRevenge-v5": NormalizationScores(random=0.0, expert=4367.0),
    "MsPacman-v5": NormalizationScores(random=307.3, expert=15693.0),
    "NameThisGame-v5": NormalizationScores(random=2292.0, expert=4076.0),
    "Pong-v5": NormalizationScores(random=-20.7, expert=9.3),
    "PrivateEye-v5": NormalizationScores(random=24.9, expert=69571.0),
    "Qbert-v5": NormalizationScores(random=163.9, expert=13455.0),
    "Riverraid-v5": NormalizationScores(random=1339.0, expert=13513.0),
    "RoadRunner-v5": NormalizationScores(random=11.5, expert=7845.0),
    "Robotank-v5": NormalizationScores(random=2.2, expert=11.9),
    "Seaquest-v5": NormalizationScores(random=68.4, expert=20182.0),
    "SpaceInvaders-v5": NormalizationScores(random=148.0, expert=1652.0),
    "StarGunner-v5": NormalizationScores(random=664.0, expert=10250.0),
    "Tennis-v5": NormalizationScores(random=-23.8, expert=-8.9),
    "TimePilot-v5": NormalizationScores(random=3568.0, expert=5925.0),
    "Tutankham-v5": NormalizationScores(random=11.4, expert=167.6),
    "UpNDown-v5": NormalizationScores(random=533.4, expert=9082.0),
    "Venture-v5": NormalizationScores(random=0.0, expert=1188.0),
    "VideoPinball-v5": NormalizationScores(random=16257.0, expert=17298.0),
    "WizardOfWor-v5": NormalizationScores(random=563.5, expert=4757.0),
    "Zaxxon-v5": NormalizationScores(random=32.5, expert=9173.0),
}


# Allowed environment suites per algorithm. PPO (MuJoCo) and DQN (Atari) are
# wired up; IPPO (PettingZoo Classic) is added in a later phase.
ENV_SUITES: dict[str, dict[str, NormalizationScores]] = {
    "PPO": _MUJOCO_SCORES,
    "DQN": _ATARI_SCORES,
}


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
