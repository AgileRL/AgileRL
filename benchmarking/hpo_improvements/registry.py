"""Per-algorithm environment registries and normalization scores.

The benchmarking script uses this module to (a) tell the user which
environments are valid for the algorithm declared in their YAML config, and
(b) normalize fitness against the random/expert baselines reported in
*"Jack of All Trades, Master of Some, a Multi-Purpose Transformer Agent"*
(Gallouedec et al., 2024), Appendix A, Table 4.
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


# Allowed environment suites per algorithm. Only PPO is wired up for now;
# DQN (Atari) and IPPO (PettingZoo Classic) are added in later phases.
ENV_SUITES: dict[str, dict[str, NormalizationScores]] = {
    "PPO": _MUJOCO_SCORES,
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
