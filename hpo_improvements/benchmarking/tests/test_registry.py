"""Tests for ``registry`` (normalization baselines and env suites).

``registry`` is pure Python (only ``dataclasses``) so these tests are exhaustive
and fast: they assert the normalization arithmetic, the structural invariants of
the baseline tables, and every branch of the public functions.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

import registry
from registry import (
    ENV_SUITE_NAMES,
    ENV_SUITES,
    MPE_ENV_CONFIGS,
    NormalizationScores,
    allowed_envs,
    env_config,
    env_suite_name,
    normalization_scores,
    resolve_env_selection,
)


# --------------------------------------------------------------------------- #
# NormalizationScores                                                          #
# --------------------------------------------------------------------------- #
class TestNormalizationScores:
    def test_normalize_maps_random_to_zero_and_expert_to_one(self):
        s = NormalizationScores(random=-10.0, expert=90.0)
        assert s.normalize(-10.0) == 0.0
        assert s.normalize(90.0) == 1.0

    def test_normalize_is_affine(self):
        s = NormalizationScores(random=0.0, expert=100.0)
        assert s.normalize(50.0) == pytest.approx(0.5)
        assert s.normalize(200.0) == pytest.approx(2.0)
        assert s.normalize(-50.0) == pytest.approx(-0.5)

    def test_normalize_degenerate_equal_baselines_returns_nan_not_raise(self):
        s = NormalizationScores(random=5.0, expert=5.0)
        assert math.isnan(s.normalize(5.0))
        assert math.isnan(s.normalize(123.0))

    def test_normalize_inverted_baselines_is_monotone(self):
        # Pusher-v4 style: expert < random (both negative).
        s = NormalizationScores(random=-149.7, expert=-25.2)
        assert s.normalize(-149.7) == pytest.approx(0.0)
        assert s.normalize(-25.2) == pytest.approx(1.0)
        # A return below random normalizes negative.
        assert s.normalize(-200.0) < 0.0

    def test_normalize_propagates_nan_input(self):
        s = NormalizationScores(random=0.0, expert=1.0)
        assert math.isnan(s.normalize(float("nan")))

    def test_is_frozen(self):
        s = NormalizationScores(random=0.0, expert=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.random = 5.0  # type: ignore[misc]

    def test_equality_and_hash(self):
        a = NormalizationScores(random=1.0, expert=2.0)
        b = NormalizationScores(random=1.0, expert=2.0)
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


# --------------------------------------------------------------------------- #
# Baseline-table structural invariants                                         #
# --------------------------------------------------------------------------- #
class TestBaselineTables:
    def test_mujoco_suite_is_eleven_v4_envs(self):
        mujoco = ENV_SUITES["PPO"]
        assert len(mujoco) == 11
        assert all(name.endswith("-v4") for name in mujoco)
        assert all(isinstance(v, NormalizationScores) for v in mujoco.values())

    def test_atari_suite_is_full_atari57_v5(self):
        atari = ENV_SUITES["DQN"]
        assert len(atari) == 57
        assert all(name.endswith("-v5") for name in atari)

    def test_mpe_suite_is_three_tasks_with_zero_expert(self):
        mpe = ENV_SUITES["IPPO"]
        assert set(mpe) == {
            "simple_spread_v3",
            "simple_reference_v3",
            "simple_speaker_listener_v4",
        }
        assert all(v.expert == 0.0 for v in mpe.values())
        assert all(v.random < 0.0 for v in mpe.values())

    def test_suite_and_name_keys_match(self):
        assert set(ENV_SUITES) == set(ENV_SUITE_NAMES) == {"PPO", "DQN", "IPPO"}

    def test_mpe_env_configs_env_id_matches_key(self):
        for key, cfg in MPE_ENV_CONFIGS.items():
            assert cfg["env_id"] == key

    def test_mpe_env_configs_keys_match_scores(self):
        assert set(MPE_ENV_CONFIGS) == set(ENV_SUITES["IPPO"])

    def test_spot_check_known_baselines(self):
        ant = ENV_SUITES["PPO"]["Ant-v4"]
        assert ant.random == pytest.approx(-59.9)
        assert ant.expert == pytest.approx(5846.4)
        pong = ENV_SUITES["DQN"]["Pong-v5"]
        assert pong.random == pytest.approx(-20.7)
        assert pong.expert == pytest.approx(14.6)


# --------------------------------------------------------------------------- #
# env_suite_name                                                               #
# --------------------------------------------------------------------------- #
class TestEnvSuiteName:
    @pytest.mark.parametrize(
        "algo,expected", [("PPO", "MuJoCo"), ("DQN", "Atari"), ("IPPO", "MPE")]
    )
    def test_known(self, algo, expected):
        assert env_suite_name(algo) == expected

    def test_unknown_falls_back_to_input(self):
        assert env_suite_name("SAC") == "SAC"
        assert env_suite_name("") == ""

    def test_case_sensitive(self):
        assert env_suite_name("ppo") == "ppo"


# --------------------------------------------------------------------------- #
# allowed_envs                                                                 #
# --------------------------------------------------------------------------- #
class TestAllowedEnvs:
    def test_counts(self):
        assert len(allowed_envs("PPO")) == 11
        assert len(allowed_envs("DQN")) == 57
        assert len(allowed_envs("IPPO")) == 3

    def test_sorted(self):
        envs = allowed_envs("PPO")
        assert envs == sorted(envs)

    def test_returns_fresh_list(self):
        a = allowed_envs("PPO")
        a.append("zzz")
        assert "zzz" not in allowed_envs("PPO")

    @pytest.mark.parametrize("algo", ["SAC", "", "ppo", "IPPO "])
    def test_unknown_algo_raises_keyerror(self, algo):
        with pytest.raises(KeyError, match="No environment suite registered"):
            allowed_envs(algo)


# --------------------------------------------------------------------------- #
# env_config                                                                   #
# --------------------------------------------------------------------------- #
class TestEnvConfig:
    def test_ippo_hit_returns_real_config(self):
        cfg = env_config("IPPO", "simple_spread_v3")
        assert cfg["env_id"] == "simple_spread_v3"
        assert cfg["N"] == 3
        assert cfg["max_cycles"] == 25

    def test_ippo_miss_falls_back_to_env_id(self):
        assert env_config("IPPO", "nonexistent") == {"env_id": "nonexistent"}

    def test_algo_without_config_table_falls_back(self):
        assert env_config("PPO", "Ant-v4") == {"env_id": "Ant-v4"}
        assert env_config("SAC", "X") == {"env_id": "X"}

    def test_ippo_hit_returns_aliased_module_dict(self):
        # Documents a real aliasing hazard: the returned dict IS the module
        # constant, not a copy. Callers must not mutate it.
        cfg = env_config("IPPO", "simple_spread_v3")
        assert cfg is MPE_ENV_CONFIGS["simple_spread_v3"]


# --------------------------------------------------------------------------- #
# normalization_scores                                                         #
# --------------------------------------------------------------------------- #
class TestNormalizationScoresLookup:
    def test_valid_pair(self):
        s = normalization_scores("PPO", "Ant-v4")
        assert isinstance(s, NormalizationScores)
        assert s.random == pytest.approx(-59.9)

    def test_bad_algo_raises(self):
        with pytest.raises(KeyError):
            normalization_scores("SAC", "Ant-v4")

    def test_bad_env_raises(self):
        with pytest.raises(KeyError):
            normalization_scores("PPO", "NotAnEnv-v4")


# --------------------------------------------------------------------------- #
# resolve_env_selection                                                        #
# --------------------------------------------------------------------------- #
class TestResolveEnvSelection:
    @pytest.mark.parametrize("token", ["all", "ALL", " all ", "All"])
    def test_all_returns_full_suite(self, token):
        assert resolve_env_selection("PPO", [token]) == allowed_envs("PPO")

    def test_all_only_special_when_alone(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            resolve_env_selection("PPO", ["all", "Ant-v4"])

    def test_whitespace_stripped_and_empties_filtered(self):
        assert resolve_env_selection("PPO", [" Ant-v4 "]) == ["Ant-v4"]
        assert resolve_env_selection("PPO", ["", "Ant-v4"]) == ["Ant-v4"]

    @pytest.mark.parametrize("selection", [[], [""], ["   "]])
    def test_empty_selection_returns_empty(self, selection):
        assert resolve_env_selection("PPO", selection) == []

    def test_unknown_env_raises_with_message(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            resolve_env_selection("PPO", ["NotReal-v4"])

    def test_output_preserves_suite_order(self):
        out = resolve_env_selection("PPO", ["Hopper-v4", "Ant-v4"])
        assert out == ["Ant-v4", "Hopper-v4"]

    def test_duplicates_collapse(self):
        out = resolve_env_selection("PPO", ["Ant-v4", "Ant-v4"])
        assert out == ["Ant-v4"]

    def test_bad_algo_propagates_keyerror(self):
        with pytest.raises(KeyError):
            resolve_env_selection("SAC", ["all"])


def test_module_has_no_heavy_imports():
    # registry must stay dependency-free (only dataclasses).
    assert not hasattr(registry, "np")
    assert not hasattr(registry, "torch")
