"""Tests for ``mpe_adapter`` (MPEv2 cooperative-env adapter).

The separate-policies rename logic is tested directly against a lightweight fake
``ParallelEnv``; the factory's success branch is tested by injecting a fake
``mpe2.<scenario>`` module so no real ``mpe2``/pygame is needed. One optional
smoke test exercises the real ``mpe2`` env when installed.
"""

from __future__ import annotations

import sys
import types

import pytest
from pettingzoo.utils.env import ParallelEnv

from mpe_adapter import _MPE_SCENARIOS, _SeparatePoliciesWrapper, make_mpe_parallel_env


class _FakeParallelEnv(ParallelEnv):
    """Minimal duck-typed ParallelEnv recording the actions it receives."""

    metadata = {"name": "fake"}

    def __init__(self, agents):
        self.possible_agents = list(agents)
        self.agents = list(agents)
        self.received_actions = None
        self.reset_args = None

    def observation_space(self, agent):
        return ("obs", agent)

    def action_space(self, agent):
        return ("act", agent)

    def reset(self, seed=None, options=None):
        self.reset_args = (seed, options)
        obs = {a: f"o_{a}" for a in self.agents}
        info = {a: {"i": a} for a in self.agents}
        return obs, info

    def step(self, actions):
        self.received_actions = actions
        obs = {a: f"o_{a}" for a in self.agents}
        rew = {a: 1.0 for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: False for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, rew, term, trunc, info


# --------------------------------------------------------------------------- #
# _SeparatePoliciesWrapper                                                     #
# --------------------------------------------------------------------------- #
class TestSeparatePoliciesWrapper:
    def test_homogeneous_agents_renamed(self):
        # All three share prefix "agent" -> each gets a "_0" suffix.
        env = _FakeParallelEnv(["agent_0", "agent_1", "agent_2"])
        w = _SeparatePoliciesWrapper(env)
        assert w.possible_agents == ["agent_0_0", "agent_1_0", "agent_2_0"]

    def test_unique_prefix_agents_unchanged(self):
        env = _FakeParallelEnv(["speaker_0", "listener_0"])
        w = _SeparatePoliciesWrapper(env)
        assert w.possible_agents == ["speaker_0", "listener_0"]

    def test_mixed_prefixes(self):
        env = _FakeParallelEnv(["agent_0", "agent_1", "speaker_0"])
        w = _SeparatePoliciesWrapper(env)
        assert w.possible_agents == ["agent_0_0", "agent_1_0", "speaker_0"]

    def test_agents_property_reflects_active(self):
        env = _FakeParallelEnv(["agent_0", "agent_1"])
        env.agents = ["agent_0"]  # only one still active
        w = _SeparatePoliciesWrapper(env)
        assert w.agents == ["agent_0_0"]

    def test_space_translation_new_to_old(self):
        env = _FakeParallelEnv(["agent_0", "agent_1"])
        w = _SeparatePoliciesWrapper(env)
        assert w.observation_space("agent_0_0") == ("obs", "agent_0")
        assert w.action_space("agent_1_0") == ("act", "agent_1")

    def test_space_with_unknown_id_raises(self):
        env = _FakeParallelEnv(["agent_0", "agent_1"])
        w = _SeparatePoliciesWrapper(env)
        with pytest.raises(KeyError):
            w.observation_space("agent_0")  # old id no longer valid

    def test_reset_renames_keys_and_forwards_seed(self):
        env = _FakeParallelEnv(["agent_0", "agent_1"])
        w = _SeparatePoliciesWrapper(env)
        obs, info = w.reset(seed=7, options={"x": 1})
        assert set(obs) == {"agent_0_0", "agent_1_0"}
        assert obs["agent_0_0"] == "o_agent_0"
        assert env.reset_args == (7, {"x": 1})

    def test_step_translates_actions_and_renames_returns(self):
        env = _FakeParallelEnv(["agent_0", "agent_1"])
        w = _SeparatePoliciesWrapper(env)
        obs, rew, term, trunc, info = w.step({"agent_0_0": 1, "agent_1_0": 2})
        # Wrapped env received OLD ids.
        assert env.received_actions == {"agent_0": 1, "agent_1": 2}
        # All five returned dicts use NEW ids.
        for d in (obs, rew, term, trunc, info):
            assert set(d) == {"agent_0_0", "agent_1_0"}

    def test_step_unknown_action_key_raises(self):
        env = _FakeParallelEnv(["agent_0", "agent_1"])
        w = _SeparatePoliciesWrapper(env)
        with pytest.raises(KeyError):
            w.step({"agent_0": 1})  # old id not accepted


# --------------------------------------------------------------------------- #
# make_mpe_parallel_env                                                        #
# --------------------------------------------------------------------------- #
class TestMakeMpeParallelEnv:
    def test_unknown_env_id_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown MPE env_id"):
            make_mpe_parallel_env(env_id="not_a_scenario")

    @pytest.mark.parametrize("bad", ["", "simple_spread_v2"])
    def test_unknown_variants(self, bad):
        with pytest.raises(ValueError, match="Supported"):
            make_mpe_parallel_env(env_id=bad)

    def test_keyword_only(self):
        with pytest.raises(TypeError):
            make_mpe_parallel_env("simple_spread_v3")  # type: ignore[misc]

    def test_requires_env_id(self):
        with pytest.raises(TypeError):
            make_mpe_parallel_env()  # type: ignore[call-arg]

    def _inject_fake_scenario(self, monkeypatch, recorder):
        fake_mod = types.ModuleType("mpe2.simple_spread_v3")

        def parallel_env(**kwargs):
            recorder["kwargs"] = kwargs
            return _FakeParallelEnv(["agent_0", "agent_1"])

        fake_mod.parallel_env = parallel_env
        monkeypatch.setitem(sys.modules, "mpe2.simple_spread_v3", fake_mod)

    def test_success_wraps_by_default(self, monkeypatch):
        recorder = {}
        self._inject_fake_scenario(monkeypatch, recorder)
        env = make_mpe_parallel_env(env_id="simple_spread_v3", N=3, max_cycles=25)
        assert isinstance(env, _SeparatePoliciesWrapper)
        # render_mode/continuous_actions passed explicitly, kwargs forwarded.
        assert recorder["kwargs"]["render_mode"] is None
        assert recorder["kwargs"]["continuous_actions"] is False
        assert recorder["kwargs"]["N"] == 3
        assert recorder["kwargs"]["max_cycles"] == 25

    def test_separate_policies_false_returns_raw_env(self, monkeypatch):
        recorder = {}
        self._inject_fake_scenario(monkeypatch, recorder)
        env = make_mpe_parallel_env(env_id="simple_spread_v3", separate_policies=False)
        assert isinstance(env, _FakeParallelEnv)
        assert not isinstance(env, _SeparatePoliciesWrapper)

    def test_render_and_continuous_plumbed(self, monkeypatch):
        recorder = {}
        self._inject_fake_scenario(monkeypatch, recorder)
        make_mpe_parallel_env(
            env_id="simple_spread_v3",
            render_mode="rgb_array",
            continuous_actions=True,
        )
        assert recorder["kwargs"]["render_mode"] == "rgb_array"
        assert recorder["kwargs"]["continuous_actions"] is True


def test_scenarios_consistent_with_registry():
    import registry

    assert set(_MPE_SCENARIOS) == set(registry.MPE_ENV_CONFIGS)
    # key == submodule name for every scenario.
    assert all(k == v for k, v in _MPE_SCENARIOS.items())


@pytest.mark.slow
def test_real_mpe2_smoke():
    pytest.importorskip("mpe2")
    env = make_mpe_parallel_env(env_id="simple_spread_v3", N=3, max_cycles=10)
    # simple_spread agents share the "agent" prefix -> renamed.
    assert all(a.endswith("_0") for a in env.possible_agents)
    obs, info = env.reset(seed=0)
    assert set(obs) == set(env.possible_agents)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rew, term, trunc, info = env.step(actions)
    assert set(rew) == set(env.possible_agents)
