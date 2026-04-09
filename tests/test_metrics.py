"""Tests for agilerl.metrics — AgentMetrics and MultiAgentMetrics."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from agilerl.metrics import AgentMetrics, MultiAgentMetrics


class TestAgentMetrics:
    def test_register_and_log(self):
        m = AgentMetrics()
        m.register("loss")
        m.log("loss", 0.1)
        m.log("loss", 0.3)
        assert m.get_mean("loss") == pytest.approx(0.2)

    def test_register_duplicate_raises(self):
        m = AgentMetrics()
        m.register("loss")
        with pytest.raises(ValueError, match="already registered"):
            m.register("loss")

    def test_get_mean_empty_returns_nan(self):
        m = AgentMetrics()
        m.register("loss")
        assert math.isnan(m.get_mean("loss"))

    def test_additional_metrics_property(self):
        m = AgentMetrics()
        assert m.additional_metrics == []
        m.register("loss")
        m.register("entropy")
        assert m.additional_metrics == ["loss", "entropy"]

    def test_clear_resets_metrics_and_scores(self):
        m = AgentMetrics()
        m.register("loss")
        m.log("loss", 1.0)
        m.add_scores([5.0, 6.0])

        m.clear()

        assert m.scores == []
        assert math.isnan(m.get_mean("loss"))

    def test_add_fitness_and_window(self):
        m = AgentMetrics(fitness_window=3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            m.add_fitness(v)
        assert list(m.fitness) == [2.0, 3.0, 4.0]

    def test_add_scores(self):
        m = AgentMetrics()
        m.add_scores([1.0, 2.0])
        m.add_scores([3.0])
        assert m.scores == [1.0, 2.0, 3.0]

    def test_increment_steps(self):
        m = AgentMetrics()
        m.increment_steps(10)
        m.increment_steps(5)
        assert m.steps == 15

    def test_init_and_finalize_evo_step(self):
        m = AgentMetrics()
        m.init_evo_step()
        time.sleep(0.05)
        m.finalize_evo_step(100)

        assert m.steps == 100
        assert m.steps_per_second > 0


class TestAgentMetricsNonScalar:
    def test_register_histogram_and_log(self):
        m = AgentMetrics()
        m.register_histogram("action_dist")
        assert "action_dist" in m.nonscalar_metrics
        assert m.get_histogram("action_dist") is None

        arr = np.array([10, 20, 30])
        m.log_histogram("action_dist", arr)
        result = m.get_histogram("action_dist")
        np.testing.assert_array_equal(result, arr)

    def test_log_histogram_accumulates_across_calls(self):
        m = AgentMetrics()
        m.register_histogram("action_dist")
        m.log_histogram("action_dist", np.array([1, 2]))
        m.log_histogram("action_dist", np.array([3, 4, 5]))
        result = m.get_histogram("action_dist")
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_register_histogram_duplicate_raises(self):
        m = AgentMetrics()
        m.register_histogram("h")
        with pytest.raises(ValueError, match="already registered"):
            m.register_histogram("h")

    def test_name_collision_scalar_then_histogram(self):
        m = AgentMetrics()
        m.register("loss")
        with pytest.raises(ValueError, match="already registered"):
            m.register_histogram("loss")

    def test_name_collision_histogram_then_scalar(self):
        m = AgentMetrics()
        m.register_histogram("loss")
        with pytest.raises(ValueError, match="already registered"):
            m.register("loss")

    def test_nonscalar_metrics_property(self):
        m = AgentMetrics()
        assert m.nonscalar_metrics == []
        m.register_histogram("a")
        m.register_histogram("b")
        assert m.nonscalar_metrics == ["a", "b"]

    def test_clear_resets_nonscalar(self):
        m = AgentMetrics()
        m.register_histogram("h")
        m.log_histogram("h", np.array([1, 2, 3]))
        m.clear()
        assert m.get_histogram("h") is None

    def test_scalar_and_nonscalar_independent(self):
        m = AgentMetrics()
        m.register("loss")
        m.register_histogram("action_dist")
        m.log("loss", 0.5)
        m.log_histogram("action_dist", np.array([1, 2]))
        assert m.additional_metrics == ["loss"]
        assert m.nonscalar_metrics == ["action_dist"]
        assert m.get_mean("loss") == pytest.approx(0.5)
        np.testing.assert_array_equal(m.get_histogram("action_dist"), [1, 2])


class TestMultiAgentMetrics:
    AGENT_IDS = ["agent_0", "agent_1"]

    def test_register_creates_per_agent_dicts(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register("loss")
        raw = m._additional_metrics["loss"]
        assert set(raw.keys()) == {"agent_0", "agent_1"}
        assert all(v == [] for v in raw.values())

    def test_log_and_get_mean(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register("loss")
        m.log("loss", "agent_0", 0.2)
        m.log("loss", "agent_0", 0.4)
        m.log("loss", "agent_1", 1.0)

        assert m.get_mean("loss", "agent_0") == pytest.approx(0.3)
        assert m.get_mean("loss", "agent_1") == pytest.approx(1.0)

    def test_get_mean_empty_per_agent(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register("loss")
        m.log("loss", "agent_0", 1.0)
        assert math.isnan(m.get_mean("loss", "agent_1"))

    def test_clear_resets_per_agent_dicts(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register("loss")
        m.log("loss", "agent_0", 1.0)
        m.log("loss", "agent_1", 2.0)
        m.add_scores([5.0])

        m.clear()

        assert m.scores == []
        assert math.isnan(m.get_mean("loss", "agent_0"))
        assert math.isnan(m.get_mean("loss", "agent_1"))

    def test_agent_ids_property(self):
        m = MultiAgentMetrics(["a", "b", "c"])
        assert m.agent_ids == ["a", "b", "c"]


class TestMultiAgentMetricsNonScalar:
    AGENT_IDS = ["agent_0", "agent_1"]

    def test_register_histogram_creates_per_agent(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register_histogram("action_dist")
        raw = m._nonscalar_metrics["action_dist"]
        assert set(raw.keys()) == {"agent_0", "agent_1"}
        assert all(v == [] for v in raw.values())

    def test_log_and_get_histogram(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register_histogram("action_dist")
        arr = np.array([5, 10, 15])
        m.log_histogram("action_dist", "agent_0", arr)

        np.testing.assert_array_equal(m.get_histogram("action_dist", "agent_0"), arr)
        assert m.get_histogram("action_dist", "agent_1") is None

    def test_clear_resets_nonscalar_per_agent(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register_histogram("h")
        m.log_histogram("h", "agent_0", np.array([1, 2]))
        m.clear()
        assert m.get_histogram("h", "agent_0") is None
        assert m.get_histogram("h", "agent_1") is None

    def test_log_histogram_accumulates_across_calls(self):
        m = MultiAgentMetrics(self.AGENT_IDS)
        m.register_histogram("action_dist")
        m.log_histogram("action_dist", "agent_0", np.array([1, 2]))
        m.log_histogram("action_dist", "agent_0", np.array([3, 4, 5]))
        result = m.get_histogram("action_dist", "agent_0")
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])


# ---------------------------------------------------------------------------
# Edge cases: __eq__, __exit__, error paths
# ---------------------------------------------------------------------------


class TestBaseMetricsEquality:
    def test_equal_instances(self):
        a = AgentMetrics()
        b = AgentMetrics()
        a.register("loss")
        b.register("loss")
        a.log("loss", 0.5)
        b.log("loss", 0.5)
        a.add_fitness(1.0)
        b.add_fitness(1.0)
        a.add_scores([10.0])
        b.add_scores([10.0])
        a.increment_steps(50)
        b.increment_steps(50)
        assert a == b

    def test_unequal_different_fitness(self):
        a = AgentMetrics()
        b = AgentMetrics()
        a.add_fitness(1.0)
        b.add_fitness(2.0)
        assert a != b

    def test_unequal_different_steps(self):
        a = AgentMetrics()
        b = AgentMetrics()
        a.increment_steps(10)
        b.increment_steps(20)
        assert a != b

    def test_unequal_different_scores(self):
        a = AgentMetrics()
        b = AgentMetrics()
        a.add_scores([1.0])
        b.add_scores([2.0])
        assert a != b

    def test_unequal_different_metrics(self):
        a = AgentMetrics()
        b = AgentMetrics()
        a.register("loss")
        b.register("loss")
        a.log("loss", 0.1)
        b.log("loss", 0.9)
        assert a != b

    def test_comparison_with_non_metrics_returns_not_implemented(self):
        m = AgentMetrics()
        assert m.__eq__("not a metrics") is NotImplemented

    def test_cross_type_comparison(self):
        a = AgentMetrics()
        b = MultiAgentMetrics(["x"])
        # Both are BaseMetrics subclasses, so __eq__ runs and compares state.
        # Fresh instances with no data differ because MultiAgentMetrics
        # has a different scores type annotation, but BaseMetrics.__eq__
        # only checks list equality — so empty vs empty compares equal.
        # The key difference is _additional_metrics structure after register.
        a.register("m")
        b.register("m")
        # After register, internal structures differ: list vs dict-of-lists.
        assert a != b


class TestBaseMetricsContextManager:
    def test_exit_calls_clear(self):
        m = AgentMetrics()
        m.register("loss")
        m.log("loss", 1.0)
        m.add_scores([5.0])
        m.__exit__(None, None, None)
        assert m.scores == []
        assert math.isnan(m.get_mean("loss"))


class TestUnregisteredMetricErrors:
    def test_log_unregistered_raises_key_error(self):
        m = AgentMetrics()
        with pytest.raises(KeyError):
            m.log("unregistered", 1.0)

    def test_get_mean_unregistered_raises_key_error(self):
        m = AgentMetrics()
        with pytest.raises(KeyError):
            m.get_mean("unregistered")

    def test_multi_agent_log_invalid_agent_id_raises(self):
        m = MultiAgentMetrics(["a0", "a1"])
        m.register("loss")
        with pytest.raises(KeyError):
            m.log("loss", "nonexistent_agent", 1.0)

    def test_multi_agent_get_mean_invalid_agent_id_raises(self):
        m = MultiAgentMetrics(["a0", "a1"])
        m.register("loss")
        with pytest.raises(KeyError):
            m.get_mean("loss", "nonexistent_agent")


class TestMultiAgentNestedScores:
    def test_add_nested_scores(self):
        m = MultiAgentMetrics(["a0", "a1"])
        m.add_scores([[1.0, 2.0], [3.0, 4.0]])
        assert len(m.scores) == 2
        assert m.scores[0] == [1.0, 2.0]
        assert m.scores[1] == [3.0, 4.0]


class TestFinalizeEvoStepNearZeroElapsed:
    def test_near_zero_elapsed_does_not_error(self):
        m = AgentMetrics()
        m._evo_start_time = time.monotonic()
        m.finalize_evo_step(1000)
        assert m.steps_per_second > 0
        assert m.steps == 1000
