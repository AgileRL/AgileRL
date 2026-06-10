"""Tests for agilerl/metrics.py."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from agilerl.metrics import AgentMetrics, BaseMetrics, MultiAgentMetrics


class TestAgentMetricsInit:
    def test_default_init(self):
        m = AgentMetrics()
        assert m.steps == 0
        assert m.steps_per_second == 0.0
        assert m.scores == []
        assert list(m.fitness) == []
        assert m.additional_metrics == []
        assert m.nonscalar_metrics == []

    def test_custom_fitness_window(self):
        m = AgentMetrics(fitness_window=3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            m.add_fitness(v)
        assert list(m.fitness) == [2.0, 3.0, 4.0]


class TestAgentMetricsLog:
    def test_log_appends_value(self):
        m = AgentMetrics()
        m.register("loss")
        m.log("loss", 0.5)
        m.log("loss", 0.3)
        assert m._additional_metrics["loss"] == [0.5, 0.3]

    def test_log_coerces_to_float(self):
        m = AgentMetrics()
        m.register("reward")
        m.log("reward", 1)
        assert isinstance(m._additional_metrics["reward"][0], float)


class TestAgentMetricsLogHistogram:
    def test_log_histogram_extends_values(self):
        m = AgentMetrics()
        m.register_histogram("actions")
        m.log_histogram("actions", np.array([1, 2, 3]))
        m.log_histogram("actions", np.array([4, 5]))
        assert list(m._nonscalar_metrics["actions"]) == [1, 2, 3, 4, 5]


class TestAgentMetricsGetMean:
    def test_get_mean_returns_mean(self):
        m = AgentMetrics()
        m.register("loss")
        m.log("loss", 2.0)
        m.log("loss", 4.0)
        assert m.get_mean("loss") == pytest.approx(3.0)

    def test_get_mean_empty_returns_nan(self):
        m = AgentMetrics()
        m.register("loss")
        result = m.get_mean("loss")
        assert math.isnan(result)


class TestAgentMetricsGetHistogram:
    def test_get_histogram_returns_array(self):
        m = AgentMetrics()
        m.register_histogram("dist")
        m.log_histogram("dist", np.array([10, 20, 30]))
        result = m.get_histogram("dist")
        np.testing.assert_array_equal(result, [10, 20, 30])

    def test_get_histogram_empty_returns_none(self):
        m = AgentMetrics()
        m.register_histogram("dist")
        assert m.get_histogram("dist") is None


class TestAgentMetricsClear:
    def test_clear_resets_accumulators(self):
        m = AgentMetrics()
        m.register("loss")
        m.register_histogram("actions")
        m.log("loss", 1.0)
        m.log_histogram("actions", np.array([1, 2]))
        m.scores.extend([10.0, 20.0])

        m.clear()

        assert m._additional_metrics["loss"] == []
        assert list(m._nonscalar_metrics["actions"]) == []
        assert m.scores == []

    def test_clear_preserves_metric_registration(self):
        m = AgentMetrics()
        m.register("loss")
        m.register_histogram("dist")
        m.clear()
        assert "loss" in m._additional_metrics
        assert "dist" in m._nonscalar_metrics


class TestMultiAgentMetricsInit:
    def test_init_stores_agent_ids(self):
        m = MultiAgentMetrics(["a", "b"])
        assert m.agent_ids == ["a", "b"]

    def test_init_copies_agent_ids(self):
        ids = ["a", "b"]
        m = MultiAgentMetrics(ids)
        ids.append("c")
        assert m.agent_ids == ["a", "b"]


class TestMultiAgentMetricsRegister:
    def test_register_creates_per_agent_storage(self):
        m = MultiAgentMetrics(["x", "y"])
        m.register("loss")
        assert m._additional_metrics["loss"] == {"x": [], "y": []}

    def test_register_histogram_creates_per_agent_storage(self):
        m = MultiAgentMetrics(["x", "y"])
        m.register_histogram("actions")
        assert {k: list(v) for k, v in m._nonscalar_metrics["actions"].items()} == {
            "x": [],
            "y": [],
        }


class TestMultiAgentMetricsLog:
    def test_log_appends_per_agent(self):
        m = MultiAgentMetrics(["a", "b"])
        m.register("loss")
        m.log("loss", "a", 0.1)
        m.log("loss", "b", 0.2)
        m.log("loss", "a", 0.3)
        assert m._additional_metrics["loss"]["a"] == [0.1, 0.3]
        assert m._additional_metrics["loss"]["b"] == [0.2]

    def test_log_histogram_per_agent(self):
        m = MultiAgentMetrics(["a", "b"])
        m.register_histogram("dist")
        m.log_histogram("dist", "a", np.array([1, 2]))
        m.log_histogram("dist", "b", np.array([3]))
        assert list(m._nonscalar_metrics["dist"]["a"]) == [1, 2]
        assert list(m._nonscalar_metrics["dist"]["b"]) == [3]


class TestMultiAgentMetricsGetMean:
    def test_get_mean_per_agent(self):
        m = MultiAgentMetrics(["a", "b"])
        m.register("reward")
        m.log("reward", "a", 10.0)
        m.log("reward", "a", 20.0)
        assert m.get_mean("reward", "a") == pytest.approx(15.0)

    def test_get_mean_empty_returns_nan(self):
        m = MultiAgentMetrics(["a"])
        m.register("loss")
        assert math.isnan(m.get_mean("loss", "a"))


class TestMultiAgentMetricsGetHistogram:
    def test_get_histogram_returns_array(self):
        m = MultiAgentMetrics(["a"])
        m.register_histogram("dist")
        m.log_histogram("dist", "a", np.array([5, 6, 7]))
        result = m.get_histogram("dist", "a")
        np.testing.assert_array_equal(result, [5, 6, 7])

    def test_get_histogram_empty_returns_none(self):
        m = MultiAgentMetrics(["a"])
        m.register_histogram("dist")
        assert m.get_histogram("dist", "a") is None


class TestMultiAgentMetricsClear:
    def test_clear_resets_per_agent_accumulators(self):
        m = MultiAgentMetrics(["a", "b"])
        m.register("loss")
        m.register_histogram("actions")
        m.log("loss", "a", 1.0)
        m.log_histogram("actions", "b", np.array([2]))
        m.scores.extend([10.0])

        m.clear()

        assert m._additional_metrics["loss"] == {"a": [], "b": []}
        assert {k: list(v) for k, v in m._nonscalar_metrics["actions"].items()} == {
            "a": [],
            "b": [],
        }
        assert m.scores == []


class TestBaseMetricsProperties:
    def test_additional_metrics_property(self):
        m = AgentMetrics()
        m.register("a")
        m.register("b")
        assert m.additional_metrics == ["a", "b"]

    def test_nonscalar_metrics_property(self):
        m = AgentMetrics()
        m.register_histogram("h1")
        assert m.nonscalar_metrics == ["h1"]

    def test_scores_property(self):
        m = AgentMetrics()
        m.add_scores([1.0, 2.0, 3.0])
        assert m.scores == [1.0, 2.0, 3.0]

    def test_fitness_property(self):
        m = AgentMetrics()
        m.add_fitness(5.0)
        m.add_fitness(10.0)
        assert list(m.fitness) == [5.0, 10.0]

    def test_steps_property(self):
        m = AgentMetrics()
        m.increment_steps(100)
        m.increment_steps(50)
        assert m.steps == 150

    def test_steps_per_second_property(self):
        m = AgentMetrics()
        assert m.steps_per_second == 0.0


class TestBaseMetricsEquality:
    def test_equal_metrics(self):
        a = AgentMetrics()
        b = AgentMetrics()
        assert a == b

    def test_not_equal_different_steps(self):
        a = AgentMetrics()
        b = AgentMetrics()
        a.increment_steps(1)
        assert a != b

    def test_not_equal_different_type(self):
        a = AgentMetrics()
        assert a.__eq__("not_metrics") is NotImplemented

    def test_equal_with_data(self):
        a = AgentMetrics()
        b = AgentMetrics()
        a.register("loss")
        b.register("loss")
        a.log("loss", 1.0)
        b.log("loss", 1.0)
        a.add_fitness(5.0)
        b.add_fitness(5.0)
        a.add_scores([1.0])
        b.add_scores([1.0])
        assert a == b


class TestBaseMetricsRegisterGuard:
    def test_duplicate_scalar_raises(self):
        m = AgentMetrics()
        m.register("loss")
        with pytest.raises(ValueError, match="already registered"):
            m.register("loss")

    def test_duplicate_histogram_raises(self):
        m = AgentMetrics()
        m.register_histogram("dist")
        with pytest.raises(ValueError, match="already registered"):
            m.register_histogram("dist")

    def test_scalar_then_histogram_same_name_raises(self):
        m = AgentMetrics()
        m.register("x")
        with pytest.raises(ValueError, match="already registered"):
            m.register_histogram("x")

    def test_histogram_then_scalar_same_name_raises(self):
        m = AgentMetrics()
        m.register_histogram("x")
        with pytest.raises(ValueError, match="already registered"):
            m.register("x")


class TestBaseMetricsExit:
    def test_exit_calls_clear(self):
        m = AgentMetrics()
        m.add_scores([1.0, 2.0])
        m.__exit__(None, None, None)
        assert m.scores == []


class TestBaseMetricsTrainingStep:
    def test_init_and_finalize_training_step(self):
        m = AgentMetrics()
        m.init_training_step()
        time.sleep(0.01)
        m.finalize_training_step(100)
        assert m.steps == 100
        assert m.steps_per_second > 0

    def test_finalize_training_step_computes_sps(self):
        m = AgentMetrics()
        m._training_start_time = time.monotonic() - 1.0
        m.finalize_training_step(500)
        assert m.steps_per_second == pytest.approx(500.0, rel=0.5)
        assert m.steps == 500


class TestBaseMetricsAddFitness:
    def test_add_fitness(self):
        m = AgentMetrics()
        m.add_fitness(1.0)
        m.add_fitness(2.0)
        assert list(m.fitness) == [1.0, 2.0]


class TestBaseMetricsAddScores:
    def test_add_scores(self):
        m = AgentMetrics()
        m.add_scores([1.0, 2.0])
        m.add_scores([3.0])
        assert m.scores == [1.0, 2.0, 3.0]


class TestBaseMetricsIncrementSteps:
    def test_increment_steps(self):
        m = AgentMetrics()
        m.increment_steps(10)
        m.increment_steps(20)
        assert m.steps == 30


class TestBaseMetricsAbstractBodies:
    """Hit the `raise NotImplementedError` lines inside each abstract method body."""

    def _make_stub(self):
        class _Stub(BaseMetrics):
            def _init_metric(self, name):
                super()._init_metric(name)

            def _init_nonscalar_metric(self, name):
                super()._init_nonscalar_metric(name)

            def log(self, name, *args, **kwargs):
                super().log(name, *args, **kwargs)

            def log_histogram(self, name, values):
                super().log_histogram(name, values)

            def get_histogram(self, name, *args):
                return super().get_histogram(name, *args)

        return _Stub()

    def test_init_metric_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_stub()._init_metric("x")

    def test_init_nonscalar_metric_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_stub()._init_nonscalar_metric("x")

    def test_log_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_stub().log("x", value=1.0)

    def test_log_histogram_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_stub().log_histogram("x", np.array([1]))

    def test_get_histogram_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_stub().get_histogram("x")


class TestNonscalarWindow:
    def test_histogram_accumulation_is_bounded(self):
        m = AgentMetrics(nonscalar_window=5)
        m.register_histogram("actions")
        m.log_histogram("actions", np.arange(10))
        assert list(m._nonscalar_metrics["actions"]) == [5, 6, 7, 8, 9]

    def test_multi_agent_histogram_accumulation_is_bounded(self):
        m = MultiAgentMetrics(["a"], nonscalar_window=3)
        m.register_histogram("actions")
        m.log_histogram("actions", "a", np.arange(5))
        assert list(m._nonscalar_metrics["actions"]["a"]) == [2, 3, 4]
