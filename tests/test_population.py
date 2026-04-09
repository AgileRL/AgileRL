"""Tests for agilerl.population and agilerl.logger.

Covers:
- get_nested_mean utility
- PopulationMetrics dataclass
- MetricRow, MetricsReport (formatting, coloring, rendering)
- Population wrapper (lifecycle, metrics gathering, logger dispatch)
- Logger implementations (StdOut, CSV, Wandb, TensorBoard)
"""

from __future__ import annotations

import csv
import math
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agilerl.algorithms.core.base import MultiAgentRLAlgorithm
from agilerl.logger import CSVLogger, StdOutLogger, TensorboardLogger, WandbLogger
from agilerl.metrics import AgentMetrics, MultiAgentMetrics
from agilerl.population import (
    MetricsReport,
    Population,
    PopulationMetrics,
)
from agilerl.utils.population_utils import (
    ScalarMetricRow,
    fmt_value,
    get_nested_mean,
)


class MockAgent:
    """Minimal single-agent mock for Population tests."""

    def __init__(self, index: int = 0) -> None:
        self.metrics = AgentMetrics()
        self.mut = "None"
        self.index = index
        self.registry = MagicMock()
        self.registry.hp_config = None

    @property
    def fitness(self):
        return self.metrics.fitness


class MockMultiAgent:
    """Minimal multi-agent mock for Population tests."""

    def __init__(self, agent_ids: list[str], index: int = 0) -> None:
        self.metrics = MultiAgentMetrics(agent_ids)
        self.mut = "None"
        self.index = index
        self.registry = MagicMock()
        self.registry.hp_config = None

    @property
    def fitness(self):
        return self.metrics.fitness


# Register so isinstance(mock, MultiAgentRLAlgorithm) returns True
MultiAgentRLAlgorithm.register(MockMultiAgent)


def _make_pop_metrics(**overrides) -> PopulationMetrics:
    """Build a PopulationMetrics with sensible two-agent defaults."""
    defaults = dict(
        fitnesses=[1.0, 3.0],
        scores=[10.0, 20.0],
        steps=[100, 200],
        steps_per_second=[50.0, 100.0],
        mutations=["None", "sigma"],
        indices=[0, 1],
        additional_metrics=[{"loss": 0.5}, {"loss": 0.3}],
        hyperparameters=[{"lr": 0.001}, {"lr": 0.002}],
    )
    defaults.update(overrides)
    return PopulationMetrics(**defaults)


@pytest.fixture()
def pop_metrics() -> PopulationMetrics:
    return _make_pop_metrics()


@pytest.fixture()
def two_agents() -> list[MockAgent]:
    agents = [MockAgent(index=i) for i in range(2)]
    for a in agents:
        a.metrics.register("loss")
        a.metrics.add_fitness(1.0)
        a.metrics.add_scores([10.0])
        a.metrics.increment_steps(100)
    return agents


class TestGetNestedMean:
    def test_basic(self):
        result = get_nested_mean([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}])
        assert result == {"a": pytest.approx(2.0), "b": pytest.approx(3.0)}

    def test_with_nan(self):
        result = get_nested_mean([{"x": 1.0}, {"x": float("nan")}])
        assert result["x"] == pytest.approx(1.0)

    def test_single_dict(self):
        result = get_nested_mean([{"k": 42.0}])
        assert result == {"k": pytest.approx(42.0)}


class TestPopulationMetrics:
    def test_global_step(self, pop_metrics):
        assert pop_metrics.global_step == 300

    def test_fps(self, pop_metrics):
        assert pop_metrics.mean_fps == pytest.approx(75.0)

    def test_mean_and_best_fitness(self, pop_metrics):
        assert pop_metrics.mean_fitness == pytest.approx(2.0)
        assert pop_metrics.best_fitness == pytest.approx(3.0)

    def test_mean_score_scalar(self, pop_metrics):
        assert pop_metrics.mean_score == pytest.approx(15.0)

    def test_mean_score_dict(self):
        pm = _make_pop_metrics(
            scores=[{"a0": 1.0, "a1": 2.0}, {"a0": 3.0, "a1": 4.0}],
        )
        mean = pm.mean_score
        assert isinstance(mean, dict)
        assert mean["a0"] == pytest.approx(2.0)
        assert mean["a1"] == pytest.approx(3.0)

    def test_mean_additional_metrics_multi_agent_flattened_keys(self):
        pm = _make_pop_metrics(
            additional_metrics=[
                {"loss/a0": 0.5, "loss/a1": 0.7},
                {"loss/a0": 0.3, "loss/a1": 0.1},
            ]
        )
        mean = pm.mean_additional_metrics
        assert mean["loss/a0"] == pytest.approx(0.4)
        assert mean["loss/a1"] == pytest.approx(0.4)

    def test_dict_scores_mean(self):
        pm = _make_pop_metrics(
            scores=[
                {"alice": 1.0, "bob": 2.0},
                {"alice": 3.0, "bob": 4.0},
            ],
        )
        mean = pm.mean_score
        assert isinstance(mean, dict)
        assert mean["alice"] == pytest.approx(2.0)
        assert mean["bob"] == pytest.approx(3.0)

    def test_to_dict_keys(self, pop_metrics):
        d = pop_metrics.to_dict()
        assert "eval/mean_fitness" in d
        assert "eval/best_fitness" in d
        assert "train/global_step" in d
        assert "train/steps_per_second" in d
        assert "train/mean_score" in d
        assert "train/agent_0/loss" in d
        assert "train/agent_1/loss" in d
        assert "train/mean_loss" in d
        assert "train/agent_0/lr" in d
        assert "train/agent_1/lr" in d


class TestFmtValue:
    def test_nan(self):
        assert fmt_value(float("nan")) == "-"

    def test_integer(self):
        assert fmt_value(5.0) == "5"

    def test_large(self):
        assert "e" in fmt_value(1_234_567.89)

    def test_small(self):
        assert "e" in fmt_value(1e-6)

    def test_normal(self):
        assert fmt_value(0.12345) == "0.1235"


class TestMetricRow:
    def test_frozen(self):
        row = ScalarMetricRow(name="x", agent_values=[1.0], pop_mean=1.0)
        with pytest.raises(FrozenInstanceError):
            row.name = "y"

    def test_fmt_name_eval(self):
        row = ScalarMetricRow(name="eval/fitness", agent_values=[1.0], pop_mean=1.0)
        assert row.fmt_name == "fitness"

    def test_fmt_name_train(self):
        row = ScalarMetricRow(name="train/score", agent_values=[1.0], pop_mean=1.0)
        assert row.fmt_name == "score"

    def test_fmt_name_noop(self):
        row = ScalarMetricRow(name="other/metric", agent_values=[1.0], pop_mean=1.0)
        assert row.fmt_name == "other/metric"


class TestMetricsReport:
    def _make_report(self, **overrides) -> MetricsReport:
        return MetricsReport(_make_pop_metrics(**overrides))

    # -- render / __str__ --

    def test_render_contains_banner(self):
        report = self._make_report()
        text = str(report)
        assert "Global Steps 300" in text
        assert "Agent 0" in text
        assert "Agent 1" in text

    def test_render_banner_formats_large_global_steps(self):
        report = self._make_report(steps=[123_000, 4_567])
        text = str(report)
        assert "Global Steps 127_567" in text

    def test_report_str_delegates_to_render(self):
        report = self._make_report()
        rendered = report.render()
        assert str(report) == rendered

    def test_render_contains_banner_and_headers(self):
        report = self._make_report()
        rendered = report.render()
        assert "Global Steps 300" in rendered
        assert "Agent 0" in rendered
        assert "Agent 1" in rendered
        assert "Mean" in rendered

    def test_render_single_agent_omits_mean_column(self):
        report = self._make_report(
            fitnesses=[1.0],
            scores=[10.0],
            steps=[100],
            steps_per_second=[50.0],
            mutations=["None"],
            indices=[0],
            additional_metrics=[{"loss": 0.5}],
            hyperparameters=[{"lr": 0.001}],
        )
        rendered = report.render()
        assert "Agent 0" in rendered
        assert "Mean" not in rendered

    def test_render_contains_metadata_rows(self):
        report = self._make_report()
        rendered = report.render()
        assert "steps" in rendered
        assert "mutations" in rendered
        assert "steps/s" in rendered

    def test_to_dict_delegates(self):
        report = self._make_report()
        assert report.to_dict() == report.metrics.to_dict()

    def test_to_nonscalar_dict_empty_by_default(self):
        report = self._make_report()
        assert report.to_nonscalar_dict() == {}

    def test_to_nonscalar_dict_with_data(self):
        arr0 = np.array([1, 2, 3])
        arr1 = np.array([4, 5, 6])
        report = self._make_report(
            nonscalar_additional_metrics=[
                {"action_dist": arr0},
                {"action_dist": arr1},
            ]
        )
        result = report.to_nonscalar_dict()
        assert set(result.keys()) == {
            "train/agent_0/action_dist",
            "train/agent_1/action_dist",
        }
        np.testing.assert_array_equal(result["train/agent_0/action_dist"], arr0)
        np.testing.assert_array_equal(result["train/agent_1/action_dist"], arr1)

    def test_to_nonscalar_dict_skips_none(self):
        report = self._make_report(
            nonscalar_additional_metrics=[
                {"action_dist": np.array([1])},
                {"action_dist": None},
            ]
        )
        result = report.to_nonscalar_dict()
        assert "train/agent_0/action_dist" in result
        assert "train/agent_1/action_dist" not in result

    def test_render_multi_agent_additional_metrics_nested_rows(self):
        report = self._make_report(
            additional_metrics=[
                {
                    "loss/a0": 0.5,
                    "loss/a1": 0.6,
                    "entropy/a0": 0.2,
                    "entropy/a1": 0.3,
                },
                {
                    "loss/a0": 0.7,
                    "loss/a1": 0.9,
                    "entropy/a0": 0.4,
                    "entropy/a1": 0.5,
                },
            ]
        )
        rendered = report.render()
        assert "Sub-agent Additional Metrics" not in rendered
        assert "loss/a0" not in rendered
        assert "entropy/a0" not in rendered
        assert "loss" in rendered
        assert "entropy" in rendered
        assert "a0" in rendered
        assert "a1" in rendered

    def test_render_multi_agent_scores_nested_rows(self):
        report = self._make_report(
            scores=[
                {"a0": 10.0, "a1": 20.0},
                {"a0": 30.0, "a1": 40.0},
            ]
        )
        rendered = report.render()
        assert "score/a0" not in rendered
        assert "score/a1" not in rendered
        assert "score" in rendered
        assert "a0" in rendered
        assert "a1" in rendered

    def test_render_multi_agent_scores_nested_rows_from_dicts(self):
        report = self._make_report(
            scores=[
                {"p0": 10.0, "p1": 20.0},
                {"p0": 30.0, "p1": 40.0},
            ],
        )
        rendered = report.render()
        assert "score" in rendered
        assert "p0" in rendered
        assert "p1" in rendered


class TestPopulation:
    def test_init_homogeneous(self, two_agents):
        pop = Population(agents=two_agents)
        assert pop.size == 2

    def test_init_heterogeneous_raises(self):
        a1 = MockAgent(index=0)
        a1.metrics.add_scores([1.0])
        a2 = MagicMock()
        a2.metrics = AgentMetrics()
        a2.metrics.add_scores([1.0])
        with pytest.raises(ValueError, match="same algorithm"):
            Population(agents=[a1, a2])

    def test_agents_property_and_size(self, two_agents):
        pop = Population(agents=two_agents)
        assert pop.agents is two_agents
        assert pop.size == 2

    def test_update_replaces_agents(self, two_agents):
        pop = Population(agents=two_agents)
        new_agents = [MockAgent(index=5)]
        new_agents[0].metrics.add_scores([1.0])
        pop.update(new_agents)
        assert pop.agents is new_agents

    def test_increment_evo_step(self, two_agents):
        pop = Population(agents=two_agents)
        assert pop.evo_steps == 0
        pop.increment_evo_step()
        pop.increment_evo_step()
        assert pop.evo_steps == 2

    def test_all_below(self, two_agents):
        pop = Population(agents=two_agents)
        assert pop.all_below(200)
        assert not pop.all_below(100)

    def test_should_stop_none_target(self, two_agents):
        pop = Population(agents=two_agents)
        assert pop.should_stop(None) is False

    def test_should_stop_below_min_evo_steps(self, two_agents):
        pop = Population(agents=two_agents, min_evo_steps=10)
        pop.evo_steps = 5
        assert pop.should_stop(0.5) is False

    def test_should_stop_true(self, two_agents):
        pop = Population(agents=two_agents, min_evo_steps=2)
        pop.evo_steps = 3
        assert pop.should_stop(0.5) is True

    def test_clear_agent_metrics(self, two_agents):
        two_agents[0].metrics.log("loss", 1.0)
        pop = Population(agents=two_agents)
        pop.clear_agent_metrics()
        assert math.isnan(two_agents[0].metrics.get_mean("loss"))
        assert two_agents[0].metrics.scores == []

    def test_report_metrics_dispatches_to_loggers(self, two_agents):
        logger = MagicMock()
        pop = Population(agents=two_agents, loggers=[logger])
        pop.report_metrics()
        logger.write.assert_called_once()
        arg = logger.write.call_args[0][0]
        assert isinstance(arg, MetricsReport)

    def test_finish_calls_close_on_loggers(self, two_agents):
        l1, l2 = MagicMock(), MagicMock()
        pop = Population(agents=two_agents, loggers=[l1, l2])
        pop.finish()
        l1.close.assert_called_once()
        l2.close.assert_called_once()

    def test_collect_scores_scalar(self, two_agents):
        pop = Population(agents=two_agents)
        scores = pop._collect_scores()
        assert scores == [pytest.approx(10.0), pytest.approx(10.0)]

    def test_collect_additional_metrics_single_agent(self, two_agents):
        two_agents[0].metrics.log("loss", 0.5)
        two_agents[1].metrics.log("loss", 0.3)
        pop = Population(agents=two_agents)
        result = pop._collect_additional_metrics()
        assert result[0]["loss"] == pytest.approx(0.5)
        assert result[1]["loss"] == pytest.approx(0.3)

    def test_collect_scores_nested(self):
        agent_ids = ["a0", "a1"]
        agents = [MockMultiAgent(agent_ids, index=i) for i in range(2)]
        for a in agents:
            a.metrics.add_scores([[1.0, 2.0], [3.0, 4.0]])
            a.metrics.add_fitness(1.0)
            a.metrics.increment_steps(100)

        pop = Population(agents=agents)
        scores = pop._collect_scores()
        assert isinstance(scores[0], dict)
        assert scores[0]["a0"] == pytest.approx(2.0)
        assert scores[0]["a1"] == pytest.approx(3.0)

    def test_collect_additional_metrics_multi_agent(self):
        agent_ids = ["a0", "a1"]
        agents = [MockMultiAgent(agent_ids, index=i) for i in range(2)]
        for a in agents:
            a.metrics.register("loss")
            a.metrics.add_scores([1.0])
            a.metrics.add_fitness(1.0)
            a.metrics.increment_steps(100)
        agents[0].metrics.log("loss", "a0", 0.5)
        agents[0].metrics.log("loss", "a1", 0.7)
        agents[1].metrics.log("loss", "a0", 0.3)
        agents[1].metrics.log("loss", "a1", 0.1)

        pop = Population(agents=agents)
        result = pop._collect_additional_metrics()
        assert result[0]["loss/a0"] == pytest.approx(0.5)
        assert result[0]["loss/a1"] == pytest.approx(0.7)
        assert result[1]["loss/a0"] == pytest.approx(0.3)

    def test_collect_nonscalar_metrics_single_agent(self, two_agents):
        for a in two_agents:
            a.metrics.register_histogram("action_dist")
        two_agents[0].metrics.log_histogram("action_dist", np.array([1, 2, 3]))
        two_agents[1].metrics.log_histogram("action_dist", np.array([4, 5, 6]))

        pop = Population(agents=two_agents)
        result = pop._collect_nonscalar_metrics()
        np.testing.assert_array_equal(result[0]["action_dist"], [1, 2, 3])
        np.testing.assert_array_equal(result[1]["action_dist"], [4, 5, 6])

    def test_collect_nonscalar_metrics_multi_agent(self):
        agent_ids = ["a0", "a1"]
        agents = [MockMultiAgent(agent_ids, index=i) for i in range(2)]
        for a in agents:
            a.metrics.register_histogram("action_dist")
            a.metrics.add_scores([1.0])
            a.metrics.add_fitness(1.0)
            a.metrics.increment_steps(100)
        agents[0].metrics.log_histogram("action_dist", "a0", np.array([10, 20]))

        pop = Population(agents=agents)
        result = pop._collect_nonscalar_metrics()
        np.testing.assert_array_equal(result[0]["action_dist/a0"], [10, 20])
        assert result[0]["action_dist/a1"] is None

    def test_nonscalar_in_report_metrics(self, two_agents):
        for a in two_agents:
            a.metrics.register_histogram("action_dist")
        two_agents[0].metrics.log_histogram("action_dist", np.array([1, 2]))

        pop = Population(agents=two_agents)
        report = pop.report_metrics()
        ns = report.to_nonscalar_dict()
        assert "train/agent_0/action_dist" in ns
        assert "train/agent_1/action_dist" not in ns


# ---------------------------------------------------------------------------
# Edge-case tests — Population / PopulationMetrics / MetricsReport
# ---------------------------------------------------------------------------


class TestPopulationEdgeCases:
    def test_empty_agents_raises(self):
        with pytest.raises(ValueError, match="at least one agent"):
            Population(agents=[])

    def test_local_step(self, two_agents):
        two_agents[0].metrics.increment_steps(50)
        two_agents[1].metrics.increment_steps(200)
        pop = Population(agents=two_agents)
        assert pop.local_step == max(a.metrics.steps for a in two_agents)

    def test_is_nested_scores_false_for_scalar(self, two_agents):
        pop = Population(agents=two_agents)
        assert pop.is_nested_scores() is False

    def test_is_nested_scores_true_for_nested(self):
        agent_ids = ["a0", "a1"]
        agents = [MockMultiAgent(agent_ids, index=i) for i in range(2)]
        for a in agents:
            a.metrics.add_scores([[1.0, 2.0]])
            a.metrics.add_fitness(1.0)
            a.metrics.increment_steps(10)
        pop = Population(agents=agents)
        assert pop.is_nested_scores() is True

    def test_is_nested_scores_empty(self, two_agents):
        two_agents[0].metrics.scores.clear()
        two_agents[1].metrics.scores.clear()
        pop = Population(agents=two_agents)
        assert pop.is_nested_scores() is False

    def test_report_metrics_clear_false_preserves(self, two_agents):
        pop = Population(agents=two_agents)
        pop.report_metrics(clear=False)
        assert two_agents[0].metrics.scores != []

    def test_collect_fitnesses_dict_values(self):
        agent_ids = ["a0", "a1"]
        agents = [MockMultiAgent(agent_ids, index=i) for i in range(2)]
        for a in agents:
            a.metrics.fitness.append({"a0": 1.0, "a1": 2.0})
            a.metrics.add_scores([1.0])
            a.metrics.increment_steps(10)
        pop = Population(agents=agents)
        fitnesses = pop._collect_fitnesses()
        assert isinstance(fitnesses[0], dict)
        assert fitnesses[0]["a0"] == 1.0
        assert fitnesses[0]["a1"] == 2.0

    def test_collect_fitnesses_empty_deque_returns_nan(self, two_agents):
        for a in two_agents:
            a.metrics.fitness.clear()
        pop = Population(agents=two_agents)
        fitnesses = pop._collect_fitnesses()
        assert all(math.isnan(f) for f in fitnesses)

    def test_collect_fitnesses_ndarray_without_agent_ids_raises(self):
        agents = [MockAgent(index=0)]
        agents[0].metrics.fitness.append(np.array([1.0, 2.0]))
        agents[0].metrics.add_scores([1.0])
        agents[0].metrics.increment_steps(10)
        pop = Population(agents=agents)
        with pytest.raises(ValueError, match="without configured agent_ids"):
            pop._collect_fitnesses()

    def test_gather_metrics_with_accelerator(self, two_agents):
        acc = MagicMock()
        acc.is_main_process = True
        acc.state = MagicMock()
        acc.state.num_processes = 4
        pop = Population(agents=two_agents, accelerator=acc)
        metrics = pop._gather_metrics()
        for step, agent in zip(metrics.steps, two_agents):
            assert step == agent.metrics.steps * 4


class TestPopulationMetricsEdgeCases:
    def test_empty_additional_metrics_names_raises(self):
        pm = _make_pop_metrics(additional_metrics=[])
        with pytest.raises(IndexError):
            _ = pm.additional_metric_names

    def test_to_dict_dict_fitnesses(self):
        pm = _make_pop_metrics(
            fitnesses=[{"a0": 1.0, "a1": 2.0}, {"a0": 3.0, "a1": 4.0}],
        )
        d = pm.to_dict()
        assert "eval/mean_fitness/a0" in d
        assert "eval/mean_fitness/a1" in d
        assert "eval/best_fitness/a0" in d
        assert "eval/best_fitness/a1" in d

    def test_to_dict_empty_scores_omits_score_key(self):
        pm = _make_pop_metrics(scores=[])
        d = pm.to_dict()
        assert "train/mean_score" not in d

    def test_empty_fitnesses_returns_nan(self):
        pm = _make_pop_metrics(fitnesses=[])
        assert math.isnan(pm.mean_fitness)
        assert math.isnan(pm.best_fitness)

    def test_empty_scores_returns_nan(self):
        pm = _make_pop_metrics(scores=[])
        assert math.isnan(pm.mean_score)


class TestMetricsReportEdgeCases:
    def _make_report(self, **overrides) -> MetricsReport:
        return MetricsReport(_make_pop_metrics(**overrides))

    def test_render_no_scores_succeeds(self):
        report = self._make_report(scores=[])
        rendered = report.render()
        assert "score" not in rendered.lower() or "score" in rendered

    def test_render_no_additional_metrics_succeeds(self):
        report = self._make_report(additional_metrics=[])
        rendered = report.render()
        assert "Global Steps" in rendered

    def test_render_empty_hyperparameters_succeeds(self):
        report = self._make_report(hyperparameters=[])
        rendered = report.render()
        assert "Global Steps" in rendered
        assert "lr" not in rendered

    def test_train_rows_empty_scores(self):
        report = self._make_report(scores=[])
        rows = report.train_rows()
        names = [r.name if hasattr(r, "name") else "" for r in rows]
        assert "train/score" not in names

    def test_train_rows_empty_additional_metrics_still_has_scores(self):
        report = self._make_report(additional_metrics=[])
        rows = report.train_rows()
        names = [r.name for r in rows if hasattr(r, "name")]
        assert "train/score" in names
        assert all("loss" not in n for n in names)


class TestStdOutLogger:
    def test_write(self, pop_metrics):
        pbar = MagicMock()
        logger = StdOutLogger(pbar)
        report = MetricsReport(pop_metrics)
        logger.write(report)
        pbar.write.assert_called_once()
        assert "Global Steps" in pbar.write.call_args[0][0]


class TestCSVLogger:
    def test_creates_file_with_header_and_row(self, tmp_path, pop_metrics):
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        report = MetricsReport(pop_metrics)
        logger.write(report)
        logger.close()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "eval/mean_fitness" in rows[0]

    def test_close_and_reopen(self, tmp_path, pop_metrics):
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        logger.write(MetricsReport(pop_metrics))
        logger.close()

        logger2 = CSVLogger(path)
        logger2.write(MetricsReport(pop_metrics))
        logger2.close()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1


class TestWandbLogger:
    @patch("agilerl.logger.wandb")
    def test_write(self, mock_wandb, pop_metrics):
        logger = WandbLogger()
        report = MetricsReport(pop_metrics)
        logger.write(report)
        mock_wandb.log.assert_called_once_with(report.to_dict())

    @patch("agilerl.logger.wandb")
    def test_close(self, mock_wandb):
        logger = WandbLogger()
        logger.close()
        mock_wandb.finish.assert_called_once()


class TestTensorboardLogger:
    @patch("agilerl.logger.SummaryWriter")
    def test_write(self, MockWriter, pop_metrics):
        mock_writer = MagicMock()
        MockWriter.return_value = mock_writer

        logger = TensorboardLogger(log_dir="/tmp/tb_test")
        report = MetricsReport(pop_metrics)
        logger.write(report)

        assert mock_writer.add_scalar.call_count > 0
        mock_writer.flush.assert_called_once()

        scalar_keys = {call.args[0] for call in mock_writer.add_scalar.call_args_list}
        assert "eval/mean_fitness" in scalar_keys

    @patch("agilerl.logger.SummaryWriter")
    def test_write_histograms(self, MockWriter):
        mock_writer = MagicMock()
        MockWriter.return_value = mock_writer

        arr = np.array([1, 2, 3, 4, 5])
        pm = _make_pop_metrics(
            nonscalar_additional_metrics=[{"action_dist": arr}, {"action_dist": None}]
        )
        logger = TensorboardLogger(log_dir="/tmp/tb_test")
        logger.write(MetricsReport(pm))

        hist_keys = {call.args[0] for call in mock_writer.add_histogram.call_args_list}
        assert "train/agent_0/action_dist" in hist_keys
        assert "train/agent_1/action_dist" not in hist_keys

    @patch("agilerl.logger.SummaryWriter")
    def test_close(self, MockWriter):
        mock_writer = MagicMock()
        MockWriter.return_value = mock_writer

        logger = TensorboardLogger(log_dir="/tmp/tb_test")
        logger.close()
        mock_writer.close.assert_called_once()
