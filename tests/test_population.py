# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl/population.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from agilerl.metrics import AgentMetrics, MultiAgentMetrics
from agilerl.population import MetricsReport, Population, PopulationMetrics


def _make_scalar_metrics(**overrides) -> PopulationMetrics:
    """Build a PopulationMetrics with sensible scalar defaults."""
    defaults = {
        "fitnesses": [1.0, 2.0],
        "scores": [10.0, 20.0],
        "steps": [100, 200],
        "steps_per_second": [50.0, 60.0],
        "mutations": ["none", "gauss"],
        "indices": [0, 1],
        "additional_metrics": [{"loss": 0.5}, {"loss": 0.3}],
        "hyperparameters": [{"lr": 0.001}, {"lr": 0.002}],
    }
    defaults.update(overrides)
    return PopulationMetrics(**defaults)


def _make_nested_metrics(**overrides) -> PopulationMetrics:
    """Build a PopulationMetrics with nested (multi-agent) fitnesses/scores."""
    defaults = {
        "fitnesses": [{"a0": 1.0, "a1": 3.0}, {"a0": 2.0, "a1": 4.0}],
        "scores": [{"a0": 10.0, "a1": 30.0}, {"a0": 20.0, "a1": 40.0}],
        "steps": [100, 200],
        "steps_per_second": [50.0, 60.0],
        "mutations": ["none", "gauss"],
        "indices": [0, 1],
        "additional_metrics": [
            {"reward/a0": 1.0, "reward/a1": 2.0},
            {"reward/a0": 3.0, "reward/a1": 4.0},
        ],
        "hyperparameters": [{"lr": 0.001}, {"lr": 0.002}],
    }
    defaults.update(overrides)
    return PopulationMetrics(**defaults)


def _make_mock_agent(
    *,
    fitness=None,
    scores=None,
    steps=100,
    steps_per_second=50.0,
    additional_metrics=None,
    nonscalar_metrics=None,
    agent_ids=None,
    mut="none",
    index=0,
    hp_config=None,
    hp_values=None,
):
    """Create a mock agent with a .metrics attribute."""
    agent = MagicMock()
    agent.mut = mut
    agent.index = index
    agent.fitness = fitness if fitness is not None else [1.0]

    # The collectors dispatch on the concrete metrics class, so the double has
    # to spec the matching one.
    metrics = MagicMock(
        spec=MultiAgentMetrics if agent_ids is not None else AgentMetrics,
    )
    metrics.scores = scores if scores is not None else [10.0]
    metrics.steps = steps
    metrics.steps_per_second = steps_per_second
    metrics.additional_metrics = additional_metrics or []
    metrics.nonscalar_metrics = nonscalar_metrics or []
    if agent_ids is not None:
        metrics.agent_ids = agent_ids
    metrics.get_mean = MagicMock(return_value=0.5)
    metrics.get_histogram = MagicMock(return_value=np.array([1.0, 2.0]))
    metrics.clear = MagicMock()
    agent.metrics = metrics

    registry = MagicMock()
    registry.hp_config = hp_config
    if hp_config is not None:
        hp_config.names = MagicMock(return_value=list((hp_values or {}).keys()))
        for k, v in (hp_values or {}).items():
            setattr(agent, k, v)
    agent.registry = registry

    return agent


def _make_population(agents, **kwargs):
    """Build a Population via __new__ to bypass __init__ validation."""
    pop = Population.__new__(Population)
    pop._agents = agents
    pop.min_evo_steps = kwargs.get("min_evo_steps", 10)
    pop.accelerator = kwargs.get("accelerator", None)
    pop.loggers = kwargs.get("loggers", [])
    pop.last_fitnesses = []
    pop.evo_steps = kwargs.get("evo_steps", 0)
    pop.is_multi_agent = kwargs.get("is_multi_agent", False)
    pop.additional_metric_names = kwargs.get("additional_metric_names", [])
    pop.nonscalar_metric_names = kwargs.get("nonscalar_metric_names", [])
    pop.agent_ids = kwargs.get("agent_ids", None)
    return pop


# ===========================================================================
# PopulationMetrics dataclass
# ===========================================================================


class TestPopulationMetrics:
    # -- mean_fitness (lines 78-82) --

    def test_mean_fitness_empty(self):
        """empty fitnesses returns nan."""
        m = _make_scalar_metrics(fitnesses=[])
        assert np.isnan(m.mean_fitness)

    def test_mean_fitness_scalar(self):
        """scalar mean."""
        m = _make_scalar_metrics(fitnesses=[2.0, 4.0])
        assert m.mean_fitness == pytest.approx(3.0)

    def test_mean_fitness_nested(self):
        """nested dict path via get_nested_mean."""
        m = _make_nested_metrics()
        result = m.mean_fitness
        assert isinstance(result, dict)
        assert result["a0"] == pytest.approx(1.5)
        assert result["a1"] == pytest.approx(3.5)

    # -- best_fitness (lines 86-94) --

    def test_best_fitness_empty(self):
        """empty returns nan."""
        m = _make_scalar_metrics(fitnesses=[])
        assert np.isnan(m.best_fitness)

    def test_best_fitness_scalar(self):
        """scalar max."""
        m = _make_scalar_metrics(fitnesses=[1.0, 5.0])
        assert m.best_fitness == pytest.approx(5.0)

    def test_best_fitness_nested(self):
        """nested dict per-key nanmax."""
        m = _make_nested_metrics()
        result = m.best_fitness
        assert isinstance(result, dict)
        assert result["a0"] == pytest.approx(2.0)
        assert result["a1"] == pytest.approx(4.0)

    # -- mean_score (lines 97-102) --

    def test_mean_score_empty(self):
        """empty returns nan."""
        m = _make_scalar_metrics(scores=[])
        assert np.isnan(m.mean_score)

    def test_mean_score_scalar(self):
        """scalar nanmean."""
        m = _make_scalar_metrics(scores=[10.0, 20.0])
        assert m.mean_score == pytest.approx(15.0)

    def test_mean_score_nested(self):
        """nested dict scores."""
        m = _make_nested_metrics()
        result = m.mean_score
        assert isinstance(result, dict)
        assert result["a0"] == pytest.approx(15.0)
        assert result["a1"] == pytest.approx(35.0)

    # -- mean_additional_metrics (line 106) --

    def test_mean_additional_metrics(self):
        m = _make_scalar_metrics(
            additional_metrics=[
                {"loss": 0.5, "reward": 1.0},
                {"loss": 0.3, "reward": 2.0},
            ]
        )
        result = m.mean_additional_metrics
        assert result["loss"] == pytest.approx(0.4)
        assert result["reward"] == pytest.approx(1.5)

    # -- additional_metric_names (line 110) --

    def test_additional_metric_names(self):
        m = _make_scalar_metrics(additional_metrics=[{"loss": 0.5, "acc": 0.9}])
        assert m.additional_metric_names == ["loss", "acc"]

    # -- pop_size, global_step, mean_fps --

    def test_pop_size(self):
        m = _make_scalar_metrics()
        assert m.pop_size == 2

    def test_global_step(self):
        m = _make_scalar_metrics(steps=[100, 200])
        assert m.global_step == 300

    def test_mean_fps(self):
        m = _make_scalar_metrics(steps_per_second=[50.0, 60.0])
        assert m.mean_fps == pytest.approx(55.0)

    # -- to_dict (lines 112-157) --

    def test_to_dict_scalar(self):
        """scalar fitness/scores path."""
        m = _make_scalar_metrics(
            additional_metrics=[{"loss": 0.5}, {"loss": 0.3}],
            hyperparameters=[{"lr": 0.001}, {"lr": 0.002}],
        )
        d = m.to_dict()
        assert "eval/mean_fitness" in d
        assert "eval/best_fitness" in d
        assert "train/mean_score" in d
        assert d["train/agent_0/local_steps"] == 100
        assert d["train/agent_1/local_steps"] == 200
        assert d["eval/agent_0/fitness"] == pytest.approx(1.0)
        assert d["eval/agent_1/fitness"] == pytest.approx(2.0)
        assert d["train/agent_0/score"] == pytest.approx(10.0)
        assert d["train/agent_1/score"] == pytest.approx(20.0)
        assert "train/agent_0/loss" in d
        assert "train/agent_1/loss" in d
        assert "train/mean_loss" in d
        assert "train/agent_0/lr" in d
        assert "train/agent_1/lr" in d

    def test_to_dict_nested(self):
        """nested fitness/scores dict path."""
        m = _make_nested_metrics()
        d = m.to_dict()
        assert "eval/mean_fitness/a0" in d
        assert "eval/best_fitness/a0" in d
        assert "train/mean_score/a0" in d
        assert d["train/agent_0/local_steps"] == 100
        assert d["eval/agent_0/fitness/a0"] == pytest.approx(1.0)
        assert d["train/agent_0/score/a0"] == pytest.approx(10.0)

    def test_to_dict_empty_scores(self):
        """mean score block skipped when empty; per-agent scores omitted too."""
        m = _make_scalar_metrics(scores=[])
        d = m.to_dict()
        assert "train/mean_score" not in d
        assert "train/agent_0/score" not in d
        assert "train/agent_0/local_steps" in d


# ===========================================================================
# MetricsReport
# ===========================================================================


class TestMetricsReport:
    def test_str_returns_render(self):
        """__str__ delegates to render()."""
        m = _make_scalar_metrics()
        report = MetricsReport(m)
        assert str(report) == report.render()

    def test_show_mean_column_true(self):
        """True when pop_size > 1."""
        m = _make_scalar_metrics()
        assert MetricsReport(m).show_mean_column is True

    def test_show_mean_column_false(self):
        """False when pop_size == 1."""
        m = _make_scalar_metrics(
            fitnesses=[1.0],
            scores=[10.0],
            steps=[100],
            steps_per_second=[50.0],
            mutations=["none"],
            indices=[0],
            additional_metrics=[{"loss": 0.5}],
            hyperparameters=[{"lr": 0.001}],
        )
        assert MetricsReport(m).show_mean_column is False

    def test_to_dict_delegates(self):
        """delegates to metrics.to_dict."""
        m = _make_scalar_metrics()
        report = MetricsReport(m)
        assert report.to_dict() == m.to_dict()

    def test_to_nonscalar_dict_with_values(self):
        """returns non-scalar metrics keyed by agent index."""
        arr = np.array([1.0, 2.0, 3.0])
        m = _make_scalar_metrics(
            nonscalar_additional_metrics=[
                {"histogram": arr, "empty_one": None},
                {"histogram": arr, "empty_one": None},
            ]
        )
        report = MetricsReport(m)
        d = report.to_nonscalar_dict()
        assert "train/agent_0/histogram" in d
        assert "train/agent_1/histogram" in d
        assert "train/agent_0/empty_one" not in d  # None filtered out
        np.testing.assert_array_equal(d["train/agent_0/histogram"], arr)

    def test_to_nonscalar_dict_empty(self):
        """empty when no nonscalar metrics."""
        m = _make_scalar_metrics()
        report = MetricsReport(m)
        assert report.to_nonscalar_dict() == {}

    def test_eval_rows(self):
        """returns fitness row."""
        m = _make_scalar_metrics()
        rows = MetricsReport(m).eval_rows()
        assert len(rows) == 1

    def test_train_rows_with_scores_and_metrics(self):
        """score + additional metric rows."""
        m = _make_scalar_metrics()
        rows = MetricsReport(m).train_rows()
        assert len(rows) >= 2  # score row + at least one additional metric row

    def test_train_rows_empty_scores(self):
        """no score row when scores empty."""
        m = _make_scalar_metrics(scores=[])
        rows = MetricsReport(m).train_rows()
        # Only additional metric rows, no score row
        score_names = [r.name for r in rows if hasattr(r, "name")]
        assert "train/score" not in score_names

    def test_train_rows_empty_additional_metrics(self):
        """no additional metric rows when empty."""
        m = _make_scalar_metrics(additional_metrics=[])
        rows = MetricsReport(m).train_rows()
        assert len(rows) == 1  # only score row

    def test_add_additional_metric_rows_nested(self):
        """nested additional metrics (multi-agent keys with /)."""
        m = _make_nested_metrics()
        report = MetricsReport(m)
        rows = report.train_rows()
        nested_rows = [r for r in rows if hasattr(r, "children")]
        assert len(nested_rows) >= 1

    def test_render_scalar_single_agent(self):
        """render with single agent (no mean column)."""
        m = _make_scalar_metrics(
            fitnesses=[3.0],
            scores=[15.0],
            steps=[300],
            steps_per_second=[75.0],
            mutations=["none"],
            indices=[0],
            additional_metrics=[{"loss": 0.5}],
            hyperparameters=[{"lr": 0.001}],
        )
        report = MetricsReport(m)
        text = report.render()
        assert "Agent 0" in text
        assert "fitness" in text
        assert "steps" in text

    def test_render_scalar_multi_agent(self):
        """render with multiple agents (mean column shown)."""
        m = _make_scalar_metrics()
        report = MetricsReport(m)
        text = report.render()
        assert "Agent 0" in text
        assert "Agent 1" in text
        assert "Mean" in text
        assert "last_mutation" in text
        assert "steps/s" in text

    def test_render_nested_metrics(self):
        """render nested metric rows."""
        m = _make_nested_metrics()
        report = MetricsReport(m)
        text = report.render()
        assert "fitness" in text
        assert "a0" in text

    def test_render_no_hyperparameters(self):
        """skip HP section when empty."""
        m = _make_scalar_metrics(hyperparameters=[])
        report = MetricsReport(m)
        text = report.render()
        assert "lr" not in text

    def test_render_shows_subpopulation_row_when_tagged(self):
        """MF-PBT subpopulation row rendered."""
        m = _make_scalar_metrics(subpopulations=[0, None])
        report = MetricsReport(m)
        text = report.render()
        assert "subpopulation" in text

    def test_render_omits_subpopulation_row_when_all_untagged(self):
        """No subpopulation row under tournament/no-HPO."""
        m = _make_scalar_metrics(subpopulations=[None, None])
        report = MetricsReport(m)
        text = report.render()
        assert "subpopulation" not in text


# ===========================================================================
# Population class
# ===========================================================================


class TestPopulationInit:
    def test_empty_agents_raises(self):
        """ValueError on empty agents list."""
        with pytest.raises(ValueError, match="at least one agent"):
            Population(agents=[])

    def test_mixed_types_raises(self):
        """ValueError when agents have different types."""

        class AlgoA:
            pass

        class AlgoB:
            pass

        a, b = AlgoA(), AlgoB()
        with pytest.raises(ValueError, match="same algorithm"):
            Population(agents=[a, b])

    def test_init_happy_path(self):
        """real __init__ with uniform agent types."""

        class FakeAlgo:
            pass

        agent = FakeAlgo()
        metrics = MagicMock()
        metrics.additional_metrics = ["loss"]
        metrics.nonscalar_metrics = ["hist"]
        metrics.agent_ids = None
        agent.metrics = metrics

        pop = Population(agents=[agent])
        assert pop._agents == [agent]
        assert pop.min_evo_steps == 100
        assert pop.accelerator is None
        assert pop.loggers == []
        assert pop.last_fitnesses == []
        assert pop.evo_steps == 0
        assert pop.is_multi_agent is False
        assert pop.additional_metric_names == ["loss"]
        assert pop.nonscalar_metric_names == ["hist"]
        assert pop.agent_ids is None

    def test_init_multi_agent_real(self):
        """is_multi_agent=True when all agents are MultiAgentAlgorithm."""
        from agilerl.algorithms.core.base import MultiAgentAlgorithm

        agent = MagicMock(spec=MultiAgentAlgorithm)
        agent.metrics = MagicMock(spec=MultiAgentMetrics)
        agent.metrics.additional_metrics = []
        agent.metrics.nonscalar_metrics = []
        agent.metrics.agent_ids = ["a0", "a1"]

        pop = Population(agents=[agent])
        assert pop.is_multi_agent is True
        assert pop.agent_ids == ["a0", "a1"]


class TestPopulationProperties:
    def test_agents_property(self):
        """returns _agents."""
        agents = [_make_mock_agent(index=i) for i in range(3)]
        pop = _make_population(agents)
        assert pop.agents is agents

    def test_size(self):
        pop = _make_population([_make_mock_agent(), _make_mock_agent()])
        assert pop.size == 2

    def test_local_step(self):
        """max of agent steps."""
        a1 = _make_mock_agent(steps=100)
        a2 = _make_mock_agent(steps=200)
        pop = _make_population([a1, a2])
        assert pop.local_step == 200


class TestPopulationIsNestedScores:
    def test_flat_scores(self):
        """returns False for flat scores."""
        a = _make_mock_agent(scores=[1.0, 2.0])
        pop = _make_population([a])
        assert pop.is_nested_scores() is False

    def test_nested_scores(self):
        """returns True when scores are list of lists."""
        a = _make_mock_agent(scores=[[1.0, 2.0], [3.0, 4.0]])
        pop = _make_population([a])
        assert pop.is_nested_scores() is True

    def test_empty_scores_skipped(self):
        """agents with empty scores are skipped."""
        a_empty = _make_mock_agent(scores=[])
        a_nested = _make_mock_agent(scores=[[1.0, 2.0]])
        pop = _make_population([a_empty, a_nested])
        assert pop.is_nested_scores() is True

    def test_all_empty_scores(self):
        """returns False when all agents have empty scores."""
        a = _make_mock_agent(scores=[])
        pop = _make_population([a])
        assert pop.is_nested_scores() is False


class TestPopulationUpdate:
    def test_update_replaces_agents(self):
        a1 = _make_mock_agent(index=0)
        a2 = _make_mock_agent(index=1)
        pop = _make_population([a1])
        pop.update([a2])
        assert pop.agents == [a2]

    def test_update_refreshes_derived_views(self):
        """No reference to a replaced agent (or its metrics) is retained."""
        a1 = _make_mock_agent(index=0, additional_metrics=["loss"])
        a2 = _make_mock_agent(index=1, additional_metrics=["loss", "entropy"])
        pop = _make_population([a1])
        pop.update([a2])
        assert pop.additional_metric_names == ["loss", "entropy"]
        assert all(vars(pop)[k] is not a1 for k in vars(pop))


class TestPopulationIncrementEvoStep:
    def test_increment(self):
        pop = _make_population([_make_mock_agent()])
        assert pop.evo_steps == 0
        pop.increment_evo_step()
        assert pop.evo_steps == 1


class TestPopulationAllBelow:
    def test_all_below_true(self):
        """all agents below max_steps."""
        a1 = _make_mock_agent(steps=50)
        a2 = _make_mock_agent(steps=99)
        pop = _make_population([a1, a2])
        assert pop.all_below(100) is True

    def test_all_below_false(self):
        """one agent at or above max_steps."""
        a1 = _make_mock_agent(steps=50)
        a2 = _make_mock_agent(steps=100)
        pop = _make_population([a1, a2])
        assert pop.all_below(100) is False


class TestPopulationShouldStop:
    def test_should_stop_none_target(self):
        """returns False when target is None."""
        pop = _make_population([_make_mock_agent()])
        assert pop.should_stop(None) is False

    def test_should_stop_below_min_evo(self):
        """returns False when evo_steps < min_evo_steps."""
        a = _make_mock_agent(fitness=[100.0])
        pop = _make_population([a], min_evo_steps=10, evo_steps=5)
        assert pop.should_stop(1.0) is False

    def test_should_stop_true(self):
        """returns True when all above target and enough evo_steps."""
        a = _make_mock_agent(fitness=[100.0])
        pop = _make_population([a], min_evo_steps=5, evo_steps=10)
        assert pop.should_stop(1.0) is True

    def test_should_stop_judges_last_10_fitnesses_only(self):
        """Early bad fitness must not prevent stopping once recent runs pass."""
        a = _make_mock_agent(fitness=[-1000.0] * 50 + [100.0] * 10)
        pop = _make_population([a], min_evo_steps=5, evo_steps=10)
        assert pop.should_stop(1.0) is True

    def test_should_stop_false_when_recent_fitness_below_target(self):
        a = _make_mock_agent(fitness=[100.0] * 50 + [-1000.0] * 10)
        pop = _make_population([a], min_evo_steps=5, evo_steps=10)
        assert pop.should_stop(1.0) is False


class TestPopulationClearAndFinish:
    def test_clear_agent_metrics(self):
        a = _make_mock_agent()
        pop = _make_population([a])
        pop.clear_agent_metrics()
        a.metrics.clear.assert_called_once()

    def test_finish_closes_loggers(self):
        logger = MagicMock()
        pop = _make_population([_make_mock_agent()], loggers=[logger])
        pop.finish()
        logger.close.assert_called_once()


class TestPopulationReportMetrics:
    def test_report_metrics_writes_to_loggers(self):
        """gathers, logs, clears."""
        logger = MagicMock()
        a = _make_mock_agent(
            additional_metrics=["loss"],
            nonscalar_metrics=[],
            hp_config=MagicMock(),
            hp_values={"lr": 0.001},
        )
        pop = _make_population(
            [a],
            loggers=[logger],
            additional_metric_names=["loss"],
            nonscalar_metric_names=[],
        )

        report = pop.report_metrics(clear=True)
        assert isinstance(report, MetricsReport)
        logger.write.assert_called_once_with(report)
        a.metrics.clear.assert_called_once()

    def test_report_metrics_no_clear(self):
        """skip clear when clear=False."""
        a = _make_mock_agent(
            additional_metrics=["loss"],
            nonscalar_metrics=[],
            hp_config=MagicMock(),
            hp_values={"lr": 0.001},
        )
        pop = _make_population(
            [a],
            additional_metric_names=["loss"],
            nonscalar_metric_names=[],
        )
        pop.report_metrics(clear=False)
        a.metrics.clear.assert_not_called()


class TestCollectFitnesses:
    def test_scalar_fitness(self):
        """scalar path."""
        a = _make_mock_agent(fitness=[3.0, 5.0])
        pop = _make_population([a])
        result = pop._collect_fitnesses()
        assert result == [5.0]

    def test_empty_fitness(self):
        """nan for empty fitness."""
        a = _make_mock_agent(fitness=[])
        pop = _make_population([a])
        result = pop._collect_fitnesses()
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_dict_fitness(self):
        """multi-agent dict fitness."""
        a = _make_mock_agent(fitness=[{"a0": 1.0, "a1": 2.0}])
        pop = _make_population([a])
        result = pop._collect_fitnesses()
        assert result == [{"a0": 1.0, "a1": 2.0}]

    def test_list_fitness_with_agent_ids(self):
        """list/tuple fitness converted to dict via agent_ids."""
        a = _make_mock_agent(fitness=[[1.0, 2.0]])
        pop = _make_population([a], agent_ids=["a0", "a1"])
        result = pop._collect_fitnesses()
        assert result == [{"a0": 1.0, "a1": 2.0}]

    def test_list_fitness_without_agent_ids_raises(self):
        """raises when nested fitness but no agent_ids."""
        a = _make_mock_agent(fitness=[[1.0, 2.0]])
        pop = _make_population([a], agent_ids=None)
        with pytest.raises(ValueError, match="without configured agent_ids"):
            pop._collect_fitnesses()

    def test_ndarray_fitness_with_agent_ids(self):
        """numpy array fitness converted to dict."""
        a = _make_mock_agent(fitness=[np.array([1.0, 2.0])])
        pop = _make_population([a], agent_ids=["a0", "a1"])
        result = pop._collect_fitnesses()
        assert result[0]["a0"] == pytest.approx(1.0)
        assert result[0]["a1"] == pytest.approx(2.0)

    def test_nested_with_empty_agent_fills_nan_dict(self):
        """nested pop where one agent has empty fitness -> nan dict keyed by agent_ids."""
        a_nested = _make_mock_agent(fitness=[{"a0": 1.0, "a1": 2.0}])
        a_empty = _make_mock_agent(fitness=[])
        pop = _make_population([a_nested, a_empty], agent_ids=["a0", "a1"])
        result = pop._collect_fitnesses()
        assert result[0] == {"a0": 1.0, "a1": 2.0}
        assert set(result[1].keys()) == {"a0", "a1"}
        assert np.isnan(result[1]["a0"])
        assert np.isnan(result[1]["a1"])


class TestCollectScores:
    def test_scalar_scores(self):
        """flat mean scores."""
        a1 = _make_mock_agent(scores=[10.0, 20.0])
        a2 = _make_mock_agent(scores=[30.0, 40.0])
        pop = _make_population([a1, a2])
        result = pop._collect_scores()
        assert result[0] == pytest.approx(15.0)
        assert result[1] == pytest.approx(35.0)

    def test_scalar_empty_scores(self):
        """nan for agent with empty scores."""
        a = _make_mock_agent(scores=[])
        pop = _make_population([a])
        result = pop._collect_scores()
        assert np.isnan(result[0])

    def test_nested_scores(self):
        """per-sub-agent mean scores."""
        a = _make_mock_agent(scores=[[1.0, 2.0], [3.0, 4.0]])
        pop = _make_population([a], agent_ids=["a0", "a1"])
        result = pop._collect_scores()
        assert result[0]["a0"] == pytest.approx(2.0)
        assert result[0]["a1"] == pytest.approx(3.0)

    def test_nested_empty_scores(self):
        """nan dict when nested but agent has no scores."""
        a_with = _make_mock_agent(scores=[[1.0, 2.0]])
        a_empty = _make_mock_agent(scores=[])
        pop = _make_population([a_with, a_empty], agent_ids=["a0", "a1"])
        result = pop._collect_scores()
        assert np.isnan(result[1]["a0"])
        assert np.isnan(result[1]["a1"])

    def test_nested_scores_without_agent_ids_raises(self):
        """raises when nested scores but no agent_ids configured."""
        a = _make_mock_agent(scores=[[1.0, 2.0]])
        pop = _make_population([a], agent_ids=None)
        with pytest.raises(ValueError, match="without configured agent_ids"):
            pop._collect_scores()


class TestCollectAdditionalMetrics:
    def test_single_agent(self):
        """single-agent path (no agent_ids)."""
        a = _make_mock_agent(additional_metrics=["loss"])
        a.metrics.get_mean.return_value = 0.42
        pop = _make_population([a], additional_metric_names=["loss"])
        result = pop._collect_additional_metrics()
        assert result == [{"loss": 0.42}]

    def test_multi_agent(self):
        """multi-agent path with agent_ids."""
        a = _make_mock_agent(additional_metrics=["reward"], agent_ids=["a0", "a1"])
        a.metrics.get_mean.side_effect = lambda name, agent_id=None: {
            ("reward", "a0"): 1.0,
            ("reward", "a1"): 2.0,
        }[(name, agent_id)]

        pop = _make_population(
            [a],
            is_multi_agent=True,
            additional_metric_names=["reward"],
            agent_ids=["a0", "a1"],
        )
        result = pop._collect_additional_metrics()
        assert result[0]["reward/a0"] == 1.0
        assert result[0]["reward/a1"] == 2.0


class TestCollectNonscalarMetrics:
    def test_single_agent(self):
        """single-agent nonscalar metrics."""
        arr = np.array([1.0, 2.0])
        a = _make_mock_agent(nonscalar_metrics=["hist"])
        a.metrics.get_histogram.return_value = arr
        pop = _make_population([a], nonscalar_metric_names=["hist"])
        result = pop._collect_nonscalar_metrics()
        assert len(result) == 1
        np.testing.assert_array_equal(result[0]["hist"], arr)

    def test_multi_agent(self):
        """multi-agent nonscalar metrics."""
        arr = np.array([1.0, 2.0])
        a = _make_mock_agent(nonscalar_metrics=["hist"], agent_ids=["a0", "a1"])
        a.metrics.get_histogram.return_value = arr

        pop = _make_population(
            [a],
            is_multi_agent=True,
            nonscalar_metric_names=["hist"],
            agent_ids=["a0", "a1"],
        )
        result = pop._collect_nonscalar_metrics()
        assert "hist/a0" in result[0]
        assert "hist/a1" in result[0]


class TestCollectHyperparameters:
    def test_with_hp_config(self):
        """collects HPs from agents."""
        hp_config = MagicMock()
        hp_config.names.return_value = ["lr", "gamma"]
        a = _make_mock_agent(
            hp_config=hp_config, hp_values={"lr": 0.001, "gamma": 0.99}
        )
        pop = _make_population([a])
        result = pop._collect_hyperparameters()
        assert result == [{"lr": 0.001, "gamma": 0.99}]

    def test_without_hp_config(self):
        """skips agent when hp_config is None."""
        a = _make_mock_agent(hp_config=None)
        pop = _make_population([a])
        result = pop._collect_hyperparameters()
        assert result == []


class TestGatherMetrics:
    def test_gather_metrics_no_accelerator(self):
        """full gather without accelerator."""
        hp_config = MagicMock()
        hp_config.names.return_value = ["lr"]
        a = _make_mock_agent(
            fitness=[2.0],
            scores=[10.0],
            additional_metrics=["loss"],
            nonscalar_metrics=[],
            hp_config=hp_config,
            hp_values={"lr": 0.001},
        )
        pop = _make_population(
            [a],
            additional_metric_names=["loss"],
            nonscalar_metric_names=[],
        )
        metrics = pop._gather_metrics()
        assert isinstance(metrics, PopulationMetrics)
        assert metrics.steps == [100]
        assert pop.last_fitnesses == metrics.fitnesses

    def test_gather_metrics_with_accelerator(self):
        """steps multiplied by num_processes."""
        hp_config = MagicMock()
        hp_config.names.return_value = []
        a = _make_mock_agent(
            fitness=[1.0],
            scores=[5.0],
            steps=100,
            additional_metrics=[],
            nonscalar_metrics=[],
            hp_config=hp_config,
            hp_values={},
        )
        accelerator = MagicMock()
        accelerator.is_main_process = True
        accelerator.num_processes = 4

        pop = _make_population(
            [a],
            accelerator=accelerator,
            additional_metric_names=[],
            nonscalar_metric_names=[],
        )
        metrics = pop._gather_metrics()
        assert metrics.steps == [400]

    def test_gather_metrics_accelerator_not_main(self):
        """steps NOT multiplied when not main process."""
        hp_config = MagicMock()
        hp_config.names.return_value = []
        a = _make_mock_agent(
            fitness=[1.0],
            scores=[5.0],
            steps=100,
            additional_metrics=[],
            nonscalar_metrics=[],
            hp_config=hp_config,
            hp_values={},
        )
        accelerator = MagicMock()
        accelerator.is_main_process = False

        pop = _make_population(
            [a],
            accelerator=accelerator,
            additional_metric_names=[],
            nonscalar_metric_names=[],
        )
        metrics = pop._gather_metrics()
        assert metrics.steps == [100]
