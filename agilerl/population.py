"""Population wrapper for evolutionary agent management."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from agilerl.algorithms.core.base import MultiAgentRLAlgorithm
from agilerl.logger import Logger
from agilerl.utils.population_utils import (
    NestedMetricRow,
    ScalarMetricRow,
    build_metric_row,
    fmt_value,
    get_nested_mean,
    get_values_for_key,
    partition_nested_metric_keys,
)

if TYPE_CHECKING:
    from accelerate import Accelerator

    from agilerl.algorithms.core.base import (
        LLMAlgorithm,
        RLAlgorithm,
    )

    AlgoType = RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm


AgentT = TypeVar("AgentT", bound="AlgoType")

ScalarRow = list[float | int]
NestedRow = list[dict[str, float]]
ScalarOrNestedRow = ScalarRow | NestedRow


@dataclass(frozen=True)
class PopulationMetrics:
    """Immutable snapshot of per-agent population metrics.

    Stores raw per-agent data and exposes computed properties for
    population-level aggregates.
    """

    fitnesses: ScalarOrNestedRow
    scores: ScalarOrNestedRow
    steps: ScalarRow
    steps_per_second: ScalarRow
    mutations: list[str]
    indices: ScalarRow
    additional_metrics: list[dict[str, float]]
    hyperparameters: list[dict[str, float]]
    nonscalar_additional_metrics: list[dict[str, np.ndarray | None]] = field(
        default_factory=list
    )

    @property
    def pop_size(self) -> int:
        return len(self.indices)

    @property
    def global_step(self) -> int:
        return sum(self.steps)

    @property
    def mean_fps(self) -> float:
        return float(np.mean(self.steps_per_second))

    @property
    def mean_fitness(self) -> float | dict[str, float]:
        if not self.fitnesses:
            return float("nan")
        if isinstance(self.fitnesses[0], dict):
            return get_nested_mean(self.fitnesses)
        return float(np.mean(self.fitnesses))

    @property
    def best_fitness(self) -> float | dict[str, float]:
        if not self.fitnesses:
            return float("nan")
        if isinstance(self.fitnesses[0], dict):
            fitnesses = self.fitnesses
            return {
                key: float(np.nanmax([fitness[key] for fitness in fitnesses]))
                for key in fitnesses[0]
            }
        return float(np.max(self.fitnesses))

    @property
    def mean_score(self) -> float | dict[str, float]:
        if not self.scores:
            return float("nan")
        if isinstance(self.scores[0], dict):
            return get_nested_mean(self.scores)
        return float(np.nanmean(self.scores))

    @property
    def mean_additional_metrics(self) -> dict[str, float]:
        return get_nested_mean(self.additional_metrics)

    @property
    def additional_metric_names(self) -> list[str]:
        return list(self.additional_metrics[0].keys())

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a flat, JSON-friendly dict for logging backends.

        :returns: Dictionary of metrics suitable for wandb/CSV logging.
        :rtype: dict[str, float | int | str]
        """
        d: dict[str, float | int | str] = {
            "train/global_step": self.global_step,
            "train/steps_per_second": self.mean_fps,
        }

        mean_fitness = self.mean_fitness
        if isinstance(mean_fitness, dict):
            best_fitness = self.best_fitness
            for agent_id, value in mean_fitness.items():
                d[f"eval/mean_fitness/{agent_id}"] = value
            if isinstance(best_fitness, dict):
                for agent_id, value in best_fitness.items():
                    d[f"eval/best_fitness/{agent_id}"] = value
        else:
            d["eval/mean_fitness"] = mean_fitness
            d["eval/best_fitness"] = self.best_fitness

        if self.scores:
            mean_score = self.mean_score
            if isinstance(mean_score, dict):
                for agent_id, score in mean_score.items():
                    d[f"train/mean_score/{agent_id}"] = score
            else:
                d["train/mean_score"] = mean_score

        # Per-agent additional metrics
        for idx, agent_metrics in enumerate(self.additional_metrics):
            for name, val in agent_metrics.items():
                d[f"train/agent_{idx}/{name}"] = val

        # Population-mean additional metrics
        for name, val in self.mean_additional_metrics.items():
            d[f"train/mean_{name}"] = val

        # Per-agent hyperparameters
        for idx, agent_hps in enumerate(self.hyperparameters):
            for hp_name, hp_value in agent_hps.items():
                d[f"train/agent_{idx}/{hp_name}"] = hp_value

        return d


MetricTableRow = ScalarMetricRow | NestedMetricRow


class MetricsReport:
    """Formats population metrics into a tabular report.

    Constructed by :meth:`Population.report_metrics` and consumed by
    :class:`~agilerl.logger.Logger` implementations.

    :param metrics: Aggregated population metrics snapshot.
    :type metrics: PopulationMetrics
    """

    def __init__(self, metrics: PopulationMetrics) -> None:
        self.metrics = metrics

    def __str__(self) -> str:
        return self.render()

    @property
    def show_mean_column(self) -> bool:
        """Whether to display a population mean column."""
        return self.metrics.pop_size > 1

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly dict for logging backends.

        :returns: Dictionary of metrics suitable for wandb/CSV logging.
        :rtype: dict[str, float | int | str]
        """
        return self.metrics.to_dict()

    def to_nonscalar_dict(self) -> dict[str, np.ndarray]:
        """Return per-agent non-scalar metrics for TensorBoard-style backends.

        :returns: Dictionary mapping ``train/agent_{idx}/{name}`` to arrays.
        :rtype: dict[str, numpy.ndarray]
        """
        d: dict[str, np.ndarray] = {}
        nonscalar_metrics = self.metrics.nonscalar_additional_metrics
        for idx, agent_metrics in enumerate(nonscalar_metrics):
            for name, val in agent_metrics.items():
                if val is not None:
                    d[f"train/agent_{idx}/{name}"] = val
        return d

    def eval_rows(self) -> list[ScalarMetricRow | NestedMetricRow]:
        """Return the evaluation metric rows.

        :returns: List of evaluation metric rows.
        :rtype: list[ScalarMetricRow | NestedMetricRow]
        """
        return [
            build_metric_row(
                name="eval/fitness",
                values=self.metrics.fitnesses,
                mean_value=self.metrics.mean_fitness,
            )
        ]

    def train_rows(self) -> list[ScalarMetricRow | NestedMetricRow]:
        """Return the training metric rows.

        :returns: List of training metric rows.
        :rtype: list[ScalarMetricRow | NestedMetricRow]
        """
        rows = []
        self._add_score_rows(rows)
        self._add_additional_metric_rows(rows)
        return rows

    def _add_score_rows(self, rows: list[ScalarMetricRow | NestedMetricRow]) -> None:
        """Add score rows to the report.

        :param rows: List of metric rows to add to.
        :type rows: list[ScalarMetricRow | NestedMetricRow]
        """
        if not self.metrics.scores:
            return

        rows.append(
            build_metric_row(
                name="train/score",
                values=self.metrics.scores,
                mean_value=self.metrics.mean_score,
            )
        )

    def _add_additional_metric_rows(
        self, rows: list[ScalarMetricRow | NestedMetricRow]
    ) -> None:
        """Add additional metric rows to the report.

        :param rows: List of metric rows to add to.
        :type rows: list[ScalarMetricRow | NestedMetricRow]
        """
        if not self.metrics.additional_metrics:
            return

        metric_keys = self.metrics.additional_metric_names
        mean_metrics = self.metrics.mean_additional_metrics
        scalar_metrics, nested_metrics = partition_nested_metric_keys(metric_keys)

        # Add scalar metrics
        rows.extend(
            build_metric_row(
                name=f"train/{name}",
                values=get_values_for_key(self.metrics.additional_metrics, name),
                mean_value=mean_metrics.get(name, float("nan")),
            )
            for name in scalar_metrics
        )

        # Transpose and add nested metrics (e.g. per-sub-agent metrics in multi-agent settings)
        for metric_name, agent_ids in nested_metrics.items():
            nested_values: list[dict[str, float]] = []
            for agent_id in agent_ids:
                key = f"{metric_name}/{agent_id}"
                value_series = get_values_for_key(self.metrics.additional_metrics, key)
                for idx, value in enumerate(value_series):
                    if idx >= len(nested_values):
                        nested_values.append({})
                    nested_values[idx][agent_id] = value

            rows.append(
                build_metric_row(
                    name=f"train/{metric_name}",
                    values=nested_values,
                    mean_value={
                        agent_id: mean_metrics.get(
                            f"{metric_name}/{agent_id}", float("nan")
                        )
                        for agent_id in agent_ids
                    },
                )
            )

    def _render_metric_row(
        self, table: Table, row: ScalarMetricRow | NestedMetricRow
    ) -> None:
        """Render a metric row into a Rich table.

        :param table: The table to add the row to.
        :type table: Table
        :param row: The metric row to render.
        :type row: ScalarMetricRow | NestedMetricRow
        """
        # Display all children of nested row with special formatting
        if isinstance(row, NestedMetricRow):
            parent_cells: list[str] = [
                row.fmt_name,
                *([""] * len(self.metrics.indices)),
            ]
            if self.show_mean_column:
                parent_cells.append("")
            table.add_row(*parent_cells)
            for child in row.children:
                child_cells: list[Text | str] = [
                    Text(f"  {child.name}", style="blue"),
                    *[fmt_value(value) for value in child.agent_values],
                ]
                if self.show_mean_column:
                    child_cells.append(fmt_value(child.pop_mean))
                table.add_row(*child_cells)
        else:
            scalar_cells = [
                row.fmt_name,
                *[fmt_value(value) for value in row.agent_values],
            ]
            if self.show_mean_column:
                scalar_cells.append(fmt_value(row.pop_mean))
            table.add_row(*scalar_cells)

    def render(self) -> str:
        """Render a `MetricsReport` snapshot of collected training metrics into a
        Rich-formatted table string.

        :returns: The report rendered by Rich as ANSI-styled text.
        :rtype: str
        """
        eval_rows = self.eval_rows()
        train_rows = self.train_rows()

        # Create a table for the report
        table = Table(
            title=f"Global Steps {self.metrics.global_step:_}",
            title_style="bold",
            show_header=True,
            header_style="bold",
            expand=False,
        )

        # Columns for each agents metric and population mean
        table.add_column("Metric", justify="left", style="bold")
        for idx in self.metrics.indices:
            table.add_column(f"Agent {idx}", justify="right")

        if self.show_mean_column:
            table.add_column("Mean", justify="right")

        # Evaluation metrics
        for row in eval_rows:
            self._render_metric_row(table, row)

        table.add_section()

        # Training metrics
        for row in train_rows:
            self._render_metric_row(table, row)

        table.add_section()

        # RL hyperparameters
        if self.metrics.hyperparameters:
            hp_names = list(self.metrics.hyperparameters[0].keys())
            for hp_name in hp_names:
                hp_vals = [
                    fmt_value(agent_hps.get(hp_name, float("nan")))
                    for agent_hps in self.metrics.hyperparameters
                ]
                hp_cells = [hp_name, *hp_vals]

                if self.show_mean_column:
                    hp_cells.append("")

                table.add_row(*hp_cells)
            table.add_section()

        # Extra info (steps, mutations, fps)
        steps_cells = ["steps", *[str(s) for s in self.metrics.steps]]
        if self.show_mean_column:
            steps_cells.append("")

        table.add_row(*steps_cells)

        mutation_cells = ["mutations", *[str(m) for m in self.metrics.mutations]]
        if self.show_mean_column:
            mutation_cells.append("")

        table.add_row(*mutation_cells)

        fps_cells = ["steps/s", *[fmt_value(s) for s in self.metrics.steps_per_second]]
        if self.show_mean_column:
            fps_cells.append(fmt_value(self.metrics.mean_fps))

        table.add_row(*fps_cells)

        # Redirect table print to buffer to avoid printing to console twice
        buf = io.StringIO()
        console = Console(file=buf, record=True, force_terminal=True, width=120)
        console.print(table)
        return console.export_text(styles=True)


class Population(Generic[AgentT]):
    """Population wrapper for evolutionary agent management.

    Owns the logger pipeline and provides a single :meth:`report_metrics`
    entry-point that gathers per-agent data, builds a :class:`MetricsReport`,
    and dispatches it to all configured loggers.

    :param agents: Initial population of RL agents.
    :type agents: list[AgentT]
    :param min_evo_steps: Minimum evolutionary steps before allowing early stopping.
    :type min_evo_steps: int
    :param accelerator: HuggingFace Accelerator for distributed training.
    :type accelerator: Accelerator | None
    :param loggers: List of loggers to use.
    :type loggers: list[Logger] | None
    """

    def __init__(
        self,
        agents: list[AgentT],
        min_evo_steps: int = 10,
        accelerator: Accelerator | None = None,
        loggers: list[Logger] | None = None,
    ) -> None:
        if not agents:
            msg = "Population requires at least one agent."
            raise ValueError(msg)

        sample_agent = agents[0]
        if not all(isinstance(agent, type(sample_agent)) for agent in agents):
            names = ", ".join(type(a).__name__ for a in agents)
            msg = f"All individuals in a population must be instances of the same algorithm. Found: {names}"
            raise ValueError(msg)

        self._agents = agents
        self.sample_agent = sample_agent
        self.min_evo_steps = min_evo_steps
        self.accelerator = accelerator
        self.loggers = loggers or []

        self.last_fitnesses: ScalarOrNestedRow = []
        self.evo_steps = 0
        self.is_multi_agent = all(
            isinstance(agent, MultiAgentRLAlgorithm) for agent in agents
        )
        self.additional_metric_names = self.sample_agent.metrics.additional_metrics
        self.nonscalar_metric_names = self.sample_agent.metrics.nonscalar_metrics
        self.agent_ids = (
            self.sample_agent.metrics.agent_ids if self.is_multi_agent else None
        )

    @property
    def agents(self) -> list[AgentT]:
        """Current population of agents."""
        return self._agents

    @property
    def size(self) -> int:
        """Number of agents in the population."""
        return len(self._agents)

    @property
    def local_step(self) -> int:
        """Local step counter for the population."""
        return max(agent.metrics.steps for agent in self.agents)

    def is_nested_scores(self) -> bool:
        """Check if the scores are nested per-sub-agent i.e. a nested list.

        :returns: True if the scores are nested per-sub-agent, False otherwise.
        :rtype: bool
        """
        pop_scores = [agent.metrics.scores for agent in self.agents]
        for agent_scores in pop_scores:
            if not agent_scores:
                continue
            if isinstance(agent_scores[0], list):
                return True
        return False

    def update(self, agents: list[AgentT]) -> None:
        """Replace the population (e.g. after tournament selection + mutation)."""
        self._agents = agents

    def increment_evo_step(self) -> None:
        """Increment the population-level evo-step counter."""
        self.evo_steps += 1

    def all_below(self, max_steps: int) -> bool:
        """Check if every agent's step count is below *max_steps*.

        :param max_steps: The maximum number of steps to check.
        :type max_steps: int
        :returns: True if every agent's step count is below *max_steps*, False otherwise.
        :rtype: bool
        """
        return bool(
            np.less(
                [a.metrics.steps for a in self._agents],
                max_steps,
            ).all(),
        )

    def should_stop(self, target: float | None) -> bool:
        """Check if all agents consistently exceed the target fitness and the minimum number
        of evo-steps has been reached.

        :param target: The target fitness to check.
        :type target: float | None
        :returns: True if all agents consistently exceed the target fitness, False otherwise.
        :rtype: bool
        """
        if target is None:
            return False

        return bool(
            np.all(
                np.greater(
                    [np.mean(a.fitness) for a in self._agents],
                    target,
                ),
            )
            and self.evo_steps >= self.min_evo_steps,
        )

    def clear_agent_metrics(self) -> None:
        """Clear scores and additional metric accumulators for all agents."""
        for agent in self._agents:
            agent.metrics.clear()

    def report_metrics(self, clear: bool = True) -> MetricsReport:
        """Gather, format, and log population metrics.

        :param clear: Whether to clear the metrics after reporting.
        :type clear: bool
        :returns: The metrics report.
        :rtype: MetricsReport
        """
        metrics = self._gather_metrics()
        report = MetricsReport(metrics)

        # Write report to all defined loggers
        for logger in self.loggers:
            logger.write(report)

        if clear:
            self.clear_agent_metrics()

        return report

    def finish(self) -> None:
        """Release resources held by all loggers."""
        for logger in self.loggers:
            logger.close()

    def _gather_metrics(self) -> PopulationMetrics:
        """Collect raw per-agent data into a :class:`PopulationMetrics` snapshot.

        :returns: The aggregated population metrics.
        :rtype: PopulationMetrics
        """
        fitnesses = self._collect_fitnesses()
        self.last_fitnesses = fitnesses

        steps = [agent.metrics.steps for agent in self.agents]
        if self.accelerator is not None and self.accelerator.is_main_process:
            steps = [step * self.accelerator.state.num_processes for step in steps]

        return PopulationMetrics(
            fitnesses=fitnesses,
            scores=self._collect_scores(),
            steps=steps,
            steps_per_second=[a.metrics.steps_per_second for a in self.agents],
            mutations=[a.mut for a in self.agents],
            indices=[a.index for a in self.agents],
            additional_metrics=self._collect_additional_metrics(),
            hyperparameters=self._collect_hyperparameters(),
            nonscalar_additional_metrics=self._collect_nonscalar_metrics(),
        )

    def _collect_fitnesses(self) -> ScalarOrNestedRow:
        """Collect the most recent fitness value from each agent.

        :returns: Fitness values for each individual in population.
        :rtype: ScalarOrNestedRow
        """
        fitnesses = []
        for agent in self.agents:
            if not agent.fitness:
                fitnesses.append(float("nan"))
                continue

            latest_fitness = agent.fitness[-1]
            if isinstance(latest_fitness, dict):  # multi-agent -> sum_scores=False
                fitnesses.append(
                    {
                        agent_id: float(value)
                        for agent_id, value in latest_fitness.items()
                    }
                )
            elif isinstance(latest_fitness, (list, tuple, np.ndarray)):
                if not self.agent_ids:
                    msg = "Received nested fitness values without configured agent_ids."
                    raise ValueError(msg)
                fitnesses.append(
                    {
                        agent_id: float(latest_fitness[idx])
                        for idx, agent_id in enumerate(self.agent_ids)
                    }
                )
            else:
                fitnesses.append(float(latest_fitness))

        return fitnesses

    def _collect_hyperparameters(self) -> list[dict[str, float]]:
        """Collect the defined evolvable RL hyperparameters from all agents.

        :returns: List of dictionaries containing hyperparameters for each agent.
        :rtype: list[dict[str, float]]
        """
        hyperparameters: list[dict[str, float]] = []
        for agent in self.agents:
            agent_hps = agent.registry.hp_config
            if agent_hps is not None:
                hyperparameters.append(
                    {hp_name: getattr(agent, hp_name) for hp_name in agent_hps.names()}
                )
        return hyperparameters

    def _collect_scores(self) -> ScalarOrNestedRow:
        """Collect per-agent mean scores across an evolution step.

        :returns: Mean scores for each individual in population or
            per-sub-agent scores for multi-agent systems.
        :rtype: ScalarOrNestedRow
        """
        if not self.is_nested_scores():
            return [
                float(np.mean(agent.metrics.scores))
                if agent.metrics.scores
                else float("nan")
                for agent in self.agents
            ]

        # Multi-agent, non-summed: per-sub-agent mean scores
        scores: NestedRow = []
        for agent in self.agents:
            if agent.metrics.scores:
                mean_score_subagent = np.mean(np.array(agent.metrics.scores), axis=0)
                scores.append(
                    {
                        agent_id: float(mean_score_subagent[idx])
                        for idx, agent_id in enumerate(self.agent_ids)
                    }
                )
            else:
                scores.append(dict.fromkeys(self.agent_ids, float("nan")))

        return scores

    def _collect_additional_metrics(self) -> list[dict[str, float]]:
        """Collect per-agent mean values for all registered additional metrics.

        :returns: Per-agent mean values for all registered additional metrics.
        :rtype: list[dict[str, float]]
        """
        result = []
        for agent in self.agents:
            d = {}
            for name in self.additional_metric_names:
                if self.is_multi_agent:
                    for agent_id in self.agent_ids:
                        d[f"{name}/{agent_id}"] = agent.metrics.get_mean(name, agent_id)
                else:
                    d[name] = agent.metrics.get_mean(name)
            result.append(d)
        return result

    def _collect_nonscalar_metrics(self) -> list[dict[str, np.ndarray | None]]:
        """Collect per-agent non-scalar metric arrays (e.g. histograms).

        :returns: One dict per agent mapping metric names to their accumulated arrays.
        :rtype: list[dict[str, numpy.ndarray | None]]
        """
        result = []
        for agent in self.agents:
            d = {}
            for name in self.nonscalar_metric_names:
                if self.is_multi_agent:
                    for agent_id in self.agent_ids:
                        d[f"{name}/{agent_id}"] = agent.metrics.get_histogram(
                            name, agent_id
                        )
                else:
                    d[name] = agent.metrics.get_histogram(name)
            result.append(d)
        return result
