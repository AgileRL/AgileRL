"""Population wrapper for evolutionary agent management."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
from tabulate import SEPARATING_LINE, tabulate

from agilerl.algorithms.core.base import MultiAgentRLAlgorithm
from agilerl.logger import Logger

if TYPE_CHECKING:
    from accelerate import Accelerator

    from agilerl.algorithms.core.base import (
        LLMAlgorithm,
        RLAlgorithm,
    )

    AlgoType = RLAlgorithm | MultiAgentRLAlgorithm | LLMAlgorithm


AgentT = TypeVar("AgentT", bound="AlgoType")


def get_nested_mean(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Transpose a list of dicts and compute the mean per key.

    :param metrics: List of dictionaries containing metrics.
    :type metrics: list[dict[str, float]]
    :returns: Dictionary mapping each key to the mean across all dicts.
    :rtype: dict[str, float]

    Example::

        >>> metrics = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        >>> get_nested_mean(metrics)
        {"a": 2.0, "b": 3.0}
    """
    metrics_T: dict[str, list[float]] = defaultdict(list)
    for indi_metrics in metrics:
        for metric_name, metric_value in indi_metrics.items():
            metrics_T[metric_name].append(metric_value)

    # Compute the mean of each metric
    return {
        metric_name: float(np.nanmean(metric_values))
        for metric_name, metric_values in metrics_T.items()
    }


@dataclass(frozen=True)
class PopulationMetrics:
    """Immutable snapshot of per-agent population metrics.

    Stores raw per-agent data and exposes computed properties for
    population-level aggregates.
    """

    fitnesses: list[float]
    scores: list[float | dict[str, float]]
    steps: list[int]
    steps_per_second: list[float]
    mutations: list[str]
    indices: list[int]
    additional_metrics: list[dict[str, float]]
    hyperparameters: list[dict[str, float]]

    @property
    def global_step(self) -> int:
        return sum(self.steps)

    @property
    def fps(self) -> float:
        return float(np.mean(self.steps_per_second))

    @property
    def mean_fitness(self) -> float:
        return float(np.mean(self.fitnesses))

    @property
    def best_fitness(self) -> float:
        return float(np.max(self.fitnesses))

    @property
    def mean_score(self) -> float | dict[str, float]:
        if self.scores and isinstance(self.scores[0], dict):
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
            "eval/mean_fitness": self.mean_fitness,
            "eval/best_fitness": self.best_fitness,
            "train/global_step": self.global_step,
            "train/steps_per_second": self.fps,
        }

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


@dataclass(frozen=True)
class MetricRow:
    """Immutable row of metric data.

    :param name: Metric name.
    :param agent_values: List of metric values per agent.
    :param pop_mean: Mean of the metric values for the population.
    """

    name: str
    agent_values: list[float]
    pop_mean: float


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
        return self.render(self._eval_rows(), self._train_rows())

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-friendly dict for logging backends.

        :returns: Dictionary of metrics suitable for wandb/CSV logging.
        :rtype: dict[str, float | int | str]
        """
        return self.metrics.to_dict()

    def _eval_rows(self) -> list[MetricRow]:
        return [
            MetricRow(
                name="eval/fitness",
                agent_values=self.metrics.fitnesses,
                pop_mean=self.metrics.mean_fitness,
            )
        ]

    def _train_rows(self) -> list[MetricRow]:
        rows: list[MetricRow] = []
        self._add_score_rows(rows)
        self._add_additional_metric_rows(rows)
        return rows

    def _add_score_rows(self, rows: list[MetricRow]) -> None:
        """Add score rows to the report.

        :param rows: List of metric rows to add to.
        :type rows: list[MetricRow]
        """
        agent_scores = self.metrics.scores
        mean_score = self.metrics.mean_score
        if agent_scores and isinstance(agent_scores[0], dict):
            for key in agent_scores[0]:
                agent_values = [score[key] for score in agent_scores]  # type: ignore[union-attr]
                mean_val = (
                    mean_score[key] if isinstance(mean_score, dict) else float("nan")
                )
                row = MetricRow(
                    name=f"train/score/{key}",
                    agent_values=agent_values,
                    pop_mean=mean_val,
                )
                rows.append(row)
        else:
            mean_val = (
                float(mean_score) if not isinstance(mean_score, dict) else float("nan")
            )
            row = MetricRow(
                name="train/score",
                agent_values=list(agent_scores),
                pop_mean=mean_val,
            )
            rows.append(row)

    def _add_additional_metric_rows(self, rows: list[MetricRow]) -> None:
        """Add additional metric rows to the report.

        :param rows: List of metric rows to add to.
        :type rows: list[MetricRow]
        """
        if not self.metrics.additional_metrics:
            return

        mean_metrics = self.metrics.mean_additional_metrics
        for name in self.metrics.additional_metric_names:
            agent_values = [
                agent_metrics.get(name, float("nan"))
                for agent_metrics in self.metrics.additional_metrics
            ]
            row = MetricRow(
                name=f"train/{name}",
                agent_values=agent_values,
                pop_mean=mean_metrics.get(name, float("nan")),
            )
            rows.append(row)

    @staticmethod
    def _fmt(v: float) -> str:
        if np.isnan(v):
            return "-"
        if isinstance(v, (int, np.integer)) or (
            isinstance(v, float) and v.is_integer()
        ):
            return str(int(v))
        if abs(v) >= 1e6:
            return f"{v:.2e}"
        if abs(v) < 5e-5:
            return f"{v:.2e}"
        return f"{v:.4f}"

    def render(
        self,
        eval_rows: list[MetricRow],
        train_rows: list[MetricRow],
    ) -> str:
        """Render eval/train/meta rows into a tabulate-formatted banner.

        :param eval_rows: List of evaluation metric rows.
        :type eval_rows: list[MetricRow]
        :param train_rows: List of training metric rows.
        :type train_rows: list[MetricRow]
        :returns: The report as a tabular string.
        :rtype: str

        Example::

            >>> report = MetricsReport(metrics)
            >>> print(report)
            ================================
            Global Steps 1000
            ================================
            Metric   Agent 0  Agent 1  Mean
            -------- ------- ------- -------
            eval/fitness  1.0     2.0     1.5
            train/score     1.0     2.0     1.5
            train/additional_metric  1.0     2.0     1.5
            -------- ------- ------- -------
            steps         1000     1000     1000
            mutations      "mut1"   "mut2"   "mut1"
            steps_per_second  1000     1000     1000
            ================================
        """
        headers = (
            ["Metric"] + [f"Agent {idx}" for idx in self.metrics.indices] + ["Mean"]
        )

        # Evaluation metrics
        table_data: list = [
            [row.name]
            + [self._fmt(v) for v in row.agent_values]
            + [self._fmt(row.pop_mean)]
            for row in eval_rows
        ]
        table_data.append(SEPARATING_LINE)

        # Training metrics
        table_data.extend(
            [
                [row.name]
                + [self._fmt(v) for v in row.agent_values]
                + [self._fmt(row.pop_mean)]
                for row in train_rows
            ]
        )
        table_data.append(SEPARATING_LINE)

        # Hyperparameters
        if self.metrics.hyperparameters:
            hp_names = list(self.metrics.hyperparameters[0].keys())
            for hp_name in hp_names:
                hp_vals = [
                    self._fmt(agent_hps.get(hp_name, float("nan")))
                    for agent_hps in self.metrics.hyperparameters
                ]
                table_data.append([hp_name, *hp_vals, ""])

        # Metadata
        steps_row = ["steps"] + [str(s) for s in self.metrics.steps] + [""]
        muts_row = ["mutations"] + [str(m) for m in self.metrics.mutations] + [""]
        fps_row = (
            ["steps_per_second"]
            + [self._fmt(s) for s in self.metrics.steps_per_second]
            + [self._fmt(self.metrics.fps)]
        )
        table_data.extend([steps_row, muts_row, fps_row])

        # Render the table
        table_str = tabulate(
            table_data,
            headers=headers,
            tablefmt="simple",
            stralign="right",
            numalign="right",
        )

        banner_text = f"Global Steps {self.metrics.global_step}"
        width = max(len(line) for line in table_str.splitlines())
        border = "=" * width

        return f"{border}\n{banner_text.center(width)}\n{border}\n{table_str}"


class Population(Generic[AgentT]):
    """Population wrapper for evolutionary agent management.

    Owns the logger pipeline and provides a single :meth:`report_metrics`
    entry-point that gathers per-agent data, builds a :class:`MetricsReport`,
    and dispatches it to all configured loggers.

    :param agents: Initial population of RL agents.
    :param min_evo_steps: Minimum evo steps before early stopping.
    :param sum_scores: Whether multi-agent scores are summed across sub-agents.
    :param accelerator: HuggingFace Accelerator for distributed training.
    :param wb: Enable Weights & Biases logging.
    :param verbose: Print training banners to console.
    :param pbar: ``tqdm`` progress bar for console output.
    """

    def __init__(
        self,
        agents: list[AgentT],
        *,
        min_evo_steps: int = 10,
        sum_scores: bool = True,
        accelerator: Accelerator | None = None,
        loggers: list[Logger] | None = None,
    ) -> None:
        if not all(isinstance(agent, type(agents[0])) for agent in agents):
            names = ", ".join(type(a).__name__ for a in agents)
            msg = f"All agents must be instances of the same algorithm. Found: {names}"
            raise ValueError(msg)

        self._agents = agents
        self.min_evo_steps = min_evo_steps
        self.sum_scores = sum_scores
        self.accelerator = accelerator
        self.loggers = loggers or []

        self.last_fitnesses: list[float] = []
        self.evo_steps: int = 0
        self.is_multi_agent: bool = all(
            isinstance(a, MultiAgentRLAlgorithm) for a in agents
        )
        self.additional_metric_names: list[str] = self._agents[
            0
        ].metrics.additional_metrics
        self.agent_ids: list[str] | None = (
            self._agents[0].metrics.agent_ids if self.is_multi_agent else None
        )

    @property
    def agents(self) -> list[AgentT]:
        """Current population of agents."""
        return self._agents

    @property
    def size(self) -> int:
        """Number of agents in the population."""
        return len(self._agents)

    def update(self, agents: list[AgentT]) -> None:
        """Replace the population (e.g. after tournament selection + mutation)."""
        self._agents = agents

    def increment_evo_step(self) -> None:
        """Increment the population-level evo-step counter."""
        self.evo_steps += 1

    def all_below(self, max_steps: int) -> bool:
        """Check if every agent's step count is below *max_steps*."""
        return bool(
            np.less(
                [a.metrics.steps for a in self._agents],
                max_steps,
            ).all(),
        )

    def should_stop(self, target: float | None) -> bool:
        """Check if all agents consistently exceed the target fitness."""
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

    def report_metrics(self) -> MetricsReport:
        """Gather, format, and log population metrics.

        :returns: The metrics report.
        :rtype: MetricsReport
        """
        metrics = self._gather_metrics()
        report = MetricsReport(metrics)

        # Write report to all defined loggers
        for logger in self.loggers:
            logger.write(report)

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
        # Extract last reported fitness for each individual
        fitnesses = [
            float(agent.fitness[-1]) if agent.fitness else float("nan")
            for agent in self.agents
        ]
        self.last_fitnesses = fitnesses

        steps = [agent.metrics.steps for agent in self.agents]
        if self.accelerator is not None and self.accelerator.is_main_process:
            steps = [step * self.accelerator.state.num_processes for step in steps]

        steps_per_second = [agent.metrics.steps_per_second for agent in self.agents]
        mutations = [agent.mut for agent in self.agents]
        indices = [agent.index for agent in self.agents]
        return PopulationMetrics(
            fitnesses=fitnesses,
            scores=self._collect_scores(),
            steps=steps,
            steps_per_second=steps_per_second,
            mutations=mutations,
            indices=indices,
            additional_metrics=self._collect_additional_metrics(),
            hyperparameters=self._collect_hyperparameters(),
        )

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

    def _collect_scores(self) -> list[float | dict[str, float]]:
        """Compute per-agent mean scores.

        :returns: List of mean scores for each agent or per-sub-agent scores for multi-agent systems.
        :rtype: list[float | dict[str, float]]
        """
        if not self.is_multi_agent or self.sum_scores:
            return [
                float(np.mean(agent.metrics.scores))
                if agent.metrics.scores
                else float("nan")
                for agent in self.agents
            ]

        # Multi-agent, non-summed: per-sub-agent mean scores
        scores: list[dict[str, float]] = []
        for agent in self.agents:
            if agent.metrics.scores:
                arr = np.mean(np.array(agent.metrics.scores), axis=0)
                scores.append(
                    {
                        agent_id: float(arr[idx])
                        for idx, agent_id in enumerate(self.agent_ids)
                    }
                )
            else:
                scores.append(dict.fromkeys(self.agent_ids, float("nan")))

        return scores

    def _collect_additional_metrics(self) -> list[dict[str, float]]:
        """Compute per-agent mean values for all registered additional metrics.

        :returns: List of dictionaries containing per-agent mean values for all registered additional metrics.
        :rtype: list[dict[str, float]]
        """
        result: list[dict[str, float]] = []
        for agent in self.agents:
            d: dict[str, float] = {}
            for name in self.additional_metric_names:
                if self.is_multi_agent and self.agent_ids is not None:
                    for agent_id in self.agent_ids:
                        d[f"{agent_id}/{name}"] = agent.metrics.get_mean(name, agent_id)
                else:
                    d[name] = agent.metrics.get_mean(name)
            result.append(d)
        return result
