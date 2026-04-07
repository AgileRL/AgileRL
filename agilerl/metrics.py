"""Per-agent training metrics."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

AdditionalMetricType = list[float] | dict[str, list[float]]


class BaseMetrics(ABC):
    """Base class for per-agent metrics.

    :param fitness_window: Maximum number of fitness values to store.
    :type fitness_window: int
    """

    def __init__(self, *, fitness_window: int = 100) -> None:
        self._additional_metrics: dict[str, AdditionalMetricType] = {}
        self._nonscalar_metrics: dict[str, list | dict[str, list]] = {}
        self._hyperparameters: dict[str, float] = {}
        self._evo_start_time: float = 0.0
        self.steps_per_second: float = 0.0
        self.steps: int = 0
        self.scores: list[float] = []
        self.fitness: deque[float] = deque(maxlen=fitness_window)

    @abstractmethod
    def _init_metric(self, name: str) -> None:
        """Initialize storage for a newly registered metric."""
        raise NotImplementedError

    @abstractmethod
    def _init_nonscalar_metric(self, name: str) -> None:
        """Initialize storage for a newly registered non-scalar metric."""
        raise NotImplementedError

    @abstractmethod
    def log(self, name: str, *, value: float) -> None:
        """Log a value to the accumulator for a registered metric."""
        raise NotImplementedError

    @abstractmethod
    def log_histogram(self, name: str, values: np.ndarray) -> None:
        """Extend the accumulator with raw sample values for a histogram metric."""
        raise NotImplementedError

    @abstractmethod
    def get_histogram(self, name: str, *args: str) -> np.ndarray | None:
        """Return accumulated raw values for a histogram metric."""
        raise NotImplementedError

    @property
    def additional_metrics(self) -> list[str]:
        """All registered scalar metric names."""
        return list(self._additional_metrics.keys())

    @property
    def nonscalar_metrics(self) -> list[str]:
        """All registered non-scalar metric names."""
        return list(self._nonscalar_metrics.keys())

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.clear()

    def __eq__(self, other: object) -> bool:
        """Compare metrics by tracked state.

        :param other: Other metrics to compare.
        :type other: object
        :returns: True if the metrics are equal, False otherwise.
        :rtype: bool
        """
        if not isinstance(other, BaseMetrics):
            return NotImplemented
        return (
            self._additional_metrics == other._additional_metrics
            and self._nonscalar_metrics == other._nonscalar_metrics
            and self._hyperparameters == other._hyperparameters
            and self.steps_per_second == other.steps_per_second
            and self.steps == other.steps
            and self.scores == other.scores
            and list(self.fitness) == list(other.fitness)
        )

    def _check_name_available(self, name: str) -> None:
        if name in self._additional_metrics or name in self._nonscalar_metrics:
            msg = f"Metric '{name}' is already registered."
            raise ValueError(msg)

    def register(self, name: str) -> None:
        """Register a named scalar metric to be tracked.

        :param name: Unique metric name (e.g. ``"loss"``, ``"entropy"``).
        :type name: str
        :raises ValueError: If *name* is already registered.
        """
        self._check_name_available(name)
        self._init_metric(name)

    def register_histogram(self, name: str) -> None:
        """Register a named non-scalar (histogram) metric.

        Non-scalar metrics store array data (e.g. action distributions) and
        are only reported by loggers that support them (TensorBoard).

        :param name: Unique metric name (e.g. ``"action_distribution"``).
        :type name: str
        :raises ValueError: If *name* is already registered.
        """
        self._check_name_available(name)
        self._init_nonscalar_metric(name)

    def clear(self) -> None:
        """Clear scores and all additional metric accumulators."""
        self.scores.clear()

    def add_fitness(self, value: float) -> None:
        """Add a fitness value to the fitness history.

        :param value: Fitness value to add.
        :type value: float
        """
        self.fitness.append(value)

    def add_scores(self, scores: list[float]) -> None:
        """Add scores to the metrics.

        :param scores: List of scores to add.
        :type scores: list[float]
        """
        self.scores.extend(scores)

    def increment_steps(self, n: int) -> None:
        """Increment the cumulative environment step count.

        :param n: Number of steps to add.
        """
        self.steps += n

    def init_evo_step(self) -> None:
        """Snapshot state at the start of an evolution step."""
        self._evo_start_time = time.monotonic()

    def finalize_evo_step(self, num_steps: int) -> None:
        """Compute fps from the delta since :meth:`init_evo_step`.

        :param num_steps: Number of steps taken during the evo step.
        :type num_steps: int
        """
        elapsed = time.monotonic() - self._evo_start_time
        self.steps_per_second = num_steps / max(elapsed, 1e-12)
        self.increment_steps(num_steps)


class AgentMetrics(BaseMetrics):
    """Tracks training metrics for a single-agent RL algorithm instance.

    :param fitness_window: Maximum number of fitness values to store.
    :type fitness_window: int
    """

    def __init__(self, *, fitness_window: int = 100) -> None:
        super().__init__(fitness_window=fitness_window)

    def _init_metric(self, name: str) -> None:
        """Initialize storage for a newly registered metric."""
        self._additional_metrics[name] = []

    def _init_nonscalar_metric(self, name: str) -> None:
        self._nonscalar_metrics[name] = []

    def log(self, name: str, value: float) -> None:
        """Append a value to the accumulator for a registered metric.

        :param name: Previously registered metric name.
        :type name: str
        :param value: Scalar metric value.
        :type value: float

        Example:
        >>> metrics = AgentMetrics()
        >>> metrics.log("loss", 0.1)
        >>> metrics.log("loss", 0.2)
        >>> metrics.get_mean("loss")
        """
        self._additional_metrics[name].append(float(value))

    def log_histogram(self, name: str, values: np.ndarray) -> None:
        """Extend the accumulator with raw sample values for a histogram metric.

        Values are accumulated across calls so that :meth:`get_histogram`
        returns the full distribution for the entire evo step.

        :param name: Previously registered non-scalar metric name.
        :type name: str
        :param values: Array of raw sample values (e.g. action indices).
        :type values: numpy.ndarray
        """
        self._nonscalar_metrics[name].extend(values.tolist())

    def get_mean(self, name: str) -> float:
        """Return the mean of accumulated values for a registered metric.

        :param name: Previously registered metric name.
        :type name: str
        :returns: Mean of accumulated values for the registered metric.
        :rtype: float
        """
        values = self._additional_metrics[name]
        if not values:
            return float("nan")

        return np.mean(values)

    def get_histogram(self, name: str) -> np.ndarray | None:
        """Return the accumulated raw values for a histogram metric.

        :param name: Previously registered non-scalar metric name.
        :type name: str
        :returns: Array of all accumulated values, or ``None`` if empty.
        :rtype: numpy.ndarray | None
        """
        values = self._nonscalar_metrics[name]
        if not values:
            return None
        return np.asarray(values)

    def clear(self) -> None:
        """Clear scores and all additional metric accumulators."""
        super().clear()
        for name in self._additional_metrics:
            self._additional_metrics[name] = []
        for name in self._nonscalar_metrics:
            self._nonscalar_metrics[name] = []


class MultiAgentMetrics(BaseMetrics):
    """Tracks training metrics for multi-agent RL algorithms.

    Assumes that we log metrics for each sub-agent separately. For settings
    where we have homogeneous/grouped agents (e.g. speaker_0, speaker_1, listener_0, listener_1),
    we assume metrics are logged for the group as a whole.

    :param agent_ids: Sub-agent identifiers from the environment.
    :type agent_ids: list[str]
    :param fitness_window: Maximum number of fitness values to store.
    :type fitness_window: int
    """

    def __init__(self, agent_ids: list[str], *, fitness_window: int = 100) -> None:
        super().__init__(fitness_window=fitness_window)

        self.agent_ids: list[str] = list(agent_ids)
        self.scores: list[float] | list[list[float]] = []

    def _init_metric(self, name: str) -> None:
        """Initialize storage for a newly registered metric.

        :param name: Unique metric name (e.g. ``"loss"``, ``"entropy"``).
        :type name: str
        """
        self._additional_metrics[name] = {agent_id: [] for agent_id in self.agent_ids}

    def _init_nonscalar_metric(self, name: str) -> None:
        self._nonscalar_metrics[name] = {agent_id: [] for agent_id in self.agent_ids}

    def log(self, name: str, agent_id: str, value: float) -> None:
        """Append a value to the accumulator for a registered metric and sub-agent.

        Example:
        >>> metrics = MultiAgentMetrics(["speaker_0", "speaker_1", "listener_0", "listener_1"])
        >>> metrics.log("loss", "speaker_0", 0.1)
        >>> metrics.log("loss", "speaker_0", 0.2)
        >>> metrics.get_mean("loss", "speaker_0")

        :param name: Previously registered metric name.
        :type name: str
        :param agent_id: Sub-agent identifier.
        :type agent_id: str
        :param value: Scalar metric value.
        :type value: float
        """
        self._additional_metrics[name][agent_id].append(float(value))

    def log_histogram(self, name: str, agent_id: str, values: np.ndarray) -> None:
        """Extend the accumulator with raw sample values for a histogram metric.

        :param name: Previously registered non-scalar metric name.
        :type name: str
        :param agent_id: Sub-agent identifier.
        :type agent_id: str
        :param values: Array of raw sample values (e.g. action indices).
        :type values: numpy.ndarray
        """
        self._nonscalar_metrics[name][agent_id].extend(values.tolist())

    def get_mean(self, name: str, agent_id: str) -> float:
        """Return the mean of accumulated values for a registered metric and sub-agent.

        :param name: Previously registered metric name.
        :type name: str
        :param agent_id: Sub-agent identifier.
        :type agent_id: str
        :returns: Mean of accumulated values for the metric and sub-agent.
        :rtype: float
        """
        values = self._additional_metrics[name][agent_id]
        if not values:
            return float("nan")

        return np.mean(values)

    def get_histogram(self, name: str, agent_id: str) -> np.ndarray | None:
        """Return accumulated raw values for a histogram metric and sub-agent.

        :param name: Previously registered non-scalar metric name.
        :type name: str
        :param agent_id: Sub-agent identifier.
        :type agent_id: str
        :returns: Array of all accumulated values, or ``None`` if empty.
        :rtype: numpy.ndarray | None
        """
        values = self._nonscalar_metrics[name][agent_id]
        if not values:
            return None
        return np.asarray(values)

    def clear(self) -> None:
        """Clear scores and all additional metric accumulators."""
        super().clear()
        for name in self._additional_metrics:
            self._additional_metrics[name] = {k: [] for k in self.agent_ids}
        for name in self._nonscalar_metrics:
            self._nonscalar_metrics[name] = {k: [] for k in self.agent_ids}
