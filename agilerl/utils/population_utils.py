"""Utility functions for population management."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def get_nested_mean(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Transpose a list of dicts and compute the mean per key.

    Works for both plain scalar keys (``"loss"``) and flattened multi-agent
    keys (``"loss/a0"``, ``"loss/a1"``).

    :param metrics: List of dictionaries containing scalar metric snapshots.
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


def get_values_for_key(snapshots: list[dict[str, float]], key: str) -> list[float]:
    """Get the values for a key from a list of metric snapshots.

    :param snapshots: List of metric snapshots.
    :type snapshots: list[dict[str, float]]
    :param key: The key to get the values for.
    :type key: str
    :returns: List of values for the key.
    :rtype: list[float]

    Example::

        >>> snapshots = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        >>> get_values_for_key(snapshots, "a")
        [1.0, 3.0]
    """
    return [snapshot.get(key, float("nan")) for snapshot in snapshots]


def fmt_value(value: float) -> str:
    """Format a numeric value for table display, using scientific notation for extremes,
    and handling NaN values specially.

    :param value: The value to format.
    :type value: float
    :returns: The formatted value.
    :rtype: str
    """
    if np.isnan(value):
        return "-"
    if isinstance(value, (int, np.integer)) or (
        isinstance(value, float) and value.is_integer()
    ):
        return str(int(value))
    if abs(value) >= 1e6:
        return f"{value:.2e}"
    if abs(value) < 5e-5:
        return f"{value:.2e}"
    return f"{value:.4f}"


@dataclass(frozen=True)
class MetricRow:
    """Base class for metric rows.

    :param name: Metric name.
    :type name: str
    """

    name: str

    @property
    def fmt_name(self) -> str:
        """Get the formatted name of the metric.

        :returns: The formatted name of the metric.
        :rtype: str
        """
        return self.name.replace("eval/", "").replace("train/", "")


@dataclass(frozen=True)
class ScalarMetricRow(MetricRow):
    """Scalar metric row with agent values and population mean.

    :param name: Metric name.
    :type name: str
    :param agent_values: List of metric values per agent.
    :type agent_values: list[float]
    :param pop_mean: Mean of the metric values for the population.
    :type pop_mean: float
    """

    agent_values: list[float]
    pop_mean: float

    @property
    def fmt_values(self) -> list[str]:
        """Get the formatted agent values of the metric.

        :returns: The formatted agent values of the metric.
        :rtype: list[str]
        """
        return [fmt_value(value) for value in self.agent_values]

    @property
    def fmt_mean(self) -> str:
        """Get the formatted population mean of the metric.

        :returns: The formatted population mean of the metric.
        :rtype: str
        """
        return fmt_value(self.pop_mean)


@dataclass(frozen=True)
class NestedMetricRow(MetricRow):
    """Nested metric row with child scalar rows. Useful for cases where you are
    reporting the same metric for sub-agents in multi-agent settings.

    :param name: Parent metric name.
    :type name: str
    :param children: Child rows (typically per sub-agent).
    :type children: list[ScalarMetricRow]
    """

    name: str
    children: list[ScalarMetricRow]


def build_metric_row(
    name: str,
    values: list[float] | list[dict[str, float]],
    mean_value: float | dict[str, float],
) -> ScalarMetricRow | NestedMetricRow:
    """Build a row from per-agent values, auto-detecting scalar vs nested layout.

    :param name: The name of the metric.
    :type name: str
    :param values: The values of the metric.
    :type values: list[float] | list[dict[str, float]]
    :param mean_value: The mean value of the metric.
    :type mean_value: float | dict[str, float]
    :returns: A row from per-agent values, auto-detecting scalar vs nested layout.
    :rtype: ScalarMetricRow | NestedMetricRow
    """
    if values and isinstance(values[0], dict):
        children = [
            ScalarMetricRow(
                name=child_name,
                agent_values=[series[child_name] for series in values],
                pop_mean=(
                    mean_value.get(child_name, float("nan"))
                    if isinstance(mean_value, dict)
                    else float("nan")
                ),
            )
            for child_name in values[0]
        ]
        return NestedMetricRow(name=name, children=children)

    return ScalarMetricRow(
        name=name,
        agent_values=[float(value) for value in values],
        pop_mean=float(mean_value)
        if not isinstance(mean_value, dict)
        else float("nan"),
    )


def partition_nested_metric_keys(
    metric_keys: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Split metric keys into scalar names and nested (multi-agent) groups.

    Keys containing ``/`` are treated as ``metric_name/agent_id`` pairs and
    grouped by metric name.  All other keys are plain scalars.

    :param metric_keys: List of metric keys.
    :type metric_keys: list[str]
    :returns: Tuple containing scalar metric names and nested metric IDs, where `nested_groups` maps
        each parent metric name to its list of agent IDs.
    :rtype: tuple[list[str], dict[str, list[str]]]
    """
    scalar_metric_names = []
    nested_metric_ids = {}
    for key in metric_keys:
        if "/" in key:
            metric_name, agent_id = key.split("/", 1)
            nested_metric_ids.setdefault(metric_name, []).append(agent_id)
        else:
            scalar_metric_names.append(key)
    return scalar_metric_names, nested_metric_ids
