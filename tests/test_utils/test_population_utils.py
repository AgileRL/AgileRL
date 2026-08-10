# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the agilerl.utils.population_utils aggregation and formatting helpers."""

import numpy as np
import pytest

from agilerl.utils.population_utils import ScalarMetricRow, fmt_value, scalar_fitness


class TestScalarMetricRowFormatting:
    def test_fmt_values_formats_each_agent_value(self):
        row = ScalarMetricRow(
            name="train/reward",
            agent_values=[1.2345, 2000.5, 0.0],
            pop_mean=667.41,
        )
        assert row.fmt_values == [fmt_value(v) for v in row.agent_values]
        # Spot-check the underlying formatting rules are applied per value:
        # plain float, large non-integer (scientific notation), integer-valued.
        assert row.fmt_values == ["1.2345", "2.00e+03", "0"]

    def test_fmt_mean_formats_population_mean(self):
        row = ScalarMetricRow(
            name="eval/score",
            agent_values=[1.0, 2.0],
            pop_mean=1.5,
        )
        assert row.fmt_mean == fmt_value(1.5)
        assert row.fmt_mean == "1.5000"


class TestScalarFitness:
    @pytest.mark.parametrize(
        ("fitness", "expected"),
        [
            # Multi-agent (sum_scores=False) fitness collapses to the mean.
            ({"agent_0": 2.0, "agent_1": 4.0}, 3.0),
            ([1.0, 2.0, 3.0], 2.0),
            ((4.0, 6.0), 5.0),
            (np.array([0.0, 10.0]), 5.0),
            (np.array(6.0), 6.0),
            (7.5, 7.5),
            (np.float32(2.5), 2.5),
        ],
        ids=["dict", "list", "tuple", "array", "0d-array", "scalar", "numpy-scalar"],
    )
    def test_reduces_fitness_to_the_mean(self, fitness, expected):
        result = scalar_fitness(fitness)
        assert result == pytest.approx(expected)
        assert isinstance(result, float)

    def test_preserves_the_ordering_of_vector_fitnesses(self):
        rows = [{"a": 9.0, "b": 0.0}, {"a": 0.0, "b": 5.0}, {"a": 4.0, "b": 4.0}]
        assert sorted(rows, key=scalar_fitness) == [rows[1], rows[2], rows[0]]
