# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for agilerl.utils.population_utils formatting helpers."""

from agilerl.utils.population_utils import ScalarMetricRow, fmt_value


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
