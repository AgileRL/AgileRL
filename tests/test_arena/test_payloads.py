"""Tests for agilerl.arena.payloads — filename_from_disposition, resolve_metrics_output_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from agilerl.arena.payloads import (
    filename_from_disposition,
    resolve_metrics_output_path,
)


# ---------------------------------------------------------------------------
# filename_from_disposition
# ---------------------------------------------------------------------------
class TestFilenameFromDisposition:
    def test_standard_form(self):
        assert (
            filename_from_disposition("attachment; filename=report.csv") == "report.csv"
        )

    def test_quoted_form(self):
        assert (
            filename_from_disposition('attachment; filename="metrics.zip"')
            == "metrics.zip"
        )

    def test_no_match_returns_none(self):
        assert filename_from_disposition("inline") is None

    def test_none_returns_none(self):
        assert filename_from_disposition(None) is None

    def test_empty_returns_none(self):
        assert filename_from_disposition("") is None


# ---------------------------------------------------------------------------
# resolve_metrics_output_path
# ---------------------------------------------------------------------------
class TestResolveMetricsOutputPath:
    def test_explicit_output_file_takes_priority(self, tmp_path):
        explicit = tmp_path / "my_output.csv"
        result = resolve_metrics_output_path(
            experiment_id=1,
            payload=b"data",
            content_type="text/csv",
            disposition='attachment; filename="server.csv"',
            output_file=explicit,
        )
        assert result == explicit

    def test_disposition_fallback(self):
        result = resolve_metrics_output_path(
            experiment_id=42,
            payload=b"data",
            content_type="text/csv",
            disposition='attachment; filename="exp42_metrics.csv"',
            output_file=None,
        )
        assert result == Path("exp42_metrics.csv")

    def test_zip_content_type_heuristic(self):
        result = resolve_metrics_output_path(
            experiment_id=7,
            payload=b"PK\x03\x04fake",
            content_type="application/zip",
            disposition=None,
            output_file=None,
        )
        assert result == Path("experiment_7_metrics.zip")

    def test_pk_magic_bytes_heuristic(self):
        result = resolve_metrics_output_path(
            experiment_id=3,
            payload=b"PK\x03\x04data",
            content_type=None,
            disposition=None,
            output_file=None,
        )
        assert result == Path("experiment_3_metrics.zip")

    def test_csv_fallback(self):
        result = resolve_metrics_output_path(
            experiment_id=5,
            payload=b"col1,col2\n1,2\n",
            content_type="text/csv",
            disposition=None,
            output_file=None,
        )
        assert result == Path("experiment_5_metrics.csv")

    def test_no_hints_defaults_to_csv(self):
        result = resolve_metrics_output_path(
            experiment_id=9,
            payload=b"some data",
            content_type=None,
            disposition=None,
            output_file=None,
        )
        assert result == Path("experiment_9_metrics.csv")
