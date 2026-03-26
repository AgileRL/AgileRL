"""Comprehensive tests for agilerl.logger — all Logger implementations.

Covers StdOutLogger, CSVLogger, WandbLogger, and TensorboardLogger
including accelerator paths, edge cases, and data integrity.
"""

from __future__ import annotations

import csv
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from agilerl.logger import CSVLogger, StdOutLogger, TensorboardLogger, WandbLogger
from agilerl.population import MetricsReport, PopulationMetrics


def _make_pop_metrics(**overrides) -> PopulationMetrics:
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
def report() -> MetricsReport:
    return MetricsReport(_make_pop_metrics())


def _make_accelerator(*, is_main: bool) -> MagicMock:
    acc = MagicMock()
    acc.is_main_process = is_main
    return acc


# ---------------------------------------------------------------------------
# StdOutLogger
# ---------------------------------------------------------------------------


class TestStdOutLogger:
    def test_write_calls_pbar_write(self, report):
        pbar = MagicMock()
        logger = StdOutLogger(pbar)
        logger.write(report)
        pbar.write.assert_called_once()

    def test_write_passes_str_of_report(self, report):
        pbar = MagicMock()
        StdOutLogger(pbar).write(report)
        written = pbar.write.call_args[0][0]
        assert written == str(report)

    def test_write_content_includes_banner(self, report):
        pbar = MagicMock()
        StdOutLogger(pbar).write(report)
        text = pbar.write.call_args[0][0]
        assert "Global Steps" in text
        assert "Agent 0" in text

    def test_multiple_writes(self, report):
        pbar = MagicMock()
        logger = StdOutLogger(pbar)
        logger.write(report)
        logger.write(report)
        assert pbar.write.call_count == 2

    def test_close_is_noop(self):
        pbar = MagicMock()
        logger = StdOutLogger(pbar)
        logger.close()


# ---------------------------------------------------------------------------
# CSVLogger
# ---------------------------------------------------------------------------


class TestCSVLogger:
    def test_creates_file_with_header(self, tmp_path, report):
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        logger.write(report)
        logger.close()

        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "eval/mean_fitness" in header
        assert "train/global_step" in header

    def test_row_values_match_to_dict(self, tmp_path, report):
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        logger.write(report)
        logger.close()

        expected = report.to_dict()
        with open(path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        for key, val in expected.items():
            assert row[key] == str(val)

    def test_multiple_writes_append_rows(self, tmp_path, report):
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        logger.write(report)
        logger.write(report)
        logger.write(report)
        logger.close()

        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3

    def test_data_readable_before_close(self, tmp_path, report):
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        logger.write(report)
        with open(path) as f:
            content = f.read()
        assert "eval/mean_fitness" in content
        logger.close()

    def test_close_then_reopen_overwrites(self, tmp_path, report):
        path = tmp_path / "log.csv"
        CSVLogger(path).write(report)

        logger2 = CSVLogger(path)
        logger2.write(report)
        logger2.close()

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_close_without_write_is_safe(self, tmp_path):
        logger = CSVLogger(tmp_path / "empty.csv")
        logger.close()

    def test_close_idempotent(self, tmp_path, report):
        path = tmp_path / "log.csv"
        logger = CSVLogger(path)
        logger.write(report)
        logger.close()
        logger.close()

    def test_accepts_string_path(self, tmp_path, report):
        path = str(tmp_path / "log.csv")
        logger = CSVLogger(path)
        logger.write(report)
        logger.close()

        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------


class TestWandbLogger:
    @patch("agilerl.logger.wandb")
    def test_write_no_accelerator(self, mock_wandb, report):
        logger = WandbLogger()
        logger.write(report)
        mock_wandb.log.assert_called_once_with(report.to_dict())

    @patch("agilerl.logger.wandb")
    def test_close_no_accelerator(self, mock_wandb):
        WandbLogger().close()
        mock_wandb.finish.assert_called_once()

    @patch("agilerl.logger.wandb")
    def test_multiple_writes(self, mock_wandb, report):
        logger = WandbLogger()
        logger.write(report)
        logger.write(report)
        assert mock_wandb.log.call_count == 2

    @patch("agilerl.logger.wandb")
    def test_write_with_accelerator_main(self, mock_wandb, report):
        acc = _make_accelerator(is_main=True)
        logger = WandbLogger(accelerator=acc)
        logger.write(report)
        mock_wandb.log.assert_called_once_with(report.to_dict())
        assert acc.wait_for_everyone.call_count == 2

    @patch("agilerl.logger.wandb")
    def test_write_with_accelerator_non_main(self, mock_wandb, report):
        acc = _make_accelerator(is_main=False)
        logger = WandbLogger(accelerator=acc)
        logger.write(report)
        mock_wandb.log.assert_not_called()
        assert acc.wait_for_everyone.call_count == 2

    @patch("agilerl.logger.wandb")
    def test_close_with_accelerator_main(self, mock_wandb):
        acc = _make_accelerator(is_main=True)
        WandbLogger(accelerator=acc).close()
        mock_wandb.finish.assert_called_once()
        assert acc.wait_for_everyone.call_count == 2

    @patch("agilerl.logger.wandb")
    def test_close_with_accelerator_non_main(self, mock_wandb):
        acc = _make_accelerator(is_main=False)
        WandbLogger(accelerator=acc).close()
        mock_wandb.finish.assert_not_called()
        assert acc.wait_for_everyone.call_count == 2


# ---------------------------------------------------------------------------
# TensorboardLogger
# ---------------------------------------------------------------------------


class TestTensorboardLogger:
    @patch("agilerl.logger.SummaryWriter")
    def test_write_calls_add_scalar_for_numeric(self, MockWriter, report):
        writer = MagicMock()
        MockWriter.return_value = writer

        TensorboardLogger(log_dir="/tmp/tb").write(report)

        scalar_keys = {c.args[0] for c in writer.add_scalar.call_args_list}
        assert "eval/mean_fitness" in scalar_keys
        assert "eval/best_fitness" in scalar_keys
        assert "train/global_step" in scalar_keys
        assert "train/mean_score" in scalar_keys

    @patch("agilerl.logger.SummaryWriter")
    def test_write_passes_global_step(self, MockWriter, report):
        writer = MagicMock()
        MockWriter.return_value = writer

        TensorboardLogger(log_dir="/tmp/tb").write(report)

        expected_step = report.to_dict()["train/global_step"]
        for c in writer.add_scalar.call_args_list:
            assert c.kwargs["global_step"] == expected_step

    @patch("agilerl.logger.SummaryWriter")
    def test_write_skips_non_numeric_values(self, MockWriter):
        writer = MagicMock()
        MockWriter.return_value = writer

        pm = _make_pop_metrics(mutations=["None", "sigma"])
        report = MetricsReport(pm)
        TensorboardLogger(log_dir="/tmp/tb").write(report)

        for c in writer.add_scalar.call_args_list:
            assert isinstance(c.args[1], (int, float))

    @patch("agilerl.logger.SummaryWriter")
    def test_write_flushes(self, MockWriter, report):
        writer = MagicMock()
        MockWriter.return_value = writer

        TensorboardLogger(log_dir="/tmp/tb").write(report)
        writer.flush.assert_called_once()

    @patch("agilerl.logger.SummaryWriter")
    def test_write_histograms(self, MockWriter):
        writer = MagicMock()
        MockWriter.return_value = writer

        arr = np.array([1, 2, 3, 4, 5])
        pm = _make_pop_metrics(
            nonscalar_additional_metrics=[{"action_dist": arr}, {"action_dist": None}]
        )
        TensorboardLogger(log_dir="/tmp/tb").write(MetricsReport(pm))

        hist_keys = {c.args[0] for c in writer.add_histogram.call_args_list}
        assert "train/agent_0/action_dist" in hist_keys
        assert "train/agent_1/action_dist" not in hist_keys

    @patch("agilerl.logger.SummaryWriter")
    def test_write_no_histograms_when_empty(self, MockWriter, report):
        writer = MagicMock()
        MockWriter.return_value = writer

        TensorboardLogger(log_dir="/tmp/tb").write(report)
        writer.add_histogram.assert_not_called()

    @patch("agilerl.logger.SummaryWriter")
    def test_histogram_global_step(self, MockWriter):
        writer = MagicMock()
        MockWriter.return_value = writer

        pm = _make_pop_metrics(nonscalar_additional_metrics=[{"h": np.array([1, 2])}])
        TensorboardLogger(log_dir="/tmp/tb").write(MetricsReport(pm))

        expected_step = pm.global_step
        for c in writer.add_histogram.call_args_list:
            assert c.kwargs["global_step"] == expected_step

    @patch("agilerl.logger.SummaryWriter")
    def test_close_no_accelerator(self, MockWriter):
        writer = MagicMock()
        MockWriter.return_value = writer

        TensorboardLogger(log_dir="/tmp/tb").close()
        writer.close.assert_called_once()

    @patch("agilerl.logger.SummaryWriter")
    def test_write_with_accelerator_main(self, MockWriter, report):
        writer = MagicMock()
        MockWriter.return_value = writer
        acc = _make_accelerator(is_main=True)

        TensorboardLogger(log_dir="/tmp/tb", accelerator=acc).write(report)

        assert writer.add_scalar.call_count > 0
        writer.flush.assert_called_once()
        assert acc.wait_for_everyone.call_count == 2

    @patch("agilerl.logger.SummaryWriter")
    def test_write_with_accelerator_non_main(self, MockWriter, report):
        writer = MagicMock()
        MockWriter.return_value = writer
        acc = _make_accelerator(is_main=False)

        TensorboardLogger(log_dir="/tmp/tb", accelerator=acc).write(report)

        writer.add_scalar.assert_not_called()
        writer.flush.assert_not_called()
        assert acc.wait_for_everyone.call_count == 2

    @patch("agilerl.logger.SummaryWriter")
    def test_close_with_accelerator_main(self, MockWriter):
        writer = MagicMock()
        MockWriter.return_value = writer
        acc = _make_accelerator(is_main=True)

        TensorboardLogger(log_dir="/tmp/tb", accelerator=acc).close()
        writer.close.assert_called_once()
        assert acc.wait_for_everyone.call_count == 2

    @patch("agilerl.logger.SummaryWriter")
    def test_close_with_accelerator_non_main(self, MockWriter):
        writer = MagicMock()
        MockWriter.return_value = writer
        acc = _make_accelerator(is_main=False)

        TensorboardLogger(log_dir="/tmp/tb", accelerator=acc).close()
        writer.close.assert_not_called()
        assert acc.wait_for_everyone.call_count == 2

    @patch("agilerl.logger.SummaryWriter")
    def test_multiple_writes(self, MockWriter, report):
        writer = MagicMock()
        MockWriter.return_value = writer

        logger = TensorboardLogger(log_dir="/tmp/tb")
        logger.write(report)
        logger.write(report)

        assert writer.flush.call_count == 2

    @patch("agilerl.logger.SummaryWriter")
    def test_log_dir_creates_timestamped_subdir(self, MockWriter):
        writer = MagicMock()
        MockWriter.return_value = writer

        logger = TensorboardLogger(log_dir="/tmp/tb_root")

        log_dir_arg = MockWriter.call_args.kwargs["log_dir"]
        assert log_dir_arg.startswith("/tmp/tb_root/")
        assert len(log_dir_arg) > len("/tmp/tb_root/")

    def test_raises_when_summary_writer_unavailable(self):
        with patch("agilerl.logger.SummaryWriter", None):
            with pytest.raises(ImportError, match="TensorBoard is not installed"):
                TensorboardLogger(log_dir="/tmp/tb")
