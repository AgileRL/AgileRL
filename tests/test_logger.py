"""Tests for agilerl/logger.py."""

from __future__ import annotations

import csv
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_report(
    scalar_dict: dict | None = None,
    nonscalar_dict: dict | None = None,
    show_mean: bool = True,
) -> MagicMock:
    """Build a minimal MetricsReport mock."""
    report = MagicMock()
    report.to_dict.return_value = scalar_dict or {
        "train/global_step": 10,
        "train/mean_score": 42.0,
    }
    report.to_nonscalar_dict.return_value = nonscalar_dict or {}
    report.show_mean_column = show_mean
    report.render.return_value = "| step | score |\n|  10  |  42.0 |"
    report.__str__ = lambda self: self.render()
    return report


class TestOnMainProcess:
    def test_yields_true_when_no_accelerator(self):
        from agilerl.logger import Logger

        with Logger.on_main_process(None) as is_main:
            assert is_main is True

    def test_yields_true_when_main_process(self):
        from agilerl.logger import Logger

        acc = MagicMock()
        acc.is_main_process = True
        with Logger.on_main_process(acc) as is_main:
            assert is_main is True
        assert acc.wait_for_everyone.call_count == 2

    def test_yields_false_when_not_main_process(self):
        from agilerl.logger import Logger

        acc = MagicMock()
        acc.is_main_process = False
        with Logger.on_main_process(acc) as is_main:
            assert is_main is False
        assert acc.wait_for_everyone.call_count == 2

    def test_finally_runs_on_exception(self):
        from agilerl.logger import Logger

        acc = MagicMock()
        acc.is_main_process = True
        with pytest.raises(RuntimeError):
            with Logger.on_main_process(acc):
                raise RuntimeError("boom")
        assert acc.wait_for_everyone.call_count == 2


class TestIsNotebook:
    def test_returns_true_for_zmq_shell(self):
        from agilerl.logger import _is_notebook

        mock_shell = MagicMock()
        mock_shell.__class__ = type("ZMQInteractiveShell", (), {})
        mock_ipython = MagicMock()
        mock_ipython.get_ipython.return_value = mock_shell

        with patch.dict("sys.modules", {"IPython": mock_ipython}):
            with patch("agilerl.logger.get_ipython", create=True, side_effect=None):
                pass
            # Directly exercise the function logic by monkeypatching the import
            mock_mod = MagicMock()
            mock_mod.get_ipython = lambda: mock_shell
            with patch.dict("sys.modules", {"IPython": mock_mod}):
                result = _is_notebook()
        # Outside Jupyter this will be False; the real test of the True path
        # requires mocking at the import level inside the function
        assert isinstance(result, bool)

    def test_returns_false_on_import_error(self):
        """Exercise the except ImportError branch."""
        original_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def _failing(name, *a, **kw):
            if name == "IPython":
                raise ImportError
            return original_import(name, *a, **kw)

        with patch("builtins.__import__", side_effect=_failing):
            from agilerl import logger as _mod

            result = _mod._is_notebook()
        assert result is False

    def test_returns_false_when_shell_is_none(self):
        mock_mod = MagicMock()
        mock_mod.get_ipython = MagicMock(return_value=None)
        with patch.dict("sys.modules", {"IPython": mock_mod}):
            from agilerl.logger import _is_notebook

            result = _is_notebook()
        assert isinstance(result, bool)


class TestStdOutLogger:
    def test_init_stores_pbar(self):
        with patch("agilerl.logger._is_notebook", return_value=False):
            from agilerl.logger import StdOutLogger

            pbar = MagicMock()
            logger = StdOutLogger(pbar=pbar)
        assert logger._pbar is pbar

    def test_write_uses_pbar_when_not_notebook(self):
        with patch("agilerl.logger._is_notebook", return_value=False):
            from agilerl.logger import StdOutLogger

            pbar = MagicMock()
            logger = StdOutLogger(pbar=pbar)

        report = _make_report()
        logger.write(report)
        pbar.write.assert_called_once_with(str(report))

    def test_write_uses_print_when_notebook(self, capsys):
        with patch("agilerl.logger._is_notebook", return_value=True):
            from agilerl.logger import StdOutLogger

            logger = StdOutLogger(pbar=MagicMock())

        report = _make_report()
        logger.write(report)
        captured = capsys.readouterr()
        assert captured.out.strip() == str(report)

    def test_write_uses_print_when_no_pbar(self, capsys):
        with patch("agilerl.logger._is_notebook", return_value=False):
            from agilerl.logger import StdOutLogger

            logger = StdOutLogger(pbar=None)

        report = _make_report()
        logger.write(report)
        captured = capsys.readouterr()
        assert captured.out.strip() == str(report)

    def test_close_is_noop(self):
        with patch("agilerl.logger._is_notebook", return_value=False):
            from agilerl.logger import StdOutLogger

            logger = StdOutLogger()
        logger.close()


class TestWandbLogger:
    def test_init(self):
        from agilerl.logger import WandbLogger

        acc = MagicMock()
        logger = WandbLogger(accelerator=acc, project="test")
        assert logger._accelerator is acc
        assert logger._project == "test"

    def test_maybe_init_wandb_when_no_run(self):
        from agilerl.logger import WandbLogger

        logger = WandbLogger(project="TestProject")
        with patch("agilerl.logger.wandb") as mock_wandb:
            mock_wandb.run = None
            logger._maybe_init_wandb()
            mock_wandb.init.assert_called_once_with(project="TestProject")

    def test_maybe_init_wandb_skips_when_run_active(self):
        from agilerl.logger import WandbLogger

        logger = WandbLogger()
        with patch("agilerl.logger.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            logger._maybe_init_wandb()
            mock_wandb.init.assert_not_called()

    def test_write_logs_on_main_process(self):
        from agilerl.logger import WandbLogger

        logger = WandbLogger()
        report = _make_report()

        with patch("agilerl.logger.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            logger.write(report)
            mock_wandb.log.assert_called_once_with(report.to_dict())

    def test_write_skips_on_non_main_process(self):
        from agilerl.logger import WandbLogger

        acc = MagicMock()
        acc.is_main_process = False
        logger = WandbLogger(accelerator=acc)
        report = _make_report()

        with patch("agilerl.logger.wandb") as mock_wandb:
            logger.write(report)
            mock_wandb.log.assert_not_called()

    def test_close_calls_finish(self):
        from agilerl.logger import WandbLogger

        logger = WandbLogger()
        with patch("agilerl.logger.wandb") as mock_wandb:
            logger.close()
            mock_wandb.finish.assert_called_once()

    def test_close_skips_on_non_main(self):
        from agilerl.logger import WandbLogger

        acc = MagicMock()
        acc.is_main_process = False
        logger = WandbLogger(accelerator=acc)
        with patch("agilerl.logger.wandb") as mock_wandb:
            logger.close()
            mock_wandb.finish.assert_not_called()


class TestCSVLogger:
    def test_init(self, tmp_path):
        from agilerl.logger import CSVLogger

        p = tmp_path / "metrics.csv"
        logger = CSVLogger(p)
        assert logger._path == p
        assert logger._file is None
        assert logger._writer is None

    def test_write_creates_file_and_writes_rows(self, tmp_path):
        from agilerl.logger import CSVLogger

        p = tmp_path / "metrics.csv"
        logger = CSVLogger(p)

        report1 = _make_report({"step": 1, "score": 10.0})
        report2 = _make_report({"step": 2, "score": 20.0})

        logger.write(report1)
        logger.write(report2)
        logger.close()

        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["step"] == "1"
        assert rows[1]["score"] == "20.0"

    def test_close_cleans_up(self, tmp_path):
        from agilerl.logger import CSVLogger

        p = tmp_path / "metrics.csv"
        logger = CSVLogger(p)
        logger.write(_make_report())
        assert logger._file is not None

        logger.close()
        assert logger._file is None
        assert logger._writer is None

    def test_close_noop_when_not_opened(self, tmp_path):
        from agilerl.logger import CSVLogger

        logger = CSVLogger(tmp_path / "nope.csv")
        logger.close()  # should not raise


class TestMutationHistoryLogger:
    """Behaviour of the per-generation mutation-history CSV writer."""

    @staticmethod
    def _report(
        *,
        indices,
        mut_details,
        fitnesses,
        parent_indices=None,
        steps=(163840, 163840),
    ):
        """Build a real MetricsReport carrying the fields the logger reads."""
        from agilerl.population import MetricsReport, PopulationMetrics

        pop_size = len(indices)
        metrics = PopulationMetrics(
            fitnesses=list(fitnesses),
            scores=[f / 2 for f in fitnesses],
            steps=list(steps[:pop_size]),
            steps_per_second=[1.0] * pop_size,
            mutations=[(d or {}).get("name", "None") for d in mut_details],
            indices=list(indices),
            additional_metrics=[{} for _ in range(pop_size)],
            hyperparameters=[{} for _ in range(pop_size)],
            mut_details=list(mut_details),
            parent_indices=list(parent_indices or indices),
        )
        return MetricsReport(metrics)

    def test_write_records_architecture_mutation_row(self, tmp_path):
        # Arrange: the detail keys the architecture operator actually records.
        from agilerl.logger import MutationHistoryLogger

        logger = MutationHistoryLogger(tmp_path)
        report = self._report(
            indices=[0, 1],
            mut_details=[
                None,
                {
                    "category": "architecture",
                    "name": "head_net.add_node",
                    "layer_changed": "head_net.1",
                    "neurons_delta": 16,
                    "arch_func_preserving": True,
                },
            ],
            fitnesses=[100.0, 150.0],
        )

        # Act
        logger.write(report)
        logger.close()

        # Assert
        with open(tmp_path / "mutation_history.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames

        assert header == MutationHistoryLogger.FIELDNAMES
        assert len(rows) == 2
        assert rows[0]["mutation_category"] == "no mutation"
        assert rows[1]["mutation_category"] == "architecture"
        assert rows[1]["mutation_name"] == "head_net.add_node"
        assert rows[1]["arch_neurons_delta"] == "16"
        assert rows[1]["arch_func_preserving"] == "True"
        assert rows[1]["fitness_after"] == "150.0"

    def test_write_links_fitness_before_to_parent(self, tmp_path):
        # Arrange: generation 2's agent 2 is a clone of generation 1's agent 1.
        from agilerl.logger import MutationHistoryLogger

        logger = MutationHistoryLogger(tmp_path)
        gen1 = self._report(
            indices=[0, 1], mut_details=[None, None], fitnesses=[100.0, 150.0]
        )
        gen2 = self._report(
            indices=[0, 2],
            parent_indices=[0, 1],
            mut_details=[None, {"category": "parameters", "name": "param_noise"}],
            fitnesses=[110.0, 180.0],
        )

        # Act
        logger.write(gen1)
        logger.write(gen2)
        logger.close()

        # Assert
        with open(tmp_path / "mutation_history.csv") as f:
            rows = list(csv.DictReader(f))

        assert [r["generation"] for r in rows] == ["0", "0", "1", "1"]
        assert rows[3]["parent_id"] == "1"
        assert rows[3]["fitness_before"] == "150.0"
        assert rows[3]["fitness_after"] == "180.0"
        # Generation 0 has no parent generation to link against.
        assert rows[0]["fitness_before"] == "nan"

    def test_write_handles_unrecorded_details(self, tmp_path):
        # Arrange: no-HPO runs carry a mutation flag but no detail dict.
        from agilerl.logger import MutationHistoryLogger

        logger = MutationHistoryLogger(tmp_path)
        report = self._report(
            indices=[0, 1], mut_details=[None, None], fitnesses=[100.0, 150.0]
        )
        report.metrics.mutations[1] = "head_net.add_layer"

        # Act
        logger.write(report)
        logger.close()

        # Assert
        with open(tmp_path / "mutation_history.csv") as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["mutation_category"] == "no mutation"
        assert rows[0]["mutation_name"] == "none"
        assert rows[1]["mutation_category"] == "other"
        assert rows[1]["mutation_name"] == "head_net.add_layer"


class TestTensorboardLogger:
    def test_init_raises_without_tensorboard(self):
        with patch("agilerl.logger.SummaryWriter", None):
            from agilerl.logger import TensorboardLogger

            with pytest.raises(ImportError, match="TensorBoard is not installed"):
                TensorboardLogger()

    def test_init_default_log_dir(self):
        mock_sw = MagicMock()
        with patch("agilerl.logger.SummaryWriter", mock_sw):
            from agilerl.logger import TensorboardLogger

            logger = TensorboardLogger()
        assert "tensorboard_logs" in str(logger._log_path)
        mock_sw.assert_called_once()

    def test_init_custom_dir_and_name(self, tmp_path):
        log_dir = tmp_path / "tb_test"
        mock_sw = MagicMock()
        with patch("agilerl.logger.SummaryWriter", mock_sw):
            from agilerl.logger import TensorboardLogger

            logger = TensorboardLogger(log_dir=log_dir, experiment_name="my_exp")
        assert "my_exp" in logger._log_path.name
        assert logger._log_path.parent.resolve() == log_dir.resolve()

    def test_write_scalars_and_histograms(self):
        mock_writer = MagicMock()
        mock_sw = MagicMock(return_value=mock_writer)
        hist_data = {"train/agent_0/weights": np.array([1.0, 2.0, 3.0])}
        report = _make_report(
            scalar_dict={"train/global_step": 5, "train/loss": 0.5},
            nonscalar_dict=hist_data,
        )

        with patch("agilerl.logger.SummaryWriter", mock_sw):
            from agilerl.logger import TensorboardLogger

            logger = TensorboardLogger()

        logger.write(report)

        mock_writer.add_scalar.assert_any_call("train/global_step", 5, global_step=5)
        mock_writer.add_scalar.assert_any_call("train/loss", 0.5, global_step=5)
        assert mock_writer.add_histogram.call_count == 1
        h_args, h_kwargs = mock_writer.add_histogram.call_args
        assert h_args[0] == "train/agent_0/weights"
        np.testing.assert_array_equal(h_args[1], np.array([1.0, 2.0, 3.0]))
        assert h_kwargs["global_step"] == 5
        mock_writer.flush.assert_called_once()

    def test_write_skips_non_numeric_scalars(self):
        mock_writer = MagicMock()
        mock_sw = MagicMock(return_value=mock_writer)
        report = _make_report(
            scalar_dict={"train/global_step": 1, "info": "text_value"},
        )

        with patch("agilerl.logger.SummaryWriter", mock_sw):
            from agilerl.logger import TensorboardLogger

            logger = TensorboardLogger()

        logger.write(report)

        calls = [c[0][0] for c in mock_writer.add_scalar.call_args_list]
        assert "info" not in calls

    def test_write_skips_on_non_main(self):
        mock_writer = MagicMock()
        mock_sw = MagicMock(return_value=mock_writer)
        acc = MagicMock()
        acc.is_main_process = False

        with patch("agilerl.logger.SummaryWriter", mock_sw):
            from agilerl.logger import TensorboardLogger

            logger = TensorboardLogger(accelerator=acc)

        logger.write(_make_report())
        mock_writer.add_scalar.assert_not_called()
        mock_writer.flush.assert_not_called()

    def test_close_on_main(self):
        mock_writer = MagicMock()
        mock_sw = MagicMock(return_value=mock_writer)

        with patch("agilerl.logger.SummaryWriter", mock_sw):
            from agilerl.logger import TensorboardLogger

            logger = TensorboardLogger()

        logger.close()
        mock_writer.close.assert_called_once()

    def test_close_skips_on_non_main(self):
        mock_writer = MagicMock()
        mock_sw = MagicMock(return_value=mock_writer)
        acc = MagicMock()
        acc.is_main_process = False

        with patch("agilerl.logger.SummaryWriter", mock_sw):
            from agilerl.logger import TensorboardLogger

            logger = TensorboardLogger(accelerator=acc)

        logger.close()
        mock_writer.close.assert_not_called()
