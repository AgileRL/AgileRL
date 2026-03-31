"""Logger hierarchy for training output.

Each logger consumes a :class:`~agilerl.population.MetricsReport` and writes
it to a specific backend (console, wandb, CSV file, TensorBoard).  A training
run typically uses several loggers in parallel, e.g. ``[StdOutLogger, WandbLogger]``.
"""

from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

import wandb

if TYPE_CHECKING:
    from accelerate import Accelerator
    from tqdm import tqdm

    from agilerl.population import MetricsReport


class Logger(ABC):
    """Base class for all training loggers.

    Subclasses must implement :meth:`write` and :meth:`close`.
    """

    @staticmethod
    @contextmanager
    def on_main_process(
        accelerator: Accelerator | None,
    ) -> Generator[bool, None, None]:
        """Synchronize distributed processes, yielding whether this is the main one.

        :param accelerator: HuggingFace Accelerator, or ``None``.
        :type accelerator: Accelerator | None
        :yields: ``True`` if the current process is the main process (or if
            *accelerator* is ``None``), ``False`` otherwise.
        """
        if accelerator is not None:
            accelerator.wait_for_everyone()
        try:
            yield accelerator is None or accelerator.is_main_process
        finally:
            if accelerator is not None:
                accelerator.wait_for_everyone()

    @abstractmethod
    def write(self, report: MetricsReport) -> None:
        """Persist one snapshot of population metrics.

        :param report: The metrics report to log.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the logger."""


class StdOutLogger(Logger):
    """Writes the tabular :class:`MetricsReport` to the console via a tqdm progress bar
    if provided, else just writes the report to the console.

    :param pbar: ``tqdm`` progress bar instance used for ``pbar.write()``.
    :type pbar: tqdm | None
    """

    def __init__(self, pbar: tqdm | None = None) -> None:
        self._pbar = pbar

    def write(self, report: MetricsReport) -> None:
        """Write the metrics report to the console.

        :param report: The metrics report to write.
        :type report: MetricsReport
        """
        if self._pbar is not None:
            self._pbar.write(str(report))
        else:
            print(report)

    def close(self) -> None:
        pass


class WandbLogger(Logger):
    """Logs a flat metrics dict to Weights & Biases. For this logger
    to work, users should call wandb.init() before training. AgileRL
    provides a helper function to do this: :func:`agilerl.utils.utils.init_wandb()`.

    Handles distributed-training synchronisation when an
    :class:`~accelerate.Accelerator` is provided.

    :param accelerator: HuggingFace Accelerator, or ``None``.
    """

    def __init__(self, accelerator: Accelerator | None = None) -> None:
        self._accelerator = accelerator

    def write(self, report: MetricsReport) -> None:
        """Write the metrics report to W&B.

        :param report: The metrics report to write.
        :type report: MetricsReport
        """
        with Logger.on_main_process(self._accelerator) as is_main:
            if is_main:
                wandb.log(report.to_dict())

    def close(self) -> None:
        """Mark a run as finished on W&B, and finish uploading all data."""
        with Logger.on_main_process(self._accelerator) as is_main:
            if is_main:
                wandb.finish()


class CSVLogger(Logger):
    """Appends one row per :meth:`write` call to a CSV file.

    :param path: Filesystem path for the CSV file.
    :type path: str | Path
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None

    def write(self, report: MetricsReport) -> None:
        """Write the metrics report to the CSV file.

        :param report: The metrics report to write.
        :type report: MetricsReport
        """
        data = report.to_dict()

        # Write header if file is not opened
        if self._writer is None:
            self._file = self._path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(data.keys()))
            self._writer.writeheader()

        # Write data to file
        self._writer.writerow(data)
        self._file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class TensorboardLogger(Logger):
    """Logs scalar metrics to TensorBoard via :class:`torch.utils.tensorboard.SummaryWriter`.

    Each key in :meth:`MetricsReport.to_dict` is written as a scalar at the
    ``train/global_step`` value.

    :param log_dir: Directory for TensorBoard event files, defaults to "tensorboard_logs"
    :type log_dir: str | Path, optional
    :param accelerator: HuggingFace Accelerator, or ``None``.
    :type accelerator: Accelerator | None
    :param experiment_name: Name of the experiment, defaults to None.
    :type experiment_name: str | None
    """

    def __init__(
        self,
        log_dir: str | Path = "tensorboard_logs",
        experiment_name: str | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        if SummaryWriter is None:
            msg = "TensorBoard is not installed. Please install it with `pip install tensorboard`."
            raise ImportError(msg)

        date = datetime.now().strftime("%m%d%Y%H%M%S")
        experiment_name = (
            date if experiment_name is None else f"{experiment_name}-{date}"
        )

        self._log_path = Path(log_dir) / experiment_name
        self._writer = SummaryWriter(log_dir=str(self._log_path))
        self._accelerator = accelerator

    def write(self, report: MetricsReport) -> None:
        """Write the metrics report to TensorBoard.

        :param report: The metrics report to write.
        :type report: MetricsReport
        """
        data = report.to_dict()
        global_step = int(data.get("train/global_step", 0))

        with Logger.on_main_process(self._accelerator) as is_main:
            if is_main:
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        self._writer.add_scalar(key, value, global_step=global_step)

                for key, value in report.to_nonscalar_dict().items():
                    self._writer.add_histogram(key, value, global_step=global_step)

                self._writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        with Logger.on_main_process(self._accelerator) as is_main:
            if is_main:
                self._writer.close()
