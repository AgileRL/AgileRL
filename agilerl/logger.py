"""Logger hierarchy for training output.

Each logger consumes a :class:`~agilerl.population.MetricsReport` and writes
it to a specific backend (console, wandb, CSV file, TensorBoard).  A training
run typically uses several loggers in parallel, e.g. ``[StdOutLogger, WandbLogger]``.
"""

from __future__ import annotations

import csv
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import wandb

if TYPE_CHECKING:
    from accelerate import Accelerator
    from tqdm import tqdm

    from agilerl.population import MetricsReport


class Logger(ABC):
    """Base class for all training loggers.

    Subclasses must implement :meth:`write` and :meth:`close`.
    """

    @abstractmethod
    def write(self, report: MetricsReport) -> None:
        """Persist one snapshot of population metrics.

        :param report: The metrics report to log.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the logger."""


class StdOutLogger(Logger):
    """Writes the tabular :class:`MetricsReport` to the console via a tqdm pbar.

    :param pbar: ``tqdm`` progress bar instance used for ``pbar.write()``.
    """

    def __init__(self, pbar: tqdm) -> None:
        self._pbar = pbar

    def write(self, report: MetricsReport) -> None:
        self._pbar.write(str(report))

    def close(self) -> None:
        pass


class WandbLogger(Logger):
    """Logs a flat metrics dict to Weights & Biases.

    Handles distributed-training synchronisation when an
    :class:`~accelerate.Accelerator` is provided.

    :param accelerator: HuggingFace Accelerator, or ``None``.
    """

    def __init__(self, accelerator: Accelerator | None = None) -> None:
        self._accelerator = accelerator

    def write(self, report: MetricsReport) -> None:
        data = report.to_dict()
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
            if self._accelerator.is_main_process:
                wandb.log(data)
            self._accelerator.wait_for_everyone()
        else:
            wandb.log(data)

    def close(self) -> None:
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
            if self._accelerator.is_main_process:
                wandb.finish()
            self._accelerator.wait_for_everyone()
        else:
            wandb.finish()


class CSVLogger(Logger):
    """Appends one row per :meth:`write` call to a CSV file.

    The header row is written lazily on the first :meth:`write` call so that
    column names adapt to whatever metrics are registered.

    :param path: Filesystem path for the CSV file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None

    def write(self, report: MetricsReport) -> None:
        data = report.to_dict()
        if self._writer is None:
            self._file = self._path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(data.keys()))
            self._writer.writeheader()
        self._writer.writerow(data)
        self._file.flush()

    def close(self) -> None:
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
    """

    def __init__(
        self,
        log_dir: str | Path = "tensorboard_logs",
        accelerator: Accelerator | None = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._accelerator = accelerator

    def write(self, report: MetricsReport) -> None:
        data = report.to_dict()
        global_step = int(data.get("train/global_step", 0))

        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
            if not self._accelerator.is_main_process:
                self._accelerator.wait_for_everyone()
                return
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, global_step=global_step)
        self._writer.flush()

        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()

    def close(self) -> None:
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
            if self._accelerator.is_main_process:
                self._writer.close()
            self._accelerator.wait_for_everyone()
        else:
            self._writer.close()
