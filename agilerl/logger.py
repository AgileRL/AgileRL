# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Logger hierarchy for training output.

Each logger consumes a :class:`~agilerl.population.MetricsReport` and writes
it to a specific backend (console, wandb, CSV file, TensorBoard).  A training
run typically uses several loggers at the same time, e.g. ``[StdOutLogger, WandbLogger]``.
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

import wandb

from agilerl.utils.distributed import barrier, is_distributed, is_main_process

if TYPE_CHECKING:
    from accelerate import Accelerator
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from agilerl.population import MetricsReport
else:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None


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
            if accelerator is not None:
                yield accelerator.is_main_process
            else:
                yield is_main_process()
        finally:
            if accelerator is not None:
                accelerator.wait_for_everyone()
            elif is_distributed():
                barrier()

    @abstractmethod
    def write(self, report: MetricsReport) -> None:
        """Persist one snapshot of population metrics.

        :param report: The metrics report to log.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the logger."""


def _is_notebook() -> bool:
    """Detect whether we are running inside a Jupyter/IPython notebook."""
    try:
        # IPython is not a dependency; only present in notebook environments.
        from IPython import get_ipython  # ty: ignore[unresolved-import]

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, AttributeError, NameError):
        return False


class StdOutLogger(Logger):
    """Writes the tabular :class:`MetricsReport` to the console via a tqdm progress bar
    if provided, else just writes the report to the console.

    :param pbar: ``tqdm`` progress bar instance used for ``pbar.write()``.
    :type pbar: tqdm | None
    """

    def __init__(
        self, pbar: tqdm | None = None, accelerator: Accelerator | None = None
    ) -> None:
        self._pbar = pbar
        self._notebook = _is_notebook()
        self._accelerator = accelerator

    def write(self, report: MetricsReport) -> None:
        """Write the metrics report to the console.

        Only the main process emits output; non-main ranks return without
        printing so distributed runs (where ``report_metrics`` is called on
        every rank to keep collectives symmetric) don't duplicate the table.

        :param report: The metrics report to write.
        :type report: MetricsReport
        """
        if self._accelerator is not None and not self._accelerator.is_main_process:
            return
        if self._accelerator is None and not is_main_process():
            return
        text = str(report)
        if self._pbar is not None and not self._notebook:
            self._pbar.write(text)
        else:
            print(text)

    def close(self) -> None:
        pass


class WandbLogger(Logger):
    """Logs a flat metrics dict to Weights & Biases.

    If ``wandb.init()`` has not been called before the first :meth:`write`,
    this logger will call it automatically with ``project="AgileRL"``.
    For more control over the run configuration, call ``wandb.init()``
    (or :func:`agilerl.utils.utils.init_wandb`) before creating this logger.

    Handles distributed-training synchronisation when an
    :class:`~accelerate.Accelerator` is provided.

    :param accelerator: HuggingFace Accelerator, or ``None``.
    :type accelerator: Accelerator | None
    """

    def __init__(
        self,
        accelerator: Accelerator | None = None,
        project: str = "AgileRL",
    ) -> None:
        self._accelerator = accelerator
        self._project = project

    def _maybe_init_wandb(self) -> None:
        """Initialize a W&B run if one is not already active."""
        if wandb.run is None:
            wandb.init(project=self._project)

    def write(self, report: MetricsReport) -> None:
        """Write the metrics report to W&B.

        :param report: The metrics report to write.
        :type report: MetricsReport
        """
        with Logger.on_main_process(self._accelerator) as is_main:
            if is_main:
                self._maybe_init_wandb()
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
        file = self._file
        if self._writer is None or file is None:
            file = self._path.open("w", newline="")
            self._file = file
            self._writer = csv.DictWriter(file, fieldnames=list(data.keys()))
            self._writer.writeheader()

        # Write data to file
        self._writer.writerow(data)
        file.flush()

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

    :param log_dir: Directory for TensorBoard event files, defaults to ``None``,
        which will use the default TensorBoard log directory ``tensorboard_logs``.
    :type log_dir: str | Path | None
    :param experiment_name: Name of the experiment, defaults to ``None``.
    :type experiment_name: str | None
    :param accelerator: HuggingFace Accelerator, or ``None``.
    :type accelerator: Accelerator | None
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        experiment_name: str | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        if SummaryWriter is None:
            msg = "TensorBoard is not installed. Please install it with `pip install tensorboard`."
            raise ImportError(msg)

        log_dir = log_dir or "tensorboard_logs"
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
