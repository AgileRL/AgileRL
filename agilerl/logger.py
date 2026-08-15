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
from typing import TYPE_CHECKING, ClassVar

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


def _is_notebook() -> bool:
    """Detect whether we are running inside a Jupyter/IPython notebook."""
    try:
        from IPython import get_ipython

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

    def __init__(self, pbar: tqdm | None = None) -> None:
        self._pbar = pbar
        self._notebook = _is_notebook()

    def write(self, report: MetricsReport) -> None:
        """Write the metrics report to the console.

        :param report: The metrics report to write.
        :type report: MetricsReport
        """
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


class MutationHistoryLogger(Logger):
    """Writes a per-generation, per-agent record of evolutionary mutations to CSV.

    One row is emitted per agent per generation, describing the mutation that
    produced the agent (applied at the previous tournament/mutation boundary and
    carried through training), together with its fitness/score *after* that
    mutation (this generation's evaluation) and *before* it (the parent's value
    from the previous generation, linked via the parent's index). Rows are written
    incrementally so the file is complete even if the run is interrupted.

    The "before" values for the final boundary mutation (agents created after the
    last :meth:`write` and never re-evaluated) are never measured, so that final
    mutation is intentionally absent from the file.

    :param out_dir: Directory to write ``mutation_history.csv`` into.
    :type out_dir: str | Path
    """

    FIELDNAMES: ClassVar[list[str]] = [
        "generation",
        "global_step",
        "agent_slot",
        "agent_id",
        "parent_id",
        "mutation_category",
        "mutation_name",
        "hp_before",
        "hp_after",
        "arch_layer_changed",
        "arch_neurons_delta",
        "arch_new_layer_position",
        "arch_new_layer_size",
        "arch_func_preserving",
        "arch_dormant_count",
        "arch_neurons_removed",
        "param_weights_reset",
        "param_weights_ordinary_noise",
        "param_weights_amplified_noise",
        "fitness_before",
        "score_before",
        "fitness_after",
        "score_after",
    ]

    def __init__(self, out_dir: str | Path) -> None:
        self._path = Path(out_dir) / "mutation_history.csv"
        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter | None = None
        self._generation = 0
        # Maps agent_id -> (fitness, score) from the previous generation.
        self._prev: dict[int, tuple[float, float]] = {}

    @staticmethod
    def _scalarize(value: object) -> float:
        """Reduce a possibly dict/sequence-valued metric to a scalar mean."""
        if value is None:
            return float("nan")
        if isinstance(value, dict):
            vals = list(value.values())
            return float(sum(vals) / len(vals)) if vals else float("nan")
        if isinstance(value, (list, tuple)):
            return float(sum(value) / len(value)) if value else float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def write(self, report: MetricsReport) -> None:
        """Append this generation's per-agent mutation rows to the CSV file.

        :param report: The metrics report to log.
        :type report: MetricsReport
        """
        metrics = report.metrics
        indices = metrics.indices
        pop_size = len(indices)

        mutations = metrics.mutations or [None] * pop_size
        mut_details = metrics.mut_details or [None] * pop_size
        parent_indices = metrics.parent_indices or list(indices)
        fitnesses = metrics.fitnesses or [float("nan")] * pop_size
        scores = metrics.scores or [float("nan")] * pop_size

        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
            self._writer.writeheader()

        current: dict[int, tuple[float, float]] = {}
        for slot in range(pop_size):
            agent_id = indices[slot]
            parent_id = parent_indices[slot]
            fitness_after = self._scalarize(fitnesses[slot])
            score_after = self._scalarize(scores[slot])
            current[agent_id] = (fitness_after, score_after)

            fitness_before, score_before = self._prev.get(
                parent_id, (float("nan"), float("nan"))
            )

            row = self._build_row(
                agent_slot=slot,
                agent_id=agent_id,
                parent_id=parent_id,
                global_step=metrics.global_step,
                mut=mutations[slot],
                details=mut_details[slot],
                fitness_before=fitness_before,
                score_before=score_before,
                fitness_after=fitness_after,
                score_after=score_after,
            )
            self._writer.writerow(row)

        self._file.flush()
        self._prev = current
        self._generation += 1

    def _build_row(
        self,
        *,
        agent_slot: int,
        agent_id: int,
        parent_id: int,
        global_step: int,
        mut: object,
        details: dict | None,
        fitness_before: float,
        score_before: float,
        fitness_after: float,
        score_after: float,
    ) -> dict:
        """Assemble a single CSV row from a snapshot entry."""
        details = details or {}
        category = details.get("category")
        name = details.get("name")
        if category is None:
            # No detail recorded (e.g. no-HPO runs): infer from the ``mut`` flag.
            category = "no mutation" if mut in (None, "None") else "other"
            name = "none" if mut in (None, "None") else str(mut)

        return {
            "generation": self._generation,
            "global_step": global_step,
            "agent_slot": agent_slot,
            "agent_id": agent_id,
            "parent_id": parent_id,
            "mutation_category": category,
            "mutation_name": name,
            "hp_before": details.get("hp_before", ""),
            "hp_after": details.get("hp_after", ""),
            "arch_layer_changed": details.get("layer_changed", ""),
            "arch_neurons_delta": details.get("neurons_delta", ""),
            "arch_new_layer_position": details.get("new_layer_position", ""),
            "arch_new_layer_size": details.get("new_layer_size", ""),
            "arch_func_preserving": details.get("arch_func_preserving", ""),
            "arch_dormant_count": details.get("arch_dormant_count", ""),
            "arch_neurons_removed": details.get("arch_neurons_removed", ""),
            "param_weights_reset": details.get("weights_reset", ""),
            "param_weights_ordinary_noise": details.get("weights_ordinary_noise", ""),
            "param_weights_amplified_noise": details.get("weights_amplified_noise", ""),
            "fitness_before": fitness_before,
            "score_before": score_before,
            "fitness_after": fitness_after,
            "score_after": score_after,
        }

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
