# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Copy finished checkpoint files from one job instance to the runs stage."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from agilerl.arena.snowflake.job import put_file, stage_directory
from agilerl.arena.snowflake.source import SnowflakeSession

CKPT_DIR = Path("/tmp/arena/ckpt")
CHECKPOINT_DIR = re.compile(r"^checkpoint_(\d+)$")
UPLOAD_ATTEMPTS = 3
UPLOAD_RETRY_S = 5.0

CHECKPOINT_UPLOADED_MARKER = ".uploaded"

logger = logging.getLogger(__name__)


def checkpoint_directories(ckpt_root: Path) -> dict[int, Path]:
    """``checkpoint_<n>`` directories under the trainer checkpoint root.

    :param ckpt_root: Trainer checkpoint directory. A missing root has none.
    :return: Directories keyed by step number.
    """
    if not ckpt_root.is_dir():
        return {}
    return {
        int(match.group(1)): child
        for child in ckpt_root.iterdir()
        if (match := CHECKPOINT_DIR.fullmatch(child.name)) and child.is_dir()
    }


def ready_checkpoint_files(ckpt_root: Path, trainer_exited: bool) -> list[Path]:
    """Checkpoint files that are finished and safe to upload.

    A ``merged/`` directory is ready when it contains ``.complete``.
    ``checkpoint_<n>/`` is ready once a higher checkpoint exists, or the trainer
    has exited. Merged files wait for ``.complete`` even after that.

    :param ckpt_root: Trainer checkpoint directory.
    :param trainer_exited: Whether the trainer process has finished.
    :return: Files to upload, in path order with each ``.complete`` after its weights.
    """
    numbered = checkpoint_directories(ckpt_root)
    if not numbered:
        return []
    highest = max(numbered)
    ready: list[Path] = []
    for number, directory in sorted(numbered.items()):
        directory_done = trainer_exited or number < highest
        merged_done = (directory / "merged" / ".complete").is_file()
        files = sorted(
            (path for path in directory.rglob("*") if path.is_file()),
            key=lambda path: (path.name == ".complete", path),
        )
        for path in files:
            if "merged" in path.relative_to(directory).parts:
                if merged_done:
                    ready.append(path)
            elif directory_done:
                ready.append(path)
    return ready


def upload_with_retries(
    action: Callable[[], object],
    final: bool,
    sleep: Callable[[float], None],
    retry_on: tuple[type[Exception], ...],
    attempts: int = UPLOAD_ATTEMPTS,
    retry_s: float = UPLOAD_RETRY_S,
) -> bool:
    """Run one upload pass with bounded exponential backoff on transient errors.

    :param action: The pass. It must skip work an earlier attempt finished.
    :param final: Whether this is the last pass of the run.
    :param sleep: Called between attempts.
    :param retry_on: Errors worth another attempt. Any other error raises at once.
    :param attempts: Tries before giving up on this pass.
    :param retry_s: First backoff delay. Each retry doubles it.
    :return: ``True`` when the pass succeeded. A failed non-final pass returns ``False``.
    :raises Exception: The final pass failed on every attempt, or an error
        outside ``retry_on`` was raised.
    """
    attempt = 1
    while True:
        try:
            action()
        except retry_on as exc:
            if attempt < attempts:
                sleep(retry_s * 2 ** (attempt - 1))
                attempt += 1
                continue
            if final:
                raise
            logger.warning(
                "Upload failed %d times, trying again next pass: %s", attempts, exc
            )
            return False
        return True


@dataclass
class CheckpointUploader:
    """Uploads this instance's finished checkpoint files, opening a session per pass.

    Once every file of a finished ``checkpoint_<n>/`` is on the stage, an empty
    ``checkpoint_<n>/.uploaded`` follows it. ``submit --resume`` only picks
    checkpoints that have it. Older local directories are deleted after their marker.
    """

    open_session: Callable[[], SnowflakeSession]
    runs_stage: str
    run_id: str
    ckpt_root: Path
    retry_on: tuple[type[Exception], ...]
    sleep: Callable[[float], None] = time.sleep
    uploaded: set[str] = field(default_factory=set)

    def upload(self, final: bool) -> bool:
        """Upload every ready file not yet on the stage.

        :param final: The trainer has exited. Failure raises instead of waiting a pass.
        :return: Whether the pass succeeded.
        """
        return upload_with_retries(
            partial(self._upload_pending, final), final, self.sleep, self.retry_on
        )

    def _upload_pending(self, final: bool) -> None:
        pending = [
            path
            for path in ready_checkpoint_files(self.ckpt_root, final)
            if self._relative(path) not in self.uploaded
        ]
        whole = self._checkpoints_whole_after(pending, final)
        if not pending and not whole:
            return
        session = self.open_session()
        try:
            for path in pending:
                self._put(session, path, self._relative(path))
            for directory in whole:
                self._put_uploaded_marker(session, directory)
        finally:
            session.close()
        self._delete_uploaded_checkpoints()

    def _checkpoints_whole_after(self, pending: list[Path], final: bool) -> list[Path]:
        on_stage = self.uploaded | {self._relative(path) for path in pending}
        numbered = checkpoint_directories(self.ckpt_root)
        return [
            directory
            for number, directory in sorted(numbered.items())
            if (final or number < max(numbered))
            and self._marker_key(directory) not in self.uploaded
            and all(
                self._relative(path) in on_stage
                for path in directory.rglob("*")
                if path.is_file()
            )
        ]

    def _put_uploaded_marker(self, session: SnowflakeSession, directory: Path) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            marker = Path(scratch) / CHECKPOINT_UPLOADED_MARKER
            marker.touch()
            self._put(session, marker, self._marker_key(directory))

    def _put(self, session: SnowflakeSession, local: Path, relative: str) -> None:
        put_file(
            session, local, stage_directory(self.runs_stage, self.run_id, relative)
        )
        self.uploaded.add(relative)

    def _delete_uploaded_checkpoints(self) -> None:
        # The highest checkpoint is kept: the trainer may still be writing it.
        numbered = checkpoint_directories(self.ckpt_root)
        for number, directory in numbered.items():
            if number < max(numbered) and self._marker_key(directory) in self.uploaded:
                shutil.rmtree(directory)

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.ckpt_root).as_posix()

    def _marker_key(self, directory: Path) -> str:
        return f"{directory.name}/{CHECKPOINT_UPLOADED_MARKER}"
