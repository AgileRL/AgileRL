# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint uploads from one Snowflake job instance."""

from __future__ import annotations

from pathlib import Path

import pytest

from agilerl.arena.snowflake.upload import (
    CheckpointUploader,
    ready_checkpoint_files,
    upload_with_retries,
)

RUN_ID = "arena-20261109163457-abc123"
RETRY_ON = (OSError,)


class StageSession:
    """Session whose PUTs copy files into a local directory that stands in for the stage."""

    def __init__(self, stage_root: Path, failures: int = 0) -> None:
        self.stage_root = stage_root
        self.failures = failures
        self.order: list[str] = []
        self.closed = 0
        self.file = self

    def put(
        self,
        local: str,
        stage: str,
        auto_compress: bool = True,
        overwrite: bool = False,
    ) -> None:
        if self.failures > 0:
            self.failures -= 1
            msg = "stage unavailable"
            raise OSError(msg)
        assert auto_compress is False
        assert overwrite is True
        target = self.stage_root / stage.removeprefix("@")
        target.mkdir(parents=True, exist_ok=True)
        source = Path(local)
        (target / source.name).write_bytes(source.read_bytes())
        self.order.append(f"{stage}/{source.name}")

    def close(self) -> None:
        self.closed += 1


def _checkpoint(root: Path, number: int, merged_complete: bool) -> Path:
    directory = root / f"checkpoint_{number}"
    (directory / "merged").mkdir(parents=True)
    (directory / "state.pt").write_text("state", encoding="utf-8")
    (directory / "merged" / "model.safetensors").write_text("w", encoding="utf-8")
    (directory / "merged" / "config.json").write_text("{}", encoding="utf-8")
    if merged_complete:
        (directory / "merged" / ".complete").write_text("", encoding="utf-8")
    return directory


class TestReadyCheckpointFiles:
    def test_complete_marker_follows_the_weights(self, tmp_path: Path) -> None:
        _checkpoint(tmp_path, 1, merged_complete=True)

        ready = ready_checkpoint_files(tmp_path, trainer_exited=True)

        assert [path.relative_to(tmp_path).as_posix() for path in ready] == [
            "checkpoint_1/merged/config.json",
            "checkpoint_1/merged/model.safetensors",
            "checkpoint_1/state.pt",
            "checkpoint_1/merged/.complete",
        ]

    def test_latest_checkpoint_waits_for_a_newer_one(self, tmp_path: Path) -> None:
        _checkpoint(tmp_path, 1, merged_complete=True)
        _checkpoint(tmp_path, 2, merged_complete=False)

        ready = ready_checkpoint_files(tmp_path, trainer_exited=False)

        assert [path.relative_to(tmp_path).as_posix() for path in ready] == [
            "checkpoint_1/merged/config.json",
            "checkpoint_1/merged/model.safetensors",
            "checkpoint_1/state.pt",
            "checkpoint_1/merged/.complete",
        ]

    def test_merged_waits_for_complete_after_exit(self, tmp_path: Path) -> None:
        _checkpoint(tmp_path, 3, merged_complete=False)
        (tmp_path / "other").mkdir()

        ready = ready_checkpoint_files(tmp_path, trainer_exited=True)

        assert [path.relative_to(tmp_path).as_posix() for path in ready] == [
            "checkpoint_3/state.pt",
        ]

    def test_missing_root_has_nothing_ready(self, tmp_path: Path) -> None:
        assert ready_checkpoint_files(tmp_path / "absent", trainer_exited=True) == []


class TestUploadWithRetries:
    def test_succeeds_after_transient_failures(self) -> None:
        calls: list[int] = []
        delays: list[float] = []

        def action() -> None:
            calls.append(1)
            if len(calls) < 3:
                msg = "transient"
                raise OSError(msg)

        result = upload_with_retries(action, True, delays.append, RETRY_ON, 3, 2.0)

        assert result is True
        assert len(calls) == 3
        assert delays == [2.0, 4.0]

    def test_non_final_failure_returns_false(self) -> None:
        delays: list[float] = []

        def action() -> None:
            msg = "down"
            raise OSError(msg)

        result = upload_with_retries(action, False, delays.append, RETRY_ON, 2, 1.0)

        assert result is False
        assert delays == [1.0]

    def test_final_failure_raises(self) -> None:
        def action() -> None:
            msg = "down"
            raise OSError(msg)

        with pytest.raises(OSError, match="down"):
            upload_with_retries(action, True, lambda seconds: None, RETRY_ON, 2, 1.0)

    def test_other_errors_raise_without_a_retry(self) -> None:
        calls: list[int] = []
        delays: list[float] = []

        def action() -> None:
            calls.append(1)
            msg = "bad stage name"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="bad stage name"):
            upload_with_retries(action, False, delays.append, RETRY_ON, 3, 1.0)

        assert len(calls) == 1
        assert delays == []


class TestCheckpointUploaderUpload:
    def test_worker_checkpoint_reaches_the_stage(self, tmp_path: Path) -> None:
        # Arrange
        ckpt = tmp_path / "worker_ckpt"
        _checkpoint(ckpt, 4, merged_complete=True)
        stage = tmp_path / "stage"
        sessions: list[StageSession] = []

        def open_session() -> StageSession:
            sessions.append(StageSession(stage))
            return sessions[-1]

        uploader = CheckpointUploader(
            open_session=open_session,
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=ckpt,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

        # Act
        result = uploader.upload(final=True)

        # Assert
        assert result is True
        run_dir = stage / "arena.runs" / RUN_ID / "checkpoint_4"
        assert (run_dir / "state.pt").read_text(encoding="utf-8") == "state"
        assert (run_dir / "merged" / "model.safetensors").is_file()
        assert (run_dir / "merged" / ".complete").is_file()
        assert sessions[0].order[-2].endswith("merged/.complete")
        assert sessions[0].order[-1] == f"@arena.runs/{RUN_ID}/checkpoint_4/.uploaded"
        assert sessions[0].closed == 1

    def test_nothing_ready_opens_no_session(self, tmp_path: Path) -> None:
        def open_session() -> StageSession:
            msg = "no session expected"
            raise AssertionError(msg)

        uploader = CheckpointUploader(
            open_session=open_session,
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=tmp_path,
            retry_on=RETRY_ON,
        )

        assert uploader.upload(final=False) is True

    def test_retry_skips_files_already_uploaded(self, tmp_path: Path) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=True)
        stage = tmp_path / "stage"
        puts: list[str] = []

        class FailsSecondPut(StageSession):
            def put(
                self,
                local: str,
                stage: str,
                auto_compress: bool = True,
                overwrite: bool = False,
            ) -> None:
                puts.append(Path(local).name)
                if len(puts) == 2:
                    msg = "stage unavailable"
                    raise OSError(msg)
                super().put(local, stage, auto_compress, overwrite)

        uploader = CheckpointUploader(
            open_session=lambda: FailsSecondPut(stage),
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=ckpt,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

        # Act
        result = uploader.upload(final=True)

        # Assert
        assert result is True
        assert puts == [
            "config.json",
            "model.safetensors",
            "model.safetensors",
            "state.pt",
            ".complete",
            ".uploaded",
        ]

    def test_non_final_failure_keeps_complete_off_the_stage(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=True)
        stage = tmp_path / "stage"

        class FailingWeights(StageSession):
            def put(
                self,
                local: str,
                stage: str,
                auto_compress: bool = True,
                overwrite: bool = False,
            ) -> None:
                if Path(local).name == "model.safetensors":
                    msg = "stage unavailable"
                    raise OSError(msg)
                super().put(local, stage, auto_compress, overwrite)

        sessions: list[StageSession] = []

        def open_session() -> StageSession:
            sessions.append(FailingWeights(stage))
            return sessions[-1]

        uploader = CheckpointUploader(
            open_session=open_session,
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=ckpt,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

        # Act
        result = uploader.upload(final=False)

        # Assert
        assert result is False
        merged = stage / "arena.runs" / RUN_ID / "checkpoint_1" / "merged"
        assert (merged / "config.json").is_file()
        assert not (merged / ".complete").exists()
        assert not (merged.parent / ".uploaded").exists()
        assert all(session.closed == 1 for session in sessions)
        assert len(sessions) == 3

    def test_final_failure_raises(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=True)
        uploader = CheckpointUploader(
            open_session=lambda: StageSession(tmp_path / "stage", failures=99),
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=ckpt,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

        with pytest.raises(OSError, match="stage unavailable"):
            uploader.upload(final=True)


class TestCheckpointUploaderUploadedMarker:
    def _uploader(
        self, ckpt: Path, sessions: list[StageSession], stage: Path
    ) -> CheckpointUploader:
        def open_session() -> StageSession:
            sessions.append(StageSession(stage))
            return sessions[-1]

        return CheckpointUploader(
            open_session=open_session,
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=ckpt,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

    def test_marker_follows_the_last_file_of_a_finished_checkpoint(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=True)
        _checkpoint(ckpt, 2, merged_complete=False)
        stage = tmp_path / "stage"
        sessions: list[StageSession] = []

        # Act
        self._uploader(ckpt, sessions, stage).upload(final=False)

        # Assert
        run_dir = f"@arena.runs/{RUN_ID}"
        assert sessions[0].order == [
            f"{run_dir}/checkpoint_1/merged/config.json",
            f"{run_dir}/checkpoint_1/merged/model.safetensors",
            f"{run_dir}/checkpoint_1/state.pt",
            f"{run_dir}/checkpoint_1/merged/.complete",
            f"{run_dir}/checkpoint_1/.uploaded",
        ]
        marker = stage / "arena.runs" / RUN_ID / "checkpoint_1" / ".uploaded"
        assert marker.read_bytes() == b""
        assert not (stage / "arena.runs" / RUN_ID / "checkpoint_2").exists()

    def test_checkpoint_waiting_on_merged_weights_gets_no_marker(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=False)
        stage = tmp_path / "stage"

        # Act
        self._uploader(ckpt, [], stage).upload(final=True)

        # Assert
        run_dir = stage / "arena.runs" / RUN_ID / "checkpoint_1"
        assert (run_dir / "state.pt").is_file()
        assert not (run_dir / ".uploaded").exists()

    def test_marker_lands_once_late_merged_weights_finish(self, tmp_path: Path) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=False)
        _checkpoint(ckpt, 2, merged_complete=False)
        sessions: list[StageSession] = []
        uploader = self._uploader(ckpt, sessions, tmp_path / "stage")
        uploader.upload(final=False)
        (ckpt / "checkpoint_1" / "merged" / ".complete").write_text(
            "", encoding="utf-8"
        )

        # Act
        uploader.upload(final=False)
        uploader.upload(final=False)

        # Assert
        assert len(sessions) == 2
        assert sessions[1].order[-1] == f"@arena.runs/{RUN_ID}/checkpoint_1/.uploaded"
        assert not (ckpt / "checkpoint_1").exists()

    def test_failed_file_put_leaves_no_marker(self, tmp_path: Path) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=True)
        stage = tmp_path / "stage"
        uploader = CheckpointUploader(
            open_session=lambda: StageSession(stage, failures=99),
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=ckpt,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

        # Act
        result = uploader.upload(final=False)

        # Assert
        assert result is False
        assert not (stage / "arena.runs" / RUN_ID / "checkpoint_1").exists()
        assert (ckpt / "checkpoint_1").is_dir()


class TestCheckpointUploaderLocalCleanup:
    def _uploader(self, ckpt: Path, stage: Path) -> CheckpointUploader:
        return CheckpointUploader(
            open_session=lambda: StageSession(stage),
            runs_stage="arena.runs",
            run_id=RUN_ID,
            ckpt_root=ckpt,
            retry_on=RETRY_ON,
            sleep=lambda seconds: None,
        )

    def test_uploaded_older_checkpoint_is_deleted(self, tmp_path: Path) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=True)
        _checkpoint(ckpt, 2, merged_complete=False)
        stage = tmp_path / "stage"

        # Act
        result = self._uploader(ckpt, stage).upload(final=False)

        # Assert
        assert result is True
        assert not (ckpt / "checkpoint_1").exists()
        assert (ckpt / "checkpoint_2" / "state.pt").is_file()
        uploaded = stage / "arena.runs" / RUN_ID / "checkpoint_1"
        assert (uploaded / "merged" / ".complete").is_file()

    def test_partly_uploaded_older_checkpoint_is_kept(self, tmp_path: Path) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 1, merged_complete=False)
        _checkpoint(ckpt, 2, merged_complete=False)

        # Act
        self._uploader(ckpt, tmp_path / "stage").upload(final=False)

        # Assert
        assert (ckpt / "checkpoint_1" / "merged" / "model.safetensors").is_file()
        assert (ckpt / "checkpoint_2").is_dir()

    def test_final_upload_keeps_the_latest_checkpoint(self, tmp_path: Path) -> None:
        # Arrange
        ckpt = tmp_path / "ckpt"
        _checkpoint(ckpt, 3, merged_complete=True)
        _checkpoint(ckpt, 10, merged_complete=True)

        # Act
        self._uploader(ckpt, tmp_path / "stage").upload(final=True)

        # Assert
        assert not (ckpt / "checkpoint_3").exists()
        assert (ckpt / "checkpoint_10" / "merged" / ".complete").is_file()
