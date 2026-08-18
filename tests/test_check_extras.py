# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Arena extra must be a compatible range covering the latest agilerl-arena/v* tag."""

from __future__ import annotations

import importlib.util
import subprocess
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check-extras.py"


def _load_check_extras() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_extras", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def check_extras() -> ModuleType:
    return _load_check_extras()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_git(repo: Path) -> None:
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _git(repo, "commit", "--allow-empty", "-m", "init")


def _write_pyproject(
    repo: Path,
    *,
    arena_reqs: list[str] | None = None,
    extras: dict[str, list[str]] | None = None,
) -> Path:
    if extras is None:
        arena = arena_reqs if arena_reqs is not None else ['"agilerl-arena>=0.2.0,<0.3"']
        extra_toml = textwrap.dedent(
            f"""
            [project.optional-dependencies]
            box2d = ["box2d-py"]
            arena = [{", ".join(arena)}]
            all = ["agilerl[box2d,arena]"]
            """
        )
    else:
        lines = ["[project.optional-dependencies]"]
        for name, reqs in extras.items():
            rendered = ", ".join(f'"{req}"' for req in reqs)
            lines.append(f"{name} = [{rendered}]")
        extra_toml = "\n".join(lines) + "\n"
    path = repo / "pyproject.toml"
    path.write_text(
        textwrap.dedent(
            """
            [project]
            name = "agilerl"
            """
        )
        + extra_toml,
        encoding="utf-8",
    )
    return path


def _bind_repo(check_extras: ModuleType, repo: Path, pyproject: Path) -> None:
    check_extras.REPO_ROOT = repo
    check_extras.PARENT_PYPROJECT = pyproject


def _assert_exit_err(
    capsys: pytest.CaptureFixture[str], needle: str, fn, *args, **kwargs
) -> None:
    with pytest.raises(SystemExit) as exc:
        fn(*args, **kwargs)
    assert exc.value.code == 1
    assert needle in capsys.readouterr().err


def test_arena_range_covers_latest_tag(check_extras: ModuleType, tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    _init_git(tmp_path)
    _git(tmp_path, "tag", "agilerl-arena/v0.1.0")
    _git(tmp_path, "tag", "agilerl-arena/v0.2.0")
    _bind_repo(check_extras, tmp_path, pyproject)

    extras = check_extras._extras(pyproject)
    assert check_extras._check_arena_extra(extras) == ">=0.2.0,<0.3"


def test_arena_range_outside_tag_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(tmp_path)
    _init_git(tmp_path)
    _git(tmp_path, "tag", "agilerl-arena/v0.3.0")
    _bind_repo(check_extras, tmp_path, pyproject)

    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "outside extra range", check_extras._check_arena_extra, extras)


def test_missing_arena_tag_fails_closed(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(tmp_path)
    _init_git(tmp_path)
    _bind_repo(check_extras, tmp_path, pyproject)

    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "agilerl-arena/v", check_extras._check_arena_extra, extras)


def test_git_missing_fails_closed(
    check_extras: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = _write_pyproject(tmp_path)
    _bind_repo(check_extras, tmp_path, pyproject)

    def _boom(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "git")

    monkeypatch.setattr(check_extras.subprocess, "run", _boom)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "git not found", check_extras._check_arena_extra, extras)


def test_git_tag_failure_fails_closed(
    check_extras: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = _write_pyproject(tmp_path)
    _bind_repo(check_extras, tmp_path, pyproject)

    def _boom(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else ["git"],
            returncode=128,
            stdout="",
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(check_extras.subprocess, "run", _boom)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "git tag", check_extras._check_arena_extra, extras)


def test_git_tag_failure_without_stderr_fails_closed(
    check_extras: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = _write_pyproject(tmp_path)
    _bind_repo(check_extras, tmp_path, pyproject)

    def _boom(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else ["git"],
            returncode=1,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(check_extras.subprocess, "run", _boom)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "exit 1", check_extras._check_arena_extra, extras)


def test_missing_arena_extra_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(
        tmp_path, extras={"box2d": ["box2d-py"], "all": ["agilerl[box2d]"]}
    )
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "Missing", check_extras._check_arena_extra, extras)


def test_committed_exact_pin_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(tmp_path, arena_reqs=['"agilerl-arena==0.2.0"'])
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "compatible range", check_extras._check_arena_extra, extras)


def test_missing_range_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(tmp_path, arena_reqs=['"agilerl-arena>=0.2.0"'])
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "Expected one", check_extras._check_arena_extra, extras)


def test_multiple_ranges_fail(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        arena_reqs=['"agilerl-arena>=0.2.0,<0.3"', '"agilerl-arena>=0.3.0,<0.4"'],
    )
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "Expected one", check_extras._check_arena_extra, extras)


def test_arena_pin_override_matches(
    check_extras: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyproject = _write_pyproject(
        tmp_path, arena_reqs=['"agilerl-arena==0.2.0+build.9"']
    )
    _bind_repo(check_extras, tmp_path, pyproject)
    monkeypatch.setenv("AGILERL_ARENA_PIN", "0.2.0+build.9")
    extras = check_extras._extras(pyproject)
    assert check_extras._check_arena_extra(extras) == "==0.2.0+build.9"


def test_arena_pin_override_requires_exact(
    check_extras: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pyproject = _write_pyproject(tmp_path)
    _bind_repo(check_extras, tmp_path, pyproject)
    monkeypatch.setenv("AGILERL_ARENA_PIN", "0.2.0+build.9")
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "AGILERL_ARENA_PIN", check_extras._check_arena_extra, extras)


def test_all_extra_unions_parts(check_extras: ModuleType, tmp_path: Path) -> None:
    pyproject = _write_pyproject(tmp_path)
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    check_extras._check_all_extra(extras)


def test_all_extra_drift_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        extras={
            "box2d": ["box2d-py"],
            "arena": ["agilerl-arena>=0.2.0,<0.3"],
            "all": ["box2d-py"],
        },
    )
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "all", check_extras._check_all_extra, extras)


def test_missing_all_extra_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(
        tmp_path, extras={"arena": ["agilerl-arena>=0.2.0,<0.3"]}
    )
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(capsys, "Missing", check_extras._check_all_extra, extras)


def test_self_reference_cycle_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        extras={"arena": ["agilerl[arena]"], "all": ["agilerl[arena]"]},
    )
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(
        capsys, "cycle", check_extras._requirements, "arena", extras, set()
    )


def test_missing_referenced_extra_fails(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(
        tmp_path,
        extras={"all": ["agilerl[missing]"]},
    )
    _bind_repo(check_extras, tmp_path, pyproject)
    extras = check_extras._extras(pyproject)
    _assert_exit_err(
        capsys, "does not exist", check_extras._requirements, "all", extras, set()
    )


def test_main_ok(
    check_extras: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pyproject = _write_pyproject(tmp_path)
    _init_git(tmp_path)
    _git(tmp_path, "tag", "agilerl-arena/v0.2.0")
    _bind_repo(check_extras, tmp_path, pyproject)
    check_extras.main()
    assert "OK:" in capsys.readouterr().out
