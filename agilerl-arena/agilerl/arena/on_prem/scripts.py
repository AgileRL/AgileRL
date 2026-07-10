"""Running bundle shell scripts locally, with quiet-on-success capture."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

import click

logger = logging.getLogger("agilerl.arena.on_prem")


class StageFailed(Exception):
    """A bundle script exited non-zero; carries its captured output for reporting."""

    def __init__(self, script: str, returncode: int, output: str) -> None:
        """Capture a failed script invocation.

        :param script: The name of the script that failed.
        :type script: str
        :param returncode: The script's non-zero exit code.
        :type returncode: int
        :param output: The captured stdout from the script, for reporting.
        :type output: str
        """
        self.script = script
        self.returncode = returncode
        self.output = output
        super().__init__(f"{script} exited {returncode}")


def stage_failure(
    label: str,
    host: str,
    exc: StageFailed,
    *,
    index: int | None = None,
    total: int | None = None,
) -> click.ClickException:
    """Turn a :class:`StageFailed` into a clean, stage-named ClickException.

    :param label: The human-readable stage label (e.g. ``"Installing Docker Engine"``).
    :type label: str
    :param host: The host the stage ran on, for the error message.
    :type host: str
    :param exc: The captured stage failure.
    :type exc: StageFailed
    :param index: The 1-based stage index, if part of a numbered sequence.
    :type index: int | None
    :param total: The total number of stages, if known.
    :type total: int | None
    :returns: A ClickException with a stage-named, output-bearing message.
    :rtype: click.ClickException
    """
    where = f"Stage {index}/{total} {label!r}" if index is not None else repr(label)
    tail = exc.output.strip()
    body = f"\n--- captured output ---\n{tail}" if tail else ""
    return click.ClickException(
        f"{where} failed on {host} (exit {exc.returncode}).{body}"
    )


def swarm_script_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Non-interactive env for remote install scripts (fresh VMs may need reboot).

    :param base: The environment to copy from; defaults to ``os.environ``.
    :type base: dict[str, str] | None
    :returns: A copy of *base* with ``DOCKER_REBOOT_ASSUME_YES`` defaulted to ``"1"``.
    :rtype: dict[str, str]
    """
    env = dict(base if base is not None else os.environ)
    env.setdefault("DOCKER_REBOOT_ASSUME_YES", "1")
    return env


def _shell_runner() -> str:
    """Resolve the shell interpreter used to run bundle scripts.

    :returns: The path to ``bash`` if available, otherwise ``sh``.
    :rtype: str
    :raises click.ClickException: If neither ``bash`` nor ``sh`` is on PATH.
    """
    bash = shutil.which("bash")
    if bash:
        return bash
    sh = shutil.which("sh")
    if sh:
        return sh
    msg = "bash or sh not found on PATH; required to run install scripts."
    raise click.ClickException(msg)


class BundleScriptRunner:
    """Runs scripts from a bundle directory with a shared environment."""

    def __init__(self, bundle_root: Path, *, env: dict[str, str]) -> None:
        """Bind the runner to a bundle directory and environment.

        :param bundle_root: The extracted bundle root that scripts live under.
        :type bundle_root: Path
        :param env: The environment variables passed to each script.
        :type env: dict[str, str]
        """
        self._bundle_root = bundle_root
        self._env = env

    def run(self, script_name: str, args: list[str]) -> None:
        """Run a bundle script, raising :class:`StageFailed` on a non-zero exit.

        Quiet by default: the script's stdout (its chatty progress) is captured
        and surfaced only on failure. stderr stays attached so ``sudo`` prompts
        and real errors remain visible/live. Run with ``--verbose`` (logger at
        DEBUG) to stream everything as it happens.

        :param script_name: The script's filename, relative to the bundle root.
        :type script_name: str
        :param args: Positional arguments passed to the script.
        :type args: list[str]
        :returns: None
        :rtype: None
        :raises click.ClickException: If the script file is missing.
        :raises StageFailed: If the script exits non-zero.
        """
        script = self._bundle_root / script_name
        if not script.is_file():
            msg = (
                f"Install bundle missing script {script.name} "
                "(re-download with arena on-prem install)."
            )
            raise click.ClickException(msg)
        runner = _shell_runner()
        cmd = [runner, str(script.resolve()), *args]
        # markup disabled: paths/args in the transcript may contain Rich-special chars.
        logger.debug("$ %s", " ".join(cmd), extra={"markup": False})
        if logger.isEnabledFor(logging.DEBUG):
            result = subprocess.run(
                cmd, check=False, cwd=self._bundle_root, env=self._env
            )
            captured = ""
        else:
            result = subprocess.run(
                cmd,
                check=False,
                cwd=self._bundle_root,
                env=self._env,
                stdout=subprocess.PIPE,
                text=True,
            )
            captured = result.stdout or ""
        if result.returncode != 0:
            raise StageFailed(script.name, result.returncode, captured)
