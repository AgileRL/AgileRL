"""SSH target parsing and local/remote command execution for Swarm installs."""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Any

import click

logger = logging.getLogger("agilerl.arena.on_prem")


@dataclass(frozen=True)
class SshTarget:
    """An ``ssh`` destination parsed once into its host, optional port, and locality.

    Accepts ``HOST``, ``user@HOST``, ``HOST:PORT``, and ``[ipv6]:PORT`` forms.
    """

    raw: str

    @classmethod
    def parse(cls, host: str) -> SshTarget:
        """Parse a host string into an :class:`SshTarget`.

        :param host: The host string (``HOST``, ``user@HOST``, ``HOST:PORT``,
            or ``[ipv6]:PORT``); surrounding whitespace is stripped.
        :type host: str
        :returns: The parsed target.
        :rtype: SshTarget
        """
        return cls(raw=host.strip())

    @property
    def hostname(self) -> str:
        """The bare hostname, stripped of any ``user@`` prefix or ``:port`` suffix.

        :returns: The hostname component of the target.
        :rtype: str
        """
        target = self.raw.split("@", 1)[-1]
        if target.startswith("[") and "]" in target:
            return target.split("]", 1)[0][1:]
        if ":" in target:
            return target.rsplit(":", 1)[0]
        return target

    @property
    def port(self) -> int | None:
        """The explicit port if the target carries one, else ``None``.

        :returns: The port number, or ``None`` if not specified.
        :rtype: int | None
        """
        target = self.raw.split("@", 1)[-1]
        if target.startswith("[") and "]:" in target:
            port_s = target.split("]:", 1)[1]
        elif ":" in target and not target.startswith("["):
            port_s = target.rsplit(":", 1)[1]
        else:
            return None
        return int(port_s) if port_s.isdigit() else None

    @property
    def is_local(self) -> bool:
        """True when commands should run on this machine rather than over ssh.

        :returns: Whether the target resolves to the local machine.
        :rtype: bool
        """
        h = self.hostname.lower()
        if h in {"localhost", "127.0.0.1", "::1"}:
            return True
        try:
            short = socket.gethostname().split(".", 1)[0]
            fqdn = socket.getfqdn()
        except OSError:
            return False
        return h == short or h == fqdn or h == fqdn.split(".", 1)[0]

    def connection_target(self, ssh_user: str | None) -> str:
        """Build the ssh(1) destination so ``~/.ssh/config`` applies like ``ssh HOST``.

        When no user is forced, use the host alias as given (e.g. ``manager-head``)
        instead of ``$USER@manager-head``, which ignores ``User`` in ssh_config.

        :param ssh_user: An explicit SSH login to force, or ``None`` to defer to
            ``~/.ssh/config`` / the ``SSH_USER`` environment variable.
        :type ssh_user: str | None
        :returns: The ssh(1) destination string (``HOST`` or ``user@HOST``).
        :rtype: str
        """
        explicit = (ssh_user or os.environ.get("SSH_USER") or "").strip()
        bare = self.hostname
        if explicit:
            return f"{explicit}@{bare}"
        if "@" in self.raw:
            user = self.raw.split("@", 1)[0]
            return f"{user}@{bare}"
        if self.port is not None:
            return bare
        return self.raw


class SshExecutor:
    """Runs shell commands on a host, locally or over ssh, with shared options."""

    def __init__(
        self,
        *,
        ssh_user: str | None = None,
        ssh_extra_opts: str | None = None,
    ) -> None:
        """Configure the executor with the SSH options applied to every host.

        :param ssh_user: SSH login to force on remote hosts, or ``None`` to defer
            to ``~/.ssh/config``.
        :type ssh_user: str | None
        :param ssh_extra_opts: Extra ssh(1) arguments (shell-split), or ``None``.
        :type ssh_extra_opts: str | None
        """
        self._ssh_user = ssh_user
        self._ssh_extra_opts = ssh_extra_opts

    def run(
        self,
        host: str,
        remote_cmd: str,
        *,
        capture: bool = False,
        check: bool = False,
    ) -> str | None:
        """Run *remote_cmd* on *host* (locally or over ssh).

        ``remote_cmd`` is evaluated by a remote (or local) shell, so any
        caller-supplied values interpolated into it MUST be escaped with
        ``shlex.quote`` to avoid shell injection.

        :param host: The target host (parsed via :meth:`SshTarget.parse`).
        :type host: str
        :param remote_cmd: The shell command to execute on the target.
        :type remote_cmd: str
        :param capture: If ``True``, capture and return stdout (stderr stays
            attached); otherwise stream output and return ``None``.
        :type capture: bool
        :param check: If ``True``, raise on a non-zero exit; otherwise log a
            warning and continue (best-effort).
        :type check: bool
        :returns: The captured stdout when ``capture`` is ``True``, else ``None``.
        :rtype: str | None
        :raises click.ClickException: If ssh is required but missing, or if the
            command exits non-zero and ``check`` is ``True``.
        """
        target = SshTarget.parse(host)
        run_kwargs: dict[str, Any] = {"check": False}
        if capture:
            run_kwargs["stdout"] = subprocess.PIPE
            run_kwargs["text"] = True

        if target.is_local:
            # markup disabled: the command transcript can contain Rich-special chars.
            logger.debug("$ %s", remote_cmd, extra={"markup": False})
            result = subprocess.run(["bash", "-lc", remote_cmd], **run_kwargs)
            self._handle_exit("command", result.returncode, check=check)
            return result.stdout if capture else None

        if not shutil.which("ssh"):
            msg = "ssh not found on PATH; required for dockerSwarm operations."
            raise click.ClickException(msg)

        ssh_cmd = [
            "ssh",
            "-tt",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
        if target.port is not None:
            ssh_cmd.extend(["-p", str(target.port)])
        if self._ssh_extra_opts:
            ssh_cmd.extend(shlex.split(self._ssh_extra_opts))
        ssh_cmd.append(target.connection_target(self._ssh_user))
        ssh_cmd.append(remote_cmd)
        # markup disabled: ssh targets like [::1] would be parsed as Rich markup.
        logger.debug("$ %s", " ".join(ssh_cmd), extra={"markup": False})
        result = subprocess.run(ssh_cmd, **run_kwargs)
        self._handle_exit("ssh", result.returncode, check=check)
        return result.stdout if capture else None

    @staticmethod
    def _handle_exit(label: str, returncode: int, *, check: bool) -> None:
        """Warn or raise on a non-zero exit, depending on *check*.

        :param label: A short label for the command kind (e.g. ``"ssh"``).
        :type label: str
        :param returncode: The process exit code.
        :type returncode: int
        :param check: If ``True``, raise on non-zero; otherwise warn.
        :type check: bool
        :returns: None
        :rtype: None
        :raises click.ClickException: If ``returncode`` is non-zero and ``check``.
        """
        if returncode == 0:
            return
        if check:
            msg = f"{label} exited {returncode}."
            raise click.ClickException(msg)
        logger.warning("%s exited %d", label, returncode)
