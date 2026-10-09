# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Terraform execution for cloud cluster provisioning."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import NoReturn

import click

from agilerl.arena.byoc.provisioning.providers import get_provider
from agilerl.arena.byoc.provisioning.spec import ClusterSpec

MISSING_STATE_MARKER = "no state file was found"


def absolute_path(path: Path) -> Path:
    """Return ``path`` as an absolute path."""
    # getcwd() fails when the process directory has been removed. PWD is still set.
    expanded = path.expanduser()
    if not expanded.is_absolute():
        try:
            base = Path.cwd()
        except FileNotFoundError:
            pwd = os.environ.get("PWD")
            if not pwd:
                raise
            base = Path(pwd)
        expanded = base / expanded
    try:
        return expanded.resolve()
    except FileNotFoundError:
        return expanded


def is_missing_state_error(stderr: str | None) -> bool:
    """Return whether ``terraform state list`` failed because state is empty."""
    return MISSING_STATE_MARKER in (stderr or "").lower()


def terraform_address_matches(address: str, resources: tuple[str, ...]) -> bool:
    """Return whether a state address is one of ``resources``, including indexes."""
    return any(
        address == resource or address.startswith(f"{resource}[")
        for resource in resources
    )


def local_terraform_state_has_resources(path: Path) -> bool:
    """Return whether a local state file records managed resources."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(raw, dict):
        return False
    resources = raw.get("resources")
    return isinstance(resources, list) and any(
        isinstance(item, dict) and item.get("mode") == "managed" for item in resources
    )


@dataclass(frozen=True)
class ClusterOutputs:
    """Values required to register and install an Arena cluster."""

    cluster_name: str
    context: str
    kubeconfig_path: Path
    storage_access_key_id: str
    storage_secret_access_key: str
    worker_node_group_ids: dict[str, str]
    storage_class_name: str | None = None
    storage_endpoint: str | None = None
    storage_bucket: str | None = None
    efs_file_system_id: str | None = None
    vpc_id: str | None = None
    cluster_endpoint: str | None = None
    project_id: str | None = None
    storage_project_id: str | None = None
    gateway_api_parent_refs: object | None = None
    inference_domain: str | None = None
    inference_hostname_template: str | None = None
    inference_tls_secret_name: str | None = None


class TerraformRunner:
    """Run Terraform for one provider module."""

    def __init__(
        self,
        terraform: str = "terraform",
        run: object = subprocess.run,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.terraform = terraform
        self.run = run
        self.extra_env = dict(extra_env or {})

    def prepare(
        self,
        spec: ClusterSpec,
        module_name: str,
        state_dir: Path,
    ) -> Path:
        """Copy a packaged module and write the cluster input."""
        if shutil.which(self.terraform) is None:
            msg = "terraform not found on PATH."
            raise click.ClickException(msg)
        source = files("agilerl.arena.byoc.provisioning.terraform_modules").joinpath(
            module_name
        )
        work_dir = absolute_path(state_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, work_dir, dirs_exist_ok=True)
        (work_dir / "terraform.tfvars.json").write_text(
            json.dumps(
                {"cluster_name": spec.name, **spec.terraform_values()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return work_dir

    def initialize(self, work_dir: Path, spec: ClusterSpec) -> None:
        """Initialize Terraform providers and the S3 state backend."""
        get_provider(spec.provider).write_backend(work_dir, spec)
        args = ["init", "-input=false"]
        local_state = work_dir / "terraform.tfstate"
        if local_state.is_file() and local_terraform_state_has_resources(local_state):
            args.extend(["-migrate-state", "-force-copy"])
        elif local_state.is_file():
            local_state.unlink()
        self._run(work_dir, *args)

    def initialize_without_backend(self, work_dir: Path) -> None:
        """Initialize Terraform with local state only.

        Terraform refuses every later command while an uninitialized backend
        block is on disk, so the S3 backend must not exist yet.
        """
        (work_dir / "backend.tf").unlink(missing_ok=True)
        self._run(work_dir, "init", "-input=false", "-backend=false")

    def adopt_resources(self, work_dir: Path, resources: dict[str, str]) -> None:
        """Import existing resources that are missing from Terraform state."""
        if not resources:
            return
        click.echo("Adopting existing experiment storage into Terraform state.")
        present = set(self.state_list(work_dir))
        for address, resource_id in resources.items():
            if address not in present:
                self._run(work_dir, "import", "-input=false", address, resource_id)

    def plan(self, work_dir: Path) -> None:
        """Create a Terraform plan without changing infrastructure."""
        self._run(work_dir, "plan", "-input=false", "-out=arena.tfplan")

    def apply(self, work_dir: Path) -> None:
        """Apply the previously generated Terraform plan."""
        self._run(work_dir, "apply", "-input=false", "arena.tfplan")

    def apply_targets(self, work_dir: Path, targets: tuple[str, ...]) -> None:
        """Create the given Terraform target resources."""
        self._run(
            work_dir,
            "apply",
            "-input=false",
            "-auto-approve",
            *(f"-target={target}" for target in targets),
        )

    def destroy(self, work_dir: Path, exclude: tuple[str, ...] = ()) -> None:
        """Destroy infrastructure recorded in Terraform state, skipping ``exclude``."""
        args = ["destroy", "-input=false", "-auto-approve"]
        if exclude:
            targets = [
                address
                for address in self.state_list(work_dir)
                if not terraform_address_matches(address, exclude)
            ]
            if not targets:
                return
            args.extend(f"-target={address}" for address in targets)
        self._run(work_dir, *args)

    def state_list(self, work_dir: Path) -> tuple[str, ...]:
        """Return resource addresses recorded in the Terraform state.

        A backend whose state object does not exist yet holds no resources.
        """
        result = self._invoke(work_dir, "state", "list", capture_output=True)
        if result.returncode:
            if is_missing_state_error(result.stderr):
                return ()
            self._raise_failure(("state", "list"), result.stderr)
        return tuple(
            line.strip() for line in result.stdout.splitlines() if line.strip()
        )

    def terraform_output(self, work_dir: Path, name: str) -> object:
        """Read one Terraform output. ``terraform output -json NAME`` prints the value JSON."""
        result = self._run(work_dir, "output", "-json", name, capture_output=True)
        try:
            return json.loads(result.stdout)
        except (AttributeError, json.JSONDecodeError) as exc:
            msg = f"Terraform returned invalid JSON for output {name!r}."
            raise click.ClickException(msg) from exc

    def terraform_outputs(self, work_dir: Path) -> dict[str, object]:
        """Read JSON outputs from the Terraform state."""
        result = self._run(work_dir, "output", "-json", capture_output=True)
        try:
            raw = json.loads(result.stdout)
        except (AttributeError, json.JSONDecodeError) as exc:
            msg = "Terraform returned invalid JSON outputs."
            raise click.ClickException(msg) from exc
        return {name: item["value"] for name, item in raw.items()}

    def _run(
        self,
        work_dir: Path,
        *args: str,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        result = self._invoke(work_dir, *args, capture_output=capture_output)
        if result.returncode:
            self._raise_failure(args, result.stderr if capture_output else None)
        return result

    def _invoke(
        self,
        work_dir: Path,
        *args: str,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return self.run(
            [self.terraform, *args],
            cwd=work_dir,
            check=False,
            capture_output=capture_output,
            text=True,
            env={**os.environ, **self.extra_env},
        )

    @staticmethod
    def _raise_failure(args: tuple[str, ...], stderr: str | None) -> NoReturn:
        detail = stderr.strip() if stderr else ""
        msg = f"terraform {' '.join(args)} failed.{f' {detail}' if detail else ''}"
        raise click.ClickException(msg)
