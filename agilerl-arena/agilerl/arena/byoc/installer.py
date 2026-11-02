# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Helm install/teardown orchestration.

:class:`ByocInstaller` holds the shared install/teardown flow (enable, require
class, download bundle, …) as template methods; :class:`HelmInstaller` fills in
the cluster steps. The module-level :func:`run_byoc_install` /
:func:`run_byoc_teardown` are a thin functional facade over the class.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, TypedDict

import click
from typing_extensions import NotRequired, Unpack

from agilerl.arena.byoc.api import ByocApi
from agilerl.arena.byoc.bundle import (
    extract_bundle,
    parse_helm_release_ids,
    validate_wireguard_bundle,
)
from agilerl.arena.byoc.endpoints import SetupKind
from agilerl.arena.byoc.scripts import BundleScriptRunner, StageFailed, stage_failure
from agilerl.arena.client import ArenaClient

logger = logging.getLogger("agilerl.arena.byoc")


def normalize_setup_type(setup_type: str) -> SetupKind:
    """Map CLI ``--setup-type`` to the Helm bundle flavor.

    :param setup_type: The raw ``--setup-type`` value (case/dash-insensitive).
    :type setup_type: str
    :returns: ``"helm"``.
    :rtype: SetupKind
    :raises click.ClickException: If *setup_type* is not ``helm``.
    """
    key = setup_type.strip().lower().replace("-", "")
    if key == "helm":
        return "helm"
    msg = f"Unsupported setup type {setup_type!r} (use helm)."
    raise click.ClickException(msg)


class ByocInstaller(ABC):
    """Shared install/teardown flow; subclasses provide the cluster-specific steps."""

    kind: ClassVar[SetupKind]

    def __init__(self, api: ByocApi, name: str) -> None:
        """Bind the installer to a BYOC API client and resource class name.

        :param api: The BYOC API wrapper to issue Arena requests through.
        :type api: ByocApi
        :param name: The resource class name to install or tear down.
        :type name: str
        """
        self.api = api
        self.name = name

    def install(
        self,
        skip_enable: bool = False,
        skip_verify: bool = False,
    ) -> None:
        """Enable BYOC, require the class, download the bundle, install.

        :param skip_enable: If ``True``, skip enabling the BYOC provider.
        :type skip_enable: bool
        :param skip_verify: If ``True``, skip post-install verification.
        :type skip_verify: bool
        :returns: None
        :rtype: None
        :raises click.ClickException: If the resource class does not exist.
        """
        if not skip_enable:
            self.api.enable()

        if self.api.find_class(self.name) is None:
            msg = (
                f"No BYOC resource class {self.name!r}. "
                "Create it first with ``arena cluster register --install`` "
                "(resource class on the cluster spec)."
            )
            raise click.ClickException(msg)

        with tempfile.TemporaryDirectory(prefix="arena-byoc-") as tmp:
            data = self.api.fetch_bundle(self.name, self.kind)
            bundle_root = extract_bundle(data, Path(tmp), class_name=self.name)
            validate_wireguard_bundle(bundle_root)
            self.install_cluster(bundle_root)
            if not skip_verify:
                self.verify(bundle_root)

        logger.info("BYOC install finished for class %r (%s).", self.name, self.kind)

    def teardown(
        self,
        skip_cluster: bool = False,
        disable_provider: bool = False,
    ) -> None:
        """Remove cluster workloads and optionally disable the BYOC provider.

        :param skip_cluster: If ``True``, skip Helm teardown steps.
        :type skip_cluster: bool
        :param disable_provider: If ``True``, disable the BYOC provider afterward.
        :type disable_provider: bool
        :returns: None
        :rtype: None
        """
        if not skip_cluster:
            self.teardown_cluster()
        if disable_provider:
            self.api.disable()
        logger.info("BYOC teardown finished for class %r (%s).", self.name, self.kind)

    def down(self) -> None:
        """Stop cluster workloads without removing the Helm release.

        :returns: None
        :rtype: None
        """
        self.down_cluster()
        logger.info("BYOC down finished for class %r (%s).", self.name, self.kind)

    # --- hooks ---------------------------------------------------------------
    @abstractmethod
    def install_cluster(self, bundle_root: Path) -> None:
        """Run the cluster install steps for an extracted bundle.

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        """

    @abstractmethod
    def verify(self, bundle_root: Path) -> None:
        """Post-install verification (best-effort; warns on problems).

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        """

    @abstractmethod
    def teardown_cluster(self) -> None:
        """Remove cluster workloads (no-op pieces are the subclass's choice).

        :returns: None
        :rtype: None
        """

    @abstractmethod
    def down_cluster(self) -> None:
        """Stop cluster workloads; the Helm release remains.

        :returns: None
        :rtype: None
        """


class HelmInstaller(ByocInstaller):
    """Local ``helm upgrade --install`` driven by the bundle's setup.sh."""

    kind: ClassVar[SetupKind] = "helm"

    def install_cluster(self, bundle_root: Path) -> None:
        """Run the bundle's ``setup.sh`` against the local kubectl context.

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        :raises click.ClickException: If ``setup.sh`` or ``helm`` is missing, or
            the script exits non-zero.
        """
        setup = bundle_root / "setup.sh"
        if not setup.is_file():
            msg = (
                "Helm bundle has no setup.sh. "
                "Re-run ``arena cluster register --install``."
            )
            raise click.ClickException(msg)
        if not shutil.which("helm"):
            msg = "helm not found on PATH; install Helm 3.x."
            raise click.ClickException(msg)
        logger.info("Running Helm setup (local kubectl context)…")
        runner = BundleScriptRunner(bundle_root, env=os.environ.copy())
        try:
            runner.run("setup.sh", [])
        except StageFailed as exc:
            err = stage_failure("Helm setup", "local", exc)
            raise err from exc

    def verify(self, bundle_root: Path) -> None:
        """Run the bundle's ``validate.sh`` if present, else warn to check kubectl.

        :param bundle_root: The extracted bundle root directory.
        :type bundle_root: Path
        :returns: None
        :rtype: None
        :raises click.ClickException: If ``validate.sh`` exits non-zero.
        """
        validate = bundle_root / "validate.sh"
        if not validate.is_file():
            logger.warning("Bundle has no validate.sh; check pods with kubectl.")
            return
        logger.info("Running Helm post-install validation…")
        runner = BundleScriptRunner(bundle_root, env=os.environ.copy())
        try:
            runner.run("validate.sh", [])
        except StageFailed as exc:
            err = stage_failure("Helm post-install validation", "local", exc)
            raise err from exc

    def down_cluster(self) -> None:
        """Scale all deployments for the release to zero replicas.

        :returns: None
        :rtype: None
        :raises click.ClickException: If ``kubectl`` is not on PATH.
        """
        if not shutil.which("kubectl"):
            msg = "kubectl not found on PATH; required for helm down."
            raise click.ClickException(msg)
        with tempfile.TemporaryDirectory(prefix="arena-byoc-down-") as tmp:
            data = self.api.fetch_bundle(self.name, self.kind)
            bundle_root = extract_bundle(data, Path(tmp), class_name=self.name)
            release, namespace = parse_helm_release_ids(bundle_root)
            logger.info(
                "Scaling Helm release %r (namespace %s) to zero replicas…",
                release,
                namespace,
            )
            result = subprocess.run(
                [
                    "kubectl",
                    "scale",
                    "deployment",
                    "-n",
                    namespace,
                    "-l",
                    f"app.kubernetes.io/instance={release}",
                    "--replicas=0",
                ],
                check=False,
            )
            if result.returncode != 0:
                logger.warning(
                    "kubectl scale exited %d (deployments may already be stopped).",
                    result.returncode,
                )

    def teardown_cluster(self) -> None:
        """Download the bundle to resolve release ids, then ``helm uninstall``.

        :returns: None
        :rtype: None
        """
        with tempfile.TemporaryDirectory(prefix="arena-byoc-teardown-") as tmp:
            data = self.api.fetch_bundle(self.name, self.kind)
            bundle_root = extract_bundle(data, Path(tmp), class_name=self.name)
            release, namespace = parse_helm_release_ids(bundle_root)
            self._helm_uninstall(release, namespace)

    @staticmethod
    def _helm_uninstall(release: str, namespace: str) -> None:
        """Run ``helm uninstall`` for *release* in *namespace* (best-effort).

        :param release: The Helm release name.
        :type release: str
        :param namespace: The Kubernetes namespace.
        :type namespace: str
        :returns: None
        :rtype: None
        :raises click.ClickException: If ``helm`` is not on PATH.
        """
        if not shutil.which("helm"):
            msg = "helm not found on PATH; install Helm 3.x or use --skip-cluster."
            raise click.ClickException(msg)
        logger.info("Removing Helm release %r (namespace %s)…", release, namespace)
        result = subprocess.run(
            ["helm", "uninstall", release, "--namespace", namespace],
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "helm uninstall exited %d (release may already be removed).",
                result.returncode,
            )


class InstallerOptions(TypedDict):
    """Keyword fields for ``build_installer``."""

    name: str


class ByocInstallOptions(TypedDict):
    """Keyword fields for ``run_byoc_install``."""

    name: str
    setup_type: str
    skip_enable: bool
    skip_verify: NotRequired[bool]


class ByocTeardownOptions(TypedDict):
    """Keyword fields for ``run_byoc_teardown``."""

    name: str
    setup_type: str
    skip_cluster: bool
    disable_provider: bool


def build_installer(api: ByocApi, **options: Unpack[InstallerOptions]) -> HelmInstaller:
    """Construct the Helm installer.

    :param api: The BYOC API wrapper.
    :type api: ByocApi
    :param name: The resource class name.
    :type name: str
    :returns: A :class:`HelmInstaller`.
    :rtype: HelmInstaller
    """
    return HelmInstaller(api, name=options["name"])


def run_byoc_install(
    client: ArenaClient, **options: Unpack[ByocInstallOptions]
) -> None:
    """Enable BYOC, require class exists, download bundle, run install scripts.

    :param client: The authenticated Arena client.
    :type client: ArenaClient
    :param name: The resource class name to install.
    :type name: str
    :param setup_type: The bundle flavor (``helm``).
    :type setup_type: str
    :param skip_enable: If ``True``, skip enabling the BYOC provider.
    :type skip_enable: bool
    :param skip_verify: If ``True``, skip post-install verification.
    :type skip_verify: bool
    :returns: None
    :rtype: None
    """
    name = options["name"]
    setup_type = options["setup_type"]
    skip_enable = options["skip_enable"]
    skip_verify = options.get("skip_verify", False)
    normalize_setup_type(setup_type)
    api = ByocApi(client)
    installer = build_installer(api, name=name)
    installer.install(skip_enable=skip_enable, skip_verify=skip_verify)


def run_byoc_teardown(
    client: ArenaClient, **options: Unpack[ByocTeardownOptions]
) -> None:
    """Remove cluster workloads and optionally disable the BYOC provider.

    :param client: The authenticated Arena client.
    :type client: ArenaClient
    :param name: The resource class name to tear down.
    :type name: str
    :param setup_type: The bundle flavor (``helm``).
    :type setup_type: str
    :param skip_cluster: If ``True``, skip Helm teardown steps.
    :type skip_cluster: bool
    :param disable_provider: If ``True``, disable the BYOC provider afterward.
    :type disable_provider: bool
    :returns: None
    :rtype: None
    """
    name = options["name"]
    setup_type = options["setup_type"]
    skip_cluster = options["skip_cluster"]
    disable_provider = options["disable_provider"]
    normalize_setup_type(setup_type)
    api = ByocApi(client)
    installer = build_installer(api, name=name)
    installer.teardown(
        skip_cluster=skip_cluster,
        disable_provider=disable_provider,
    )


def run_byoc_down(
    client: ArenaClient,
    name: str,
    setup_type: str,
) -> None:
    """Stop cluster workloads without removing the Helm release.

    :param client: The authenticated Arena client.
    :type client: ArenaClient
    :param name: The resource class name.
    :type name: str
    :param setup_type: The bundle flavor (``helm``).
    :type setup_type: str
    :returns: None
    :rtype: None
    """
    normalize_setup_type(setup_type)
    api = ByocApi(client)
    installer = build_installer(api, name=name)
    installer.down()
