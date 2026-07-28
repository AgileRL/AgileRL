# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Filesystem handling for the downloaded deployment bundle.

Covers unzipping, locating the bundle root, marking scripts executable,
validating the rendered WireGuard config, and reading Helm release identifiers.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

import click

from agilerl.arena.on_prem.endpoints import SetupKind


def resolve_bundle_root(extract_dir: Path) -> Path:
    """Find directory containing ``setup.sh`` after unzipping a deployment bundle.

    Bundles built by the Arena backend use the root prefix ``arena-train/``.
    Extracting into a pre-created ``…/arena-train`` folder would nest
    ``arena-train/arena-train/``.

    :param extract_dir: The directory the bundle zip was extracted into.
    :type extract_dir: Path
    :returns: The directory that holds ``setup.sh`` (the bundle root).
    :rtype: Path
    """
    for candidate in (extract_dir / "arena-train", extract_dir):
        if (candidate / "setup.sh").is_file():
            return candidate
    for setup in extract_dir.rglob("setup.sh"):
        parent = setup.parent
        if (parent / "chart").is_dir() or (parent / "install-docker.sh").is_file():
            return parent
    return extract_dir / "arena-train"


def prepare_bundle_scripts(bundle_root: Path) -> None:
    """Make every ``*.sh`` under the bundle executable.

    Zip extracts often drop the executable bit, so restore it for safety.

    :param bundle_root: The extracted bundle root directory.
    :type bundle_root: Path
    :returns: None
    :rtype: None
    """
    for path in bundle_root.rglob("*.sh"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o755)


def extract_bundle(data: bytes, dest_dir: Path, *, class_name: str) -> Path:
    """Write *data* as a zip under *dest_dir*, unzip it, and return its root.

    Scripts in the extracted tree are made executable before returning.

    :param data: The raw bytes of the deployment bundle zip.
    :type data: bytes
    :param dest_dir: The directory to write and extract the bundle into.
    :type dest_dir: Path
    :param class_name: The resource class name, used to name the archive file.
    :type class_name: str
    :returns: The extracted bundle root directory.
    :rtype: Path
    """
    archive = dest_dir / f"{class_name}-setup.zip"
    archive.write_bytes(data)
    extract_dir = dest_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)
    bundle_root = resolve_bundle_root(extract_dir)
    prepare_bundle_scripts(bundle_root)
    return bundle_root


def _validate_tun0_conf(path: Path) -> None:
    """Validate that the WireGuard ``tun0.conf`` is present and complete.

    :param path: Path to the bundle's ``config.d/tun0.conf``.
    :type path: Path
    :returns: None
    :rtype: None
    :raises click.ClickException: If the file is missing or missing required keys.
    """
    if not path.is_file():
        msg = f"Bundle missing WireGuard config {path.name}."
        raise click.ClickException(msg)
    text = path.read_text(encoding="utf-8")
    required = (
        "[Interface]",
        "[Peer]",
        "PrivateKey = ",
        "PublicKey = ",
        "PresharedKey = ",
        "Endpoint = ",
        "AllowedIPs = ",
    )
    missing = [token for token in required if token not in text]
    if missing:
        msg = (
            f"Invalid {path.name} in install bundle (missing: {', '.join(missing)}). "
            "Re-create the on-prem class or re-download the bundle."
        )
        raise click.ClickException(msg)


def validate_wireguard_bundle(bundle_root: Path, kind: SetupKind) -> None:
    """Check the bundle has a rendered WireGuard config for *kind*.

    :param bundle_root: The extracted bundle root directory.
    :type bundle_root: Path
    :param kind: The bundle flavor (``dockerSwarm`` or ``helm``).
    :type kind: SetupKind
    :returns: None
    :rtype: None
    :raises click.ClickException: If the required WireGuard config is missing.
    """
    if kind == "helm":
        values = bundle_root / "chart" / "values.yaml"
        if not values.is_file():
            msg = "Helm bundle missing chart/values.yaml."
            raise click.ClickException(msg)
        text = values.read_text(encoding="utf-8")
        for key in (
            "wireguard:",
            "gatewayHost:",
            "gatewayPublicKey:",
            "peerPrivateKey:",
            "peerIp:",
            "preSharedKey:",
        ):
            if key not in text:
                msg = f"Helm values.yaml missing {key!r} (WireGuard not rendered)."
                raise click.ClickException(msg)
    else:
        _validate_tun0_conf(bundle_root / "config.d" / "tun0.conf")
        stack = bundle_root / "arena-stack.yaml"
        if not stack.is_file():
            msg = "Docker Swarm bundle missing arena-stack.yaml."
            raise click.ClickException(msg)
        stack_text = stack.read_text(encoding="utf-8")
        if "/etc/wireguard/" not in stack_text:
            msg = "arena-stack.yaml does not mount WireGuard config (/etc/wireguard/)."
            raise click.ClickException(msg)


def parse_helm_release_ids(bundle_root: Path) -> tuple[str, str]:
    """Read ``clusterName`` from rendered chart values (matching the bundle's setup.sh).

    :param bundle_root: The extracted bundle root directory.
    :type bundle_root: Path
    :returns: A ``(release, namespace)`` pair for ``helm uninstall``.
    :rtype: tuple[str, str]
    :raises click.ClickException: If ``chart/values.yaml`` is missing.
    """
    values_file = bundle_root / "chart" / "values.yaml"
    if not values_file.is_file():
        msg = f"Helm bundle missing {values_file.name}; cannot determine release name."
        raise click.ClickException(msg)
    text = values_file.read_text(encoding="utf-8")
    match = re.search(
        r"^clusterName:\s*[\"']?([^\"'\s]+)[\"']?\s*$",
        text,
        re.MULTILINE,
    )
    cluster_name = match.group(1) if match else "arena-train"
    release = os.environ.get("RELEASE_NAME", cluster_name)
    namespace = os.environ.get("NAMESPACE", cluster_name)
    return release, namespace
