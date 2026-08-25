# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Download INTERNAL agilerl / agilerl-arena dists for a public PyPI upload.

CodeArtifact is the source of truth. This script does not rebuild or publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

ALLOWED_PACKAGES = frozenset({"agilerl", "agilerl-arena"})
DIST_PREFIX = {"agilerl": "agilerl", "agilerl-arena": "agilerl_arena"}
# Hub CodeArtifact SoT; the GitHub fetch job must pass the same values.
DEFAULT_DOMAIN = "agilerl-pypi"
DEFAULT_OWNER = "761296822462"
DEFAULT_REPOSITORY = "arena"
DEFAULT_REGION = "us-east-1"


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _require_allowed_package(package: str) -> str:
    if package not in ALLOWED_PACKAGES:
        _fail(f"error: package must be agilerl or agilerl-arena (got {package!r})")
    return package


def _require_public_version(version: str) -> str:
    if not version or version[0] in "-+" or not version[0].isdigit():
        _fail(
            "error: version must be a public PEP 440 string starting with a "
            f"digit (got {version!r})"
        )
    for bad in ("+", "/", "\\", "..", " ", "\n", "\t"):
        if bad in version:
            _fail(
                "error: version must be immutable PEP 440 with no local suffix "
                f"(got {version!r})"
            )
    return version


def _origin_type(describe_payload: dict[str, object]) -> str:
    package_version = describe_payload.get("packageVersion")
    if not isinstance(package_version, dict):
        return ""
    origin = package_version.get("origin")
    if not isinstance(origin, dict):
        return ""
    origin_type = origin.get("originType")
    if not isinstance(origin_type, str):
        return ""
    return origin_type


def _require_internal_origin(
    describe_payload: dict[str, object], package: str, version: str
) -> None:
    origin = _origin_type(describe_payload)
    if origin != "INTERNAL":
        shown = origin or "missing"
        _fail(
            f"error: {package}=={version} origin={shown}; "
            "need INTERNAL (already-published release). Refusing EXTERNAL "
            "pull-through and missing versions."
        )


def _asset_matches(name: str, dist_prefix: str, version: str) -> bool:
    wheel_prefix = f"{dist_prefix}-{version}-"
    sdist_name = f"{dist_prefix}-{version}.tar.gz"
    return name == sdist_name or (
        name.startswith(wheel_prefix) and name.endswith(".whl")
    )


def _matching_assets(
    assets: list[dict[str, object]], dist_prefix: str, version: str
) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    for asset in assets:
        name = asset.get("name")
        if isinstance(name, str) and _asset_matches(name, dist_prefix, version):
            matched.append(asset)
    return matched


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, asset: dict[str, object]) -> None:
    hashes = asset.get("hashes")
    if not isinstance(hashes, dict):
        _fail(f"error: {path.name} missing hashes; refusing to publish")
    expected = hashes.get("SHA-256") or hashes.get("SHA256")
    if not expected:
        _fail(f"error: {path.name} missing SHA-256; refusing to publish")
    actual = _sha256_hex(path)
    if actual.casefold() != str(expected).casefold():
        _fail(
            f"error: {path.name} SHA-256 mismatch (got {actual}, expected {expected})"
        )


def _require_wheel_and_sdist(names: list[str], package: str, version: str) -> None:
    has_wheel = any(name.endswith(".whl") for name in names)
    has_sdist = any(name.endswith(".tar.gz") for name in names)
    if not has_wheel or not has_sdist:
        _fail(
            f"error: {package}=={version} must include wheel and sdist "
            f"(got {names!r}); refusing to publish a partial set"
        )


class CodeArtifactClient:
    """aws CLI wrapper for CodeArtifact describe / list / get-package-version-asset."""

    def __init__(
        self,
        *,
        domain: str,
        domain_owner: str,
        repository: str,
        region: str,
    ) -> None:
        self.domain = domain
        self.domain_owner = domain_owner
        self.repository = repository
        self.region = region

    def _run(
        self, extra: list[str], *, outfile: Path | None = None
    ) -> dict[str, object]:
        cmd = [
            "aws",
            "codeartifact",
            *extra,
            "--domain",
            self.domain,
            "--domain-owner",
            self.domain_owner,
            "--repository",
            self.repository,
            "--format",
            "pypi",
            "--region",
            self.region,
        ]
        try:
            if outfile is not None:
                cmd.append(str(outfile))
                subprocess.run(cmd, check=True)
                return {}
            cmd.extend(["--output", "json"])
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            err = exc.stderr
            if isinstance(err, str) and err.strip():
                print(err.strip(), file=sys.stderr)
            raise
        if not completed.stdout.strip():
            return {}
        payload: object = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            _fail(f"error: unexpected aws output type {type(payload)!r}")
        return payload

    def describe_package_version(self, package: str, version: str) -> dict[str, object]:
        """Return the CodeArtifact package-version describe payload."""
        try:
            return self._run(
                [
                    "describe-package-version",
                    "--package",
                    package,
                    "--package-version",
                    version,
                ]
            )
        except subprocess.CalledProcessError:
            _fail(
                f"error: {package}=={version} is not on {self.repository}. "
                "The INTERNAL release must exist before this job can upload it."
            )

    def list_assets(self, package: str, version: str) -> list[dict[str, object]]:
        """Return every CodeArtifact asset for one package version."""
        payload = self._run(
            [
                "list-package-version-assets",
                "--package",
                package,
                "--package-version",
                version,
            ]
        )
        assets = payload.get("assets")
        if not isinstance(assets, list):
            _fail(f"error: {package}=={version} listed no assets; refusing to publish")
        if payload.get("nextToken"):
            _fail(
                f"error: {package}=={version} asset list is paginated; "
                "refusing a partial set"
            )
        return [asset for asset in assets if isinstance(asset, dict)]

    def download_asset(
        self, package: str, version: str, asset_name: str, dest: Path
    ) -> None:
        """Write one named asset to dest."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._run(
            [
                "get-package-version-asset",
                "--package",
                package,
                "--package-version",
                version,
                "--asset",
                asset_name,
            ],
            outfile=dest,
        )


def fetch_internal_dist(
    *,
    package: str,
    version: str,
    out_dir: Path,
    client: CodeArtifactClient,
) -> list[Path]:
    """Download and SHA-256-check the INTERNAL wheel and sdist into out_dir."""
    package = _require_allowed_package(package)
    version = _require_public_version(version)
    dist_prefix = DIST_PREFIX[package]
    _require_internal_origin(
        client.describe_package_version(package, version), package, version
    )
    assets = _matching_assets(
        client.list_assets(package, version), dist_prefix, version
    )
    if not assets:
        _fail(
            f"error: {package}=={version} has no matching wheel/sdist on "
            f"{client.repository}"
        )

    downloaded: list[Path] = []
    names: list[str] = []
    for asset in assets:
        name = asset["name"]
        if not isinstance(name, str):
            continue
        dest = out_dir / name
        client.download_asset(package, version, name, dest)
        _require_sha256(dest, asset)
        downloaded.append(dest)
        names.append(name)
    _require_wheel_and_sdist(names, package, version)
    return downloaded


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, choices=sorted(ALLOWED_PACKAGES))
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--domain-owner", default=DEFAULT_OWNER)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--region", default=DEFAULT_REGION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Fetch INTERNAL dist files into --out-dir."""
    args = _parse_args(argv)
    client = CodeArtifactClient(
        domain=args.domain,
        domain_owner=args.domain_owner,
        repository=args.repository,
        region=args.region,
    )
    paths = fetch_internal_dist(
        package=args.package,
        version=args.version,
        out_dir=args.out_dir,
        client=client,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
