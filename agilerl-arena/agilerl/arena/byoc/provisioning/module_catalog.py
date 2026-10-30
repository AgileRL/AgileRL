# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Versioned Terraform modules for ``arena cluster``."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, Protocol
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import click
import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from packaging.version import Version

MODULE_VERSION_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
OUTPUT_NAMES = re.compile(r'output\s+"([^"]+)"')
RESOURCE_ADDRESSES = re.compile(r'resource\s+"([^"]+)"\s+"([^"]+)"')
VARIABLE_BLOCK = re.compile(r'variable\s+"([^"]+)"\s*\{', re.MULTILINE)
VARIABLE_DEFAULT = re.compile(r"^\s*default\s*=", re.MULTILINE)
PIN_FILE = "module.json"
BUNDLED_MODULE_VERSION = "1.0.0"
# Files the CLI writes. They are not part of a published module.
MODULE_FILE_SKIP = frozenset({"manifest.json", "provider.tf"})
CATALOG_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class OrchestrationContract:
    """What this CLI knows how to run for one cloud."""

    name: str
    provider: str
    variables: frozenset[str]
    required_outputs: frozenset[str]
    required_resources: tuple[str, ...]
    pre_apply_targets: tuple[str, ...]
    retain_on_destroy: tuple[str, ...]
    always_destroy_excluded: tuple[str, ...]


AWS_V1 = OrchestrationContract(
    name="aws.v1",
    provider="aws",
    variables=frozenset(
        {
            "cluster_name",
            "region",
            "storage_bucket_name",
            "storage_endpoint",
            "kubernetes_version",
            "arena_min_node_count",
            "arena_max_node_count",
            "arena_instance_type",
            "provision_shared_filesystem",
            "enable_gateway_api",
            "workers",
        }
    ),
    required_outputs=frozenset(
        {
            "cluster_name",
            "context",
            "cluster_id",
            "region",
            "cluster_endpoint",
            "vpc_id",
            "efs_file_system_id",
            "storage_access_key_id",
            "storage_secret_access_key",
            "worker_node_group_ids",
        }
    ),
    required_resources=(
        "aws_eks_cluster.arena",
        "aws_eks_node_group.arena",
        "aws_s3_bucket.arena_data",
        "aws_s3_bucket_public_access_block.arena_data",
        "aws_iam_user.storage",
        "aws_iam_access_key.storage",
        "aws_iam_user_policy.storage",
    ),
    pre_apply_targets=("aws_eks_cluster.arena",),
    retain_on_destroy=(
        "aws_s3_bucket.arena_data",
        "aws_s3_bucket_public_access_block.arena_data",
        "aws_iam_user.storage",
        "aws_iam_access_key.storage",
        "aws_iam_user_policy.storage",
    ),
    always_destroy_excluded=(),
)
NEBIUS_V1 = OrchestrationContract(
    name="nebius.v1",
    provider="nebius",
    variables=frozenset(
        {
            "cluster_name",
            "tenant_id",
            "region",
            "project_id",
            "storage_project_id",
            "subnet_id",
            "service_account_id",
            "storage_bucket_name",
            "storage_endpoint",
            "kubernetes_version",
            "etcd_cluster_size",
            "arena_min_node_count",
            "arena_max_node_count",
            "arena_platform",
            "arena_preset",
            "workers",
            "object_storage_size_gib",
            "provision_shared_filesystem",
            "filesystem_type",
            "filesystem_size_gib",
            "filesystem_mount_tag",
        }
    ),
    required_outputs=frozenset(
        {
            "cluster_name",
            "context",
            "cluster_id",
            "project_id",
            "storage_project_id",
            "storage_access_key_id",
            "storage_secret_access_key",
            "worker_node_group_ids",
        }
    ),
    required_resources=(
        "nebius_mk8s_v1_cluster.arena",
        "nebius_iam_v2_project.arena",
        "nebius_iam_v2_project.storage",
        "nebius_storage_v1_bucket.arena_data",
        "nebius_iam_v2_access_key.storage",
        "nebius_iam_v1_group_membership.storage_editor",
        "nebius_iam_v1_group.storage_editors",
        "nebius_iam_v1_service_account.storage",
    ),
    pre_apply_targets=(),
    retain_on_destroy=(
        "nebius_storage_v1_bucket.arena_data",
        "nebius_iam_v2_access_key.storage",
        "nebius_iam_v1_group_membership.storage_editor",
        "nebius_iam_v1_group.storage_editors",
        "nebius_iam_v1_service_account.storage",
    ),
    always_destroy_excluded=(
        "nebius_iam_v2_project.arena",
        "nebius_iam_v2_project.storage",
    ),
)
CONTRACTS = {"aws.v1": AWS_V1, "nebius.v1": NEBIUS_V1}
MODULE_NAMES = {"aws": "arena-cluster-aws", "nebius": "arena-cluster-nebius"}


@dataclass(frozen=True)
class ModuleManifest:
    """One published module release."""

    schema: int
    name: str
    provider: str
    version: str
    orchestration: str
    terraform_version: str
    digest: str
    files: tuple[str, ...]

    def pin_record(self, previous_version: str | None) -> ModulePin:
        """Return the local pin written when this release is installed."""
        return ModulePin(
            name=self.name,
            version=self.version,
            digest=self.digest,
            orchestration=self.orchestration,
            files=self.files,
            previous_version=previous_version,
        )


@dataclass(frozen=True)
class ModulePin:
    """The module release installed for one cluster."""

    name: str
    version: str
    digest: str
    orchestration: str
    files: tuple[str, ...]
    previous_version: str | None = None

    def public_record(self) -> dict[str, str]:
        """Return the pin stored on the Arena cluster record."""
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "orchestration": self.orchestration,
        }


@dataclass(frozen=True)
class CatalogRelease:
    """One row of a provider index."""

    version: str
    orchestration: str
    digest: str
    yanked: bool


class CatalogBytes(Protocol):
    """Reads one catalog URL. Tests substitute a local map."""

    def read(self, url: str) -> bytes:
        """Return the bytes at ``url``."""


def _fail(message: str) -> NoReturn:
    """Raise a command error."""
    raise click.ClickException(message)


def pin_path(state_dir: Path) -> Path:
    """Return ``module.json`` beside a Terraform work directory."""
    return state_dir.parent / PIN_FILE


def module_digest(module_dir: Path, names: tuple[str, ...]) -> str:
    """Hash module file names and bytes. The digest is not of the tarball."""
    hasher = hashlib.sha256()
    for name in sorted(names):
        payload = (
            (module_dir / name)
            .read_bytes()
            .replace(b"\r\n", b"\n")
            .replace(b"\r", b"\n")
        )
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(payload)
        hasher.update(b"\0")
    return "sha256:" + hasher.hexdigest()


def module_file_names(module_dir: Path) -> tuple[str, ...]:
    """Return the module files a manifest publishes."""
    return tuple(
        sorted(
            path.name
            for path in module_dir.iterdir()
            if path.is_file()
            and path.name not in MODULE_FILE_SKIP
            and not path.name.endswith(".sig")
        )
    )


def load_manifest(path: Path) -> ModuleManifest:
    """Load a module manifest from disk."""
    return parse_manifest(path.read_bytes())


def parse_manifest(payload: bytes) -> ModuleManifest:
    """Parse a module manifest. The caller has already checked its signature when it came from a catalog."""
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        msg = "Module manifest is not JSON."
        raise click.ClickException(msg) from exc
    if not isinstance(raw, dict):
        _fail("Module manifest must be a JSON object.")
    schema = raw.get("schema")
    if schema != 1:
        _fail(f"Unsupported module manifest schema {schema!r}.")
    version = _require_text(raw, "version")
    if MODULE_VERSION_RE.fullmatch(version) is None:
        _fail(f"Module version {version!r} is not MAJOR.MINOR.PATCH.")
    orchestration = _require_text(raw, "orchestration")
    if orchestration not in CONTRACTS:
        _fail(
            f"Module orchestration {orchestration!r} is not implemented. Upgrade the arena CLI."
        )
    files = raw.get("files")
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(item, str) for item in files)
    ):
        _fail("Module manifest field 'files' must be a list of file names.")
    names = tuple(_safe_file_name(item) for item in files)
    digest = _require_text(raw, "digest")
    if not digest.startswith("sha256:") or len(digest) != 7 + 64:
        _fail("Module digest must be sha256 and 64 hex characters.")
    return ModuleManifest(
        schema=1,
        name=_require_text(raw, "name"),
        provider=_require_text(raw, "provider"),
        version=version,
        orchestration=orchestration,
        terraform_version=_require_text(raw, "terraform_version"),
        digest=digest,
        files=names,
    )


def read_module_pin(state_dir: Path) -> ModulePin | None:
    """Return the pin for a Terraform work directory, or ``None`` when the cluster has none."""
    path = pin_path(state_dir)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Module pin {path} is not JSON."
        raise click.ClickException(msg) from exc
    if not isinstance(raw, dict):
        _fail(f"Module pin {path} must be a JSON object.")
    version = _require_text(raw, "version")
    files = raw.get("files")
    if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
        _fail(f"Module pin {path} field 'files' must be a list of file names.")
    previous = raw.get("previous_version")
    if previous is not None and not isinstance(previous, str):
        _fail(f"Module pin {path} field 'previous_version' must be a string.")
    return ModulePin(
        name=_require_text(raw, "name"),
        version=version,
        digest=_require_text(raw, "digest"),
        orchestration=_require_text(raw, "orchestration"),
        files=tuple(files),
        previous_version=previous,
    )


def write_module_pin(state_dir: Path, pin: ModulePin) -> Path:
    """Write the pin next to a Terraform work directory."""
    path = pin_path(state_dir)
    body: dict[str, object] = {
        "name": pin.name,
        "version": pin.version,
        "digest": pin.digest,
        "orchestration": pin.orchestration,
        "files": list(pin.files),
    }
    if pin.previous_version is not None:
        body["previous_version"] = pin.previous_version
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return path


def selected_module_version(
    state_dir: Path,
    spec_version: str | None,
    flag_version: str | None,
) -> str | None:
    """Choose the module version. A pin wins unless ``--module-version`` upgrades it."""
    if flag_version is not None and MODULE_VERSION_RE.fullmatch(flag_version) is None:
        _fail(f"Module version {flag_version!r} is not MAJOR.MINOR.PATCH.")
    pin = read_module_pin(state_dir)
    if flag_version is not None:
        return flag_version
    if pin is not None and spec_version is not None and pin.version != spec_version:
        _fail(
            f"Cluster module is pinned to {pin.version}. "
            f"Spec module_version is {spec_version}. "
            "Pass --module-version to upgrade."
        )
    if pin is not None:
        return pin.version
    return spec_version


def check_module(module_dir: Path, manifest: ModuleManifest) -> None:
    """Reject a module that does not match the orchestration it names."""
    contract = CONTRACTS[manifest.orchestration]
    if manifest.provider != contract.provider:
        _fail(
            f"Module {manifest.name} provider is {manifest.provider!r}, "
            f"orchestration {contract.name} is {contract.provider}."
        )
    if manifest.name != MODULE_NAMES[contract.provider]:
        _fail(
            f"Module name {manifest.name!r} does not match {MODULE_NAMES[contract.provider]}."
        )
    text = "\n".join(
        (module_dir / name).read_text(encoding="utf-8")
        for name in manifest.files
        if name.endswith(".tf")
    )
    declared = _variable_defaults(text)
    missing = sorted(contract.variables - declared.keys())
    if missing:
        _fail(
            f"Module {manifest.name} {manifest.version} is missing variables: {', '.join(missing)}."
        )
    required_extras = sorted(
        name
        for name in declared
        if name not in contract.variables and not declared[name]
    )
    if required_extras:
        _fail(
            f"Module {manifest.name} {manifest.version} adds required variables "
            f"{', '.join(required_extras)}. That needs a new orchestration contract."
        )
    outputs = set(OUTPUT_NAMES.findall(text))
    missing_outputs = sorted(contract.required_outputs - outputs)
    if missing_outputs:
        _fail(
            f"Module {manifest.name} {manifest.version} is missing outputs: {', '.join(missing_outputs)}."
        )
    resources = {f"{kind}.{name}" for kind, name in RESOURCE_ADDRESSES.findall(text)}
    missing_resources = [
        address for address in contract.required_resources if address not in resources
    ]
    if missing_resources:
        _fail(
            "Module "
            f"{manifest.name} {manifest.version} is missing resources: {', '.join(missing_resources)}."
        )
    digest = module_digest(module_dir, manifest.files)
    if digest != manifest.digest:
        _fail(
            f"Module {manifest.name} {manifest.version} digest is {digest}, manifest says {manifest.digest}."
        )


def install_pinned_module(
    packaged_dir: Path,
    work_dir: Path,
    *,
    requested_version: str | None,
    catalog: ModuleCatalog | None,
) -> ModulePin:
    """Copy the selected module into ``work_dir`` and write the pin."""
    bundled = load_manifest(packaged_dir / "manifest.json")
    source, manifest = _resolve_source(
        packaged_dir, bundled, requested_version, catalog
    )
    check_module(source, manifest)
    previous = read_module_pin(work_dir)
    previous_version = _previous_version(previous, manifest.version)
    _copy_module_files(source, work_dir, manifest.files, previous)
    pin = manifest.pin_record(previous_version)
    write_module_pin(work_dir, pin)
    return pin


def verify_signature(public_key_pem: bytes, payload: bytes, signature: bytes) -> None:
    """Reject a catalog file whose Ed25519 signature does not match."""
    key = load_pem_public_key(public_key_pem)
    if not isinstance(key, Ed25519PublicKey):
        _fail("Module catalog public key must be Ed25519.")
    try:
        key.verify(signature, payload)
    except InvalidSignature as exc:
        msg = "Module catalog signature is invalid."
        raise click.ClickException(msg) from exc


class ModuleCatalog:
    """A signed index of module tarballs."""

    def __init__(
        self,
        base_url: str,
        public_key_pem: bytes,
        cache_dir: Path | None = None,
        reader: CatalogBytes | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.public_key_pem = public_key_pem
        self.cache_dir = cache_dir or (Path.home() / ".cache" / "arena" / "modules")
        self.reader = reader or HttpCatalogReader()

    def latest(self, provider: str) -> CatalogRelease | None:
        """Return the newest release this CLI can run, skipping yanked rows."""
        compatible = [
            release
            for release in self.index(provider)
            if not release.yanked and release.orchestration in CONTRACTS
        ]
        if not compatible:
            return None
        return max(compatible, key=lambda release: Version(release.version))

    def index(self, provider: str) -> tuple[CatalogRelease, ...]:
        """Return the signed provider index."""
        payload = self._signed(f"modules/{provider}/index.json")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            msg = "Module catalog index is not JSON."
            raise click.ClickException(msg) from exc
        if not isinstance(raw, dict) or raw.get("provider") != provider:
            _fail(f"Module catalog index is not for {provider}.")
        releases = raw.get("releases")
        if not isinstance(releases, list):
            _fail("Module catalog index field 'releases' must be a list.")
        return tuple(_parse_release(item) for item in releases)

    def fetch(self, provider: str, version: str) -> Path:
        """Download one release into the cache and return its directory."""
        if MODULE_VERSION_RE.fullmatch(version) is None:
            _fail(f"Module version {version!r} is not MAJOR.MINOR.PATCH.")
        listed = next(
            (item for item in self.index(provider) if item.version == version), None
        )
        if listed is None:
            _fail(f"Module catalog has no {provider} {version}.")
        prefix = f"modules/{provider}/{version}"
        manifest_bytes = self._signed(f"{prefix}/manifest.json")
        manifest = parse_manifest(manifest_bytes)
        if manifest.provider != provider or manifest.version != version:
            _fail(f"Module manifest identity does not match {provider} {version}.")
        if (
            manifest.digest != listed.digest
            or manifest.orchestration != listed.orchestration
        ):
            _fail(
                f"Module {provider} {version} manifest does not match the catalog index."
            )
        cached = self.cache_dir / provider / version
        if (
            cached.is_dir()
            and (cached / "manifest.json").read_bytes() == manifest_bytes
        ):
            check_module(cached, manifest)
            return cached
        archive = self.reader.read(self._url(f"{prefix}/module.tar.gz"))
        staging = self.cache_dir / provider / f"{version}.partial"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _extract_module(archive, staging, manifest.files)
        try:
            check_module(staging, manifest)
        except click.ClickException:
            shutil.rmtree(staging)
            raise
        (staging / "manifest.json").write_bytes(manifest_bytes)
        if cached.exists():
            shutil.rmtree(cached)
        staging.rename(cached)
        return cached

    def _signed(self, relative: str) -> bytes:
        payload = self.reader.read(self._url(relative))
        signature = self.reader.read(self._url(relative + ".sig"))
        try:
            decoded = base64.b64decode(signature.strip(), validate=True)
        except ValueError as exc:
            msg = f"Module catalog signature for {relative} is not base64."
            raise click.ClickException(msg) from exc
        verify_signature(self.public_key_pem, payload, decoded)
        return payload

    def _url(self, relative: str) -> str:
        return f"{self.base_url}/{relative}"


class HttpCatalogReader:
    """Read catalog files over HTTP or from a ``file:`` URL."""

    def read(self, url: str) -> bytes:
        """Return the document at ``url``."""
        parsed = urlsplit(url)
        if parsed.scheme == "file":
            try:
                return Path(url2pathname(unquote(parsed.path))).read_bytes()
            except OSError as exc:
                msg = f"Module catalog file is missing: {url}."
                raise CatalogUnavailable(msg) from exc
        if parsed.scheme not in ("http", "https"):
            _fail(f"Unsupported module catalog URL {url}.")
        try:
            response = httpx.get(
                url, timeout=CATALOG_TIMEOUT_SECONDS, follow_redirects=True
            )
        except httpx.HTTPError as exc:
            msg = f"Module catalog request failed for {url}."
            raise CatalogUnavailable(msg) from exc
        if response.status_code != 200:
            _fail(f"Module catalog returned {response.status_code} for {url}.")
        return response.content


def open_module_catalog(
    base_url: str | None, public_key_path: str | None
) -> ModuleCatalog | None:
    """Open the catalog from CLI config. No URL means the bundled modules only."""
    if base_url is None or not base_url.strip():
        return None
    if public_key_path is None or not public_key_path.strip():
        _fail("Pass --module-catalog-public-key to verify the module catalog.")
    key_path = Path(public_key_path).expanduser()
    if not key_path.is_file():
        _fail(f"Module catalog public key is not a file: {key_path}.")
    return ModuleCatalog(base_url.strip(), key_path.read_bytes())


def _resolve_source(
    packaged_dir: Path,
    bundled: ModuleManifest,
    requested_version: str | None,
    catalog: ModuleCatalog | None,
) -> tuple[Path, ModuleManifest]:
    if requested_version is None and catalog is not None:
        try:
            latest = catalog.latest(bundled.provider)
        except CatalogUnavailable as exc:
            click.echo(
                f"Module catalog is unreachable ({exc}). "
                f"Using bundled {bundled.name} {bundled.version}."
            )
            return packaged_dir, bundled
        if latest is None or latest.version == bundled.version:
            return packaged_dir, bundled
        fetched = catalog.fetch(bundled.provider, latest.version)
        return fetched, load_manifest(fetched / "manifest.json")
    if requested_version is None or requested_version == bundled.version:
        return packaged_dir, bundled
    if catalog is None:
        _fail(
            f"Module {bundled.provider} {requested_version} is not bundled with this CLI. "
            "Pass --module-catalog-url to fetch it."
        )
    fetched = catalog.fetch(bundled.provider, requested_version)
    return fetched, load_manifest(fetched / "manifest.json")


def _copy_module_files(
    source: Path,
    work_dir: Path,
    names: tuple[str, ...],
    previous: ModulePin | None,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    incoming = set(names)
    if previous is not None:
        for name in previous.files:
            if name not in incoming:
                (work_dir / name).unlink(missing_ok=True)
    for name in names:
        target = work_dir / name
        shutil.copyfile(source / name, target)


def _previous_version(previous: ModulePin | None, version: str) -> str | None:
    if previous is None:
        return None
    if previous.version == version:
        return previous.previous_version
    return previous.version


def _extract_module(archive: bytes, dest: Path, names: tuple[str, ...]) -> None:
    allowed = set(names)
    try:
        tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz")
    except tarfile.TarError as exc:
        msg = "Module archive is not a gzip tar."
        raise click.ClickException(msg) from exc
    with tar:
        members = [
            member for member in tar.getmembers() if member.name not in ("", ".", "./")
        ]
        seen: set[str] = set()
        for member in members:
            name = _archive_name(member.name)
            if not member.isfile() or member.issym() or member.islnk():
                _fail(f"Module archive entry {member.name!r} is not a file.")
            if name not in allowed:
                _fail(f"Module archive contains unexpected file {name!r}.")
            extracted = tar.extractfile(member)
            if extracted is None:
                _fail(f"Module archive entry {name!r} is empty.")
            (dest / name).write_bytes(extracted.read())
            seen.add(name)
    missing = sorted(allowed - seen)
    if missing:
        _fail(f"Module archive is missing {', '.join(missing)}.")


def _archive_name(name: str) -> str:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        _fail(f"Module archive entry {name!r} escapes the archive.")
    if len(pure.parts) != 1:
        _fail(f"Module archive entry {name!r} is not a module file.")
    return pure.name


def _parse_release(item: object) -> CatalogRelease:
    if not isinstance(item, dict):
        _fail("Module catalog release must be an object.")
    version = _require_text(item, "version")
    if MODULE_VERSION_RE.fullmatch(version) is None:
        _fail(f"Module version {version!r} is not MAJOR.MINOR.PATCH.")
    yanked = item.get("yanked", False)
    if not isinstance(yanked, bool):
        _fail(f"Module {version} field 'yanked' must be a boolean.")
    return CatalogRelease(
        version=version,
        orchestration=_require_text(item, "orchestration"),
        digest=_require_text(item, "digest"),
        yanked=yanked,
    )


def _require_text(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        _fail(f"Module field {key!r} must be a string.")
    return value.strip()


def _safe_file_name(name: str) -> str:
    if Path(name).name != name or name in {".", ".."}:
        _fail(f"Module file name {name!r} is not a single path segment.")
    return name


def _variable_defaults(text: str) -> dict[str, bool]:
    matches = list(VARIABLE_BLOCK.finditer(text))
    declared: dict[str, bool] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        declared[match.group(1)] = (
            VARIABLE_DEFAULT.search(text[match.end() : end]) is not None
        )
    return declared


class CatalogUnavailable(click.ClickException):
    """The catalog host could not be read."""
