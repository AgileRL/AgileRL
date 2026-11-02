# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Versioned Terraform module pins and the signed catalog."""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from agilerl.arena.byoc.provisioning.module_catalog import (
    AWS_V1,
    NEBIUS_V1,
    CatalogUnavailable,
    HttpCatalogReader,
    ModuleCatalog,
    ModulePin,
    _archive_name,
    _extract_module,
    _parse_release,
    _previous_version,
    _require_text,
    _resolve_source,
    _safe_file_name,
    check_module,
    install_pinned_module,
    load_manifest,
    module_digest,
    module_file_names,
    open_module_catalog,
    parse_manifest,
    read_module_pin,
    selected_module_version,
    write_module_pin,
)
from agilerl.arena.byoc.provisioning.providers.aws.provider import (
    AWS_EKS_CLUSTER_RESOURCE,
    AWS_OBJECT_STORAGE_RESOURCES,
)
from agilerl.arena.byoc.provisioning.providers.nebius.provider import (
    NEBIUS_COMPUTE_PROJECT_RESOURCE,
    NEBIUS_OBJECT_STORAGE_RESOURCES,
    NEBIUS_STORAGE_PROJECT_RESOURCE,
)
from agilerl.arena.byoc.provisioning.spec import ClusterSpec
from agilerl.arena.byoc.provisioning.terraform import TerraformRunner
from agilerl.arena.byoc.provisioning.terraform_modules import (
    __path__ as terraform_modules_path,
)

MODULES = Path(terraform_modules_path[0])


def _aws_spec() -> ClusterSpec:
    return ClusterSpec.model_validate(
        {
            "provider": "aws",
            "name": "eks-byoc",
            "terraform_state": {"bucket": "eks-tfstate"},
            "arena": {
                "storage": {
                    "bucket": "eks-data",
                    "endpoint": "https://s3.eu-west-1.amazonaws.com",
                },
                "inference": {"domain": "inference.example.com"},
            },
            "aws": {
                "region": "eu-west-1",
                "object_storage": {"size_gib": 1024},
                "shared_filesystem": {"provision": False},
            },
        }
    )


def _keypair() -> tuple[Ed25519PrivateKey, bytes]:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    return private, public


def _sign(private: Ed25519PrivateKey, payload: bytes) -> bytes:
    return base64.b64encode(private.sign(payload)) + b"\n"


def _publish_aws(
    root: Path, private: Ed25519PrivateKey, *, yanked: bool = False
) -> None:
    source = MODULES / "aws"
    manifest = load_manifest(source / "manifest.json")
    release = root / "modules" / "aws" / manifest.version
    release.mkdir(parents=True)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name in manifest.files:
            tar.add(source / name, arcname=name)
    (release / "module.tar.gz").write_bytes(archive.getvalue())
    manifest_bytes = (source / "manifest.json").read_bytes()
    (release / "manifest.json").write_bytes(manifest_bytes)
    (release / "manifest.json.sig").write_bytes(_sign(private, manifest_bytes))
    index = {
        "provider": "aws",
        "releases": [
            {
                "version": manifest.version,
                "orchestration": manifest.orchestration,
                "digest": manifest.digest,
                "yanked": yanked,
            },
            {
                "version": "9.0.0",
                "orchestration": "aws.v1",
                "digest": manifest.digest,
                "yanked": True,
            },
        ],
    }
    index_bytes = json.dumps(index).encode()
    index_path = root / "modules" / "aws" / "index.json"
    index_path.write_bytes(index_bytes)
    index_path.with_name("index.json.sig").write_bytes(_sign(private, index_bytes))


class TestOrchestrationContract:
    def test_aws_addresses_match_the_provider(self) -> None:
        assert AWS_V1.pre_apply_targets == (AWS_EKS_CLUSTER_RESOURCE,)
        assert AWS_V1.retain_on_destroy == AWS_OBJECT_STORAGE_RESOURCES

    def test_nebius_addresses_match_the_provider(self) -> None:
        assert NEBIUS_V1.retain_on_destroy == NEBIUS_OBJECT_STORAGE_RESOURCES
        assert NEBIUS_V1.always_destroy_excluded == (
            NEBIUS_COMPUTE_PROJECT_RESOURCE,
            NEBIUS_STORAGE_PROJECT_RESOURCE,
        )

    def test_bundled_modules_match_their_manifests(self) -> None:
        for provider in ("aws", "nebius"):
            directory = MODULES / provider
            manifest = load_manifest(directory / "manifest.json")

            assert manifest.files == module_file_names(directory)
            check_module(directory, manifest)

    def test_module_digest_treats_crlf_as_lf(self, tmp_path: Path) -> None:
        names = ("main.tf",)
        lf_dir = tmp_path / "lf"
        crlf_dir = tmp_path / "crlf"
        lf_dir.mkdir()
        crlf_dir.mkdir()
        (lf_dir / "main.tf").write_bytes(b'resource "x" {}\n')
        (crlf_dir / "main.tf").write_bytes(b'resource "x" {}\r\n')

        assert module_digest(lf_dir, names) == module_digest(crlf_dir, names)

    def test_file_catalog_url_reads_the_document(self, tmp_path: Path) -> None:
        catalog_file = tmp_path / "index.json"
        catalog_file.write_bytes(b'{"provider":"aws"}')

        assert HttpCatalogReader().read(catalog_file.as_uri()) == b'{"provider":"aws"}'

    def test_nebius_module_omits_provider_auth(self) -> None:
        manifest = load_manifest(MODULES / "nebius" / "manifest.json")

        assert "provider.tf" not in manifest.files


class TestModulePin:
    def test_spec_version_that_disagrees_with_the_pin_stops(
        self, tmp_path: Path
    ) -> None:
        state_dir = tmp_path / "terraform"
        state_dir.mkdir()
        write_module_pin(
            state_dir,
            ModulePin(
                name="arena-cluster-aws",
                version="1.0.0",
                digest="sha256:" + "ab" * 32,
                orchestration="aws.v1",
                files=("main.tf",),
            ),
        )

        with pytest.raises(click.ClickException, match=r"pinned to 1\.0\.0"):
            selected_module_version(state_dir, "1.2.0", None)

        assert selected_module_version(state_dir, "1.2.0", "1.2.0") == "1.2.0"

    def test_prepare_pins_the_bundled_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.terraform.shutil.which",
            lambda _name: "terraform",
        )
        work_dir = TerraformRunner().prepare(
            _aws_spec(), module_name="aws", state_dir=tmp_path / "terraform"
        )

        assert (work_dir / "main.tf").is_file()
        assert not (work_dir / "manifest.json").exists()
        pin = read_module_pin(work_dir)
        assert pin is not None
        assert pin.name == "arena-cluster-aws"
        assert pin.version == "1.0.0"
        assert "backend.tf" not in pin.files

    def test_reinstall_drops_files_the_new_pin_does_not_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agilerl.arena.byoc.provisioning.terraform.shutil.which",
            lambda _name: "terraform",
        )
        state_dir = tmp_path / "terraform"
        work_dir = TerraformRunner().prepare(
            _aws_spec(), module_name="aws", state_dir=state_dir
        )
        pin = read_module_pin(work_dir)
        assert pin is not None
        write_module_pin(
            work_dir,
            ModulePin(
                name=pin.name,
                version="0.9.0",
                digest=pin.digest,
                orchestration=pin.orchestration,
                files=(*pin.files, "gone.tf"),
            ),
        )
        (work_dir / "gone.tf").write_text("removed\n", encoding="utf-8")

        TerraformRunner(module_version="1.0.0").prepare(
            _aws_spec(), module_name="aws", state_dir=state_dir
        )

        assert not (work_dir / "gone.tf").exists()
        upgraded = read_module_pin(work_dir)
        assert upgraded is not None
        assert upgraded.version == "1.0.0"
        assert upgraded.previous_version == "0.9.0"


class TestModuleCatalog:
    def test_fetches_a_signed_release(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        assert catalog.latest("aws") is not None
        assert catalog.latest("aws").version == "1.0.0"
        fetched = catalog.fetch("aws", "1.0.0")

        check_module(fetched, load_manifest(fetched / "manifest.json"))

    def test_rejects_a_bad_signature(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        signature = tmp_path / "catalog" / "modules" / "aws" / "index.json.sig"
        signature.write_bytes(base64.b64encode(b"x" * 64) + b"\n")
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="signature is invalid"):
            catalog.index("aws")

    def test_unreachable_catalog_uses_the_bundled_module(self, tmp_path: Path) -> None:
        _private, public = _keypair()
        catalog = ModuleCatalog(
            (tmp_path / "missing").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        pin = install_pinned_module(
            MODULES / "aws",
            tmp_path / "terraform",
            requested_version=None,
            catalog=catalog,
        )

        assert pin.version == "1.0.0"


def _valid_digest() -> str:
    return "sha256:" + "ab" * 32


def _manifest_bytes(**overrides: object) -> bytes:
    body: dict[str, object] = {
        "schema": 1,
        "name": "arena-cluster-aws",
        "provider": "aws",
        "version": "1.0.0",
        "orchestration": "aws.v1",
        "terraform_version": ">= 1.10.0",
        "digest": _valid_digest(),
        "files": ["main.tf"],
    }
    body.update(overrides)
    return json.dumps(body).encode()


class TestParseManifest:
    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(click.ClickException, match="is not JSON"):
            parse_manifest(b"{")

    def test_rejects_a_non_object(self) -> None:
        with pytest.raises(click.ClickException, match="JSON object"):
            parse_manifest(b"[]")

    def test_rejects_unsupported_schema(self) -> None:
        with pytest.raises(click.ClickException, match="Unsupported module manifest"):
            parse_manifest(_manifest_bytes(schema=2))

    def test_rejects_a_bad_version(self) -> None:
        with pytest.raises(click.ClickException, match=r"MAJOR\.MINOR\.PATCH"):
            parse_manifest(_manifest_bytes(version="v1"))

    def test_rejects_unknown_orchestration(self) -> None:
        with pytest.raises(click.ClickException, match="not implemented"):
            parse_manifest(_manifest_bytes(orchestration="aws.v9"))

    def test_rejects_bad_files(self) -> None:
        with pytest.raises(click.ClickException, match="list of file names"):
            parse_manifest(_manifest_bytes(files="main.tf"))

    def test_rejects_a_bad_digest(self) -> None:
        with pytest.raises(click.ClickException, match="sha256"):
            parse_manifest(_manifest_bytes(digest="sha256:abc"))


class TestReadModulePin:
    def test_rejects_invalid_json(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "terraform"
        state_dir.mkdir()
        (tmp_path / "module.json").write_text("{", encoding="utf-8")

        with pytest.raises(click.ClickException, match="is not JSON"):
            read_module_pin(state_dir)

    def test_rejects_a_non_object(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "terraform"
        state_dir.mkdir()
        (tmp_path / "module.json").write_text("[]", encoding="utf-8")

        with pytest.raises(click.ClickException, match="JSON object"):
            read_module_pin(state_dir)

    def test_rejects_bad_files_and_previous(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "terraform"
        state_dir.mkdir()
        (tmp_path / "module.json").write_text(
            json.dumps(
                {
                    "name": "arena-cluster-aws",
                    "version": "1.0.0",
                    "digest": _valid_digest(),
                    "orchestration": "aws.v1",
                    "files": "main.tf",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(click.ClickException, match="list of file names"):
            read_module_pin(state_dir)

        (tmp_path / "module.json").write_text(
            json.dumps(
                {
                    "name": "arena-cluster-aws",
                    "version": "1.0.0",
                    "digest": _valid_digest(),
                    "orchestration": "aws.v1",
                    "files": ["main.tf"],
                    "previous_version": 1,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(click.ClickException, match="previous_version"):
            read_module_pin(state_dir)


class TestSelectedModuleVersion:
    def test_rejects_a_bad_flag_version(self, tmp_path: Path) -> None:
        with pytest.raises(click.ClickException, match=r"MAJOR\.MINOR\.PATCH"):
            selected_module_version(tmp_path, None, "v1")

    def test_returns_the_pin_version(self, tmp_path: Path) -> None:
        write_module_pin(
            tmp_path,
            ModulePin(
                name="arena-cluster-aws",
                version="1.0.0",
                digest=_valid_digest(),
                orchestration="aws.v1",
                files=("main.tf",),
            ),
        )

        assert selected_module_version(tmp_path, None, None) == "1.0.0"


class TestCheckModule:
    def test_rejects_provider_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "main.tf").write_text("", encoding="utf-8")
        manifest = parse_manifest(_manifest_bytes(provider="nebius"))

        with pytest.raises(click.ClickException, match="provider is"):
            check_module(tmp_path, manifest)

    def test_rejects_name_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "main.tf").write_text("", encoding="utf-8")
        manifest = parse_manifest(_manifest_bytes(name="other"))

        with pytest.raises(click.ClickException, match="does not match"):
            check_module(tmp_path, manifest)

    def test_rejects_missing_variables_outputs_resources_and_digest(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "main.tf").write_text(
            'variable "cluster_name" {}\n', encoding="utf-8"
        )
        manifest = load_manifest(MODULES / "aws" / "manifest.json")
        files = tuple(name for name in manifest.files if name.endswith(".tf"))
        slim = parse_manifest(
            _manifest_bytes(files=list(files), digest=manifest.digest)
        )
        for name in files:
            (tmp_path / name).write_text(
                'variable "cluster_name" {}\n', encoding="utf-8"
            )

        with pytest.raises(click.ClickException, match="missing variables"):
            check_module(tmp_path, slim)

        required = "\n".join(
            f'variable "{item}" {{ default = null }}' for item in AWS_V1.variables
        )
        (tmp_path / "main.tf").write_text(
            required + '\nvariable "extra" {}\n',
            encoding="utf-8",
        )
        for name in files:
            if name != "main.tf":
                (tmp_path / name).write_text("", encoding="utf-8")
        with pytest.raises(click.ClickException, match="adds required variables"):
            check_module(tmp_path, slim)

        (tmp_path / "main.tf").write_text(
            "\n".join(
                f'variable "{item}" {{ default = null }}' for item in AWS_V1.variables
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(click.ClickException, match="missing outputs"):
            check_module(tmp_path, slim)

        outputs = "\n".join(
            f'output "{item}" {{ value = 1 }}' for item in AWS_V1.required_outputs
        )
        (tmp_path / "main.tf").write_text(
            "\n".join(
                f'variable "{item}" {{ default = null }}' for item in AWS_V1.variables
            )
            + "\n"
            + outputs
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(click.ClickException, match="missing resources"):
            check_module(tmp_path, slim)

        resources = "\n".join(
            f'resource "{kind}" "{name}" {{}}'
            for address in AWS_V1.required_resources
            for kind, name in [address.split(".", 1)]
        )
        (tmp_path / "main.tf").write_text(
            "\n".join(
                f'variable "{item}" {{ default = null }}' for item in AWS_V1.variables
            )
            + "\n"
            + outputs
            + "\n"
            + resources
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(click.ClickException, match="digest is"):
            check_module(tmp_path, slim)


class TestCatalogIndexAndFetch:
    def test_latest_returns_none_when_every_release_is_yanked(
        self, tmp_path: Path
    ) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private, yanked=True)
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        assert catalog.latest("aws") is None

    def test_index_rejects_invalid_json(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        index = tmp_path / "catalog" / "modules" / "aws" / "index.json"
        index.write_bytes(b"{")
        index.with_name("index.json.sig").write_bytes(_sign(private, b"{"))
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="index is not JSON"):
            catalog.index("aws")

    def test_index_rejects_the_wrong_provider(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        payload = json.dumps({"provider": "nebius", "releases": []}).encode()
        index = tmp_path / "catalog" / "modules" / "aws" / "index.json"
        index.write_bytes(payload)
        index.with_name("index.json.sig").write_bytes(_sign(private, payload))
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="is not for aws"):
            catalog.index("aws")

    def test_index_rejects_a_non_list_of_releases(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        payload = json.dumps({"provider": "aws", "releases": {}}).encode()
        index = tmp_path / "catalog" / "modules" / "aws" / "index.json"
        index.write_bytes(payload)
        index.with_name("index.json.sig").write_bytes(_sign(private, payload))
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="must be a list"):
            catalog.index("aws")

    def test_fetch_rejects_a_bad_version(self, tmp_path: Path) -> None:
        _private, public = _keypair()
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match=r"MAJOR\.MINOR\.PATCH"):
            catalog.fetch("aws", "v1")

    def test_fetch_rejects_a_mismatched_manifest_identity(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        manifest_path = (
            tmp_path / "catalog" / "modules" / "aws" / "1.0.0" / "manifest.json"
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["version"] = "9.9.9"
        manifest_bytes = json.dumps(payload).encode()
        manifest_path.write_bytes(manifest_bytes)
        manifest_path.with_name("manifest.json.sig").write_bytes(
            _sign(private, manifest_bytes)
        )
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="identity does not match"):
            catalog.fetch("aws", "1.0.0")

    def test_fetch_rejects_a_manifest_that_disagrees_with_the_index(
        self, tmp_path: Path
    ) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        manifest_path = (
            tmp_path / "catalog" / "modules" / "aws" / "1.0.0" / "manifest.json"
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["digest"] = _valid_digest()
        manifest_bytes = json.dumps(payload).encode()
        manifest_path.write_bytes(manifest_bytes)
        manifest_path.with_name("manifest.json.sig").write_bytes(
            _sign(private, manifest_bytes)
        )
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(
            click.ClickException, match="does not match the catalog index"
        ):
            catalog.fetch("aws", "1.0.0")

    def test_fetch_rejects_a_missing_version(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match=r"has no aws 0\.0\.1"):
            catalog.fetch("aws", "0.0.1")

    def test_fetch_reuses_a_cached_release(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        first = catalog.fetch("aws", "1.0.0")
        second = catalog.fetch("aws", "1.0.0")

        assert first == second

    def test_fetch_replaces_a_partial_cache_dir(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )
        staging = tmp_path / "cache" / "aws" / "1.0.0.partial"
        staging.mkdir(parents=True)
        (staging / "stale.tf").write_text("stale\n", encoding="utf-8")

        fetched = catalog.fetch("aws", "1.0.0")

        assert fetched.is_dir()
        assert not staging.exists()

    def test_fetch_cleans_up_when_check_module_fails(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        archive = tmp_path / "catalog" / "modules" / "aws" / "1.0.0" / "module.tar.gz"
        archive.write_bytes(b"not-a-tar")
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="gzip tar"):
            catalog.fetch("aws", "1.0.0")

    def test_fetch_removes_staging_when_check_module_fails(
        self, tmp_path: Path
    ) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        source = MODULES / "aws"
        manifest = load_manifest(source / "manifest.json")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            for name in manifest.files:
                payload = b'resource "x" {}\n'
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        (
            tmp_path
            / "catalog"
            / "modules"
            / "aws"
            / manifest.version
            / "module.tar.gz"
        ).write_bytes(archive.getvalue())
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="missing variables"):
            catalog.fetch("aws", manifest.version)

        assert not (tmp_path / "cache" / "aws" / f"{manifest.version}.partial").exists()

    def test_fetch_replaces_a_stale_cached_release(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )
        cached = catalog.fetch("aws", "1.0.0")
        (cached / "manifest.json").write_bytes(b'{"stale": true}')

        fetched = catalog.fetch("aws", "1.0.0")
        manifest = json.loads((fetched / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["version"] == "1.0.0"

    def test_signed_rejects_bad_base64(self, tmp_path: Path) -> None:
        private, public = _keypair()
        _publish_aws(tmp_path / "catalog", private)
        sig = tmp_path / "catalog" / "modules" / "aws" / "index.json.sig"
        sig.write_bytes(b"!!!!\n")
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        with pytest.raises(click.ClickException, match="is not base64"):
            catalog.index("aws")

    def test_verify_rejects_a_non_ed25519_key(self, tmp_path: Path) -> None:
        from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        rsa = generate_private_key(public_exponent=65537, key_size=2048)
        public = rsa.public_key().public_bytes(
            Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
        )
        catalog = ModuleCatalog(
            (tmp_path / "catalog").as_uri(),
            public,
            cache_dir=tmp_path / "cache",
        )

        def read(url: str) -> bytes:
            if url.endswith(".sig"):
                return base64.b64encode(b"x" * 64)
            return b"{}"

        catalog.reader = MagicMock(read=read)

        with pytest.raises(click.ClickException, match="must be Ed25519"):
            catalog._signed("modules/aws/index.json")


class TestHttpCatalogReader:
    def test_rejects_unsupported_schemes(self) -> None:
        with pytest.raises(
            click.ClickException, match="Unsupported module catalog URL"
        ):
            HttpCatalogReader().read("ftp://example.com/index.json")

    def test_http_errors_are_unavailable(self) -> None:
        with patch(
            "agilerl.arena.byoc.provisioning.module_catalog.httpx.get",
            side_effect=httpx.ConnectError("down"),
        ):
            with pytest.raises(CatalogUnavailable, match="request failed"):
                HttpCatalogReader().read("https://example.com/index.json")

    def test_http_non_200(self) -> None:
        response = MagicMock(status_code=404, content=b"missing")
        with patch(
            "agilerl.arena.byoc.provisioning.module_catalog.httpx.get",
            return_value=response,
        ):
            with pytest.raises(click.ClickException, match="returned 404"):
                HttpCatalogReader().read("https://example.com/index.json")

    def test_http_success(self) -> None:
        response = MagicMock(status_code=200, content=b"ok")
        with patch(
            "agilerl.arena.byoc.provisioning.module_catalog.httpx.get",
            return_value=response,
        ):
            assert HttpCatalogReader().read("https://example.com/index.json") == b"ok"


class TestOpenModuleCatalog:
    def test_returns_none_without_a_url(self) -> None:
        assert open_module_catalog(None, None) is None
        assert open_module_catalog("  ", "key.pem") is None

    def test_requires_a_public_key(self) -> None:
        with pytest.raises(
            click.ClickException, match="Pass --module-catalog-public-key"
        ):
            open_module_catalog("https://example.com", None)

    def test_requires_a_key_file(self, tmp_path: Path) -> None:
        with pytest.raises(click.ClickException, match="is not a file"):
            open_module_catalog("https://example.com", str(tmp_path / "missing.pem"))

    def test_opens_a_catalog(self, tmp_path: Path) -> None:
        _private, public = _keypair()
        key = tmp_path / "key.pem"
        key.write_bytes(public)

        catalog = open_module_catalog("https://example.com/catalog", str(key))

        assert catalog is not None
        assert catalog.base_url == "https://example.com/catalog"


class TestResolveSource:
    def test_keeps_the_bundled_module_when_catalog_matches(self) -> None:
        bundled = load_manifest(MODULES / "aws" / "manifest.json")
        catalog = MagicMock()
        catalog.latest.return_value = MagicMock(version=bundled.version)

        source, manifest = _resolve_source(MODULES / "aws", bundled, None, catalog)

        assert source == MODULES / "aws"
        assert manifest is bundled
        catalog.fetch.assert_not_called()

    def test_uses_a_newer_catalog_release(self, tmp_path: Path) -> None:
        bundled = load_manifest(MODULES / "aws" / "manifest.json")
        fetched = tmp_path / "fetched"
        fetched.mkdir()
        (fetched / "manifest.json").write_bytes(
            (MODULES / "aws" / "manifest.json").read_bytes()
        )
        catalog = MagicMock()
        catalog.latest.return_value = MagicMock(version="9.9.9")
        catalog.fetch.return_value = fetched

        source, manifest = _resolve_source(MODULES / "aws", bundled, None, catalog)

        assert source == fetched
        assert manifest.version == bundled.version

    def test_requires_a_catalog_for_an_unbundled_version(self) -> None:
        bundled = load_manifest(MODULES / "aws" / "manifest.json")

        with pytest.raises(click.ClickException, match="is not bundled"):
            _resolve_source(MODULES / "aws", bundled, "9.9.9", None)

    def test_fetches_a_requested_version(self, tmp_path: Path) -> None:
        bundled = load_manifest(MODULES / "aws" / "manifest.json")
        fetched = tmp_path / "fetched"
        fetched.mkdir()
        (fetched / "manifest.json").write_bytes(
            (MODULES / "aws" / "manifest.json").read_bytes()
        )
        catalog = MagicMock()
        catalog.fetch.return_value = fetched

        source, manifest = _resolve_source(MODULES / "aws", bundled, "9.9.9", catalog)

        catalog.fetch.assert_called_once_with("aws", "9.9.9")
        assert source == fetched
        assert manifest.version == bundled.version


class TestArchiveHelpers:
    def test_previous_version_keeps_the_older_pin(self) -> None:
        pin = ModulePin(
            name="arena-cluster-aws",
            version="1.0.0",
            digest=_valid_digest(),
            orchestration="aws.v1",
            files=("main.tf",),
            previous_version="0.9.0",
        )

        assert _previous_version(pin, "1.0.0") == "0.9.0"
        assert _previous_version(pin, "1.1.0") == "1.0.0"

    def test_extract_rejects_a_bad_archive(self, tmp_path: Path) -> None:
        with pytest.raises(click.ClickException, match="gzip tar"):
            _extract_module(b"nope", tmp_path, ("main.tf",))

    def test_extract_rejects_directories_and_unexpected_files(
        self, tmp_path: Path
    ) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="subdir")
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        with pytest.raises(click.ClickException, match="is not a file"):
            _extract_module(archive.getvalue(), tmp_path, ("main.tf",))

        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            payload = b"x"
            info = tarfile.TarInfo(name="extra.tf")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        with pytest.raises(click.ClickException, match="unexpected file"):
            _extract_module(archive.getvalue(), tmp_path, ("main.tf",))

    def test_extract_rejects_an_empty_file_entry(self, tmp_path: Path) -> None:
        member = tarfile.TarInfo(name="main.tf")
        member.type = tarfile.REGTYPE
        tar = MagicMock()
        tar.__enter__.return_value = tar
        tar.__exit__.return_value = False
        tar.getmembers.return_value = [member]
        tar.extractfile.return_value = None

        with patch(
            "agilerl.arena.byoc.provisioning.module_catalog.tarfile.open",
            return_value=tar,
        ):
            with pytest.raises(click.ClickException, match="is empty"):
                _extract_module(b"archive", tmp_path, ("main.tf",))

    def test_extract_rejects_a_missing_file(self, tmp_path: Path) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            payload = b"x"
            info = tarfile.TarInfo(name="main.tf")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        with pytest.raises(click.ClickException, match="is missing"):
            _extract_module(archive.getvalue(), tmp_path, ("main.tf", "variables.tf"))

    def test_archive_name_allows_a_dot_prefix(self) -> None:
        assert _archive_name("./main.tf") == "main.tf"

    def test_archive_name_rejects_paths(self) -> None:
        with pytest.raises(click.ClickException, match="escapes the archive"):
            _archive_name("../main.tf")
        with pytest.raises(click.ClickException, match="is not a module file"):
            _archive_name("nested/main.tf")

    def test_parse_release_errors(self) -> None:
        with pytest.raises(click.ClickException, match="must be an object"):
            _parse_release("nope")
        with pytest.raises(click.ClickException, match=r"MAJOR\.MINOR\.PATCH"):
            _parse_release(
                {
                    "version": "v1",
                    "orchestration": "aws.v1",
                    "digest": _valid_digest(),
                }
            )
        with pytest.raises(click.ClickException, match="yanked"):
            _parse_release(
                {
                    "version": "1.0.0",
                    "orchestration": "aws.v1",
                    "digest": _valid_digest(),
                    "yanked": "yes",
                }
            )

    def test_require_text_and_safe_file_name(self) -> None:
        with pytest.raises(click.ClickException, match="must be a string"):
            _require_text({"version": 1}, "version")
        with pytest.raises(click.ClickException, match="single path segment"):
            _safe_file_name("nested/main.tf")
        with pytest.raises(click.ClickException, match="single path segment"):
            _safe_file_name("..")
