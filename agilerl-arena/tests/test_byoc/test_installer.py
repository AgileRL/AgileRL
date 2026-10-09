# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Helm installer and the functional facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click import ClickException

from agilerl.arena.byoc import ByocApi
from agilerl.arena.byoc.installer import (
    HelmInstaller,
    build_installer,
    normalize_setup_type,
    run_byoc_down,
    run_byoc_install,
    run_byoc_teardown,
)
from agilerl.arena.byoc.scripts import BundleScriptRunner, StageFailed
from agilerl.arena.client import ArenaClient


@pytest.fixture
def api() -> ByocApi:
    return ByocApi(MagicMock(spec=ArenaClient))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("helm", "helm"),
        ("HELM", "helm"),
    ],
)
def test_normalize_setup_type_maps_helm(raw: str, expected: str) -> None:
    assert normalize_setup_type(raw) == expected


@pytest.mark.parametrize("raw", ["dockerSwarm", "kubernetes", "nomad"])
def test_normalize_setup_type_rejects_unknown(raw: str) -> None:
    with pytest.raises(ClickException, match="Unsupported setup type"):
        normalize_setup_type(raw)


class TestHelmInstaller:
    def test_install_cluster_runs_setup(self, api: ByocApi, helm_bundle: Path) -> None:
        inst = HelmInstaller(api, name="pool")
        with (
            patch(
                "agilerl.arena.byoc.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            patch.object(BundleScriptRunner, "run") as run_mock,
        ):
            inst.install_cluster(helm_bundle)
        run_mock.assert_called_once_with("setup.sh", [])

    def test_install_cluster_requires_setup_script(
        self, api: ByocApi, tmp_path: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with pytest.raises(ClickException, match=r"no setup\.sh"):
            inst.install_cluster(tmp_path)

    def test_install_cluster_requires_helm_on_path(
        self, api: ByocApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with (
            patch("agilerl.arena.byoc.installer.shutil.which", return_value=None),
            pytest.raises(ClickException, match="helm not found"),
        ):
            inst.install_cluster(helm_bundle)

    def test_verify_runs_validate_script(self, api: ByocApi, helm_bundle: Path) -> None:
        (helm_bundle / "validate.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        inst = HelmInstaller(api, name="pool")
        with patch.object(BundleScriptRunner, "run") as run_mock:
            inst.verify(helm_bundle)
        run_mock.assert_called_once_with("validate.sh", [])

    def test_verify_warns_without_validate_script(
        self, api: ByocApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with patch("agilerl.arena.byoc.installer.logger") as log:
            inst.verify(helm_bundle)
        log.warning.assert_called_once()

    def test_helm_uninstall_invokes_cli(self) -> None:
        completed = MagicMock(returncode=0)
        with (
            patch(
                "agilerl.arena.byoc.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            patch(
                "agilerl.arena.byoc.installer.subprocess.run",
                return_value=completed,
            ) as run_mock,
        ):
            HelmInstaller._helm_uninstall("rel", "ns")
        assert run_mock.call_args.args[0] == [
            "helm",
            "uninstall",
            "rel",
            "--namespace",
            "ns",
        ]

    def test_helm_uninstall_tolerates_nonzero_exit(self) -> None:
        completed = MagicMock(returncode=1)
        with (
            patch(
                "agilerl.arena.byoc.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            patch(
                "agilerl.arena.byoc.installer.subprocess.run",
                return_value=completed,
            ),
            patch("agilerl.arena.byoc.installer.logger") as log,
        ):
            HelmInstaller._helm_uninstall("rel", "ns")
        log.warning.assert_called_once()
        assert log.warning.call_args.args[1] == 1

    def test_helm_uninstall_requires_helm_on_path(self) -> None:
        with (
            patch("agilerl.arena.byoc.installer.shutil.which", return_value=None),
            pytest.raises(ClickException, match="helm not found"),
        ):
            HelmInstaller._helm_uninstall("rel", "ns")


def test_build_installer_returns_helm(api: ByocApi) -> None:
    assert isinstance(build_installer(api, name="p"), HelmInstaller)


class TestRunByocInstall:
    def test_helm_flow_uses_name_in_bundle_query(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            {},  # enable
            [{"name": "k8s-pool", "num_nodes": 3}],  # find_class
            (b"zip", "application/zip", None),  # fetch_bundle
        ]
        with (
            patch(
                "agilerl.arena.byoc.installer.extract_bundle",
                return_value=Path("/tmp/fake"),
            ),
            patch("agilerl.arena.byoc.installer.validate_wireguard_bundle"),
            patch.object(HelmInstaller, "install_cluster") as install_mock,
            patch.object(HelmInstaller, "verify"),
        ):
            run_byoc_install(
                client, name="k8s-pool", setup_type="helm", skip_enable=False
            )
        install_mock.assert_called_once()
        bundle_call = client._invoke_manifest_command.call_args_list[2]
        assert bundle_call.args[1] == {
            "name": "k8s-pool",
            "setupType": "helm",
            "archivedType": "zip",
        }

    def test_install_fails_when_class_missing(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            {},  # enable
            [],  # find_class
        ]
        with (
            patch(
                "agilerl.arena.byoc.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            pytest.raises(ClickException, match="No BYOC resource class"),
        ):
            run_byoc_install(
                client, name="missing-pool", setup_type="helm", skip_enable=False
            )
        assert client._invoke_manifest_command.call_count == 2


class TestRunByocTeardown:
    def test_helm_uninstalls_without_deleting_class(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            (b"zip", "application/zip", None),  # fetch_bundle (teardown_cluster)
        ]
        with (
            patch(
                "agilerl.arena.byoc.installer.extract_bundle",
                return_value=Path("/tmp/fake"),
            ),
            patch(
                "agilerl.arena.byoc.installer.parse_helm_release_ids",
                return_value=("k8s-pool", "k8s-pool"),
            ),
            patch.object(HelmInstaller, "_helm_uninstall") as helm_mock,
        ):
            run_byoc_teardown(
                client,
                name="k8s-pool",
                setup_type="helm",
                skip_cluster=False,
                disable_provider=False,
            )
        helm_mock.assert_called_once_with("k8s-pool", "k8s-pool")
        delete_calls = [
            c
            for c in client._invoke_manifest_command.call_args_list
            if c.args[0]["path"].endswith("/classes/delete")
        ]
        assert delete_calls == []


class TestByocInstallerDown:
    def test_down_delegates_to_down_cluster(self, api: ByocApi) -> None:
        inst = HelmInstaller(api, name="pool")
        with (
            patch.object(inst, "down_cluster") as down_cluster,
            patch("agilerl.arena.byoc.installer.logger") as log,
        ):
            inst.down()
        down_cluster.assert_called_once_with()
        log.info.assert_called()


class TestHelmInstallerExtraPaths:
    def test_install_cluster_wraps_stage_failure(
        self, api: ByocApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with (
            patch(
                "agilerl.arena.byoc.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            patch.object(
                BundleScriptRunner,
                "run",
                side_effect=StageFailed(
                    "setup.sh", 1, "setup failed: cluster unreachable"
                ),
            ),
        ):
            with pytest.raises(ClickException):
                inst.install_cluster(helm_bundle)

    def test_verify_wraps_stage_failure(self, api: ByocApi, helm_bundle: Path) -> None:
        (helm_bundle / "validate.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        inst = HelmInstaller(api, name="pool")
        with patch.object(
            BundleScriptRunner,
            "run",
            side_effect=StageFailed(
                "validate.sh", 1, "validation failed: resources missing"
            ),
        ):
            with pytest.raises(ClickException):
                inst.verify(helm_bundle)

    def test_down_cluster_requires_kubectl(self, api: ByocApi) -> None:
        inst = HelmInstaller(api, name="pool")
        with (
            patch("agilerl.arena.byoc.installer.shutil.which", return_value=None),
            pytest.raises(ClickException, match="kubectl not found"),
        ):
            inst.down_cluster()

    def test_down_cluster_scales_release_to_zero(
        self, api: ByocApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        completed = MagicMock(returncode=0)
        with (
            patch(
                "agilerl.arena.byoc.installer.shutil.which",
                return_value="/usr/bin/kubectl",
            ),
            patch.object(inst.api, "fetch_bundle", return_value=b"zip-bytes"),
            patch(
                "agilerl.arena.byoc.installer.extract_bundle",
                return_value=helm_bundle,
            ),
            patch(
                "agilerl.arena.byoc.installer.parse_helm_release_ids",
                return_value=("rel", "ns"),
            ),
            patch(
                "agilerl.arena.byoc.installer.subprocess.run",
                return_value=completed,
            ) as run_mock,
        ):
            inst.down_cluster()
        assert run_mock.call_args.args[0][:3] == ["kubectl", "scale", "deployment"]

    def test_down_cluster_warns_on_nonzero_scale(
        self, api: ByocApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        completed = MagicMock(returncode=2)
        with (
            patch(
                "agilerl.arena.byoc.installer.shutil.which",
                return_value="/usr/bin/kubectl",
            ),
            patch.object(inst.api, "fetch_bundle", return_value=b"zip-bytes"),
            patch(
                "agilerl.arena.byoc.installer.extract_bundle",
                return_value=helm_bundle,
            ),
            patch(
                "agilerl.arena.byoc.installer.parse_helm_release_ids",
                return_value=("rel", "ns"),
            ),
            patch(
                "agilerl.arena.byoc.installer.subprocess.run",
                return_value=completed,
            ),
            patch("agilerl.arena.byoc.installer.logger") as log,
        ):
            inst.down_cluster()
        log.warning.assert_called_once()


class TestRunByocDown:
    def test_builds_installer_and_calls_down(self) -> None:
        client = MagicMock(spec=ArenaClient)
        fake_installer = MagicMock()
        with patch(
            "agilerl.arena.byoc.installer.build_installer",
            return_value=fake_installer,
        ) as build:
            run_byoc_down(client, name="pool", setup_type="helm")
        build.assert_called_once()
        fake_installer.down.assert_called_once_with()
