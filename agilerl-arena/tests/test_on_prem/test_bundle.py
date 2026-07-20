"""Tests for bundle extraction, WireGuard validation, and Helm id parsing."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from click import ClickException

from agilerl.arena.on_prem.bundle import (
    extract_bundle,
    parse_helm_release_ids,
    resolve_bundle_root,
    validate_wireguard_bundle,
)


class TestExtractBundle:
    def test_extracts_and_marks_scripts_executable(
        self, tmp_path: Path, make_zip: Callable[[dict[str, str]], bytes]
    ) -> None:
        data = make_zip(
            {
                "arena-train/setup.sh": "#!/bin/sh\necho hi\n",
                "arena-train/chart/values.yaml": "clusterName: pool\n",
            }
        )
        root = extract_bundle(data, tmp_path, class_name="pool")
        assert (root / "setup.sh").is_file()
        assert os.access(root / "setup.sh", os.X_OK)


class TestResolveBundleRoot:
    def test_finds_setup_sh_with_arena_train_prefix(self, tmp_path: Path) -> None:
        nested = tmp_path / "extracted" / "arena-train"
        nested.mkdir(parents=True)
        (nested / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        assert resolve_bundle_root(tmp_path / "extracted") == nested

    def test_finds_setup_sh_at_extract_root(self, tmp_path: Path) -> None:
        root = tmp_path / "extracted"
        root.mkdir()
        (root / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        assert resolve_bundle_root(root) == root

    def test_finds_nested_setup_sh_via_rglob_with_chart(self, tmp_path: Path) -> None:
        # setup.sh is neither at <root>/arena-train nor <root>; rglob finds it
        # nested, and the sibling chart/ dir identifies it as the bundle root.
        root = tmp_path / "extracted"
        nested = root / "deep" / "pkg"
        (nested / "chart").mkdir(parents=True)
        (nested / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        assert resolve_bundle_root(root) == nested

    def test_finds_nested_setup_sh_via_rglob_with_install_docker(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "extracted"
        nested = root / "deep" / "pkg"
        nested.mkdir(parents=True)
        (nested / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        (nested / "install-docker.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        assert resolve_bundle_root(root) == nested

    def test_falls_back_to_arena_train_when_unresolvable(self, tmp_path: Path) -> None:
        # No setup.sh anywhere -> rglob yields nothing -> deterministic fallback.
        root = tmp_path / "extracted"
        root.mkdir()
        assert resolve_bundle_root(root) == root / "arena-train"


class TestValidateWireguardBundle:
    def test_accepts_valid_swarm_bundle(self, swarm_bundle: Path) -> None:
        validate_wireguard_bundle(swarm_bundle, "dockerSwarm")

    def test_accepts_valid_helm_bundle(self, helm_bundle: Path) -> None:
        validate_wireguard_bundle(helm_bundle, "helm")

    def test_rejects_incomplete_tun0(self, swarm_bundle: Path) -> None:
        (swarm_bundle / "config.d" / "tun0.conf").write_text(
            "[Interface]\n", encoding="utf-8"
        )
        with pytest.raises(ClickException, match=r"Invalid tun0\.conf"):
            validate_wireguard_bundle(swarm_bundle, "dockerSwarm")

    def test_rejects_stack_without_wireguard_mount(self, swarm_bundle: Path) -> None:
        (swarm_bundle / "arena-stack.yaml").write_text(
            "volumes: []\n", encoding="utf-8"
        )
        with pytest.raises(ClickException, match="does not mount WireGuard"):
            validate_wireguard_bundle(swarm_bundle, "dockerSwarm")

    def test_rejects_helm_values_missing_wireguard_key(self, helm_bundle: Path) -> None:
        (helm_bundle / "chart" / "values.yaml").write_text(
            "clusterName: pool\n", encoding="utf-8"
        )
        with pytest.raises(ClickException, match="WireGuard not rendered"):
            validate_wireguard_bundle(helm_bundle, "helm")

    def test_rejects_helm_bundle_missing_values_yaml(self, helm_bundle: Path) -> None:
        (helm_bundle / "chart" / "values.yaml").unlink()
        with pytest.raises(ClickException, match=r"missing chart/values\.yaml"):
            validate_wireguard_bundle(helm_bundle, "helm")

    def test_rejects_swarm_bundle_missing_tun0(self, swarm_bundle: Path) -> None:
        (swarm_bundle / "config.d" / "tun0.conf").unlink()
        with pytest.raises(ClickException, match="missing WireGuard config"):
            validate_wireguard_bundle(swarm_bundle, "dockerSwarm")

    def test_rejects_swarm_bundle_missing_stack_yaml(self, swarm_bundle: Path) -> None:
        (swarm_bundle / "arena-stack.yaml").unlink()
        with pytest.raises(ClickException, match=r"missing arena-stack\.yaml"):
            validate_wireguard_bundle(swarm_bundle, "dockerSwarm")


class TestParseHelmReleaseIds:
    def test_parses_cluster_name(self, tmp_path: Path) -> None:
        root = tmp_path / "bundle"
        (root / "chart").mkdir(parents=True)
        (root / "chart" / "values.yaml").write_text(
            'clusterName: "my-k3d-pool"\n', encoding="utf-8"
        )
        assert parse_helm_release_ids(root) == ("my-k3d-pool", "my-k3d-pool")

    def test_requires_values_file(self, tmp_path: Path) -> None:
        with pytest.raises(ClickException, match="cannot determine release name"):
            parse_helm_release_ids(tmp_path)
