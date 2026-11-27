# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for choosing and checking the Kubernetes cluster a BYOC command uses."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
import yaml

from agilerl.byoc.api import ByocApi
from agilerl.byoc.kube_target import (
    KubeTarget,
    NoKubeContextError,
    kube_target,
    require_registered_cluster,
)

from .conftest import FAKE_CLUSTER_UID

MODULE = "agilerl.byoc.kube_target"
MINIFIED = {
    "apiVersion": "v1",
    "kind": "Config",
    "clusters": [{"name": "b", "cluster": {"server": "https://b.example:6443"}}],
    "contexts": [{"name": "ctx-b", "context": {"cluster": "b", "user": "u"}}],
    "users": [{"name": "u", "user": {"token": "x"}}],
    "current-context": "ctx-b",
}


class FakeKubectl:
    """Answers ``kubectl config view`` and ``get namespace kube-system``."""

    def __init__(
        self,
        view_returncode: int = 0,
        uid_result: tuple[int, str, str] = (0, FAKE_CLUSTER_UID, ""),
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.view_returncode = view_returncode
        self.uid_result = uid_result
        self.pinned_at_uid_read: dict[str, object] = {}

    def __call__(self, argv: list[str], **kwargs: object) -> MagicMock:
        env = kwargs["env"]
        assert isinstance(env, dict)
        self.calls.append((argv, env))
        if argv[1:3] == ["config", "view"]:
            if self.view_returncode != 0:
                return MagicMock(
                    returncode=1,
                    stdout="",
                    stderr="error: current-context is not set",
                )
            return MagicMock(returncode=0, stdout=yaml.safe_dump(MINIFIED), stderr="")
        pinned = Path(env["KUBECONFIG"])
        self.pinned_at_uid_read = {
            "config": yaml.safe_load(pinned.read_text(encoding="utf-8")),
            "mode": stat.S_IMODE(pinned.stat().st_mode),
        }
        returncode, stdout, stderr = self.uid_result
        return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def kubectl(monkeypatch: pytest.MonkeyPatch) -> FakeKubectl:
    fake = FakeKubectl()
    monkeypatch.setattr(f"{MODULE}.subprocess.run", fake)
    monkeypatch.setattr(f"{MODULE}.require_kubectl", lambda: None)
    return fake


def _set_tty(monkeypatch: pytest.MonkeyPatch, tty: bool) -> None:
    monkeypatch.setattr(f"{MODULE}.sys.stdin", MagicMock(isatty=lambda: tty))


class TestKubeTarget:
    def test_named_context_is_pinned_without_asking(
        self, kubectl: FakeKubectl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _set_tty(monkeypatch, False)
        source = tmp_path / "kubeconfig"
        source.write_text("apiVersion: v1\n", encoding="utf-8")

        with kube_target("corp-k8s", source, "ctx-b") as target:
            pinned = target.kubeconfig
            assert pinned.is_file()

        view_argv, view_env = kubectl.calls[0]
        assert view_argv[view_argv.index("--context") + 1] == "ctx-b"
        assert view_env["KUBECONFIG"] == str(source.resolve())
        assert target == KubeTarget(
            kubeconfig=pinned,
            context="ctx-b",
            server="https://b.example:6443",
            cluster_uid=FAKE_CLUSTER_UID,
        )
        assert kubectl.pinned_at_uid_read["config"] == MINIFIED
        if sys.platform != "win32":
            assert kubectl.pinned_at_uid_read["mode"] == 0o600
        assert not pinned.exists()

    def test_provisioned_kubeconfig_needs_no_flag(
        self, kubectl: FakeKubectl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _set_tty(monkeypatch, False)
        provisioned = tmp_path / "arena-cluster-corp-k8s" / "kubeconfig"
        provisioned.parent.mkdir()
        provisioned.write_text("apiVersion: v1\n", encoding="utf-8")

        with kube_target("corp-k8s", None, None) as target:
            assert target.context == "ctx-b"

        assert kubectl.calls[0][1]["KUBECONFIG"] == str(provisioned.resolve())
        assert "--context" not in kubectl.calls[0][0]

    def test_unnamed_cluster_uses_kubectls_default_and_logs_to_stderr(
        self,
        kubectl: FakeKubectl,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KUBECONFIG", str(tmp_path / "default-kubeconfig"))
        _set_tty(monkeypatch, False)

        with kube_target(None, None, "ctx-b") as target:
            assert target.context == "ctx-b"

        assert kubectl.calls[0][1]["KUBECONFIG"] == str(tmp_path / "default-kubeconfig")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Using Kubernetes context 'ctx-b'" in captured.err

    def test_current_context_needs_a_flag_when_nobody_can_answer(
        self, kubectl: FakeKubectl, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _set_tty(monkeypatch, False)

        with pytest.raises(click.ClickException, match="Pass --context ctx-b"):
            with kube_target("corp-k8s", None, None):
                pass

        assert len(kubectl.calls) == 1

    @pytest.mark.parametrize("answer", [True, False])
    def test_current_context_is_used_only_after_the_user_agrees(
        self,
        kubectl: FakeKubectl,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        answer: bool,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("KUBECONFIG", str(tmp_path / "env-kubeconfig"))
        _set_tty(monkeypatch, True)
        asked: list[str] = []

        def confirm(text: str, **_kwargs: object) -> bool:
            asked.append(text)
            if not answer:
                raise click.Abort
            return True

        with patch(f"{MODULE}.click.confirm", side_effect=confirm):
            if answer:
                with kube_target("corp-k8s", None, None) as target:
                    assert target.cluster_uid == FAKE_CLUSTER_UID
            else:
                with pytest.raises(click.Abort):
                    with kube_target("corp-k8s", None, None):
                        pass

        expected = (
            "Use the current Kubernetes context 'ctx-b' "
            "(API server https://b.example:6443)?"
        )
        assert asked == [expected]
        assert kubectl.calls[0][1]["KUBECONFIG"] == os.environ["KUBECONFIG"]
        assert len(kubectl.calls) == (2 if answer else 1)

    def test_no_context_at_all_is_its_own_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(f"{MODULE}.subprocess.run", FakeKubectl(view_returncode=1))
        monkeypatch.setattr(f"{MODULE}.require_kubectl", lambda: None)

        with pytest.raises(NoKubeContextError, match="current-context is not set"):
            with kube_target("corp-k8s", None, None):
                pass

    @pytest.mark.parametrize(
        ("uid_result", "detail"),
        [
            (
                (1, "", "Error from server (Forbidden): namespaces is forbidden\n"),
                "Error from server (Forbidden): namespaces is forbidden",
            ),
            ((0, "\n", ""), ""),
        ],
        ids=["kubectl-fails", "empty-uid"],
    )
    def test_unreadable_cluster_uid_is_an_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        uid_result: tuple[int, str, str],
        detail: str,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            f"{MODULE}.subprocess.run", FakeKubectl(uid_result=uid_result)
        )
        monkeypatch.setattr(f"{MODULE}.require_kubectl", lambda: None)
        source = tmp_path / "kubeconfig"
        source.write_text("apiVersion: v1\n", encoding="utf-8")

        with pytest.raises(click.ClickException) as raised:
            with kube_target("corp-k8s", source, "ctx-b"):
                pass

        assert raised.value.message == (
            f"Could not read the kube-system namespace UID: {detail}"
        )


class TestRequireRegisteredCluster:
    TARGET = KubeTarget(
        kubeconfig=Path("pinned"),
        context="ctx-b",
        server="https://b.example:6443",
        cluster_uid=FAKE_CLUSTER_UID,
    )

    def _check(self, cluster: dict[str, object] | None) -> None:
        with patch.object(ByocApi, "find_cluster", return_value=cluster):
            require_registered_cluster(ByocApi(MagicMock()), "corp-k8s", self.TARGET)

    def test_accepts_the_stored_uid(self) -> None:
        self._check({"name": "corp-k8s", "clusterUid": FAKE_CLUSTER_UID})

    def test_says_when_arena_has_no_uid_yet(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._check({"name": "corp-k8s"})

        assert "Arena has no cluster UID for 'corp-k8s' yet" in capsys.readouterr().out

    def test_refuses_another_cluster(self) -> None:
        with pytest.raises(
            click.ClickException,
            match="is not Arena cluster 'corp-k8s', which runs in cluster UID other",
        ):
            self._check({"name": "corp-k8s", "clusterUid": "other"})

    def test_refuses_an_unregistered_name(self) -> None:
        with pytest.raises(click.ClickException, match="No registered BYOC cluster"):
            self._check(None)
