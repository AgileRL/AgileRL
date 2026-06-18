"""Tests for the provider installers and the functional facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click import ClickException

from agilerl.arena.client import ArenaClient
from agilerl.arena.on_prem import OnPremApi
from agilerl.arena.on_prem.installer import (
    HelmInstaller,
    SwarmInstaller,
    all_hosts,
    build_installer,
    normalize_setup_type,
    report_stack_readiness,
    resolve_stack_name,
    run_on_prem_install,
    run_on_prem_teardown,
    stack_readiness_state,
    warn_ignored_swarm_flags,
)
from agilerl.arena.on_prem.scripts import BundleScriptRunner, StageFailed
from agilerl.arena.on_prem.ssh import SshExecutor


@pytest.fixture
def api() -> OnPremApi:
    return OnPremApi(MagicMock(spec=ArenaClient))


# --------------------------------------------------------------------------- #
# Module-level helpers                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dockerSwarm", "dockerSwarm"),
        ("docker-swarm", "dockerSwarm"),
        ("DOCKERSWARM", "dockerSwarm"),
        ("kubernetes", "dockerSwarm"),
        ("helm", "helm"),
        ("HELM", "helm"),
    ],
)
def test_normalize_setup_type_maps_known_aliases(raw: str, expected: str) -> None:
    assert normalize_setup_type(raw) == expected


def test_normalize_setup_type_rejects_unknown() -> None:
    with pytest.raises(ClickException, match="Unsupported setup type"):
        normalize_setup_type("nomad")


def test_resolve_stack_name_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARENA_STACK_NAME", raising=False)
    assert resolve_stack_name("arena") == "arena"
    monkeypatch.setenv("ARENA_STACK_NAME", "custom")
    assert resolve_stack_name("arena") == "custom"


@pytest.mark.parametrize(
    ("manager", "workers", "expected"),
    [
        ("m", ("w1", "w2"), ["m", "w1", "w2"]),
        ("m", ("m", "w1"), ["m", "w1"]),  # manager not duplicated
        (" m ", (" ", "w1", "w1"), ["m", "w1"]),  # strip + dedup + drop blanks
        ("m", (), ["m"]),
    ],
)
def test_all_hosts_dedups_and_preserves_order(
    manager: str, workers: tuple[str, ...], expected: list[str]
) -> None:
    assert all_hosts(manager, workers) == expected


def test_warn_ignored_swarm_flags_lists_set_flags() -> None:
    with patch("agilerl.arena.on_prem.installer.logger") as log:
        warn_ignored_swarm_flags(
            manager="m",
            workers=("w1",),
            ssh_user="ubuntu",
            ssh_extra_opts=None,
            advertise_addr=None,
        )
    log.warning.assert_called_once()
    flags = log.warning.call_args.args[1]
    assert "--manager" in flags
    assert "--workers" in flags
    assert "--ssh-user" in flags
    assert "--ssh-extra-opts" not in flags


def test_warn_ignored_swarm_flags_silent_when_none() -> None:
    with patch("agilerl.arena.on_prem.installer.logger") as log:
        warn_ignored_swarm_flags(
            manager=None,
            workers=(),
            ssh_user=None,
            ssh_extra_opts=None,
            advertise_addr=None,
        )
    log.warning.assert_not_called()


def test_stack_readiness_state_detects_scheduling_errors() -> None:
    ready, not_ready, scheduling_errors = stack_readiness_state(
        "arena_ray-worker\t0/1",
        service_ps_output="x  Pending  no suitable node (insufficient resources on 1 node)",
    )
    assert ready is False
    assert not_ready == ["arena_ray-worker 0/1"]
    assert scheduling_errors


def test_report_stack_readiness_warns_on_scheduling_errors() -> None:
    with patch("agilerl.arena.on_prem.installer.logger") as log:
        report_stack_readiness(
            "arena",
            "arena_ray-worker\t0/1",
            service_ps_output="x  Pending  no suitable node (insufficient resources on 1 node)",
        )
    assert any("scheduling issue" in str(c) for c in log.warning.call_args_list)


def test_report_stack_readiness_warns_on_partial() -> None:
    with patch("agilerl.arena.on_prem.installer.logger") as log:
        report_stack_readiness("arena", "arena_ray-head\t0/1\narena_ray-worker\t1/1")
    log.warning.assert_called_once()
    assert "arena_ray-head 0/1" in log.warning.call_args.args[2]


def test_report_stack_readiness_ok_when_all_running() -> None:
    with patch("agilerl.arena.on_prem.installer.logger") as log:
        report_stack_readiness("arena", "arena_ray-head\t1/1\narena_ray-worker\t2/2")
    log.warning.assert_not_called()
    log.info.assert_called_once()


def test_report_stack_readiness_warns_when_no_output() -> None:
    with patch("agilerl.arena.on_prem.installer.logger") as log:
        report_stack_readiness("arena", None)
    log.warning.assert_called_once()


# --------------------------------------------------------------------------- #
# SwarmInstaller                                                               #
# --------------------------------------------------------------------------- #


class TestSwarmInstaller:
    def test_install_cluster_strips_ssh_port_from_advertise_addr(
        self, api: OnPremApi
    ) -> None:
        inst = SwarmInstaller(api, name="pool", manager="127.0.0.1:5043")
        captured: dict[str, str] = {}

        class _CapturingRunner:
            def __init__(
                self, bundle_root: Path, env: dict[str, str] | None = None
            ) -> None:
                captured.update(env or {})

            def run(self, script: str, args: list[str]) -> None:
                return None

        with patch(
            "agilerl.arena.on_prem.installer.BundleScriptRunner", _CapturingRunner
        ):
            inst.install_cluster(Path("/tmp/bundle"))
        assert captured["SWARM_ADVERTISE_ADDR"] == "127.0.0.1"

    def test_install_cluster_strips_ssh_port_from_explicit_advertise_addr(
        self, api: OnPremApi
    ) -> None:
        inst = SwarmInstaller(
            api, name="pool", manager="localhost", advertise_addr="127.0.0.1:5043"
        )
        captured: dict[str, str] = {}

        class _CapturingRunner:
            def __init__(
                self, bundle_root: Path, env: dict[str, str] | None = None
            ) -> None:
                captured.update(env or {})

            def run(self, script: str, args: list[str]) -> None:
                return None

        with patch(
            "agilerl.arena.on_prem.installer.BundleScriptRunner", _CapturingRunner
        ):
            inst.install_cluster(Path("/tmp/bundle"))
        assert captured["SWARM_ADVERTISE_ADDR"] == "127.0.0.1"

    @pytest.mark.parametrize(
        ("workers", "expect_join"),
        [((), False), (("w1",), True)],
    )
    def test_install_cluster_runs_expected_scripts(
        self, api: OnPremApi, workers: tuple[str, ...], expect_join: bool
    ) -> None:
        inst = SwarmInstaller(api, name="pool", manager="m", workers=workers)
        with patch.object(BundleScriptRunner, "run") as run_mock:
            inst.install_cluster(Path("/tmp/bundle"))
        scripts = [c.args[0] for c in run_mock.call_args_list]
        assert scripts[0] == "install-docker.sh"
        assert scripts[-1] == "deploy-arena-stack.sh"
        assert ("join-docker-swarm.sh" in scripts) is expect_join

    def test_install_cluster_reports_failing_stage(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(api, name="pool", manager="m")
        with (
            patch.object(
                BundleScriptRunner,
                "run",
                side_effect=StageFailed("install-docker.sh", 1, "kaboom"),
            ),
            pytest.raises(ClickException) as excinfo,
        ):
            inst.install_cluster(Path("/tmp/bundle"))
        message = str(excinfo.value)
        assert "Stage 1/" in message
        assert "Installing Docker Engine" in message  # human label, not script name
        assert "kaboom" in message  # captured output surfaced

    def test_verify_quotes_stack_name(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(
            api, name="pool", manager="manager-host", stack_name="arena; rm -rf /"
        )
        with patch.object(SshExecutor, "run", return_value="svc\t1/1") as ssh_mock:
            inst.verify(Path("/tmp"))
        remote_cmd = ssh_mock.call_args_list[0].args[1]
        assert "'arena; rm -rf /'" in remote_cmd
        assert "; rm -rf / --format" not in remote_cmd
        assert ssh_mock.call_count == 2

    def test_teardown_cluster_quotes_stack_name(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(
            api, name="pool", manager="manager-host", stack_name="arena$(whoami)"
        )
        with patch.object(SshExecutor, "run") as ssh_mock:
            inst.teardown_cluster()
        assert ssh_mock.call_args.args[1] == "sudo docker stack rm 'arena$(whoami)'"

    def test_down_cluster_scales_services(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(api, name="pool", manager="m", stack_name="arena")
        with patch.object(
            SshExecutor,
            "run",
            side_effect=["svc-a\nsvc-b", None],
        ) as ssh_mock:
            inst.down_cluster()
        scale_cmd = ssh_mock.call_args_list[1].args[1]
        assert "service scale" in scale_cmd
        assert "=0" in scale_cmd
        assert "svc-a=0" in scale_cmd
        assert "svc-b=0" in scale_cmd

    def test_down_cluster_missing_stack_warns(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(api, name="pool", manager="m", stack_name="arena")
        with (
            patch.object(SshExecutor, "run", return_value=""),
            patch("agilerl.arena.on_prem.installer.logger") as log,
        ):
            inst.down_cluster()
        log.warning.assert_called_once()
        assert "not found" in str(log.warning.call_args)

    def test_down_cluster_quotes_stack_name(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(
            api, name="pool", manager="manager-host", stack_name="arena$(whoami)"
        )
        with patch.object(
            SshExecutor,
            "run",
            side_effect=["svc$(whoami)", None],
        ) as ssh_mock:
            inst.down_cluster()
        list_cmd = ssh_mock.call_args_list[0].args[1]
        assert "'arena$(whoami)'" in list_cmd
        scale_cmd = ssh_mock.call_args_list[1].args[1]
        assert "'svc$(whoami)=0'" in scale_cmd

    def test_teardown_waits_before_leave_swarm(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(
            api, name="pool", manager="m", workers=("w1",), leave_swarm=True
        )
        with (
            patch.object(
                SshExecutor,
                "run",
                side_effect=[
                    None,  # stack rm
                    "arena\nother",  # first stack ls — still present
                    "",  # second stack ls — gone
                    None,  # leave m
                    None,  # leave w1
                ],
            ) as ssh_mock,
            patch("agilerl.arena.on_prem.installer.time.sleep"),
        ):
            inst.teardown_cluster()
        stack_ls_calls = [
            c.args[1] for c in ssh_mock.call_args_list if "stack ls" in c.args[1]
        ]
        assert len(stack_ls_calls) == 2
        assert sum("swarm leave" in c.args[1] for c in ssh_mock.call_args_list) == 2

    def test_teardown_cluster_leaves_swarm_on_all_hosts(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(
            api, name="pool", manager="m", workers=("w1",), leave_swarm=True
        )
        with (
            patch.object(
                SshExecutor,
                "run",
                side_effect=[None, "", None, None],
            ) as ssh_mock,
            patch("agilerl.arena.on_prem.installer.time.sleep"),
        ):
            inst.teardown_cluster()
        cmds = [c.args[1] for c in ssh_mock.call_args_list]
        assert any("stack rm" in c for c in cmds)
        assert sum("swarm leave" in c for c in cmds) == 2  # m + w1

    def test_preflight_requires_manager(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(api, name="pool", manager=None)
        with pytest.raises(ClickException, match="--manager is required"):
            inst.preflight_install()

    def test_preflight_requires_ssh_on_path(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(api, name="pool", manager="m")
        with (
            patch("agilerl.arena.on_prem.installer.shutil.which", return_value=None),
            pytest.raises(ClickException, match="ssh not found"),
        ):
            inst.preflight_install()

    def test_teardown_cluster_requires_manager(self, api: OnPremApi) -> None:
        inst = SwarmInstaller(api, name="pool", manager=None)
        with pytest.raises(ClickException, match="--manager is required"):
            inst.teardown_cluster()


# --------------------------------------------------------------------------- #
# HelmInstaller                                                                #
# --------------------------------------------------------------------------- #


class TestHelmInstaller:
    def test_install_cluster_runs_setup(
        self, api: OnPremApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with (
            patch(
                "agilerl.arena.on_prem.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            patch.object(BundleScriptRunner, "run") as run_mock,
        ):
            inst.install_cluster(helm_bundle)
        run_mock.assert_called_once_with("setup.sh", [])

    def test_install_cluster_requires_setup_script(
        self, api: OnPremApi, tmp_path: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with pytest.raises(ClickException, match="no setup.sh"):
            inst.install_cluster(tmp_path)

    def test_install_cluster_requires_helm_on_path(
        self, api: OnPremApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with (
            patch("agilerl.arena.on_prem.installer.shutil.which", return_value=None),
            pytest.raises(ClickException, match="helm not found"),
        ):
            inst.install_cluster(helm_bundle)

    def test_verify_runs_validate_script(
        self, api: OnPremApi, helm_bundle: Path
    ) -> None:
        (helm_bundle / "validate.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        inst = HelmInstaller(api, name="pool")
        with patch.object(BundleScriptRunner, "run") as run_mock:
            inst.verify(helm_bundle)
        run_mock.assert_called_once_with("validate.sh", [])

    def test_verify_warns_without_validate_script(
        self, api: OnPremApi, helm_bundle: Path
    ) -> None:
        inst = HelmInstaller(api, name="pool")
        with patch("agilerl.arena.on_prem.installer.logger") as log:
            inst.verify(helm_bundle)
        log.warning.assert_called_once()

    def test_helm_uninstall_invokes_cli(self) -> None:
        completed = MagicMock(returncode=0)
        with (
            patch(
                "agilerl.arena.on_prem.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            patch(
                "agilerl.arena.on_prem.installer.subprocess.run",
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
                "agilerl.arena.on_prem.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            patch(
                "agilerl.arena.on_prem.installer.subprocess.run",
                return_value=completed,
            ),
            patch("agilerl.arena.on_prem.installer.logger") as log,
        ):
            HelmInstaller._helm_uninstall("rel", "ns")  # must not raise
        log.warning.assert_called_once()
        assert log.warning.call_args.args[1] == 1

    def test_helm_uninstall_requires_helm_on_path(self) -> None:
        with (
            patch("agilerl.arena.on_prem.installer.shutil.which", return_value=None),
            pytest.raises(ClickException, match="helm not found"),
        ):
            HelmInstaller._helm_uninstall("rel", "ns")


# --------------------------------------------------------------------------- #
# Factory + functional facade                                                 #
# --------------------------------------------------------------------------- #


def test_build_installer_selects_subclass(api: OnPremApi) -> None:
    assert isinstance(
        build_installer("dockerSwarm", api, name="p", manager="m"), SwarmInstaller
    )
    assert isinstance(build_installer("helm", api, name="p"), HelmInstaller)


class TestRunOnPremInstall:
    def test_swarm_flow_uses_name_in_bundle_query(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            {},  # enable
            [{"name": "pool", "id": 9}],  # find_class
            (b"zip", "application/zip", None),  # fetch_bundle
        ]
        with (
            patch(
                "agilerl.arena.on_prem.installer.shutil.which",
                return_value="/usr/bin/ssh",
            ),
            patch(
                "agilerl.arena.on_prem.installer.extract_bundle",
                return_value=Path("/tmp/fake"),
            ),
            patch("agilerl.arena.on_prem.installer.validate_wireguard_bundle"),
            patch.object(SwarmInstaller, "install_cluster") as install_mock,
            patch.object(SwarmInstaller, "verify"),
        ):
            run_on_prem_install(
                client,
                name="pool",
                manager="10.0.0.1",
                workers=("10.0.0.2",),
                setup_type="dockerSwarm",
                ssh_user="ubuntu",
                skip_enable=False,
            )
        install_mock.assert_called_once()
        bundle_call = client._invoke_manifest_command.call_args_list[2]
        assert bundle_call.args[1] == {
            "name": "pool",
            "setupType": "dockerSwarm",
            "archivedType": "zip",
        }

    def test_swarm_requires_ssh_on_path(self) -> None:
        client = MagicMock(spec=ArenaClient)
        with (
            patch("agilerl.arena.on_prem.installer.shutil.which", return_value=None),
            pytest.raises(ClickException, match="ssh not found"),
        ):
            run_on_prem_install(
                client,
                name="pool",
                setup_type="dockerSwarm",
                manager="10.0.0.1",
                skip_enable=True,
            )
        client._invoke_manifest_command.assert_not_called()

    def test_swarm_requires_manager(self) -> None:
        client = MagicMock(spec=ArenaClient)
        with pytest.raises(ClickException, match="--manager"):
            run_on_prem_install(
                client, name="pool", setup_type="dockerSwarm", skip_enable=True
            )

    def test_helm_does_not_require_manager(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            {},  # enable
            [{"name": "k8s-pool", "num_nodes": 3}],  # find_class
            (b"zip", "application/zip", None),  # fetch_bundle
        ]
        with (
            patch(
                "agilerl.arena.on_prem.installer.extract_bundle",
                return_value=Path("/tmp/fake"),
            ),
            patch("agilerl.arena.on_prem.installer.validate_wireguard_bundle"),
            patch.object(HelmInstaller, "install_cluster") as install_mock,
            patch.object(HelmInstaller, "verify"),
        ):
            run_on_prem_install(
                client, name="k8s-pool", setup_type="helm", skip_enable=False
            )
        install_mock.assert_called_once()
        assert client._invoke_manifest_command.call_count == 3

    def test_install_fails_when_class_missing(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            {},  # enable
            [],  # find_class
        ]
        with (
            patch(
                "agilerl.arena.on_prem.installer.shutil.which",
                return_value="/usr/bin/helm",
            ),
            pytest.raises(ClickException, match="No on-prem resource class"),
        ):
            run_on_prem_install(
                client, name="missing-pool", setup_type="helm", skip_enable=False
            )
        assert client._invoke_manifest_command.call_count == 2


class TestRunOnPremTeardown:
    def test_swarm_requires_manager(self) -> None:
        client = MagicMock(spec=ArenaClient)
        with pytest.raises(ClickException, match="--manager is required"):
            run_on_prem_teardown(
                client,
                name="pool",
                setup_type="dockerSwarm",
                skip_cluster=False,
                disable_provider=False,
            )

    def test_swarm_removes_stack_and_disables(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.return_value = {}
        with patch.object(SwarmInstaller, "teardown_cluster") as teardown_mock:
            run_on_prem_teardown(
                client,
                name="pool",
                setup_type="dockerSwarm",
                skip_cluster=False,
                disable_provider=True,
                manager="10.0.0.1",
            )
        teardown_mock.assert_called_once()
        paths = [
            c.args[0]["path"] for c in client._invoke_manifest_command.call_args_list
        ]
        assert any(p.endswith("/disable") for p in paths)

    def test_helm_uninstalls_without_deleting_class(self) -> None:
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            (b"zip", "application/zip", None),  # fetch_bundle (teardown_cluster)
        ]
        with (
            patch(
                "agilerl.arena.on_prem.installer.extract_bundle",
                return_value=Path("/tmp/fake"),
            ),
            patch(
                "agilerl.arena.on_prem.installer.parse_helm_release_ids",
                return_value=("k8s-pool", "k8s-pool"),
            ),
            patch.object(HelmInstaller, "_helm_uninstall") as helm_mock,
        ):
            run_on_prem_teardown(
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
