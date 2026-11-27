# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``arena cluster`` Click commands and verbosity helper."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner, Result

from agilerl.byoc import (
    ClusterDynamicGroup,
    build_cluster_connect_command,
    build_cluster_plan_command,
    build_cluster_preflight_command,
    build_cluster_register_command,
    build_cluster_revoke_token_command,
    build_cluster_rotate_token_command,
    build_cluster_unregister_command,
)
from agilerl.byoc.api import ByocApi
from agilerl.byoc.commands import (
    _apply_verbosity,
    _byoc_provider_with_module,
    _load_destroy_target,
    _load_provisioning,
    _node_groups_for_classes,
    _recover_spec,
    _register_gpu_resource_classes,
    _register_runtime_fields,
    _registration_values_from_spec,
    _resource_classes_from_outputs,
    _run_cluster_destroy,
    _selected_worker_names,
    build_cluster_generate_spec_command,
    build_cluster_provision_command,
    build_cluster_register_classes_command,
    build_cluster_status_command,
)
from agilerl.byoc.kube_target import NoKubeContextError, kubeconfig_environment
from agilerl.byoc.preflight import CheckStatus, PreflightCheck
from agilerl.byoc.provisioning.module_catalog import ModulePin, write_module_pin
from agilerl.byoc.provisioning.providers.nebius.platforms import GpuPreset
from agilerl.byoc.provisioning.providers.nebius.spec import (
    NEBIUS_CREDENTIALS_PATH,
    NEBIUS_PROFILE_NAME,
    attach_service_account_credentials,
    recover_stored_config,
    registration_payload,
    resource_classes,
    service_account_credentials,
)
from agilerl.byoc.provisioning.providers.nebius.terraform import (
    storage_project_id,
)
from agilerl.byoc.provisioning.spec import ClusterSpec
from agilerl.byoc.provisioning.terraform import ClusterOutputs
from agilerl.client import ArenaClient
from agilerl.config import CommandConfig
from agilerl.exceptions import ArenaAPIError

COMMANDS = "agilerl.byoc.commands"
INSTALLED_BYOC_RELEASES = [
    ("arena-byoc-agent", "arena"),
    ("arena-byoc-storage", "storage"),
]
AGENT_HELM_RELEASE = [("arena-byoc-agent", "arena")]


ARCHIVED = {
    "unregistered": True,
    "status": "Archived",
    "cleanupCommand": "helm uninstall arena-byoc-agent --namespace arena --timeout 10m",
}
DEREGISTERING = {"unregistered": False, "status": "Deregistering"}


@pytest.fixture(autouse=True)
def pinned_target(
    fake_kube_target: Callable[..., MagicMock], monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """Point every cluster command at a fixed context that Arena accepts."""
    monkeypatch.setattr(f"{COMMANDS}.require_registered_cluster", MagicMock())
    return fake_kube_target(COMMANDS)


@pytest.fixture
def no_kube_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """No kubeconfig names a context for the command to use."""
    monkeypatch.setattr(
        f"{COMMANDS}.kube_target",
        MagicMock(side_effect=NoKubeContextError("No Kubernetes context to use.")),
    )


@dataclass
class UnregisterRun:
    """What ``arena cluster unregister`` printed and called."""

    result: Result
    unregister: MagicMock
    uninstall: MagicMock
    find_cluster: MagicMock
    list_releases: MagicMock


def run_unregister(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    args: list[str],
    installed: list[tuple[str, str]] | None = None,
    response: dict[str, object] | None = None,
    input_text: str | None = None,
    still_listed: int = 0,
    unregister_error: Exception | None = None,
    uninstall_error: Exception | None = None,
    helm: bool = True,
) -> UnregisterRun:
    """Run ``unregister --name prod-cluster`` plus *args* against mocks.

    Arena answers *response* (``ARCHIVED`` by default) and keeps listing the
    cluster for *still_listed* polls.
    """
    listed = [{"name": "prod-cluster", "status": "Deregistering"}] * still_listed
    with (
        patch(f"{COMMANDS}.helm_available", return_value=helm),
        patch(
            f"{COMMANDS}.list_installed_byoc_releases",
            return_value=installed or [],
        ) as list_releases,
        patch(
            f"{COMMANDS}.uninstall_byoc_releases",
            side_effect=uninstall_error,
        ) as uninstall,
        patch(
            f"{COMMANDS}.arena_client",
            return_value=client_context(MagicMock()),
        ),
        patch(f"{COMMANDS}.time.sleep"),
        patch.object(
            ByocApi,
            "unregister_cluster",
            return_value=response or ARCHIVED,
            side_effect=unregister_error,
        ) as unregister,
        patch.object(ByocApi, "find_cluster", side_effect=[*listed, None]) as find,
    ):
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "prod-cluster", *args],
            obj=command_config,
            input=input_text,
        )
    return UnregisterRun(result, unregister, uninstall, find, list_releases)


@pytest.fixture
def kubeconfig(tmp_path: Path) -> Path:
    """The kubeconfig pinned to the agent cluster's context."""
    return tmp_path / "pinned-kubeconfig"


class TestClusterUnregister:
    def test_requires_confirmation(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(command_config, client_context, [], input_text="\n")

        assert run.result.exit_code == 0, run.result.output
        assert "Aborted." in run.result.output
        run.unregister.assert_not_called()

    def test_sends_no_flags_by_default(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(command_config, client_context, [], input_text="y\n")

        assert run.result.exit_code == 0, run.result.output
        run.unregister.assert_called_once_with(
            "prod-cluster", stop_runs=False, force=False
        )
        assert "Unregistered BYOC cluster 'prod-cluster'." in run.result.output

    @pytest.mark.parametrize(
        ("flag", "sent"),
        [
            ("--stop-runs", {"stop_runs": True, "force": False}),
            ("--force", {"stop_runs": False, "force": True}),
        ],
    )
    def test_passes_stop_runs_and_force(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        flag: str,
        sent: dict[str, bool],
    ) -> None:
        run = run_unregister(command_config, client_context, [flag, "--yes"])

        assert run.result.exit_code == 0, run.result.output
        run.unregister.assert_called_once_with("prod-cluster", **sent)

    @pytest.mark.parametrize(
        ("flag", "sent"),
        [
            ("--stop-runs", {"stop_runs": True, "force": False}),
            ("--force", {"stop_runs": False, "force": True}),
        ],
    )
    def test_stop_runs_and_force_uninstall_on_the_named_context(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        pinned_target: MagicMock,
        kubeconfig: Path,
        flag: str,
        sent: dict[str, bool],
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            [flag, "--context", "corp", "--uninstall-helm", "--yes"],
            installed=AGENT_HELM_RELEASE,
        )

        assert run.result.exit_code == 0, run.result.output
        pinned_target.assert_called_once_with("prod-cluster", None, "corp")
        run.unregister.assert_called_once_with("prod-cluster", **sent)
        run.uninstall.assert_called_once_with(AGENT_HELM_RELEASE, kubeconfig)

    def test_refusal_with_active_runs_is_shown(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--yes"],
            installed=AGENT_HELM_RELEASE,
            unregister_error=ArenaAPIError(
                'experiment 4 "run" (Running). Stop them first, or unregister '
                "with --stop-runs."
            ),
        )

        assert run.result.exit_code != 0
        assert "--stop-runs" in str(run.result.exception)
        run.uninstall.assert_not_called()

    def test_waits_for_the_agent_then_uninstalls_it(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        kubeconfig: Path,
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--uninstall-helm", "--yes"],
            installed=INSTALLED_BYOC_RELEASES,
            response=DEREGISTERING,
            still_listed=2,
        )

        assert run.result.exit_code == 0, run.result.output
        assert "Waiting for the agent" in run.result.output
        assert run.find_cluster.call_count == 3
        assert "Keeping MinIO storage Helm release" in run.result.output
        run.uninstall.assert_called_once_with(AGENT_HELM_RELEASE, kubeconfig)
        assert "kubectl delete namespace" not in run.result.output

    def test_older_agent_is_waited_for_then_cleaned_up_by_uninstall(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--yes"],
            response={**DEREGISTERING, "cleanupCommand": "helm uninstall x"},
            still_listed=1,
        )

        assert run.result.exit_code == 0, run.result.output
        assert run.find_cluster.call_count == 2
        assert "did not remove its job namespaces" in run.result.output
        assert "helm uninstall x" in run.result.output

    def test_yes_alone_never_touches_the_cluster_and_prints_how_to_remove_the_agent(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        pinned_target: MagicMock,
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--yes"],
            installed=AGENT_HELM_RELEASE,
            response=DEREGISTERING,
        )

        assert run.result.exit_code == 0, run.result.output
        pinned_target.assert_not_called()
        run.list_releases.assert_not_called()
        run.uninstall.assert_not_called()
        assert "The agent removed its job namespaces." in run.result.output
        assert "helm uninstall arena-byoc-agent --namespace arena" in run.result.output

    def test_gives_up_waiting_and_keeps_the_agent(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--yes", "--wait-timeout", "0"],
            installed=AGENT_HELM_RELEASE,
            response=DEREGISTERING,
            still_listed=1,
        )

        assert run.result.exit_code != 0
        assert "still Deregistering" in run.result.output
        assert "--force" in run.result.output
        run.uninstall.assert_not_called()

    def test_agent_is_uninstalled_by_default(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        kubeconfig: Path,
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            [],
            installed=INSTALLED_BYOC_RELEASES,
            input_text="\n\ny\n",
        )

        assert run.result.exit_code == 0, run.result.output
        assert "checkpoints, metrics, and datasets" in run.result.output
        run.uninstall.assert_called_once_with(AGENT_HELM_RELEASE, kubeconfig)

    def test_uninstall_storage_removes_minio(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        kubeconfig: Path,
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--uninstall-helm", "--uninstall-storage", "--yes"],
            installed=INSTALLED_BYOC_RELEASES,
        )

        assert run.result.exit_code == 0, run.result.output
        run.uninstall.assert_called_once_with(INSTALLED_BYOC_RELEASES, kubeconfig)

    def test_uninstalls_an_agent_outside_the_default_namespace(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        kubeconfig: Path,
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--uninstall-helm", "--yes"],
            installed=[("arena-byoc-agent", "arena-agent")],
        )

        assert run.result.exit_code == 0, run.result.output
        run.uninstall.assert_called_once_with(
            [("arena-byoc-agent", "arena-agent")], kubeconfig
        )

    def test_kept_agent_after_a_finished_wait_says_namespaces_were_removed(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--keep-agent", "--yes"],
            installed=AGENT_HELM_RELEASE,
            response=DEREGISTERING,
            still_listed=1,
        )

        assert run.result.exit_code == 0, run.result.output
        assert run.find_cluster.call_count == 2
        run.uninstall.assert_not_called()
        assert "The agent removed its job namespaces. To uninstall it, run:" in (
            run.result.output
        )
        assert (
            "helm uninstall arena-byoc-agent --namespace arena --timeout 10m"
        ) in run.result.output

    def test_kept_agent_after_force_prints_its_cleanup_command(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        pinned_target: MagicMock,
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--force", "--keep-agent", "--context", "corp"],
            installed=[("prod-cluster", "arena-agent")],
            input_text="y\n",
        )

        assert run.result.exit_code == 0, run.result.output
        pinned_target.assert_called_once_with("prod-cluster", None, "corp")
        run.uninstall.assert_not_called()
        assert "did not remove its job namespaces" in run.result.output
        assert (
            "helm uninstall prod-cluster --namespace arena-agent --timeout 10m || "
            "kubectl delete namespace "
            "--selector=arena.agilerl.com/agent-namespace=arena-agent"
        ) in run.result.output

    @pytest.mark.usefixtures("no_kube_context")
    def test_without_kube_context_prints_arenas_cleanup_command(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(
            command_config, client_context, ["--force"], input_text="y\n"
        )

        assert run.result.exit_code == 0, run.result.output
        assert (
            "Could not check Helm releases. No Kubernetes context" in run.result.output
        )
        run.uninstall.assert_not_called()
        run.unregister.assert_called_once_with(
            "prod-cluster", stop_runs=False, force=True
        )
        assert ARCHIVED["cleanupCommand"] in run.result.output

    def test_without_helm_skips_the_release_check(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        pinned_target: MagicMock,
    ) -> None:
        run = run_unregister(
            command_config, client_context, [], input_text="y\n", helm=False
        )

        assert run.result.exit_code == 0, run.result.output
        assert "helm is not on PATH" in run.result.output
        pinned_target.assert_not_called()
        run.uninstall.assert_not_called()
        run.unregister.assert_called_once()

    @pytest.mark.usefixtures("no_kube_context")
    def test_uninstall_flag_needs_a_kube_context(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(
            command_config, client_context, ["--uninstall-helm", "--yes"]
        )

        assert run.result.exit_code != 0
        assert "No Kubernetes context" in run.result.output
        run.unregister.assert_not_called()

    def test_checks_the_cluster_uid_before_uninstalling(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        check = MagicMock(side_effect=click.ClickException("is not Arena cluster"))
        monkeypatch.setattr(f"{COMMANDS}.require_registered_cluster", check)

        run = run_unregister(
            command_config,
            client_context,
            ["--context", "corp", "--stop-runs", "--uninstall-helm", "--yes"],
            installed=AGENT_HELM_RELEASE,
        )

        assert run.result.exit_code != 0
        assert "is not Arena cluster" in run.result.output
        assert check.call_args.args[1] == "prod-cluster"
        run.list_releases.assert_not_called()
        run.uninstall.assert_not_called()
        run.unregister.assert_not_called()

    def test_older_arena_archives_at_once(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
        kubeconfig: Path,
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--uninstall-helm", "--yes"],
            installed=AGENT_HELM_RELEASE,
            response={"unregistered": True},
        )

        assert run.result.exit_code == 0, run.result.output
        run.find_cluster.assert_not_called()
        run.uninstall.assert_called_once_with(AGENT_HELM_RELEASE, kubeconfig)

    def test_helm_uninstall_failure_is_reported_after_arena_unregisters(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        run = run_unregister(
            command_config,
            client_context,
            ["--uninstall-helm", "--yes"],
            installed=AGENT_HELM_RELEASE,
            uninstall_error=click.ClickException("helm uninstall failed"),
        )

        assert run.result.exit_code != 0
        assert "helm uninstall failed" in run.result.output
        run.unregister.assert_called_once()


def test_cluster_rotate_token_requires_confirmation(
    command_config: CommandConfig,
) -> None:
    with patch("agilerl.byoc.commands.run_cluster_rotate_token") as rotate_mock:
        result = CliRunner().invoke(
            build_cluster_rotate_token_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="\n",
        )

    assert result.exit_code == 0, result.output
    assert "Aborted." in result.output
    rotate_mock.assert_not_called()


@pytest.mark.parametrize(
    ("build", "target", "args"),
    [
        (build_cluster_rotate_token_command, "run_cluster_rotate_token", []),
        (build_cluster_revoke_token_command, "run_cluster_revoke_token", []),
        (
            build_cluster_revoke_token_command,
            "run_cluster_revoke_token",
            ["--platform"],
        ),
    ],
)
def test_cluster_credential_commands_yes_skips_prompt(
    command_config: CommandConfig,
    client_context: Callable[[MagicMock], MagicMock],
    build: Callable[[], click.Command],
    target: str,
    args: list[str],
) -> None:
    client = MagicMock()
    with (
        patch(
            "agilerl.byoc.commands.arena_client",
            return_value=client_context(client),
        ),
        patch(f"agilerl.byoc.commands.{target}") as run_mock,
    ):
        result = CliRunner().invoke(
            build(), ["--name", " prod-cluster ", "--yes", *args], obj=command_config
        )

    assert result.exit_code == 0, result.output
    assert "?" not in result.output
    run_mock.assert_called_once_with(client, name="prod-cluster", platform=bool(args))


def test_cluster_revoke_token_asks_first(command_config: CommandConfig) -> None:
    with patch("agilerl.byoc.commands.run_cluster_revoke_token") as revoke_mock:
        result = CliRunner().invoke(
            build_cluster_revoke_token_command(),
            ["--name", "prod-cluster"],
            obj=command_config,
            input="n\n",
        )

    assert result.exit_code == 0, result.output
    assert "Revoke every credential of 'prod-cluster' now?" in result.output
    revoke_mock.assert_not_called()


def test_cluster_credential_command_rejects_a_blank_name(
    command_config: CommandConfig,
) -> None:
    result = CliRunner().invoke(
        build_cluster_rotate_token_command(),
        ["--name", "  ", "--yes"],
        obj=command_config,
    )

    assert result.exit_code != 0
    assert "--name is required." in result.output


PREFLIGHT_PASS = PreflightCheck(
    "kubernetes_version", CheckStatus.PASS, "Kubernetes 1.31"
)
PREFLIGHT_FAIL = PreflightCheck(
    "one_agent", CheckStatus.FAIL, "Arena agents already in this cluster: b/x"
)


def run_preflight_command(
    command_config: CommandConfig,
    args: list[str],
    checks: list[PreflightCheck],
    registered: dict[str, object] | None = None,
) -> tuple[Result, MagicMock]:
    """Run ``arena cluster preflight`` with *checks* as the cluster's results."""
    with (
        patch(f"{COMMANDS}.run_preflight", return_value=checks) as preflight,
        patch(f"{COMMANDS}.arena_client", return_value=MagicMock()),
        patch(f"{COMMANDS}.registered_cluster", return_value=registered),
        patch(
            f"{COMMANDS}.own_agent_release",
            side_effect=lambda _kubeconfig, _namespace, name: name,
        ),
    ):
        result = CliRunner().invoke(
            build_cluster_preflight_command(), args, obj=command_config
        )
    return result, preflight


class TestClusterPreflightCommand:
    def test_prints_the_table_and_exits_zero_when_nothing_failed(
        self, command_config: CommandConfig, pinned_target: MagicMock, kubeconfig: Path
    ) -> None:
        result, preflight = run_preflight_command(
            command_config, ["--context", "corp"], [PREFLIGHT_PASS]
        )

        assert result.exit_code == 0, result.output
        assert "Kubernetes 1.31" in result.output
        pinned_target.assert_called_once_with(None, None, "corp")
        assert preflight.call_args.args == (kubeconfig, "arena", None, None)

    def test_failed_check_exits_non_zero(self, command_config: CommandConfig) -> None:
        result, _ = run_preflight_command(
            command_config, ["--context", "corp"], [PREFLIGHT_PASS, PREFLIGHT_FAIL]
        )

        assert result.exit_code == 1
        assert "Preflight failed: one_agent." in result.output

    def test_json_report_is_the_only_stdout(
        self, command_config: CommandConfig
    ) -> None:
        result, _ = run_preflight_command(
            command_config, ["--context", "corp", "--json"], [PREFLIGHT_FAIL]
        )

        assert result.exit_code == 1
        report = json.loads(result.stdout)
        assert report["context"] == "corp"
        assert report["passed"] is False
        assert report["checks"][0]["status"] == "fail"

    def test_json_report_of_a_passing_cluster_exits_zero(
        self, command_config: CommandConfig
    ) -> None:
        result, _ = run_preflight_command(
            command_config, ["--context", "corp", "--json"], [PREFLIGHT_PASS]
        )

        assert result.exit_code == 0
        assert json.loads(result.stdout)["passed"] is True

    def test_name_lets_the_clusters_own_agent_through(
        self, command_config: CommandConfig
    ) -> None:
        result, preflight = run_preflight_command(
            command_config,
            [
                "--name",
                "corp-k8s",
                "--context",
                "corp",
                "--agent-namespace",
                "arena-agent",
                "--ray-data-storage-class-name",
                "fast",
            ],
            [PREFLIGHT_PASS],
            {"name": "corp-k8s"},
        )

        assert result.exit_code == 0, result.output
        assert preflight.call_args.args[1:] == ("arena-agent", "corp-k8s", "fast")

    def test_registered_name_keeps_an_installed_default_agent(
        self, command_config: CommandConfig
    ) -> None:
        deployment = {
            "metadata": {
                "name": "arena-byoc-agent",
                "namespace": "arena",
                "labels": {
                    "app.kubernetes.io/name": "arena-byoc-agent",
                    "app.kubernetes.io/instance": "arena-byoc-agent",
                },
            }
        }

        def kubectl(*_args: object, **_kwargs: object) -> MagicMock:
            return MagicMock(
                returncode=0,
                stdout=json.dumps({"items": [deployment]}),
                stderr="",
            )

        with (
            patch(
                f"{COMMANDS}.run_preflight", return_value=[PREFLIGHT_PASS]
            ) as preflight,
            patch(f"{COMMANDS}.arena_client", return_value=MagicMock()),
            patch(f"{COMMANDS}.registered_cluster", return_value={"name": "corp-k8s"}),
            patch("agilerl.byoc.preflight.subprocess.run", side_effect=kubectl),
        ):
            result = CliRunner().invoke(
                build_cluster_preflight_command(),
                ["--name", "corp-k8s", "--context", "corp"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert preflight.call_args.args[2] == "arena-byoc-agent"

    def test_name_arena_does_not_have_counts_every_agent(
        self, command_config: CommandConfig
    ) -> None:
        result, preflight = run_preflight_command(
            command_config,
            ["--name", "corp-k8s", "--context", "corp"],
            [PREFLIGHT_PASS],
        )

        assert result.exit_code == 0, result.output
        assert preflight.call_args.args[2] is None


class TestClusterConnectCommand:
    def test_passes_every_option(self, command_config: CommandConfig) -> None:
        with patch(f"{COMMANDS}.run_cluster_connect") as connect:
            result = CliRunner().invoke(
                build_cluster_connect_command(),
                [
                    "--name",
                    " corp-k8s ",
                    "--context",
                    "corp",
                    "--storage-bucket",
                    "arena-data",
                    "--rotate-token",
                    "--wait-timeout",
                    "30",
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        options = connect.call_args.args[1]
        assert options.name == "corp-k8s"
        assert options.kube_context == "corp"
        assert options.agent_namespace == "arena"
        assert options.storage_bucket == "arena-data"
        assert options.rotate_token is True
        assert options.wait_timeout == 30.0
        assert options.yes is True

    def test_blank_name_is_rejected(self, command_config: CommandConfig) -> None:
        with patch(f"{COMMANDS}.run_cluster_connect") as connect:
            result = CliRunner().invoke(
                build_cluster_connect_command(), ["--name", " "], obj=command_config
            )

        assert result.exit_code != 0
        assert "--name is required." in result.output
        connect.assert_not_called()


def test_cluster_group_includes_hardcoded_commands() -> None:
    cluster = ClusterDynamicGroup()

    assert cluster.name == "cluster"
    assert "unregister" in cluster.commands
    assert "rotate-token" in cluster.commands
    assert "upgrade" in cluster.commands
    assert "smoke" in cluster.commands
    assert "register" in cluster.commands
    assert "register-classes" in cluster.commands
    assert "preflight" in cluster.commands
    assert "connect" in cluster.commands


def test_apply_verbosity_toggles_debug_level() -> None:
    arena_logger = logging.getLogger("agilerl")
    original = arena_logger.level
    try:
        _apply_verbosity(verbose=False)
        assert arena_logger.level == original  # unchanged
        _apply_verbosity(verbose=True)
        assert arena_logger.level == logging.DEBUG
    finally:
        arena_logger.setLevel(original)


def _nebius_spec(**overrides: object) -> ClusterSpec:
    raw = dict(overrides)
    nebius: dict[str, object] = {
        "tenant_id": "tenant",
        "project_id": "project",
        "storage_project_id": "storage-project",
        "subnet_id": "subnet",
        "object_storage": {"size_gib": 1024},
    }
    nebius.update(raw)
    return ClusterSpec.model_validate(
        {
            "provider": "nebius",
            "name": "test",
            "terraform_state": {"bucket": "test-tfstate", "endpoint": "https://s3"},
            "arena": {
                "storage": {
                    "bucket": "bucket",
                    "endpoint": "https://storage.example.com",
                },
                "inference": {"domain": "inference.example.com"},
            },
            "nebius": nebius,
        }
    )


class TestNebiusByocProvider:
    def test_includes_state_endpoint(self) -> None:
        provider = registration_payload(_nebius_spec())

        assert provider["config"]["terraform_state"]["endpoint"] == "https://s3"

    def test_fills_project_ids_the_spec_omits(self) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        provider = registration_payload(
            spec, project_id="project-created", storage_project_id="storage-created"
        )

        assert provider["config"]["project_id"] == "project-created"
        assert provider["config"]["storage_project_id"] == "storage-created"
        assert "service_account_key" not in provider["config"]
        assert "profile" not in provider["config"]


class TestServiceAccountCredentials:
    def test_builds_key_and_profile_for_the_created_account(self) -> None:
        pem_label = "PRIVATE " + "KEY"
        private_key = f"-----BEGIN {pem_label}-----\nYWJj\n-----END {pem_label}-----"
        outputs = {
            "service_account_id": "serviceaccount-e00m886x7epbx082j3",
            "service_account_public_key_id": "publickey-e00abc",
            "service_account_private_key": private_key + "\n",
            "project_id": "project-e00",
        }

        documents = service_account_credentials(outputs)

        assert documents is not None
        key, profile = documents
        parsed = json.loads(key)
        assert parsed["subject-credentials"] == {
            "type": "JWT",
            "alg": "RS256",
            "private-key": private_key,
            "kid": "publickey-e00abc",
            "iss": "serviceaccount-e00m886x7epbx082j3",
            "sub": "serviceaccount-e00m886x7epbx082j3",
        }
        assert profile == "\n".join(
            (
                f"default: {NEBIUS_PROFILE_NAME}",
                "profiles:",
                f"  {NEBIUS_PROFILE_NAME}:",
                "    endpoint: api.nebius.cloud",
                "    parent-id: project-e00",
                f"    service-account-credentials-file-path: {NEBIUS_CREDENTIALS_PATH}",
                "",
            )
        )

    def test_omits_credentials_when_the_account_was_not_created(self) -> None:
        payload = registration_payload(_nebius_spec())

        attach_service_account_credentials(
            payload,
            {"project_id": "project-e00", "service_account_id": None},
        )

        assert "service_account_key" not in payload["config"]
        assert "profile" not in payload["config"]

    def test_register_body_stores_the_created_credentials(self) -> None:
        payload = _byoc_provider_with_module(
            _nebius_spec(),
            ClusterOutputs(
                cluster_name="test",
                context="test",
                kubeconfig_path=Path("kubeconfig"),
                storage_access_key_id="access",
                storage_secret_access_key="secret",
                worker_node_group_ids={"gpu": "nodegroup-1"},
                project_id="project-e00",
                service_account_key='{"subject-credentials":{}}',
                service_account_profile="default: nebius-default\n",
            ),
            None,
        )

        assert payload["config"]["service_account_key"] == '{"subject-credentials":{}}'
        assert payload["config"]["profile"] == "default: nebius-default\n"


class TestStoredSpecHelpers:
    def test_stored_subnet_id_none_when_project_missing(self) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        recovered = recover_stored_config(spec, {"subnet_id": "subnet-2"})

        assert recovered.nebius.subnet_id is None

    def test_stored_state_location_empty_when_not_a_mapping(self) -> None:
        spec = _nebius_spec()

        assert recover_stored_config(spec, {"terraform_state": "nope"}) is spec


class TestGpuResourceClasses:
    def test_requires_node_group_id(self) -> None:
        with pytest.raises(click.ClickException, match="missing node group id"):
            resource_classes(_nebius_spec(), worker_node_group_ids={})

    def test_wraps_invalid_preset(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "2vcpu-8gb"}])

        with pytest.raises(click.ClickException, match="GPU worker preset"):
            resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})

    def test_registers_the_preset_machine_size(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "1gpu-4vcpu-16gb"}])

        classes = resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})

        compute = classes[0].metadata["computeResource"]
        assert compute["numCpus"] == 4
        assert compute["memoryBytes"] == "16 GiB"

    def test_keeps_a_one_vcpu_preset(self) -> None:
        spec = _nebius_spec(workers=[{"name": "gpu", "preset": "1gpu-1vcpu-1gb"}])

        classes = resource_classes(spec, worker_node_group_ids={"gpu": "ng-1"})

        compute = classes[0].metadata["computeResource"]
        assert compute["numCpus"] == 1
        assert compute["memoryBytes"] == "1 GiB"


class TestRegistrationValuesFromSpec:
    def test_rejects_non_list_parent_refs(self) -> None:
        with pytest.raises(click.ClickException, match="JSON list"):
            _registration_values_from_spec(
                _nebius_spec(), gateway_api_parent_refs={"name": "gw"}
            )


class TestKubeconfigEnvironment:
    def test_restores_previous_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KUBECONFIG", "/old/config")
        kubeconfig = Path("/new/config")

        with kubeconfig_environment(kubeconfig):
            assert os.environ["KUBECONFIG"] == str(kubeconfig)

        assert os.environ["KUBECONFIG"] == "/old/config"


class TestLoadProvisioning:
    def test_wraps_invalid_spec(self, tmp_path: Path) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text("not: valid: yaml: [\n", encoding="utf-8")

        with pytest.raises(click.ClickException, match="Invalid YAML"):
            _load_provisioning(spec, None)


class TestUnregisterNameRequired:
    def test_blank_name_is_rejected(self, command_config: CommandConfig) -> None:
        result = CliRunner().invoke(
            build_cluster_unregister_command(),
            ["--name", "   "],
            obj=command_config,
            input="y\n",
        )

        assert result.exit_code != 0
        assert "--name is required" in result.output


class TestGenerateSpecCommand:
    def test_wraps_existing_file_without_force(self, tmp_path: Path) -> None:
        destination = tmp_path / "arena-nebius.yaml"
        destination.write_text("existing\n", encoding="utf-8")

        result = CliRunner().invoke(
            build_cluster_generate_spec_command(),
            ["--output", str(destination)],
        )

        assert result.exit_code != 0
        assert "already exists" in result.output


class TestStatusCommand:
    def test_prints_outputs(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch("agilerl.byoc.commands.ClusterProvisioner") as provisioner_cls,
        ):
            provisioner_cls.return_value.status.return_value = {
                "cluster_id": "mk8s-1",
                "service_account_private_key": "pem-should-not-print",
                "storage_secret_access_key": "secret-access-should-not-print",
                "catalog_secret_access_key": "catalog-secret-should-not-print",
            }
            result = CliRunner().invoke(
                build_cluster_status_command(),
                ["--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "cluster_id: mk8s-1" in result.output
        assert "service_account_private_key: <sensitive>" in result.output
        assert "storage_secret_access_key: <sensitive>" in result.output
        assert "pem-should-not-print" not in result.output
        assert "secret-access-should-not-print" not in result.output
        assert "catalog_secret_access_key: <sensitive>" in result.output
        assert "catalog-secret-should-not-print" not in result.output


class TestProvisionCommand:
    def test_rejects_provider_mismatch(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        mismatched = MagicMock()
        mismatched.provider = "other"
        with (
            patch(
                "agilerl.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.byoc.commands._registration_values_from_spec",
                return_value=(mismatched, {"byoc_provider": {}}),
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_provision_command(),
                ["nebius", "--spec", str(spec), "--yes"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "does not match spec provider" in result.output


class TestLoadDestroyTarget:
    def test_wraps_invalid_stored_cluster(self, command_config: CommandConfig) -> None:
        with (
            patch(
                "agilerl.byoc.commands.arena_client",
            ) as client_cm,
            patch.object(ByocApi, "byoc_cluster", return_value={"name": "broken"}),
        ):
            client = MagicMock()
            cm = MagicMock()
            cm.__enter__.return_value = client
            cm.__exit__.return_value = False
            client_cm.return_value = cm
            with pytest.raises(click.ClickException, match="no stored cloud settings"):
                _load_destroy_target(command_config, None, "broken", None)


class TestRegistrationValuesFromSpecJson:
    def test_parses_json_string_parent_refs(self) -> None:
        spec, registration = _registration_values_from_spec(
            _nebius_spec(),
            gateway_api_parent_refs='[{"name": "arena", "namespace": "arena"}]',
        )

        assert registration["gateway_api_parent_refs"] == [
            {"name": "arena", "namespace": "arena"}
        ]
        assert spec.arena.gateway.parent_refs == [
            {"name": "arena", "namespace": "arena"}
        ]

    def test_defaults_placement_and_applies_override(self) -> None:
        spec, registration = _registration_values_from_spec(_nebius_spec())
        overridden, overridden_registration = _registration_values_from_spec(
            _nebius_spec(),
            placement="ephemeral",
        )

        assert spec.values.placement == "in-cluster"
        assert registration["placement"] == "in-cluster"
        assert overridden.values.placement == "ephemeral"
        assert overridden_registration["placement"] == "ephemeral"

    def test_platform_scope_defaults_placement_when_omitted(self) -> None:
        spec, registration = _registration_values_from_spec(
            _nebius_spec(),
            byoc_scope="platform",
        )

        assert spec.values.byoc_scope == "platform"
        assert spec.values.placement == "ephemeral"
        assert registration["byoc_scope"] == "platform"
        assert registration["placement"] == "ephemeral"

    def test_name_only_register_defaults_placement(self) -> None:
        fields = _register_runtime_fields(
            None,
            {
                "placement": None,
                "domain": None,
                "gateway_api_parent_refs": None,
            },
        )

        assert fields["placement"] == "in-cluster"

    def test_name_only_platform_register_defaults_placement(self) -> None:
        fields = _register_runtime_fields(
            None,
            {
                "placement": None,
                "byoc_scope": "platform",
                "domain": None,
                "gateway_api_parent_refs": None,
            },
        )

        assert fields["placement"] == "ephemeral"
        assert fields["byoc_scope"] == "platform"


class TestKubeconfigEnvironmentUnset:
    def test_clears_when_previously_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("KUBECONFIG", raising=False)
        kubeconfig = Path("/new/config")

        with kubeconfig_environment(kubeconfig):
            assert os.environ["KUBECONFIG"] == str(kubeconfig)

        assert "KUBECONFIG" not in os.environ


class TestRecoverNebiusSpec:
    def test_rejects_unknown_providers(self, command_config: CommandConfig) -> None:
        spec = ClusterSpec.model_construct(
            **{**_nebius_spec().model_dump(), "provider": "other"}
        )

        with pytest.raises(ValueError, match="Unsupported cluster provider"):
            _recover_spec(command_config, spec)

    def test_skips_lookup_when_both_project_ids_are_set(
        self, command_config: CommandConfig
    ) -> None:
        spec = _nebius_spec()

        with patch("agilerl.byoc.commands.arena_client") as client_cm:
            assert _recover_spec(command_config, spec) is spec
            client_cm.assert_not_called()

    def test_fills_missing_ids_from_arena(self, command_config: CommandConfig) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        with (
            patch("agilerl.byoc.commands.arena_client") as client_cm,
            patch.object(
                ByocApi,
                "stored_provider_config",
                return_value={
                    "project_id": "project-stored",
                    "storage_project_id": "storage-stored",
                    "terraform_state": {
                        "key": "clusters/test/terraform.tfstate",
                        "endpoint": "https://s3.stored",
                    },
                },
            ),
        ):
            client = MagicMock()
            cm = MagicMock()
            cm.__enter__.return_value = client
            cm.__exit__.return_value = False
            client_cm.return_value = cm
            recovered = _recover_spec(command_config, spec)

        assert recovered.nebius.project_id == "project-stored"
        assert recovered.nebius.storage_project_id == "storage-stored"

    def test_org_without_byoc_enabled_keeps_the_spec(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        spec = _nebius_spec(
            project_id=None, storage_project_id=None, subnet_id=None, region="eu-north1"
        )

        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch.object(
                ByocApi,
                "stored_provider_config",
                side_effect=ArenaAPIError("Not Found", status_code=404),
            ),
        ):
            assert _recover_spec(command_config, spec) is spec

    def test_other_lookup_errors_propagate(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        spec = _nebius_spec(
            project_id=None, storage_project_id=None, subnet_id=None, region="eu-north1"
        )

        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch.object(
                ByocApi,
                "stored_provider_config",
                side_effect=ArenaAPIError("boom", status_code=500),
            ),
            pytest.raises(ArenaAPIError, match="boom"),
        ):
            _recover_spec(command_config, spec)


class TestSpecWithStoredNebiusConfig:
    def test_returns_spec_when_nothing_is_missing(self) -> None:
        spec = _nebius_spec()

        assert recover_stored_config(spec, {"project_id": "other"}) is spec

    def test_recovered_project_ids_are_not_terraform_inputs(self) -> None:
        spec = _nebius_spec(
            project_id=None,
            storage_project_id=None,
            subnet_id=None,
            region="eu-north1",
        )

        recovered = recover_stored_config(
            spec,
            {"project_id": "project-stored", "storage_project_id": "storage-stored"},
        )

        values = recovered.terraform_values()
        assert values["project_id"] == ""
        assert values["storage_project_id"] == ""
        assert storage_project_id(recovered) == "storage-stored"
        assert recovered.nebius_cloud().project_id == "project-stored"

    def test_spec_project_ids_stay_terraform_inputs(self) -> None:
        spec = _nebius_spec(storage_project_id=None, region="eu-north1")

        recovered = recover_stored_config(
            spec, {"project_id": "other", "storage_project_id": "storage-stored"}
        )

        values = recovered.terraform_values()
        assert values["project_id"] == "project"
        assert values["subnet_id"] == "subnet"
        assert values["storage_project_id"] == ""
        assert storage_project_id(recovered) == "storage-stored"


class TestStoredStateLocationEndpoint:
    def test_fills_endpoint_when_spec_omits_it(self) -> None:
        spec = _nebius_spec()
        spec = spec.model_copy(
            update={
                "terraform_state": spec.terraform_state.model_copy(
                    update={"endpoint": None, "key": None}
                )
            }
        )

        recovered = recover_stored_config(
            spec, {"terraform_state": {"endpoint": "https://s3.stored"}}
        )

        assert recovered.terraform_state.endpoint == "https://s3.stored"


class TestRegisterCommand:
    def test_requires_name_without_spec(self, command_config: CommandConfig) -> None:
        result = CliRunner().invoke(
            build_cluster_register_command(),
            [],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "--name is required" in result.output

    def test_requires_spec_when_provider_is_given(
        self, command_config: CommandConfig
    ) -> None:
        result = CliRunner().invoke(
            build_cluster_register_command(),
            ["nebius", "--name", "prod-cluster"],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "--spec is required" in result.output

    def test_wraps_invalid_spec(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text("not: valid: yaml: [\n", encoding="utf-8")

        result = CliRunner().invoke(
            build_cluster_register_command(),
            ["--spec", str(spec)],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "Invalid YAML" in result.output

    def test_aborts_when_provision_is_declined(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch("agilerl.byoc.commands.ClusterProvisioner") as provisioner_cls,
            patch("agilerl.byoc.commands.click.confirm", return_value=False),
        ):
            provisioner_cls.return_value.cluster_exists.return_value = False
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Aborted." in result.output

    def test_wraps_invalid_domain_without_spec(
        self, command_config: CommandConfig
    ) -> None:
        with patch("agilerl.byoc.commands.run_cluster_register"):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["--name", "prod-cluster", "--domain", "https://bad.example.com"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "bare DNS name" in result.output

    def test_parses_parent_refs_without_spec(
        self, command_config: CommandConfig
    ) -> None:
        with patch("agilerl.byoc.commands.run_cluster_register") as register_mock:
            result = CliRunner().invoke(
                build_cluster_register_command(),
                [
                    "--name",
                    "prod-cluster",
                    "--gateway-api-parent-refs",
                    '[{"name": "arena"}]',
                    "--skip-enable",
                    "--no-write",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert register_mock.call_args.kwargs["gateway_api_parent_refs"] == [
            {"name": "arena"}
        ]


class TestPlanCommand:
    def test_wraps_invalid_registration_overrides(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.byoc.commands._registration_values_from_spec",
                side_effect=ValueError("bare DNS name"),
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_plan_command(),
                ["--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "bare DNS name" in result.output


class TestProvisionCommandOverrides:
    def test_wraps_invalid_registration_overrides(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        with (
            patch(
                "agilerl.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.byoc.commands._registration_values_from_spec",
                side_effect=ValueError("bare DNS name"),
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_provision_command(),
                ["nebius", "--spec", str(spec), "--yes"],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "bare DNS name" in result.output


class TestRotateTokenCommand:
    def test_rejects_blank_name(self, command_config: CommandConfig) -> None:
        result = CliRunner().invoke(
            build_cluster_rotate_token_command(),
            ["--name", "   "],
            obj=command_config,
        )

        assert result.exit_code != 0
        assert "--name is required" in result.output


class TestRegisterProviderMismatch:
    def test_rejects_provider_that_does_not_match_spec(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )

        def recover(_config: CommandConfig, cluster_spec: ClusterSpec) -> ClusterSpec:
            object.__setattr__(cluster_spec, "provider", "other")
            return cluster_spec

        with patch("agilerl.byoc.commands._recover_spec", side_effect=recover):
            result = CliRunner().invoke(
                build_cluster_register_command(),
                ["nebius", "--spec", str(spec)],
                obj=command_config,
            )

        assert result.exit_code != 0
        assert "does not match spec provider" in result.output


H100_PRESET = GpuPreset(
    platform="gpu-h100-sxm",
    preset="1gpu-16vcpu-200gb",
    gpu_count=1,
    vcpu_count=16,
    memory_gib=200,
    vram_gib=80,
)


class TestByocProviderModulePin:
    def test_adds_the_installed_module_pin(self, tmp_path: Path) -> None:
        write_module_pin(
            tmp_path,
            ModulePin(
                name="arena-cluster-nebius",
                version="1.0.0",
                digest="sha256:" + "ab" * 32,
                orchestration="nebius.v1",
                files=("main.tf",),
            ),
        )

        payload = _byoc_provider_with_module(
            _nebius_spec(),
            ClusterOutputs(
                cluster_name="test",
                context="test",
                kubeconfig_path=Path("kubeconfig"),
                storage_access_key_id="access",
                storage_secret_access_key="secret",
                worker_node_group_ids={},
            ),
            tmp_path,
        )

        assert payload["config"]["module"] == {
            "name": "arena-cluster-nebius",
            "version": "1.0.0",
            "digest": "sha256:" + "ab" * 32,
            "orchestration": "nebius.v1",
        }


class TestResourceClassesFromOutputs:
    def test_rejects_missing_worker_ids(self) -> None:
        with pytest.raises(click.ClickException, match="no worker_node_group_ids"):
            _resource_classes_from_outputs(_nebius_spec(), {})

    def test_rejects_a_non_map(self) -> None:
        with pytest.raises(click.ClickException, match="must be a string map"):
            _resource_classes_from_outputs(
                _nebius_spec(), {"worker_node_group_ids": "ng-1"}
            )


class TestRegisterGpuResourceClasses:
    def test_rejects_non_nebius_specs(self, command_config: CommandConfig) -> None:
        spec = ClusterSpec.model_validate(
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
                },
            }
        )

        with pytest.raises(click.ClickException, match="supports Nebius"):
            _register_gpu_resource_classes(
                command_config,
                MagicMock(),
                spec,
                Path("cluster.yaml"),
                Path("terraform"),
                Path("out"),
                register_all=True,
                requested_presets=(),
                yes=True,
                module_version=None,
                register_node_group=False,
            )

    def test_rejects_an_empty_preset_list(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        runner = MagicMock()

        with (
            patch(
                "agilerl.byoc.commands._cluster_provisioner",
                return_value=runner,
            ),
            patch(
                "agilerl.byoc.commands.list_registerable_gpu_presets",
                return_value=[],
            ),
        ):
            with pytest.raises(click.ClickException, match="No GPU presets"):
                _register_gpu_resource_classes(
                    command_config,
                    api,
                    _nebius_spec(),
                    tmp_path / "cluster.yaml",
                    tmp_path / "terraform",
                    tmp_path / "out",
                    register_all=True,
                    requested_presets=(),
                    yes=True,
                    module_version=None,
                    register_node_group=False,
                )

    def test_rejects_an_empty_preset_list_for_an_ephemeral_cluster(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        api = MagicMock()
        api.find_cluster.return_value = {"name": "test"}
        spec = _nebius_spec().model_copy(
            update={
                "values": _nebius_spec().values.model_copy(
                    update={"placement": "ephemeral"}
                )
            }
        )

        with (
            patch(
                "agilerl.byoc.commands._cluster_provisioner",
                return_value=MagicMock(),
            ),
            patch(
                "agilerl.byoc.commands.list_registerable_gpu_presets",
                return_value=[],
            ),
            pytest.raises(
                click.ClickException, match="No GPU presets in region projects"
            ),
        ):
            _register_gpu_resource_classes(
                command_config,
                api,
                spec,
                tmp_path / "cluster.yaml",
                tmp_path / "terraform",
                tmp_path / "out",
                register_all=True,
                requested_presets=(),
                yes=True,
                module_version=None,
                register_node_group=False,
            )


class TestSelectedWorkerNames:
    def test_rejects_a_preset_missing_from_the_spec(self) -> None:
        with pytest.raises(click.ClickException, match="was not added"):
            _selected_worker_names(_nebius_spec(), [H100_PRESET])


class TestNodeGroupsForClasses:
    def test_returns_recorded_ids_when_every_group_exists(self, tmp_path: Path) -> None:
        spec = _nebius_spec(
            workers=[
                {
                    "name": "gpu-h100-sxm-1gpu-16vcpu-200gb",
                    "platform": "gpu-h100-sxm",
                    "preset": "1gpu-16vcpu-200gb",
                    "register_node_group": True,
                }
            ]
        )
        runner = MagicMock()
        ids = {"gpu-h100-sxm-1gpu-16vcpu-200gb": "ng-1"}

        group_ids = _node_groups_for_classes(
            spec,
            original=spec,
            spec_path=tmp_path / "cluster.yaml",
            state_dir=tmp_path / "terraform",
            output_dir=tmp_path / "out",
            new_workers=[],
            selected=[H100_PRESET],
            terraform_outputs={"worker_node_group_ids": ids},
            runner=runner,
        )

        assert group_ids == ids
        runner.status.assert_not_called()


class TestLoadDestroyTargetStoredRow:
    def test_wraps_invalid_stored_provider_config(
        self, command_config: CommandConfig
    ) -> None:
        with (
            patch("agilerl.byoc.commands.arena_client") as client_cm,
            patch.object(
                ByocApi,
                "byoc_cluster",
                return_value={
                    "name": "broken",
                    "byocProvider": {"provider": "aws", "config": "nope"},
                },
            ),
        ):
            client = MagicMock()
            cm = MagicMock()
            cm.__enter__.return_value = client
            cm.__exit__.return_value = False
            client_cm.return_value = cm
            with pytest.raises(click.ClickException, match="no stored AWS settings"):
                _load_destroy_target(command_config, None, "broken", None)


class TestProvisionPrintsModulePin:
    def test_echoes_the_installed_module(
        self, command_config: CommandConfig, tmp_path: Path
    ) -> None:
        spec = tmp_path / "cluster.yaml"
        spec.write_text(
            """
provider: nebius
name: arena-nebius
terraform_state:
  bucket: arena-nebius-tfstate
arena:
  storage:
    bucket: arena-data
    endpoint: https://storage.example.com
  inference:
    domain: inference.example.com
nebius:
  tenant_id: tenant-1
  project_id: project-1
  storage_project_id: storage-project-1
  subnet_id: subnet-1
  object_storage:
    size_gib: 1024
""",
            encoding="utf-8",
        )
        state_dir = tmp_path / "terraform"
        write_module_pin(
            state_dir,
            ModulePin(
                name="arena-cluster-nebius",
                version="1.0.0",
                digest="sha256:" + "ab" * 32,
                orchestration="nebius.v1",
                files=("main.tf",),
            ),
        )
        outputs = ClusterOutputs(
            cluster_name="arena-nebius",
            context="arena-nebius",
            kubeconfig_path=tmp_path / "kubeconfig",
            storage_access_key_id="AKIA-test",
            storage_secret_access_key="secret-test",
            worker_node_group_ids={"workers": "ng-1"},
        )

        with (
            patch(
                "agilerl.byoc.commands._recover_spec",
                side_effect=lambda _config, cluster_spec: cluster_spec,
            ),
            patch(
                "agilerl.byoc.commands.ClusterProvisioner.provision",
                return_value=outputs,
            ),
        ):
            result = CliRunner().invoke(
                build_cluster_provision_command(),
                [
                    "nebius",
                    "--spec",
                    str(spec),
                    "--state-dir",
                    str(state_dir),
                    "--yes",
                ],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert "Module arena-cluster-nebius 1.0.0" in result.output


def _platform_spec() -> ClusterSpec:
    spec, _registration = _registration_values_from_spec(
        _nebius_spec(), name="arena", byoc_scope="platform"
    )
    return spec


class TestPlatformClusters:
    def test_unregister_platform_sends_the_platform_scope(
        self,
        command_config: CommandConfig,
        http_client: ArenaClient,
        sent_scope: Callable[[ArenaClient, str], dict[str, object]],
    ) -> None:
        with (
            patch("agilerl.byoc.commands.helm_available", return_value=False),
            patch("agilerl.config.build_client", return_value=http_client),
        ):
            result = CliRunner().invoke(
                build_cluster_unregister_command(),
                ["--platform", "--name", "arena", "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert sent_scope(http_client, "/clusters/unregister")["body"] == "platform"

    def test_unregister_without_the_flag_sends_no_scope(
        self,
        command_config: CommandConfig,
        http_client: ArenaClient,
        sent_scope: Callable[[ArenaClient, str], dict[str, object]],
    ) -> None:
        with (
            patch("agilerl.byoc.commands.helm_available", return_value=False),
            patch("agilerl.config.build_client", return_value=http_client),
        ):
            result = CliRunner().invoke(
                build_cluster_unregister_command(),
                ["--name", "arena", "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert sent_scope(http_client, "/clusters/unregister") == {
            "query": None,
            "body": None,
        }

    def test_rotate_token_passes_the_flag(
        self,
        command_config: CommandConfig,
        client_context: Callable[[MagicMock], MagicMock],
    ) -> None:
        with (
            patch(
                "agilerl.byoc.commands.arena_client",
                return_value=client_context(MagicMock()),
            ),
            patch("agilerl.byoc.commands.run_cluster_rotate_token") as rotate,
        ):
            result = CliRunner().invoke(
                build_cluster_rotate_token_command(),
                ["--platform", "--name", "arena", "--yes"],
                obj=command_config,
            )

        assert result.exit_code == 0, result.output
        assert rotate.call_args.kwargs["platform"] is True

    @pytest.mark.parametrize(
        "command", ["unregister", "rotate-token", "upgrade", "manifest", "smoke"]
    )
    def test_help_says_staff_only(self, command: str) -> None:
        result = CliRunner().invoke(ClusterDynamicGroup().commands[command], ["--help"])

        help_text = " ".join(result.output.split())
        assert (
            "--platform Target AgileRL's own platform clusters, "
            "which needs AgileRL staff access." in help_text
        )

    def test_register_classes_uses_the_spec_scope(
        self,
        command_config: CommandConfig,
        http_client: ArenaClient,
        sent_scope: Callable[[ArenaClient, str], dict[str, object]],
        tmp_path: Path,
    ) -> None:
        # Arrange
        spec_file = tmp_path / "cluster.yaml"
        spec_file.write_text("unused\n", encoding="utf-8")

        def register(
            _config: CommandConfig, api: ByocApi, spec: ClusterSpec, **_kwargs: object
        ) -> None:
            api.find_cluster(spec.name)

        # Act
        with (
            patch(
                "agilerl.byoc.commands._load_provisioning",
                return_value=(_platform_spec(), tmp_path),
            ),
            patch("agilerl.byoc.commands._recover_spec", side_effect=lambda _c, s: s),
            patch(
                "agilerl.byoc.commands._register_gpu_resource_classes",
                side_effect=register,
            ),
            patch("agilerl.config.build_client", return_value=http_client),
        ):
            result = CliRunner().invoke(
                build_cluster_register_classes_command(),
                ["--spec", str(spec_file), "--all"],
                obj=command_config,
            )

        # Assert
        assert result.exit_code == 0, result.output
        assert sent_scope(http_client, "/clusters/list")["query"] == "platform"

    def test_destroy_unregisters_a_platform_spec_cluster(
        self,
        command_config: CommandConfig,
        http_client: ArenaClient,
        sent_scope: Callable[[ArenaClient, str], dict[str, object]],
        tmp_path: Path,
    ) -> None:
        with (
            patch(
                "agilerl.byoc.commands._load_destroy_target",
                return_value=(_platform_spec(), tmp_path, None),
            ),
            patch("agilerl.byoc.commands._cluster_provisioner") as provisioner,
            patch("agilerl.byoc.commands._echo_destroy_result"),
            patch("agilerl.config.build_client", return_value=http_client),
        ):
            _run_cluster_destroy(
                command_config,
                spec=None,
                name="arena",
                state_dir=None,
                delete_storage=False,
                delete_storage_project=False,
                delete_all=False,
                module_version=None,
                yes=True,
            )

        provisioner.return_value.destroy.assert_called_once()
        assert sent_scope(http_client, "/clusters/list")["query"] == "platform"
        assert sent_scope(http_client, "/clusters/unregister")["body"] == "platform"
