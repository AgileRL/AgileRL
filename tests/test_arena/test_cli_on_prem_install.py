"""Tests for on-prem install helpers (bundle validation, env)."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click import ClickException
from click.testing import CliRunner

from agilerl.arena.client import ArenaClient
from agilerl.arena.cli_on_prem_install import (
    _all_hosts,
    _class_by_name,
    _delete_class_if_present,
    _download_bundle,
    _ensure_class,
    _helm_uninstall,
    _num_nodes_for_create,
    _parse_helm_release_ids,
    _run_docker_swarm_install,
    _run_docker_swarm_teardown,
    _run_helm_install,
    _run_script,
    _shell_runner,
    _ssh_connection_target,
    _ssh_remote_command,
    _ssh_target_host,
    _ssh_target_port,
    _swarm_script_env,
    _validate_tun0_conf,
    _validate_wireguard_bundle,
    _verify_swarm_stack,
    _warn_ignored_swarm_flags,
    build_install_command,
    build_teardown_command,
    normalize_setup_type,
    run_on_prem_install,
    run_on_prem_teardown,
)
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaAPIError


def _command_config() -> CommandConfig:
    return CommandConfig(
        api_key="test-key",
        base_url=None,
        keycloak_url=None,
        realm=None,
        client_id=None,
        request_timeout=30,
        upload_timeout=300,
    )


def _client_context(client: MagicMock) -> MagicMock:
    """A context manager (like ``arena_client``) that yields *client*."""
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__.return_value = client
    ctx_mgr.__exit__.return_value = False
    return ctx_mgr


def test_ssh_connection_target_uses_host_alias_without_ssh_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSH_USER", raising=False)
    assert _ssh_connection_target("op-ray-head", None) == "op-ray-head"
    assert _ssh_connection_target("ubuntu@op-ray-head", None) == "ubuntu@op-ray-head"
    assert _ssh_connection_target("op-ray-head", "deploy") == "deploy@op-ray-head"
    assert _ssh_connection_target("op-ray-head:2222", None) == "op-ray-head"


def test_swarm_script_env_sets_docker_reboot_assume_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOCKER_REBOOT_ASSUME_YES", raising=False)
    env = _swarm_script_env({"FOO": "bar"})
    assert env["DOCKER_REBOOT_ASSUME_YES"] == "1"
    assert env["FOO"] == "bar"


def test_validate_tun0_conf_accepts_complete_file(tmp_path: Path) -> None:
    conf = tmp_path / "tun0.conf"
    conf.write_text(
        """
[Interface]
Address = 172.24.0.50/32
PrivateKey = abc

[Peer]
PublicKey = def
PresharedKey = ghi
Endpoint = gw.example.com:51820
AllowedIPs = 10.0.0.0/8
""",
        encoding="utf-8",
    )
    _validate_tun0_conf(conf)


def test_validate_tun0_conf_rejects_incomplete_file(tmp_path: Path) -> None:
    conf = tmp_path / "tun0.conf"
    conf.write_text("[Interface]\n", encoding="utf-8")
    with pytest.raises(ClickException, match="Invalid tun0.conf"):
        _validate_tun0_conf(conf)


def test_validate_wireguard_bundle_swarm(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    (root / "config.d").mkdir(parents=True)
    (root / "config.d" / "tun0.conf").write_text(
        "[Interface]\n[Peer]\nPrivateKey = x\nPublicKey = y\n"
        "PresharedKey = z\nEndpoint = h:1\nAllowedIPs = 0.0.0.0/0\n",
        encoding="utf-8",
    )
    (root / "arena-stack.yaml").write_text(
        "volumes:\n  - ./config.d:/etc/wireguard/\n",
        encoding="utf-8",
    )
    _validate_wireguard_bundle(root, "dockerSwarm")


def test_validate_wireguard_bundle_helm(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    chart = root / "chart"
    chart.mkdir(parents=True)
    (chart / "values.yaml").write_text(
        """
wireguard:
  gatewayHost: "gw"
  gatewayPublicKey: "pub"
  peerPrivateKey: "priv"
  peerIp: "172.24.0.1"
  preSharedKey: "psk"
  allowedIps: []
""",
        encoding="utf-8",
    )
    _validate_wireguard_bundle(root, "helm")


def test_verify_swarm_stack_quotes_stack_name() -> None:
    """A malicious stack name must not break out of the remote shell command."""
    with patch("agilerl.arena.cli_on_prem_install._ssh_remote_command") as ssh_mock:
        _verify_swarm_stack(
            "manager-host",
            "arena; rm -rf /",
            ssh_user=None,
            ssh_extra_opts=None,
        )
    remote_cmd = ssh_mock.call_args.args[1]
    # The injected command is neutralized by shlex.quote (single-quoted, no bare ;).
    assert "'arena; rm -rf /'" in remote_cmd
    assert "; rm -rf / --format" not in remote_cmd


def test_docker_swarm_teardown_quotes_stack_name() -> None:
    """``docker stack rm`` must escape a caller-supplied stack name."""
    with patch("agilerl.arena.cli_on_prem_install._ssh_remote_command") as ssh_mock:
        _run_docker_swarm_teardown(
            manager="manager-host",
            workers=(),
            stack_name="arena$(whoami)",
            ssh_user=None,
            ssh_extra_opts=None,
            leave_swarm=False,
        )
    remote_cmd = ssh_mock.call_args.args[1]
    assert remote_cmd == "sudo docker stack rm 'arena$(whoami)'"


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
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


@pytest.mark.parametrize(
    ("manager", "workers", "expected"),
    [
        ("m", ("w1", "w2"), ["m", "w1", "w2"]),
        ("m", ("m", "w1"), ["m", "w1"]),  # manager not duplicated
        (" m ", (" ", "w1", "w1"), ["m", "w1"]),  # strips + dedups + drops blanks
        ("m", (), ["m"]),
    ],
)
def test_all_hosts_dedups_and_preserves_order(
    manager: str, workers: tuple[str, ...], expected: list[str]
) -> None:
    assert _all_hosts(manager, workers) == expected


@pytest.mark.parametrize(
    ("host", "expected_host", "expected_port"),
    [
        ("host", "host", None),
        ("user@host", "host", None),
        ("host:2222", "host", 2222),
        ("user@host:2222", "host", 2222),
        ("[::1]:22", "::1", 22),
        ("[::1]", "::1", None),
    ],
)
def test_ssh_target_parsing(
    host: str, expected_host: str, expected_port: int | None
) -> None:
    assert _ssh_target_host(host) == expected_host
    assert _ssh_target_port(host) == expected_port


@pytest.mark.parametrize(
    ("existing", "kind", "explicit", "manager", "workers", "expected"),
    [
        ({"num_nodes": 5}, "dockerSwarm", None, "m", ("w1",), 5),  # existing wins
        (None, "dockerSwarm", 3, "m", ("w1",), 3),  # explicit next
        (None, "helm", None, None, (), 1),  # helm default
        (None, "dockerSwarm", None, "m", ("w1", "w2"), 3),  # manager + workers
        ({"num_nodes": 0}, "dockerSwarm", 4, "m", (), 4),  # invalid existing ignored
    ],
)
def test_num_nodes_for_create(
    existing: dict[str, int] | None,
    kind: str,
    explicit: int | None,
    manager: str | None,
    workers: tuple[str, ...],
    expected: int,
) -> None:
    assert (
        _num_nodes_for_create(
            existing,
            kind=kind,  # type: ignore[arg-type]
            explicit=explicit,
            manager=manager,
            workers=workers,
        )
        == expected
    )


def test_class_by_name_returns_single_match() -> None:
    classes = [{"name": "a", "id": 1}, {"name": "b", "id": 2}]
    assert _class_by_name(classes, "b") == {"name": "b", "id": 2}


@pytest.mark.parametrize("classes", [[], [{"name": "other"}], "not-a-list", None])
def test_class_by_name_returns_none_when_absent(classes: object) -> None:
    assert _class_by_name(classes, "missing") is None


def test_class_by_name_rejects_duplicates() -> None:
    classes = [{"name": "dup"}, {"name": "dup"}]
    with pytest.raises(ArenaAPIError, match="Multiple on-prem classes"):
        _class_by_name(classes, "dup")


def test_warn_ignored_swarm_flags_lists_set_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _warn_ignored_swarm_flags(
        manager="m",
        workers=("w1",),
        ssh_user="ubuntu",
        ssh_extra_opts=None,
        advertise_addr=None,
    )
    err = capsys.readouterr().err
    assert "--manager" in err
    assert "--workers" in err
    assert "--ssh-user" in err
    assert "--ssh-extra-opts" not in err


def test_warn_ignored_swarm_flags_silent_when_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _warn_ignored_swarm_flags(
        manager=None,
        workers=(),
        ssh_user=None,
        ssh_extra_opts=None,
        advertise_addr=None,
    )
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------- #
# Remote command dispatch                                                      #
# --------------------------------------------------------------------------- #


def test_ssh_remote_command_runs_locally_without_ssh() -> None:
    completed = MagicMock(returncode=0)
    with (
        patch(
            "agilerl.arena.cli_on_prem_install._is_local_swarm_host",
            return_value=True,
        ),
        patch(
            "agilerl.arena.cli_on_prem_install.subprocess.run",
            return_value=completed,
        ) as run_mock,
    ):
        _ssh_remote_command("localhost", "echo hi", ssh_user=None, ssh_extra_opts=None)
    run_mock.assert_called_once()
    assert run_mock.call_args.args[0] == ["bash", "-lc", "echo hi"]


def test_ssh_remote_command_builds_ssh_invocation() -> None:
    completed = MagicMock(returncode=0)
    with (
        patch(
            "agilerl.arena.cli_on_prem_install._is_local_swarm_host",
            return_value=False,
        ),
        patch(
            "agilerl.arena.cli_on_prem_install.shutil.which",
            return_value="/usr/bin/ssh",
        ),
        patch(
            "agilerl.arena.cli_on_prem_install.subprocess.run",
            return_value=completed,
        ) as run_mock,
    ):
        _ssh_remote_command(
            "host:2222", "echo hi", ssh_user="ubuntu", ssh_extra_opts="-v"
        )
    cmd = run_mock.call_args.args[0]
    assert cmd[0] == "ssh"
    assert "-p" in cmd and "2222" in cmd
    assert "-v" in cmd  # extra opts are split in
    assert "ubuntu@host" in cmd
    assert cmd[-1] == "echo hi"  # remote command stays a single argv entry


def test_ssh_remote_command_requires_ssh_on_path() -> None:
    with (
        patch(
            "agilerl.arena.cli_on_prem_install._is_local_swarm_host",
            return_value=False,
        ),
        patch(
            "agilerl.arena.cli_on_prem_install.shutil.which",
            return_value=None,
        ),
        pytest.raises(ClickException, match="ssh not found"),
    ):
        _ssh_remote_command("host", "echo hi", ssh_user=None, ssh_extra_opts=None)


# --------------------------------------------------------------------------- #
# Docker Swarm / Helm orchestration                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("workers", "expect_join"),
    [((), False), (("w1",), True)],
)
def test_run_docker_swarm_install_runs_expected_scripts(
    workers: tuple[str, ...], expect_join: bool
) -> None:
    with patch("agilerl.arena.cli_on_prem_install._run_script") as run_script:
        _run_docker_swarm_install(
            Path("/tmp/bundle"),
            manager="m",
            workers=workers,
            ssh_user=None,
            ssh_extra_opts=None,
            advertise_addr=None,
        )
    scripts = [call.args[0].name for call in run_script.call_args_list]
    assert scripts[0] == "install-docker.sh"
    assert scripts[-1] == "deploy-arena-stack.sh"
    assert ("join-docker-swarm.sh" in scripts) is expect_join


def test_helm_uninstall_invokes_cli() -> None:
    completed = MagicMock(returncode=0)
    with (
        patch(
            "agilerl.arena.cli_on_prem_install.shutil.which",
            return_value="/usr/bin/helm",
        ),
        patch(
            "agilerl.arena.cli_on_prem_install.subprocess.run",
            return_value=completed,
        ) as run_mock,
    ):
        _helm_uninstall("rel", "ns")
    assert run_mock.call_args.args[0] == [
        "helm",
        "uninstall",
        "rel",
        "--namespace",
        "ns",
    ]


def test_helm_uninstall_tolerates_nonzero_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    completed = MagicMock(returncode=1)
    with (
        patch(
            "agilerl.arena.cli_on_prem_install.shutil.which",
            return_value="/usr/bin/helm",
        ),
        patch(
            "agilerl.arena.cli_on_prem_install.subprocess.run",
            return_value=completed,
        ),
    ):
        _helm_uninstall("rel", "ns")  # must not raise
    assert "helm uninstall exited 1" in capsys.readouterr().err


def test_helm_uninstall_requires_helm_on_path() -> None:
    with (
        patch(
            "agilerl.arena.cli_on_prem_install.shutil.which",
            return_value=None,
        ),
        pytest.raises(ClickException, match="helm not found"),
    ):
        _helm_uninstall("rel", "ns")


# --------------------------------------------------------------------------- #
# Bundle download + script runners                                            #
# --------------------------------------------------------------------------- #


def _zip_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()


def test_download_bundle_extracts_and_marks_scripts_executable(
    tmp_path: Path,
) -> None:
    client = MagicMock(spec=ArenaClient)
    client._invoke_manifest_command.return_value = (
        _zip_bytes(
            {
                "arena-train/setup.sh": "#!/bin/sh\necho hi\n",
                "arena-train/chart/values.yaml": "clusterName: pool\n",
            }
        ),
        "application/zip",
        None,
    )

    root = _download_bundle(
        client, class_name="pool", setup_type="helm", dest_dir=tmp_path
    )

    assert (root / "setup.sh").is_file()
    assert os.access(root / "setup.sh", os.X_OK)  # _prepare_bundle_scripts chmod
    _invoke, parsed = client._invoke_manifest_command.call_args.args
    assert parsed == {"name": "pool", "setupType": "helm", "archivedType": "zip"}


@pytest.mark.parametrize(
    ("which_map", "expected"),
    [
        ({"bash": "/usr/bin/bash", "sh": "/bin/sh"}, "/usr/bin/bash"),
        ({"sh": "/bin/sh"}, "/bin/sh"),  # falls back when bash absent
    ],
)
def test_shell_runner_resolves_interpreter(
    which_map: dict[str, str], expected: str
) -> None:
    with patch(
        "agilerl.arena.cli_on_prem_install.shutil.which",
        side_effect=lambda name: which_map.get(name),
    ):
        assert _shell_runner() == expected


def test_shell_runner_raises_when_no_shell() -> None:
    with (
        patch("agilerl.arena.cli_on_prem_install.shutil.which", return_value=None),
        pytest.raises(ClickException, match="bash or sh not found"),
    ):
        _shell_runner()


def test_run_script_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ClickException, match="missing script"):
        _run_script(tmp_path / "absent.sh", [], env={}, cwd=tmp_path)


def test_run_script_invokes_runner_with_args(tmp_path: Path) -> None:
    script = tmp_path / "go.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    completed = MagicMock(returncode=0)
    with (
        patch(
            "agilerl.arena.cli_on_prem_install._shell_runner",
            return_value="/bin/bash",
        ),
        patch(
            "agilerl.arena.cli_on_prem_install.subprocess.run",
            return_value=completed,
        ) as run_mock,
    ):
        _run_script(script, ["a1", "a2"], env={"K": "V"}, cwd=tmp_path)

    assert run_mock.call_args.args[0] == [
        "/bin/bash",
        str(script.resolve()),
        "a1",
        "a2",
    ]
    assert run_mock.call_args.kwargs["check"] is True
    assert run_mock.call_args.kwargs["cwd"] == tmp_path
    assert run_mock.call_args.kwargs["env"] == {"K": "V"}


def test_run_helm_install_runs_setup_script(tmp_path: Path) -> None:
    (tmp_path / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    with (
        patch(
            "agilerl.arena.cli_on_prem_install.shutil.which",
            return_value="/usr/bin/helm",
        ),
        patch("agilerl.arena.cli_on_prem_install._run_script") as run_script,
    ):
        _run_helm_install(tmp_path)
    assert run_script.call_args.args[0] == tmp_path / "setup.sh"


def test_run_helm_install_requires_setup_script(tmp_path: Path) -> None:
    with pytest.raises(ClickException, match="no setup.sh"):
        _run_helm_install(tmp_path)


def test_run_helm_install_requires_helm_on_path(tmp_path: Path) -> None:
    (tmp_path / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    with (
        patch("agilerl.arena.cli_on_prem_install.shutil.which", return_value=None),
        pytest.raises(ClickException, match="helm not found"),
    ):
        _run_helm_install(tmp_path)


def test_parse_helm_release_ids_requires_values_file(tmp_path: Path) -> None:
    with pytest.raises(ClickException, match="cannot determine release name"):
        _parse_helm_release_ids(tmp_path)


# --------------------------------------------------------------------------- #
# Resource-class API helpers                                                   #
# --------------------------------------------------------------------------- #


def test_ensure_class_creates_when_absent() -> None:
    client = MagicMock(spec=ArenaClient)
    client._invoke_manifest_command.side_effect = [[], {"name": "pool", "id": 7}]

    row = _ensure_class(client, name="pool", num_nodes=2)

    assert row["id"] == 7
    assert client._invoke_manifest_command.call_count == 2


def test_ensure_class_rejects_non_object_create_response() -> None:
    client = MagicMock(spec=ArenaClient)
    client._invoke_manifest_command.side_effect = [[], "oops-not-a-dict"]
    with pytest.raises(ArenaAPIError, match="not an object"):
        _ensure_class(client, name="pool", num_nodes=1)


def test_delete_class_if_present_skips_when_absent() -> None:
    client = MagicMock(spec=ArenaClient)
    client._invoke_manifest_command.return_value = []  # list returns no classes
    _delete_class_if_present(client, "pool")
    client._invoke_manifest_command.assert_called_once()  # only the list call


# --------------------------------------------------------------------------- #
# Flow-level guards + teardown                                                 #
# --------------------------------------------------------------------------- #


def test_run_on_prem_install_requires_ssh_on_path() -> None:
    client = MagicMock(spec=ArenaClient)
    with (
        patch("agilerl.arena.cli_on_prem_install.shutil.which", return_value=None),
        pytest.raises(ClickException, match="ssh not found"),
    ):
        run_on_prem_install(
            client,
            name="pool",
            setup_type="dockerSwarm",
            manager="10.0.0.1",
            skip_enable=True,
        )


def test_teardown_docker_swarm_requires_manager() -> None:
    client = MagicMock(spec=ArenaClient)
    with pytest.raises(ClickException, match="--manager is required"):
        run_on_prem_teardown(
            client,
            name="pool",
            setup_type="dockerSwarm",
            skip_cluster=False,
            delete_class=False,
            disable_provider=False,
        )


def test_teardown_docker_swarm_removes_stack_and_disables() -> None:
    client = MagicMock(spec=ArenaClient)
    with patch(
        "agilerl.arena.cli_on_prem_install._run_docker_swarm_teardown"
    ) as teardown_mock:
        run_on_prem_teardown(
            client,
            name="pool",
            setup_type="dockerSwarm",
            skip_cluster=False,
            delete_class=False,
            disable_provider=True,
            manager="10.0.0.1",
        )
    teardown_mock.assert_called_once()
    paths = [
        call.args[0].get("path")
        for call in client._invoke_manifest_command.call_args_list
    ]
    assert any(p.endswith("/disable") for p in paths)


# --------------------------------------------------------------------------- #
# Click command callbacks                                                      #
# --------------------------------------------------------------------------- #


def test_install_command_parses_workers_and_delegates() -> None:
    client = MagicMock(spec=ArenaClient)
    with (
        patch(
            "agilerl.arena.cli.arena_client",
            return_value=_client_context(client),
        ),
        patch("agilerl.arena.cli_on_prem_install.run_on_prem_install") as run_mock,
    ):
        result = CliRunner().invoke(
            build_install_command(),
            ["pool", "--manager", "10.0.0.1", "--workers", "w1, w2"],
            obj=_command_config(),
        )

    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["name"] == "pool"
    assert kwargs["manager"] == "10.0.0.1"
    assert kwargs["workers"] == ("w1", "w2")  # comma-split + stripped
    assert kwargs["setup_type"] == "dockerSwarm"  # default


def test_teardown_command_maps_flags() -> None:
    client = MagicMock(spec=ArenaClient)
    with (
        patch(
            "agilerl.arena.cli.arena_client",
            return_value=_client_context(client),
        ),
        patch("agilerl.arena.cli_on_prem_install.run_on_prem_teardown") as run_mock,
    ):
        result = CliRunner().invoke(
            build_teardown_command(),
            [
                "pool",
                "--manager",
                "m",
                "--keep-class",
                "--disable-provider",
                "--leave-swarm",
            ],
            obj=_command_config(),
        )

    assert result.exit_code == 0, result.output
    kwargs = run_mock.call_args.kwargs
    assert kwargs["name"] == "pool"
    assert kwargs["delete_class"] is False  # --keep-class inverts
    assert kwargs["disable_provider"] is True
    assert kwargs["leave_swarm"] is True
