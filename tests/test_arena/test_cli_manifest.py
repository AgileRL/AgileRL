"""Tests for manifest-backed Arena CLI plumbing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.cli_manifest import (
    ArenaRootGroup,
    _manifest_spec_to_click_option,
    build_manifest_click_command,
    capabilities_show_on_prem_root,
    caps_allow_on_prem_at_root,
    pythonize_manifest_param_name,
    register_on_prem_manifest_group,
    write_binary_atomic,
)
from agilerl.arena.client import ArenaClient
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaValidationError


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


@pytest.fixture
def api_key_client() -> ArenaClient:
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        return ArenaClient(api_key="test-key")


class TestCapsAllowOnPremAtRoot:
    def test_enterprise_true(self) -> None:
        assert caps_allow_on_prem_at_root({"enterprise": True})

    def test_onprem_cli_feature_without_enterprise(self) -> None:
        assert caps_allow_on_prem_at_root(
            {"enterprise": False, "features": {"onPremCli": True}},
        )

    def test_neither(self) -> None:
        assert not caps_allow_on_prem_at_root(
            {"enterprise": False, "features": {"onPremCli": False}},
        )

    def test_missing_features(self) -> None:
        assert not caps_allow_on_prem_at_root({"enterprise": False})

    def test_cli_manifest_schema_without_flags(self) -> None:
        assert not caps_allow_on_prem_at_root(
            {
                "enterprise": False,
                "features": {"onPremCli": False},
                "cli": {"manifestSchemaVersion": 1, "root": {}},
            },
        )


class TestCapabilitiesShowOnPremRoot:
    @pytest.mark.parametrize(
        ("caps", "expected"),
        [
            (None, None),  # capabilities unavailable
            ({"enterprise": True}, True),
            ({"enterprise": False, "features": {"onPremCli": False}}, False),
        ],
    )
    def test_resolves_visibility_and_closes_client(
        self, caps: dict[str, object] | None, expected: bool | None
    ) -> None:
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = caps
        with patch(
            "agilerl.arena.cli_manifest.build_client",
            return_value=client_mock,
        ):
            result = capabilities_show_on_prem_root(_command_config())
        assert result is expected
        client_mock.close.assert_called_once()


class TestPythonizeManifestParamName:
    def test_camel_case(self) -> None:
        assert pythonize_manifest_param_name("archivedType") == "archived_type"

    def test_snake_case(self) -> None:
        assert pythonize_manifest_param_name("num_nodes") == "num_nodes"


class TestGetCliCapabilities:
    def test_404_returns_none(self, api_key_client: ArenaClient) -> None:
        resp = MagicMock()
        resp.status_code = 404
        with patch.object(api_key_client._http, "request", return_value=resp):
            assert api_key_client._get_cli_capabilities(force_refresh=True) is None

    def test_invalid_json_body_returns_none(self, api_key_client: ArenaClient) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("no JSON object")
        with patch.object(api_key_client._http, "request", return_value=resp):
            assert api_key_client._get_cli_capabilities(force_refresh=True) is None

    def test_wrong_schema_returns_none(self, api_key_client: ArenaClient) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "ok": True,
            "data": {"schemaVersion": 999},
        }
        with patch.object(api_key_client._http, "request", return_value=resp):
            assert api_key_client._get_cli_capabilities(force_refresh=True) is None

    def test_uses_bounded_timeout(self, api_key_client: ArenaClient) -> None:
        """Capability checks gate ``--help``; the request must not block on the
        full request timeout when the API is slow."""
        api_key_client._request_timeout = 30
        resp = MagicMock()
        resp.status_code = 404
        with patch.object(
            api_key_client._http, "request", return_value=resp
        ) as req_mock:
            api_key_client._get_cli_capabilities(force_refresh=True)
        assert (
            req_mock.call_args.kwargs["timeout"]
            == ArenaClient._CAPABILITIES_TIMEOUT_SECS
        )

    def test_schema_version_string_one_accepted(
        self, api_key_client: ArenaClient
    ) -> None:
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "ok": True,
            "data": {"schemaVersion": "1", "enterprise": False},
        }
        with patch.object(api_key_client._http, "request", return_value=resp):
            caps = api_key_client._get_cli_capabilities(force_refresh=True)
        assert caps == {"schemaVersion": "1", "enterprise": False}


class TestInvokeManifestCommand:
    def test_delete_class_sends_name_as_query_not_body(
        self, api_key_client: ArenaClient
    ) -> None:
        """The classes/delete endpoint reads ``name`` from the query string."""
        from agilerl.arena.cli_on_prem_install import _DELETE_CLASS_INVOKE

        with patch.object(
            api_key_client, "_request", return_value={"ok": True}
        ) as mocked:
            api_key_client._invoke_manifest_command(
                _DELETE_CLASS_INVOKE, {"name": "pool"}
            )
        _args, kwargs = mocked.call_args
        assert kwargs.get("params") == {"name": "pool"}
        assert "json" not in kwargs  # must NOT be sent as a JSON body

    def test_rejects_non_allowlisted_path(self, api_key_client: ArenaClient) -> None:
        invoke = {
            "method": "GET",
            "path": "/api/evil",
            "responseKind": "json",
            "params": [],
        }
        with pytest.raises(ArenaValidationError):
            api_key_client._invoke_manifest_command(invoke, {})

    def test_accepts_on_prem_prefix_json(self, api_key_client: ArenaClient) -> None:
        invoke = {
            "method": "POST",
            "path": "/api/cli/v1/on-prem/enable",
            "responseKind": "json",
            "params": [],
        }
        with patch.object(
            api_key_client, "_request", return_value={"ok": True}
        ) as mocked:
            api_key_client._invoke_manifest_command(invoke, {})
        mocked.assert_called_once_with("POST", "/api/cli/v1/on-prem/enable")

    def test_post_with_empty_manifest_params_sends_json_body(
        self, api_key_client: ArenaClient
    ) -> None:
        invoke = {
            "method": "POST",
            "path": "/api/cli/v1/on-prem/classes/create",
            "responseKind": "json",
            "params": [],
        }
        body = {"name": "pool", "num_nodes": 2, "enabled": True}
        with patch.object(api_key_client, "_request", return_value={"id": 1}) as mocked:
            api_key_client._invoke_manifest_command(invoke, body)
        _args, kwargs = mocked.call_args
        assert kwargs["json"] == body

    def test_get_with_empty_manifest_params_sends_query(
        self, api_key_client: ArenaClient
    ) -> None:
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/deployment-setup",
            "responseKind": "binary",
            "params": [],
        }
        query = {"name": "pool", "setupType": "helm", "archivedType": "zip"}
        fake = (b"zip", "application/zip", None)
        with patch.object(api_key_client, "_request_raw", return_value=fake) as mocked:
            api_key_client._invoke_manifest_command(invoke, query)
        _args, kwargs = mocked.call_args
        assert kwargs["params"] == query

    def test_binary_branch_uses_raw(self, api_key_client: ArenaClient) -> None:
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/deployment-setup",
            "responseKind": "binary",
            "params": [
                {
                    "name": "name",
                    "click": {"option": ["--name"], "metavar": "STR"},
                    "in": "query",
                    "type": "string",
                    "required": True,
                    "help": "",
                }
            ],
        }
        fake = (b"hello", "application/octet-stream", None)
        with patch.object(api_key_client, "_request_raw", return_value=fake) as mocked:
            out = api_key_client._invoke_manifest_command(invoke, {"name": "my-pool"})
        assert out == fake
        mocked.assert_called_once()
        _args, kwargs = mocked.call_args
        assert kwargs["params"] == {"name": "my-pool"}


class TestBuildManifestClickCommand:
    def test_help_registers_options(self) -> None:
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/get",
            "responseKind": "json",
            "params": [
                {
                    "name": "name",
                    "click": {"option": ["--name"], "metavar": "STR"},
                    "in": "query",
                    "type": "string",
                    "required": True,
                    "help": "Class name",
                }
            ],
        }
        cmd = build_manifest_click_command("get", "Fetch class", invoke)
        runner = CliRunner()
        result = runner.invoke(cmd, ["--help"], obj=_command_config())
        assert result.exit_code == 0
        assert "--name" in result.output


class TestWriteBinaryAtomic:
    def test_writes_via_replace(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        write_binary_atomic(dest, b"abc", force=False)
        assert dest.read_bytes() == b"abc"

    def test_refuses_existing_without_force(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        dest.write_bytes(b"old")
        with pytest.raises(click.ClickException, match="Refusing to overwrite"):
            write_binary_atomic(dest, b"new", force=False)
        assert dest.read_bytes() == b"old"

    def test_force_overwrites(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.bin"
        dest.write_bytes(b"old")
        write_binary_atomic(dest, b"new", force=True)
        assert dest.read_bytes() == b"new"


class TestRegisterOnPremManifestGroup:
    def test_registers_named_subgroup(self) -> None:
        @click.group()
        def root() -> None:
            pass

        register_on_prem_manifest_group(root)
        names = [n for n, _c in root.commands.items()]
        assert "on-prem" in names


class TestArenaRootGroupVisibility:
    @staticmethod
    def _root_with_on_prem() -> click.Group:
        @click.group(cls=ArenaRootGroup)
        def root() -> None:
            """Arena root."""

        register_on_prem_manifest_group(root)
        return root

    def test_help_lists_on_prem_when_enterprise(self) -> None:
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            return_value=True,
        ):
            r = CliRunner().invoke(
                root,
                ["--help"],
                obj=_command_config(),
            )
        assert r.exit_code == 0
        assert "on-prem" in r.output

    def test_help_hides_on_prem_when_not_enterprise(self) -> None:
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            return_value=False,
        ):
            r = CliRunner().invoke(
                root,
                ["--help"],
                obj=_command_config(),
            )
        assert r.exit_code == 0
        assert "on-prem" not in r.output

    def test_help_hides_on_prem_when_caps_unavailable(self) -> None:
        root = self._root_with_on_prem()
        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            return_value=None,
        ):
            r = CliRunner().invoke(
                root,
                ["--help"],
                obj=_command_config(),
            )
        assert r.exit_code == 0
        assert "on-prem" not in r.output

    def test_main_help_uses_argv_before_callback_for_capabilities(self) -> None:
        """Eager ``--help`` runs before ``main`` sets ``ctx.obj``; config must come from params."""
        from agilerl.arena.cli import main

        captured: dict[str, object] = {}

        def capture(cfg: CommandConfig) -> bool:
            captured["cfg"] = cfg
            return True

        with patch(
            "agilerl.arena.cli_manifest.capabilities_show_on_prem_root",
            side_effect=capture,
        ):
            r = CliRunner().invoke(
                main,
                [
                    "--base-url",
                    "http://localhost:3001",
                    "--keycloak-url",
                    "http://localhost:8023",
                    "--api-key",
                    "arena_pat_testtoken",
                    "--help",
                ],
            )
        assert r.exit_code == 0
        cfg = captured["cfg"]
        assert isinstance(cfg, CommandConfig)
        assert cfg.api_key == "arena_pat_testtoken"
        assert cfg.base_url == "http://localhost:3001"
        assert cfg.keycloak_url == "http://localhost:8023"
        assert "on-prem" in r.output


CAP_FIXTURE_V2 = {
    "schemaVersion": 1,
    "enterprise": True,
    "features": {"onPremCli": True},
    "cli": {
        "manifestSchemaVersion": 2,
        "root": {
            "type": "group",
            "name": "on-prem",
            "help": "root",
            "children": [
                {
                    "type": "group",
                    "name": "providers",
                    "help": "providers",
                    "children": [
                        {
                            "type": "command",
                            "name": "get",
                            "help": "Get provider",
                            "invoke": {
                                "method": "GET",
                                "path": "/api/cli/v1/on-prem/provider",
                                "responseKind": "json",
                                "params": [],
                            },
                        }
                    ],
                },
                {
                    "type": "group",
                    "name": "classes",
                    "help": "classes",
                    "children": [],
                },
                {
                    "type": "group",
                    "name": "install",
                    "help": "install",
                    "children": [],
                },
            ],
        },
    },
}


class TestOnPremDynamicIntegration:
    def test_lazy_group_loads_fixture_manifest(self) -> None:
        @click.group()
        def root() -> None:
            """root"""

        register_on_prem_manifest_group(root)

        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = CAP_FIXTURE_V2

        build_patch = patch(
            "agilerl.arena.cli_manifest.build_client",
            return_value=client_mock,
        )

        runner = CliRunner()
        with build_patch:
            res = runner.invoke(
                root,
                ["on-prem", "providers", "get", "--help"],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        assert client_mock._get_cli_capabilities.call_count >= 1
        assert client_mock.close.call_count >= 1

    def test_on_prem_install_command_help(self) -> None:
        @click.group()
        def root() -> None:
            """root"""

        register_on_prem_manifest_group(root)
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = CAP_FIXTURE_V2

        with patch(
            "agilerl.arena.cli_manifest.build_client",
            return_value=client_mock,
        ):
            res = CliRunner().invoke(
                root,
                ["on-prem", "install", "--help"],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        assert "--manager" in res.output
        assert "helm" in res.output.lower()
        assert "install-docker" in res.output

    def test_entitlement_failure_shows_user_friendly_notice(self) -> None:
        @click.group()
        def root() -> None:
            """root"""

        register_on_prem_manifest_group(root)
        client_mock = MagicMock(spec=ArenaClient)
        client_mock._get_cli_capabilities.return_value = {
            "schemaVersion": 1,
            "enterprise": False,
            "features": {"onPremCli": False},
            "cli": {"manifestSchemaVersion": 2, "root": {}},
        }

        with patch(
            "agilerl.arena.cli_manifest.build_client",
            return_value=client_mock,
        ):
            res = CliRunner().invoke(
                root,
                ["on-prem", "help"],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        assert "arena user profile" in res.output
        assert "/api/cli/v1/capabilities" not in res.output


class TestHelmReleaseIds:
    def test_parse_cluster_name_from_values(self, tmp_path: Path) -> None:
        from agilerl.arena.cli_on_prem_install import _parse_helm_release_ids

        root = tmp_path / "bundle"
        (root / "chart").mkdir(parents=True)
        (root / "chart" / "values.yaml").write_text(
            'clusterName: "my-k3d-pool"\n',
            encoding="utf-8",
        )
        release, namespace = _parse_helm_release_ids(root)
        assert release == "my-k3d-pool"
        assert namespace == "my-k3d-pool"


class TestResolveBundleRoot:
    def test_finds_setup_sh_when_zip_has_arena_train_prefix(
        self, tmp_path: Path
    ) -> None:
        from agilerl.arena.cli_on_prem_install import resolve_bundle_root

        nested = tmp_path / "extracted" / "arena-train"
        nested.mkdir(parents=True)
        (nested / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")

        assert resolve_bundle_root(tmp_path / "extracted") == nested

    def test_finds_setup_sh_at_extract_root(self, tmp_path: Path) -> None:
        from agilerl.arena.cli_on_prem_install import resolve_bundle_root

        root = tmp_path / "extracted"
        root.mkdir()
        (root / "setup.sh").write_text("#!/bin/sh\n", encoding="utf-8")

        assert resolve_bundle_root(root) == root


class TestOnPremInstall:
    def test_ensure_class_reuses_existing_by_name(self) -> None:
        from agilerl.arena.cli_on_prem_install import _ensure_class

        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.return_value = [{"name": "pool", "id": 9}]

        row = _ensure_class(client, name="pool", num_nodes=2)
        assert row["name"] == "pool"
        client._invoke_manifest_command.assert_called_once()

    def test_install_flow_uses_name_in_bundle_query(self) -> None:
        from agilerl.arena.cli_on_prem_install import run_on_prem_install

        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            {},
            [{"name": "pool", "id": 9}],
            [{"name": "pool", "id": 9}],
        ]

        with (
            patch(
                "agilerl.arena.cli_on_prem_install.shutil.which",
                return_value="/usr/bin/ssh",
            ),
            patch(
                "agilerl.arena.cli_on_prem_install._download_bundle",
                return_value=Path("/tmp/fake-bundle"),
            ) as download_mock,
            patch("agilerl.arena.cli_on_prem_install._validate_wireguard_bundle"),
            patch(
                "agilerl.arena.cli_on_prem_install._run_docker_swarm_install",
            ) as swarm_mock,
            patch("agilerl.arena.cli_on_prem_install._verify_swarm_stack"),
        ):
            run_on_prem_install(
                client,
                name="pool",
                manager="10.0.0.1",
                workers=("10.0.0.2",),
                setup_type="dockerSwarm",
                ssh_user="ubuntu",
                ssh_extra_opts=None,
                advertise_addr=None,
                skip_enable=False,
            )

        download_mock.assert_called_once()
        assert download_mock.call_args.kwargs["class_name"] == "pool"
        assert download_mock.call_args.kwargs["setup_type"] == "dockerSwarm"
        swarm_mock.assert_called_once()

    def test_helm_install_does_not_require_manager(self) -> None:
        from agilerl.arena.cli_on_prem_install import run_on_prem_install

        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            {},
            [{"name": "k8s-pool", "num_nodes": 3}],
            [{"name": "k8s-pool", "num_nodes": 3}],
        ]

        with (
            patch(
                "agilerl.arena.cli_on_prem_install._download_bundle",
                return_value=Path("/tmp/fake-helm-bundle"),
            ),
            patch("agilerl.arena.cli_on_prem_install._validate_wireguard_bundle"),
            patch(
                "agilerl.arena.cli_on_prem_install._wait_for_gateway_peer_registration",
            ),
            patch("agilerl.arena.cli_on_prem_install._run_helm_install") as helm_mock,
        ):
            run_on_prem_install(
                client,
                name="k8s-pool",
                setup_type="helm",
                skip_enable=False,
            )

        helm_mock.assert_called_once()
        assert client._invoke_manifest_command.call_count == 3

    def test_docker_swarm_requires_manager(self) -> None:
        from agilerl.arena.cli_on_prem_install import run_on_prem_install

        client = MagicMock(spec=ArenaClient)
        with pytest.raises(click.ClickException, match="--manager"):
            run_on_prem_install(
                client,
                name="pool",
                setup_type="dockerSwarm",
                skip_enable=True,
            )

    def test_teardown_helm_uninstalls_and_deletes_class(self) -> None:
        from agilerl.arena.cli_on_prem_install import run_on_prem_teardown

        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.side_effect = [
            [{"name": "k8s-pool", "id": 1}],
            {},
        ]

        with (
            patch(
                "agilerl.arena.cli_on_prem_install._download_bundle",
                return_value=Path("/tmp/fake"),
            ),
            patch(
                "agilerl.arena.cli_on_prem_install._parse_helm_release_ids",
                return_value=("k8s-pool", "k8s-pool"),
            ),
            patch("agilerl.arena.cli_on_prem_install._helm_uninstall") as helm_mock,
        ):
            run_on_prem_teardown(
                client,
                name="k8s-pool",
                setup_type="helm",
                skip_cluster=False,
                delete_class=True,
                disable_provider=False,
            )

        helm_mock.assert_called_once_with("k8s-pool", "k8s-pool")
        delete_call = [
            c
            for c in client._invoke_manifest_command.call_args_list
            if c[0][0].get("path", "").endswith("/classes/delete")
        ]
        assert len(delete_call) == 1
        assert delete_call[0][0][1] == {"name": "k8s-pool"}


def _param_spec(
    name: str = "x",
    *,
    in_: str = "query",
    type_: str = "string",
    required: bool = False,
    option: list[str] | None = None,
) -> dict[str, object]:
    """Build a manifest param spec for option/partition tests."""
    return {
        "name": name,
        "in": in_,
        "type": type_,
        "required": required,
        "help": "h",
        "click": {"option": option or [f"--{name}"]},
    }


class TestManifestSpecToClickOption:
    @staticmethod
    def _option(spec: dict[str, object]) -> click.Option:
        decorator = _manifest_spec_to_click_option(spec)  # type: ignore[arg-type]

        @decorator
        def f(**_kw: object) -> None: ...

        return f.__click_params__[0]  # type: ignore[attr-defined,no-any-return]

    @pytest.mark.parametrize(
        ("spec", "type_name"),
        [
            (_param_spec("count", type_="int"), "integer"),
            (_param_spec("body", type_="json"), "text"),
            (_param_spec("name", type_="string"), "text"),
        ],
    )
    def test_scalar_types_map_to_click_types(
        self, spec: dict[str, object], type_name: str
    ) -> None:
        assert self._option(spec).type.name == type_name

    def test_client_bool_becomes_flag(self) -> None:
        opt = self._option(_param_spec("verbose", in_="client", type_="bool"))
        assert opt.is_flag is True

    def test_optional_body_bool_becomes_toggle_pair(self) -> None:
        opt = self._option(_param_spec("enabled", in_="body", type_="bool"))
        assert opt.is_bool_flag is True
        assert any("--no-enabled" in s for s in opt.secondary_opts)

    def test_required_is_propagated(self) -> None:
        assert self._option(_param_spec("name", required=True)).required is True

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(ArenaValidationError):
            _manifest_spec_to_click_option(_param_spec("x", type_="float"))  # type: ignore[arg-type]


class TestPartitionManifestArgs:
    def test_splits_query_and_body(self, api_key_client: ArenaClient) -> None:
        query, body = api_key_client._partition_manifest_args(
            method="POST",
            params_list=[
                _param_spec("a", in_="query", required=True),
                _param_spec("b", in_="body", required=True),
            ],
            parsed_args={"a": "x", "b": "y"},
        )
        assert query == {"a": "x"}
        assert body == {"b": "y"}

    def test_client_params_are_not_sent(self, api_key_client: ArenaClient) -> None:
        query, body = api_key_client._partition_manifest_args(
            method="POST",
            params_list=[_param_spec("v", in_="client", type_="bool")],
            parsed_args={"v": True},
        )
        assert query == {}
        assert body is None

    def test_missing_required_raises(self, api_key_client: ArenaClient) -> None:
        with pytest.raises(ArenaValidationError, match="Missing required"):
            api_key_client._partition_manifest_args(
                method="POST",
                params_list=[_param_spec("a", in_="query", required=True)],
                parsed_args={},
            )

    @pytest.mark.parametrize(
        ("method", "expected_query", "expected_body"),
        [
            ("GET", {"name": "p"}, None),
            ("POST", {}, {"name": "p"}),
        ],
    )
    def test_paramless_invoke_routed_by_method(
        self,
        api_key_client: ArenaClient,
        method: str,
        expected_query: dict[str, object],
        expected_body: dict[str, object] | None,
    ) -> None:
        query, body = api_key_client._partition_manifest_args(
            method=method,
            params_list=[],
            parsed_args={"name": "p"},
        )
        assert query == expected_query
        assert body == expected_body


class TestValidateManifestInvokeErrors:
    @pytest.mark.parametrize(
        "invoke",
        [
            {
                "method": "OPTIONS",
                "path": "/api/cli/v1/on-prem/x",
                "responseKind": "json",
            },
            {"method": "GET", "path": "/api/cli/v1/on-prem/x", "responseKind": "text"},
            {"method": "GET", "path": "/api/evil", "responseKind": "json"},
            {
                "method": "GET",
                "path": "/api/cli/v1/on-prem/../x",
                "responseKind": "json",
            },
            {
                "method": "GET",
                "path": "/api/cli/v1/on-prem/x",
                "responseKind": "json",
                "params": [{"name": "p", "in": "header", "type": "string"}],
            },
            {
                "method": "GET",
                "path": "/api/cli/v1/on-prem/x",
                "responseKind": "json",
                "params": [{"name": "p", "in": "query", "type": "float"}],
            },
        ],
    )
    def test_rejects_malformed_invoke(
        self, api_key_client: ArenaClient, invoke: dict[str, object]
    ) -> None:
        with pytest.raises(ArenaValidationError):
            api_key_client._validate_manifest_invoke(invoke)  # type: ignore[arg-type]


class TestManifestCommandCallback:
    def test_json_command_forwards_parsed_args(self) -> None:
        invoke = {
            "method": "POST",
            "path": "/api/cli/v1/on-prem/classes/create",
            "responseKind": "json",
            "params": [_param_spec("name", in_="body", required=True)],
        }
        cmd = build_manifest_click_command("create", "help", invoke)
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.return_value = {"id": 1}
        ctx_mgr = MagicMock()
        ctx_mgr.__enter__.return_value = client
        ctx_mgr.__exit__.return_value = False

        with patch("agilerl.arena.cli.arena_client", return_value=ctx_mgr):
            result = CliRunner().invoke(cmd, ["--name", "pool"], obj=_command_config())

        assert result.exit_code == 0
        client._invoke_manifest_command.assert_called_once()
        _invoke, parsed = client._invoke_manifest_command.call_args.args
        assert parsed == {"name": "pool"}

    def test_binary_command_writes_output_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "bundle.zip"
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/deployment-setup",
            "responseKind": "binary",
            "params": [
                _param_spec("name", in_="query", required=True),
                _param_spec("outputPath", in_="client", option=["--output-path"]),
            ],
        }
        cmd = build_manifest_click_command("download", "help", invoke)
        client = MagicMock(spec=ArenaClient)
        client._invoke_manifest_command.return_value = (
            b"data",
            "application/zip",
            None,
        )
        ctx_mgr = MagicMock()
        ctx_mgr.__enter__.return_value = client
        ctx_mgr.__exit__.return_value = False

        with patch("agilerl.arena.cli.arena_client", return_value=ctx_mgr):
            result = CliRunner().invoke(
                cmd,
                ["--name", "pool", "--output-path", str(dest)],
                obj=_command_config(),
            )

        assert result.exit_code == 0
        assert dest.read_bytes() == b"data"


def _invoke_on_prem_notice(caps: dict[str, object] | None) -> str:
    """Render ``arena on-prem help`` for a given capabilities payload."""

    @click.group()
    def root() -> None:
        """root"""

    register_on_prem_manifest_group(root)
    client_mock = MagicMock(spec=ArenaClient)
    client_mock._get_cli_capabilities.return_value = caps

    with patch(
        "agilerl.arena.cli_manifest.build_client",
        return_value=client_mock,
    ):
        result = CliRunner().invoke(root, ["on-prem", "help"], obj=_command_config())
    assert result.exit_code == 0
    return result.output


class TestOnPremDynamicNotices:
    @pytest.mark.parametrize(
        ("caps", "expected"),
        [
            (None, "not available from this Arena server"),
            ({"schemaVersion": 999}, "does not support on-prem CLI"),
            (
                {
                    "schemaVersion": 1,
                    "enterprise": False,
                    "features": {"onPremCli": False},
                },
                "not enabled for your account",
            ),
            (
                {"schemaVersion": 1, "enterprise": True, "cli": None},
                "temporarily unavailable",
            ),
            (
                {
                    "schemaVersion": 1,
                    "enterprise": True,
                    "cli": {"manifestSchemaVersion": 1},
                },
                "too old",
            ),
            (
                {
                    "schemaVersion": 1,
                    "enterprise": True,
                    "cli": {"manifestSchemaVersion": 2},
                },
                "configuration from Arena is invalid",
            ),
        ],
    )
    def test_unavailable_capabilities_show_friendly_notice(
        self, caps: dict[str, object] | None, expected: str
    ) -> None:
        output = _invoke_on_prem_notice(caps)
        assert expected in output
        assert "/api/" not in output  # no backend endpoints leak to users
