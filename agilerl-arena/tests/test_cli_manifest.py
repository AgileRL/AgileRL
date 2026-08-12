# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic manifest -> Click machinery and client dispatch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from agilerl.arena.cli_manifest import (
    _manifest_spec_to_click_option,
    _parse_json_cli_value,
    attach_manifest_tree,
    build_manifest_click_command,
    pythonize_manifest_param_name,
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


def _client_context(client: MagicMock) -> MagicMock:
    """A context manager (like ``arena_client``) that yields *client*."""
    cm = MagicMock()
    cm.__enter__.return_value = client
    cm.__exit__.return_value = False
    return cm


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


class TestPythonizeManifestParamName:
    def test_camel_case(self) -> None:
        assert pythonize_manifest_param_name("archivedType") == "archived_type"

    def test_snake_case(self) -> None:
        assert pythonize_manifest_param_name("num_nodes") == "num_nodes"

    def test_id_is_preserved(self) -> None:
        assert pythonize_manifest_param_name("id") == "id"


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
        resp.json.return_value = {"ok": True, "data": {"schemaVersion": 999}}
        with patch.object(api_key_client._http, "request", return_value=resp):
            assert api_key_client._get_cli_capabilities(force_refresh=True) is None

    def test_uses_bounded_timeout(self, api_key_client: ArenaClient) -> None:
        """Capability checks gate ``--help``; the request must not block on the
        full request timeout when the API is slow.
        """
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
        from agilerl.arena.on_prem.endpoints import DELETE_CLASS

        with patch.object(
            api_key_client, "_request", return_value={"ok": True}
        ) as mocked:
            api_key_client._invoke_manifest_command(DELETE_CLASS, {"name": "pool"})
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
        result = CliRunner().invoke(cmd, ["--help"], obj=_command_config())
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

    def test_cleans_up_temp_when_replace_fails(self, tmp_path: Path) -> None:
        # If os.replace fails, the .tmp file is removed before the error surfaces.
        dest = tmp_path / "out.bin"
        with patch(
            "agilerl.arena.cli_manifest.os.replace", side_effect=OSError("nope")
        ):
            with pytest.raises(OSError, match="nope"):
                write_binary_atomic(dest, b"data")
        assert not dest.with_name(dest.name + ".tmp").exists()
        assert not dest.exists()

    def test_ignores_temp_cleanup_failure(self, tmp_path: Path) -> None:
        # A failing cleanup unlink must not mask the original replace error.
        dest = tmp_path / "out.bin"
        with (
            patch("agilerl.arena.cli_manifest.os.replace", side_effect=OSError("nope")),
            patch.object(Path, "unlink", side_effect=OSError("locked")),
        ):
            with pytest.raises(OSError, match="nope"):
                write_binary_atomic(dest, b"data")


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

        with patch(
            "agilerl.arena.cli_manifest.arena_client",
            return_value=_client_context(client),
        ):
            result = CliRunner().invoke(cmd, ["--name", "pool"], obj=_command_config())

        assert result.exit_code == 0
        _invoke, parsed = client._invoke_manifest_command.call_args.args
        assert parsed == {"name": "pool"}

    def test_missing_required_param_raises_usage_error(self) -> None:
        """The callback's defensive guard rejects a required param left as None.

        Invoking the callback without parsing (``ctx.params`` empty) bypasses
        Click's own required-option check, exercising the in-callback guard.
        """
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/get",
            "responseKind": "json",
            "params": [_param_spec("name", in_="query", required=True)],
        }
        cmd = build_manifest_click_command("get", "help", invoke)
        ctx = click.Context(cmd, obj=_command_config())
        with pytest.raises(
            click.UsageError, match="Missing required option for 'name'"
        ):
            cmd.invoke(ctx)

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

        with patch(
            "agilerl.arena.cli_manifest.arena_client",
            return_value=_client_context(client),
        ):
            result = CliRunner().invoke(
                cmd,
                ["--name", "pool", "--output-path", str(dest)],
                obj=_command_config(),
            )

        assert result.exit_code == 0
        assert dest.read_bytes() == b"data"


class TestParseJsonCliValue:
    def test_parses_inline_json(self) -> None:
        assert _parse_json_cli_value('{"a": 1}') == {"a": 1}

    def test_reads_json_from_at_file(self, tmp_path: Path) -> None:
        f = tmp_path / "payload.json"
        f.write_text('{"k": [1, 2, 3]}', encoding="utf-8")
        assert _parse_json_cli_value(f"@{f}") == {"k": [1, 2, 3]}


class TestManifestCommandCallbackBinary:
    @staticmethod
    def _binary_get_command() -> click.Command:
        invoke = {
            "method": "GET",
            "path": "/api/cli/v1/on-prem/classes/get",
            "responseKind": "binary",
            "params": [
                {
                    "name": "name",
                    "click": {"option": ["--name"]},
                    "in": "query",
                    "type": "string",
                    "required": True,
                    "help": "h",
                },
                {
                    "name": "extra",
                    "click": {"option": ["--extra"]},
                    "in": "query",
                    "type": "json",
                    "required": False,
                    "help": "h",
                },
            ],
        }
        return build_manifest_click_command("get", "Fetch", invoke)

    def test_optional_arg_omitted_and_binary_echoed(self) -> None:
        client = MagicMock()
        client._invoke_manifest_command.return_value = (b"hello", None, None)
        with patch(
            "agilerl.arena.cli_manifest.arena_client",
            return_value=_client_context(client),
        ):
            res = CliRunner().invoke(
                self._binary_get_command(),
                ["--name", "pool"],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        assert "hello" in res.output
        # Optional 'extra' was omitted -> not forwarded to the server call.
        sent_args = client._invoke_manifest_command.call_args.args[1]
        assert "extra" not in sent_args

    def test_json_option_is_parsed_before_send(self) -> None:
        client = MagicMock()
        client._invoke_manifest_command.return_value = (b"ok", None, None)
        with patch(
            "agilerl.arena.cli_manifest.arena_client",
            return_value=_client_context(client),
        ):
            res = CliRunner().invoke(
                self._binary_get_command(),
                ["--name", "pool", "--extra", '{"nested": true}'],
                obj=_command_config(),
            )
        assert res.exit_code == 0
        sent_args = client._invoke_manifest_command.call_args.args[1]
        assert sent_args["extra"] == {"nested": True}


class TestArenaArchOptional:
    def test_arch_present_passes_network_raw(self) -> None:
        from agilerl.arena.models.manifest import TrainingManifest

        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "merge-env", "version": "v1"},
            "training": {"max_steps": 10_000, "evo_steps": 100, "pop_size": 1},
            "network": {
                "arch": "mlp",
                "encoder_config": {"hidden_size": [64]},
                "head_config": {"hidden_size": [64]},
            },
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        # `arch` is resolved server-side: the network section is passed through
        # untouched, with no promotion or `name` injection.
        assert out["network"] == raw["network"]

    def test_arch_absent_passes_network_raw(self) -> None:
        from agilerl.arena.models.manifest import TrainingManifest

        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "merge-env", "version": "v1"},
            "training": {"max_steps": 10_000, "evo_steps": 100, "pop_size": 1},
            "network": {"latent_dim": 64, "encoder_config": {"hidden_size": [64]}},
        }
        out = TrainingManifest.get_validated(raw, mode="json")
        # Network section is left raw for the server to validate.
        assert out["network"] == {
            "latent_dim": 64,
            "encoder_config": {"hidden_size": [64]},
        }
        assert "arch" not in out["network"]["encoder_config"]


class TestArenaSimbaRecurrentConflict:
    """``simba`` and ``recurrent`` are contradictory encoder requests.

    A network cannot simultaneously be a SimBa encoder and a recurrent
    (LSTM) encoder. Mirrors the core guard in
    ``agilerl.models.manifest.TrainingManifest._process_manifest``.
    """

    def test_simba_and_recurrent_raises(self) -> None:
        """A top-level ``simba`` flag with ``recurrent`` is a contradiction."""
        from agilerl.arena.models.manifest import TrainingManifest

        raw = {
            "algorithm": {"name": "PPO", "recurrent": True},
            "environment": {"name": "merge-env", "version": "v1"},
            "training": {"max_steps": 10_000, "evo_steps": 100, "pop_size": 1},
            "network": {"simba": True, "head_config": {"hidden_size": [64]}},
        }
        with pytest.raises(ValueError, match="cannot both be set"):
            TrainingManifest.model_validate(raw)

    def test_only_simba_validates(self) -> None:
        """Either request on its own is fine; only the combination is rejected."""
        from agilerl.arena.models.manifest import TrainingManifest

        raw = {
            "algorithm": {"name": "PPO"},
            "environment": {"name": "merge-env", "version": "v1"},
            "training": {"max_steps": 10_000, "evo_steps": 100, "pop_size": 1},
            "network": {"simba": True, "head_config": {"hidden_size": [64]}},
        }
        manifest = TrainingManifest.model_validate(raw)
        assert manifest.network.get("simba") is True


class TestAttachManifestTree:
    def test_warns_on_unknown_node_type(self) -> None:
        group = click.Group(name="root")
        node = {"children": [{"type": "mystery", "name": "weird"}]}
        with patch("agilerl.arena.cli_manifest.logger") as log:
            attach_manifest_tree(group, node)
        log.warning.assert_called_once()
        assert "weird" not in group.commands
