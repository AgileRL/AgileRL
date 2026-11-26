# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``arena secrets`` commands and their ArenaClient methods."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from unittest.mock import patch

import httpx
import pytest
from click.testing import CliRunner

from agilerl.arena.cli import main
from agilerl.arena.cli_secrets import _stdin_is_interactive

SECRETS_PATH = "/api/cli/v1/secrets"
ROW = {
    "name": "ANTHROPIC_API_KEY",
    "last_four": "wxyz",
    "created_at": "2026-11-10T09:00:00Z",
    "updated_at": "2026-11-11T09:00:00Z",
    "updated_by": "ada@example.com",
}
# What the CLI may show: the server's row without anything derived from the value.
SHOWN_ROW = {k: v for k, v in ROW.items() if k != "last_four"}


class FakeSecretsServer:
    """In-memory stand-in for the platform's org secrets endpoint."""

    def __init__(self, rows: list[dict[str, str]] | None = None) -> None:
        self.rows = {row["name"]: dict(row) for row in rows or []}
        self.requests: list[httpx.Request] = []
        self.reject: tuple[int, str] | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.path == SECRETS_PATH
        if self.reject is not None:
            status, message = self.reject
            return httpx.Response(status, json={"error": message})
        if request.method == "GET":
            return httpx.Response(
                200, json={"ok": True, "data": list(self.rows.values())}
            )
        body = json.loads(request.content)
        if request.method == "PUT":
            row = {
                "name": body["name"],
                "last_four": body["value"][-4:],
                "created_at": "2026-11-11T10:00:00Z",
                "updated_at": "2026-11-11T10:00:00Z",
                "updated_by": "grace@example.com",
            }
            self.rows[body["name"]] = row
            return httpx.Response(200, json={"ok": True, "data": row})
        if request.method == "DELETE":
            if self.rows.pop(body["name"], None) is None:
                return httpx.Response(404, json={"error": "secret not found"})
            return httpx.Response(200, json={"deleted": True, "name": body["name"]})
        return httpx.Response(405)


@pytest.fixture
def server() -> FakeSecretsServer:
    return FakeSecretsServer([ROW])


@pytest.fixture
def run(server: FakeSecretsServer) -> Callable[..., object]:
    """Invoke the real CLI against the fake server."""
    real_client = httpx.Client

    def fake_client(**kwargs: object) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(server.handler), **kwargs)

    def _run(
        *args: str,
        stdin: str | None = None,
        interactive: bool = False,
        root_args: tuple[str, ...] = (),
    ):
        with (
            patch("agilerl.arena.auth.KeycloakOpenID"),
            patch("agilerl.arena.client.httpx.Client", fake_client),
            patch(
                "agilerl.arena.cli_secrets._stdin_is_interactive",
                return_value=interactive,
            ),
        ):
            return CliRunner().invoke(
                main,
                [*root_args, "--api-key", "test-key", "secrets", *args],
                input=stdin,
            )

    return _run


class TestSecretsList:
    def test_prints_names_timestamps_and_editor_only(self, run):
        result = run("list")

        assert result.exit_code == 0, result.output
        assert "ANTHROPIC_API_KEY" in result.output
        assert "2026-11-10T09:00:00Z" in result.output
        assert "Updated by" in result.output
        assert "ada@example.com" in result.output
        assert "wxyz" not in result.output
        assert "Last four" not in result.output

    def test_json_returns_rows(self, run):
        result = run("list", root_args=("--json",))

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == [SHOWN_ROW]

    def test_empty(self, server, run):
        server.rows.clear()

        result = run("list")

        assert result.exit_code == 0, result.output
        assert "No secrets." in result.output


class TestSecretsSet:
    def test_reads_piped_stdin(self, server, run):
        result = run("set", "JUDGE_KEY", stdin="dummy-value-1234\n")

        assert result.exit_code == 0, result.output
        sent = json.loads(server.requests[-1].content)
        assert sent == {"name": "JUDGE_KEY", "value": "dummy-value-1234"}
        assert result.output.strip() == "Stored secret JUDGE_KEY."

    def test_json_hides_value_suffix(self, run):
        result = run(
            "set", "JUDGE_KEY", stdin="dummy-value-1234\n", root_args=("--json",)
        )

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout) == {
            "name": "JUDGE_KEY",
            "created_at": "2026-11-11T10:00:00Z",
            "updated_at": "2026-11-11T10:00:00Z",
            "updated_by": "grace@example.com",
        }

    def test_reads_from_env(self, server, run, monkeypatch):
        monkeypatch.setenv("LOCAL_JUDGE_KEY", "from-env-abcd")

        result = run("set", "JUDGE_KEY", "--from-env", "LOCAL_JUDGE_KEY")

        assert result.exit_code == 0, result.output
        assert json.loads(server.requests[-1].content)["value"] == "from-env-abcd"
        assert "abcd" not in result.output

    def test_hidden_prompt_on_terminal(self, server, run):
        result = run("set", "JUDGE_KEY", stdin="typed-secret-9876\n", interactive=True)

        assert result.exit_code == 0, result.output
        assert json.loads(server.requests[-1].content)["value"] == "typed-secret-9876"
        assert "Secret value" in result.output
        assert "9876" not in result.output

    def test_missing_env_var_is_usage_error(self, server, run, monkeypatch):
        monkeypatch.delenv("UNSET_VAR", raising=False)

        result = run("set", "JUDGE_KEY", "--from-env", "UNSET_VAR")

        assert result.exit_code == 2
        assert "UNSET_VAR is not set" in result.output
        assert server.requests == []

    def test_empty_stdin_is_usage_error(self, server, run):
        result = run("set", "JUDGE_KEY", stdin="\n")

        assert result.exit_code == 2
        assert "empty" in result.output
        assert server.requests == []

    def test_value_is_not_an_argument(self, server, run):
        result = run("set", "JUDGE_KEY", "plain-value")

        assert result.exit_code == 2
        assert server.requests == []

    def test_surfaces_server_rejection(self, server, run):
        server.reject = (400, "secret name PATH is reserved")

        result = run("set", "PATH", stdin="dummy-value\n")

        assert result.exit_code != 0
        assert "PATH is reserved" in result.output
        assert "dummy-value" not in result.output


class TestSecretsDelete:
    @pytest.mark.parametrize(
        ("args", "stdin"), [(("--yes",), None), ((), "y\n")], ids=["yes", "confirm"]
    )
    def test_deletes(self, server, run, args, stdin):
        result = run("delete", "ANTHROPIC_API_KEY", *args, stdin=stdin)

        assert result.exit_code == 0, result.output
        assert "ANTHROPIC_API_KEY" not in server.rows
        assert "Deleted secret ANTHROPIC_API_KEY." in result.stdout

    def test_confirm_declined_sends_nothing(self, server, run):
        result = run("delete", "ANTHROPIC_API_KEY", stdin="n\n")

        assert result.exit_code == 0, result.output
        assert "Aborted." in result.output
        assert server.requests == []
        assert "ANTHROPIC_API_KEY" in server.rows

    def test_declined_keeps_json_stdout_empty(self, server, run):
        result = run("delete", "ANTHROPIC_API_KEY", stdin="n\n", root_args=("--json",))

        assert result.exit_code == 0, result.output
        assert result.stdout == ""
        assert "Aborted." in result.stderr

    def test_surfaces_missing_secret(self, server, run):
        result = run("delete", "NO_SUCH_SECRET", "--yes")

        assert result.exit_code != 0
        assert "secret not found" in result.output


class TestStdinIsInteractive:
    def test_false_for_piped_stdin(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", io.StringIO("piped"))

        assert _stdin_is_interactive() is False
