# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Local connections, the setup.json profile, and connection checks."""

from __future__ import annotations

import errno
import os
import re
import sys
from pathlib import Path

import pytest

from agilerl.arena.snowflake.profile import (
    GPU_FAMILIES,
    NO_POOLS,
    admin_connection_fields,
    admin_connection_rejection,
    browser_admin_rejection,
    checked_job_connection,
    job_connection_fields,
    job_connection_name,
    missing_key_rejection,
    profile_from_mapping,
    remove_connection,
    replaced_connection,
    restrict_private_key,
    write_connection,
    write_private_text,
)
from agilerl.arena.snowflake.session import connection_section
from tests.test_snowflake_setup import ACCOUNT, PROFILE_DOC, mode, paths


class TestAdminConnectionFields:
    def test_snowflake_oauth_without_a_role(self) -> None:
        assert admin_connection_fields(ACCOUNT, "NICK") == {
            "account": ACCOUNT,
            "user": "NICK",
            "authenticator": "OAUTH_AUTHORIZATION_CODE",
        }


class TestRemoveConnection:
    def test_removes_one_section_and_keeps_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))
        write_connection("mine", {"account": "X", "user": "ME"})

        removed = remove_connection("arena")

        assert removed is True
        assert connection_section("arena") is None
        assert connection_section("mine") == {"account": "X", "user": "ME"}
        if sys.platform != "win32":
            assert mode(connections) == 0o600

    def test_a_missing_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)

        assert remove_connection("arena") is False

    def test_a_file_without_the_section_is_left_as_it_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, _key = paths(tmp_path, monkeypatch)
        write_connection("mine", {"account": "X", "user": "ME"})
        before = connections.read_text(encoding="utf-8")

        removed = remove_connection("arena")

        assert removed is False
        assert connections.read_text(encoding="utf-8") == before


class TestReplacedConnection:
    def test_warns_about_a_different_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        line = replaced_connection("arena", {"account": ACCOUNT, "user": "Ada"})

        assert line == (
            f"[arena] in {connections} signs in as arena_job_user on {ACCOUNT}. "
            "It is replaced."
        )

    def test_an_identical_or_absent_section_is_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        fields = job_connection_fields(ACCOUNT, key)

        absent = replaced_connection("arena", fields)
        write_connection("arena", fields)

        assert absent is None
        assert replaced_connection("arena", fields) is None


class TestJobConnectionName:
    def test_uses_arena_when_connections_toml_has_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        assert job_connection_name(None) == "arena"

    def test_an_explicit_name_wins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        assert job_connection_name("mine") == "mine"

    def test_without_arena_uses_the_snowflake_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)

        assert job_connection_name(None) is None


class TestCheckedJobConnection:
    def test_returns_arena_with_its_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        key.write_text("key", encoding="utf-8")
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        assert checked_job_connection(None) == "arena"

    def test_refuses_the_browser_admin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))

        with pytest.raises(ValueError, match="signs in with a browser"):
            checked_job_connection("arena-admin")

    def test_refuses_a_missing_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        with pytest.raises(ValueError, match=re.escape(f"No private key at {key}")):
            checked_job_connection(None)


class TestProfileFromMapping:
    def test_reads_pools_and_integrations(self) -> None:
        profile = profile_from_mapping(PROFILE_DOC)

        assert profile.pools.broker.pool == "arena_pulsar_pool"
        assert profile.pools.gpu["GPU_NV_S"].gpus_per_node == 1
        assert profile.external_access_integrations == ("arena_hf_eai",)

    @pytest.mark.parametrize(
        "pools",
        [
            None,
            {"broker": {"pool": "b", "family": "CPU_X64_L"}},
            {**PROFILE_DOC["pools"], "gpu": {"GPU_NV_S": "arena_gpu_GPU_NV_S"}},
        ],
    )
    def test_without_pools_says_to_rerun_setup(self, pools: object) -> None:
        section = {**PROFILE_DOC, "pools": pools}

        with pytest.raises(ValueError, match="no compute pools") as raised:
            profile_from_mapping(section)

        assert str(raised.value) == NO_POOLS


class TestGpuFamilies:
    @pytest.mark.parametrize(
        ("family", "gpus", "gpu"),
        [
            ("GPU_NV_S", 1, "A10G"),
            ("GPU_NV_SM", 1, "A10"),
            ("GPU_NV_SL", 4, "A100"),
            ("GPU_GCP_NV_L4_1_24G", 1, "L4"),
            ("GPU_GCP_NV_A100_8_40G", 8, "A100"),
        ],
    )
    def test_covers_aws_azure_and_gcp(self, family: str, gpus: int, gpu: str) -> None:
        spec = GPU_FAMILIES[family]

        assert spec.gpus == gpus
        assert spec.gpu == gpu


class TestBrowserAdminRejection:
    def test_rejects_the_admin_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))

        message = browser_admin_rejection("arena-admin")

        assert message == (
            "Connection arena-admin signs in with a browser as ACCOUNTADMIN.\n"
            "Job commands use the arena connection (user arena_job_user)."
        )

    def test_rejects_any_oauth_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("mine", admin_connection_fields(ACCOUNT, "NICK"))

        assert browser_admin_rejection("mine") is not None

    def test_allows_the_job_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        assert browser_admin_rejection("arena") is None


class TestAdminConnectionRejection:
    def test_rejects_the_service_user(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        message = admin_connection_rejection("arena")

        assert message == (
            "Connection arena signs in as user arena_job_user.\n"
            "Admin commands use the arena-admin connection, or --connection "
            "with a user that has ACCOUNTADMIN."
        )

    def test_rejects_a_data_scientist_on_arena_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection(
            "ada", {"account": ACCOUNT, "user": "ADA", "role": "arena_trainer"}
        )

        assert admin_connection_rejection("ada") is not None

    def test_allows_an_admin_login(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))

        assert admin_connection_rejection("arena-admin") is None


class TestMissingKeyRejection:
    def test_a_missing_key_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        gone = tmp_path / "gone.p8"
        write_connection("arena", job_connection_fields(ACCOUNT, gone))

        message = missing_key_rejection("arena")

        assert message == (
            f"No private key at {gone}.\n"
            "The private key is not in Snowflake.\n"
            "Run arena snowflake join to sign in as your own Snowflake user."
        )

    def test_a_present_key_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _connections, key = paths(tmp_path, monkeypatch)
        key.write_text("key", encoding="utf-8")
        write_connection("arena", job_connection_fields(ACCOUNT, key))

        assert missing_key_rejection("arena") is None

    def test_a_connection_without_a_key_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        paths(tmp_path, monkeypatch)
        write_connection("arena-admin", admin_connection_fields(ACCOUNT, "NICK"))

        assert missing_key_rejection("arena-admin") is None


def failing_chmod(monkeypatch: pytest.MonkeyPatch, target: Path, code: int) -> None:
    """Make ``os.chmod`` on ``target`` fail with ``code``, as a full or read-only disk does."""
    real_chmod = os.chmod

    def chmod(path: str | os.PathLike[str], mode: int) -> None:
        if Path(path) == target:
            raise OSError(code, os.strerror(code), str(path))
        real_chmod(path, mode)

    monkeypatch.setattr(os, "chmod", chmod)


class TestWritePrivateText:
    def test_writes_an_owner_only_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "connections.toml"

        write_private_text(path, "[arena]\n")

        assert path.read_text(encoding="utf-8") == "[arena]\n"
        if sys.platform != "win32":
            assert mode(path) == 0o600

    def test_a_full_disk_is_a_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "connections.toml"
        failing_chmod(monkeypatch, path, errno.ENOSPC)

        with pytest.raises(ValueError, match="out of space") as raised:
            write_private_text(path, "[arena]\n")

        assert str(raised.value) == (
            f"The disk is out of space, so {path} was not written."
        )

    def test_other_os_errors_propagate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "connections.toml"
        failing_chmod(monkeypatch, path, errno.EACCES)

        with pytest.raises(PermissionError):
            write_private_text(path, "[arena]\n")


class TestRestrictPrivateKey:
    def test_leaves_the_key_owner_only(self, tmp_path: Path) -> None:
        key = tmp_path / "arena_job_user.p8"
        key.write_text("key", encoding="utf-8")
        key.chmod(0o644)

        restrict_private_key(key)

        assert key.read_text(encoding="utf-8") == "key"
        if sys.platform != "win32":
            assert mode(key) == 0o600

    def test_a_read_only_disk_is_a_value_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key = tmp_path / "arena_job_user.p8"
        key.write_text("key", encoding="utf-8")
        failing_chmod(monkeypatch, key, errno.EROFS)

        with pytest.raises(ValueError, match="read-only") as raised:
            restrict_private_key(key)

        assert str(raised.value) == f"The disk is read-only, so {key} was not written."

    def test_other_os_errors_propagate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key = tmp_path / "arena_job_user.p8"
        key.write_text("key", encoding="utf-8")
        failing_chmod(monkeypatch, key, errno.EACCES)

        with pytest.raises(PermissionError):
            restrict_private_key(key)
