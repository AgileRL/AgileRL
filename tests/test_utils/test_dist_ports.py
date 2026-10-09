# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for xdist-safe MASTER_PORT allocation."""

from __future__ import annotations

import socket

from tests.dist_ports import get_free_port, gloo_rank_env, gloo_socket_ifname


class TestGetFreePort:
    def test_returns_an_int_port(self):
        port = get_free_port()

        assert isinstance(port, int)
        assert 0 < port < 65536

    def test_successive_calls_return_distinct_ports(self):
        first = get_free_port()
        second = get_free_port()

        assert first != second

    def test_port_is_bindable_after_return(self):
        port = get_free_port()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
            assert sock.getsockname()[1] == port


class TestGlooSocketIfname:
    def test_linux_uses_lo(self, monkeypatch):
        monkeypatch.setattr("tests.dist_ports.sys.platform", "linux")

        assert gloo_socket_ifname() == "lo"

    def test_darwin_uses_lo0(self, monkeypatch):
        monkeypatch.setattr("tests.dist_ports.sys.platform", "darwin")

        assert gloo_socket_ifname() == "lo0"

    def test_windows_does_not_pin_lo(self, monkeypatch):
        monkeypatch.setattr("tests.dist_ports.sys.platform", "win32")

        assert gloo_socket_ifname() is None
        assert "GLOO_SOCKET_IFNAME" not in gloo_rank_env(0, 2, 12345)
