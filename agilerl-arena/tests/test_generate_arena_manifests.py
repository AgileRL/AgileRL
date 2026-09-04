# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HELPER_PATH = Path(__file__).with_name("generate_arena_manifests.py")
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "generate_arena_manifests", _HELPER_PATH
)
assert _HELPER_SPEC is not None
assert _HELPER_SPEC.loader is not None
generate_arena_manifests = importlib.util.module_from_spec(_HELPER_SPEC)
sys.modules["generate_arena_manifests"] = generate_arena_manifests
_HELPER_SPEC.loader.exec_module(generate_arena_manifests)

_attach_default_network = generate_arena_manifests._attach_default_network
_default_algorithm_spec = generate_arena_manifests._default_algorithm_spec
write_arena_manifest = generate_arena_manifests.write_arena_manifest


@pytest.mark.parametrize("algo_name", ["DQN", "IPPO", "MADDPG", "MATD3"])
def test_rl_and_multi_agent_manifests_include_network(
    tmp_path: Path, algo_name: str
) -> None:
    _path, payload = write_arena_manifest(algo_name, tmp_path)
    assert payload["network"] is not None


def test_attach_default_network_keeps_existing_multi_agent_net_config() -> None:
    algorithm = _default_algorithm_spec("IPPO")
    _attach_default_network(algorithm)
    existing = algorithm.net_config
    _attach_default_network(algorithm)
    assert algorithm.net_config is existing
