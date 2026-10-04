# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import agilerl


@pytest.mark.parametrize(("package", "extra"), [("agilerl", "llm")])
def test_get_extra_dependencies(package, extra):
    result = agilerl.get_extra_dependencies(package, extra)
    assert isinstance(result, list)
    assert all(isinstance(d, str) for d in result)


def test_llm_packages_and_has_llm_dependencies():
    _ = agilerl.LLM_PACKAGES
    _ = agilerl.HAS_LLM_DEPENDENCIES


def test_agent_type_reexports_arena_registry():
    from agilerl.arena.models.registry import AgentType as ArenaAgentType

    assert agilerl.AgentType is ArenaAgentType


def test_agent_type_loads_after_extend_path():
    source = Path(agilerl.__file__).read_text(encoding="utf-8")
    extend_at = source.index("__path__ = extend_path")
    attach_at = source.index("lazy.attach")
    assert extend_at < attach_at
    assert "AgentType" in source[attach_at:]


class TestIsDistributionInstalled:
    def test_is_distribution_installed_found(self):
        assert agilerl._is_distribution_installed("agilerl") is True

    def test_is_distribution_installed_not_found(self):
        assert (
            agilerl._is_distribution_installed("_nonexistent_pkg_xyz_12345_") is False
        )
