# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy public surface of ``agilerl.algorithms``."""

import importlib

import pytest

import agilerl
import agilerl.algorithms as algorithms

_LLM_DEPS_SKIP = pytest.mark.skipif(
    not agilerl.HAS_LLM_DEPENDENCIES, reason="agilerl[llm] not installed"
)

_LLM_ALGORITHM = "GRPO"


class TestLazyAlgorithmSurface:
    def test_dir_matches_exported_names(self):
        assert dir(algorithms) == algorithms.__all__

    def test_non_llm_algorithm_always_exported(self):
        assert "DQN" in algorithms.__all__
        assert algorithms.DQN.__name__ == "DQN"

    @_LLM_DEPS_SKIP
    def test_llm_algorithms_exported_with_extra(self):
        assert _LLM_ALGORITHM in algorithms.__all__
        assert getattr(algorithms, _LLM_ALGORITHM).__name__ == _LLM_ALGORITHM

    def test_llm_algorithms_withheld_without_extra(self, monkeypatch):
        """Without the extra the names leave ``__all__``/``dir()`` and raise."""
        # Arrange
        monkeypatch.setattr(agilerl, "HAS_LLM_DEPENDENCIES", False)

        try:
            # Act
            reloaded = importlib.reload(algorithms)

            # Assert
            assert _LLM_ALGORITHM not in reloaded.__all__
            assert _LLM_ALGORITHM not in dir(reloaded)
            assert "DQN" in reloaded.__all__
            with pytest.raises(AttributeError, match="requires the LLM extra"):
                getattr(reloaded, _LLM_ALGORITHM)
        finally:
            monkeypatch.undo()
            importlib.reload(algorithms)

    def test_unknown_name_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            _ = algorithms.NotAnAlgorithm


@_LLM_DEPS_SKIP
class TestRenamedAlgorithmAliases:
    @pytest.mark.parametrize(
        ("module_name", "old_name", "new_name"),
        [
            ("agilerl.algorithms.ppo_llm", "PPO", "LLMPPO"),
            ("agilerl.algorithms.reinforce_llm", "REINFORCE", "LLMREINFORCE"),
        ],
    )
    def test_old_name_resolves_with_deprecation_warning(
        self,
        module_name,
        old_name,
        new_name,
    ):
        # Arrange
        module = importlib.import_module(module_name)

        # Act
        with pytest.warns(DeprecationWarning, match=f"renamed to {new_name}"):
            aliased = getattr(module, old_name)

        # Assert
        assert aliased is getattr(module, new_name)

    @pytest.mark.parametrize(
        "module_name",
        ["agilerl.algorithms.ppo_llm", "agilerl.algorithms.reinforce_llm"],
    )
    def test_unknown_name_raises_attribute_error(self, module_name):
        module = importlib.import_module(module_name)

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = module.NotAnAlgorithm
