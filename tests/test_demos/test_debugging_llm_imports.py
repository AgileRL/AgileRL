# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""LLM debugging demos import names that ``agilerl.llm_envs`` actually exports."""

from __future__ import annotations

import ast
from pathlib import Path

from agilerl.llm_envs import __all__ as LLM_ENVS_EXPORTS

DEBUGGING_DEMOS = Path(__file__).resolve().parents[2] / "demos" / "llm" / "debugging"


class TestDebuggingLlmImports:
    def test_llm_envs_imports_are_exported(self) -> None:
        exported = set(LLM_ENVS_EXPORTS)
        imported: list[str] = []
        missing: list[str] = []
        for path in sorted(DEBUGGING_DEMOS.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.module != "agilerl.llm_envs":
                    continue
                for alias in node.names:
                    imported.append(alias.name)
                    if alias.name not in exported:
                        missing.append(f"{path.name}:{alias.name}")

        assert missing == []
        assert "RolloutHarness" in imported
