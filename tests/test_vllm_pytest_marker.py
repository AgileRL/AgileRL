# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Module-level ``importorskip("vllm")`` must carry ``pytestmark = pytest.mark.vllm``.

Hosted Linux CPU installs ``llm`` without vLLM and runs ``-m "not gpu and not vllm"``.
Those modules skip on CPU unless they also take the vllm mark so the GPU job
(``-m "gpu or vllm"``) collects them.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_importorskip_vllm(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    if _call_name(node.value.func) not in {"importorskip", "pytest.importorskip"}:
        return False
    if not node.value.args:
        return False
    first = node.value.args[0]
    return isinstance(first, ast.Constant) and first.value == "vllm"


def _assigns_pytestmark_vllm(node: ast.AST) -> bool:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not isinstance(target, ast.Name) or target.id != "pytestmark":
        return False
    value = node.value
    if not isinstance(value, ast.Attribute) or value.attr != "vllm":
        return False
    return _call_name(value.value) == "pytest.mark"


class TestVllmImportorskipUsesVllmMarker:
    def test_module_level_importorskip_sets_pytestmark(self) -> None:
        tests_root = Path(__file__).resolve().parent
        missing: list[str] = []
        checked = 0
        for path in sorted(tests_root.rglob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            body = list(tree.body)
            if not any(_is_importorskip_vllm(node) for node in body):
                continue
            checked += 1
            if not any(_assigns_pytestmark_vllm(node) for node in body):
                missing.append(path.relative_to(tests_root).as_posix())

        assert checked >= 1
        assert missing == []
