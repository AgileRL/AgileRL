# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Guard: every algorithm constructor argument must be documented.

With so many tunable settings across the RL and LLM algorithms it is easy for a
new ``__init__`` parameter to be added without a matching ``:param:`` entry (or for
a removed parameter to leave a stale one behind). This test parses the source
with :mod:`ast` -- it imports nothing, so it runs in every CI lane regardless of
optional dependencies -- and fails if an algorithm class that documents *some*
of its constructor parameters does not document *all* of them.

Rule: a class is audited when it defines ``__init__`` and its class (or
``__init__``) docstring contains at least one ``:param:`` entry. Audited classes
must document every constructor parameter (excluding ``self`` and
``*args`` / ``**kwargs``) and must not carry ``:param:`` entries for
parameters absent from the class. Classes that document no parameters
at all (internal helpers) are exempt.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ALGORITHMS_DIR = Path(__file__).resolve().parents[2] / "agilerl" / "algorithms"
REPO_ROOT = ALGORITHMS_DIR.parents[1]
_PARAM_RE = re.compile(r":param (\w+):")
_SKIP = {"self", "args", "kwargs"}


def _documented_params(node: ast.AST) -> set[str]:
    return set(_PARAM_RE.findall(ast.get_docstring(node) or ""))


def _init_params(init: ast.FunctionDef) -> list[str]:
    a = init.args
    names = [arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return [n for n in names if n not in _SKIP]


def _audited_classes() -> list[tuple[str, str, list[str], list[str]]]:
    """Return ``(relpath, classname, missing, stale)`` for every audited class."""
    results: list[tuple[str, str, list[str], list[str]]] = []
    for path in sorted(ALGORITHMS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            init = next(
                (
                    n
                    for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            documented = _documented_params(cls) | _documented_params(init)
            if not documented:
                continue  # documents no params -> exempt (internal helper)
            params = _init_params(init)
            missing = [p for p in params if p not in documented]
            stale = [d for d in sorted(documented) if d not in params]
            rel = path.relative_to(REPO_ROOT).as_posix()
            results.append((rel, cls.name, missing, stale))
    return results


_AUDITED = _audited_classes()


def test_audit_discovers_algorithm_classes() -> None:
    """Fail loudly if discovery finds nothing (e.g. the package moved)."""
    assert len(_AUDITED) >= 15, (
        f"Constructor-docstring audit found only {len(_AUDITED)} documented "
        "algorithm classes under "
        f"{ALGORITHMS_DIR}; discovery is probably broken."
    )


@pytest.mark.parametrize(
    ("classname", "missing", "stale"),
    [pytest.param(c, m, s, id=f"{rel}::{c}") for rel, c, m, s in _AUDITED],
)
def test_constructor_params_documented(
    classname: str, missing: list[str], stale: list[str]
) -> None:
    problems = []
    if missing:
        problems.append(f"undocumented constructor args {missing}")
    if stale:
        problems.append(f":param: entries with no matching arg {stale}")
    assert not problems, (
        f"{classname}: docstring is out of sync with __init__ -- "
        + "; ".join(problems)
        + ". Add a ':param <name>:' / ':type <name>:' block for each argument "
        "(or remove the stale entry)."
    )
