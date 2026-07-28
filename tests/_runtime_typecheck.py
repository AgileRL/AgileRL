"""Runtime checker for the jaxtyping import hook installed in ``tests/conftest.py``."""

import typing

import jaxtyping
from beartype import beartype


def _is_array_hint(hint: object) -> bool:
    """Report whether ``hint`` is, or contains, a jaxtyping array type."""
    if isinstance(hint, type) and issubclass(hint, jaxtyping.AbstractArray):
        return True
    return any(_is_array_hint(arg) for arg in typing.get_args(hint))


def typechecker(fn):
    """Wrap ``fn`` so beartype validates only its jaxtyping array annotations.

    Checking every annotation instead surfaces thousands of pre-existing
    inaccuracies unrelated to array shapes (ints declared where Mocks are passed,
    open ``**kwargs`` bags, protocol parameters that tests fill with sentinels).
    Restricting the check to array hints keeps what the static checker cannot
    see - dtype, rank, and axis sizes agreeing across arguments - without making
    the suite hostage to the rest of the annotation surface. It also sidesteps
    beartype's refusal to handle PEP 673 ``Self`` on an undecorated class, since
    those annotations are dropped before it ever sees them.
    """
    hints = getattr(fn, "__annotations__", None)
    if not hints:
        return fn

    resolved = hints
    if any(isinstance(v, str) for v in hints.values()):
        # ``from __future__ import annotations`` modules hand us strings.
        try:
            resolved = typing.get_type_hints(fn, include_extras=True)
        except Exception:
            return fn

    array_hints = {k: v for k, v in resolved.items() if _is_array_hint(v)}
    if not array_hints:
        return fn

    original = fn.__annotations__
    fn.__annotations__ = array_hints
    try:
        checked = beartype(fn)
    except Exception:
        return fn
    finally:
        fn.__annotations__ = original

    checked.__annotations__ = original
    return checked
