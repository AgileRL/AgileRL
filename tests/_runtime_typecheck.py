"""Runtime checker for the jaxtyping import hook installed in ``tests/conftest.py``."""

import copyreg
import functools
import typing
import weakref

import jaxtyping
import torch
from beartype import beartype

_JAXTYPING_DECORATOR = jaxtyping._decorator.__file__


class _DeadReferent:
    """Stands in for a jaxtyping wrapper whose weakref did not survive pickling."""


_DEAD_REFERENT = _DeadReferent()


def _rebuild_jaxtyping_weakref() -> weakref.ReferenceType:
    """Rebuild a jaxtyping wrapper weakref in a subprocess."""
    return weakref.ref(_DEAD_REFERENT)


def _reduce_weakref(ref: weakref.ReferenceType):
    """Make jaxtyping's own wrapper weakrefs picklable, and nothing else."""
    code = getattr(ref(), "__code__", None)
    if code is None or code.co_filename != _JAXTYPING_DECORATOR:
        msg = "cannot pickle 'weakref.ReferenceType' object"
        raise TypeError(msg)
    return (_rebuild_jaxtyping_weakref, ())


# Each wrapper jaxtyping builds closes over a list holding a weakref to itself,
# consulted only to read ``__no_type_check__`` off the wrapper. cloudpickle
# serializes closures by value, so any function reached through a wrapped one
# becomes unpicklable and the async vectorized envs cannot spawn their workers.
# The substitute referent has no ``__no_type_check__``, which is what the live
# weakref reports for every ordinary function, so behaviour is unchanged.
copyreg.pickle(weakref.ReferenceType, _reduce_weakref)


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

    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        if torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        return checked(*args, **kwargs)

    guarded.__annotations__ = original
    return guarded
