"""Runtime checker for the jaxtyping import hook installed in ``tests/conftest.py``."""

import copyreg
import os
import pathlib
import sys
import typing
import weakref

import jaxtyping
from beartype import beartype
from jaxtyping import _import_hook

_JAXTYPING_DECORATOR = jaxtyping._decorator.__file__

_instrumented_exec_module = _import_hook._JaxtypingLoader.exec_module


def _exec_module_uncached(loader, module) -> None:
    """Execute an instrumented module without letting it reach the bytecode cache."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        _instrumented_exec_module(loader, module)
    finally:
        sys.dont_write_bytecode = previous


# jaxtyping redirects instrumented modules to their own ``.pyc`` tag by patching
# ``cache_from_source`` process-wide for the duration of one ``exec_module``,
# reasoning that the import lock makes that safe. The import lock is per module,
# so anything imported inside that window - a transitive dependency, another
# thread, an xdist worker - has its bytecode written to the jaxtyping-tagged path
# WITHOUT being instrumented. The next run finds a tagged file, takes it for
# already-transformed, and silently loads that module unchecked while the suite
# stays green. Not writing the tagged file at all removes the failure mode; the
# instrumented modules are recompiled each session, which is cheap because it is
# only agilerl's own pure-Python sources and not its dependencies.
_import_hook._JaxtypingLoader.exec_module = _exec_module_uncached


def _purge_stale_instrumented_bytecode() -> None:
    """Delete jaxtyping-tagged bytecode left by a run that still wrote it.

    Only the xdist controller sweeps, before any worker starts, so no process can
    be reading a file another is unlinking. Nothing writes these any more, so on
    every run after the first this finds nothing.
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    root = pathlib.Path(__file__).resolve().parent.parent / "agilerl"
    for stale in root.rglob("__pycache__/*jaxtyping*.pyc"):
        stale.unlink(missing_ok=True)


_purge_stale_instrumented_bytecode()


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

    checked.__annotations__ = original
    return checked
