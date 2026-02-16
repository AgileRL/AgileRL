"""Shared utilities for AgileRL tutorials."""

import re
from collections.abc import Generator
from contextlib import contextmanager


def _format_install_message(package: str) -> str:
    """Build install hint for a single package."""
    return (
        f"This tutorial requires the '{package}' package, which is not installed.\n"
        f"Install it with:\n\n"
        f"    pip install {package}\n\n"
        f"Or with uv:\n\n"
        f"    uv pip install {package}\n"
    )


class MissingPackageError(ImportError):
    """Raised when a required optional package is not installed.

    Use with the :func:`require_package` context manager in tutorials
    to prompt users to install missing dependencies.
    """

    def __init__(self, package: str, message: str | None = None):
        self.package = package
        self._message = message or _format_install_message(package)
        super().__init__(self._message)

    def __str__(self) -> str:
        return self._message


def _package_from_import_error(exc: ImportError) -> str | None:
    """Extract the missing package name from an ImportError."""
    # ImportError.name is set in Python 3.3+ for "from X import Y" / "import X"
    if getattr(exc, "name", None):
        # name can be the module we tried to import (e.g. 'ucimlrepo')
        return exc.name
    # Fallback: parse "No module named 'foo'" or "No module named \"foo\""
    msg = str(exc)
    match = re.search(r"No module named\s+['\"]([^'\"]+)['\"]", msg)
    if match:
        return match.group(1)
    return None


@contextmanager
def require_package() -> Generator[None, None, None]:
    """Context manager that turns import failures into a clear install message.

    Wrap optional tutorial imports in this context. If an :exc:`ImportError`
    is raised inside the block, it is caught and re-raised as
    :exc:`MissingPackageError` with a message telling the user how to install
    the missing package (inferred from the exception).

    Examples
    --------
    >>> with require_package():
    ...     from ucimlrepo import fetch_ucirepo
    ...     from scipy.ndimage import gaussian_filter1d

    """
    try:
        yield
    except ImportError as e:
        inferred = _package_from_import_error(e)
        if inferred:
            raise MissingPackageError(inferred) from e
        raise
