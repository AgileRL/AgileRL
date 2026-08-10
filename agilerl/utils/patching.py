# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Primitives shared by every class-level patch module.

This sits below both :mod:`agilerl.utils.zero3_patches` and
:mod:`agilerl.architectures` so the generic and per-family patches can share
resolution and idempotence without importing each other.
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType

logger = logging.getLogger(__name__)

__all__ = ["class_is_patched", "try_import"]


def try_import(module_path: str) -> ModuleType | None:
    """Import *module_path*, or None when the package cannot be loaded.

    :param module_path: Importable module path.
    :type module_path: str
    :return: The imported module, or None.
    :rtype: ModuleType | None
    """
    try:
        return importlib.import_module(module_path)
    except Exception:
        logger.debug("[patches] %s could not be imported", module_path)
        return None


def class_is_patched(cls: type, flag: str) -> bool:
    """Whether *cls* itself carries *flag*, ignoring any inherited value.

    :param cls: Class to inspect.
    :type cls: type
    :param flag: Marker attribute name.
    :type flag: str
    :return: True when this exact class has already been patched.
    :rtype: bool
    """
    return bool(vars(cls).get(flag, False))
