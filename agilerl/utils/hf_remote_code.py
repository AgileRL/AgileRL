# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Cross-process lock for Hugging Face loads that run checkpoint code."""

from __future__ import annotations

import contextlib
from pathlib import Path

from filelock import FileLock
from transformers import AutoConfig, dynamic_module_utils

REMOTE_CODE_LOCK_NAME = "agilerl_remote_code.lock"


def remote_code_lock(
    trust_remote_code: bool | None,
) -> contextlib.AbstractContextManager[object]:
    """Lock the Hugging Face modules cache across processes for a remote-code load.

    :param trust_remote_code: The load's ``trust_remote_code``; falsy skips the lock.
    :type trust_remote_code: bool | None
    :return: Context manager holding the lock, or a no-op one.
    :rtype: contextlib.AbstractContextManager[object]
    """
    if not trust_remote_code:
        return contextlib.nullcontext()
    # transformers rewrites checkpoint code in this cache in place under an
    # in-process lock only, so a concurrent rank can import a half-written file.
    lock_path = Path(dynamic_module_utils.HF_MODULES_CACHE) / REMOTE_CODE_LOCK_NAME
    # One instance per path, so a nested lock in the same process re-enters.
    return FileLock(lock_path, is_singleton=True)


def load_remote_code(
    model_name_or_path: str,
    auto_class: type,
    trust_remote_code: bool | None,
) -> None:
    """Copy and import the checkpoint code ``auto_class.from_pretrained`` runs, under the lock.

    Weights can then load outside the lock: transformers only rewrites a cached
    module file when it is missing or differs from the checkpoint's.

    :param model_name_or_path: Hugging Face model id or local path.
    :type model_name_or_path: str
    :param auto_class: Auto class whose ``auto_map`` entry the load resolves.
    :type auto_class: type
    :param trust_remote_code: The load's ``trust_remote_code``; falsy does nothing.
    :type trust_remote_code: bool | None
    """
    if not trust_remote_code:
        return
    with remote_code_lock(trust_remote_code):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        class_reference = getattr(config, "auto_map", {}).get(auto_class.__name__)
        if class_reference is not None:
            dynamic_module_utils.get_class_from_dynamic_module(
                class_reference, model_name_or_path
            )
