# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Run a reward file locally to catch faults no dataset row can excuse, before Arena prices a run."""

from __future__ import annotations

import inspect
import os
import traceback
from pathlib import Path
from typing import Any

from agilerl.exceptions import ArenaValidationError
from agilerl.models.env import LLMEnvSpec


def check_reward_file(
    reward_file: str | os.PathLike[str] | bytes, environment: LLMEnvSpec
) -> None:
    """Raise on a syntax error, an undefined name, a missing reward function, or a reward that is not a number.

    The file runs once in this process; a dataset-backed rollout's reward
    function is then called on an empty row. Other errors can come from the
    row or from packages only the trainer has, so they are left to Arena's
    check at submit.

    :param reward_file: The reward module the submit uploads, as a path or its bytes.
    :param environment: The validated manifest's environment.
    :raises ArenaValidationError: The file cannot score any row.
    """
    if isinstance(reward_file, bytes):
        label, source = "reward.py", reward_file
    else:
        label, source = str(reward_file), Path(reward_file).read_bytes()
    namespace: dict[str, Any] = {"__name__": "arena_reward_check", "__file__": label}
    try:
        exec(compile(source, label, "exec"), namespace)
    except SyntaxError as err:
        raise reward_error(label, err.lineno, f"SyntaxError: {err.msg}") from err
    except NameError as err:
        raise reward_error(label, error_line(err, label), f"NameError: {err}") from err
    except Exception:
        # Imports and module setup may need packages or secrets only the trainer has.
        return
    name = environment.rubric_name
    if not environment.dataset_backed_rollout or name is None:
        return
    if name not in namespace:
        msg = f"{label} defines no {name!r}, the reward environment.rubric_name names."
        raise ArenaValidationError(msg)
    reward_fn = namespace[name]
    if not inspect.isfunction(reward_fn):
        return
    row = {"question": "", "answer": "", "completion": ""}
    try:
        inspect.signature(reward_fn).bind(**row)
    except TypeError:
        # Arena's check at submit names the arguments it expects.
        return
    try:
        score = reward_fn(**row)
    except NameError as err:
        raise reward_error(label, error_line(err, label), f"NameError: {err}") from err
    except Exception:
        # The empty row may cause it; Arena runs the reward on a real row at submit.
        return
    # The trainer takes float(score), so ints and NumPy or torch scalars train too.
    if isinstance(score, (str, bytes, bool)) or not hasattr(score, "__float__"):
        problem = f"{name} returned {type(score).__name__} {score!r}; a reward must return a float."
        raise reward_error(label, reward_fn.__code__.co_firstlineno, problem)


def error_line(err: BaseException, label: str) -> int | None:
    """The line in the reward file where *err* was raised."""
    lines = [
        lineno
        for frame, lineno in traceback.walk_tb(err.__traceback__)
        if frame.f_code.co_filename == label
    ]
    return lines[-1] if lines else None


def reward_error(label: str, line: int | None, problem: str) -> ArenaValidationError:
    """An error naming the reward file, the line and what went wrong there."""
    where = f"{label}, line {line}" if line is not None else label
    return ArenaValidationError(f"{where}: {problem}")
