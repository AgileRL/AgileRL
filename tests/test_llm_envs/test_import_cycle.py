# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Env-host workers import ``llm_envs`` before any algorithm module."""

from __future__ import annotations

import subprocess
import sys


def test_llm_envs_import_does_not_load_algorithms() -> None:
    """Ray EnvHostActor does ``from agilerl.llm_envs import openenv_server``."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from agilerl.llm_envs import RolloutHarness, openenv_server\n"
                "import sys\n"
                "assert 'agilerl.algorithms.cqn' not in sys.modules\n"
                "assert 'agilerl.algorithms.core.base' not in sys.modules\n"
                "assert RolloutHarness is not None\n"
                "assert openenv_server is not None\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
