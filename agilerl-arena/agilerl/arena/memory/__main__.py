# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Run ``arena memory`` as ``python -m agilerl.arena.memory``."""

import importlib

from agilerl.arena.memory.cli import memory_group

if __name__ == "__main__":
    # Installs the rich log handler the ``arena`` entry point loads via agilerl.arena.cli.
    importlib.import_module("agilerl.arena._console")
    memory_group()
