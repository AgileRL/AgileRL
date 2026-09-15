# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Allow ``python -m agilerl.arena.memory`` as a shorthand for the sizing check."""

from agilerl.arena.memory.cli import memory_group

if __name__ == "__main__":
    memory_group()
