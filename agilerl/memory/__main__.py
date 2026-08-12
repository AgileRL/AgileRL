# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0
"""Allow ``python -m agilerl.memory`` as a shorthand for the sizing check."""

import sys

from agilerl.memory.cli import main

if __name__ == "__main__":
    sys.exit(main())
