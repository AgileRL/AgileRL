"""Sitecustomize module to enable coverage tracking in subprocesses.

This file is automatically loaded by Python when the tests directory is in PYTHONPATH.
It starts coverage measurement in subprocesses when COVERAGE_PROCESS_START is set.
"""

import os

if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()
