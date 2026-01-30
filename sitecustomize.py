"""Sitecustomize module to enable coverage tracking in subprocesses.

This file is automatically loaded by Python when the repository root is in PYTHONPATH.
It starts coverage measurement in subprocesses when COVERAGE_PROCESS_START is set.
"""

import os

import coverage

if os.environ.get("COVERAGE_PROCESS_START"):
    coverage.process_startup()
