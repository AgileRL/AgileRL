"""Runner for subprocess test execution with coverage support.

This module is invoked by spawn_new_process_for_each_test to execute
pickled test functions in a fresh subprocess. Coverage is automatically
started via sitecustomize.py when COVERAGE_PROCESS_START is set.
"""

import sys

import cloudpickle

if __name__ == "__main__":
    data = sys.stdin.buffer.read()
    if data:
        func, args, kwargs, _ = cloudpickle.loads(data)
        func(*args, **kwargs)
