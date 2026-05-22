"""Pytest fixtures for MuJoCo Playground env wrapper tests."""

from __future__ import annotations

import jax.experimental.layout as _layout
import pytest

if not hasattr(_layout, "Format"):
    _layout.Format = _layout.DeviceLocalLayout

pytest.importorskip("mujoco_playground")
