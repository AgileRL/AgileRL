# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared type aliases for the arena client package."""

from __future__ import annotations

# A decoded JSON value: any JSON primitive, array, or object (recursive).
# Defined here (rather than imported from the agilerl core package, where the
# same alias also lives) because agilerl-arena is a standalone distribution that
# does not depend on agilerl core.
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
