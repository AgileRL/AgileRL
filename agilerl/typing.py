# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Shared type aliases for the Arena client package."""

from __future__ import annotations

# A decoded JSON value: any JSON primitive, array, or object (recursive).
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
