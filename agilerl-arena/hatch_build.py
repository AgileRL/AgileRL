# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Stamp the ML bundle version into stable agilerl-arena wheels.

``arena snowflake setup`` copies ``snowflake-job:<bundle.version>-py313-cu130``
for the bundle the installed wheel was released in. Stable release builds set
``ML_BUNDLE_CHANNEL=stable`` and ``ML_BUNDLE_VERSION``; other builds carry no stamp.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

STAMP_NAME = "bundle_version.txt"
STAMP_PATH = f"agilerl/arena/snowflake/{STAMP_NAME}"


class BundleVersionHook(BuildHookInterface):
    """Add ``bundle_version.txt`` to stable wheels."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Write the stamp and include it in the wheel.

        :param version: ``standard`` or ``editable``.
        :param build_data: Hatch build data.
        :raises RuntimeError: A stable build has no ``ML_BUNDLE_VERSION``.
        """
        self.stamp_dir: Path | None = None
        if version == "editable" or os.environ.get("ML_BUNDLE_CHANNEL") != "stable":
            return
        bundle_version = os.environ.get("ML_BUNDLE_VERSION", "").strip()
        if bundle_version == "":
            msg = "A stable agilerl-arena wheel needs ML_BUNDLE_VERSION."
            raise RuntimeError(msg)
        self.stamp_dir = Path(tempfile.mkdtemp(prefix="agilerl-arena-stamp-"))
        stamp = self.stamp_dir / STAMP_NAME
        stamp.write_text(f"{bundle_version}\n", encoding="utf-8")
        build_data["force_include"][str(stamp)] = STAMP_PATH

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        """Remove the temporary stamp directory.

        :param version: ``standard`` or ``editable``.
        :param build_data: Hatch build data.
        :param artifact_path: Built wheel.
        """
        if self.stamp_dir is not None:
            shutil.rmtree(self.stamp_dir)
