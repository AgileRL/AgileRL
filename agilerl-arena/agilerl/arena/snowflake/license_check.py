# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Run the head's Arena licence check and print one result line.

Setup's driver-check job runs ``python -m agilerl.arena.snowflake.license_check``
in the trainer image with the head's Arena secrets and external access
integration, so it reaches Arena the way the head does.
"""

from __future__ import annotations

import os

from agilerl.arena.snowflake.license import (
    SnowflakeLicenseError,
    check_snowflake_license,
)

RESULT_PREFIX = "arena-license-check: "
RESULT_OK = "ok"


def main() -> int:
    """Print ``arena-license-check: ok``, or the licence error after the prefix.

    :return: 0 when Arena allows Snowflake training, else 1.
    """
    try:
        check_snowflake_license(os.environ)
    except SnowflakeLicenseError as exc:
        print(RESULT_PREFIX + " ".join(str(exc).split()), flush=True)
        return 1
    print(RESULT_PREFIX + RESULT_OK, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
