# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest

from tests.helper_functions import restore_moe_routers


@pytest.fixture
def pristine_moe_routers() -> Iterator[None]:
    """Restore each listed MoE router class after a class-level patch."""
    with restore_moe_routers():
        yield
