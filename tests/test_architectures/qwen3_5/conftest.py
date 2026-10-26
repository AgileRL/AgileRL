# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import pytest

from agilerl import HAS_LLM_DEPENDENCIES

if HAS_LLM_DEPENDENCIES:
    from tests.test_architectures.qwen3_5.gdn_helpers import restored_qwen_gdn_classes


@pytest.fixture
def pristine_qwen_gdn_classes():
    """Run with the unpatched Qwen GDN forwards, and undo the family patches afterwards."""
    with restored_qwen_gdn_classes():
        yield
