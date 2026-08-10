# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rejecting unhonoured DeepSpeed activation-checkpointing config."""

from __future__ import annotations

import pytest

from agilerl.utils.llm_utils import (
    ACTIVATION_CHECKPOINTING_KEY,
    assert_no_activation_checkpointing_config,
)

_SOURCE = "the test config"


def test_accepts_a_config_without_the_section() -> None:
    assert_no_activation_checkpointing_config(
        {"zero_optimization": {"stage": 3}},
        source=_SOURCE,
    )


def test_accepts_a_missing_config() -> None:
    assert_no_activation_checkpointing_config(None, source=_SOURCE)


def test_rejects_a_populated_section_naming_every_key() -> None:
    config = {
        ACTIVATION_CHECKPOINTING_KEY: {
            "partition_activations": True,
            "cpu_checkpointing": False,
        },
    }

    with pytest.raises(RuntimeError) as excinfo:
        assert_no_activation_checkpointing_config(config, source=_SOURCE)

    message = str(excinfo.value)
    assert "cpu_checkpointing, partition_activations" in message
    assert "not honoured" in message
    assert _SOURCE in message


def test_rejects_an_empty_section() -> None:
    with pytest.raises(RuntimeError, match="no keys"):
        assert_no_activation_checkpointing_config(
            {ACTIVATION_CHECKPOINTING_KEY: {}},
            source=_SOURCE,
        )


def test_rejects_a_non_mapping_section() -> None:
    with pytest.raises(RuntimeError, match="True"):
        assert_no_activation_checkpointing_config(
            {ACTIVATION_CHECKPOINTING_KEY: True},
            source=_SOURCE,
        )


def test_rejects_a_config_that_is_not_a_mapping() -> None:
    with pytest.raises(TypeError, match="not a mapping"):
        assert_no_activation_checkpointing_config(
            [ACTIVATION_CHECKPOINTING_KEY],
            source=_SOURCE,
        )
