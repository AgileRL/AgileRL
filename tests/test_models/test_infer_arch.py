"""Tests for infer_encoder_arch: obs-space -> encoder arch inference."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from agilerl.models.networks import (
    CnnSpec,
    LstmSpec,
    MlpSpec,
    MultiInputSpec,
    SimbaSpec,
    encoder_spec_for_arch,
    infer_encoder_arch,
    network_arch_is_resolvable,
    normalize_manifest_network,
)
from agilerl.modules.configs import (
    CnnNetConfig,
    LstmNetConfig,
    MlpNetConfig,
    MultiInputNetConfig,
    SimBaNetConfig,
)
from agilerl.utils.evolvable_networks import (
    config_from_dict,
    get_default_encoder_config,
)

VECTOR = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
IMAGE = spaces.Box(0, 255, shape=(3, 32, 32), dtype=np.uint8)
DICT = spaces.Dict(
    {"a": spaces.Box(-1.0, 1.0, shape=(2,)), "b": spaces.Box(-1.0, 1.0, shape=(3,))}
)
TUPLE = spaces.Tuple((spaces.Box(-1.0, 1.0, shape=(2,)), spaces.Discrete(3)))


@pytest.mark.parametrize(
    ("space", "recurrent", "simba", "expected"),
    [
        (DICT, False, False, "multiinput"),
        (TUPLE, False, False, "multiinput"),
        (IMAGE, False, False, "cnn"),
        (VECTOR, False, False, "mlp"),
        (VECTOR, True, False, "lstm"),
        (VECTOR, False, True, "simba"),
        (VECTOR, True, True, "simba"),  # simba wins over recurrent
        (DICT, True, True, "multiinput"),  # space wins over flags
        (IMAGE, True, True, "cnn"),
    ],
)
def test_infer_encoder_arch(space, recurrent, simba, expected):
    assert infer_encoder_arch(space, recurrent=recurrent, simba=simba) == expected


_CONFIG_TO_ARCH = {
    MlpNetConfig: "mlp",
    CnnNetConfig: "cnn",
    LstmNetConfig: "lstm",
    SimBaNetConfig: "simba",
    MultiInputNetConfig: "multiinput",
}


@pytest.mark.parametrize(
    ("space", "recurrent", "simba"),
    [
        (DICT, False, False),
        (IMAGE, False, False),
        (VECTOR, False, False),
        (VECTOR, True, False),
        (VECTOR, False, True),
    ],
)
def test_drift_guard_matches_get_default_encoder_config(space, recurrent, simba):
    cfg = config_from_dict(
        get_default_encoder_config(space, simba=simba, recurrent=recurrent)
    )
    assert (
        infer_encoder_arch(space, recurrent=recurrent, simba=simba)
        == _CONFIG_TO_ARCH[type(cfg)]
    )


@pytest.mark.parametrize("value", ["not-a-dict", None, 42, ["arch"]])
def test_network_arch_is_resolvable_non_dict(value):
    assert network_arch_is_resolvable(value) is False


@pytest.mark.parametrize(
    ("arch", "expected_cls"),
    [
        ("mlp", MlpSpec),
        ("cnn", CnnSpec),
        ("lstm", LstmSpec),
        ("simba", SimbaSpec),
        ("multiinput", MultiInputSpec),
    ],
)
def test_encoder_spec_for_arch_maps_every_arch(arch, expected_cls):
    assert encoder_spec_for_arch(arch) is expected_cls


def test_encoder_spec_for_arch_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown encoder arch"):
        encoder_spec_for_arch("bogus")


@pytest.mark.parametrize("value", ["not-a-dict", None, 7])
def test_normalize_manifest_network_non_dict_passthrough(value):
    assert normalize_manifest_network(value) == value
