# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""A contract default has to be the algorithm constructor's default.

The framework forwards only the fields a manifest sets, so an omitted field
gets the constructor's default locally. The payload Arena stores carries every
field, defaults included, so the same manifest run remotely gets the contract's.
Any drift between the two makes one manifest mean two different runs.
"""

import enum
import inspect

import pytest

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.arena.models.algorithms import LLMAlgorithmSpec
from agilerl.arena.models.registry import MANIFEST_REGISTRY
from agilerl.builders import select_builder

# Names that resolve to the same class as another entry; checked once under it.
_ALIASES = {"Rainbow DQN", "Recurrent PPO"}


def _norm(value):
    return value.value if isinstance(value, enum.Enum) else value


def _cases():
    for name, spec_cls in MANIFEST_REGISTRY.items():
        if name in _ALIASES:
            continue
        if not HAS_LLM_DEPENDENCIES and issubclass(spec_cls, LLMAlgorithmSpec):
            # algo_class import-gates on the llm extra; without it there is no
            # constructor to compare the contract against.
            yield pytest.param(
                spec_cls,
                None,
                None,
                id=name,
                marks=pytest.mark.skip(reason="LLM deps not installed"),
            )
            continue
        spec = spec_cls.model_construct()
        params = inspect.signature(
            select_builder(spec).algo_class(spec).__init__
        ).parameters
        for field_name, field in spec_cls.model_fields.items():
            param = params.get(field_name)
            if param is None or param.default is inspect._empty or field.is_required():
                continue
            yield pytest.param(spec_cls, field, param, id=f"{name}.{field_name}")


@pytest.mark.parametrize(("spec_cls", "field", "param"), list(_cases()))
def test_contract_default_matches_the_constructor(spec_cls, field, param):
    contract = field.default_factory() if field.default_factory else field.default
    assert _norm(contract) == _norm(param.default), (
        f"{spec_cls.__name__}.{field.alias or param.name}: contract default "
        f"{contract!r} but {spec_cls.__name__.removesuffix('Spec')}.__init__ "
        f"defaults to {param.default!r}. Change one so a manifest means the "
        "same run locally and on Arena."
    )
