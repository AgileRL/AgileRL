# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Turn a validated algorithm spec plus run-time arguments into a live algorithm."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agilerl import algorithms
from agilerl.algorithms.core import EvolvableAlgorithm
from agilerl.arena.models.algorithms import (
    AlgoSpec,
    LLMAlgorithmSpec,
    MultiAgentAlgorithmSpec,
    SingleAgentAlgorithmSpec,
)

if TYPE_CHECKING:
    from agilerl.algorithms.core.registry import HyperparameterConfig
else:
    HyperparameterConfig = object

ALGO_CLASSES: dict[type[AlgoSpec], type[EvolvableAlgorithm]] = {}


def registry_names(spec_cls: type[AlgoSpec]) -> Iterator[str]:
    """Yield the registry name of *spec_cls*, then of each spec class above it.

    Walking the MRO lets a spec subclassed outside the framework still resolve
    to its parent's algorithm class.

    :param spec_cls: The spec class.
    :type spec_cls: type[AlgoSpec]
    :returns: Registry names, most specific first.
    :rtype: Iterator[str]
    """
    for spec_type in spec_cls.__mro__:
        if spec_type in (
            SingleAgentAlgorithmSpec,
            MultiAgentAlgorithmSpec,
            LLMAlgorithmSpec,
        ):
            return
        if issubclass(spec_type, BaseModel):
            yield spec_type.__name__.removesuffix("Spec")


class AlgorithmBuilder:
    """Paradigm-keyed factory that builds a live algorithm from a spec.

    Concrete builders own the paradigm-specific ``build`` signature; the
    callers dispatch on paradigm and call the concrete class directly.
    """

    @classmethod
    def algo_class(cls, spec: AlgoSpec) -> type[EvolvableAlgorithm]:
        """Resolve the algorithm class from :mod:`agilerl.algorithms`.

        Uses the naming convention ``<Name>Spec`` -> ``<Name>``, walking the
        spec's MRO so subclasses resolve to their parent's algorithm. Cached
        per spec class; the import is deferred so specs stay cheap to import.

        :param spec: The algorithm spec.
        :type spec: AlgoSpec
        :returns: The algorithm class.
        :rtype: type[EvolvableAlgorithm]
        :raises AttributeError: If no algorithm matches the spec's name.
        """
        spec_cls = type(spec)
        if spec_cls not in ALGO_CLASSES:
            for name in registry_names(spec_cls):
                resolved = getattr(algorithms, name, None)
                if isinstance(resolved, type) and issubclass(
                    resolved, EvolvableAlgorithm
                ):
                    ALGO_CLASSES[spec_cls] = resolved
                    break
            else:
                msg = (
                    f"No algorithm class in agilerl.algorithms for {spec_cls.__name__}."
                )
                raise AttributeError(msg)
        return ALGO_CLASSES[spec_cls]


# Values whose type makes a meaningful ``!=`` comparison; anything else (an
# Accelerator, a LoraConfig, a registry) is skipped when diffing hyperparameters.
COMPARABLE_HP_TYPES = (bool, int, float, str, type(None))


def apply_checkpoint(
    algo: EvolvableAlgorithm,
    resume_from_checkpoint: str | None,
    load_weights_from: str | None,
    *,
    index: int,
) -> None:
    """Seed a freshly-built agent from a checkpoint, if asked to.

    The two options are mutually exclusive:

    * ``resume_from_checkpoint`` continues a run from the checkpoint's optimizer
      state and hyperparameters, warning when they drift from the spec.
    * ``load_weights_from`` warm-starts a new run from prior weights only, keeping
      the spec's hyperparameters.

    :param algo: A freshly-built algorithm, configured from its spec.
    :param resume_from_checkpoint: Checkpoint to resume the run from.
    :type resume_from_checkpoint: str | None
    :param load_weights_from: Checkpoint to take weights from.
    :type load_weights_from: str | None
    :param index: Population slot the agent occupies; restored after a resume,
        since the checkpoint's slot is not this agent's identity.
    :type index: int
    """
    if resume_from_checkpoint is not None and load_weights_from is not None:
        msg = (
            "Provide exactly one of 'resume_from_checkpoint' (continue a run, "
            "restoring optimizer state and its hyperparameters) or "
            "'load_weights_from' (warm-start a new run from prior weights)."
        )
        raise ValueError(msg)

    if load_weights_from is not None:
        algo.load_weights(load_weights_from)
    elif resume_from_checkpoint is not None:
        _resume_and_warn_on_drift(algo, resume_from_checkpoint, index=index)


def _resume_and_warn_on_drift(
    algo: EvolvableAlgorithm, path: str, *, index: int
) -> None:
    """Restore a checkpoint, warning about hyperparameters it overrode.

    The checkpoint's hyperparameters win (the restored optimizer state belongs to
    them), so any drift from the spec is warned about.

    :param algo: A freshly-built algorithm, configured from its spec.
    :param path: Checkpoint to resume from.
    :type path: str
    """
    configured = EvolvableAlgorithm.inspect_attributes(algo, input_args_only=True)

    algo.load_checkpoint(path)
    algo.index = index

    drifted = {
        name: (configured[name], getattr(algo, name))
        for name in configured
        if isinstance(configured[name], COMPARABLE_HP_TYPES)
        and hasattr(algo, name)
        and configured[name] != getattr(algo, name)
    }
    if drifted:
        changes = ", ".join(
            f"{name}: {new!r} (checkpoint) overrides {old!r} (spec)"
            for name, (old, new) in sorted(drifted.items())
        )
        warnings.warn(
            f"Resuming from {path} restored hyperparameters that differ from the "
            f"spec, and the checkpoint's values win because the optimizer state "
            f"belongs to them -- {changes}. Update the spec to match, or use "
            f"'load_weights_from' to warm-start with the spec's values instead.",
            UserWarning,
            stacklevel=3,
        )


def spec_kwargs(
    spec: AlgoSpec, *, hp_config: HyperparameterConfig | None
) -> dict[str, Any]:
    """The constructor kwargs a spec contributes: its explicitly-set fields.

    Only set fields are forwarded so the algorithm's own defaults apply to
    everything a manifest omits. ``net_config`` -- the one nested model on the
    RL specs, resolved from the network section -- goes as the plain dict the
    constructors take, again with only its set fields.

    :param spec: The algorithm spec.
    :type spec: AlgoSpec
    :param hp_config: Resolved hyperparameter config, forwarded when given.
    :type hp_config: HyperparameterConfig | None
    :returns: Keyword arguments for the algorithm constructor.
    :rtype: dict[str, Any]
    """
    kwargs: dict[str, Any] = {
        name: getattr(spec, name) for name in spec.model_fields_set
    }
    net_config = kwargs.get("net_config")
    if isinstance(net_config, (BaseModel, Mapping)):
        kwargs["net_config"] = _as_plain(net_config)
    if hp_config is not None:
        kwargs["hp_config"] = hp_config
    return kwargs


def constructor_kwargs(
    algo_cls: type[EvolvableAlgorithm], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Keep only names the algorithm constructor accepts.

    Arena payloads dump the full contract surface, including fields the
    constructor does not take (zero_stage, deepspeed, vllm_engine_args, …).
    """
    params = inspect.signature(algo_cls.__init__).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {name: value for name, value in kwargs.items() if name in params}


def _as_plain(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    """Dump a network spec, or a per-group mapping of them, to a plain dict."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python", exclude_unset=True)
    return {
        key: _as_plain(item) if isinstance(item, (BaseModel, dict)) else item
        for key, item in value.items()
    }
