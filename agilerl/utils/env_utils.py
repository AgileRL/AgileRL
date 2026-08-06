# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from pettingzoo import ParallelEnv

from agilerl.typing import WrapperSpec
from agilerl.vector import AsyncPettingZooVecEnv

GymEnvType = AsyncVectorEnv | SyncVectorEnv
PzEnvType = ParallelEnv | AsyncPettingZooVecEnv

# Single envs, vectorized gym envs, and PettingZoo parallel / vec envs.
# Wrappers duck-type as the env they wrap, so wrapping preserves EnvT.
EnvT = TypeVar("EnvT", bound=gym.Env | GymEnvType | PzEnvType)


def make_conversation_template(prompt_template: dict[str, str]) -> list[dict[str, str]]:
    """Make a conversation template for the model.

    :param prompt_template: Template dictionary with prompt keys
    :type prompt_template: dict[str, str]
    :return: List of conversation dictionaries
    :rtype: list[dict[str, str]]
    """
    return [
        {
            "role": key.split("_")[0],
            "content": escape_non_format_braces(prompt_template[key]),
        }
        for key in prompt_template
    ]


def get_reward_fn(reward_fn_name: str, file_path: str) -> Callable[..., float]:
    """Removed — use :func:`get_rubric`."""
    msg = (
        "get_reward_fn is removed; use get_rubric(rubric_name, file_path) and "
        "export a Rubric instance (or subclass) from the file. Wrap a legacy "
        "(completion, answer, question) -> float callable with "
        "agilerl.llm_envs.rubrics.reward_fn_to_rubric(...)."
    )
    raise ValueError(msg)


def get_rubric(rubric_name: str, file_path: str) -> Any:  # noqa: ANN401
    """Load a Rubric from a file, wrapping legacy reward callables.

    Accepts a ``Rubric`` instance, a ``Rubric`` subclass (instantiated here), or a
    ``(completion, answer, question) -> float`` callable (Arena / older manifests),
    which is wrapped with :func:`~agilerl.llm_envs.rubrics.reward_fn_to_rubric`.

    :param rubric_name: Name of the symbol in ``file_path``.
    :param file_path: Path to the Python module defining the rubric or reward fn.
    :returns: An OpenEnv ``Rubric`` instance.
    :raises TypeError: If the symbol is not a Rubric, Rubric subclass, or callable.
    """
    from openenv.core.rubrics.base import Rubric

    from agilerl.llm_envs.rubrics import reward_fn_to_rubric

    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        msg = f"{file_path} not found"
        raise ValueError(msg)
    try:
        abs_file_path = file_path_obj.resolve()
        module_name = file_path_obj.stem
        spec = importlib_util.spec_from_file_location(
            module_name,
            str(abs_file_path),
        )
        if spec is None or spec.loader is None:
            msg = f"Could not create spec for {abs_file_path}"
            raise ValueError(msg)
        rubric_module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(rubric_module)
        obj = getattr(rubric_module, rubric_name)
    except Exception as e:
        msg = f"Error importing rubric {rubric_name} from {file_path}: {e}"
        raise ValueError(msg) from e

    if isinstance(obj, Rubric):
        return obj
    if isinstance(obj, type) and issubclass(obj, Rubric):
        return obj()
    if callable(obj):
        return reward_fn_to_rubric(obj)
    msg = (
        f"{rubric_name!r} in {file_path} must be a Rubric instance, Rubric "
        f"subclass, or (completion, answer, question) -> float callable, "
        f"got {type(obj).__name__}."
    )
    raise TypeError(msg)


def escape_non_format_braces(
    template: str,
    format_keys: list[str] | None = None,
) -> str:
    """Replace {word} with {{word}} except for known format keys.

    :param template: The template to escape
    :type template: str
    :param format_keys: The keys to not escape
    :type format_keys: list[str]
    :return: The escaped template
    :rtype: str
    """
    if format_keys is None:
        format_keys = ["question", "answer"]

    # Escape every brace, then restore the known placeholders. Escaping via
    # regex instead would miss ``{}``, lone braces, and nested-brace JSON —
    # all of which make ``str.format`` raise on real system prompts.
    escaped = template.replace("{", "{{").replace("}", "}}")
    for key in format_keys:
        escaped = escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}")
    return escaped


def _parse_entrypoint(entrypoint: str) -> tuple[str, str]:
    """Parse an entrypoint string into a module reference and target name.

    :param entrypoint: Entrypoint string to parse.
    :type entrypoint: str
    :returns: Tuple of module reference and target name.
    :rtype: tuple[str, str]
    """
    if ":" not in entrypoint:
        msg = (
            "Invalid entrypoint format. Expected '{file_name}:ClassOrCallable', "
            f"got '{entrypoint}'."
        )
        raise ValueError(msg)
    module_ref, target_name = entrypoint.rsplit(":", maxsplit=1)
    if not module_ref or not target_name:
        msg = f"Entrypoint must include both module and target. Got '{entrypoint}'."
        raise ValueError(msg)
    return module_ref, target_name


def _load_module_from_path(module_ref: str, script_path: Path) -> ModuleType:
    """Load a module from a file path.

    :param module_ref: Module reference.
    :type module_ref: str
    :param script_path: File path to the module.
    :type script_path: Path
    :returns: Loaded module.
    :rtype: Any
    """
    module_name = (
        f"agilerl_custom_env_{module_ref.replace(os.sep, '_').replace('.', '_')}"
    )
    spec = importlib_util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load module '{module_ref}' from '{script_path}'."
        raise ImportError(msg)
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_module(module_ref: str, path: str | None = None) -> ModuleType:
    """Resolve a module from a module reference and environment path.

    :param module_ref: Module reference.
    :type module_ref: str
    :param path: Environment path.
    :type env_path: str or None
    :returns: Resolved module.
    :rtype: Any
    """
    base_dir = Path(path).resolve() if path is not None else Path.cwd()
    normalized_path = module_ref.replace(".", os.sep)
    candidates = [Path(module_ref), Path(normalized_path)]
    if not module_ref.endswith(".py"):
        candidates.extend([Path(f"{module_ref}.py"), Path(f"{normalized_path}.py")])

    for candidate in candidates:
        script_path = candidate if candidate.is_absolute() else base_dir / candidate
        if script_path.is_file():
            return _load_module_from_path(module_ref, script_path)

    try:
        return import_module(module_ref)
    except ModuleNotFoundError as err:
        msg = (
            f"Could not resolve module '{module_ref}' from path '{base_dir}'. "
            "Expected a python file relative to path/cwd or an importable module."
        )
        raise ModuleNotFoundError(msg) from err


def resolve_entrypoint_target(entrypoint: str, path: str | None = None) -> Any:  # noqa: ANN401 -- resolves an arbitrary user-defined class/callable via getattr
    """Resolve an entrypoint target from an entrypoint string and environment path.

    :param entrypoint: Entrypoint string to resolve.
    :type entrypoint: str
    :param path: Environment path.
    :type path: str or None
    :returns: Resolved entrypoint target.
    :rtype: Any
    """
    module_ref, target_name = _parse_entrypoint(entrypoint)
    module = _resolve_module(module_ref, path=path)
    if not hasattr(module, target_name):
        msg = f"Module '{module_ref}' does not define '{target_name}'."
        raise AttributeError(msg)
    return getattr(module, target_name)


def _resolve_wrapper(
    wrapper: WrapperSpec, path: str | None = None
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Resolve a wrapper from a wrapper specification and environment path.

    :param wrapper: Wrapper specification.
    :type wrapper: WrapperSpec
    :param path: Environment path.
    :type path: str or None
    :returns: Resolved wrapper callable and wrapper kwargs.
    :rtype: tuple[Callable[..., Any], dict[str, Any]]
    """
    wrapper_kwargs: dict[str, Any]
    if isinstance(wrapper, tuple):
        # ``isinstance(tuple)`` cannot exclude the ``Callable`` arm of
        # ``WrapperSpec``, so it erases the tuple's element types; restate them
        # through an Any-typed unpack.
        spec_and_kwargs: Any = wrapper
        wrapper_spec, wrapper_kwargs = spec_and_kwargs
    else:
        wrapper_spec, wrapper_kwargs = wrapper, {}

    if isinstance(wrapper_spec, str):
        if ":" in wrapper_spec:
            wrapper_fn = resolve_entrypoint_target(wrapper_spec, path=path)
        elif "." in wrapper_spec:
            module_ref, target_name = wrapper_spec.rsplit(".", maxsplit=1)
            wrapper_fn = getattr(import_module(module_ref), target_name)
        else:
            msg = (
                f"Invalid wrapper '{wrapper_spec}'. Use a callable, "
                "a tuple(callable, kwargs), or a string import path."
            )
            raise ValueError(msg)
    else:
        wrapper_fn = wrapper_spec

    if not callable(wrapper_fn):
        msg = f"Wrapper '{wrapper_spec}' resolved to non-callable object."
        raise TypeError(msg)
    return wrapper_fn, wrapper_kwargs


def apply_wrappers(
    env: EnvT,
    wrappers: list[WrapperSpec] | None,
    path: str | None = None,
) -> EnvT:
    """Apply environment wrappers to an environment.

    :param env: Environment to apply wrappers to.
    :type env: GymEnvType | PzEnvType
    :param wrappers: List of environment wrappers.
    :type wrappers: list[WrapperSpec] or None
    :param env_path: Environment path.
    :type path: str or None
    :returns: Wrapped environment.
    :rtype: GymEnvType | PzEnvType
    """
    if not wrappers:
        return env

    wrapped_env = env
    for wrapper in wrappers:
        wrapper_fn, wrapper_kwargs = _resolve_wrapper(wrapper, path=path)
        wrapped_env = wrapper_fn(wrapped_env, **wrapper_kwargs)

    return wrapped_env
