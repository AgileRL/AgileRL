from __future__ import annotations

import os
from collections.abc import Callable
from enum import Enum
from importlib import import_module
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from pettingzoo import ParallelEnv
from pydantic import Field
from pydantic.dataclasses import dataclass

from agilerl.vector import AsyncPettingZooVecEnv


class LLMEnvType(str, Enum):
    """Type of LLM environment."""

    REASONING = "reasoning"
    PREFERENCE = "preference"

    def __str__(self) -> str:
        return str(self.value)


GymEnvType = AsyncVectorEnv | SyncVectorEnv
PzEnvType = ParallelEnv | AsyncPettingZooVecEnv
EnvFactory = Callable[[], Any]
WrapperSpec = tuple[Any, dict[str, Any]] | str | Callable[..., Any]


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


def _load_module_from_path(module_ref: str, script_path: Path) -> Any:
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


def _resolve_module(module_ref: str, path: str | None = None) -> Any:
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


def _resolve_entrypoint_target(entrypoint: str, path: str | None = None) -> Any:
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
) -> tuple[Any, dict[str, Any]]:
    """Resolve a wrapper from a wrapper specification and environment path.

    :param wrapper: Wrapper specification.
    :type wrapper: WrapperSpec
    :param path: Environment path.
    :type path: str or None
    :returns: Resolved wrapper and wrapper kwargs.
    :rtype: tuple[Any, dict[str, Any]]
    """
    if isinstance(wrapper, tuple):
        wrapper_spec, wrapper_kwargs = wrapper
    else:
        wrapper_spec, wrapper_kwargs = wrapper, {}

    if isinstance(wrapper_spec, str):
        if ":" in wrapper_spec:
            wrapper_fn = _resolve_entrypoint_target(wrapper_spec, path=path)
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


def _apply_wrappers(
    env: Any,
    wrappers: list[WrapperSpec] | None,
    path: str | None = None,
) -> Any:
    """Apply environment wrappers to an environment.

    :param env: Environment to apply wrappers to.
    :type env: Any
    :param wrappers: List of environment wrappers.
    :type wrappers: list[WrapperSpec] or None
    :param env_path: Environment path.
    :type path: str or None
    :returns: Wrapped environment.
    :rtype: Any
    """
    if not wrappers:
        return env

    wrapped_env = env
    for wrapper in wrappers:
        wrapper_fn, wrapper_kwargs = _resolve_wrapper(wrapper, path=path)
        wrapped_env = wrapper_fn(wrapped_env, **wrapper_kwargs)

    return wrapped_env


@dataclass
class EnvSpec:
    """Environment specification from an Arena manifest.

    Provides information that allows us to construct both gymnasium as well as
    pettingzoo environments, and also custom environments from an entrypoint.

    :param name: Name of the environment
    :type name: str
    :param num_envs: Number of environments to run in parallel
    :type num_envs: int
    """

    name: str
    num_envs: int = Field(default=1, ge=1)


@dataclass
class GymEnvSpec(EnvSpec):
    """Gym environment specification.

    :param entrypoint: Entrypoint for the environment, if custom. Defaults to None.
    :type entrypoint: str or None
    :param path: Path to the environment, if custom. Defaults to None.
    :type path: str or None
    :param config: Environment configuration, if custom. Defaults to None.
    :type config: dict[str, Any] or None
    :param wrappers: Environment wrappers, if custom. Defaults to None.
    :type wrappers: list[tuple[Any, dict[str, Any]] | str] or None
    :param sync: Use synchronous vectorization instead of async.
    :type sync: bool
    """

    entrypoint: str | None = Field(default=None)
    path: str | None = Field(default=None)
    config: dict[str, Any] | None = Field(default=None)
    wrappers: list[WrapperSpec] | None = Field(default=None)
    sync: bool = Field(default=False)

    @staticmethod
    def constuct_custom_env_fn(
        entrypoint: str,
        path: str | None = None,
        config: dict[str, Any] | None = None,
        wrappers: list[WrapperSpec] | None = None,
    ) -> EnvFactory:
        """Construct a custom environment given the configuration.

        :param entrypoint: Entrypoint for the environment, if custom. Defaults to None.
        :type entrypoint: str or None
        :param path: Path to the environment, if custom. Defaults to None.
        :type path: str or None
        :param config: Environment configuration, if custom. Defaults to None.
        :type config: dict[str, Any] or None
        :param wrappers: Environment wrappers, if custom. Defaults to None.
        :type wrappers: list[tuple[Any, dict[str, Any]] | str] or None
        :returns: Custom environment factory function.
        :rtype: Callable[[], Any]
        """

        def default_make_env() -> Any:
            constructor = _resolve_entrypoint_target(entrypoint, path=path)
            if not callable(constructor):
                msg = f"Entrypoint '{entrypoint}' resolved to non-callable object."
                raise TypeError(msg)
            env = constructor(**(config or {}))
            return _apply_wrappers(env, wrappers, path=path)

        return default_make_env

    def make_env(self) -> GymEnvType:
        """Instantiate the vectorized environment given the configuration.

        :returns: Vectorized environment
        :rtype: GymEnvType
        """
        from agilerl.utils.utils import make_vect_envs

        if self.entrypoint is not None:
            make_env = self.constuct_custom_env_fn(
                self.entrypoint,
                self.path,
                self.config,
                self.wrappers,
            )
        else:
            make_env = None

        return make_vect_envs(
            env_name=self.name,
            num_envs=self.num_envs,
            make_env=make_env,
            should_async_vector=(not self.sync),
        )


@dataclass
class PzEnvSpec(EnvSpec):
    """PettingZoo environment specification.

    :param entrypoint: Entrypoint for the environment, if custom. Defaults to None.
    :type entrypoint: str or None
    :param path: Path to the environment, if custom. Defaults to None.
    :type path: str or None
    :param config: Environment configuration, if custom. Defaults to None.
    :type config: dict[str, Any] or None
    :param wrappers: Environment wrappers, if custom. Defaults to None.
    :type wrappers: list[WrapperSpec] or None
    """

    entrypoint: str | None = Field(default=None)
    path: str | None = Field(default=None)
    config: dict[str, Any] | None = Field(default=None)
    wrappers: list[WrapperSpec] | None = Field(default=None)

    @staticmethod
    def constuct_custom_env_fn(
        entrypoint: str,
        path: str | None = None,
        config: dict[str, Any] | None = None,
        wrappers: list[WrapperSpec] | None = None,
    ) -> Callable[[], ParallelEnv]:
        """Construct a custom PettingZoo environment factory.

        For PettingZoo, we always require an explicit constructor/entrypoint.

        :param entrypoint: Entrypoint for the environment, if custom. Defaults to None.
        :type entrypoint: str or None
        :param path: Path to the environment, if custom. Defaults to None.
        :type path: str or None
        :param config: Environment configuration, if custom. Defaults to None.
        :type config: dict[str, Any] or None
        :param wrappers: Environment wrappers, if custom. Defaults to None.
        :type wrappers: list[WrapperSpec] or None
        :returns: Custom PettingZoo environment factory.
        :rtype: Callable[[], ParallelEnv]
        """

        def default_make_env() -> ParallelEnv:
            constructor = _resolve_entrypoint_target(entrypoint, path=path)
            if not callable(constructor):
                msg = f"Entrypoint '{entrypoint}' resolved to non-callable object."
                raise TypeError(msg)
            env = constructor(**(config or {}))
            return _apply_wrappers(env, wrappers, path=path)

        return default_make_env

    def make_env(self) -> AsyncPettingZooVecEnv:
        """Instantiate vectorized PettingZoo environments from a constructor.

        :returns: Vectorized PettingZoo environments.
        :rtype: AsyncPettingZooVecEnv
        """
        if self.entrypoint is not None:
            make_env = self.constuct_custom_env_fn(
                self.entrypoint,
                self.path,
                self.config,
                self.wrappers,
            )
        else:
            # PettingZoo environments still need a constructor path.
            def make_env() -> ParallelEnv:
                module = import_module(self.name)
                if not hasattr(module, "parallel_env"):
                    msg = f"PettingZoo module '{self.name}' has no 'parallel_env' constructor."
                    raise AttributeError(msg)
                constructor = module.parallel_env
                env = constructor(**(self.config or {}))
                return _apply_wrappers(env, self.wrappers, path=self.path)

        from agilerl.utils.utils import make_multi_agent_vect_envs

        return make_multi_agent_vect_envs(
            env=make_env,
            num_envs=self.num_envs,
        )


@dataclass
class LLMEnvSpec:
    """Environment specification for LLM reasoning and preference training.

    Declaratively captures the HuggingFace dataset, tokenizer, reward
    function, and conversation template needed to construct a
    :class:`~agilerl.utils.llm_utils.ReasoningGym` or
    :class:`~agilerl.utils.llm_utils.PreferenceGym`.

    :param dataset_name: HuggingFace dataset identifier (e.g. ``"gsm8k"``).
    :type dataset_name: str
    :param tokenizer_name: HuggingFace tokenizer / model identifier used to
        load an ``AutoTokenizer``.
    :type tokenizer_name: str
    :param env_type: One of ``"reasoning"`` or ``"preference"``.
    :type env_type: str
    :param reward_fn: An entrypoint string (``"module:callable"``) or an
        already-resolved callable that scores completions.  Required for
        reasoning environments; ignored for preference environments.
    :type reward_fn: str | Callable[..., float] | None
    :param conversation_template: Chat-template messages with ``{question}``
        / ``{answer}`` placeholders.  Required for reasoning environments.
    :type conversation_template: list[dict[str, str]] | None
    :param dataset_config: Optional config name passed to
        ``datasets.load_dataset`` (e.g. ``"main"``).
    :type dataset_config: str | None
    :param train_split: Dataset split used for training.
    :type train_split: str
    :param test_split: Dataset split used for evaluation.
    :type test_split: str
    :param data_batch_size_per_gpu: Per-GPU DataLoader batch size.
    :type data_batch_size_per_gpu: int
    :param max_context_length: Maximum tokenized context length.  Samples
        exceeding this are filtered out.
    :type max_context_length: int | None
    :param min_completion_length: Minimum completion length (preference only).
    :type min_completion_length: int | None
    :param seed: RNG seed for the DataLoader.
    :type seed: int
    """

    dataset_name: str
    tokenizer_name: str
    name: str = Field(default="")
    num_envs: int = Field(default=1, ge=1)
    env_type: LLMEnvType = Field(default=LLMEnvType.REASONING)
    reward_fn: str | Callable[..., float] | None = Field(default=None)
    conversation_template: list[dict[str, str]] | None = Field(default=None)
    dataset_config: str | None = Field(default=None)
    train_split: str = Field(default="train")
    test_split: str = Field(default="test")
    data_batch_size_per_gpu: int = Field(default=8, ge=1)
    max_context_length: int | None = Field(default=None)
    min_completion_length: int | None = Field(default=None)
    seed: int = Field(default=42)

    def __post_init__(self) -> None:
        if self.env_type not in (LLMEnvType.REASONING, LLMEnvType.PREFERENCE):
            msg = f"env_type must be {LLMEnvType.REASONING} or {LLMEnvType.PREFERENCE}, got {self.env_type!r}"
            raise ValueError(msg)
        if self.env_type == LLMEnvType.REASONING and self.reward_fn is None:
            msg = "reward_fn is required for reasoning environments"
            raise ValueError(msg)
        if self.env_type == LLMEnvType.REASONING and self.conversation_template is None:
            msg = "conversation_template is required for reasoning environments"
            raise ValueError(msg)

    def _load_datasets(self) -> tuple[Any, Any]:
        """Load train and test splits from HuggingFace.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        from datasets import load_dataset

        ds = load_dataset(self.dataset_name, self.dataset_config)
        return ds[self.train_split], ds[self.test_split]

    def _resolve_reward_fn(self) -> Callable[..., float]:
        """Resolve ``reward_fn`` to a callable.

        If the value is already callable it is returned as-is; otherwise
        the string is treated as an entrypoint (``"module:target"``).

        :returns: The resolved reward callable.
        :rtype: Callable[..., float]
        """
        if callable(self.reward_fn):
            return self.reward_fn
        return _resolve_entrypoint_target(self.reward_fn)

    def _load_tokenizer(self) -> Any:
        """Load the tokenizer from HuggingFace.

        :returns: An ``AutoTokenizer`` instance.
        :rtype: AutoTokenizer
        """
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.tokenizer_name)

    def make_env(self, accelerator: Any | None = None) -> Any:
        """Construct the LLM gym environment.

        :param accelerator: Optional HuggingFace ``Accelerator``.
        :type accelerator: Accelerator | None
        :returns: A :class:`~agilerl.utils.llm_utils.ReasoningGym` or
            :class:`~agilerl.utils.llm_utils.PreferenceGym`.
        """
        from agilerl.utils.llm_utils import PreferenceGym, ReasoningGym

        train_ds, test_ds = self._load_datasets()
        tokenizer = self._load_tokenizer()

        if self.env_type == LLMEnvType.REASONING:
            return ReasoningGym(
                train_dataset=train_ds,
                test_dataset=test_ds,
                tokenizer=tokenizer,
                reward_fn=self._resolve_reward_fn(),
                conversation_template=self.conversation_template,
                data_batch_size_per_gpu=self.data_batch_size_per_gpu,
                accelerator=accelerator,
                max_context_length=self.max_context_length,
                seed=self.seed,
            )

        return PreferenceGym(
            train_dataset=train_ds,
            test_dataset=test_ds,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=self.data_batch_size_per_gpu,
            accelerator=accelerator,
            max_context_length=self.max_context_length,
            min_completion_length=self.min_completion_length,
            seed=self.seed,
        )


@dataclass
class OfflineEnvSpec(GymEnvSpec):
    """Environment specification for offline RL training.

    Wraps a standard Gymnasium evaluation environment together with the
    dataset source used to fill the replay buffer before training begins.

    Exactly one of ``minari_dataset_id`` or ``dataset_path`` must be
    provided.  When ``minari_dataset_id`` is set, the dataset is loaded
    via the `Minari <https://minari.farama.org/>`_ library.  When
    ``dataset_path`` is set, the dataset is loaded from a local HDF5 file.

    :param minari_dataset_id: Identifier for a Minari dataset (e.g.
        ``"cartpole-v0"``).
    :type minari_dataset_id: str | None
    :param dataset_path: Path to a local HDF5 dataset file.
    :type dataset_path: str | None
    :param remote: If ``True``, download the Minari dataset from the
        remote repository when it is not available locally.
    :type remote: bool
    """

    minari_dataset_id: str | None = Field(default=None)
    dataset_path: str | None = Field(default=None)
    remote: bool = Field(default=False)

    def __post_init__(self) -> None:

        if self.dataset_path is not None:
            import h5py

            self.dataset = h5py.File(self.dataset_path, "r")
        elif self.minari_dataset_id is not None:
            self.dataset = None
        else:
            msg = "OfflineEnvSpec requires either 'minari_dataset_id' or 'dataset_path' to be set."
            raise ValueError(msg)


@dataclass
class ArenaEnvSpec(EnvSpec):
    """Environment specification for Arena environments.

    The specified environment name and version should correspond to a registered and
    validated environment on Arena.

    :param name: Name of the environment
    :type name: str
    :param num_envs: Number of environments to run in parallel
    :type num_envs: int
    :param version: Version of the environment
    :type version: str
    """

    version: str = Field(default="latest")
