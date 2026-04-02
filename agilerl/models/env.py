from __future__ import annotations

import os
from collections.abc import Callable
from enum import Enum
from importlib import import_module
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from pettingzoo import ParallelEnv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

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


class EnvSpec(BaseModel):
    """Environment specification from an Arena manifest.

    Provides information that allows us to construct both gymnasium as well as
    pettingzoo environments, and also custom environments from an entrypoint.

    :param name: Name of the environment
    :type name: str
    :param num_envs: Number of environments to run in parallel
    :type num_envs: int
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    num_envs: int = Field(default=1, ge=1)


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

    def make_single_env(self) -> gym.Env:
        """Create a single (non-vectorized) environment instance.

        Useful for probing the observation/action space without the overhead
        of spinning up a full vectorized environment.

        :returns: A single gymnasium environment.
        :rtype: gymnasium.Env
        """
        if self.entrypoint is not None:
            return self.constuct_custom_env_fn(
                self.entrypoint,
                self.path,
                self.config,
                self.wrappers,
            )()
        return gym.make(self.name)

    def make_env(self, extra_wrappers: list[type] | None = None) -> GymEnvType:
        """Instantiate the vectorized environment given the configuration.

        :param extra_wrappers: Optional list of wrapper classes to apply to each
            individual environment before vectorization.
        :type extra_wrappers: list[type] or None, optional
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
            extra_wrappers=extra_wrappers,
        )


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

    def make_single_env(self) -> ParallelEnv:
        """Create a single (non-vectorized) PettingZoo environment instance.

        Useful for probing the observation/action spaces without the overhead
        of spinning up a full vectorized environment.

        :returns: A single PettingZoo parallel environment.
        :rtype: ParallelEnv
        """
        if self.entrypoint is not None:
            return self.constuct_custom_env_fn(
                self.entrypoint,
                self.path,
                self.config,
                self.wrappers,
            )()
        module = import_module(self.name)
        if not hasattr(module, "parallel_env"):
            msg = f"PettingZoo module '{self.name}' has no 'parallel_env' constructor."
            raise AttributeError(msg)

        env = module.parallel_env(**(self.config or {}))
        return _apply_wrappers(env, self.wrappers, path=self.path)

    def make_env(
        self, extra_wrappers: list[type] | None = None
    ) -> AsyncPettingZooVecEnv:
        """Instantiate vectorized PettingZoo environments from a constructor.

        :param extra_wrappers: Optional list of wrapper classes to apply to each
            individual environment before vectorization.
        :type extra_wrappers: list[type] or None, optional
        :returns: Vectorized PettingZoo environments.
        :rtype: AsyncPettingZooVecEnv
        """
        from agilerl.utils.utils import make_multi_agent_vect_envs

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

        return make_multi_agent_vect_envs(
            env=make_env,
            num_envs=self.num_envs,
            extra_wrappers=extra_wrappers,
        )


class LLMEnvSpec(BaseModel):
    """Environment specification for LLM reasoning and preference training.

    Declaratively captures the dataset, reward function, and prompt template
    needed to construct a :class:`~agilerl.utils.llm_utils.ReasoningGym` or
    :class:`~agilerl.utils.llm_utils.PreferenceGym`.  Fields are aligned
    with what Arena expects for LLM training jobs.

    :param env_type: The type of LLM environment (``"reasoning"`` or
        ``"preference"``).
    :type env_type: LLMEnvType
    :param columns: Optional mapping from source dataset column names to the
        names expected by the gym (e.g. ``{"question": "input", "answer":
        "output"}`` for reasoning).
    :type columns: dict[str, str] | None
    :param prompt_template: Chat-template configuration passed as
        ``conversation_template`` to :class:`ReasoningGym`.
    :type prompt_template: dict[str, Any] | None
    :param max_reward: Maximum achievable reward, forwarded to the LLM
        training loop for accuracy logging.
    :type max_reward: float | None
    :param train_test_split: Fraction of the dataset used for training.
    :type train_test_split: float
    :param reward_file_path: Path to a Python file containing the reward
        function.  Required for reasoning environments.
    :type reward_file_path: str | None
    :param dataset_path: Path to a Parquet dataset file.
    :type dataset_path: str
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_type: LLMEnvType
    columns: dict[str, str] | None = Field(default=None)
    prompt_template: dict[str, Any] | None = Field(default=None)
    max_reward: float | None = Field(default=None)
    train_test_split: float = Field(default=0.9, ge=0.0, le=1.0)
    reward_file_path: str | None = Field(default="reward.py")
    dataset_path: str = Field(default="dataset.parquet")

    @model_validator(mode="after")
    def _validate_reasoning_fields(self) -> Self:
        if self.env_type == LLMEnvType.REASONING and self.reward_file_path is None:
            msg = "reward_file_path is required for reasoning environments"
            raise ValueError(msg)
        return self

    def _load_dataset(self) -> tuple[Any, Any]:
        """Load and split the Parquet dataset into train/test.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        import pandas as pd
        from datasets import Dataset

        df = pd.read_parquet(self.dataset_path)
        if self.columns:
            df = df.rename(columns=self.columns)

        ds = Dataset.from_pandas(df)
        split = ds.train_test_split(test_size=1.0 - self.train_test_split)
        return split["train"], split["test"]

    def _resolve_reward_fn(self) -> Callable[..., float]:
        """Resolve the reward function from ``reward_file_path``.

        The file is expected to contain a ``reward`` callable at module
        level.

        :returns: The resolved reward callable.
        :rtype: Callable[..., float]
        """
        return _resolve_entrypoint_target(f"{self.reward_file_path}:reward")

    def make_env(self, tokenizer: Any, accelerator: Any | None = None) -> Any:
        """Construct the LLM gym environment.

        :param tokenizer: A HuggingFace tokenizer instance.
        :type tokenizer: Any
        :param accelerator: Optional HuggingFace ``Accelerator``.
        :type accelerator: Accelerator | None
        :returns: A :class:`~agilerl.utils.llm_utils.ReasoningGym` or
            :class:`~agilerl.utils.llm_utils.PreferenceGym`.
        """
        from agilerl.utils.llm_utils import PreferenceGym, ReasoningGym

        train_ds, test_ds = self._load_dataset()

        if self.env_type == LLMEnvType.REASONING:
            return ReasoningGym(
                train_dataset=train_ds,
                test_dataset=test_ds,
                tokenizer=tokenizer,
                reward_fn=self._resolve_reward_fn(),
                conversation_template=self.prompt_template,
                accelerator=accelerator,
            )

        return PreferenceGym(
            train_dataset=train_ds,
            test_dataset=test_ds,
            tokenizer=tokenizer,
            accelerator=accelerator,
        )


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
    dataset: Any = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_and_load_dataset(self) -> Self:
        if self.dataset_path is not None:
            # NOTE: Not too sure this is a robust way to load any kind of dataset
            self.dataset = h5py.File(self.dataset_path, "r")
        elif self.minari_dataset_id is not None:
            self.dataset = None
        else:
            msg = "OfflineEnvSpec requires either 'minari_dataset_id' or 'dataset_path' to be set."
            raise ValueError(msg)
        return self


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
