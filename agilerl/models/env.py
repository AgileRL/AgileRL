from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import pandas as pd
from datasets import Dataset
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from pettingzoo import ParallelEnv
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from typing_extensions import Self

from agilerl.protocols import BanditEnvProtocol
from agilerl.utils.env_utils import (
    apply_wrappers,
    get_reward_fn,
    make_conversation_template,
    resolve_entrypoint_target,
)
from agilerl.vector import AsyncPettingZooVecEnv

if TYPE_CHECKING:
    from accelerate import Accelerator

    from agilerl.utils.llm_utils import PreferenceGym, ReasoningGym


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
    num_envs: int = Field(default=16, ge=1)


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
            constructor = resolve_entrypoint_target(entrypoint, path=path)
            if not callable(constructor):
                msg = f"Entrypoint '{entrypoint}' resolved to non-callable object."
                raise TypeError(msg)
            env = constructor(**(config or {}))
            return apply_wrappers(env, wrappers, path=path)

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
            constructor = resolve_entrypoint_target(entrypoint, path=path)
            if not callable(constructor):
                msg = f"Entrypoint '{entrypoint}' resolved to non-callable object."
                raise TypeError(msg)
            env = constructor(**(config or {}))
            return apply_wrappers(env, wrappers, path=path)

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
        return apply_wrappers(env, self.wrappers, path=self.path)

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
                return apply_wrappers(env, self.wrappers, path=self.path)

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
    reward_file_path: str | None = Field(default=None)
    reward_fn_name: str | None = Field(default=None)
    dataset_path: str = Field(default="dataset.parquet")
    data_batch_size_per_gpu: int = Field(default=8, ge=1)
    max_context_length: int | None = Field(default=None)
    return_raw_completions: bool = Field(default=False)
    seed: int | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_reasoning_fields(self) -> Self:
        if self.env_type == LLMEnvType.REASONING:
            if self.reward_file_path is None:
                msg = "reward_file_path is required for reasoning environments"
                raise ValueError(msg)
            if self.reward_fn_name is None:
                msg = "reward_fn_name is required for reasoning environments"
                raise ValueError(msg)
            if self.prompt_template is None:
                msg = "Prompt template is required for reasoning environments"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_preference_fields(self) -> Self:
        if self.env_type == LLMEnvType.PREFERENCE and self.reward_file_path is not None:
            msg = "Reward file path has been specified, but is not supported for preference environments."
            raise ValueError(msg)
        return self

    def load_dataset(self) -> tuple[Dataset, Dataset]:
        """Load and split the Parquet dataset into train/test.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        df = pd.read_parquet(self.dataset_path)
        if self.columns:
            df = df.rename(columns=self.columns)

        ds = Dataset.from_pandas(df)
        split = ds.train_test_split(test_size=1.0 - self.train_test_split)
        return split["train"], split["test"]

    def make_env(
        self, tokenizer: Any, accelerator: Accelerator | None = None
    ) -> ReasoningGym | PreferenceGym:
        """Make the environment for the LLM agent.

        :param tokenizer: The tokenizer.
        :type tokenizer: Any
        :param accelerator: The accelerator.
        :type accelerator: Accelerator | None
        :return: The reasoning or preference gym environment.
        :rtype: ReasoningGym | PreferenceGym
        """
        train_ds, test_ds = self.load_dataset()

        if self.env_type == LLMEnvType.REASONING:
            return self._make_reasoning_env(train_ds, test_ds, tokenizer, accelerator)
        if self.env_type == LLMEnvType.PREFERENCE:
            return self._make_preference_env(train_ds, test_ds, tokenizer, accelerator)
        msg = f"Invalid environment type: {self.env_type}"
        raise ValueError(msg)

    def _make_reasoning_env(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: Any,
        accelerator: Accelerator | None = None,
    ) -> ReasoningGym:
        """Make the reasoning gym environment.

        :param train_dataset: The training dataset.
        :type train_dataset: Dataset
        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :param tokenizer: The tokenizer.
        :type tokenizer: Any
        :param accelerator: The accelerator.
        :type accelerator: Accelerator | None
        :return: The reasoning gym environment.
        :rtype: ReasoningGym
        """
        from agilerl.utils.llm_utils import ReasoningGym

        reward_fn = get_reward_fn(
            reward_fn_name=self.reward_fn_name, file_path=self.reward_file_path
        )
        conversation_template = make_conversation_template(
            prompt_template=self.prompt_template
        )
        return ReasoningGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            conversation_template=conversation_template,
            data_batch_size_per_gpu=self.data_batch_size_per_gpu,
            accelerator=accelerator,
            max_context_length=self.max_context_length,
            return_raw_completions=self.return_raw_completions,
            seed=self.seed,
        )

    def _make_preference_env(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: Any,
        accelerator: Accelerator | None = None,
    ) -> PreferenceGym:
        """Make the environment for the LLM agent.

        :param train_dataset: The training dataset.
        :type train_dataset: Dataset
        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :param tokenizer: The tokenizer.
        :type tokenizer: Any
        :param accelerator: The accelerator.
        :type accelerator: Accelerator | None
        :return: The preference gym environment.
        :rtype: PreferenceGym
        """
        from agilerl.utils.llm_utils import PreferenceGym

        return PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=self.data_batch_size_per_gpu,
            accelerator=accelerator,
            max_context_length=self.max_context_length,
            seed=self.seed,
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
        if self.dataset_path is None and self.minari_dataset_id is None:
            msg = "OfflineEnvSpec requires either 'minari_dataset_id' or 'dataset_path' to be set."
            raise ValueError(msg)

        return self


class BanditEnvSpec(BaseModel):
    """Environment specification for contextual bandit training.

    Supports two modes:

    **Dataset mode:** provide ``features`` and ``targets`` (as DataFrames or
    file paths) to construct a :class:`~agilerl.wrappers.learning.BanditEnv`
    from a labelled dataset.

    **Custom entrypoint mode:** provide an ``entrypoint`` (e.g.
    ``"my_module:MyBanditEnv"``) to instantiate an arbitrary bandit
    environment.  The resolved callable is invoked with ``**config``..

    Exactly one of (``features`` + ``targets``) or ``entrypoint`` must be
    provided.

    :param name: Human-readable name for the environment / dataset.
    :type name: str
    :param features: Dataset features. A pd.DataFrame or a path to a file.
    :type features: pandas.DataFrame | str | Path | None
    :param targets: Dataset targets. A pd.DataFrame or a path to a file.
    :type targets: pandas.DataFrame | str | Path | None
    :param entrypoint: Dotted path to a callable that returns a bandit
        environment (e.g. ``"my_module:MyBanditEnv"``).
    :type entrypoint: str | None
    :param path: Optional filesystem path added to ``sys.path`` before
        resolving the entrypoint.
    :type path: str | None
    :param config: Keyword arguments forwarded to the entrypoint callable.
    :type config: dict[str, Any] | None
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="BanditEnv")
    features: pd.DataFrame | str | Path | None = Field(default=None)
    targets: pd.DataFrame | str | Path | None = Field(default=None)
    entrypoint: str | None = Field(default=None)
    path: str | None = Field(default=None)
    config: dict[str, Any] | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        has_features = self.features is not None
        has_targets = self.targets is not None
        has_entrypoint = self.entrypoint is not None

        if has_features != has_targets:
            msg = "Both 'features' and 'targets' must be provided together."
            raise ValueError(msg)

        has_dataset = has_features and has_targets

        if has_dataset and has_entrypoint:
            msg = "Provide either (features + targets) or entrypoint, not both."
            raise ValueError(msg)

        if not has_dataset and not has_entrypoint:
            msg = (
                "BanditEnvSpec requires either (features + targets) for "
                "dataset-based environments, or an entrypoint for custom environments."
            )
            raise ValueError(msg)

        return self

    @field_serializer("features", "targets")
    @classmethod
    def _serialize_dataframe_fields(
        cls, v: pd.DataFrame | str | Path | None
    ) -> str | None:
        if isinstance(v, (str, Path)):
            return str(v)
        return None

    def _load_dataframe(self, value: str | Path) -> pd.DataFrame:
        """Load a DataFrame from a path.

        Supports CSV, Parquet, and HDF5 files based on file extension.

        :param value: A file path.
        :returns: A pandas DataFrame.
        :rtype: pd.DataFrame
        """
        path = Path(value) if isinstance(value, str) else value
        if path.suffix in {".parquet", ".pq"}:
            data = pd.read_parquet(path)
        elif path.suffix == ".csv":
            data = pd.read_csv(path)
        elif path.suffix in {".h5", ".hdf5"}:
            data = pd.read_hdf(path)
        else:
            msg = f"Unsupported file type: {path.suffix}"
            raise ValueError(msg)
        return data

    def make_env(self) -> BanditEnvProtocol:
        """Construct a bandit environment.

        In dataset mode, returns a :class:`~agilerl.wrappers.learning.BanditEnv`
        built from ``features`` and ``targets``.  In entrypoint mode, resolves
        the callable and invokes it with ``**config``.

        :returns: A bandit environment satisfying :class:`~agilerl.protocols.BanditEnvProtocol`.
        :rtype: BanditEnvProtocol
        """
        if self.entrypoint is not None:
            constructor = resolve_entrypoint_target(self.entrypoint, path=self.path)
            if not callable(constructor):
                msg = f"Entrypoint '{self.entrypoint}' resolved to non-callable object."
                raise TypeError(msg)
            return constructor(**(self.config or {}))

        from agilerl.wrappers.learning import BanditEnv

        features = (
            self._load_dataframe(self.features)
            if isinstance(self.features, str | Path)
            else self.features
        )
        targets = (
            self._load_dataframe(self.targets)
            if isinstance(self.targets, str | Path)
            else self.targets
        )
        return BanditEnv(features=features, targets=targets)


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
