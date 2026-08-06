# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import pandas as pd
from pettingzoo import ParallelEnv
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)
from typing_extensions import Self

from agilerl.models.env_types import LLMEnvType
from agilerl.protocols import BanditEnvProtocol
from agilerl.typing import EnvFactory, WrapperSpec
from agilerl.utils.env_utils import (
    GymEnvType,
    apply_wrappers,
    get_reward_fn,
    make_conversation_template,
    resolve_entrypoint_target,
)
from agilerl.vector import AsyncPettingZooVecEnv

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from agilerl.llm_envs import TokenObservationWrapper
    from agilerl.protocols import MultiTurnEnv
    from agilerl.wrappers.llm_envs import PreferenceGym, ReasoningGym, SFTGym


def _require_datasets() -> tuple[type[Dataset], Callable[..., Any]]:
    """Import HuggingFace ``datasets`` (provided by the ``agilerl[llm]`` extra)."""
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        msg = (
            "The 'datasets' package is required for LLM environments. "
            "Install with: pip install 'agilerl[llm]'"
        )
        raise ImportError(msg) from exc
    return Dataset, load_dataset


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
    def construct_custom_env_fn(
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
        :rtype: EnvFactory
        """

        def default_make_env() -> gym.Env:
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
            return self.construct_custom_env_fn(
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
            make_env = self.construct_custom_env_fn(
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
    def construct_custom_env_fn(
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
            return self.construct_custom_env_fn(
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
            make_env = self.construct_custom_env_fn(
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
    :param dataset: Path to a Parquet dataset file or a HuggingFace dataset.
        Required for reasoning/preference/sft environments.
    :type dataset: str
    :param env_name: GEM environment id (e.g. ``"game:Sudoku-v0-easy"``).
        Mutually exclusive with ``entrypoint``.
    :type env_name: str | None
    :param entrypoint: Dotted path to a callable that returns a
        :class:`~agilerl.protocols.MultiTurnEnv`.  Mutually exclusive with
        ``env_name``.
    :type entrypoint: str | None
    :param env_config: Keyword arguments forwarded to the entrypoint callable.
        Only used when ``entrypoint`` is set.
    :type env_config: dict[str, Any] | None
    :param max_turns: Maximum interaction turns per episode.  If ``None``
        for multiturn environments, the value is probed from the environment.
    :type max_turns: int | None
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_type: LLMEnvType
    dataset: str | None = Field(
        default=None, validation_alias=AliasChoices("dataset", "name")
    )
    columns: dict[str, str] | None = Field(default=None)
    prompt_template: dict[str, Any] | None = Field(default=None)
    max_reward: float | None = Field(default=None)
    train_test_split: float = Field(default=0.9, ge=0.0, le=1.0)
    reward_file_path: str | None = Field(default=None)
    reward_fn_name: str | None = Field(default=None)
    response_column: str = Field(default="response")

    # Multi-turn specific fields
    env_name: str | None = Field(default=None)
    entrypoint: str | None = Field(default=None)
    env_config: dict[str, Any] | None = Field(default=None)
    max_turns: int | None = Field(default=None, ge=1)

    # These fields are overridden given the rest of the training configuration
    data_batch_size_per_gpu: int = Field(default=8, ge=1, exclude=True)
    return_raw_completions: bool = Field(default=False, exclude=True)
    max_context_length: int | None = Field(default=None, exclude=True)
    seed: int | None = Field(default=None, exclude=True)

    @property
    def name(self) -> str:
        """Human-readable name: dataset path, GEM env name, or entrypoint."""
        if self.dataset is not None:
            return self.dataset
        if self.env_name is not None:
            return self.env_name
        return self.entrypoint or "multiturn"

    def _seed_kwargs(self) -> dict[str, int]:
        """Seed kwarg for gym construction; omitted when unset so the gym default applies."""
        return {} if self.seed is None else {"seed": self.seed}

    @model_validator(mode="after")
    def _validate_reasoning_fields(self) -> Self:
        if self.env_type == LLMEnvType.REASONING:
            if self.dataset is None:
                msg = "dataset is required for reasoning environments"
                raise ValueError(msg)
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
        if self.env_type == LLMEnvType.PREFERENCE:
            if self.dataset is None:
                msg = "dataset is required for preference environments"
                raise ValueError(msg)
            if self.reward_file_path is not None:
                msg = "Reward file path has been specified, but is not supported for preference environments."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_sft_fields(self) -> Self:
        if self.env_type == LLMEnvType.SFT:
            if self.dataset is None:
                msg = "dataset is required for SFT environments"
                raise ValueError(msg)
            if self.reward_file_path is not None:
                msg = "Reward file path has been specified, but is not supported for SFT environments."
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_multiturn_fields(self) -> Self:
        if self.env_type == LLMEnvType.MULTITURN:
            has_gem = self.env_name is not None
            has_ep = self.entrypoint is not None
            if has_gem == has_ep:
                msg = (
                    "Exactly one of env_name or entrypoint is required "
                    "for multiturn environments."
                )
                raise ValueError(msg)
            if self.env_config is not None and not has_ep:
                msg = "env_config is only used with entrypoint, not env_name."
                raise ValueError(msg)
        return self

    def _load_dataset_hf(self) -> tuple[Dataset, Dataset]:
        """Load the HuggingFace dataset.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        dataset = self.dataset
        if dataset is None:
            msg = "dataset is required to load reasoning/preference/sft data"
            raise ValueError(msg)
        _, load_dataset = _require_datasets()
        ds = load_dataset(dataset, split="train").shuffle(seed=self.seed)
        if self.columns:
            ds = ds.rename_columns(self.columns)

        split = ds.train_test_split(test_size=1.0 - self.train_test_split)
        return split["train"], split["test"]

    def _load_dataset_file(self) -> tuple[Dataset, Dataset]:
        """Load and split the Parquet dataset into train/test.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        dataset = self.dataset
        if dataset is None:
            msg = "dataset is required to load reasoning/preference/sft data"
            raise ValueError(msg)
        Dataset, _ = _require_datasets()
        df = pd.read_parquet(dataset)
        if self.columns:
            df = df.rename(columns=self.columns)

        ds = Dataset.from_pandas(df)
        split = ds.train_test_split(test_size=1.0 - self.train_test_split)
        return split["train"], split["test"]

    def _load_dataset(self) -> tuple[Dataset, Dataset]:
        """Load the dataset.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        dataset = self.dataset
        if dataset is None:
            msg = "dataset is required to load reasoning/preference/sft data"
            raise ValueError(msg)
        if dataset.endswith((".parquet", ".pq")):
            return self._load_dataset_file()
        return self._load_dataset_hf()

    def make_env(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> ReasoningGym | PreferenceGym | SFTGym:
        """Make the environment for the LLM agent.

        For multiturn environments, use :meth:`make_multiturn_env_factory`
        instead — the training loop needs a factory, not a single env.

        :param tokenizer: The tokenizer.
        :type tokenizer: PreTrainedTokenizerBase
        :return: The reasoning or preference gym environment.
        :rtype: ReasoningGym | PreferenceGym | SFTGym
        """
        if self.env_type == LLMEnvType.MULTITURN:
            msg = (
                "Multiturn environments cannot be constructed with make_env(). "
                "Use make_multiturn_env_factory() instead."
            )
            raise TypeError(msg)

        train_ds, test_ds = self._load_dataset()

        if self.env_type == LLMEnvType.REASONING:
            return self._make_reasoning_env(train_ds, test_ds, tokenizer)
        if self.env_type == LLMEnvType.PREFERENCE:
            return self._make_preference_env(train_ds, test_ds, tokenizer)
        if self.env_type == LLMEnvType.SFT:
            return self._make_sft_env(train_ds, test_ds, tokenizer)
        msg = f"Invalid environment type: {self.env_type}"
        raise ValueError(msg)

    def make_multiturn_env_factory(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_model_len: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Callable[[], TokenObservationWrapper]:
        """Build a factory that creates wrapped multi-turn env instances.

        Each call to the returned factory creates a fresh
        :class:`~agilerl.llm_envs.TokenObservationWrapper`.  The underlying
        environment is either a GEM environment (``env_name``) or a custom
        class resolved from ``entrypoint``.

        If :attr:`max_turns` is ``None``, it is probed from a temporary
        environment instance and stored back on the spec.

        :param tokenizer: The tokenizer (shared across all instances).
        :type tokenizer: PreTrainedTokenizerBase
        :param max_model_len: Maximum model context length for sliding-window
            prompt truncation inside the wrapper.
        :type max_model_len: int | None
        :param max_output_tokens: Maximum newly generated tokens per turn.
        :type max_output_tokens: int | None
        :returns: A zero-argument callable that creates a wrapped env.
        :rtype: Callable[[], TokenObservationWrapper]
        """
        from agilerl.llm_envs import TokenObservationWrapper

        if self.env_name is not None:
            try:
                # gem-llm is an optional runtime dependency, never installed for checks.
                import gem  # ty: ignore[unresolved-import]
            except ImportError:
                msg = (
                    f"The 'gem-llm' package is required to use env_name={self.env_name!r}. "
                    "Install it with: pip install gem-llm"
                )
                raise ImportError(msg) from None

            env_name = self.env_name

            def _make_raw_env() -> MultiTurnEnv:
                return gem.make(env_name)
        else:
            if self.entrypoint is None:
                msg = (
                    "Exactly one of env_name or entrypoint is required "
                    "for multiturn environments."
                )
                raise ValueError(msg)
            constructor = resolve_entrypoint_target(self.entrypoint)
            if not callable(constructor):
                msg = f"Entrypoint '{self.entrypoint}' resolved to non-callable object."
                raise TypeError(msg)
            cfg = self.env_config or {}

            def _make_raw_env() -> MultiTurnEnv:
                return constructor(**cfg)

        max_turns = self.max_turns
        if max_turns is None:
            probe = _make_raw_env()
            max_turns = probe.max_turns
            if hasattr(probe, "close"):
                probe.close()
            self.max_turns = max_turns

        pad_id = tokenizer.pad_token_id

        def _factory() -> TokenObservationWrapper:
            env = _make_raw_env()
            return TokenObservationWrapper(
                env=env,
                tokenizer=tokenizer,
                max_turns=max_turns,
                pad_id=pad_id,
                apply_chat_template=True,
                max_model_len=max_model_len,
                max_output_tokens=max_output_tokens,
            )

        return _factory

    def _make_reasoning_env(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
    ) -> ReasoningGym:
        """Make the reasoning gym environment.

        :param train_dataset: The training dataset.
        :type train_dataset: Dataset
        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :param tokenizer: The tokenizer.
        :type tokenizer: PreTrainedTokenizerBase
        :return: The reasoning gym environment.
        :rtype: ReasoningGym
        """
        from agilerl.wrappers.llm_envs import ReasoningGym

        if (
            self.reward_fn_name is None
            or self.reward_file_path is None
            or self.prompt_template is None
        ):
            msg = (
                "reward_fn_name, reward_file_path, and prompt_template are "
                "required for reasoning environments"
            )
            raise ValueError(msg)

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
            max_context_length=self.max_context_length,
            return_raw_completions=self.return_raw_completions,
            **self._seed_kwargs(),
        )

    def _make_preference_env(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
    ) -> PreferenceGym:
        """Make the environment for the LLM agent.

        :param train_dataset: The training dataset.
        :type train_dataset: Dataset
        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :param tokenizer: The tokenizer.
        :type tokenizer: PreTrainedTokenizerBase
        :return: The preference gym environment.
        :rtype: PreferenceGym
        """
        from agilerl.wrappers.llm_envs import PreferenceGym

        return PreferenceGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=self.data_batch_size_per_gpu,
            max_context_length=self.max_context_length,
            **self._seed_kwargs(),
        )

    def _make_sft_env(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
    ) -> SFTGym:
        """Make the SFT gym environment.

        :param train_dataset: The training dataset.
        :type train_dataset: Dataset
        :param test_dataset: The test dataset.
        :type test_dataset: Dataset
        :param tokenizer: The tokenizer.
        :type tokenizer: PreTrainedTokenizerBase
        :return: The SFT gym environment.
        :rtype: SFTGym
        """
        from agilerl.wrappers.llm_envs import SFTGym

        return SFTGym(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            data_batch_size_per_gpu=self.data_batch_size_per_gpu,
            response_column=self.response_column,
            max_context_length=self.max_context_length,
            **self._seed_kwargs(),
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
        if features is None or targets is None:
            msg = "Both 'features' and 'targets' are required for dataset-mode bandit environments."
            raise ValueError(msg)
        return BanditEnv(features=features, targets=targets)
