# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

from agilerl.llm_envs import DatasetEnv, RolloutHarness
from agilerl.llm_envs.env_packages import ensure_importable, package_list
from agilerl.llm_envs.env_specs import name_source
from agilerl.models.env_types import LLMEnvType
from agilerl.protocols import BanditEnvProtocol
from agilerl.typing import EnvFactory, WrapperSpec
from agilerl.utils.env_utils import (
    GymEnvType,
    apply_wrappers,
    get_rubric_factory,
    make_conversation_template,
    resolve_entrypoint_target,
)
from agilerl.utils.llm_utils import render_chat_template
from agilerl.utils.utils import make_multi_agent_vect_envs, make_vect_envs
from agilerl.vector import AsyncPettingZooVecEnv
from agilerl.wrappers.learning import BanditEnv

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from agilerl.protocols import TextEnvProtocol


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
    """Environment specification for LLM training.

    Captures what is needed to build either a
    :class:`~agilerl.llm_envs.RolloutHarness` (generative) or a
    :class:`~agilerl.llm_envs.DatasetEnv` (teacher-forced).

    A ``rollout`` env comes from exactly one source:

    * ``dataset`` + ``rubric_file_path`` / ``rubric_name`` / ``prompt_template``
      -- labelled ``(question, answer)`` rows scored by an OpenEnv rubric
      (single-turn; ``max_turns`` is 1).
    * ``entrypoint`` -- a dotted path to a callable returning a text env, with
      ``env_config`` as its keyword arguments (a library's factory works as-is,
      e.g. ``entrypoint: gem:make`` with ``env_config: {env_id: game:Sudoku-v0-easy}``).
    * ``env_url`` -- a URL to an already-hosted OpenEnv service (driven over
      HTTP); ``max_turns`` must be set since it can't be probed remotely.

    ``entrypoint`` runs **in-process**, with ``env_packages`` naming anything it
    needs installed first. To run an env elsewhere, host it as an OpenEnv
    service (a container, a Space, or a server) and point ``env_url`` at it.

    A ``dataset`` env always comes from ``dataset`` and an ``objective``.

    :param env_type: ``"rollout"`` (generative) or ``"dataset"`` (teacher-forced).
    :type env_type: LLMEnvType
    :param objective: Teacher-forced objective, required when
        ``env_type="dataset"``: ``"preference"`` (DPO) or ``"sft"``.
    :type objective: Literal["preference", "sft"] | None
    :param columns: Optional mapping from source dataset column names to the
        names expected downstream (e.g. ``{"nums": "question", "target": "answer"}``).
    :type columns: dict[str, str] | None
    :param prompt_template: Chat-template configuration rendered into the prompt
        served on reset for a dataset-backed rollout env.
    :type prompt_template: dict[str, Any] | None
    :param chat_template_kwargs: Extra kwargs for every ``apply_chat_template``
        render (e.g. ``{"enable_thinking": False}``).
    :type chat_template_kwargs: dict[str, Any]
    :param max_reward: Maximum achievable reward, forwarded to the LLM
        training loop for accuracy logging.
    :type max_reward: float | None
    :param train_test_split: Fraction of the dataset used for training.
    :type train_test_split: float
    :param rubric_file_path: Path to a Python file containing the rubric
        (a ``Rubric`` instance or subclass, or a reward callable wrapped
        by :func:`~agilerl.llm_envs.rubrics.reward_fn_to_rubric`). Required for
        a dataset-backed rollout env. Accepts the alias ``reward_file_path``.
    :type rubric_file_path: str | None
    :param rubric_name: Name of the rubric (or reward callable) symbol in
        ``rubric_file_path``. Accepts the alias ``reward_fn_name``.
    :type rubric_name: str | None
    :param dataset: Path to a Parquet dataset file or a HuggingFace dataset.
    :type dataset: str | None
    :param entrypoint: Dotted path to a callable returning a text env.
    :type entrypoint: str | None
    :param env_config: Keyword arguments forwarded to the entrypoint callable.
    :type env_config: dict[str, Any] | None
    :param env_packages: What the entrypoint needs installed, as ``{"uv": [...]}``
        or ``{"pip": [...]}``. Installed into the environment training runs in
        the first time the env is not importable.
    :type env_packages: dict[str, Any] | None
    :param max_turns: Maximum interaction turns per episode. If ``None`` for an
        env-backed rollout, the value is probed from the environment.
    :type max_turns: int | None
    :param env_url: URL of an already-hosted OpenEnv env service, driven over
        HTTP. Mutually exclusive with the other sources; requires ``max_turns``.
    :type env_url: str | None
    :param mcp_tool: For an MCP-backed ``env_url``, the tool the model's text is
        sent to. Only applies to ``env_url``.
    :type mcp_tool: str | None
    :param action_field: The action field an env puts the model's text in.
        OpenEnv envs name it themselves — ``message`` by default, but ``code``
        or ``action_str`` elsewhere — and for an MCP tool it is the argument
        name.
    :type action_field: str
    :param observation_field: The field an env's observation carries its text
        in. Unset reads the specified shapes and, failing those, the single
        text field the observation carries — which warns, naming what it found.
    :type observation_field: str | None
    :param observation_processor: ``module:fn`` / ``path.py:fn`` entrypoint
        naming a callable that renders an observation payload to prompt text,
        for envs whose observations need more than a field lookup (composite
        renderings, board states). Mutually exclusive with
        ``observation_field``.
    :type observation_processor: str | None
    :param request_timeout_s: Per-message client timeout in seconds for an
        ``env_url``. ``None`` (the default) applies a 300 s bound; ``0``
        disables the bound (e.g. an env step that legitimately runs a very
        long tool job). Only applies to ``env_url``.
    :type request_timeout_s: float | None
    :param strict_chat_template_boundary: When ``True``, a chat template that
        cannot render a multi-turn boundary raises; when ``False``, it warns
        and falls back to ChatML markers.
    :type strict_chat_template_boundary: bool
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_type: LLMEnvType
    objective: Literal["preference", "sft"] | None = Field(default=None)
    dataset: str | None = Field(
        default=None, validation_alias=AliasChoices("dataset", "name")
    )
    columns: dict[str, str] | None = Field(default=None)
    prompt_template: dict[str, Any] | None = Field(default=None)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    max_reward: float | None = Field(default=None)
    train_test_split: float = Field(default=0.9, ge=0.0, le=1.0)
    rubric_file_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("rubric_file_path", "reward_file_path"),
    )
    rubric_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("rubric_name", "reward_fn_name"),
    )
    response_column: str = Field(default="response")

    # Env-backed rollout fields
    entrypoint: str | None = Field(default=None)
    env_config: dict[str, Any] | None = Field(default=None)
    env_packages: dict[str, Any] | None = Field(default=None)
    max_turns: int | None = Field(default=None, ge=1)
    strict_chat_template_boundary: bool = Field(default=True)

    # How an env-backed rollout's observations render to prompt text.
    observation_field: str | None = Field(default=None)
    observation_processor: str | None = Field(default=None)

    # Remote rollout fields: an already-hosted OpenEnv service driven over /ws.
    env_url: str | None = Field(default=None)
    mcp_tool: str | None = Field(default=None)
    action_field: str = Field(default="message")
    request_timeout_s: float | None = Field(default=None, ge=0.0)

    # These fields are overridden given the rest of the training configuration
    data_batch_size_per_gpu: int = Field(default=8, ge=1, exclude=True)
    max_context_length: int | None = Field(default=None, exclude=True)
    seed: int | None = Field(default=None, exclude=True)

    @property
    def name(self) -> str:
        """Human-readable name: dataset path, URL, or entrypoint."""
        if self.dataset is not None:
            return self.dataset
        if self.env_url is not None:
            return self.env_url
        return self.entrypoint or "rollout"

    @property
    def dataset_backed_rollout(self) -> bool:
        """Whether this rollout env serves labelled rows scored by a reward fn."""
        return self.env_type == LLMEnvType.ROLLOUT and self.dataset is not None

    @property
    def _split_seed(self) -> int:
        """Seed for the shuffle and train/test split, defaulting when unset.

        Every rank loads the dataset for itself and the rollout task assigner
        hands out rows *by index*, so an unseeded split would leave ranks
        disagreeing about which row is row ``n`` — their shards would overlap
        instead of partitioning, and each would hold out a different test set.
        """
        return self.seed if self.seed is not None else 42

    def _http_timeout_s(self) -> float | None:
        """Per-message ``env_url`` timeout: manifest value, 300 s default, ``0`` unbounds."""
        if self.request_timeout_s is None:
            return 300.0
        return self.request_timeout_s or None

    @model_validator(mode="after")
    def _validate_rollout_fields(self) -> Self:
        if self.env_type != LLMEnvType.ROLLOUT:
            return self
        name_source(
            dataset=self.dataset,
            entrypoint=self.entrypoint,
            env_url=self.env_url,
        )
        if self.env_config is not None and self.entrypoint is None:
            msg = "env_config is only used with entrypoint."
            raise ValueError(msg)
        if self.env_packages is not None:
            if self.entrypoint is None:
                msg = (
                    "env_packages installs the dependencies of an entrypoint env; "
                    "a dataset or env_url source has nothing to install for."
                )
                raise ValueError(msg)
            package_list(self.env_packages)
        if (
            self.mcp_tool is not None or self.request_timeout_s is not None
        ) and self.env_url is None:
            msg = (
                "mcp_tool / request_timeout_s only apply to a remote rollout "
                "env; set env_url to the hosted OpenEnv service."
            )
            raise ValueError(msg)
        if self.action_field != "message" and self.dataset is not None:
            msg = (
                "action_field names the field an env receives the model's text "
                "in; a dataset-backed rollout has no such env."
            )
            raise ValueError(msg)
        if (
            self.observation_field is not None
            and self.observation_processor is not None
        ):
            msg = (
                "observation_field and observation_processor are mutually "
                "exclusive: the field names where the default processor reads "
                "the text, the processor replaces that default."
            )
            raise ValueError(msg)
        if (
            self.observation_field is not None or self.observation_processor is not None
        ) and self.dataset is not None:
            msg = (
                "observation_field / observation_processor render an env's "
                "observations; a dataset-backed rollout builds its own prompts."
            )
            raise ValueError(msg)
        if self.env_url is not None and self.max_turns is None:
            msg = (
                "max_turns is required with env_url: a remote env's turn budget "
                "cannot be probed, so set it explicitly (1 for single-turn)."
            )
            raise ValueError(msg)
        if self.dataset is not None:
            if self.rubric_file_path is None:
                msg = "rubric_file_path is required for dataset-backed rollout environments"
                raise ValueError(msg)
            if self.rubric_name is None:
                msg = "rubric_name is required for dataset-backed rollout environments"
                raise ValueError(msg)
            if self.prompt_template is None:
                msg = "Prompt template is required for dataset-backed rollout environments"
                raise ValueError(msg)
            if self.max_turns not in (None, 1):
                msg = "A dataset-backed rollout environment is single-turn; max_turns must be 1."
                raise ValueError(msg)
            self.max_turns = 1
        elif self.rubric_file_path is not None:
            msg = (
                "rubric_file_path is not supported for env-backed rollout environments."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_dataset_fields(self) -> Self:
        if self.env_type != LLMEnvType.DATASET:
            return self
        if self.dataset is None:
            msg = "dataset is required for dataset environments"
            raise ValueError(msg)
        if self.objective is None:
            msg = "objective is required for dataset environments"
            raise ValueError(msg)
        if self.rubric_file_path is not None:
            msg = "rubric_file_path has been specified, but is not supported for dataset environments."
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
        ds = load_dataset(dataset, split="train").shuffle(seed=self._split_seed)
        if self.columns:
            ds = ds.rename_columns(self.columns)

        split = ds.train_test_split(
            test_size=1.0 - self.train_test_split, seed=self._split_seed
        )
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
        split = ds.train_test_split(
            test_size=1.0 - self.train_test_split, seed=self._split_seed
        )
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

    def make_dataset_env(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> DatasetEnv:
        """Make the teacher-forced dataset environment for the LLM agent.

        Named for what it builds: the sibling :meth:`make_rollout_env_factory`
        covers the generative half, and the training loop needs a factory there
        rather than a single env.

        :param tokenizer: The tokenizer.
        :type tokenizer: PreTrainedTokenizerBase
        :param rank: This process's data-parallel shard index (from the runtime).
        :type rank: int
        :param world_size: Number of data-parallel shards.
        :type world_size: int
        :return: The dataset environment.
        :rtype: DatasetEnv
        """
        if self.env_type != LLMEnvType.DATASET:
            msg = (
                "Rollout environments cannot be constructed with "
                "make_dataset_env(). Use make_rollout_env_factory() instead."
            )
            raise TypeError(msg)

        if self.objective is None:
            msg = "objective is required for dataset environments."
            raise ValueError(msg)

        train_ds, test_ds = self._load_dataset()
        return DatasetEnv(
            train_dataset=train_ds,
            test_dataset=test_ds,
            tokenizer=tokenizer,
            objective=self.objective,
            response_column=self.response_column,
            chat_template_kwargs=self.chat_template_kwargs,
            data_batch_size_per_gpu=self.data_batch_size_per_gpu,
            max_context_length=self.max_context_length,
            seed=self.seed if self.seed is not None else 42,
            rank=rank,
            world_size=world_size,
        )

    def make_rollout_env_factory(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_model_len: int | None = None,
    ) -> Callable[[], RolloutHarness]:
        """Build a factory that creates fresh :class:`RolloutHarness` instances.

        Each call to the returned factory creates an independent env, so
        concurrent trajectories never share state. Which of the three builders
        below runs is decided by the one source the spec names — dataset rows,
        an ``env_url``, or an ``entrypoint``.

        :param tokenizer: The tokenizer (shared across all instances).
        :type tokenizer: PreTrainedTokenizerBase
        :param max_model_len: Maximum model context length for prompt truncation.
        :type max_model_len: int | None
        :returns: A zero-argument callable that creates a ``RolloutHarness``.
        :rtype: Callable[[], RolloutHarness]
        """
        if self.env_type != LLMEnvType.ROLLOUT:
            msg = (
                "Dataset environments cannot be constructed with "
                "make_rollout_env_factory(). Use make_dataset_env() instead."
            )
            raise TypeError(msg)

        harness_kwargs: dict[str, Any] = {
            "max_model_len": max_model_len,
            "chat_template_kwargs": dict(self.chat_template_kwargs),
        }
        if self.dataset_backed_rollout:
            return self._make_dataset_rollout_factory(tokenizer, harness_kwargs)
        if self.env_url is not None:
            return self._make_url_rollout_factory(tokenizer, harness_kwargs)
        return self._make_entrypoint_rollout_factory(tokenizer, harness_kwargs)

    def _make_dataset_rollout_factory(
        self,
        tokenizer: PreTrainedTokenizerBase,
        harness_kwargs: dict[str, Any],
    ) -> Callable[[], RolloutHarness]:
        """Factory serving labelled dataset rows scored by a rubric, in-process.

        :param tokenizer: The tokenizer, shared across every built env.
        :type tokenizer: PreTrainedTokenizerBase
        :param harness_kwargs: Context-budget and chat-template settings common to
            all three builders.
        :type harness_kwargs: dict[str, Any]
        :rtype: Callable[[], RolloutHarness]
        """
        chat_template_kwargs = harness_kwargs["chat_template_kwargs"]
        # The rubric file and the dataset are read once, on the first env
        # build, so constructing the factory stays free of I/O.
        resolved: dict[str, Any] = {}

        def _resolve() -> dict[str, Any]:
            if not resolved:
                if (
                    self.prompt_template is None
                    or self.rubric_name is None
                    or self.rubric_file_path is None
                ):
                    msg = (
                        "rubric_name, rubric_file_path, and prompt_template "
                        "are required for dataset-backed rollout environments"
                    )
                    raise ValueError(msg)

                conversation = make_conversation_template(
                    prompt_template=self.prompt_template
                )

                def _prompt_builder(row: Mapping[str, Any]) -> str:
                    """Render the manifest's chat template for one dataset row."""
                    return render_chat_template(
                        conversation,
                        tokenizer,
                        chat_template_kwargs=chat_template_kwargs,
                        **row,
                    )

                train_ds, test_ds = self._load_dataset()
                resolved.update(
                    train_ds=train_ds,
                    test_ds=test_ds,
                    prompt_builder=_prompt_builder,
                    rubric_factory=get_rubric_factory(
                        rubric_name=self.rubric_name,
                        file_path=self.rubric_file_path,
                    ),
                )
            return resolved

        def _dataset_factory() -> RolloutHarness:
            parts = _resolve()
            return RolloutHarness.from_dataset(
                parts["train_ds"],
                # One rubric per env: the collector steps its slots on
                # concurrent threads, and a rubric carries scoring state.
                parts["rubric_factory"](),
                tokenizer,
                test_dataset=parts["test_ds"],
                prompt_builder=parts["prompt_builder"],
                # ``prompt_builder`` already rendered the chat template.
                apply_chat_template=False,
                max_model_len=harness_kwargs["max_model_len"],
            )

        return _dataset_factory

    def _make_url_rollout_factory(
        self,
        tokenizer: PreTrainedTokenizerBase,
        harness_kwargs: dict[str, Any],
    ) -> Callable[[], RolloutHarness]:
        """Factory dialling an env that is already running at ``env_url``.

        :param tokenizer: The tokenizer, shared across every built env.
        :type tokenizer: PreTrainedTokenizerBase
        :param harness_kwargs: Context-budget and chat-template settings common to
            all three builders.
        :type harness_kwargs: dict[str, Any]
        :rtype: Callable[[], RolloutHarness]
        """
        from agilerl.llm_envs.openenv import RemoteEnvClient  # optional extra: llm

        url = self.env_url
        if url is None:
            msg = "env_url is required to dial a remote rollout env"
            raise RuntimeError(msg)
        # A remote env's turn budget can't be probed; validation requires max_turns.
        url_max_turns = self.max_turns or 1
        mcp_tool = self.mcp_tool
        action_field = self.action_field
        timeout_s = self._http_timeout_s()
        observation_field = self.observation_field
        observation_processor = self._resolved_observation_processor()
        strict_boundary = self.strict_chat_template_boundary

        def _url_factory() -> RolloutHarness:
            # The server's max_concurrent_envs must cover batch*group + the eval env.
            return RolloutHarness(
                RemoteEnvClient(
                    url,
                    mcp_tool=mcp_tool,
                    action_field=action_field,
                    timeout_s=timeout_s,
                ),
                tokenizer,
                max_turns=url_max_turns,
                observation_field=observation_field,
                observation_processor=observation_processor,
                strict_chat_template_boundary=strict_boundary,
                apply_chat_template=True,
                **harness_kwargs,
            )

        return _url_factory

    def _make_entrypoint_rollout_factory(
        self,
        tokenizer: PreTrainedTokenizerBase,
        harness_kwargs: dict[str, Any],
    ) -> Callable[[], RolloutHarness]:
        """Factory importing ``entrypoint`` and driving the env it builds in-process.

        An unset :attr:`max_turns` is probed off one throwaway env and cached.

        :param tokenizer: The tokenizer, shared across every built env.
        :type tokenizer: PreTrainedTokenizerBase
        :param harness_kwargs: Context-budget and chat-template settings common to
            all three builders.
        :type harness_kwargs: dict[str, Any]
        :rtype: Callable[[], RolloutHarness]
        """
        if self.entrypoint is None:
            msg = "An entrypoint is required for env-backed rollout environments."
            raise ValueError(msg)
        if self.env_packages is not None:
            ensure_importable(self.entrypoint, self.env_packages)
        constructor = resolve_entrypoint_target(self.entrypoint)
        if not callable(constructor):
            msg = f"Entrypoint '{self.entrypoint}' resolved to non-callable object."
            raise TypeError(msg)
        cfg = dict(self.env_config or {})
        system_prompt = cfg.pop("system_prompt", None)

        def _make_raw_env() -> TextEnvProtocol:
            env = constructor(**cfg)
            if system_prompt is not None:
                env.system_prompt = system_prompt
            return env

        max_turns = self.max_turns
        if max_turns is None:
            probe = _make_raw_env()
            max_turns = int(getattr(probe, "max_turns", 1) or 1)
            closer = getattr(probe, "close", None)
            if callable(closer):
                closer()
            self.max_turns = max_turns

        entry_action_field = self.action_field
        observation_field = self.observation_field
        observation_processor = self._resolved_observation_processor()
        strict_boundary = self.strict_chat_template_boundary

        def _env_factory() -> RolloutHarness:
            return RolloutHarness.local(
                _make_raw_env(),
                tokenizer,
                max_turns=max_turns,
                action_field=entry_action_field,
                observation_field=observation_field,
                observation_processor=observation_processor,
                apply_chat_template=True,
                strict_chat_template_boundary=strict_boundary,
                **harness_kwargs,
            )

        return _env_factory

    def _resolved_observation_processor(self) -> Callable[[Any], str] | None:
        """Import :attr:`observation_processor`, or ``None`` when the spec names none.

        :raises TypeError: If the dotted path resolves to something not callable.
        :rtype: Callable[[Any], str] | None
        """
        if self.observation_processor is None:
            return None
        resolved = resolve_entrypoint_target(self.observation_processor)
        if not callable(resolved):
            msg = (
                f"observation_processor '{self.observation_processor}' "
                "resolved to a non-callable object."
            )
            raise TypeError(msg)
        return resolved


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
