from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import pandas as pd
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
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
    from datasets import Dataset

    from agilerl.llm_envs import DatasetEnv, RolloutEnv


def _require_datasets() -> tuple[type[Dataset], Callable[[], Dataset]]:
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


class LLMEnvType(str, Enum):
    """Type of LLM environment.

    ``ROLLOUT`` covers every generative regime (a ``RolloutEnv``): single-turn
    reasoning is simply ``max_turns=1``. ``DATASET`` covers the teacher-forced
    regimes (a ``DatasetEnv``), selected by :attr:`LLMEnvSpec.objective`.
    """

    ROLLOUT = "rollout"
    DATASET = "dataset"

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
    """Environment specification for LLM training.

    Declaratively captures what is needed to build either a
    :class:`~agilerl.llm_envs.RolloutEnv` (generative) or a
    :class:`~agilerl.llm_envs.DatasetEnv` (teacher-forced). Fields are aligned
    with what Arena expects for LLM training jobs.

    A ``rollout`` env comes from exactly one source:

    * ``dataset`` + ``reward_file_path`` / ``reward_fn_name`` / ``prompt_template``
      -- labelled ``(question, answer)`` rows scored by a reward function
      (single-turn; ``max_turns`` is 1).
    * ``env_name`` -- a GEM environment id (e.g. ``"game:Sudoku-v0-easy"``).
    * ``entrypoint`` -- a dotted path to a callable returning a text env.
    * ``env_url`` -- a URL to an already-hosted OpenEnv service (driven over
      HTTP); ``max_turns`` must be set since it can't be probed remotely.

    ``env_name`` / ``entrypoint`` run **in-process** by default; set
    ``served: true`` to instead host them on an in-process OpenEnv HTTP server
    -- the local rehearsal of the ``env_url`` production transport. Transport
    is thus a deployment concern, not a code change: the same env declaration
    trains in-process for dev and against a served endpoint in production.

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
    :param max_reward: Maximum achievable reward, forwarded to the LLM
        training loop for accuracy logging.
    :type max_reward: float | None
    :param train_test_split: Fraction of the dataset used for training.
    :type train_test_split: float
    :param reward_file_path: Path to a Python file containing the reward
        function. Required for a dataset-backed rollout env.
    :type reward_file_path: str | None
    :param dataset: Path to a Parquet dataset file or a HuggingFace dataset.
    :type dataset: str | None
    :param env_name: GEM environment id. Mutually exclusive with ``entrypoint``.
    :type env_name: str | None
    :param entrypoint: Dotted path to a callable returning a text env.
        Mutually exclusive with ``env_name``.
    :type entrypoint: str | None
    :param env_config: Keyword arguments forwarded to the entrypoint callable.
    :type env_config: dict[str, Any] | None
    :param max_turns: Maximum interaction turns per episode. If ``None`` for an
        env-backed rollout, the value is probed from the environment.
    :type max_turns: int | None
    :param env_url: URL of an already-hosted OpenEnv env service, driven over
        HTTP. Mutually exclusive with the other sources; requires ``max_turns``.
    :type env_url: str | None
    :param served: Host an ``env_name`` / ``entrypoint`` env on an in-process
        OpenEnv HTTP server instead of running it in-process. Defaults to False.
    :type served: bool
    :param mcp_tool: For an MCP-backed served env, the tool the model's text is
        sent to. Only applies to a served env (``env_url`` or ``served``).
    :type mcp_tool: str | None
    :param request_timeout_s: Per-request HTTP timeout for a served env
        (``None`` = unbounded). Only applies to a served env.
    :type request_timeout_s: float | None
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env_type: LLMEnvType
    objective: Literal["preference", "sft"] | None = Field(default=None)
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

    # Env-backed rollout fields
    env_name: str | None = Field(default=None)
    entrypoint: str | None = Field(default=None)
    env_config: dict[str, Any] | None = Field(default=None)
    max_turns: int | None = Field(default=None, ge=1)

    # Served rollout fields: drive an env over HTTP instead of in-process. Point
    # at an already-hosted OpenEnv service (``env_url``), or host an
    # ``env_name`` / ``entrypoint`` env on an in-process OpenEnv server
    # (``served``). ``mcp_tool`` / ``request_timeout_s`` tune the HTTP client.
    env_url: str | None = Field(default=None)
    served: bool = Field(default=False)
    mcp_tool: str | None = Field(default=None)
    request_timeout_s: float | None = Field(default=None, ge=0.0)

    # These fields are overridden given the rest of the training configuration
    data_batch_size_per_gpu: int = Field(default=8, ge=1, exclude=True)
    return_raw_completions: bool = Field(default=False, exclude=True)
    max_context_length: int | None = Field(default=None, exclude=True)
    seed: int | None = Field(default=None, exclude=True)

    @property
    def name(self) -> str:
        """Human-readable name: dataset path, GEM env name, URL, or entrypoint."""
        if self.dataset is not None:
            return self.dataset
        if self.env_name is not None:
            return self.env_name
        if self.env_url is not None:
            return self.env_url
        return self.entrypoint or "rollout"

    @property
    def dataset_backed_rollout(self) -> bool:
        """Whether this rollout env serves labelled rows scored by a reward fn."""
        return self.env_type == LLMEnvType.ROLLOUT and self.dataset is not None

    @model_validator(mode="after")
    def _validate_rollout_fields(self) -> Self:
        if self.env_type != LLMEnvType.ROLLOUT:
            return self
        sources = [self.dataset, self.env_name, self.entrypoint, self.env_url]
        if sum(source is not None for source in sources) != 1:
            msg = (
                "Exactly one of dataset, env_name, entrypoint or env_url is "
                "required for rollout environments."
            )
            raise ValueError(msg)
        if self.env_config is not None and self.entrypoint is None:
            msg = "env_config is only used with entrypoint, not env_name."
            raise ValueError(msg)
        if self.served and self.env_name is None and self.entrypoint is None:
            msg = (
                "served=True hosts an env_name / entrypoint env over HTTP; it "
                "does not apply to a dataset or an already-served env_url."
            )
            raise ValueError(msg)
        transport_is_http = self.env_url is not None or self.served
        if (
            self.mcp_tool is not None or self.request_timeout_s is not None
        ) and not transport_is_http:
            msg = (
                "mcp_tool / request_timeout_s only apply to a served rollout "
                "env (set env_url, or served=True on an env_name / entrypoint)."
            )
            raise ValueError(msg)
        if self.env_url is not None and self.max_turns is None:
            msg = (
                "max_turns is required with env_url: a remote env's turn budget "
                "cannot be probed, so set it explicitly (1 for single-turn)."
            )
            raise ValueError(msg)
        if self.dataset is not None:
            if self.reward_file_path is None:
                msg = "reward_file_path is required for dataset-backed rollout environments"
                raise ValueError(msg)
            if self.reward_fn_name is None:
                msg = (
                    "reward_fn_name is required for dataset-backed rollout environments"
                )
                raise ValueError(msg)
            if self.prompt_template is None:
                msg = "Prompt template is required for dataset-backed rollout environments"
                raise ValueError(msg)
            if self.max_turns not in (None, 1):
                msg = "A dataset-backed rollout environment is single-turn; max_turns must be 1."
                raise ValueError(msg)
            self.max_turns = 1
        elif self.reward_file_path is not None:
            msg = (
                "Reward file path is not supported for env-backed rollout environments."
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
        if self.reward_file_path is not None:
            msg = "Reward file path has been specified, but is not supported for dataset environments."
            raise ValueError(msg)
        return self

    def _load_dataset_hf(self) -> tuple[Dataset, Dataset]:
        """Load the HuggingFace dataset.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        _, load_dataset = _require_datasets()
        ds = load_dataset(self.dataset, split="train").shuffle(seed=self.seed)
        if self.columns:
            ds = ds.rename_columns(self.columns)

        split = ds.train_test_split(test_size=1.0 - self.train_test_split)
        return split["train"], split["test"]

    def _load_dataset_file(self) -> tuple[Dataset, Dataset]:
        """Load and split the Parquet dataset into train/test.

        :returns: A ``(train_dataset, test_dataset)`` tuple.
        :rtype: tuple[Dataset, Dataset]
        """
        Dataset, _ = _require_datasets()
        df = pd.read_parquet(self.dataset)
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
        if self.dataset.endswith((".parquet", ".pq")):
            return self._load_dataset_file()
        return self._load_dataset_hf()

    def make_env(
        self, tokenizer: Any, accelerator: Accelerator | None = None
    ) -> DatasetEnv:
        """Make the teacher-forced dataset environment for the LLM agent.

        For rollout environments, use :meth:`make_rollout_env_factory` instead —
        the training loop needs a factory, not a single env.

        :param tokenizer: The tokenizer.
        :type tokenizer: Any
        :param accelerator: The accelerator.
        :type accelerator: Accelerator | None
        :return: The dataset environment.
        :rtype: DatasetEnv
        """
        from agilerl.llm_envs import DatasetEnv

        if self.env_type != LLMEnvType.DATASET:
            msg = (
                "Rollout environments cannot be constructed with make_env(). "
                "Use make_rollout_env_factory() instead."
            )
            raise TypeError(msg)

        train_ds, test_ds = self._load_dataset()
        return DatasetEnv(
            train_dataset=train_ds,
            test_dataset=test_ds,
            tokenizer=tokenizer,
            objective=self.objective,
            response_column=self.response_column,
            data_batch_size_per_gpu=self.data_batch_size_per_gpu,
            accelerator=accelerator,
            max_context_length=self.max_context_length,
            seed=self.seed if self.seed is not None else 42,
        )

    def make_rollout_env_factory(
        self,
        tokenizer: Any,
        *,
        max_model_len: int | None = None,
        max_output_tokens: int | None = None,
    ) -> Callable[[], RolloutEnv]:
        """Build a factory that creates fresh :class:`RolloutEnv` instances.

        Each call to the returned factory creates an independent env, so
        concurrent trajectories never share state. The source is either labelled
        dataset rows plus a reward function, a GEM environment (``env_name``), or
        a custom callable resolved from ``entrypoint``.

        For an env-backed rollout, if :attr:`max_turns` is ``None`` it is probed
        from a temporary environment instance and stored back on the spec.

        :param tokenizer: The tokenizer (shared across all instances).
        :type tokenizer: Any
        :param max_model_len: Maximum model context length for prompt truncation.
        :type max_model_len: int | None
        :param max_output_tokens: Maximum newly generated tokens per turn.
        :type max_output_tokens: int | None
        :returns: A zero-argument callable that creates a ``RolloutEnv``.
        :rtype: Callable[[], RolloutEnv]
        """
        from agilerl.llm_envs import RolloutEnv

        if self.env_type != LLMEnvType.ROLLOUT:
            msg = (
                "Dataset environments cannot be constructed with "
                "make_rollout_env_factory(). Use make_env() instead."
            )
            raise TypeError(msg)

        pad_id = tokenizer.pad_token_id

        if self.dataset_backed_rollout:
            # The reward file and the dataset are read once, on the first env
            # build, so constructing the factory stays free of I/O.
            resolved: dict[str, Any] = {}

            def _resolve() -> dict[str, Any]:
                if not resolved:
                    from agilerl.utils.llm_utils import render_chat_template

                    conversation = make_conversation_template(
                        prompt_template=self.prompt_template
                    )

                    def _prompt_builder(row: Any) -> str:
                        """Render the manifest's chat template for one dataset row."""
                        return render_chat_template(conversation, tokenizer, **row)

                    train_ds, test_ds = self._load_dataset()
                    resolved.update(
                        train_ds=train_ds,
                        test_ds=test_ds,
                        prompt_builder=_prompt_builder,
                        reward_fn=get_reward_fn(
                            reward_fn_name=self.reward_fn_name,
                            file_path=self.reward_file_path,
                        ),
                    )
                return resolved

            def _dataset_factory() -> RolloutEnv:
                parts = _resolve()
                return RolloutEnv.from_dataset(
                    parts["train_ds"],
                    parts["reward_fn"],
                    tokenizer,
                    test_dataset=parts["test_ds"],
                    prompt_builder=parts["prompt_builder"],
                    pad_id=pad_id,
                    # ``prompt_builder`` already rendered the chat template.
                    apply_chat_template=False,
                    max_model_len=max_model_len,
                    max_output_tokens=max_output_tokens,
                )

            return _dataset_factory

        if self.env_url is not None:
            url = self.env_url
            # A remote env's turn budget can't be probed; validation requires
            # max_turns to be set explicitly for env_url.
            url_max_turns = self.max_turns or 1
            mcp_tool = self.mcp_tool
            timeout_s = self.request_timeout_s

            def _url_factory() -> RolloutEnv:
                return RolloutEnv(
                    url,
                    tokenizer,
                    max_turns=url_max_turns,
                    mcp_tool=mcp_tool,
                    timeout_s=timeout_s,
                    pad_id=pad_id,
                    apply_chat_template=True,
                    max_model_len=max_model_len,
                    max_output_tokens=max_output_tokens,
                )

            return _url_factory

        if self.env_name is not None:
            try:
                import gem
            except ImportError:
                msg = (
                    f"The 'gem-llm' package is required to use env_name={self.env_name!r}. "
                    "Install it with: pip install gem-llm"
                )
                raise ImportError(msg) from None

            env_name = self.env_name

            def _make_raw_env() -> Any:
                return gem.make(env_name)
        else:
            constructor = resolve_entrypoint_target(self.entrypoint)
            if not callable(constructor):
                msg = f"Entrypoint '{self.entrypoint}' resolved to non-callable object."
                raise TypeError(msg)
            cfg = self.env_config or {}

            def _make_raw_env() -> Any:
                return constructor(**cfg)

        max_turns = self.max_turns
        if max_turns is None:
            probe = _make_raw_env()
            max_turns = probe.max_turns
            if hasattr(probe, "close"):
                probe.close()
            self.max_turns = max_turns

        if self.served:
            # Host the same env on an in-process OpenEnv server and drive it
            # over HTTP loopback (one server per concurrent rollout) — the
            # in-process rehearsal of the ``env_url`` production transport.
            env_display_name = self.name
            mcp_tool = self.mcp_tool
            timeout_s = self.request_timeout_s

            def _served_factory() -> RolloutEnv:
                return RolloutEnv.serving(
                    _make_raw_env,
                    tokenizer,
                    max_turns=max_turns,
                    env_name=env_display_name,
                    mcp_tool=mcp_tool,
                    timeout_s=timeout_s,
                    pad_id=pad_id,
                    apply_chat_template=True,
                    max_model_len=max_model_len,
                    max_output_tokens=max_output_tokens,
                )

            return _served_factory

        def _env_factory() -> RolloutEnv:
            return RolloutEnv.local(
                _make_raw_env(),
                tokenizer,
                max_turns=max_turns,
                pad_id=pad_id,
                apply_chat_template=True,
                max_model_len=max_model_len,
                max_output_tokens=max_output_tokens,
            )

        return _env_factory


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
