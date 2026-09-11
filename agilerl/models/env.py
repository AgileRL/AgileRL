# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Environment specs (the arena contract) and the functions that build them."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import pandas as pd
from pettingzoo import ParallelEnv

from agilerl.arena.models.env import (
    BanditEnvSpec,
    EnvSpec,
    GymEnvSpec,
    LLMEnvSpec,
    LLMEnvType,
    OfflineEnvSpec,
)
from agilerl.llm_envs import DatasetEnv, RolloutHarness
from agilerl.llm_envs.env_packages import ensure_importable
from agilerl.protocols import BanditEnvProtocol
from agilerl.typing import EnvFactory, WrapperSpec
from agilerl.utils.env_utils import (
    GymEnvType,
    apply_wrappers,
    get_rubric_factory,
    make_conversation_template,
    resolve_entrypoint_target,
)
from agilerl.utils.llm_utils import render_chat_template, validate_llm_context_lengths
from agilerl.utils.utils import make_multi_agent_vect_envs, make_vect_envs
from agilerl.vector import AsyncPettingZooVecEnv
from agilerl.wrappers.learning import BanditEnv

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from agilerl.protocols import TextEnvProtocol

__all__ = [
    "BanditEnvSpec",
    "EnvSpec",
    "GymEnvSpec",
    "LLMEnvSpec",
    "LLMEnvType",
    "OfflineEnvSpec",
    "construct_custom_env_fn",
    "construct_custom_pz_env_fn",
    "make_bandit_env",
    "make_env",
    "make_gym_env",
    "make_llm_env",
    "make_pz_env",
    "make_rollout_env_factory",
    "make_single_env",
]


def _env_kwargs(spec: GymEnvSpec | BanditEnvSpec) -> dict[str, Any]:
    config = spec.env_config
    return config if isinstance(config, dict) else {}


def _wrappers(
    spec: GymEnvSpec,
) -> list[str | tuple[str, dict[str, Any]]] | None:
    return spec.env_wrappers


def construct_custom_env_fn(
    entrypoint: str,
    path: str | None = None,
    config: dict[str, Any] | None = None,
    wrappers: Sequence[WrapperSpec] | None = None,
) -> EnvFactory:
    """Build a factory for a custom gym environment from an entrypoint."""

    def default_make_env() -> gym.Env:
        constructor = resolve_entrypoint_target(entrypoint, path=path)
        if not callable(constructor):
            msg = f"Entrypoint '{entrypoint}' resolved to non-callable object."
            raise TypeError(msg)
        env = constructor(**(config or {}))
        return apply_wrappers(env, wrappers, path=path)

    return default_make_env


def construct_custom_pz_env_fn(
    entrypoint: str,
    path: str | None = None,
    config: dict[str, Any] | None = None,
    wrappers: Sequence[WrapperSpec] | None = None,
) -> Callable[[], ParallelEnv]:
    """Build a factory for a custom PettingZoo environment from an entrypoint."""

    def default_make_env() -> ParallelEnv:
        constructor = resolve_entrypoint_target(entrypoint, path=path)
        if not callable(constructor):
            msg = f"Entrypoint '{entrypoint}' resolved to non-callable object."
            raise TypeError(msg)
        env = constructor(**(config or {}))
        return apply_wrappers(env, wrappers, path=path)

    return default_make_env


def make_single_env(
    spec: GymEnvSpec, *, multi_agent: bool = False
) -> gym.Env | ParallelEnv:
    """Create a single (non-vectorized) environment instance."""
    if multi_agent:
        return _make_single_pz_env(spec)
    if spec.entrypoint is not None:
        return construct_custom_env_fn(
            spec.entrypoint,
            spec.path,
            _env_kwargs(spec),
            _wrappers(spec),
        )()
    return gym.make(spec.name)


def _make_single_pz_env(spec: GymEnvSpec) -> ParallelEnv:
    if spec.entrypoint is not None:
        return construct_custom_pz_env_fn(
            spec.entrypoint,
            spec.path,
            _env_kwargs(spec),
            _wrappers(spec),
        )()
    module = import_module(spec.name)
    if not hasattr(module, "parallel_env"):
        msg = f"PettingZoo module '{spec.name}' has no 'parallel_env' constructor."
        raise AttributeError(msg)
    env = module.parallel_env(**_env_kwargs(spec))
    return apply_wrappers(env, _wrappers(spec), path=spec.path)


def make_gym_env(
    spec: GymEnvSpec,
    extra_wrappers: list[type] | None = None,
    wrappers: Sequence[WrapperSpec] | None = None,
) -> GymEnvType:
    """Instantiate the vectorized gym environment."""
    resolved = wrappers if wrappers is not None else _wrappers(spec)
    if spec.entrypoint is not None:
        make_one = construct_custom_env_fn(
            spec.entrypoint,
            spec.path,
            _env_kwargs(spec),
            resolved,
        )
    else:
        make_one = None

    return make_vect_envs(
        env_name=spec.name,
        num_envs=spec.num_envs,
        make_env=make_one,
        should_async_vector=(not spec.sync),
        extra_wrappers=extra_wrappers,
    )


def make_pz_env(
    spec: GymEnvSpec,
    extra_wrappers: list[type] | None = None,
    wrappers: Sequence[WrapperSpec] | None = None,
) -> AsyncPettingZooVecEnv:
    """Instantiate vectorized PettingZoo environments."""
    resolved = wrappers if wrappers is not None else _wrappers(spec)
    if spec.entrypoint is not None:
        make_one = construct_custom_pz_env_fn(
            spec.entrypoint,
            spec.path,
            _env_kwargs(spec),
            resolved,
        )
    else:

        def make_one() -> ParallelEnv:
            module = import_module(spec.name)
            if not hasattr(module, "parallel_env"):
                msg = f"PettingZoo module '{spec.name}' has no 'parallel_env' constructor."
                raise AttributeError(msg)
            env = module.parallel_env(**_env_kwargs(spec))
            return apply_wrappers(env, resolved, path=spec.path)

    return make_multi_agent_vect_envs(
        env=make_one,
        num_envs=spec.num_envs,
        extra_wrappers=extra_wrappers,
    )


def _load_dataframe(value: str | Path) -> pd.DataFrame:
    path = Path(value) if isinstance(value, str) else value
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".h5", ".hdf5"}:
        return pd.read_hdf(path)
    msg = f"Unsupported file type: {path.suffix}"
    raise ValueError(msg)


def make_bandit_env(
    spec: BanditEnvSpec,
    *,
    features: pd.DataFrame | str | Path | None = None,
    targets: pd.DataFrame | str | Path | None = None,
) -> BanditEnvProtocol:
    """Construct a bandit environment from a spec, or from in-memory tables."""
    resolved_features = spec.features if features is None else features
    resolved_targets = spec.targets if targets is None else targets
    has_features = resolved_features is not None
    has_targets = resolved_targets is not None
    has_entrypoint = spec.entrypoint is not None

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

    if spec.entrypoint is not None:
        constructor = resolve_entrypoint_target(spec.entrypoint, path=spec.path)
        if not callable(constructor):
            msg = f"Entrypoint '{spec.entrypoint}' resolved to non-callable object."
            raise TypeError(msg)
        return constructor(**_env_kwargs(spec))

    loaded_features = (
        _load_dataframe(resolved_features)
        if isinstance(resolved_features, str | Path)
        else resolved_features
    )
    loaded_targets = (
        _load_dataframe(resolved_targets)
        if isinstance(resolved_targets, str | Path)
        else resolved_targets
    )
    if loaded_features is None or loaded_targets is None:
        msg = (
            "Both 'features' and 'targets' are required for dataset-mode "
            "bandit environments."
        )
        raise ValueError(msg)
    return BanditEnv(features=loaded_features, targets=loaded_targets)


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


def _split_seed(seed: int | None) -> int:
    """Seed for the shuffle and train/test split, defaulting when unset.

    Every rank loads the dataset for itself and the rollout task assigner hands
    out rows *by index*, so an unseeded split would leave ranks disagreeing
    about which row is row ``n`` — their shards would overlap instead of
    partitioning, and each would hold out a different test set.
    """
    return 42 if seed is None else seed


def _load_llm_dataset(
    spec: LLMEnvSpec, *, seed: int | None = None
) -> tuple[Dataset, Dataset]:
    """Load the spec's dataset and split it into train and test."""
    dataset = spec.dataset
    if dataset is None:
        msg = "dataset is required to load rollout/preference/sft data"
        raise ValueError(msg)
    if dataset.endswith((".parquet", ".pq")):
        return _load_dataset_file(spec, dataset, seed=seed)
    return _load_dataset_hf(spec, dataset, seed=seed)


def _load_dataset_hf(
    spec: LLMEnvSpec, dataset: str, *, seed: int | None = None
) -> tuple[Dataset, Dataset]:
    _, load_dataset = _require_datasets()
    ds = load_dataset(dataset, split="train").shuffle(seed=_split_seed(seed))
    if spec.columns:
        ds = ds.rename_columns(spec.columns)
    split = ds.train_test_split(
        test_size=1.0 - spec.train_test_split, seed=_split_seed(seed)
    )
    return split["train"], split["test"]


def _load_dataset_file(
    spec: LLMEnvSpec, dataset: str, *, seed: int | None = None
) -> tuple[Dataset, Dataset]:
    Dataset, _ = _require_datasets()
    df = pd.read_parquet(dataset)
    if spec.columns:
        df = df.rename(columns=spec.columns)
    ds = Dataset.from_pandas(df)
    split = ds.train_test_split(
        test_size=1.0 - spec.train_test_split, seed=_split_seed(seed)
    )
    return split["train"], split["test"]


def make_llm_env(
    spec: LLMEnvSpec,
    tokenizer: PreTrainedTokenizerBase,
    *,
    data_batch_size_per_gpu: int = 8,
    max_context_length: int | None = None,
    seed: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> DatasetEnv:
    """Build the teacher-forced dataset environment for an LLM spec.

    Rollout envs are built per-trajectory instead: use
    :func:`make_rollout_env_factory`.

    :param spec: The env spec, with ``env_type="dataset"``.
    :type spec: LLMEnvSpec
    :param tokenizer: The tokenizer.
    :type tokenizer: PreTrainedTokenizerBase
    :param data_batch_size_per_gpu: Rows each rank draws per step.
    :type data_batch_size_per_gpu: int
    :param max_context_length: Token budget a row is truncated to.
    :type max_context_length: int | None
    :param seed: Seed for the dataset split and the env's row order.
    :type seed: int | None
    :param rank: This process's data-parallel shard index.
    :type rank: int
    :param world_size: Number of data-parallel shards.
    :type world_size: int
    :return: The dataset environment.
    :rtype: DatasetEnv
    """
    if spec.env_type != LLMEnvType.DATASET:
        msg = (
            "Rollout environments cannot be constructed with make_llm_env(). "
            "Use make_rollout_env_factory() instead."
        )
        raise TypeError(msg)
    if spec.objective is None:
        msg = "objective is required for dataset environments."
        raise ValueError(msg)

    train_ds, test_ds = _load_llm_dataset(spec, seed=seed)
    return DatasetEnv(
        train_dataset=train_ds,
        test_dataset=test_ds,
        tokenizer=tokenizer,
        objective=spec.objective,
        response_column=spec.response_column,
        chat_template_kwargs=spec.chat_template_kwargs,
        data_batch_size_per_gpu=data_batch_size_per_gpu,
        max_context_length=max_context_length,
        seed=_split_seed(seed),
        rank=rank,
        world_size=world_size,
    )


def make_rollout_env_factory(
    spec: LLMEnvSpec,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_model_len: int | None = None,
    max_output_tokens: int | None = None,
    seed: int | None = None,
) -> tuple[Callable[[], RolloutHarness], int]:
    """Build a factory that creates fresh :class:`RolloutHarness` instances.

    Each call to the returned factory creates an independent env, so concurrent
    trajectories never share state. Which of the three builders below runs is
    decided by the one source the spec names — dataset rows, an ``env_url``, or
    an ``entrypoint``. ``env_image`` needs an orchestrator that can place Pods,
    which this process is not.

    The resolved turn budget rides along because only the builder can know it:
    an entrypoint env that does not declare ``max_turns`` in the manifest is
    probed for one.

    :param spec: The env spec, with ``env_type="rollout"``.
    :type spec: LLMEnvSpec
    :param tokenizer: The tokenizer (shared across all instances).
    :type tokenizer: PreTrainedTokenizerBase
    :param max_model_len: Maximum model context length for prompt truncation.
    :type max_model_len: int | None
    :param max_output_tokens: Maximum newly generated tokens per turn.
    :type max_output_tokens: int | None
    :param seed: Seed for a dataset-backed rollout's train/test split.
    :type seed: int | None
    :returns: A zero-argument callable that creates a ``RolloutHarness``, and
        the rollout's turn budget.
    :rtype: tuple[Callable[[], RolloutHarness], int]
    """
    if spec.env_type != LLMEnvType.ROLLOUT:
        msg = (
            "Dataset environments cannot be constructed with "
            "make_rollout_env_factory(). Use make_llm_env() instead."
        )
        raise TypeError(msg)
    if spec.env_image is not None:
        msg = (
            "env_image runs the env as its own Pod, which needs the Ray "
            "runtime; this process can only build dataset, entrypoint, or "
            "env_url rollouts."
        )
        raise ValueError(msg)

    # max_output_tokens >= max_model_len zeroes the prompt budget: every episode
    # truncates at reset and training silently never advances. Fail at setup.
    if max_model_len is not None:
        validate_llm_context_lengths(max_model_len, max_output_tokens)

    harness_kwargs: dict[str, Any] = {
        "max_model_len": max_model_len,
        "max_output_tokens": max_output_tokens,
        "chat_template_kwargs": dict(spec.chat_template_kwargs),
    }
    if spec.dataset_backed_rollout:
        return _make_dataset_rollout_factory(spec, tokenizer, harness_kwargs, seed=seed)
    if spec.env_url is not None:
        return _make_url_rollout_factory(spec, tokenizer, harness_kwargs)
    return _make_entrypoint_rollout_factory(spec, tokenizer, harness_kwargs)


def _http_timeout_s(spec: LLMEnvSpec) -> float | None:
    """Per-message ``env_url`` timeout: manifest value, 300 s default, ``0`` unbounds."""
    if spec.request_timeout_s is None:
        return 300.0
    return spec.request_timeout_s or None


def _make_dataset_rollout_factory(
    spec: LLMEnvSpec,
    tokenizer: PreTrainedTokenizerBase,
    harness_kwargs: dict[str, Any],
    *,
    seed: int | None,
) -> tuple[Callable[[], RolloutHarness], int]:
    """Factory serving labelled dataset rows scored by a rubric, in-process."""
    chat_template_kwargs = harness_kwargs["chat_template_kwargs"]
    # The rubric file and the dataset are read once, on the first env
    # build, so constructing the factory stays free of I/O.
    resolved: dict[str, Any] = {}

    def _resolve() -> dict[str, Any]:
        if not resolved:
            if (
                spec.prompt_template is None
                or spec.rubric_name is None
                or spec.rubric_file_path is None
            ):
                msg = (
                    "rubric_name, rubric_file_path, and prompt_template "
                    "are required for dataset-backed rollout environments"
                )
                raise ValueError(msg)

            conversation = make_conversation_template(
                prompt_template=spec.prompt_template
            )

            def _prompt_builder(row: Mapping[str, Any]) -> str:
                """Render the manifest's chat template for one dataset row."""
                return render_chat_template(
                    conversation,
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,
                    **row,
                )

            train_ds, test_ds = _load_llm_dataset(spec, seed=seed)
            resolved.update(
                train_ds=train_ds,
                test_ds=test_ds,
                prompt_builder=_prompt_builder,
                rubric_factory=get_rubric_factory(
                    rubric_name=spec.rubric_name,
                    file_path=spec.rubric_file_path,
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
            max_output_tokens=harness_kwargs["max_output_tokens"],
        )

    # A dataset-backed rollout is single-turn; the contract pins max_turns to 1.
    return _dataset_factory, int(spec.max_turns or 1)


def _make_url_rollout_factory(
    spec: LLMEnvSpec,
    tokenizer: PreTrainedTokenizerBase,
    harness_kwargs: dict[str, Any],
) -> tuple[Callable[[], RolloutHarness], int]:
    """Factory dialling an env that is already running at ``env_url``.

    A list of URLs is a set of interchangeable replicas; successive factory
    calls deal them round-robin, one per rollout slot.
    """
    from agilerl.llm_envs.openenv import RemoteEnvClient  # optional extra: llm

    if spec.env_url is None:
        msg = "env_url is required to dial a remote rollout env"
        raise RuntimeError(msg)
    urls = [spec.env_url] if isinstance(spec.env_url, str) else list(spec.env_url)
    if not urls:
        msg = "`env_url` is an empty list; give at least one hosted env URL."
        raise ValueError(msg)
    next_url = count()
    # A remote env's turn budget can't be probed; validation requires max_turns.
    url_max_turns = spec.max_turns or 1
    mcp_tool = spec.mcp_tool
    action_field = spec.action_field
    timeout_s = _http_timeout_s(spec)
    observation_field = spec.observation_field
    observation_processor = _resolved_observation_processor(spec)
    strict_boundary = spec.strict_chat_template_boundary

    def _url_factory() -> RolloutHarness:
        # The server's max_concurrent_envs must cover batch*group + the eval env.
        return RolloutHarness(
            RemoteEnvClient(
                urls[next(next_url) % len(urls)],
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

    return _url_factory, url_max_turns


def _make_entrypoint_rollout_factory(
    spec: LLMEnvSpec,
    tokenizer: PreTrainedTokenizerBase,
    harness_kwargs: dict[str, Any],
) -> tuple[Callable[[], RolloutHarness], int]:
    """Factory importing ``entrypoint`` and driving the env it builds in-process.

    An unset ``max_turns`` is probed off one throwaway env.
    """
    if spec.entrypoint is None:
        msg = "An entrypoint is required for env-backed rollout environments."
        raise ValueError(msg)
    if spec.env_packages is not None:
        ensure_importable(spec.entrypoint, spec.env_packages)
    constructor = resolve_entrypoint_target(spec.entrypoint)
    if not callable(constructor):
        msg = f"Entrypoint '{spec.entrypoint}' resolved to non-callable object."
        raise TypeError(msg)
    cfg = dict(spec.env_config or {})
    # Constructors like ``gem.make`` reject a system_prompt kwarg; it is set
    # on the built env as an attribute instead.
    system_prompt = cfg.pop("system_prompt", None)

    def _make_raw_env() -> TextEnvProtocol:
        env = constructor(**cfg)
        if system_prompt is not None:
            env.system_prompt = system_prompt
        return env

    max_turns = spec.max_turns
    if max_turns is None:
        probe = _make_raw_env()
        max_turns = int(getattr(probe, "max_turns", 1) or 1)
        closer = getattr(probe, "close", None)
        if callable(closer):
            closer()

    action_field = spec.action_field
    observation_field = spec.observation_field
    observation_processor = _resolved_observation_processor(spec)
    strict_boundary = spec.strict_chat_template_boundary

    def _env_factory() -> RolloutHarness:
        return RolloutHarness.local(
            _make_raw_env(),
            tokenizer,
            max_turns=max_turns,
            action_field=action_field,
            observation_field=observation_field,
            observation_processor=observation_processor,
            apply_chat_template=True,
            strict_chat_template_boundary=strict_boundary,
            **harness_kwargs,
        )

    return _env_factory, max_turns


def _resolved_observation_processor(spec: LLMEnvSpec) -> Callable[[Any], str] | None:
    """Import ``spec.observation_processor``, or ``None`` when the spec names none.

    :raises TypeError: If the dotted path resolves to something not callable.
    """
    if spec.observation_processor is None:
        return None
    resolved = resolve_entrypoint_target(spec.observation_processor)
    if not callable(resolved):
        msg = (
            f"observation_processor '{spec.observation_processor}' "
            "resolved to a non-callable object."
        )
        raise TypeError(msg)
    return resolved


def make_env(
    spec: EnvSpec,
    *,
    multi_agent: bool = False,
    extra_wrappers: list[type] | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    features: pd.DataFrame | str | Path | None = None,
    targets: pd.DataFrame | str | Path | None = None,
    seed: int | None = None,
    data_batch_size_per_gpu: int = 8,
    max_context_length: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> GymEnvType | AsyncPettingZooVecEnv | BanditEnvProtocol | DatasetEnv:
    """Build the live environment described by *spec*."""
    if isinstance(spec, LLMEnvSpec):
        if tokenizer is None:
            msg = "LLM environment construction requires a tokenizer."
            raise TypeError(msg)
        return make_llm_env(
            spec,
            tokenizer,
            data_batch_size_per_gpu=data_batch_size_per_gpu,
            max_context_length=max_context_length,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
    if isinstance(spec, BanditEnvSpec):
        return make_bandit_env(spec, features=features, targets=targets)
    if not isinstance(spec, GymEnvSpec):
        msg = f"Cannot build a gym environment from {type(spec).__name__}"
        raise TypeError(msg)
    if multi_agent:
        return make_pz_env(spec, extra_wrappers=extra_wrappers)
    return make_gym_env(spec, extra_wrappers=extra_wrappers)
