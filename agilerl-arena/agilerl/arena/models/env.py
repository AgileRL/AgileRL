# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The manifest's ``environment`` section, discriminated on ``env_type``."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class LLMEnvType(str, Enum):
    """Type of LLM environment.

    ``ROLLOUT`` covers every generative regime: single-turn reasoning is simply
    ``max_turns=1``. ``DATASET`` covers the teacher-forced regimes, selected by
    :attr:`LLMEnvSpec.objective`.
    """

    ROLLOUT = "rollout"
    DATASET = "dataset"

    def __str__(self) -> str:
        return str(self.value)


class EnvSpecBase(BaseModel):
    """Fields every environment section carries."""

    model_config = ConfigDict(extra="forbid")

    num_envs: int = Field(
        default=1, ge=1, description="Environment copies stepped in parallel."
    )
    version: str | int | None = Field(
        default=None,
        description=(
            "Version of the environment implementation. For a custom "
            "environment this selects which uploaded archive the run uses; "
            "unset takes the latest."
        ),
    )


class GymEnvSpec(EnvSpecBase):
    """A gymnasium / PettingZoo environment, or a custom one from an entrypoint."""

    env_type: Literal["gym"] = Field(
        default="gym", description="Selects a gymnasium or PettingZoo environment."
    )
    name: str = Field(
        min_length=1,
        description="Registered environment id, e.g. LunarLander-v3.",
    )
    num_envs: int = Field(
        default=16,
        ge=1,
        description=(
            "Environment copies stepped in parallel. More environments collect "
            "experience faster at a proportional CPU cost."
        ),
    )
    custom: bool = Field(
        default=False,
        description="The environment is user-supplied rather than registered.",
    )
    default_type: str = Field(
        default="float32",
        description="NumPy dtype observations and buffers are stored as.",
    )
    entrypoint: str | None = Field(
        default=None,
        description="Dotted path to a callable returning the environment, for custom environments.",
    )
    path: str | None = Field(
        default=None,
        description="Filesystem path added to sys.path before the entrypoint is imported.",
    )
    env_config: dict[str, Any] | str | None = Field(
        default=None,
        validation_alias=AliasChoices("env_config", "config"),
        description="Keyword arguments forwarded to the environment constructor.",
    )
    env_wrappers: list[str | tuple[str, dict[str, Any]]] | None = Field(
        default=None,
        validation_alias=AliasChoices("env_wrappers", "wrappers"),
        description="Wrappers applied to each environment, outermost last.",
    )
    sync: bool = Field(
        default=False,
        description=(
            "Vectorize in one process rather than forking workers. "
            "Slower, but avoids gRPC and fork issues."
        ),
    )


class OfflineEnvSpec(GymEnvSpec):
    """A gym environment whose transitions come from a fixed dataset."""

    env_type: Literal["offline"] = Field(
        default="offline",
        description="Selects an environment replayed from a fixed dataset.",
    )
    minari_dataset_id: str | None = Field(
        default=None, description="Minari dataset id to pre-fill the buffer from."
    )
    dataset_path: str | None = Field(
        default=None,
        description="Path to a local HDF5 dataset to pre-fill the buffer from.",
    )
    remote: bool = Field(
        default=False,
        description="Download the Minari dataset rather than reading it locally.",
    )


class BanditEnvSpec(EnvSpecBase):
    """A tabular dataset wrapped as a contextual-bandit environment."""

    env_type: Literal["bandit"] = Field(
        default="bandit", description="Selects a contextual-bandit environment."
    )
    name: str = Field(default="BanditEnv", description="Label for the dataset in logs.")
    features: str | None = Field(
        default=None, description="Path to the feature table, one row per context."
    )
    targets: str | None = Field(
        default=None, description="Path to the target table, aligned with the features."
    )
    entrypoint: str | None = Field(
        default=None,
        description="Dotted path to a callable returning a bandit environment, instead of a table.",
    )
    path: str | None = Field(
        default=None,
        description="Filesystem path added to sys.path before the entrypoint is imported.",
    )
    env_config: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("env_config", "config"),
        description="Keyword arguments forwarded to the entrypoint callable.",
    )


class LLMEnvSpec(EnvSpecBase):
    """The dataset or interactive environment an LLM algorithm fine-tunes against.

    A ``rollout`` env is generative and comes from exactly one source:

    * dataset rows plus a rubric — a single-turn env over labelled rows;
    * ``entrypoint`` — a callable returning a text env, run where training runs
      (or on env hosts under the hosting fields, which need the Ray runtime);
    * ``env_url`` — an OpenEnv service someone else hosts;
    * ``env_image`` — a prebuilt env image the Ray runtime runs as its own Pod.

    A ``dataset`` env is teacher-forced over the rows; ``objective`` picks the
    loss.
    """

    env_type: Literal["rollout", "dataset"] = Field(
        description=(
            "Which LLM regime drives the run: a generative environment the "
            "model acts in (rollout), or teacher-forced learning over dataset "
            "rows (dataset)."
        )
    )
    objective: Literal["preference", "sft"] | None = Field(
        default=None,
        description=(
            "Teacher-forced loss, required with env_type dataset: sft, or "
            "preference (DPO) over chosen/rejected pairs."
        ),
    )

    dataset: str | None = Field(
        default=None,
        validation_alias=AliasChoices("dataset", "name"),
        description="HuggingFace dataset id, or a Parquet path, for the dataset sources.",
    )
    dataset_path: str | None = Field(
        default=None, description="Object-store path to a user-supplied dataset."
    )
    hf_dataset_id: str | None = Field(
        default=None,
        description="HuggingFace dataset id, when set apart from `dataset`.",
    )
    columns: dict[str, str] | None = Field(
        default=None,
        description=(
            "Maps your dataset's column names onto the ones the run expects "
            "(question, answer). Unset assumes they already match."
        ),
    )
    response_column: str = Field(
        default="response", description="Column holding the target response. SFT only."
    )
    prompt_template: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Conversation template, keyed 'role_index' (system_0, user_1, ...) "
            "so ordering is explicit. Values may interpolate dataset columns. "
            "Required for a rollout over dataset rows."
        ),
    )
    rubric_file_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("rubric_file_path", "reward_file_path"),
        description=(
            "Path to the Python module holding the rubric (or reward function) "
            "that scores a rollout over dataset rows."
        ),
    )
    rubric_name: str = Field(
        default="reward_fn",
        min_length=1,
        validation_alias=AliasChoices("rubric_name", "reward_fn_name"),
        description="Name of the rubric (or reward callable) to import from that module.",
    )
    max_reward: float | None = Field(
        default=None,
        description="Reward at which a rollout counts as solved, used for the success metric.",
    )
    train_test_split: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Fraction of the dataset used for training; the rest is held out for evaluation.",
    )

    entrypoint: str | None = Field(
        default=None,
        description=(
            "module:attr path to any callable returning a text env — your own "
            "class, or an env library's factory (gem:make, with the env id "
            "passed through env_config)."
        ),
    )
    env_config: dict[str, Any] | None = Field(
        default=None,
        description="Keyword arguments forwarded to the environment constructor.",
    )
    env_packages: dict[str, Any] | None = Field(
        default=None,
        description=(
            "What the entrypoint needs installed, as {'uv': [...]} or "
            "{'pip': [...]}. Installed where the env runs the first time the "
            "entrypoint is not importable."
        ),
    )
    max_turns: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Turn budget per episode. Unset probes it from an entrypoint env; "
            "required with env_url, where it cannot be probed."
        ),
    )
    strict_chat_template_boundary: bool = Field(
        default=True,
        description=(
            "Fail when the chat template cannot render a clean multi-turn "
            "boundary, rather than falling back to ChatML markers."
        ),
    )

    observation_field: str | None = Field(
        default=None,
        description=(
            "The field an env's observation carries its text in. Unset reads "
            "the standard shapes, or the single text field the observation "
            "carries."
        ),
    )
    observation_processor: str | None = Field(
        default=None,
        description=(
            "module:fn entrypoint rendering an observation payload to prompt "
            "text, for envs that need more than a field lookup. Mutually "
            "exclusive with observation_field."
        ),
    )

    env_url: str | list[str] | None = Field(
        default=None,
        description=(
            "URL of an already-hosted OpenEnv service. A list is a set of "
            "interchangeable replicas dealt across rollout slots."
        ),
    )
    mcp_tool: str | None = Field(
        default=None,
        description="For an MCP-backed env, the tool the model's text is sent to.",
    )
    action_field: str = Field(
        default="message",
        description=(
            "The action field an env receives the model's text in — message by "
            "default, but code or action_str elsewhere; for an MCP tool it is "
            "the argument name."
        ),
    )
    request_timeout_s: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Per-message client timeout in seconds for an env reached over "
            "HTTP. Unset applies a 300 s bound; 0 disables the bound."
        ),
    )

    env_hosts: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of env hosts to serve the env on (Ray runtime only). Unset "
            "with no image or packages runs an entrypoint in-process in each "
            "rollout slot."
        ),
    )
    env_image: str | None = Field(
        default=None,
        description=(
            "Prebuilt env image run verbatim as its own Pod per host, serving "
            "on env_port (Ray runtime only)."
        ),
    )
    env_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="TCP port the env_image server listens on.",
    )
    cpus_per_env_host: float = Field(
        default=1.0, gt=0.0, description="CPUs reserved per env host."
    )
    env_host_memory_bytes: int | None = Field(
        default=None, gt=0, description="Memory reserved per env host."
    )
    env_host_resource: str | None = Field(
        default=None,
        description=(
            "Where env hosts land: a custom Ray resource, or (for env_image) "
            "the Pod nodeSelector — key=value, or a bare pool name."
        ),
    )

    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Defaults injected into apply_chat_template, e.g. {'enable_thinking': false}.",
    )
    chat_template_path: str | None = Field(
        default=None,
        description="Jinja file whose template replaces the tokenizer's own.",
    )

    @property
    def name(self) -> str:
        """Human-readable name: dataset, URL, entrypoint, or image."""
        named = self._named_dataset()
        if named is not None:
            return named
        if isinstance(self.env_url, str):
            return self.env_url
        if self.env_url:
            return str(self.env_url[0])
        return self.entrypoint or self.env_image or "rollout"

    @property
    def dataset_backed_rollout(self) -> bool:
        """Whether this rollout env serves labelled rows scored by a rubric."""
        return self.env_type == "rollout" and self._named_dataset() is not None

    def _named_dataset(self) -> str | None:
        for value in (self.dataset, self.dataset_path, self.hf_dataset_id):
            if value is not None:
                return str(value)
        return None

    @model_validator(mode="after")
    def _check_rollout(self) -> Self:
        if self.env_type != "rollout":
            return self
        dataset = self._named_dataset()
        sources = {
            "a dataset": dataset,
            "entrypoint": self.entrypoint,
            "env_url": self.env_url,
            "env_image": self.env_image,
        }
        named = [name for name, value in sources.items() if value is not None]
        if len(named) != 1:
            got = ", ".join(named) or "none"
            msg = (
                "A rollout environment needs exactly one source: dataset rows "
                "plus a rubric, `entrypoint` (an env we build), `env_url` (an "
                "OpenEnv service someone else hosts), or `env_image` (a "
                f"prebuilt image we run). Got {got}."
            )
            raise ValueError(msg)
        if self.env_config is not None and dataset is not None:
            msg = (
                "env_config configures an environment; a rollout over dataset "
                "rows has no env to configure."
            )
            raise ValueError(msg)
        if self.env_packages is not None:
            if self.entrypoint is None:
                msg = (
                    "env_packages installs the dependencies of an entrypoint "
                    "env; a dataset, env_url or env_image source has nothing "
                    "to install for."
                )
                raise ValueError(msg)
            _check_env_packages(self.env_packages)
        if (
            self.env_hosts is not None
            and self.entrypoint is None
            and (self.env_image is None)
        ):
            msg = (
                "env_hosts serves an env we build (`entrypoint` or "
                "`env_image`); it does not apply to this source."
            )
            raise ValueError(msg)
        if (
            self.mcp_tool is not None or self.request_timeout_s is not None
        ) and dataset is not None:
            msg = (
                "mcp_tool / request_timeout_s configure an env reached over "
                "HTTP; a rollout over dataset rows has no such env."
            )
            raise ValueError(msg)
        if self.action_field != "message" and dataset is not None:
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
        ) and dataset is not None:
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
        if dataset is not None:
            if self.rubric_file_path is None:
                msg = "rubric_file_path is required for dataset-backed rollout environments"
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
    def _check_dataset(self) -> Self:
        if self.env_type != "dataset":
            return self
        if self._named_dataset() is None:
            msg = (
                "A dataset environment must name its dataset via 'dataset', "
                "'dataset_path' or 'hf_dataset_id'."
            )
            raise ValueError(msg)
        if self.objective is None:
            msg = "objective is required for dataset environments"
            raise ValueError(msg)
        named_envs = [
            f"{field}={value!r}"
            for field, value in (
                ("entrypoint", self.entrypoint),
                ("env_url", self.env_url),
                ("env_image", self.env_image),
            )
            if value is not None
        ]
        if named_envs:
            msg = (
                "A dataset environment is teacher-forced over its rows; drop "
                f"{', '.join(named_envs)} — keeping both would silently ignore "
                "the env."
            )
            raise ValueError(msg)
        if self.rubric_file_path is not None:
            msg = "rubric_file_path has been specified, but is not supported for dataset environments."
            raise ValueError(msg)
        return self


def _check_env_packages(env_packages: dict[str, Any]) -> None:
    """Check ``env_packages`` names at least one package under ``uv`` or ``pip``."""
    packages: list[Any] = []
    for field in ("uv", "pip"):
        value = env_packages.get(field)
        if value is None:
            continue
        if isinstance(value, dict):
            packages.extend(value.get("packages") or [])
        elif isinstance(value, list):
            packages.extend(value)
        else:
            msg = (
                f"env_packages.{field} must be a list of packages or "
                f"{{'packages': [...]}}, got {type(value).__name__}."
            )
            raise ValueError(msg)
    if not packages:
        msg = (
            f"env_packages named no packages to install: {env_packages!r}. "
            "Expected {'uv': [...]} or {'pip': [...]}."
        )
        raise ValueError(msg)


EnvSpec = Annotated[
    GymEnvSpec | OfflineEnvSpec | BanditEnvSpec | LLMEnvSpec,
    Field(discriminator="env_type"),
]
