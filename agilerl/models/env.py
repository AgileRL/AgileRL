# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The manifest's ``environment`` section, discriminated on ``env_type``."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from agilerl.models.judge import ENV_VAR_NAME_PATTERN, JudgeSpec
from agilerl.models.prompt_template import check_row_placeholders
from agilerl.models.rewards import RewardSpec, check_reward_fields
from agilerl.models.traces import TRACES_DATASET_FORMAT
from agilerl.models.visibility import ADVANCED, ADVANCED_PLATFORM_WRITTEN


def reject_legacy_llm_env_spelling(
    *,
    entrypoint: str | None,
    factory: str | None,
    env_config: dict[str, Any] | None,
) -> None:
    """Reject ``env_config.env_id`` and factory without an entrypoint.

    :param entrypoint: The env to build.
    :param factory: Optional ``module:attr`` callable that receives the entrypoint.
    :param env_config: Constructor kwargs; must not contain ``env_id``.
    """
    if env_config is not None and "env_id" in env_config:
        msg = (
            "env_config cannot contain env_id. Put the env to build in "
            "entrypoint and the callable that builds it in factory."
        )
        raise ValueError(msg)
    if factory is not None and not entrypoint:
        msg = (
            "factory is set but entrypoint is missing. factory is the "
            "callable; entrypoint is the env it builds."
        )
        raise ValueError(msg)


def check_factoryless_entrypoint(entrypoint: str) -> None:
    """Reject an entrypoint the trainer cannot import when no factory is set.

    Without a factory the trainer imports ``module:attr`` and calls ``attr``,
    so the part after the last ``:`` must be a Python identifier. A registry
    id such as ``game:GuessTheNumber-v0-easy`` fails that and needs a factory.
    A registry id whose name is an identifier passes, since telling it from an
    import path would need the import.

    :param entrypoint: The env to build.
    """
    if entrypoint.startswith(("http://", "https://")):
        msg = f"entrypoint {entrypoint!r} is a URL; use env_url instead."
        raise ValueError(msg)
    module_ref, _, attr = entrypoint.rpartition(":")
    if not module_ref or not attr.isidentifier():
        msg = (
            f"environment.entrypoint {entrypoint!r} is a registry id, not a "
            "module:attr import path; set environment.factory: gem:make (or "
            "another factory) to build it."
        )
        raise ValueError(msg)


DNS_LABEL_PATTERN = r"^[a-z]([-a-z0-9]*[a-z0-9])?$"
# Env Pods are I/O-bound; this fits beside Ray workers that request a node's CPUs.
ENV_POD_DEFAULT_CPUS = 0.01
# A multi-GB env image cold-pulls in minutes.
ENV_POD_READY_TIMEOUT_S = 600.0
DEFAULT_ADAPTIVE_TASK_SAMPLING = True
# Arena fills a named built-in's source when the run starts.
BUNDLED_ENVIRONMENTS = frozenset({"AgentWorldModel", "VisualWebArena"})


class EnvContainerSpec(BaseModel):
    """One container of an ``env_services`` Pod."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        pattern=DNS_LABEL_PATTERN,
        max_length=40,
        description="Container name, unique within its service.",
    )
    image: str = Field(description="Container image.")
    command: list[str] | None = Field(
        default=None,
        description="Replaces the image ENTRYPOINT. Required with memory_rootfs.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Container environment, in order. A value may reference an earlier "
            "variable or any service's url_env as $(NAME)."
        ),
    )
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description=(
            "Port this container serves the service on. Exactly one container "
            "of a service sets it."
        ),
    )
    setup_command: list[str] | None = Field(
        default=None,
        description=(
            "Runs once the container starts; the service is not ready until it "
            "exits 0, and a failure restarts the container. Output goes to the "
            "container log."
        ),
    )
    readiness_command: list[str] | None = Field(
        default=None,
        description="Exec readiness probe. Unset probes TCP on port, if set.",
    )
    readiness_path: str | None = Field(
        default=None,
        description="HTTP GET readiness probe on port.",
    )
    cpu: str = Field(
        default="10m",
        description="CPU request, as a Kubernetes quantity. No CPU limit is set.",
    )
    memory: str | None = Field(default=None, description="Memory request.")
    memory_limit: str | None = Field(default=None, description="Memory limit.")
    ephemeral_storage: str | None = Field(
        default=None,
        description=(
            "Ephemeral-storage request, so the Pod lands on a node with room "
            "for a large image."
        ),
    )
    memory_rootfs: str | None = Field(
        default=None,
        description=(
            "Size of a memory-backed volume the image is unpacked into, for an "
            "image larger than a node's disk. The container then runs command "
            "under chroot in a privileged container: the image's ENTRYPOINT, "
            "ENV and WORKDIR do not apply, so set command and env. The unpack "
            "uses the Pod's image pull secrets."
        ),
    )

    @model_validator(mode="after")
    def _check_container(self) -> Self:
        if self.readiness_command is not None and self.readiness_path is not None:
            msg = (
                f"Container {self.name!r} sets readiness_command and "
                "readiness_path; a container has one readiness probe."
            )
            raise ValueError(msg)
        if self.readiness_path is not None and self.port is None:
            msg = f"Container {self.name!r}: readiness_path probes port; set port."
            raise ValueError(msg)
        if self.memory_rootfs is not None and self.command is None:
            msg = (
                f"Container {self.name!r}: memory_rootfs runs the unpacked image "
                "without its ENTRYPOINT; set command."
            )
            raise ValueError(msg)
        return self


class EnvHostVolume(BaseModel):
    """A node directory mounted read-only into every container of a service."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        pattern=DNS_LABEL_PATTERN,
        max_length=40,
        description="Volume name, unique within its service.",
    )
    host_path: str = Field(
        description=(
            "Directory on the node under /var/cache/agilerl-, where the platform "
            "mounts the data it provides. The Pod does not start on a node "
            "without it."
        ),
    )
    mount_path: str = Field(description="Where the containers see the directory.")

    @model_validator(mode="after")
    def _check_paths(self) -> Self:
        paths = {"host_path": self.host_path, "mount_path": self.mount_path}
        for field, path in paths.items():
            normal = PurePosixPath(path)
            if (
                not normal.is_absolute()
                or path == "/"
                or str(normal) != path
                or ".." in normal.parts
            ):
                msg = (
                    f"Host volume {self.name!r}: {field} must be an absolute path "
                    f"without '..', a trailing slash or '/' itself; got {path!r}."
                )
                raise ValueError(msg)
        node_data_root = "/var/cache/agilerl-"
        if not self.host_path.startswith(node_data_root):
            msg = (
                f"Host volume {self.name!r}: host_path must be under "
                f"{node_data_root}*; got {self.host_path!r}."
            )
            raise ValueError(msg)
        # Every env container already mounts a memory volume at /dev/shm.
        shm = PurePosixPath("/dev/shm")
        mount = PurePosixPath(self.mount_path)
        if mount == shm or shm in mount.parents:
            msg = (
                f"Host volume {self.name!r}: mount_path {self.mount_path!r} is in "
                "/dev/shm, the containers' shared-memory mount; pick another path."
            )
            raise ValueError(msg)
        return self


class EnvServiceSpec(BaseModel):
    """A backing service the env sessions share: one Pod behind a stable DNS name."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        pattern=DNS_LABEL_PATTERN,
        max_length=30,
        description="Service name, unique within the environment.",
    )
    port: int = Field(
        ge=1,
        le=65535,
        description="Port clients dial; the service URL is http://<host>:<port>.",
    )
    url_env: str = Field(
        pattern=ENV_VAR_NAME_PATTERN,
        description=(
            "Environment variable carrying the service URL into every env "
            "session and service container."
        ),
    )
    containers: list[EnvContainerSpec] = Field(
        min_length=1,
        description="Containers of the service Pod; they share its network.",
    )
    ready_timeout_s: float = Field(
        default=600.0,
        gt=0.0,
        description="How long the service may take to become ready.",
    )
    host_volumes: list[EnvHostVolume] = Field(
        default_factory=list,
        description=(
            "Node directories mounted read-only into every container, for data "
            "the node provides (a disk the cluster attaches) rather than the image."
        ),
    )

    @model_validator(mode="after")
    def _check_service(self) -> Self:
        names = [container.name for container in self.containers]
        if len(set(names)) != len(names):
            msg = f"Service {self.name!r} repeats a container name: {names}."
            raise ValueError(msg)
        volume_names = [volume.name for volume in self.host_volumes]
        if len(set(volume_names)) != len(volume_names):
            msg = f"Service {self.name!r} repeats a host volume name: {volume_names}."
            raise ValueError(msg)
        mount_paths = [volume.mount_path for volume in self.host_volumes]
        for path in mount_paths:
            if mount_paths.count(path) > 1:
                msg = (
                    f"Service {self.name!r} mounts more than one host volume at {path}."
                )
                raise ValueError(msg)
        rootfs = [c.name for c in self.containers if c.memory_rootfs is not None]
        if self.host_volumes and rootfs:
            msg = (
                f"Service {self.name!r}: host_volumes are not visible under "
                f"memory_rootfs, which containers {rootfs} set."
            )
            raise ValueError(msg)
        serving = [c.name for c in self.containers if c.port is not None]
        if len(serving) != 1:
            msg = (
                f"Service {self.name!r} needs exactly one container with a port "
                f"to serve on; got {serving or 'none'}."
            )
            raise ValueError(msg)
        return self


def check_restart_keep_turns(
    restart_keep_turns: int,
    segment_prompt_tokens: int | None,
    segment_max_images: int | None,
    restart_older_obs_field: str,
    restart_older_images: bool,
) -> None:
    """Reject kept turns without a restart limit, or with no room for the next turn's image.

    Also reject settings for older kept turns when fewer than two turns are kept.

    :param restart_keep_turns: Turns a restarted context repeats word for word.
    :param segment_prompt_tokens: Token count that restarts the context.
    :param segment_max_images: Image count that restarts the context.
    :param restart_older_obs_field: Field shown for kept turns before the latest.
    :param restart_older_images: Whether kept turns before the latest keep images.
    """
    if (restart_older_obs_field or not restart_older_images) and restart_keep_turns < 2:
        msg = (
            "restart_older_obs_field and restart_older_images apply to kept "
            "turns before the latest; set restart_keep_turns to 2 or more."
        )
        raise ValueError(msg)
    if not restart_keep_turns:
        return
    if segment_prompt_tokens is None and segment_max_images is None:
        msg = (
            "restart_keep_turns applies at a context restart; set "
            "segment_prompt_tokens or segment_max_images."
        )
        raise ValueError(msg)
    if segment_max_images is not None and restart_keep_turns >= segment_max_images:
        msg = (
            f"restart_keep_turns ({restart_keep_turns}) must be below "
            f"segment_max_images ({segment_max_images}): a restarted prompt "
            "keeps at most segment_max_images - 1 images so the next turn's "
            "image fits."
        )
        raise ValueError(msg)


def check_env_pod_fields(
    *,
    env_image: str | None,
    env_vars: dict[str, str],
    env_sessions_per_host: int | None,
    env_session_ports: dict[str, int],
    env_host_memory_limit_bytes: int | None,
    env_host_ready_timeout_s: float | None,
    env_services: list[EnvServiceSpec],
) -> None:
    """Reject ``env_image`` Pod settings without an image, unbindable sessions, and colliding env names.

    :param env_image: The env image, or ``None``.
    :param env_vars: Container environment of each session.
    :param env_sessions_per_host: Sessions per Pod.
    :param env_session_ports: Per-session port variables.
    :param env_host_memory_limit_bytes: Memory limit per Pod.
    :param env_host_ready_timeout_s: Readiness timeout of each session Pod.
    :param env_services: Backing services.
    """
    if env_image is None:
        if env_vars:
            msg = "env_vars set container environment on an env_image Pod."
            raise ValueError(msg)
        if (
            env_sessions_per_host is not None
            or env_session_ports
            or env_host_memory_limit_bytes is not None
            or env_host_ready_timeout_s is not None
            or env_services
        ):
            msg = (
                "env_sessions_per_host, env_session_ports, "
                "env_host_memory_limit_bytes, env_host_ready_timeout_s and "
                "env_services configure env_image Pods; set env_image."
            )
            raise ValueError(msg)
        return
    # Sessions share the Pod's network; only env_session_ports tells each its port.
    if (env_sessions_per_host or 1) > 1 and not env_session_ports:
        msg = (
            f"env_sessions_per_host={env_sessions_per_host} needs "
            "env_session_ports: sessions share one Pod network, so each "
            "container must be told its own port (e.g. {PORT: 8000} gives "
            "session s PORT=8000+s)."
        )
        raise ValueError(msg)
    service_names = [service.name for service in env_services]
    if len(set(service_names)) != len(service_names):
        msg = f"env_services repeats a service name: {service_names}."
        raise ValueError(msg)
    names = [service.url_env for service in env_services]
    names += list(env_vars) + list(env_session_ports)
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        msg = (
            f"{repeated} set more than once across env_vars, env_session_ports "
            "and env_services url_env."
        )
        raise ValueError(msg)


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


class LLMEnvSpec(EnvSpecBase):
    """The dataset or interactive environment an LLM algorithm fine-tunes against.

    A ``rollout`` env is generative and comes from exactly one source:

    * dataset rows plus a rubric — a single-turn env over labelled rows;
    * ``entrypoint`` — a callable returning a text env, run where training runs
      (or on env hosts under the hosting fields, which need the Ray runtime);
    * ``env_url`` — an OpenEnv service someone else hosts;
    * ``env_image`` — a prebuilt env image the Ray runtime runs as its own Pod.

    A built-in (:data:`BUNDLED_ENVIRONMENTS`) may name no source.

    A ``dataset`` env is teacher-forced over the rows; ``objective`` picks the
    loss.
    """

    num_envs: int | None = Field(
        default=None,
        ge=1,
        description="Environment copies stepped in parallel.",
    )
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
        description="HuggingFace dataset id, or a Parquet or JSONL path, for the dataset sources.",
    )
    dataset_format: Literal["traces"] | None = Field(
        default=None,
        description=(
            "traces: rows are recorded session segments, OpenAI chat messages "
            "in a messages column with optional tools. A rollout renders each "
            "target assistant turn's history as a single-turn prompt and "
            "needs no prompt_template; with no rubric_file_path it scores 0, "
            "which OPD trains on as pure distillation. objective sft trains "
            "every target assistant turn. Takes no columns mapping."
        ),
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
            "(question, answer; prompt, target, or multi-turn messages and "
            "tools for SFT). Unset assumes they already match."
        ),
    )
    response_column: str | None = Field(
        default=None,
        description="Column holding the target response. SFT only.",
    )
    prompt_template: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Conversation template, keyed 'role_index' (system_0, user_1, ...) "
            "so ordering is explicit. Values may use {question} and {answer}, "
            "filled from the mapped columns; other braces are literal. "
            "Required for a rollout over dataset rows, except traces."
        ),
    )
    rubric_file_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("rubric_file_path", "reward_file_path"),
        description=(
            "Path to the Python module holding the rubric (or reward function) "
            "that scores every turn. Required for a rollout over dataset rows, "
            "except under OPD, where unset scores every turn 0; on an env, its "
            "score replaces the env's reward, which it can "
            "read as an input."
        ),
    )
    rubric_name: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("rubric_name", "reward_fn_name"),
        description="Name of the rubric (or reward callable) to import from that module.",
    )
    group_rubric_name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Name of a GroupRubric in the rubric_file_path module. It scores each "
            "finished group of completions together, and each score is added to "
            "that completion's final-turn reward. Leave rubric_name unset to keep "
            "the env's own turn reward. Group replay is off with a group rubric."
        ),
    )
    judge: JudgeSpec | None = Field(
        default=None,
        description=(
            "LLM judge handed to the rubric: a Rubric class whose __init__ takes "
            "judge is built with it. Needs rubric_file_path."
        ),
    )
    reward: RewardSpec | None = Field(
        default=None,
        description="Built-in reward scoring each turn, in place of rubric_file_path.",
    )
    max_reward: float | None = Field(
        default=None,
        description="Reward at which a rollout counts as solved, used for the success metric.",
    )
    min_reward: float | None = Field(
        default=None,
        description=(
            "Lowest reward the env gives. Scores a turn whose tool calls are "
            "malformed or missing, which never reaches the env. Unset scores 0."
        ),
    )
    include_reasoning: bool | None = Field(
        default=None,
        description=(
            "Send the env each turn's reasoning with its markers, not only "
            "the text after the reasoning ends; a Rubric class reads that text "
            "as TurnAction.message. Unset sends it over dataset rows and not "
            "to an env that runs the action. A reward function gets the whole "
            "turn as completion either way."
        ),
    )
    train_test_split: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of the dataset used for training; the rest is held out for evaluation.",
    )

    entrypoint: str | None = Field(
        default=None,
        description=(
            "The env to build: a registry id, or `module:Class` (or "
            "`module:attr` constructor) for a custom upload."
        ),
    )
    factory: str | None = Field(
        default=None,
        description=(
            "Optional `module:attr` callable that receives the entrypoint as "
            "its first argument, e.g. gem:make. Unset constructs entrypoint "
            "as a class or callable."
        ),
    )
    env_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Keyword arguments forwarded to the factory or constructor after "
            "the env id. Not a place for env_id. system_prompt is reserved "
            "and set on the built env."
        ),
    )
    name: str | None = Field(
        default=None,
        description="Catalog / dataset label only. Not the env id.",
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
            "Turn budget per episode. Unset puts no cap on turns: the episode "
            "ends when the env terminates or truncates, or when the next prompt "
            "no longer fits the model's context. Required with env_url, "
            "segment_prompt_tokens or segment_max_images."
        ),
    )
    segment_prompt_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Restart the episode's context when the next prompt would exceed "
            "this many tokens; each segment trains as its own row. Unset keeps "
            "one row per episode."
        ),
    )
    segment_max_images: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Restart the episode's context when the next prompt would carry "
            "more than this many images; set it to the engine's "
            "limit_mm_per_prompt image count. Unset puts no image limit on a "
            "segment."
        ),
    )
    restart_keep_turns: int = Field(
        default=0,
        ge=0,
        description=(
            "Turns a restarted context repeats word for word (the model's "
            "tokens and the observation after each), oldest dropped to leave "
            "room for one more turn under segment_prompt_tokens. Must be below "
            "segment_max_images. 0 keeps only the list of past actions."
        ),
    )
    restart_older_obs_field: str = Field(
        default="",
        description=(
            "Observation field a restarted context shows, as 'field: value', "
            "for each kept turn before the latest, in place of that turn's "
            "full observation text. Needs restart_keep_turns of 2 or more. "
            "Empty repeats every kept observation in full."
        ),
    )
    restart_older_images: bool = Field(
        default=True,
        description=(
            "Whether kept turns before the latest keep their images in a "
            "restarted context. False needs restart_keep_turns of 2 or more."
        ),
    )
    action_error_field: str = Field(
        default="",
        description=(
            "Observation field holding the error text of the action just "
            "taken. A restarted context lists it after that action. Empty "
            "lists the actions alone."
        ),
    )
    repeat_system_prompt: bool = Field(
        default=False,
        description=(
            "Also put env_config.system_prompt after the first observation and "
            "after each image observation, ahead of a trailing 'Question:' "
            "block. False sends each observation unchanged."
        ),
    )
    strict_chat_template_boundary: bool | None = Field(
        default=None,
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
    action_processor: str | None = Field(
        default=None,
        description=(
            "module:fn entrypoint that turns the post-</think> action text "
            "into what the env receives. Unset sends that text unchanged."
        ),
    )
    screenshot_field: str | None = Field(
        default=None,
        description=(
            "Observation field holding the page image. Empty or unset reads no "
            "page image. A field named image is always read when present."
        ),
    )
    goal_images_field: str | None = Field(
        default=None,
        description=(
            "Observation field holding the task images that follow the first "
            "prompt. Empty or unset reads none."
        ),
    )
    modality: Literal["text", "vision"] | None = Field(
        default=None,
        description=(
            "What the env's observations carry: text only, or text and images "
            "(vision). text lets a vision-language model's rollout engine skip "
            "its multimodal towers. Unset is unknown and keeps them."
        ),
    )

    env_url: str | list[str] | None = Field(
        default=None,
        description=(
            "URL of an already-hosted OpenEnv service. A list is a set of "
            "interchangeable replicas dealt across rollout slots."
        ),
    )
    answer_tool: str | None = Field(
        default=None,
        description=(
            "For an MCP-backed env, one of its real tools receiving a turn's "
            "prose when the turn has no tool call. Unset answers such a turn "
            "with retry feedback."
        ),
    )
    action_field: str | None = Field(
        default=None,
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
    max_concurrent_resets: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Env resets one rollout engine runs at once (Ray runtime only). "
            "Unset runs every reset at once; set it to what the env's backing "
            "services can serve."
        ),
    )
    adaptive_task_sampling: bool = Field(
        default=DEFAULT_ADAPTIVE_TASK_SAMPLING,
        description=(
            "Draw each group's task row in proportion to how often that row's "
            "recent groups had differing rewards, so tasks that keep giving tied "
            "groups are drawn less (Ray async rollouts over task rows only). "
            "false keeps the per-epoch shuffle."
        ),
    )
    task_family_field: str | None = Field(
        default=None,
        description=(
            "Field of each env_config.tasks row naming the row's family, such "
            "as a template id. Under adaptive_task_sampling, a row with few "
            "groups is drawn by its family's pooled rate of differing rewards. "
            "The field is not sent to the env. Unset puts no row in a family."
        ),
    )
    task_family_prior_strength: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Groups' worth of weight a family's rate carries in a row's "
            "estimate; a row with more groups than this is drawn mostly by "
            "its own rate."
        ),
    )

    env_hosts: int | None = Field(
        default=None,
        json_schema_extra=ADVANCED_PLATFORM_WRITTEN,
        ge=1,
        description=(
            "Number of env hosts to serve the env on (Ray runtime only). Unset "
            "with no image or packages runs an entrypoint in-process in each "
            "rollout slot, unless cpus_per_env_host, env_host_memory_bytes or "
            "env_host_resource is set, which asks for one host."
        ),
    )
    env_image: str | None = Field(
        default=None,
        description=(
            "Prebuilt env image run verbatim as its own Pod per host, serving "
            "on env_port (Ray runtime only)."
        ),
    )
    env_port: int | None = Field(
        default=None,
        json_schema_extra=ADVANCED_PLATFORM_WRITTEN,
        ge=1,
        le=65535,
        description="TCP port the env_image server listens on.",
    )
    env_sessions_per_host: int | None = Field(
        default=None,
        json_schema_extra=ADVANCED,
        ge=1,
        description=(
            "Sessions per env_image Pod: one container each, serving on "
            "env_port, env_port + 1, ... Unset is 1. More than 1 needs "
            "env_session_ports."
        ),
    )
    env_session_ports: dict[str, int] = Field(
        default_factory=dict,
        json_schema_extra=ADVANCED,
        description=(
            "Container environment set to base + session index in each session "
            "container, so sessions sharing a Pod bind distinct ports."
        ),
    )
    cpus_per_env_host: float | None = Field(
        default=None,
        json_schema_extra=ADVANCED_PLATFORM_WRITTEN,
        gt=0.0,
        description=(
            "CPUs reserved per env host; an env_image Pod requests them with no "
            "CPU limit."
        ),
    )
    env_host_memory_bytes: int | None = Field(
        default=None,
        json_schema_extra=ADVANCED_PLATFORM_WRITTEN,
        gt=0,
        description="Memory reserved per env host.",
    )
    env_host_memory_limit_bytes: int | None = Field(
        default=None,
        json_schema_extra=ADVANCED,
        gt=0,
        description="Memory limit per env_image Pod. Unset sets no limit.",
    )
    env_host_ready_timeout_s: float | None = Field(
        default=None,
        json_schema_extra=ADVANCED,
        gt=0.0,
        description=(
            "Seconds each env_image session Pod may take to become ready, "
            "image pull included. Unset is 600."
        ),
    )
    env_services: list[EnvServiceSpec] = Field(
        default_factory=list,
        json_schema_extra=ADVANCED,
        description=(
            "Backing services the env_image sessions share, started with the "
            "job and deleted with it (Ray runtime on Kubernetes only)."
        ),
    )
    env_host_resource: str | None = Field(
        default=None,
        json_schema_extra=ADVANCED,
        description=(
            "Where env hosts land: a custom Ray resource, or (for env_image) "
            "the Pod nodeSelector — key=value, or a bare pool name."
        ),
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Container environment for each env_image Pod.",
    )

    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Defaults injected into apply_chat_template, e.g. {'enable_thinking': false}.",
    )
    chat_template_path: str | None = Field(
        default=None,
        description="Jinja file whose template replaces the tokenizer's own.",
    )
    apply_chat_template: bool = Field(
        default=True,
        description=(
            "Render prompts through the tokenizer chat template. Digit-token "
            "multi-turn probes set this false and encode observations raw."
        ),
    )

    @property
    def label(self) -> str:
        """Human-readable label: catalog name, dataset, URL, entrypoint, or image."""
        if self.name is not None:
            return self.name
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

    @property
    def name_only_built_in(self) -> bool:
        """Whether this rollout names a built-in and no source."""
        if self.env_type != "rollout" or self.name not in BUNDLED_ENVIRONMENTS:
            return False
        return (
            self._named_dataset() is None
            and self.entrypoint is None
            and self.env_url is None
            and self.env_image is None
        )

    @property
    def sends_reasoning(self) -> bool:
        """Whether each turn's reasoning reaches the env or reward function."""
        if self.include_reasoning is not None:
            return self.include_reasoning
        return self.dataset_backed_rollout

    @property
    def text_only(self) -> bool:
        """Whether every observation is text: declared text, or dataset rows."""
        return self.modality == "text" or self.dataset_backed_rollout

    def _named_dataset(self) -> str | None:
        for value in (self.dataset, self.dataset_path, self.hf_dataset_id):
            if value is not None:
                return str(value)
        return None

    def _dataset_identity_backed(self) -> bool:
        return any(
            value is not None
            for value in (
                self.dataset,
                self.dataset_path,
                self.hf_dataset_id,
                self.columns,
                self.prompt_template,
            )
        )

    @model_validator(mode="after")
    def _check_rollout(self) -> Self:
        reject_legacy_llm_env_spelling(
            entrypoint=self.entrypoint,
            factory=self.factory,
            env_config=self.env_config,
        )
        if self.entrypoint is not None and self.factory is None:
            check_factoryless_entrypoint(self.entrypoint)
        if self.env_type != "rollout":
            return self
        if self.task_family_field is not None and not self.adaptive_task_sampling:
            msg = (
                "task_family_field weights rows under adaptive_task_sampling; "
                "set adaptive_task_sampling: true or drop it."
            )
            raise ValueError(msg)
        dataset = self._named_dataset()
        sources = {
            "a dataset": dataset,
            "entrypoint": self.entrypoint,
            "env_url": self.env_url,
            "env_image": self.env_image,
        }
        named = [name for name, value in sources.items() if value is not None]
        if len(named) != 1 and not self.name_only_built_in:
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
            self.answer_tool is not None or self.request_timeout_s is not None
        ) and dataset is not None:
            msg = (
                "answer_tool / request_timeout_s configure an env reached over "
                "HTTP; a rollout over dataset rows has no such env."
            )
            raise ValueError(msg)
        check_env_pod_fields(
            env_image=self.env_image,
            env_vars=self.env_vars,
            env_sessions_per_host=self.env_sessions_per_host,
            env_session_ports=self.env_session_ports,
            env_host_memory_limit_bytes=self.env_host_memory_limit_bytes,
            env_host_ready_timeout_s=self.env_host_ready_timeout_s,
            env_services=self.env_services,
        )
        if (
            self.action_field is not None
            and self.action_field != "message"
            and dataset is not None
        ):
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
        if self.action_processor is not None and dataset is not None:
            msg = (
                "action_processor rewrites the text an env receives; a "
                "dataset-backed rollout has no such env."
            )
            raise ValueError(msg)
        if dataset is not None and (
            (self.screenshot_field is not None and self.screenshot_field != "")
            or (self.goal_images_field is not None and self.goal_images_field != "")
        ):
            msg = (
                "screenshot_field and goal_images_field name env observation "
                "keys; a dataset-backed rollout has no such observation."
            )
            raise ValueError(msg)
        if self.env_url is not None and self.max_turns is None:
            msg = (
                "max_turns is required with env_url: set the remote env's turn "
                "budget explicitly (1 for single-turn)."
            )
            raise ValueError(msg)
        if dataset is not None:
            if self.dataset_format != TRACES_DATASET_FORMAT:
                if self.prompt_template is None:
                    msg = "Prompt template is required for dataset-backed rollout environments"
                    raise ValueError(msg)
                check_row_placeholders(self.prompt_template, self.columns)
            if self.max_turns not in (None, 1):
                msg = "A dataset-backed rollout environment is single-turn; max_turns must be 1."
                raise ValueError(msg)
            self.max_turns = 1
        if self.max_turns is None and (
            self.segment_prompt_tokens is not None
            or self.segment_max_images is not None
        ):
            msg = (
                "max_turns is required with segment_prompt_tokens or "
                "segment_max_images: a restarted context never fills, so only "
                "the turn cap ends an episode the env does not end."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_modality(self) -> Self:
        if self.modality == "text" and (
            self.screenshot_field or self.goal_images_field
        ):
            msg = (
                "modality text means the env sends no images; drop "
                "screenshot_field and goal_images_field, or set modality vision."
            )
            raise ValueError(msg)
        if self.modality == "vision" and self.dataset_backed_rollout:
            msg = (
                "modality vision needs image observations; a dataset-backed "
                "rollout sends text only."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_restart_keep_turns(self) -> Self:
        check_restart_keep_turns(
            restart_keep_turns=self.restart_keep_turns,
            segment_prompt_tokens=self.segment_prompt_tokens,
            segment_max_images=self.segment_max_images,
            restart_older_obs_field=self.restart_older_obs_field,
            restart_older_images=self.restart_older_images,
        )
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
        pod_fields = [
            field
            for field, value in (
                ("env_vars", self.env_vars),
                ("env_sessions_per_host", self.env_sessions_per_host),
                ("env_session_ports", self.env_session_ports),
                ("env_host_memory_limit_bytes", self.env_host_memory_limit_bytes),
                ("env_host_ready_timeout_s", self.env_host_ready_timeout_s),
                ("env_services", self.env_services),
            )
            if value
        ]
        if pod_fields:
            msg = (
                f"{', '.join(pod_fields)} configure env_image Pods; a dataset "
                "environment runs no env, so drop them."
            )
            raise ValueError(msg)
        if self.rubric_file_path is not None:
            msg = "rubric_file_path has been specified, but is not supported for dataset environments."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_traces_format(self) -> Self:
        if self.dataset_format != TRACES_DATASET_FORMAT:
            return self
        if self._named_dataset() is None:
            msg = "dataset_format traces describes dataset rows; name the dataset too."
            raise ValueError(msg)
        expected = "sft" if self.env_type == "dataset" else None
        if self.objective != expected:
            msg = (
                "A traces dataset trains SFT (env_type dataset, objective sft) or a "
                f"rollout (env_type rollout, no objective); objective {self.objective!r} "
                f"on env_type {self.env_type!r} is not supported."
            )
            raise ValueError(msg)
        mapped = [
            field
            for field, value in (
                ("columns", self.columns),
                ("prompt_template", self.prompt_template),
            )
            if value is not None
        ]
        if mapped:
            msg = (
                f"A traces dataset's column names are the spec; drop {', '.join(mapped)}. "
                "Each row's prompt messages are the rollout prompt."
            )
            raise ValueError(msg)
        if self.response_column not in (None, "completion"):
            msg = "A traces dataset trains its completion column; drop response_column."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _resolve_env_image_host_defaults(self) -> Self:
        if self.env_image is not None:
            if self.env_port is None:
                self.env_port = 8000
            if self.cpus_per_env_host is None:
                self.cpus_per_env_host = ENV_POD_DEFAULT_CPUS
            if self.env_sessions_per_host is None:
                self.env_sessions_per_host = 1
            if self.env_host_ready_timeout_s is None:
                self.env_host_ready_timeout_s = ENV_POD_READY_TIMEOUT_S
        elif (
            self.env_hosts is None
            and self.entrypoint is not None
            and (
                self.cpus_per_env_host is not None
                or self.env_host_memory_bytes is not None
                or self.env_host_resource is not None
            )
        ):
            self.env_hosts = 1
        return self

    @model_validator(mode="after")
    def _check_reward_fields(self) -> Self:
        check_reward_fields(self)
        return self

    @model_validator(mode="after")
    def _apply_conditional_defaults(self) -> Self:
        if self._dataset_identity_backed() and self.train_test_split is None:
            self.train_test_split = 0.9
        if (
            self.rubric_file_path is not None
            and self.rubric_name is None
            and self.group_rubric_name is None
        ):
            self.rubric_name = "reward_fn"
        if self.env_type == "dataset" and self.response_column is None:
            self.response_column = (
                "completion"
                if self.dataset_format == TRACES_DATASET_FORMAT
                else "response"
            )
        if self.env_type == "rollout":
            # Unset here: a submitted default replaces the bundled profile's value.
            if not self.name_only_built_in:
                if self.num_envs is None:
                    self.num_envs = 1
                if self.strict_chat_template_boundary is None:
                    self.strict_chat_template_boundary = True
            env_backed = (
                self.entrypoint is not None
                or self.env_url is not None
                or self.env_image is not None
            )
            if env_backed and self.action_field is None:
                self.action_field = "message"
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


EnvSpec = LLMEnvSpec
