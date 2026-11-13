# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, TypeVar

import click
import yaml

from agilerl.arena import console
from agilerl.arena.byoc import ArenaRootGroup, register_byoc_manifest_group
from agilerl.arena.cli_agent import register_deployment_commands
from agilerl.arena.cli_datasets import datasets_group
from agilerl.arena.cli_manifest import handle_help_option
from agilerl.arena.cli_secrets import secrets_group
from agilerl.arena.client import MEMORY_SCOPES, MemoryScope
from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.exceptions import ArenaError
from agilerl.arena.inference import Agent, SessionInfo
from agilerl.arena.inference.cache import (
    clear_active_session,
    load_active_agent,
    load_active_session,
    save_active_agent,
    save_active_session,
)
from agilerl.arena.memory.cli import memory_group
from agilerl.arena.models import verdict as manifest_verdict
from agilerl.arena.models.model_info import SUPPORTED_MODEL_INFO
from agilerl.arena.models.schema import manifest_schema
from agilerl.arena.output import (
    JSON_OUTPUT_META_KEY,
    emit_csv_preview,
    emit_json_result,
    emit_result,
    emit_session_transcript,
    json_output_enabled,
    select_row,
    supports_interactive_selection,
)
from agilerl.arena.snowflake.cli import snowflake_group

ArenaError.enable_cli_mode()

logger = logging.getLogger(__name__)

ClickDecorated = TypeVar("ClickDecorated", bound=Callable[..., Any] | click.Command)


@click.group(
    cls=ArenaRootGroup,
    add_help_option=False,
    context_settings={"show_default": True},
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    envvar="ARENA_JSON",
    help=(
        "Print each result as one JSON document on stdout, with every field. "
        "Progress, logs and errors go to stderr. Commands that stream text "
        "(agent run/generate, sessions) stay as text."
    ),
)
@click.option(
    "--api-key",
    default=None,
    help="Bearer secret: profile CLI PAT (arena_pat_…) or access token.",
)
@click.option(
    "--org-key",
    default=None,
    help="Organisation API key (arena_org_…). Requires --external-user-id.",
)
@click.option(
    "--external-user-id",
    default=None,
    help="Partner user id sent as X-External-User-Id. Requires --org-key.",
)
@click.option(
    "--base-url",
    default=None,
    help="Override Arena API base URL (for local/dev environments).",
)
@click.option(
    "--keycloak-url",
    default=None,
    help="Override Keycloak URL for authentication.",
)
@click.option("--realm", default=None, help="Override Keycloak realm.")
@click.option("--client-id", default=None, help="Override Keycloak client ID.")
@click.option(
    "--request-timeout",
    type=click.IntRange(1),
    default=30,
    show_default=True,
    help="Default timeout in seconds for API requests.",
)
@click.option(
    "--upload-timeout",
    type=click.IntRange(1),
    default=300,
    show_default=True,
    help="Timeout in seconds for file uploads and cluster register.",
)
@click.option(
    "--module-catalog-url",
    default=None,
    help="Base URL of the signed Terraform module catalog.",
)
@click.option(
    "--module-catalog-public-key",
    default=None,
    help="Ed25519 public key PEM used to verify the module catalog.",
)
@click.option(
    "-h",
    "--help",
    is_flag=True,
    is_eager=False,
    expose_value=False,
    help="Show this message and exit.",
    callback=handle_help_option,
)
@click.pass_context
def main(
    ctx: click.Context,
    as_json: bool,
    api_key: str | None,
    org_key: str | None,
    external_user_id: str | None,
    base_url: str | None,
    keycloak_url: str | None,
    realm: str | None,
    client_id: str | None,
    request_timeout: int,
    upload_timeout: int,
    module_catalog_url: str | None,
    module_catalog_public_key: str | None,
) -> None:
    """Arena CLI - Interact with the Arena RLOps platform directly from the command-line."""
    ctx.meta[JSON_OUTPUT_META_KEY] = as_json
    ctx.obj = CommandConfig(
        api_key=api_key,
        org_key=org_key,
        external_user_id=external_user_id,
        base_url=base_url,
        keycloak_url=keycloak_url,
        realm=realm,
        client_id=client_id,
        request_timeout=request_timeout,
        upload_timeout=upload_timeout,
        module_catalog_url=module_catalog_url,
        module_catalog_public_key=module_catalog_public_key,
    )


@main.command()
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Run browser login even if an API key is set or the stored session is still valid.",
)
@click.option(
    "--timeout",
    type=click.IntRange(1),
    default=300,
    show_default=True,
    help="Maximum seconds to wait for device authorization.",
)
@click.pass_obj
def login(
    config: CommandConfig,
    force: bool,
    timeout: int,
) -> None:
    """Authenticate with Arena."""
    with arena_client(config) as client:
        client.login(timeout=timeout, force=force)


@main.command()
@click.pass_obj
def logout(
    config: CommandConfig,
) -> None:
    """Log out and clear persisted credentials."""
    with arena_client(config) as client:
        client.logout()


@main.group("manifest")
def manifest_group() -> None:
    """Validate a training manifest, or print its JSON Schema."""


@manifest_group.command("validate")
@click.argument("manifest")
@click.option("--payload", is_flag=True, help="Print the resolved manifest.")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help=(
        "Report the verdict as one JSON document on stdout and exit 0 for "
        "either verdict; a non-zero exit then means the tool itself failed."
    ),
)
def manifest_validate(manifest: str, payload: bool, as_json: bool) -> None:
    """Validate MANIFEST (a path, or - for stdin) against the contract."""
    try:
        document = manifest_verdict.read_manifest(manifest)
    except (OSError, ValueError, yaml.YAMLError) as err:
        click.echo(f"could not read {manifest}: {err}", err=True)
        sys.exit(manifest_verdict.EXIT_FAILED)

    result = manifest_verdict.verdict(document)
    if as_json or json_output_enabled():
        click.echo(json.dumps(result))
        return
    if not result["ok"]:
        for error in result["errors"]:
            where = f"{error['path']}: " if error["path"] else ""
            click.echo(f"{where}{error['message']}", err=True)
        sys.exit(manifest_verdict.EXIT_INVALID)
    for warning in result["warnings"]:
        click.echo(f"warning: {warning}", err=True)
    if payload:
        click.echo(json.dumps(result["payload"], indent=2))
    else:
        click.echo(f"{manifest}: valid ({result['payload']['algorithm']['name']})")


@manifest_group.command("template")
@click.argument("algorithm")
@click.option("--model", default=None, help="Catalog model id for an LLM algorithm.")
@click.option("--dataset", default=None, help="Dataset name to fill in.")
@click.option("--gym-env", default=None, help="Gym environment name to fill in.")
@click.option(
    "--gym-env-version",
    default=None,
    help="Gym environment version; required with --gym-env-entrypoint.",
)
@click.option("--gym-env-entrypoint", default=None, help="Gym entrypoint to fill in.")
@click.option(
    "--section",
    type=click.Choice(
        [
            "algorithm",
            "environment",
            "mutation",
            "network",
            "replay_buffer",
            "tournament_selection",
            "training",
        ]
    ),
    default=None,
    help="Print one section instead of the whole manifest.",
)
@click.pass_obj
def manifest_template(
    config: CommandConfig,
    algorithm: str,
    model: str | None,
    dataset: str | None,
    gym_env: str | None,
    gym_env_version: str | None,
    gym_env_entrypoint: str | None,
    section: str | None,
) -> None:
    """Print Arena's default manifest for ALGORITHM as YAML, ready to edit."""
    with arena_client(config) as client:
        spec = client.get_default_run_spec(
            algorithm,
            gym_env=gym_env,
            gym_env_version=gym_env_version,
            gym_env_entrypoint=gym_env_entrypoint,
            dataset_name=dataset,
            field=section,
            pretrained_model_name_or_path=model,
        )
    if json_output_enabled():
        emit_json_result(spec)
    else:
        click.echo(yaml.safe_dump(spec, sort_keys=False), nl=False)


@manifest_group.command("schema")
def manifest_schema_command() -> None:
    """Print the training manifest JSON Schema."""
    click.echo(json.dumps(manifest_schema(), indent=2))


@main.group("user")
def user_group() -> None:
    """Retrieve user / account information."""


@user_group.command("profile")
@click.pass_obj
def user_profile(
    config: CommandConfig,
) -> None:
    """Get current authenticated user profile."""
    with arena_client(config) as client:
        user = client.get_current_user()
        if json_output_enabled():
            emit_json_result(user)
            return
        click.echo(f"User: {user.get('first_name', '')} {user.get('last_name', '')}")
        click.echo(f"Email: {user.get('email', '')}")


@user_group.command("credits")
@click.pass_obj
def user_credits(
    config: CommandConfig,
) -> None:
    """Get remaining account credits."""
    with arena_client(config) as client:
        emit_result(client.get_user_credits())


@main.group("resources")
def resources_group() -> None:
    """List available training compute tiers."""


@resources_group.command("list")
@click.pass_obj
def resources_list(config: CommandConfig) -> None:
    """List resource tiers: ids, specs, and credits per node-hour."""
    with arena_client(config) as client:
        data = client.list_resources()
        tiers = data.get("tiers", {})
        sorted_tiers = sorted(tiers.values(), key=lambda t: t["price_per_node_hour"])
        emit_result(
            sorted_tiers,
            display=lambda tiers: [
                {
                    "Resource ID": tier["name"],
                    "GPUs": f"{tier['num_gpus']}x {tier['gpu_type']}"
                    if tier.get("gpu_type")
                    else "—",
                    "CPUs": tier["num_cpus"],
                    "RAM (GB)": tier["ram_gb"],
                    "credits/node-hour": f"{tier['price_per_node_hour']:.2f}",
                }
                for tier in tiers
            ],
        )


@main.group("env")
def env() -> None:
    """Manage your custom Gym / PettingZoo / GEM environments."""


@env.command("list")
@click.option("--name", default=None, hidden=False)
@click.option(
    "--include-arena/--no-include-arena",
    "include_arena",
    default=False,
    show_default=True,
    type=bool,
)
@click.pass_obj
def env_list(
    config: CommandConfig,
    name: str | None,
    include_arena: bool,
) -> None:
    """List all available environments in Arena, and whether they have been validated and profiled."""
    with arena_client(config) as client:
        emit_result(client.list_environments(name=name, include_arena=include_arena))


@env.command("exists")
@click.argument("name")
@click.option("--version", default=None, show_default=True)
@click.pass_obj
def env_exists(
    config: CommandConfig,
    name: str,
    version: str | None,
) -> None:
    """Check if an environment version exists in Arena."""
    with arena_client(config) as client:
        emit_result(client.environment_exists(name=name, version=version))


@env.command("entrypoints")
@click.argument("name")
@click.option("--version", default=None, show_default=True)
@click.pass_obj
def env_entrypoints(
    config: CommandConfig,
    name: str,
    version: str | None,
) -> None:
    """List available entrypoints for an existing environment version."""
    with arena_client(config) as client:
        emit_result(
            client.list_environment_entrypoints(name=name, version=version),
            columns=["Entrypoints"],
        )


@env.command("validate")
@click.argument("name", type=str)
@click.option("--version", default=None, show_default=True)
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to environment source directory or .tar.gz archive.",
)
@click.option(
    "--env-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to env_config.yaml file.",
)
@click.option(
    "--requirements",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to requirements.txt file.",
)
@click.option(
    "--entrypoint",
    default=None,
    help="Optional entrypoint override. Useful when multiple entrypoints exist.",
)
@click.option(
    "--description",
    default=None,
    help="Optional description of the environment.",
)
@click.option("--multi-agent/--single-agent", default=False, show_default=True)
@click.option(
    "--language-based/--classic",
    default=False,
    show_default=True,
    help="Environment follows the GEM API.",
)
@click.option("--do-rollouts/--no-do-rollouts", default=False, show_default=True)
@click.pass_obj
def env_validate(
    config: CommandConfig,
    name: str,
    version: str | None,
    source: Path | None,
    env_config: Path | None,
    requirements: Path | None,
    entrypoint: str | None,
    description: str | None,
    multi_agent: bool,
    language_based: bool,
    do_rollouts: bool,
) -> None:
    """Validate an environment on Arena.

    Pass NAME to validate an already-registered environment.  Pass --source
    to register and validate in one step.  NAME is always required and is never
    inferred from --source.
    """
    with arena_client(config) as client:
        result = client.validate_environment(
            name=name,
            version=version,
            source=source,
            env_config=env_config,
            requirements=requirements,
            entrypoint=entrypoint,
            description=description,
            multi_agent=multi_agent,
            language_based=language_based,
            do_rollouts=do_rollouts,
        )
    emit_json_result(result)


_PROFILE_METRIC_LABELS: dict[str, str] = {
    "avg_cpu_per_env": "Avg CPU per Env (%)",
    "memory_per_env_gb": "Memory per Env (GB)",
    "peak_cpu_per_env": "Peak CPU per Env (%)",
    "steps_per_second": "Steps per Second",
}


def _format_profile_metrics(raw: dict[str, float]) -> dict[str, str]:
    """Transform raw profiling metric keys into human-readable labels with formatted values."""
    formatted: dict[str, str] = {}
    for key, value in raw.items():
        label = _PROFILE_METRIC_LABELS.get(key, key.replace("_", " ").title())
        formatted[label] = f"{value:.3f}"
    return formatted


@env.command("profile")
@click.argument("name")
@click.option("--version", default=None, show_default=True)
@click.option(
    "--multi-agent/--single-agent",
    "multi_agent",
    default=None,
    help="Override the stored multi-agent flag. Omit it to use the saved value.",
)
@click.pass_obj
def env_profile(
    config: CommandConfig,
    name: str,
    version: str | None,
    multi_agent: bool | None,
) -> None:
    """Profile a validated environment in Arena and get its resource requirements."""
    with arena_client(config) as client:
        result = client.profile_environment(
            name=name, version=version, multi_agent=multi_agent
        )
        if result:
            emit_result(
                _format_profile_metrics(result),
                columns=["Metric", "Estimate"],
            )


@env.command("delete")
@click.argument("name")
@click.option("--version", default=None, show_default=True)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without interactive prompt.",
)
@click.pass_obj
def env_delete(
    config: CommandConfig,
    name: str,
    version: str | None,
    yes: bool,
) -> None:
    """Delete an environment version (or all versions if version is None) from Arena."""
    with arena_client(config) as client:
        client.delete_environment(name=name, version=version, confirm=yes)


@env.command("duplicate")
@click.argument("name")
@click.argument("new_version")
@click.option("--version", default=None, show_default=True)
@click.pass_obj
def env_duplicate(
    config: CommandConfig,
    name: str,
    new_version: str,
    version: str | None,
) -> None:
    """Copy an existing environment version to a new version name. If no version is provided, the latest version will be duplicated."""
    with arena_client(config) as client:
        client.duplicate_environment_version(
            name=name,
            new_version=new_version,
            version=version,
        )


@main.group("models")
def models_group() -> None:
    """List HuggingFace models in the Arena catalog."""


@models_group.command("list")
@click.pass_obj
def models_list(config: CommandConfig) -> None:
    """List HuggingFace models in the Arena catalog.

    LoRA modules and ranks for one model: ``arena models info MODEL_NAME``.
    """
    with arena_client(config) as client:
        rows = client.list_models()
    emit_result(rows, display=lambda rows: [_model_summary_row(row) for row in rows])


@models_group.command("supported")
def models_supported() -> None:
    """Print every supported model's static info as JSON (no server call)."""
    click.echo(
        json.dumps(
            [info.model_dump(mode="json") for info in SUPPORTED_MODEL_INFO.values()]
        )
    )


@models_group.command("info")
@click.argument("model_name")
@click.pass_obj
def models_info(config: CommandConfig, model_name: str) -> None:
    """Fetch LoRA modules and catalog metadata for a model."""
    with arena_client(config) as client:
        emit_result(client.get_model_info(model_name))


@main.group("experiments")
def experiment() -> None:
    """Manage experiments (training jobs) by name."""


@experiment.command("submit")
@click.argument(
    "manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--resource-id",
    type=str,
    default=None,
    help=(
        "Arena cluster type to submit the experiment to. Omit every resource "
        "flag to let Arena pick the cheapest fit."
    ),
)
@click.option(
    "--num-nodes",
    type=int,
    default=None,
    help=(
        "Number of nodes to use for the experiment. Omit every resource flag "
        "to let Arena pick the cheapest fit."
    ),
)
@click.option(
    "--num-gpus",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Total GPUs. Arena uses the fewest nodes. Not combinable with "
        "--resource-id or --num-nodes."
    ),
)
@click.option(
    "--gpu-type",
    type=str,
    default=None,
    help=(
        "Short GPU name such as h100-80, b200 or l4. Not combinable with "
        "--resource-id or --num-nodes."
    ),
)
@click.option(
    "--project",
    type=str,
    default=None,
    help="Project to submit the experiment to (defaults to the default project in ~/.arena/config.json).",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="Name of the experiment.",
)
@click.option(
    "--reward-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Python rubric or reward function scoring a rollout job (multipart "
        "submit). Required when the manifest targets a reasoning dataset."
    ),
)
@click.option(
    "--completion",
    default=None,
    help=(
        "Model completion for reward validation. Defaults to the reference "
        "answer from the first dataset row when omitted."
    ),
)
@click.option(
    "--idempotency-key",
    default=None,
    help="Reuse this key to replay the same submit instead of starting another job.",
)
@click.pass_obj
def experiment_submit(
    config: CommandConfig,
    manifest: Path,
    resource_id: str | None,
    num_nodes: int | None,
    num_gpus: int | None,
    gpu_type: str | None,
    project: str | None,
    experiment_name: str | None,
    reward_file: Path | None,
    completion: str | None,
    idempotency_key: str | None,
) -> None:
    """Submit an experiment from a manifest."""
    with arena_client(config) as client:
        result = client.submit_experiment(
            manifest=manifest,
            resource_id=resource_id,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            gpu_type=gpu_type,
            project=project,
            experiment_name=experiment_name,
            reward_file=reward_file,
            completion=completion,
            idempotency_key=idempotency_key,
        )
    emit_json_result(result)


@experiment.command("list")
@click.option(
    "--project",
    default=None,
    help="Project whose experiments should be listed (defaults to the default project in ~/.arena/config.json).",
)
@click.pass_obj
def experiment_list(config: CommandConfig, project: str | None) -> None:
    """List experiments in a project."""
    with arena_client(config) as client:
        rows = client.list_experiments(project=project)
    emit_result(
        rows,
        empty_message=(
            "No experiments yet. Submit one with: arena experiments submit <manifest>"
        ),
    )


@experiment.command("resume")
@click.argument("experiment_name")
@click.option(
    "--max-steps",
    type=int,
    required=True,
    help="Maximum training steps for the resumed run.",
)
@click.option(
    "--checkpoint",
    default=None,
    help="Checkpoint to resume from. Omit it to use the best checkpoint.",
)
@click.option(
    "--new-name",
    default=None,
    help="Name for the resumed experiment.",
)
@click.option(
    "--project",
    default=None,
    help="Project, when the experiment name exists in more than one.",
)
@click.pass_obj
def experiment_resume(
    config: CommandConfig,
    experiment_name: str,
    max_steps: int,
    checkpoint: str | None,
    new_name: str | None,
    project: str | None,
) -> None:
    """Resume an experiment by name."""
    with arena_client(config) as client:
        emit_result(
            client.resume_experiment(
                experiment_name=experiment_name,
                max_steps=max_steps,
                checkpoint=checkpoint,
                new_name=new_name,
                project=project,
            )
        )


@experiment.command("stop")
@click.argument("experiment_name")
@click.pass_obj
def experiment_stop(
    config: CommandConfig,
    experiment_name: str,
) -> None:
    """Stop a running experiment by name (same as the platform training job name)."""
    with arena_client(config) as client:
        emit_result(client.stop_experiment(experiment_name))


@experiment.command("checkpoints")
@click.argument("experiment_name")
@click.pass_obj
def experiment_checkpoints(
    config: CommandConfig,
    experiment_name: str,
) -> None:
    """List checkpoints for an experiment."""
    with arena_client(config) as client:
        rows = client.list_checkpoints(experiment_name=experiment_name)
    emit_result(
        rows,
        display=_format_checkpoint_rows_for_display,
        empty_message=f"No checkpoints for {experiment_name}.",
    )


@experiment.command("metrics")
@click.argument("experiment_name")
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    help="Metric to download. Omit to request all metrics. Repeat for multiple.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination file path. Defaults to a name derived from the experiment.",
)
@click.option(
    "--preview-rows",
    type=click.IntRange(0),
    default=5,
    show_default=True,
    help="When CSV is returned, preview this many rows in a rich table.",
)
@click.pass_obj
def experiment_metrics(
    config: CommandConfig,
    experiment_name: str,
    metrics: tuple[str, ...],
    output_file: Path | None,
    preview_rows: int,
) -> None:
    """Download metrics for an experiment by name."""
    with arena_client(config) as client:
        metrics_list: list[str] | None = list(metrics) if metrics else None
        target_path = client.download_experiment_metrics(
            experiment_name=experiment_name,
            output_path=output_file,
            metrics=metrics_list,
        )

        # Preview the metrics if requested
        if preview_rows > 0:
            preview_payload, preview_ct, _ = client.preview_experiment_metrics_csv(
                experiment_name=experiment_name,
                preview_rows=preview_rows,
                metrics=metrics_list,
            )
            if (preview_ct or "").startswith("text/csv"):
                emit_csv_preview(preview_payload, max_rows=preview_rows)
            elif target_path.suffix == ".csv":
                emit_csv_preview(target_path.read_bytes(), max_rows=preview_rows)


def _redact_agent_rows_for_display(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop any credential a deployment row still carries before it is displayed."""
    out: list[dict[str, Any]] = []
    for row in rows:
        copy_row = dict(row)
        copy_row.pop("api_key", None)
        spec = copy_row.get("spec")
        if isinstance(spec, dict):
            spec_copy = dict(spec)
            spec_copy.pop("api_key", None)
            copy_row["spec"] = spec_copy
        out.append(copy_row)
    return out


def _model_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    """One catalog row reduced to the columns that fit a terminal table."""
    num_params = row.get("num_params")
    return {
        "model": row["model_name"],
        "params": f"{num_params / 1e9:.2f}B" if num_params else "—",
        "max_context": row.get("max_context_length") or "—",
        "status": "disabled"
        if row.get("enabled") is False
        else row.get("status") or "—",
    }


def _format_checkpoint_rows_for_display(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Put ``steps`` first and round ``size_mb`` to two decimals for the table."""
    out: list[dict[str, Any]] = []
    for row in rows:
        ordered: dict[str, Any] = {}
        if "steps" in row:
            ordered["steps"] = row["steps"]
        for key, value in row.items():
            if key == "steps":
                continue
            if key == "size_mb" and isinstance(value, (int, float)):
                value = f"{value:.2f}"
            ordered[key] = value
        out.append(ordered)
    return out


def _resolve_agent_target(
    deployment_name: str | None,
    experiment_name: str | None,
    project_name: str | None,
) -> tuple[str, str | None, str | None]:
    """Use an explicit deployment name or the active agent from ``arena agent run``."""
    if deployment_name:
        return deployment_name.strip(), experiment_name, project_name
    active = load_active_agent()
    if active is None:
        msg = "No active agent. Set one with: arena agent run <deployment>"
        raise click.UsageError(msg)
    exp = experiment_name if experiment_name is not None else active.experiment_name
    proj = project_name if project_name is not None else active.project_name
    return active.deployment_name, exp, proj


@contextmanager
def _open_agent(
    config: CommandConfig,
    deployment_name: str | None,
    refresh: bool,
    experiment_name: str | None,
    project_name: str | None,
) -> Iterator[tuple[str, Agent]]:
    """Open the named deployment, or the agent set by ``arena agent run``.

    Yields the resolved deployment name alongside the agent, since it is the key
    the session cache is stored under.
    """
    target, experiment_name, project_name = _resolve_agent_target(
        deployment_name, experiment_name, project_name
    )
    with (
        arena_client(config) as client,
        client.open_inference_agent(
            target,
            refresh=refresh,
            experiment_name=experiment_name,
            project_name=project_name,
        ) as agent,
    ):
        yield target, agent


def _agent_deployment_options(func: ClickDecorated) -> ClickDecorated:
    """Shared ``--refresh`` / experiment / project options for agent commands."""
    func = click.option(
        "--refresh",
        is_flag=True,
        default=False,
        help="Refetch the deployment URL from Arena instead of using the local cache.",
    )(func)
    func = click.option(
        "--experiment-name",
        "experiment_name",
        default=None,
        help="Experiment name (exact) — narrows duplicate deployment names.",
    )(func)
    return click.option(
        "--project-name",
        "project_name",
        default=None,
        help="Project name (exact) — narrows duplicate deployment names.",
    )(func)


@main.group("agent")
def agent_cli() -> None:
    """List, deploy, run, and generate with Arena agents."""


@agent_cli.command("list")
@click.option("--name", default=None, help="Exact deployment name filter.")
@click.option(
    "--experiment-name",
    "experiment_name",
    default=None,
    help="Exact experiment name filter.",
)
@click.option(
    "--project-name",
    "project_name",
    default=None,
    help="Exact project name filter.",
)
@click.pass_obj
def agent_list(
    config: CommandConfig,
    name: str | None,
    experiment_name: str | None,
    project_name: str | None,
) -> None:
    """List deployed agents."""
    with arena_client(config) as client:
        rows = client.list_inference_deployments(
            name=name,
            experiment_name=experiment_name,
            project_name=project_name,
        )
    filtered = name or experiment_name or project_name
    emit_result(
        _redact_agent_rows_for_display(rows),
        empty_message=(
            "No deployed agents match those filters."
            if filtered
            else "No deployed agents. Deploy one with: arena agent deploy <experiment>"
        ),
    )


@agent_cli.command("deploy")
@click.argument("experiment_name")
@click.option(
    "--checkpoint",
    default=None,
    help="Checkpoint to deploy. Omit to deploy the best checkpoint.",
)
@click.option(
    "--memory-scope",
    "memory_scope",
    type=click.Choice(MEMORY_SCOPES),
    default=None,
    help="Who an LLM deployment keeps chat sessions for. New deployments default "
    "to 'user'. Redeploying without this flag keeps the stored scope.",
)
@click.option(
    "--num-gpus",
    type=click.IntRange(min=1),
    default=None,
    help="GPUs on the deployment's node.",
)
@click.option(
    "--gpu-type",
    type=str,
    default=None,
    help="Short GPU name such as h100-80, b200 or l4.",
)
@click.pass_obj
def agent_deploy(
    config: CommandConfig,
    experiment_name: str,
    checkpoint: str | None,
    memory_scope: MemoryScope | None,
    num_gpus: int | None,
    gpu_type: str | None,
) -> None:
    """Deploy an agent from an experiment checkpoint.

    A deployment's memory scope is fixed once it exists, so ``--memory-scope``
    only takes effect on the first deploy.

    Without ``--num-gpus`` or ``--gpu-type``, Arena picks the cheapest node
    type that fits.
    """
    with arena_client(config) as client:
        result = client.deploy_agent(
            experiment_name=experiment_name,
            checkpoint=checkpoint,
            memory_scope=memory_scope,
            num_gpus=num_gpus,
            gpu_type=gpu_type,
        )
    emit_json_result(result)


register_deployment_commands(agent_cli)


@agent_cli.command("run")
@click.argument("deployment_name")
@_agent_deployment_options
@click.pass_obj
def agent_run(
    config: CommandConfig,
    deployment_name: str,
    refresh: bool,
    experiment_name: str | None,
    project_name: str | None,
) -> None:
    """Select a deployment as the active agent to use for inference."""
    with arena_client(config) as client:
        client._ensure_inference_binding(
            deployment_name,
            refresh=refresh,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        save_active_agent(
            deployment_name,
            experiment_name=experiment_name,
            project_name=project_name,
        )


def _session_for_prompt(
    deployment: str,
    session_id: str | None,
    *,
    new_session: bool,
) -> str:
    """Pick the session a prompt runs in, starting and recording one if needed.

    A new conversation gets its id here rather than from the deployment, which is
    what lets the next prompt carry on from it: a streamed reply is plain text
    and has nowhere to hand an id back.
    """
    if session_id:
        return session_id.strip()
    if not new_session:
        resumed = load_active_session(deployment)
        if resumed:
            return resumed
    started = str(uuid.uuid4())
    save_active_session(deployment, started)
    logger.info("Started session %s.", started)
    return started


@agent_cli.command("generate")
@click.argument("deployment_name", required=False, default=None)
@_agent_deployment_options
@click.option(
    "--prompt",
    required=True,
    help="Prompt text for the LLM deployment.",
)
@click.option(
    "--session-id",
    "session_id",
    default=None,
    help="Chat session to continue. Overrides a session set by "
    "'arena agent sessions resume'.",
)
@click.option(
    "--new-session",
    "new_session",
    is_flag=True,
    default=False,
    help="Start a new conversation and continue it in later prompts.",
)
@click.pass_obj
def agent_generate(
    config: CommandConfig,
    deployment_name: str | None,
    refresh: bool,
    experiment_name: str | None,
    project_name: str | None,
    prompt: str,
    session_id: str | None,
    new_session: bool,
) -> None:
    """Stream a completion from a prompt through a deployed LLM agent.

    Omit *deployment_name* to use the agent set by ``arena agent run``. Prompts
    continue the current conversation, starting one on the first prompt, so a
    deployment that keeps chat history remembers what you asked.
    """
    if session_id and new_session:
        msg = "Pass either --session-id or --new-session, not both."
        raise click.UsageError(msg)
    with _open_agent(
        config, deployment_name, refresh, experiment_name, project_name
    ) as (deployment, agent):
        resolved = _session_for_prompt(deployment, session_id, new_session=new_session)
        for chunk in agent.generate_stream(prompt, session_id=resolved):
            console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
        console.print()


SESSION_COLUMNS = ["Session Id", "Created At", "Last Updated"]


@agent_cli.group("sessions")
def agent_sessions() -> None:
    """Browse chat sessions held by a deployed LLM agent."""


@agent_sessions.command("list")
@click.argument("deployment_name", required=False, default=None)
@_agent_deployment_options
@click.pass_obj
def agent_sessions_list(
    config: CommandConfig,
    deployment_name: str | None,
    refresh: bool,
    experiment_name: str | None,
    project_name: str | None,
) -> None:
    """List your chat sessions with a deployed LLM agent.

    Omit *deployment_name* to use the agent set by ``arena agent run``. The
    session being resumed, if any, is marked in the first column.
    """
    with _open_agent(
        config, deployment_name, refresh, experiment_name, project_name
    ) as (deployment, agent):
        resumed = load_active_session(deployment)
        rows = [
            {
                "current": "●" if session.session_id == resumed else "",
                **session.model_dump(),
            }
            for session in agent.list_sessions()
        ]
        emit_result(
            rows,
            columns=["Current", *SESSION_COLUMNS],
            empty_message=f"No chat sessions on {deployment} yet.",
        )


@agent_sessions.command("get")
@click.argument("session_id")
@click.argument("deployment_name", required=False, default=None)
@_agent_deployment_options
@click.pass_obj
def agent_sessions_get(
    config: CommandConfig,
    session_id: str,
    deployment_name: str | None,
    refresh: bool,
    experiment_name: str | None,
    project_name: str | None,
) -> None:
    """Show one chat session and its messages.

    Omit *deployment_name* to use the agent set by ``arena agent run``.
    """
    with _open_agent(
        config, deployment_name, refresh, experiment_name, project_name
    ) as (_, agent):
        emit_session_transcript(agent.get_session(session_id).model_dump())


def _choose_session(
    sessions: list[SessionInfo],
    session_id: str | None,
    resumed: str | None,
) -> str:
    """Take the caller's session id, or have them pick one from the list."""
    known = {session.session_id for session in sessions}
    if session_id:
        wanted = session_id.strip()
        if wanted not in known:
            msg = f"No session {wanted!r} on this deployment. See: arena agent sessions list"
            raise click.UsageError(msg)
        return wanted

    if not supports_interactive_selection():
        msg = "Choosing from a list needs an interactive terminal. Pass --session-id."
        raise click.UsageError(msg)

    start = next(
        (i for i, session in enumerate(sessions) if session.session_id == resumed), 0
    )
    chosen = select_row(
        [session.model_dump() for session in sessions],
        columns=SESSION_COLUMNS,
        selected=start,
    )
    if chosen is None:
        raise click.Abort
    return str(chosen["session_id"])


@agent_sessions.command("resume")
@click.argument("deployment_name", required=False, default=None)
@_agent_deployment_options
@click.option(
    "--session-id",
    "session_id",
    default=None,
    help="Session to resume. Omit to pick one from a list.",
)
@click.pass_obj
def agent_sessions_resume(
    config: CommandConfig,
    deployment_name: str | None,
    refresh: bool,
    experiment_name: str | None,
    project_name: str | None,
    session_id: str | None,
) -> None:
    """Continue a chat session in later commands without repeating its id.

    Omit ``--session-id`` to choose one with the arrow keys. The session is
    remembered per deployment, so ``arena agent generate --prompt '...'`` carries
    on the conversation until you run ``arena agent sessions clear``.
    """
    with _open_agent(
        config, deployment_name, refresh, experiment_name, project_name
    ) as (deployment, agent):
        sessions = agent.list_sessions()
        if not sessions:
            msg = (
                f"No chat sessions on {deployment!r} yet. Start one with: "
                "arena agent generate --prompt '...'"
            )
            raise click.UsageError(msg)
        chosen = _choose_session(sessions, session_id, load_active_session(deployment))

    save_active_session(deployment, chosen)
    console.print(f"Resuming session {chosen} on {deployment}.")


@agent_sessions.command("delete")
@click.argument("session_id")
@click.argument("deployment_name", required=False, default=None)
@_agent_deployment_options
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without interactive prompt.",
)
@click.pass_obj
def agent_sessions_delete(
    config: CommandConfig,
    session_id: str,
    deployment_name: str | None,
    refresh: bool,
    experiment_name: str | None,
    project_name: str | None,
    yes: bool,
) -> None:
    """Delete a chat session and its history from the deployment.

    Omit *deployment_name* to use the agent set by ``arena agent run``. Unlike
    ``arena agent sessions clear``, this cannot be undone: the messages are gone
    from the deployment, not just from this machine.
    """
    if not yes and not click.confirm(
        f"Delete session {session_id!r} and its messages?", default=False
    ):
        click.echo("Aborted.")
        return

    with _open_agent(
        config, deployment_name, refresh, experiment_name, project_name
    ) as (deployment, agent):
        deleted = agent.delete_session(session_id)

    if load_active_session(deployment) == session_id.strip():
        clear_active_session(deployment)
    if deleted:
        console.print(f"Deleted session {session_id} on {deployment}.")
    else:
        console.print(f"No session {session_id} on {deployment}.")


@agent_sessions.command("clear")
@click.argument("deployment_name", required=False, default=None)
def agent_sessions_clear(deployment_name: str | None) -> None:
    """Clear the current session, so the next prompt starts a new conversation.

    Omit *deployment_name* to use the agent set by ``arena agent run``. Only the
    local pointer is cleared; the session itself stays on the deployment and
    ``arena agent sessions resume`` can pick it up again.
    """
    target, _, _ = _resolve_agent_target(deployment_name, None, None)
    if clear_active_session(target):
        console.print(f"Cleared the session on {target}.")
    else:
        console.print(f"No session to clear on {target}.")


@main.group("projects")
def projects() -> None:
    """Manage your projects in Arena."""


@projects.command("list")
@click.pass_obj
def projects_list(config: CommandConfig) -> None:
    """List all projects in Arena."""
    with arena_client(config) as client:
        emit_result(client.list_projects(), columns=["Name", "Type", "Description"])


@projects.command("create")
@click.argument("name")
@click.option("--description", default=None, help="Description of the project.")
@click.option("--llm-based/--simulation-based", default=False, show_default=True)
@click.pass_obj
def projects_create(
    config: CommandConfig,
    name: str,
    description: str | None,
    llm_based: bool,
) -> None:
    """Create a new project in Arena.

    :param name: The name of the project.
    :type name: str
    :param description: The description of the project.
    :type description: str | None
    :param llm_based: Whether the project is LLM-based.
    :type llm_based: bool
    :returns: A dictionary containing the created project.
    :rtype: dict[str, Any]
    """
    with arena_client(config) as client:
        client.create_project(name=name, description=description, llm_based=llm_based)


@projects.command("delete")
@click.argument("name", type=str)
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without interactive prompt.",
)
@click.pass_obj
def projects_delete(config: CommandConfig, name: str, yes: bool) -> None:
    """Delete a project in Arena."""
    if not yes and not click.confirm(f"Delete project {name!r}?", default=False):
        click.echo("Aborted.")
        return

    with arena_client(config) as client:
        client.delete_project(name=name)


@projects.command("set-default")
@click.argument("name")
@click.pass_obj
def projects_set_default(config: CommandConfig, name: str) -> None:
    """Set the default project for commands that accept --project."""
    with arena_client(config) as client:
        client.set_default_project(name)


@projects.command("get-default")
@click.pass_obj
def projects_get_default(config: CommandConfig) -> None:
    """Show the current default project."""
    with arena_client(config) as client:
        project = client.get_default_project()
    if json_output_enabled():
        emit_json_result({"project": project})
    elif project:
        click.echo(project)
    else:
        click.echo("No default project set. Use 'arena projects set-default <name>'.")


CLI_PLUGIN_GROUP = "agilerl.arena.cli"


def load_cli_plugins(group: click.Group) -> None:
    """Let installed packages add commands to *group*.

    Each entry point in ``agilerl.arena.cli`` names a callable that takes the
    group, e.g. ``[project.entry-points."agilerl.arena.cli"] my = "pkg.cli:register"``.

    :param group: The CLI group plugins register commands on.
    """
    for plugin in entry_points(group=CLI_PLUGIN_GROUP):
        plugin.load()(group)


main.add_command(datasets_group)
main.add_command(memory_group)
main.add_command(secrets_group)
main.add_command(snowflake_group)

if __name__ == "__main__":
    # Plugins import agilerl.arena.cli, a second copy of this module under
    # ``-m``; run that copy's group, which loads them once.
    from agilerl.arena.cli import main as package_main

    package_main()
else:
    load_cli_plugins(main)
    register_byoc_manifest_group(main)
