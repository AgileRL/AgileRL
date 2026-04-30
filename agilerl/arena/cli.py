from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import click

from agilerl.arena.client import ArenaClient
from agilerl.arena.config import CommandConfig, build_client
from agilerl.arena.exceptions import ArenaError
from agilerl.arena.output import (
    emit_csv_preview,
    emit_result,
    handle_error,
)
from agilerl.arena.payloads import resolve_metrics_output_path

ArenaError.enable_cli_mode()


@contextmanager
def arena_client(
    config: CommandConfig,
) -> Generator[ArenaClient, None, None]:
    """Build an :class:`ArenaClient`, handle errors, and guarantee cleanup."""
    client = build_client(config)
    try:
        yield client
    except Exception as exc:
        handle_error(exc)
    finally:
        client.close()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--api-key", default=None, help="Arena API key.")
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
    help="Timeout in seconds for file upload requests.",
)
@click.pass_context
def main(
    ctx: click.Context,
    api_key: str | None,
    base_url: str | None,
    keycloak_url: str | None,
    realm: str | None,
    client_id: str | None,
    request_timeout: int,
    upload_timeout: int,
) -> None:
    """Arena CLI - Interact with the Arena RLOps platform directly from the command-line."""
    ctx.obj = CommandConfig(
        api_key=api_key,
        base_url=base_url,
        keycloak_url=keycloak_url,
        realm=realm,
        client_id=client_id,
        request_timeout=request_timeout,
        upload_timeout=upload_timeout,
    )


@main.command()
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
    timeout: int,
) -> None:
    """Authenticate with Arena."""
    with arena_client(config) as client:
        client.login(timeout=timeout)


@main.command()
@click.pass_obj
def logout(
    config: CommandConfig,
) -> None:
    """Log out and clear persisted credentials."""
    with arena_client(config) as client:
        client.logout()


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
    """List training compute tiers (resource_id values) available on Arena."""


@resources_group.command("list")
@click.pass_obj
def resources_list(config: CommandConfig) -> None:
    """List resource tiers: ids, specs, and credits per node-hour."""
    with arena_client(config) as client:
        emit_result(client.list_resources())


@main.group("env")
def env() -> None:
    """Manage your custom Gym / PettingZoo environments in Arena."""


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
    include_arena: bool | None,
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
@click.argument("name", required=False, default=None, type=str)
@click.option("--name", "name_opt", default=None, hidden=True)
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
@click.option("--do-rollouts/--no-do-rollouts", default=False, show_default=True)
@click.pass_obj
def env_validate(
    config: CommandConfig,
    name: str | None,
    name_opt: str | None,
    version: str | None,
    source: Path | None,
    env_config: Path | None,
    requirements: Path | None,
    entrypoint: str | None,
    description: str | None,
    multi_agent: bool,
    do_rollouts: bool,
) -> None:
    """Validate an environment on Arena.

    Pass 'name' to validate an already-registered environment.  Pass --source
    to register and validate in one step.  When using --source without a positional
    'name', the source directory/file name is used by default.
    """
    env_name = name or name_opt or (source.stem if source else None)
    if env_name is None:
        msg = "Provide a name of an already-registered environment or use --source to upload and validate from scratch."
        raise click.UsageError(msg)

    with arena_client(config) as client:
        client.validate_environment(
            name=env_name,
            version=version,
            source=source,
            env_config=env_config,
            requirements=requirements,
            entrypoint=entrypoint,
            description=description,
            multi_agent=multi_agent,
            do_rollouts=do_rollouts,
        )


@env.command("profile")
@click.argument("name")
@click.option("--version", default=None, show_default=True)
@click.pass_obj
def env_profile(
    config: CommandConfig,
    name: str,
    version: str | None,
) -> None:
    """Profile a validated environment in Arena and get its resource requirements."""
    with arena_client(config) as client:
        result = client.profile_environment(name=name, version=version)
        if result:
            emit_result(result)


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
    """Delete an environment version from Arena."""
    if not yes and not click.confirm(
        f"Delete environment {name}:{version}?", default=False
    ):
        click.echo("Aborted.")
        return

    with arena_client(config) as client:
        result = client.delete_environment(name=name, version=version)
        if result in ("", None):
            emit_result({"deleted": True, "name": name, "version": version})
            return
        emit_result(result)


@env.command("duplicate")
@click.argument("name")
@click.argument("new_version_name")
@click.option("--version", default=None, show_default=True)
@click.pass_obj
def env_duplicate(
    config: CommandConfig,
    name: str,
    new_version_name: str,
    version: str | None,
) -> None:
    """Copy an existing environment version to a new version name (S3 + registry)."""
    with arena_client(config) as client:
        emit_result(
            client.duplicate_environment_version(
                name=name,
                new_version_name=new_version_name,
                version=version,
            ),
        )


@main.group("train")
def train() -> None:
    """Submit, validate, and inspect training jobs."""


@train.command("submit")
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to manifest file.",
)
@click.option(
    "--resource-id",
    type=str,
    default="arena-medium",
    help="Arena cluster type to submit the training job to.",
)
@click.option(
    "--num-nodes",
    type=int,
    default=2,
    help="Number of nodes to use for the training job.",
)
@click.option(
    "--project",
    type=str,
    default=None,
    help="Project to submit the training job to.",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="Name of the experiment to submit the training job to.",
)
@click.pass_obj
def train_submit(
    config: CommandConfig,
    manifest: Path,
    resource_id: int | None,
    num_nodes: int | None,
    project: str | None,
    experiment_name: str | None,
) -> None:
    """Submit a training job from manifest and/or existing training job IDs."""
    with arena_client(config) as client:
        client.submit_training_job(
            manifest=manifest,
            resource_id=resource_id,
            num_nodes=num_nodes,
            project=project,
            experiment_name=experiment_name,
        )


@train.command("get-metrics")
@click.argument("experiment_id", type=int)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    required=True,
    help="Metric to download. Repeat for multiple metrics.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination file path. Defaults to ./experiment_<id>_metrics.csv (or .zip).",
)
@click.option(
    "--preview-rows",
    type=click.IntRange(0),
    default=10,
    show_default=True,
    help="When CSV is returned, preview this many rows in a rich table.",
)
@click.pass_obj
def train_get_metrics(
    config: CommandConfig,
    experiment_id: int,
    metrics: tuple[str, ...],
    output_file: Path | None,
    preview_rows: int,
) -> None:
    """Download metrics data for an experiment as CSV (or zip)."""
    with arena_client(config) as client:
        payload, content_type, disposition = client.download_experiment_metrics(
            experiment_id=experiment_id, metrics=list(metrics)
        )
        target_path = resolve_metrics_output_path(
            experiment_id=experiment_id,
            payload=payload,
            content_type=content_type,
            disposition=disposition,
            output_file=output_file,
        )
        target_path.write_bytes(payload)
        emit_result(
            {
                "saved": str(target_path),
                "bytes": len(payload),
                "content_type": content_type or "unknown",
            },
        )

        if preview_rows > 0 and (content_type or "").startswith("text/csv"):
            emit_csv_preview(payload, max_rows=preview_rows)


@main.group("experiment")
def experiment() -> None:
    """Manage experiments (training jobs) by name."""


@experiment.command("submit")
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to manifest file.",
)
@click.option(
    "--resource-id",
    type=str,
    default="arena-medium",
    help="Arena cluster type to submit the experiment to.",
)
@click.option(
    "--num-nodes",
    type=int,
    default=2,
    help="Number of nodes to use for the experiment.",
)
@click.option(
    "--project",
    type=str,
    default=None,
    help="Project to submit the experiment to.",
)
@click.option(
    "--experiment-name",
    type=str,
    default=None,
    help="Name of the experiment.",
)
@click.pass_obj
def experiment_submit(
    config: CommandConfig,
    manifest: Path,
    resource_id: str | None,
    num_nodes: int | None,
    project: str | None,
    experiment_name: str | None,
) -> None:
    """Submit an experiment from a manifest (training job submit API)."""
    with arena_client(config) as client:
        emit_result(
            client.submit_experiment(
                manifest=manifest,
                resource_id=resource_id,
                num_nodes=num_nodes,
                project=project,
                experiment_name=experiment_name,
            )
        )


@experiment.command("list")
@click.option(
    "--project",
    required=True,
    help="Project whose experiments should be listed.",
)
@click.pass_obj
def experiment_list(config: CommandConfig, project: str) -> None:
    """List experiments in a project."""
    with arena_client(config) as client:
        emit_result(client.list_experiments(project=project))


@experiment.command("resume")
@click.argument("experiment_name")
@click.option(
    "--max-steps",
    type=int,
    required=True,
    help="Maximum training steps for the resumed run.",
)
@click.pass_obj
def experiment_resume(
    config: CommandConfig,
    experiment_name: str,
    max_steps: int,
) -> None:
    """Resume an experiment by name."""
    with arena_client(config) as client:
        emit_result(
            client.resume_experiment(
                experiment_name=experiment_name, max_steps=max_steps
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
        emit_result(client.list_checkpoints(experiment_name=experiment_name))


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
    default=10,
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
    """Download metrics for an experiment by name (CSV or zip)."""
    with arena_client(config) as client:
        metrics_list: list[str] | None = list(metrics) if metrics else None
        payload, content_type, disposition = client.download_experiment_metrics(
            experiment_name=experiment_name,
            metrics=metrics_list,
        )
        target_path = resolve_metrics_output_path(
            experiment_name=experiment_name,
            payload=payload,
            content_type=content_type,
            disposition=disposition,
            output_file=output_file,
        )
        target_path.write_bytes(payload)
        emit_result(
            {
                "saved": str(target_path),
                "bytes": len(payload),
                "content_type": content_type or "unknown",
            },
        )

        if preview_rows > 0:
            preview_payload, preview_ct, _ = client.preview_experiment_metrics_csv(
                experiment_name=experiment_name,
                preview_rows=preview_rows,
                metrics=metrics_list,
            )
            if (preview_ct or "").startswith("text/csv"):
                emit_csv_preview(preview_payload, max_rows=preview_rows)
            elif (content_type or "").startswith("text/csv"):
                emit_csv_preview(payload, max_rows=preview_rows)


@experiment.command("deploy")
@click.argument("experiment_name")
@click.option(
    "--checkpoint",
    default=None,
    help="Checkpoint to deploy. Omit to deploy the best checkpoint.",
)
@click.pass_obj
def experiment_deploy(
    config: CommandConfig,
    experiment_name: str,
    checkpoint: str | None,
) -> None:
    """Deploy an agent from an experiment to Arena inference."""
    with arena_client(config) as client:
        client.deploy_agent(experiment_name=experiment_name, checkpoint=checkpoint)
        emit_result(
            {
                "deployed": True,
                "experiment_name": experiment_name,
                "checkpoint": checkpoint,
            }
        )


@main.group("projects")
def projects() -> None:
    """Manage your projects in Arena."""


@projects.command("list")
@click.pass_obj
def projects_list(config: CommandConfig) -> None:
    """List all projects in Arena."""
    with arena_client(config) as client:
        emit_result(client.list_projects())


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
        emit_result(
            client.create_project(
                name=name, description=description, llm_based=llm_based
            )
        )


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
        emit_result(client.delete_project(name=name))
