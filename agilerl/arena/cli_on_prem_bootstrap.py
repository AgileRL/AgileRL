"""Composite ``arena on-prem install bootstrap`` (enable → create class → download bundle)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from agilerl.arena.cli_manifest import write_binary_atomic
from agilerl.arena.config import CommandConfig
from agilerl.arena.exceptions import ArenaAPIError

_ENABLE_INVOKE: dict[str, Any] = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/enable",
    "responseKind": "json",
    "params": [],
}

_CREATE_INVOKE: dict[str, Any] = {
    "method": "POST",
    "path": "/api/cli/v1/on-prem/classes/create",
    "responseKind": "json",
    "params": [],
}

_BUNDLE_INVOKE: dict[str, Any] = {
    "method": "GET",
    "path": "/api/cli/v1/on-prem/classes/deployment-setup",
    "responseKind": "binary",
    "params": [],
}


def _class_id_from_create_response(data: Any) -> int:
    if isinstance(data, dict):
        raw = data.get("id")
        if raw is not None:
            return int(raw)
    msg = "Create class response did not include resource class id."
    raise ArenaAPIError(msg, cli_hint="Check platform logs; retry with manual classes create.")


def _build_metadata(*, cpus: int, gpus: int, memory: str) -> dict[str, Any]:
    return {
        "computeResource": {
            "numCpus": cpus,
            "numGpus": gpus,
            "memoryBytes": memory,
        },
    }


def run_install_bootstrap(
    client: Any,
    *,
    name: str,
    num_nodes: int,
    output: Path,
    setup_type: str,
    archived_type: str,
    cpus: int,
    gpus: int,
    memory: str,
    description: str | None,
    enabled: bool,
    force: bool,
) -> None:
    """Run enable → create → bundle download using an authenticated client."""
    click.echo("Step 1/3: enabling on-prem provider…")
    client.invoke_manifest_command(_ENABLE_INVOKE, {})

    body: dict[str, Any] = {
        "name": name,
        "num_nodes": num_nodes,
        "enabled": enabled,
        "metadata": _build_metadata(cpus=cpus, gpus=gpus, memory=memory),
    }
    if description is not None:
        body["description"] = description

    click.echo(f"Step 2/3: creating resource class {name!r}…")
    created = client.invoke_manifest_command(_CREATE_INVOKE, body)
    class_id = _class_id_from_create_response(created)

    click.echo(f"Step 3/3: downloading install bundle (class id={class_id})…")
    raw_b, _ctype, _disp = client.invoke_manifest_command(
        _BUNDLE_INVOKE,
        {
            "id": class_id,
            "setupType": setup_type,
            "archivedType": archived_type,
        },
    )
    write_binary_atomic(output.expanduser().resolve(), raw_b, force=force)
    click.echo(f"Wrote {output} (class id={class_id})")


def build_install_bootstrap_command() -> click.Command:
    """Build the static ``install bootstrap`` subcommand."""

    @click.command("bootstrap")
    @click.option("--name", required=True, help="Resource class name (worker pool).")
    @click.option(
        "--num-nodes",
        type=click.IntRange(1),
        required=True,
        help="Worker node count for this class.",
    )
    @click.option(
        "-o",
        "--output",
        "output",
        type=click.Path(dir_okay=False, path_type=Path),
        required=True,
        help="Destination path for the install archive.",
    )
    @click.option(
        "--setup-type",
        "setup_type",
        default="helm",
        show_default=True,
        help="Stack flavor: helm, dockerSwarm, or kubernetes.",
    )
    @click.option(
        "--archived-type",
        "archived_type",
        default="zip",
        show_default=True,
        help="Archive format: zip or tar.",
    )
    @click.option("--cpus", type=click.IntRange(0), default=8, show_default=True)
    @click.option("--gpus", type=click.IntRange(0), default=0, show_default=True)
    @click.option(
        "--memory",
        default="64 GB",
        show_default=True,
        help="Per-node memory (ByteSize string, e.g. 64 GB).",
    )
    @click.option("--description", default=None, help="Optional class description.")
    @click.option(
        "--enabled/--disabled",
        "enabled",
        default=True,
        show_default=True,
        help="Whether the new class is enabled.",
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Overwrite an existing output file.",
    )
    @click.pass_obj
    def bootstrap_cmd(
        config: CommandConfig,
        name: str,
        num_nodes: int,
        output: Path,
        setup_type: str,
        archived_type: str,
        cpus: int,
        gpus: int,
        memory: str,
        description: str | None,
        enabled: bool,
        force: bool,
    ) -> None:
        """Enable on-prem, create a class, and download its worker install bundle."""
        from agilerl.arena.cli import arena_client

        with arena_client(config) as client:
            run_install_bootstrap(
                client,
                name=name,
                num_nodes=num_nodes,
                output=output,
                setup_type=setup_type,
                archived_type=archived_type,
                cpus=cpus,
                gpus=gpus,
                memory=memory,
                description=description,
                enabled=enabled,
                force=force,
            )

    return bootstrap_cmd


def register_install_bootstrap(install_group: click.Group) -> None:
    """Attach ``bootstrap`` under the manifest ``install`` group."""
    if "bootstrap" not in install_group.commands:
        install_group.add_command(build_install_bootstrap_command())
