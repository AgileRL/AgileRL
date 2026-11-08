# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""``arena datasets`` commands."""

from __future__ import annotations

from pathlib import Path

import click

from agilerl.arena.client import DATASET_CATEGORIES
from agilerl.arena.config import CommandConfig, arena_client
from agilerl.arena.output import emit_result
from agilerl.arena.utils import sort_dataset_search_by_downloads


@click.group("datasets")
def datasets_group() -> None:
    """Manage your language model datasets in Arena."""


@datasets_group.command("list")
@click.option("--search", default=None, help="Search HuggingFace datasets.")
@click.pass_obj
def datasets_list(config: CommandConfig, search: str | None) -> None:
    """List datasets registered for your organization."""
    with arena_client(config) as client:
        result = client.list_datasets(search=search)
        if search is not None:
            for res in result:
                res.pop("description", None)
            result = sort_dataset_search_by_downloads(result)

        emit_result(result)


@datasets_group.command("exists")
@click.argument("name")
@click.pass_obj
def datasets_exists(config: CommandConfig, name: str) -> None:
    """Check whether a dataset name exists in Arena."""
    with arena_client(config) as client:
        emit_result(client.dataset_exists(name=name))


@datasets_group.command("create")
@click.argument("name")
@click.option(
    "--category",
    required=True,
    type=click.Choice(sorted(DATASET_CATEGORIES), case_sensitive=False),
    help="Dataset category.",
)
@click.option(
    "--column-mapping",
    default=None,
    help="Column mapping as a JSON object string.",
)
@click.option(
    "--column-mapping-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON file containing the column mapping.",
)
@click.option("--description", default=None, help="Optional description.")
@click.option(
    "--file",
    "dataset_file",
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    default=None,
    help="Local CSV or parquet file, or a Hugging Face parquet folder.",
)
@click.option(
    "--config",
    "parquet_config",
    default=None,
    help="Parquet config when --file is a folder with more than one config.",
)
@click.option(
    "--hf-dataset",
    "hf_dataset_name",
    default=None,
    help="HuggingFace dataset id to import.",
)
@click.option(
    "--hf-config",
    default=None,
    help="HuggingFace dataset config (default on the server: default).",
)
@click.option(
    "--hf-split",
    default=None,
    help="HuggingFace split (required when using --hf-dataset).",
)
@click.pass_obj
def datasets_create(
    config: CommandConfig,
    name: str,
    category: str,
    column_mapping: str | None,
    column_mapping_file: Path | None,
    description: str | None,
    dataset_file: Path | None,
    parquet_config: str | None,
    hf_dataset_name: str | None,
    hf_config: str | None,
    hf_split: str | None,
) -> None:
    """Create a dataset on Arena."""
    mapping = (
        column_mapping_file.read_text()
        if column_mapping_file is not None
        else column_mapping
    )
    if mapping is None:
        msg = "Provide --column-mapping or --column-mapping-file."
        raise click.UsageError(msg)
    with arena_client(config) as client:
        client.create_dataset(
            name=name,
            category=category,
            column_mapping=mapping,
            description=description,
            file=dataset_file,
            config=parquet_config,
            hf_dataset_name=hf_dataset_name,
            hf_config=hf_config,
            hf_split=hf_split,
        )


@datasets_group.command("delete")
@click.argument("name")
@click.option(
    "--yes",
    is_flag=True,
    default=False,
    help="Confirm deletion without interactive prompt.",
)
@click.pass_obj
def datasets_delete(
    config: CommandConfig,
    name: str,
    yes: bool,
) -> None:
    """Archive a dataset by name."""
    with arena_client(config) as client:
        client.delete_dataset(name=name, confirm=yes)
