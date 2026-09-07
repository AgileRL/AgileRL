# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Dataset listing, creation, and deletion."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from agilerl.arena.exceptions import ArenaValidationError
from agilerl.arena.utils import (
    multipart_text_fields,
    order_dataset_fields,
    prepare_file_upload,
)

logger = logging.getLogger("agilerl.arena.client")

DATASET_CATEGORIES = frozenset({"sft", "preference", "reasoning"})
PARQUET_CONTENT_TYPE = "application/vnd.apache.parquet"
CSV_CONTENT_TYPE = "text/csv"


class DatasetClientMixin:
    """Arena dataset catalog."""

    def list_datasets(
        self,
        *,
        name: str | None = None,
        search: str | None = None,
    ) -> list[dict[str, Any]]:
        """List datasets or search HuggingFace datasets.

        :param name: Filter by registered dataset name.
        :type name: str | None
        :param search: HuggingFace dataset search query.
        :type search: str | None
        :returns: List of datasets or search results from Arena.
        :rtype: list[dict[str, Any]]
        """
        params: dict[str, str] | None = None
        if name is not None or search is not None:
            params = {}
            if name is not None:
                params["name"] = name
            if search is not None:
                params["search"] = search
        result = self._request("GET", "/api/cli/v1/datasets", params=params)
        if not isinstance(result, list):
            return result
        return [
            order_dataset_fields(item) if isinstance(item, dict) else item
            for item in result
        ]

    def dataset_exists(self, name: str) -> dict[str, bool | str]:
        """Check whether a dataset name is registered for the active org.

        :param name: Dataset name.
        :type name: str
        :returns: ``exists``, optional ``id``, and ``datasetType`` when present.
        :rtype: dict[str, bool | str]
        """
        return self._request(
            "GET",
            "/api/cli/v1/datasets/exists",
            params={"name": name},
        )

    def create_dataset(
        self,
        *,
        name: str,
        category: str,
        column_mapping: str | dict[str, Any],
        description: str | None = None,
        file: str | os.PathLike[str] | bytes | None = None,
        config: str | None = None,
        hf_dataset_name: str | None = None,
        hf_config: str | None = None,
        hf_split: str | None = None,
    ) -> dict[str, Any]:
        """Create an LLM dataset on Arena.

        Upload a local CSV, a parquet file, a Hugging Face parquet folder,
        import from HuggingFace, or create metadata only. Validation is
        performed by the Arena API.

        :param name: Dataset name.
        :type name: str
        :param category: Dataset category (``reasoning``, ``preference``,
            or ``sft``).
        :type category: str
        :param column_mapping: Column mapping as a JSON string or dict.
        :type column_mapping: str | dict[str, Any]
        :param description: Optional description.
        :type description: str | None
        :param file: Local CSV or parquet path, a directory of parquet shards,
            or raw CSV bytes (bytes are uploaded as ``dataset.csv``).
        :type file: str | os.PathLike[str] | bytes | None
        :param config: Parquet config name when *file* is a folder with more
            than one config (e.g. gsm8k ``main`` vs ``socratic``).
        :type config: str | None
        :param hf_dataset_name: HuggingFace dataset id for import.
        :type hf_dataset_name: str | None
        :param hf_config: HuggingFace dataset config name.
        :type hf_config: str | None
        :param hf_split: HuggingFace split (required when importing from HF).
        :type hf_split: str | None
        :returns: Created dataset metadata from Arena.
        :rtype: dict[str, Any]
        """
        data, upload_files = self._build_create_dataset_multipart(
            name=name,
            category=category,
            column_mapping=column_mapping,
            description=description,
            file=file,
            config=config,
            hf_dataset_name=hf_dataset_name,
            hf_config=hf_config,
            hf_split=hf_split,
        )
        files: list[tuple[str, tuple[None, str] | tuple[str, Any, str]]] = [
            *multipart_text_fields(data).items(),
            *upload_files,
        ]
        try:
            resp: dict[str, Any] = self._request(
                "POST",
                "/api/cli/v1/datasets/create",
                files=files,
                timeout=self._upload_timeout,
            )
        finally:
            self._close_upload_files(upload_files)
        if resp and resp.get("is_ready", False) and resp.get("uploaded", False):
            # Avoid circular import with agilerl.arena.client barrel
            import agilerl.arena.client as client

            client.logger.info("Dataset %s created successfully.", name)

        return resp

    def delete_dataset(
        self,
        name: str,
        *,
        confirm: bool = False,
    ) -> dict[str, Any] | None:
        """Archive a dataset by name.

        :param name: Dataset name.
        :type name: str
        :param confirm: When ``True``, skip the interactive confirmation prompt.
        :type confirm: bool
        :returns: Archive result, or ``None`` if the user declined confirmation.
        :rtype: dict[str, Any] | None
        """
        if not confirm:
            confirm_prompt = (
                input(
                    f"Delete dataset {name!r}? [y/N]: ",
                )
                .strip()
                .lower()
            )
            if confirm_prompt not in ("y", "yes"):
                logger.info("Dataset %s was not deleted.", name)
                return None

        resp = self._request(
            "DELETE",
            "/api/cli/v1/datasets/delete",
            json={"name": name},
        )
        logger.info("Dataset %s deleted successfully.", name)
        return resp

    @staticmethod
    def _validate_dataset_category(category: str) -> str:
        normalized = category.strip().lower()
        if normalized not in DATASET_CATEGORIES:
            supported = ", ".join(sorted(DATASET_CATEGORIES))
            msg = (
                f"Invalid dataset category {category!r}. "
                f"Supported categories: {supported}"
            )
            raise ArenaValidationError(msg)
        return normalized

    @staticmethod
    def _build_create_dataset_multipart(
        *,
        name: str,
        category: str,
        column_mapping: str | dict[str, Any],
        description: str | None = None,
        file: str | os.PathLike[str] | bytes | None = None,
        config: str | None = None,
        hf_dataset_name: str | None = None,
        hf_config: str | None = None,
        hf_split: str | None = None,
    ) -> tuple[dict[str, str | None], list[tuple[str, tuple[str, Any, str]]]]:
        """Build multipart form fields for dataset creation."""
        category = DatasetClientMixin._validate_dataset_category(category)
        column_mapping_str = (
            json.dumps(column_mapping)
            if isinstance(column_mapping, dict)
            else column_mapping
        )
        data: dict[str, str | None] = {
            "name": name,
            "category": category,
            "column_mapping": column_mapping_str,
            "description": description,
            "hf_dataset_name": hf_dataset_name,
            "hf_config": hf_config,
            "hf_split": hf_split,
        }
        if config is not None:
            data["config"] = config

        return data, DatasetClientMixin._dataset_upload_parts(file, config=config)

    @staticmethod
    def _content_type_for_upload_path(path: Path) -> str:
        if path.suffix.lower() == ".parquet":
            return PARQUET_CONTENT_TYPE
        return CSV_CONTENT_TYPE

    @staticmethod
    def _parquet_shard_paths(root: Path) -> list[Path]:
        shards = sorted(
            child
            for child in root.rglob("*")
            if child.is_file() and child.suffix.lower() == ".parquet"
        )
        if not shards:
            msg = f"No parquet files found in {root}"
            raise ArenaValidationError(msg)
        return shards

    @staticmethod
    def _posix_path_components(relative: str) -> list[str]:
        return [part for part in relative.split("/") if part not in ("", ".")]

    @staticmethod
    def _strip_common_path_prefixes(paths: list[list[str]]) -> list[list[str]]:
        remaining = [list(parts) for parts in paths]
        while remaining and all(len(parts) > 1 for parts in remaining):
            head = remaining[0][0]
            if any(parts[0] != head for parts in remaining):
                break
            remaining = [parts[1:] for parts in remaining]
        return remaining

    @staticmethod
    def _parquet_stripped_relatives(root: Path, shards: list[Path]) -> list[str]:
        components = [
            DatasetClientMixin._posix_path_components(
                shard.relative_to(root).as_posix()
            )
            for shard in shards
        ]
        stripped = DatasetClientMixin._strip_common_path_prefixes(components)
        return ["/".join(parts) for parts in stripped]

    @staticmethod
    def _parquet_config_name(relative: str) -> str:
        parts = DatasetClientMixin._posix_path_components(relative)
        if len(parts) > 1:
            return parts[0]
        return "default"

    @staticmethod
    def _parquet_config_names(root: Path, shards: list[Path]) -> list[str]:
        return sorted(
            {
                DatasetClientMixin._parquet_config_name(relative)
                for relative in DatasetClientMixin._parquet_stripped_relatives(
                    root, shards
                )
            }
        )

    @staticmethod
    def _dataset_upload_parts(
        file: str | os.PathLike[str] | bytes | None,
        *,
        config: str | None,
    ) -> list[tuple[str, tuple[str, Any, str]]]:
        if file is None:
            return []
        if isinstance(file, bytes):
            return [
                (
                    "file",
                    prepare_file_upload(
                        file,
                        default_name="dataset.csv",
                        content_type=CSV_CONTENT_TYPE,
                    ),
                )
            ]

        path = Path(os.fspath(file)).expanduser().resolve()
        if path.is_dir():
            return DatasetClientMixin._parquet_directory_parts(path, config=config)

        content_type = DatasetClientMixin._content_type_for_upload_path(path)
        return [
            (
                "file",
                prepare_file_upload(
                    file,
                    default_name="dataset.csv",
                    content_type=content_type,
                ),
            )
        ]

    @staticmethod
    def _parquet_directory_parts(
        root: Path,
        *,
        config: str | None,
    ) -> list[tuple[str, tuple[str, Any, str]]]:
        shards = DatasetClientMixin._parquet_shard_paths(root)
        stripped = DatasetClientMixin._parquet_stripped_relatives(root, shards)
        configs = sorted(
            {
                DatasetClientMixin._parquet_config_name(relative)
                for relative in stripped
            }
        )
        selected = list(zip(shards, stripped, strict=True))
        if config is None and len(configs) > 1:
            listed = ", ".join(configs)
            msg = (
                f"Parquet folder {root} has multiple configs ({listed}); "
                "pass config= to choose one."
            )
            raise ArenaValidationError(msg)
        if config is not None:
            selected = [
                (shard, relative)
                for shard, relative in selected
                if DatasetClientMixin._parquet_config_name(relative) == config
            ]
            if not selected:
                msg = f"No parquet files for config {config!r} in {root}"
                raise ArenaValidationError(msg)

        parts: list[tuple[str, tuple[str, Any, str]]] = []
        for shard, relative in selected:
            parts.append(
                (
                    "file",
                    prepare_file_upload(
                        shard,
                        default_name=relative,
                        content_type=PARQUET_CONTENT_TYPE,
                        filename=relative,
                    ),
                )
            )
        return parts
