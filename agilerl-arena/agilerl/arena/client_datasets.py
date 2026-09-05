# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Dataset listing, creation, and deletion."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agilerl.arena.exceptions import ArenaValidationError
from agilerl.arena.utils import (
    multipart_text_fields,
    order_dataset_fields,
    prepare_file_upload,
)

logger = logging.getLogger("agilerl.arena.client")

DATASET_CATEGORIES = frozenset({"sft", "preference", "reasoning"})


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
        hf_dataset_name: str | None = None,
        hf_config: str | None = None,
        hf_split: str | None = None,
    ) -> dict[str, Any]:
        """Create an LLM dataset on Arena.

        Upload a local CSV, import from HuggingFace, or create metadata only
        (no file or HF fields). Validation is performed by the Arena API.

        :param name: Dataset name.
        :type name: str
        :param category: Dataset category (e.g. ``reasoning``, ``preference``).
        :type category: str
        :param column_mapping: Column mapping as a JSON string or dict.
        :type column_mapping: str | dict[str, Any]
        :param description: Optional description.
        :type description: str | None
        :param file: Local CSV file path or raw bytes.
        :type file: str | os.PathLike[str] | bytes | None
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
            hf_dataset_name=hf_dataset_name,
            hf_config=hf_config,
            hf_split=hf_split,
        )
        files: dict[str, tuple[None, str] | tuple[str, Any, str]] = {
            **multipart_text_fields(data),
            **upload_files,
        }
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
        hf_dataset_name: str | None = None,
        hf_config: str | None = None,
        hf_split: str | None = None,
    ) -> tuple[dict[str, str | None], dict[str, tuple[str, Any, str]]]:
        """Build multipart form fields for dataset creation."""
        category = DatasetClientMixin._validate_dataset_category(category)
        column_mapping_str = (
            json.dumps(column_mapping)
            if isinstance(column_mapping, dict)
            else column_mapping
        )
        data = {
            "name": name,
            "category": category,
            "column_mapping": column_mapping_str,
            "description": description,
            "hf_dataset_name": hf_dataset_name,
            "hf_config": hf_config,
            "hf_split": hf_split,
        }

        files: dict[str, tuple[str, Any, str]] = {}
        if file is not None:
            files["file"] = prepare_file_upload(
                file,
                default_name="dataset.csv",
                content_type="text/csv",
            )
        return data, files
