# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Models, experiments, projects, resources, and metrics."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO

from agilerl.arena.exceptions import ArenaAPIError, ArenaConfigError
from agilerl.arena.typing import JSONValue
from agilerl.arena.utils import (
    extract_filename,
    multipart_text_fields,
    prepare_file_upload,
)

logger = logging.getLogger("agilerl.arena.client")


def _check_writable_target(path: Path) -> None:
    """Ensure *path* can be created as a new file, making parent dirs as needed.

    :param path: The intended destination file.
    :type path: Path
    :raises FileExistsError: If something already exists at *path*.
    :raises NotADirectoryError: If the parent path exists but is not a directory.
    """
    if path.exists():
        kind = "directory" if path.is_dir() else "file"
        msg = (
            f"Cannot write to {path}: a {kind} of that name already exists. "
            f"Remove it, or choose a different output path."
        )
        raise FileExistsError(msg)

    parent = path.parent
    if parent.exists() and not parent.is_dir():
        msg = f"Cannot write to {path}: {parent} is not a directory."
        raise NotADirectoryError(msg)
    parent.mkdir(parents=True, exist_ok=True)


class ExperimentClientMixin:
    """Arena training jobs, projects, and metrics."""

    def list_models(self) -> list[dict[str, Any]]:
        """List HuggingFace models in the Arena catalog.

        :returns: Catalog rows including ``model_name``, ``num_params``,
            ``lora_info``, and ``max_context_length``.
        :rtype: list[dict[str, Any]]
        """
        return self._request("GET", "/api/cli/v1/models")

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Fetch LoRA modules and catalog metadata for a model.

        :param model_name: HuggingFace model id as stored in the catalog.
        :type model_name: str
        :returns: ``modules``, ``parameters``, ``lora_info``, ``num_params``, and
            ``max_context_length``.
        :rtype: dict[str, Any]
        """
        return self._request(
            "POST",
            "/api/cli/v1/models/info",
            json={"model_name": model_name},
        )

    # -------------------------------------------------------------------------
    ### Training Jobs ###
    # -------------------------------------------------------------------------

    def get_default_run_spec(
        self,
        algorithm: str,
        *,
        gym_env: str | None = None,
        gym_env_entrypoint: str | None = None,
        dataset_name: str | None = None,
        field: str | None = None,
        pretrained_model_name_or_path: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a default RunSpec for an algorithm.

        ``field`` returns one section instead of the full spec.

        :param algorithm: Catalog algorithm name.
        :type algorithm: str
        :param gym_env: Gym environment display name.
        :type gym_env: str | None
        :param gym_env_entrypoint: Gym entrypoint.
        :type gym_env_entrypoint: str | None
        :param dataset_name: Dataset name.
        :type dataset_name: str | None
        :param field: RunSpec section to return (``algorithm``, ``environment``,
            ``mutation``, ``network``, ``replay_buffer``, ``tournament_selection``,
            or ``training``).
        :type field: str | None
        :param pretrained_model_name_or_path: HuggingFace model name in the catalog.
        :type pretrained_model_name_or_path: str | None
        :returns: A filled RunSpec, or one section when ``field`` is set.
        :rtype: dict[str, Any]
        """
        params: dict[str, Any] = {"algorithm": algorithm}
        optional = {
            "gymEnv": gym_env,
            "gymEnvEntrypoint": gym_env_entrypoint,
            "datasetName": dataset_name,
            "field": field,
            "pretrainedModelNameOrPath": pretrained_model_name_or_path,
        }
        params.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return self._request(
            "GET",
            "/api/cli/v1/experiments/run-spec/default",
            params=params,
        )

    def submit_experiment(
        self,
        manifest: str | Path | dict[str, Any],
        *,
        resource_id: str | int | None = None,
        num_nodes: int | None = None,
        project: str | None = None,
        experiment_name: str | None = None,
        reward_file: str | os.PathLike[str] | bytes | None = None,
        completion: str | None = None,
    ) -> dict[str, Any]:
        """Submit an experiment (a training job).

        :param manifest: Training manifest as a YAML/JSON file path, raw YAML
            string, or a pre-parsed dict.
        :type manifest: str | Path | dict[str, Any]
        :param resource_id: Arena cluster type or resource id for the job.
        :type resource_id: str | int | None
        :param num_nodes: The number of nodes to use for training.
        :type num_nodes: int | None
        :param project: The project to submit the experiment to.
        :type project: str | None
        :param experiment_name: The name of the experiment to submit.
        :type experiment_name: str | None
        :param reward_file: Python reward module for reasoning dataset jobs.
            When set, the request is sent as ``multipart/form-data``.
        :type reward_file: str | os.PathLike[str] | bytes | None
        :param completion: Optional model completion for reward validation.
            When omitted, the server uses the reference answer from the first
            dataset row.
        :type completion: str | None
        """
        import agilerl.arena.client as client

        validated = client.TrainingManifest.get_validated(manifest, mode="json")
        resolved_project = self._resolve_project(project)

        if reward_file is not None:
            files = self._build_submit_experiment_multipart(
                manifest=validated,
                project=resolved_project,
                resource_id=resource_id,
                num_nodes=num_nodes,
                experiment_name=experiment_name,
                reward_file=reward_file,
                completion=completion,
            )
            try:
                return self._open_stream(
                    "POST",
                    "/api/cli/v1/experiments/jobs/submit",
                    files=files,
                    timeout=self._upload_timeout,
                ).collect()
            finally:
                self._close_upload_files(files)

        payload: dict[str, Any] = {
            "manifest": validated,
            "resource_id": resource_id,
            "num_nodes": num_nodes,
            "project": resolved_project,
            "experiment_name": experiment_name,
        }
        return self._open_stream(
            "POST",
            "/api/cli/v1/experiments/jobs/submit",
            json=payload,
            timeout=self._upload_timeout,
        ).collect()

    @staticmethod
    def _build_submit_experiment_multipart(
        *,
        manifest: dict[str, Any],
        project: str | None,
        resource_id: str | int | None,
        num_nodes: int | None,
        experiment_name: str | None,
        reward_file: str | os.PathLike[str] | bytes,
        completion: str | None,
    ) -> dict[str, tuple[None, str] | tuple[str, BinaryIO | bytes, str]]:
        """Build multipart form parts for reasoning submit with reward validation."""
        text_fields: dict[str, str | None] = {
            "manifest": json.dumps(manifest),
            "project": project,
            "resource_id": str(resource_id) if resource_id is not None else None,
            "num_nodes": str(num_nodes) if num_nodes is not None else None,
            "experiment_name": experiment_name,
            "completion": completion,
        }
        files: dict[str, tuple[None, str] | tuple[str, BinaryIO | bytes, str]] = {
            **multipart_text_fields(text_fields)
        }
        files["reward_file"] = prepare_file_upload(
            reward_file,
            default_name="reward.py",
            content_type="text/x-python",
        )
        return files

    def list_experiments(self, project: str | None = None) -> list[dict[str, Any]]:
        """List all experiments in a project.

        :param project: The name of the project. Falls back to the default
            project from ``~/.arena/config.json`` if not provided.
        :type project: str | None
        :returns: A list of experiments.
        :rtype: list[dict[str, Any]]
        """
        resolved = self._resolve_project(project)
        if not resolved:
            msg = "No project specified."
            raise ArenaConfigError(
                msg,
                sdk_hint="Pass a project name or set a default with ArenaClient.set_default_project().",
                cli_hint="Use --project or set a default with 'arena projects set-default <name>'.",
            )
        return self._request(
            "GET", "/api/cli/v1/experiments/list", params={"project": resolved}
        )

    def resume_experiment(self, experiment_name: str, max_steps: int) -> dict[str, Any]:
        """Resume an experiment (a training job).

        :param experiment_name: The name of the experiment to resume.
        :type experiment_name: str
        :param max_steps: The maximum number of steps to train for.
        :type max_steps: int
        :returns: A dictionary containing the resume result.
        :rtype: dict[str, Any]
        """
        return self._request(
            "POST",
            "/api/cli/v1/experiments/jobs/resume",
            json={"experiment_name": experiment_name, "max_steps": max_steps},
        )

    def list_checkpoints(self, experiment_name: str) -> list[dict[str, Any]]:
        """List all checkpoints for an experiment.

        :param experiment_name: The name of the experiment to list checkpoints for.
        :type experiment_name: str
        :returns: A list of checkpoints.
        :rtype: list[dict[str, Any]]
        """
        return self._request(
            "GET",
            "/api/cli/v1/experiments/jobs/checkpoints",
            params={"experiment_name": experiment_name},
        )

    def preview_experiment_metrics_csv(
        self,
        experiment_name: str,
        *,
        preview_rows: int,
        metrics: Sequence[str] | None = None,
        project: str | None = None,
    ) -> tuple[bytes, str | None, str | None]:
        """Fetch a capped CSV snippet (Arena CLI ``--metric`` / ``--preview-rows``).

        Uses ``GET /api/cli/v1/experiments/metrics`` with ``preview_rows`` set.
        Omit ``metrics`` to include all columns.

        :param experiment_name: Experiment name (latest match in scope).
        :type experiment_name: str
        :param preview_rows: Maximum number of **data** rows in the CSV (server-capped).
        :type preview_rows: int
        :param metrics: Metric column names to include (repeat query param ``metric``).
        :type metrics: Sequence[str] | None
        :param project: Optional exact project name.
        :type project: str | None
        :returns: A tuple of the metrics payload, content type, and disposition.
        :rtype: tuple[bytes, str | None, str | None]
        """
        resolved_project = self._resolve_project(project)
        params: list[tuple[str, Any]] = [
            ("experiment_name", experiment_name),
            ("preview_rows", preview_rows),
        ]
        if resolved_project is not None:
            params.append(("project", resolved_project))
        if metrics:
            params.extend(("metric", m) for m in metrics)
        return self._request_raw(
            "GET",
            "/api/cli/v1/experiments/metrics",
            params=params,
        )

    def list_experiment_metric_names(
        self,
        experiment_name: str,
        *,
        project: str | None = None,
        details: bool = False,
    ) -> list[str] | dict[str, Any]:
        r"""List metric column names recorded for an experiment (JSON).

        For a **CSV preview** with ``--metric`` / ``--preview-rows``-style filters,
        use :meth:`preview_experiment_metrics_csv`.

        :param experiment_name: Experiment name (latest updated match in scope).
        :type experiment_name: str
        :param project: Optional exact project name in the current org.
        :type project: str | None
        :param details: When True, the API returns ``{\"experiment_id\", \"metrics\"}``.
        :type details: bool
        :returns: Sorted unique metric names, or that object when ``details`` is True.
        :rtype: list[str] | dict[str, Any]
        """
        resolved_project = self._resolve_project(project)
        params: dict[str, Any] = {"experiment_name": experiment_name}
        if resolved_project is not None:
            params["project"] = resolved_project
        if details:
            params["details"] = True
        return self._request(
            "GET",
            "/api/cli/v1/experiments/metrics",
            params=params,
        )

    def list_resources(self) -> dict[str, Any]:
        """List compute resource tiers for Arena training jobs.

        Any of the listed resource IDs can be used as the `resource_id` parameter when submitting a training job
        through `ArenaClient.submit_experiment`.

        :returns: A dictionary of resource tiers.
        :rtype: dict[str, Any]
        """
        return self._request("GET", "/api/cli/v1/resources/list")

    def download_experiment_metrics(
        self,
        experiment_name: str,
        output_path: str | os.PathLike[str] | None = None,
        metrics: list[str] | None = None,
    ) -> Path:
        """Download experiment metrics to a local file.

        :param experiment_name: The name of the experiment to download metrics for.
        :type experiment_name: str
        :param output_path: Destination file path or directory. If a directory,
            the filename is inferred from the server response.
            Defaults to ``{experiment_name}_metrics.csv`` in the current directory.
        :type output_path: str | os.PathLike[str] | None
        :param metrics: The metrics to download. If None, download all metrics.
        :type metrics: list[str] | None
        :returns: The path to the written file.
        :rtype: Path
        :raises FileExistsError: If the resolved output path already exists.
        :raises NotADirectoryError: If the parent path exists but is not a directory.
        """
        path = (
            Path(f"{experiment_name}_metrics.csv")
            if output_path is None
            else Path(output_path)
        )
        # A directory target takes its filename from the response's
        # content-disposition, so it can only be checked after the download.
        resolve_after_download = path.is_dir()
        if not resolve_after_download:
            _check_writable_target(path)

        payload, content_type, disposition = self.preview_experiment_metrics_csv(
            experiment_name,
            preview_rows=50_000,
            metrics=metrics,
        )
        if not (content_type or "").startswith("text/csv"):
            body_preview = payload.decode("utf-8", errors="replace")[:500]
            raise ArenaAPIError.from_response_body(body_preview)

        if resolve_after_download:
            filename = extract_filename(disposition) or f"{experiment_name}_metrics.csv"
            path = path / filename
            _check_writable_target(path)

        path.write_bytes(payload)
        logger.info("Metrics saved to %s", path)
        return path

    def stop_experiment(self, experiment_name: str) -> JSONValue:
        """Stop a running experiment in Arena.

        :param experiment_name: Experiment name to halt.
        :type experiment_name: str
        """
        return self._request(
            "POST",
            "/api/cli/v1/experiments/jobs/stop",
            json={"experiment_name": experiment_name.strip()},
        )

    # -------------------------------------------------------------------------
    ### Projects ###
    # -------------------------------------------------------------------------

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects in Arena for the authenticated user.

        :returns: A list of projects.
        :rtype: list[dict[str, Any]]
        """
        resp = self._request("GET", "/api/cli/v1/projects")
        if resp:
            return [
                {"name": p["name"], "type": p["type"], "description": p["description"]}
                for p in resp
            ]
        return []

    def create_project(
        self, name: str, description: str | None, llm_based: bool
    ) -> dict[str, Any]:
        """Create a new project in Arena.

        :param name: The name of the project to create.
        :type name: str
        :param description: The description of the project to create.
        :type description: str | None
        :param llm_based: Whether the project is based on an LLM.
        :type llm_based: bool
        :returns: A dictionary containing the project creation result.
        :rtype: dict[str, Any]
        """
        resp = self._request(
            "POST",
            "/api/cli/v1/projects/create",
            json={"name": name, "description": description, "llm_based": llm_based},
        )
        logger.info("Project %s created successfully.", name)
        return resp

    def delete_project(self, name: str) -> None:
        """Delete a project in Arena.

        :param name: The name of the project to delete.
        :type name: str
        """
        resp = self._request(
            "DELETE", "/api/cli/v1/projects/delete", json={"name": name}
        )
        logger.info("Project %s deleted successfully.", name)
        return resp
