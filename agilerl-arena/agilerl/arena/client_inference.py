# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Inference deployments and Agent bindings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from agilerl.arena.exceptions import (
    ArenaAPIError,
    ArenaInferenceError,
    ArenaValidationError,
)
from agilerl.arena.inference.cache import normalized_deployment_name

if TYPE_CHECKING:
    from agilerl.arena.inference import Agent

logger = logging.getLogger("agilerl.arena.client")

MemoryScope = Literal["user", "organization"]
MEMORY_SCOPES: tuple[MemoryScope, ...] = ("user", "organization")


class InferenceClientMixin:
    """Arena inference deployments."""

    @staticmethod
    def _deployment_lookup_params(
        *,
        name: str | None = None,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if name is not None and name.strip():
            params["name"] = name.strip()
        en = experiment_name.strip() if experiment_name else ""
        if en:
            params["experimentName"] = en
        pn = project_name.strip() if project_name else ""
        if pn:
            params["projectName"] = pn
        return params

    @staticmethod
    def _validated_memory_scope(memory_scope: str) -> str:
        """Reject an unknown memory scope here rather than at the API."""
        scope = memory_scope.strip().lower()
        if scope not in MEMORY_SCOPES:
            msg = (
                f"Unknown memory scope {memory_scope!r}. "
                f"Choose one of: {', '.join(MEMORY_SCOPES)}."
            )
            raise ArenaValidationError(msg)
        return scope

    def deploy_agent(
        self,
        experiment_name: str,
        checkpoint: str | None = None,
        memory_scope: MemoryScope | None = None,
    ) -> dict[str, Any]:
        """Create an inference deployment from an experiment checkpoint.

        :param experiment_name: The name of the experiment to deploy.
        :type experiment_name: str
        :param checkpoint: The checkpoint to deploy. If None, deploy the best checkpoint.
        :type checkpoint: str | None
        :param memory_scope: Who an LLM deployment keeps chat sessions for. ``"user"``
            gives every caller their own conversations, ``"organization"`` shares them
            across the organisation. A new deployment defaults to ``"user"``. Redeploying
            with ``None`` keeps the scope already stored, it does not reset it, and the
            scope cannot be changed after the deployment exists.
        :type memory_scope: MemoryScope | None
        :returns: A dictionary containing the deployment result.
        :rtype: dict[str, Any]
        """
        body: dict[str, Any] = {
            "experiment_name": experiment_name,
            "checkpoint": checkpoint,
        }
        if memory_scope is not None:
            body["memoryScope"] = self._validated_memory_scope(memory_scope)
        result = self._request(
            "POST",
            "/api/cli/v1/inference/deploy",
            json=body,
        )
        checkpoint_suffix = (
            f" (checkpoint {checkpoint})" if checkpoint else " (best checkpoint)"
        )
        logger.info(
            "Agent submitted for deployment from experiment %s%s. "
            "Check its status in Arena.",
            experiment_name,
            checkpoint_suffix,
        )
        return result

    def list_inference_deployments(
        self,
        *,
        name: str | None = None,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List inference deployments available to the user.

        :param name: The name of the deployment to list.
        :type name: str | None
        :param experiment_name: The name of the experiment to list deployments for.
        :type experiment_name: str | None
        :param project_name: The name of the project to list deployments for.
        :type project_name: str | None
        :returns: A list of deployments.
        :rtype: list[dict[str, Any]]
        """
        q = self._deployment_lookup_params(
            name=name,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        rows = self._request(
            "GET",
            "/api/cli/v1/inference/deployments/list",
            params=q or None,
        )
        if not isinstance(rows, list):
            return []
        return [r for r in rows if isinstance(r, dict)]

    def _inference_credential(self) -> str | None:
        """Return :meth:`_credential`, refreshing an expiring OAuth token first."""
        self._proactively_refresh_oauth()
        return self._credential()

    def open_inference_agent(
        self,
        deployment_name: str,
        *,
        refresh: bool = False,
        experiment_name: str | None = None,
        project_name: str | None = None,
        timeout: int | None = None,
    ) -> Agent:
        """Build an :class:`~arena.inference.Agent` for a named deployment.

        Attempts to load the deployment from the cache, and if not found, fetches it from the API.

        The agent carries this client's own credential, so run :meth:`login` or set
        ``ARENA_API_KEY`` before using a deployment that keeps memory per user.

        :param deployment_name: The name of the deployment to open.
        :type deployment_name: str
        :param refresh: Whether to refresh the deployment metadata.
        :type refresh: bool
        :param experiment_name: Experiment name to disambiguate when multiple deployments
            share the same deployment name.
        :type experiment_name: str | None
        :param project_name: Project name to disambiguate when multiple deployments
            share the same deployment name.
        :type project_name: str | None
        A redeploy moves the deployment to a new URL, which leaves the cached one
        answering 404. Rather than make you pass *refresh* to find that out, a
        404 from the cached URL refetches it once and retries.

        :param timeout: HTTP timeout in seconds for the returned agent's inference requests.
        :type timeout: int | None
        :returns: An :class:`~arena.inference.Agent` instance.
        :rtype: Agent
        """
        url = self._ensure_inference_binding(
            deployment_name,
            refresh=refresh,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        try:
            return self._open_agent_at(url, timeout)
        except ArenaInferenceError as exc:
            if refresh or exc.status_code != 404:
                raise
            fresh_url = self._ensure_inference_binding(
                deployment_name,
                refresh=True,
                experiment_name=experiment_name,
                project_name=project_name,
            )
            if fresh_url == url:
                raise
            logger.info("Deployment %s moved to %s.", deployment_name, fresh_url)
            return self._open_agent_at(fresh_url, timeout)

    def _open_agent_at(self, url: str, timeout: int | None) -> Agent:
        """Build an agent for *url*, probing it to confirm the URL still serves."""
        import agilerl.arena.client as client

        return client.Agent(
            url,
            api_key=self._inference_credential(),
            timeout=timeout or self._request_timeout,
        )

    @staticmethod
    def _deployment_url(row: dict[str, Any]) -> str:
        """Parse ``url`` from an API deployment row."""
        url = row.get("url")
        if not isinstance(url, str) or not url.strip():
            msg = "Deployment has no inference URL."
            raise ArenaAPIError(
                msg,
                cli_hint="Wait until provisioning completes, then retry with --refresh.",
            )
        return url.strip()

    def _fetch_deployment_for_inference(
        self,
        deployment_name: str,
        *,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> dict[str, Any]:
        """Fetch one deployment detail row for inference binding."""
        params = self._deployment_lookup_params(
            name=normalized_deployment_name(deployment_name),
            experiment_name=experiment_name,
            project_name=project_name,
        )
        if "name" not in params:
            msg = "deployment name is required."
            raise ArenaAPIError(msg)

        try:
            row = self._request(
                "GET",
                "/api/cli/v1/inference/deployments/one",
                params=params,
            )
        except ArenaAPIError as exc:
            hint = (
                "Pass --experiment-name and/or --project-name when multiple deployments "
                "share this deployment name."
            )
            if not exc.cli_hint:
                raise ArenaAPIError(
                    exc.detail, cli_hint=hint, status_code=exc.status_code
                ) from exc
            raise

        if not isinstance(row, dict):
            msg = "Unexpected deployment detail response shape."
            raise ArenaAPIError(msg)
        return row

    def _ensure_inference_binding(
        self,
        deployment_name: str,
        *,
        refresh: bool = False,
        experiment_name: str | None = None,
        project_name: str | None = None,
    ) -> str:
        """Return the cached deployment URL, or fetch it from the API and persist it."""
        import agilerl.arena.client as client

        key = normalized_deployment_name(deployment_name)

        if not refresh:
            cached = client.load_binding(key)
            if cached is not None:
                return cached

        row = self._fetch_deployment_for_inference(
            deployment_name,
            experiment_name=experiment_name,
            project_name=project_name,
        )
        url = self._deployment_url(row)
        client.save_binding(key, url)
        return url
