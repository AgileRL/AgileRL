# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Arena capability check and Snowflake GPU-hour usage posts."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import httpx

from agilerl.arena.auth import EXTERNAL_USER_ID_HEADER
from agilerl.arena.client_version import (
    CLIENT_VERSION_HEADER,
    SNOWFLAKE_UPGRADE_COMMAND,
    print_update_notice,
    upgrade_required_message,
)
from agilerl.arena.models import __version__

CAPABILITIES_PATH = "/api/cli/v1/capabilities"
RUNS_PATH = "/api/cli/v1/snowflake/runs"
CAPABILITIES_SCHEMA_VERSION = 1
CAPABILITIES_TIMEOUT_S = 5.0
USAGE_TIMEOUT_S = 30.0
RETRY_FIRST_S = 2.0
RETRY_MAX_S = 60.0
RETRY_BUDGET_S = 300.0
DEFAULT_BASE_URL = "https://arena.agilerl.com"
MISSING_CREDENTIALS = (
    "Snowflake training needs a signed-in account. "
    "Run arena login, or set ARENA_ORG_KEY and ARENA_EXTERNAL_USER_ID."
)
SNOWFLAKE_DENIED = (
    "Snowflake training is not enabled for this organization. "
    "Ask AgileRL to enable Snowflake training for your organisation."
)
AUTH_REJECTED = "Arena rejected the org key or external user id."
OPEN_FAILED = "Could not record the Snowflake run with Arena."
HEARTBEAT_FAILED = "Could not renew the Snowflake run with Arena."
RENEWAL_REFUSED = "Arena will not renew this Snowflake run."
COMPLETE_FAILED = "Could not record the end of the Snowflake run with Arena."
UPDATE_TRAINER_IMAGE = "then run arena snowflake setup to update the trainer image."

logger = logging.getLogger(__name__)


class SnowflakeLicenseError(Exception):
    """Snowflake training cannot start, or the usage post did not land."""


class RunRenewalRefusedError(SnowflakeLicenseError):
    """Arena will not renew the run: Snowflake training is off, or the run ended."""


def arena_api_hostname(base_url: str) -> str:
    """Hostname of the Arena API, with no port and no wildcard.

    :param base_url: Origin to parse. A bare host is read as ``https``.
    :return: The hostname.
    :raises ValueError: The value does not name one host.
    """
    raw = base_url if "://" in base_url else f"https://{base_url}"
    host = urlsplit(raw).hostname
    if host is None or any(char in host for char in "*?/ \t'\"\\"):
        msg = f"ARENA_BASE_URL must name one host, got {raw!r}."
        raise ValueError(msg)
    return host


def arena_api_base_url(base_url: str) -> str:
    """Origin of the Arena API, ``https`` plus the hostname.

    :param base_url: Origin to parse. A bare host is read as ``https``.
    :return: ``https://<hostname>`` with no path.
    :raises ValueError: The value does not name one host.
    """
    return f"https://{arena_api_hostname(base_url)}"


def snowflake_training_enabled(caps: Mapping[str, object]) -> bool:
    """Whether a capabilities document allows Snowflake training.

    :param caps: The ``data`` object from :func:`fetch_capabilities`.
    :return: True only when ``features.snowflakeTraining`` is true.
    """
    features = caps.get("features")
    if not isinstance(features, dict):
        return False
    return features.get("snowflakeTraining") is True


def fetch_capabilities(base_url: str, headers: Mapping[str, str]) -> dict[str, object]:
    """GET the capabilities document from the user's machine.

    Prints the update notice Arena sends.

    :param base_url: Arena origin.
    :param headers: Authorization headers.
    :return: The ``data`` object.
    :raises SnowflakeLicenseError: This agilerl-arena is too old for Arena,
        Arena rejected the credentials, Arena could not be reached, it returned
        an error status, or the body is not the envelope. The message names
        ``base_url``.
    """
    response = _get_capabilities(base_url, headers)
    print_update_notice(response, SNOWFLAKE_UPGRADE_COMMAND)
    if response.status_code == httpx.codes.UPGRADE_REQUIRED:
        raise SnowflakeLicenseError(upgrade_required_message(response.text))
    return _read_capabilities(response, base_url)


def _get_capabilities(base_url: str, headers: Mapping[str, str]) -> httpx.Response:
    try:
        return _send(
            "GET", base_url, headers, CAPABILITIES_PATH, None, CAPABILITIES_TIMEOUT_S
        )
    except httpx.HTTPError as exc:
        cause = " ".join(str(exc).split()).rstrip(".") or type(exc).__name__
        msg = f"Could not reach Arena at {base_url.rstrip('/')}: {cause}."
        raise SnowflakeLicenseError(msg) from exc


def _read_capabilities(response: httpx.Response, base_url: str) -> dict[str, object]:
    origin = base_url.rstrip("/")
    if response.status_code in {401, 403}:
        raise SnowflakeLicenseError(AUTH_REJECTED)
    if not response.is_success:
        msg = f"Arena at {origin} returned HTTP {response.status_code}."
        raise SnowflakeLicenseError(msg)
    try:
        data = _capabilities_data(response.json())
    except ValueError:
        data = None
    if data is None:
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        msg = (
            f"{origin} answered HTTP {response.status_code} with "
            f"{content_type or 'no content type'}, which is not the Arena API; "
            "the Arena URL must be reachable from the public internet."
        )
        raise SnowflakeLicenseError(msg)
    return data


def check_snowflake_license(env: Mapping[str, str]) -> None:
    """Fail unless Arena allows Snowflake training for the head's credentials.

    The head runs in the trainer image, so it prints no pip update notice.

    :param env: Process environment, including the Snowflake secret mounts.
    :raises SnowflakeLicenseError: Credentials are missing or rejected, the
        trainer image's agilerl-arena is too old for Arena, Arena cannot be
        read, or the flag is not true.
    """
    org_key, external_user_id, _account, base_url = _head_credentials(env)
    response = _get_capabilities(base_url, partner_headers(org_key, external_user_id))
    _raise_for_outdated_image(response)
    if not snowflake_training_enabled(_read_capabilities(response, base_url)):
        raise SnowflakeLicenseError(SNOWFLAKE_DENIED)


def open_head_run(
    job: Mapping[str, object],
    env: Mapping[str, str],
    started_at: datetime,
    sleep: Callable[[float], None],
) -> str:
    """Open the usage row once every node has joined.

    Transport errors, 5xx, and 429 are retried for about five minutes.

    :param job: ``job.json`` fields.
    :param env: Process environment, including the Snowflake secret mounts.
    :param started_at: Earliest GPU worker container start.
    :param sleep: Called between retries.
    :return: The signed run token.
    :raises SnowflakeLicenseError: The trainer image's agilerl-arena is too old
        for Arena, Snowflake training is off for the org, Arena rejected the
        credentials, or the open POST did not land.
    """
    org_key, external_user_id, account, base_url = _head_credentials(env)
    run_id, gpu_count, gpu_family = _run_fields(job)
    payload: dict[str, object] = {
        "runId": run_id,
        "snowflakeAccount": account,
        "gpuCount": gpu_count,
        "gpuFamily": gpu_family,
        "startedAt": started_at.isoformat(),
    }
    response = _post_with_retries(
        base_url,
        partner_headers(org_key, external_user_id),
        RUNS_PATH,
        payload,
        sleep,
        OPEN_FAILED,
    )
    _raise_for_outdated_image(response)
    _raise_for_rejection(response)
    return _run_token(response, OPEN_FAILED)


def heartbeat_snowflake_run(
    run_id: str, env: Mapping[str, str], sleep: Callable[[float], None]
) -> str:
    """Mark the run alive and get a fresh run token.

    :param run_id: Run id the row was opened with.
    :param env: Process environment.
    :param sleep: Called between retries.
    :return: The new signed run token.
    :raises RunRenewalRefusedError: Snowflake training is off for the org, or
        the run has ended.
    :raises SnowflakeLicenseError: Arena rejected the credentials, or the
        heartbeat did not land.
    """
    org_key, external_user_id, _account, base_url = _head_credentials(env)
    response = _post_with_retries(
        base_url,
        partner_headers(org_key, external_user_id),
        f"{RUNS_PATH}/{run_id}/heartbeat",
        {},
        sleep,
        HEARTBEAT_FAILED,
    )
    if response.status_code == 409 or snowflake_training_denied(response):
        msg = f"{RENEWAL_REFUSED} Arena answered {response.status_code}."
        raise RunRenewalRefusedError(msg)
    _raise_for_rejection(response)
    return _run_token(response, HEARTBEAT_FAILED)


def complete_snowflake_run(
    run_id: str,
    env: Mapping[str, str],
    sleep: Callable[[float], None],
    retry_budget_s: float = RETRY_BUDGET_S,
    timeout_s: float = USAGE_TIMEOUT_S,
) -> None:
    """POST the end of a run, stamped now.

    Transport errors, 5xx, and 429 are retried within ``retry_budget_s``.

    :param run_id: Run id the row was opened with.
    :param env: Process environment.
    :param sleep: Called between retries.
    :param retry_budget_s: Seconds of waiting allowed between attempts. ``0``
        makes one attempt.
    :param timeout_s: Timeout of each attempt.
    :raises SnowflakeLicenseError: Arena rejected the credentials, or the
        complete POST did not land.
    """
    org_key, external_user_id, _account, base_url = _head_credentials(env)
    response = _post_with_retries(
        base_url,
        partner_headers(org_key, external_user_id),
        f"{RUNS_PATH}/{run_id}/complete",
        {"endedAt": datetime.now(timezone.utc).isoformat()},
        sleep,
        COMPLETE_FAILED,
        retry_budget_s,
        timeout_s,
    )
    _raise_for_rejection(response)
    if not response.is_success:
        msg = f"{COMPLETE_FAILED} Arena answered {response.status_code}."
        raise SnowflakeLicenseError(msg)


def snowflake_training_denied(response: httpx.Response) -> bool:
    """Whether a response is Arena's 403 for an org without Snowflake training.

    The run handlers send it as JSON ``{"error": "forbidden", ...}``. The auth
    layer's 403, for an org key that may not call the path, is plain text.

    :param response: Arena response.
    :return: True for the flag-off 403.
    """
    if response.status_code != 403:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("error") == "forbidden"


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Delay a ``Retry-After`` header asks for.

    :param response: Arena response.
    :return: Seconds to wait, or ``None`` when the header is absent or unreadable.
    """
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
        return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)
    except (TypeError, ValueError):
        logger.warning("Ignoring unreadable Retry-After header %r", value)
        return None


def _post_with_retries(
    base_url: str,
    headers: Mapping[str, str],
    path: str,
    payload: dict[str, object],
    sleep: Callable[[float], None],
    failure: str,
    retry_budget_s: float = RETRY_BUDGET_S,
    timeout_s: float = USAGE_TIMEOUT_S,
) -> httpx.Response:
    waited = 0.0
    backoff = RETRY_FIRST_S
    while True:
        error: httpx.TransportError | None = None
        try:
            response = _send("POST", base_url, headers, path, payload, timeout_s)
        except httpx.TransportError as exc:
            error = exc
            delay = backoff
        else:
            status = response.status_code
            if status != 429 and status < 500:
                return response
            retry_after = retry_after_seconds(response)
            delay = backoff if retry_after is None else retry_after
        if waited + delay > retry_budget_s:
            raise SnowflakeLicenseError(failure) from error
        sleep(delay)
        waited += delay
        backoff = min(backoff * 2, RETRY_MAX_S)


def _raise_for_outdated_image(response: httpx.Response) -> None:
    if response.status_code == httpx.codes.UPGRADE_REQUIRED:
        message = upgrade_required_message(response.text).rstrip(".")
        msg = f"{message}, {UPDATE_TRAINER_IMAGE}"
        raise SnowflakeLicenseError(msg)


def _raise_for_rejection(response: httpx.Response) -> None:
    if snowflake_training_denied(response):
        raise SnowflakeLicenseError(SNOWFLAKE_DENIED)
    if response.status_code in {401, 403}:
        raise SnowflakeLicenseError(AUTH_REJECTED)


def _run_token(response: httpx.Response, failure: str) -> str:
    if not response.is_success:
        msg = f"{failure} Arena answered {response.status_code}."
        raise SnowflakeLicenseError(msg)
    try:
        body = response.json()
    except ValueError as exc:
        msg = f"{failure} The response is not JSON."
        raise SnowflakeLicenseError(msg) from exc
    data = body.get("data") if isinstance(body, dict) else None
    token = data.get("runToken") if isinstance(data, dict) else None
    if not isinstance(token, str) or token == "":
        msg = f"{failure} The response has no runToken."
        raise SnowflakeLicenseError(msg)
    return token


def _head_credentials(env: Mapping[str, str]) -> tuple[str, str, str, str]:
    org_key = env.get("ARENA_ORG_KEY", "").strip()
    external_user_id = env.get("ARENA_EXTERNAL_USER_ID", "").strip()
    if org_key == "" or external_user_id == "":
        raise SnowflakeLicenseError(MISSING_CREDENTIALS)
    account = env.get("SNOWFLAKE_ACCOUNT", "").strip()
    if account == "":
        msg = "SNOWFLAKE_ACCOUNT is required."
        raise SnowflakeLicenseError(msg)
    base_url = env.get("ARENA_BASE_URL", "").strip() or DEFAULT_BASE_URL
    return org_key, external_user_id, account, base_url.rstrip("/")


def _run_fields(job: Mapping[str, object]) -> tuple[str, int, str]:
    return str(job["run_id"]), int(str(job["gpus"])), str(job["family"])


def partner_headers(org_key: str, external_user_id: str) -> dict[str, str]:
    """Headers that authenticate an Arena org key for one external user.

    :param org_key: Arena organisation key.
    :param external_user_id: External user id the key acts for.
    :return: Request headers.
    """
    return {
        "Authorization": f"Bearer {org_key}",
        EXTERNAL_USER_ID_HEADER: external_user_id,
    }


def _capabilities_data(body: object) -> dict[str, object] | None:
    if not isinstance(body, dict) or body.get("ok") is not True:
        return None
    data = body.get("data")
    if not isinstance(data, dict):
        return None
    schema = data.get("schemaVersion")
    version = CAPABILITIES_SCHEMA_VERSION
    if schema != version and schema != str(version):
        return None
    return data


def _send(
    method: str,
    base_url: str,
    headers: Mapping[str, str],
    path: str,
    payload: dict[str, object] | None,
    timeout_s: float,
) -> httpx.Response:
    url = f"{base_url.rstrip('/')}{path}"
    sent = {**headers, CLIENT_VERSION_HEADER: __version__}
    with httpx.Client() as client:
        if method == "GET":
            return client.get(url, headers=sent, timeout=timeout_s)
        return client.post(url, headers=sent, json=payload, timeout=timeout_s)
