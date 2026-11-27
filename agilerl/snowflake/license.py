# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Arena capability check, Snowflake GPU-hour usage posts, and Native App org key claims."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlsplit

import httpx

from agilerl.auth import EXTERNAL_USER_ID_HEADER
from agilerl.client_version import (
    CLIENT_VERSION_HEADER,
    SNOWFLAKE_UPGRADE_COMMAND,
    is_upgrade_required,
    print_update_notice,
    upgrade_required_message,
)
from agilerl.models import __version__

CAPABILITIES_PATH = "/api/cli/v1/capabilities"
RUNS_PATH = "/api/cli/v1/snowflake/runs"
DEPLOYMENTS_PATH = "/api/cli/v1/snowflake/deployments"
CLAIMS_PATH = "/api/cli/v1/snowflake/claims"
APP_KEY_ROTATE_PATH = "/api/cli/v1/snowflake/app-keys/rotate"
USAGE_PATHS = {
    "run": f"{RUNS_PATH}/{{usage_id}}/usage",
    "deployment": f"{DEPLOYMENTS_PATH}/{{usage_id}}/usage",
}
CAPABILITIES_SCHEMA_VERSION = 1
CAPABILITIES_TIMEOUT_S = 5.0
USAGE_TIMEOUT_S = 30.0
RETRY_FIRST_S = 2.0
RETRY_MAX_S = 60.0
RETRY_BUDGET_S = 300.0
# An admin waits on each claim call, so a rate limit is shown rather than waited out.
CLAIM_RETRY_BUDGET_S = 10.0
DEFAULT_BASE_URL = "https://arena.agilerl.com"
# Set only in the head spec of a Native App run; Arena bills such runs through Snowflake.
CHANNEL_ENV = "ARENA_SNOWFLAKE_CHANNEL"
NATIVE_APP_CHANNEL = "native_app"
# Set with CHANNEL_ENV; Arena binds Native App keys to <region>.<locator>.
REGION_ENV = "ARENA_SNOWFLAKE_REGION"
# "true" or "false": the app's listing_purchased(), so Arena sees a purchase before its daily sync.
PURCHASED_ENV = "ARENA_SNOWFLAKE_PURCHASED"
MISSING_CREDENTIALS = (
    "Snowflake training needs a signed-in account. "
    "Run arena login, or set ARENA_ORG_KEY and ARENA_EXTERNAL_USER_ID."
)
SNOWFLAKE_DENIED = (
    "Snowflake training is not enabled for this organization. "
    "Ask AgileRL to enable Snowflake training for your organisation."
)
AUTH_REJECTED = "Arena rejected the org key or external user id."
TRIAL_ENDED = (
    "The free trial of AgileRL Arena has ended. Purchase the AgileRL Arena "
    "listing in Snowflake Marketplace to train and serve again."
)
OPEN_FAILED = "Could not record the Snowflake run with Arena."
HEARTBEAT_FAILED = "Could not renew the Snowflake run with Arena."
RENEWAL_REFUSED = "Arena will not renew this Snowflake run."
COMPLETE_FAILED = "Could not record the end of the Snowflake run with Arena."
USAGE_FAILED = "Could not read Snowflake usage from Arena."
DEPLOYMENT_OPEN_FAILED = "Could not record the Snowflake deployment with Arena."
DEPLOYMENT_HEARTBEAT_FAILED = "Could not renew the Snowflake deployment with Arena."
DEPLOYMENT_CLOSE_FAILED = (
    "Could not record the end of the Snowflake deployment with Arena."
)
CLAIM_START_FAILED = "Could not start connecting this Snowflake account to Arena."
CLAIM_PROVE_FAILED = "Could not prove this Snowflake install to Arena."
CLAIM_KEY_FAILED = "Could not collect the Arena org key."
ROTATE_FAILED = "Could not rotate the Arena org key."
UPDATE_TRAINER_IMAGE = "then upgrade the Arena app: ALTER APPLICATION <app> UPGRADE."

logger = logging.getLogger(__name__)


class SnowflakeLicenseError(Exception):
    """Snowflake training cannot start, or the usage post did not land."""


class SnowflakeUpgradeRequiredError(SnowflakeLicenseError):
    """Arena refuses this agilerl release as below its minimum."""


class AuthRejectedError(SnowflakeLicenseError):
    """Arena answered 401 or 403 to the org key or external user id."""


class RunRenewalRefusedError(SnowflakeLicenseError):
    """Arena will not renew the run: Snowflake training is off, or the run ended."""


class UnknownUsageError(SnowflakeLicenseError):
    """Arena has no such run or deployment for this organisation."""


class TrainingDisabledError(SnowflakeLicenseError):
    """Arena refused a deployment heartbeat: Snowflake training is off, or the trial ended."""


class DeploymentClosedError(SnowflakeLicenseError):
    """Arena has closed the deployment and stops billing it."""


class DeploymentRefusedError(SnowflakeLicenseError):
    """Arena answered 400, 403, 404, 409, or 422 to a deployment open."""


class ClaimWaitingError(SnowflakeLicenseError):
    """Arena answered 409 or 429: not visible yet, not linked yet, or too many proven claims."""


class ClaimLinkedError(ClaimWaitingError):
    """Arena answered 409 ``claim_linked`` to a prove: the key can be collected now."""


class ClaimClosedError(SnowflakeLicenseError):
    """Arena answered 403, 404, or 410: the claim was refused, is unknown, expired, or its key was collected."""


@dataclass(frozen=True)
class SnowflakeUsage:
    """Billable hours Arena reports for one run or deployment."""

    state: str
    billable_hours: list[object]


@dataclass(frozen=True)
class SnowflakeDeployment:
    """Fields Arena records when a served model opens.

    :param deployment_id: Letters, digits, ``.``, ``_``, ``:`` and ``-``.
    :param snowflake_account: Snowflake account locator.
    :param snowflake_region: Snowflake region, like ``AWS_EU_WEST_2``.
    :param gpu_count: GPUs the service requests.
    :param gpu_family: Snowflake instance family of the serving pool.
    :param started_at: When the service started, timezone-aware.
    :param model: Model Registry model name.
    :param version: Model Registry version name.
    :param run_id: Run the model was published from, or ``None``.
    :param purchased: The app's ``listing_purchased()``.
    """

    deployment_id: str
    snowflake_account: str
    snowflake_region: str
    gpu_count: int
    gpu_family: str
    started_at: datetime
    model: str
    version: str
    run_id: str | None
    purchased: bool


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


def snowflake_priced_families(caps: Mapping[str, object]) -> frozenset[str] | None:
    """Snowflake GPU families Arena has a price for, from ``features.snowflakePricedGpuFamilies``.

    :param caps: The ``data`` object from :func:`fetch_capabilities`.
    :return: Upper-case family names, or ``None`` when Arena does not list them.
    """
    features = caps.get("features")
    if not isinstance(features, dict):
        return None
    families = features.get("snowflakePricedGpuFamilies")
    if not isinstance(families, list):
        return None
    return frozenset(str(family).upper() for family in families)


def fetch_capabilities(base_url: str, headers: Mapping[str, str]) -> dict[str, object]:
    """GET the capabilities document from the user's machine.

    Prints the update notice Arena sends.

    :param base_url: Arena origin.
    :param headers: Authorization headers.
    :return: The ``data`` object.
    :raises SnowflakeLicenseError: This agilerl is too old for Arena,
        Arena rejected the credentials, Arena could not be reached, it returned
        an error status, or the body is not the envelope. The message names
        ``base_url``.
    """
    response = _get_capabilities(base_url, headers)
    print_update_notice(response, SNOWFLAKE_UPGRADE_COMMAND)
    if response.status_code == httpx.codes.UPGRADE_REQUIRED:
        message = upgrade_required_message(response.text)
        if is_upgrade_required(response.status_code, response.text):
            raise SnowflakeUpgradeRequiredError(message)
        raise SnowflakeLicenseError(message)
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
        raise AuthRejectedError(AUTH_REJECTED)
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
        trainer image's agilerl is too old for Arena, Arena cannot be
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
        :data:`CHANNEL_ENV` adds ``channel``, :data:`REGION_ENV` adds
        ``snowflakeRegion``, and :data:`PURCHASED_ENV` adds ``purchased``.
    :param started_at: Earliest GPU worker container start.
    :param sleep: Called between retries.
    :return: The signed run token.
    :raises SnowflakeLicenseError: The trainer image's agilerl is too old
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
    for name, field in ((CHANNEL_ENV, "channel"), (REGION_ENV, "snowflakeRegion")):
        value = env.get(name, "").strip()
        if value:
            payload[field] = value
    purchased = env.get(PURCHASED_ENV, "").strip()
    if purchased in {"true", "false"}:
        payload["purchased"] = purchased == "true"
    response = _send_with_retries(
        "POST",
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
    response = _send_with_retries(
        "POST",
        base_url,
        partner_headers(org_key, external_user_id),
        f"{RUNS_PATH}/{run_id}/heartbeat",
        {},
        sleep,
        HEARTBEAT_FAILED,
    )
    if snowflake_trial_ended(response):
        msg = f"{RENEWAL_REFUSED} {TRIAL_ENDED}"
        raise RunRenewalRefusedError(msg)
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
    response = _send_with_retries(
        "POST",
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


def snowflake_usage(
    kind: str,
    usage_id: str,
    base_url: str,
    headers: Mapping[str, str],
    sleep: Callable[[float], None],
    retry_budget_s: float = RETRY_BUDGET_S,
) -> SnowflakeUsage:
    """GET the billable hours of a run or deployment.

    Transport errors, 5xx, and 429 are retried within ``retry_budget_s``.

    :param kind: ``run`` or ``deployment``.
    :param usage_id: Run or deployment id.
    :param base_url: Arena origin.
    :param headers: Authorization headers from :func:`partner_headers`.
    :param sleep: Called between retries.
    :param retry_budget_s: Seconds of waiting allowed between attempts.
    :return: ``open`` or ``closed``, and every closed hour so far. A closed
        run or deployment includes its partial last hour.
    :raises ValueError: The kind is not ``run`` or ``deployment``.
    :raises UnknownUsageError: Arena answered 404.
    :raises SnowflakeLicenseError: Arena rejected the credentials, the GET did
        not land, or the body is not a usage document.
    """
    if kind not in USAGE_PATHS:
        msg = f"Unknown usage kind {kind}. Use run or deployment."
        raise ValueError(msg)
    path = USAGE_PATHS[kind].format(usage_id=quote(usage_id, safe=""))
    response = _send_with_retries(
        "GET", base_url, headers, path, None, sleep, USAGE_FAILED, retry_budget_s
    )
    if response.status_code == httpx.codes.NOT_FOUND:
        msg = f"Arena has no {kind} {usage_id} for this organisation."
        raise UnknownUsageError(msg)
    _raise_for_rejection(response)
    if not response.is_success:
        msg = f"{USAGE_FAILED} Arena answered {response.status_code}."
        raise SnowflakeLicenseError(msg)
    try:
        body = response.json()
    except ValueError as exc:
        msg = f"{USAGE_FAILED} The response is not JSON."
        raise SnowflakeLicenseError(msg) from exc
    data = body.get("data") if isinstance(body, dict) else None
    state = data.get("state") if isinstance(data, dict) else None
    hours = data.get("billableHours") if isinstance(data, dict) else None
    if state not in {"open", "closed"} or not isinstance(hours, list):
        msg = f"{USAGE_FAILED} The response has no state and billableHours."
        raise SnowflakeLicenseError(msg)
    return SnowflakeUsage(str(state), hours)


def open_snowflake_deployment(
    deployment: SnowflakeDeployment,
    base_url: str,
    headers: Mapping[str, str],
    sleep: Callable[[float], None],
) -> None:
    """POST the start of a served model as a Native App deployment.

    Transport errors, 5xx, and 429 are retried for about five minutes. Opening
    the same deployment again with the same fields succeeds.

    :param deployment: What Arena records.
    :param base_url: Arena origin.
    :param headers: Authorization headers from :func:`partner_headers`.
    :param sleep: Called between retries.
    :raises DeploymentRefusedError: Arena answered 400, 403, 404, 409, or
        422, with Arena's description: Snowflake training is off, the key may
        not call the path, this Arena has no deployment route, or Arena refused
        the deployment.
    :raises SnowflakeLicenseError: Arena rejected the credentials, or the
        POST did not land.
    """
    payload: dict[str, object] = {
        "deploymentId": deployment.deployment_id,
        "snowflakeAccount": deployment.snowflake_account,
        "snowflakeRegion": deployment.snowflake_region,
        "gpuCount": deployment.gpu_count,
        "gpuFamily": deployment.gpu_family,
        "startedAt": deployment.started_at.astimezone(timezone.utc).isoformat(),
        "channel": NATIVE_APP_CHANNEL,
        "model": deployment.model,
        "version": deployment.version,
        "purchased": deployment.purchased,
    }
    if deployment.run_id is not None:
        payload["runId"] = deployment.run_id
    response = _send_with_retries(
        "POST",
        base_url,
        headers,
        DEPLOYMENTS_PATH,
        payload,
        sleep,
        DEPLOYMENT_OPEN_FAILED,
    )
    if snowflake_training_denied(response):
        raise DeploymentRefusedError(SNOWFLAKE_DENIED)
    if snowflake_trial_ended(response):
        raise DeploymentRefusedError(TRIAL_ENDED)
    if response.status_code == httpx.codes.FORBIDDEN:
        raise DeploymentRefusedError(AUTH_REJECTED)
    if response.status_code in {400, 404, 409, 422}:
        raise DeploymentRefusedError(_answered(DEPLOYMENT_OPEN_FAILED, response))
    _raise_for_rejection(response)
    if not response.is_success:
        raise SnowflakeLicenseError(_answered(DEPLOYMENT_OPEN_FAILED, response))


def heartbeat_snowflake_deployment(
    deployment_id: str,
    observed_at: datetime,
    base_url: str,
    headers: Mapping[str, str],
    sleep: Callable[[float], None],
    retry_budget_s: float = RETRY_BUDGET_S,
) -> None:
    """POST that the deployment's service was seen running.

    Transport errors, 5xx, and 429 are retried within ``retry_budget_s``.

    :param deployment_id: Deployment id the row was opened with.
    :param observed_at: When the service was seen, timezone-aware.
    :param base_url: Arena origin.
    :param headers: Authorization headers from :func:`partner_headers`.
    :param sleep: Called between retries.
    :param retry_budget_s: Seconds of waiting allowed between attempts.
    :raises TrainingDisabledError: Snowflake training is off for the org.
    :raises DeploymentClosedError: Arena has closed the deployment.
    :raises UnknownUsageError: Arena has no such deployment.
    :raises SnowflakeLicenseError: Arena rejected the credentials, or the POST
        did not land.
    """
    response = _send_with_retries(
        "POST",
        base_url,
        headers,
        f"{DEPLOYMENTS_PATH}/{quote(deployment_id, safe='')}/heartbeat",
        {"observedAt": observed_at.astimezone(timezone.utc).isoformat()},
        sleep,
        DEPLOYMENT_HEARTBEAT_FAILED,
        retry_budget_s,
    )
    if snowflake_training_denied(response):
        raise TrainingDisabledError(SNOWFLAKE_DENIED)
    if snowflake_trial_ended(response):
        raise TrainingDisabledError(TRIAL_ENDED)
    if response.status_code == httpx.codes.CONFLICT:
        msg = f"Arena has closed deployment {deployment_id}."
        raise DeploymentClosedError(msg)
    if response.status_code == httpx.codes.NOT_FOUND:
        msg = f"Arena has no deployment {deployment_id} for this organisation."
        raise UnknownUsageError(msg)
    _raise_for_rejection(response)
    if not response.is_success:
        raise SnowflakeLicenseError(_answered(DEPLOYMENT_HEARTBEAT_FAILED, response))


def close_snowflake_deployment(
    deployment_id: str,
    ended_at: datetime,
    base_url: str,
    headers: Mapping[str, str],
    sleep: Callable[[float], None],
    retry_budget_s: float = RETRY_BUDGET_S,
) -> None:
    """POST the end of a deployment. Closing it again succeeds.

    Transport errors, 5xx, and 429 are retried within ``retry_budget_s``.

    :param deployment_id: Deployment id the row was opened with.
    :param ended_at: When the service stopped, timezone-aware.
    :param base_url: Arena origin.
    :param headers: Authorization headers from :func:`partner_headers`.
    :param sleep: Called between retries.
    :param retry_budget_s: Seconds of waiting allowed between attempts.
    :raises UnknownUsageError: Arena has no such deployment.
    :raises SnowflakeLicenseError: Arena rejected the credentials, or the POST
        did not land.
    """
    response = _send_with_retries(
        "POST",
        base_url,
        headers,
        f"{DEPLOYMENTS_PATH}/{quote(deployment_id, safe='')}/close",
        {"endedAt": ended_at.astimezone(timezone.utc).isoformat()},
        sleep,
        DEPLOYMENT_CLOSE_FAILED,
        retry_budget_s,
    )
    if response.status_code == httpx.codes.NOT_FOUND:
        msg = f"Arena has no deployment {deployment_id} for this organisation."
        raise UnknownUsageError(msg)
    _raise_for_rejection(response)
    if not response.is_success:
        raise SnowflakeLicenseError(_answered(DEPLOYMENT_CLOSE_FAILED, response))


@dataclass(frozen=True)
class ClaimAccount:
    """What a Native App install tells Arena about itself: in a claim, and on run opens.

    :param region: ``CURRENT_REGION()``, like ``AWS_EU_WEST_2``.
    :param account: ``CURRENT_ACCOUNT()``, the account locator.
    :param account_name: ``CURRENT_ACCOUNT_NAME()``.
    :param organization_name: ``CURRENT_ORGANIZATION_NAME()``.
    :param application_name: ``CURRENT_DATABASE()``, the install's name.
    :param purchased: The app's secure ``listing_purchased()``.
    """

    region: str
    account: str
    account_name: str
    organization_name: str
    application_name: str
    purchased: bool


@dataclass(frozen=True)
class SnowflakeClaim:
    """A claim Arena started. Prove, link, and key collection all end at ``expires_at``."""

    claim_id: str
    claim_secret: str
    expires_at: datetime


@dataclass(frozen=True)
class AppKey:
    """An org key Arena issued to the app.

    :param org_key: The key.
    :param external_user_id: External user id the key acts for.
    :param organization_name: Arena organisation, or ``None`` when Arena does not say.
    :param trial_ends_at: End of the Arena trial, or ``None`` once purchased.
    """

    org_key: str
    external_user_id: str
    organization_name: str | None
    trial_ends_at: str | None


def start_snowflake_claim(
    account: ClaimAccount, base_url: str, sleep: Callable[[float], None]
) -> SnowflakeClaim:
    """POST a new claim; Arena writes its code to the package's ``claims.challenges``.

    :param account: The install's identity.
    :param base_url: Arena origin.
    :param sleep: Called between retries.
    :return: The claim.
    :raises ClaimWaitingError: Arena cannot see the install yet; nothing was written.
    :raises SnowflakeLicenseError: Arena refused the claim (the trial ended, or
        the install lost access), or the POST did not land.
    """
    response = _send_with_retries(
        "POST",
        base_url,
        {},
        CLAIMS_PATH,
        {
            "snowflakeRegion": account.region,
            "snowflakeAccount": account.account,
            "accountName": account.account_name,
            "organizationName": account.organization_name,
            "applicationName": account.application_name,
            "purchased": account.purchased,
            "appVersion": __version__,
        },
        sleep,
        CLAIM_START_FAILED,
        CLAIM_RETRY_BUDGET_S,
    )
    if response.status_code == httpx.codes.CONFLICT:
        raise ClaimWaitingError(_answered(CLAIM_START_FAILED, response))
    if not response.is_success:
        raise SnowflakeLicenseError(_answered(CLAIM_START_FAILED, response))
    data = _data(response, CLAIM_START_FAILED)
    claim_id, claim_secret, expires_at = (
        _text_field(data, name, CLAIM_START_FAILED)
        for name in ("claimId", "claimSecret", "expiresAt")
    )
    try:
        expires = datetime.fromisoformat(expires_at)
    except ValueError as exc:
        msg = f"{CLAIM_START_FAILED} The response has no RFC 3339 expiresAt."
        raise SnowflakeLicenseError(msg) from exc
    return SnowflakeClaim(claim_id, claim_secret, expires)


def prove_snowflake_claim(
    claim_id: str,
    claim_secret: str,
    code: str,
    base_url: str,
    sleep: Callable[[float], None],
) -> str:
    """POST the code the app read from ``claims.challenges``.

    :param claim_id: Claim id.
    :param claim_secret: Claim secret from :func:`start_snowflake_claim`.
    :param code: The claim's code.
    :param base_url: Arena origin.
    :param sleep: Called between retries.
    :return: The signup link for the admin.
    :raises ClaimLinkedError: The claim is already linked; collect its key.
    :raises ClaimWaitingError: Arena cannot see the install yet, or the account
        already holds three proven claims.
    :raises ClaimClosedError: Arena refused the proof or the claim expired.
    :raises SnowflakeLicenseError: The POST did not land.
    """
    data = _claim_call(
        claim_id,
        "prove",
        {"claimSecret": claim_secret, "code": code},
        base_url,
        sleep,
        CLAIM_PROVE_FAILED,
    )
    return _text_field(data, "signupUrl", CLAIM_PROVE_FAILED)


def collect_snowflake_claim_key(
    claim_id: str, claim_secret: str, base_url: str, sleep: Callable[[float], None]
) -> AppKey:
    """POST for the org key of a linked claim. Arena issues it once.

    :param claim_id: Claim id.
    :param claim_secret: Claim secret from :func:`start_snowflake_claim`.
    :param base_url: Arena origin.
    :param sleep: Called between retries.
    :return: The key.
    :raises ClaimWaitingError: Nobody has linked the claim yet.
    :raises ClaimClosedError: Arena refused the claim, it expired, or its key
        was already collected.
    :raises SnowflakeLicenseError: The POST did not land.
    """
    data = _claim_call(
        claim_id,
        "key",
        {"claimSecret": claim_secret},
        base_url,
        sleep,
        CLAIM_KEY_FAILED,
    )
    return _app_key(data, CLAIM_KEY_FAILED)


def rotate_snowflake_app_key(
    base_url: str, headers: Mapping[str, str], sleep: Callable[[float], None]
) -> AppKey:
    """POST for a new app key. Arena retires the old one 5 minutes after the new one is first used.

    :param base_url: Arena origin.
    :param headers: Partner headers of the current key.
    :param sleep: Called between retries.
    :return: The new key.
    :raises AuthRejectedError: Arena does not accept the current key for rotation.
    :raises SnowflakeLicenseError: The POST did not land.
    """
    response = _send_with_retries(
        "POST",
        base_url,
        headers,
        APP_KEY_ROTATE_PATH,
        {},
        sleep,
        ROTATE_FAILED,
        CLAIM_RETRY_BUDGET_S,
    )
    _raise_for_rejection(response)
    if not response.is_success:
        raise SnowflakeLicenseError(_answered(ROTATE_FAILED, response))
    return _app_key(_data(response, ROTATE_FAILED), ROTATE_FAILED)


def _claim_call(
    claim_id: str,
    step: str,
    payload: dict[str, object],
    base_url: str,
    sleep: Callable[[float], None],
    failure: str,
) -> dict[str, object]:
    response = _send_with_retries(
        "POST",
        base_url,
        {},
        f"{CLAIMS_PATH}/{quote(claim_id, safe='')}/{step}",
        payload,
        sleep,
        failure,
        CLAIM_RETRY_BUDGET_S,
        final_errors=frozenset({"too_many_claims"}),
    )
    if response.status_code in {409, 429}:
        # 429 here is too_many_claims: the account's other proven claims must
        # be linked or expire first, so this claim waits.
        if _error_code(response) == "claim_linked":
            raise ClaimLinkedError(_answered(failure, response))
        raise ClaimWaitingError(_answered(failure, response))
    if response.status_code in {403, 404, 410}:
        raise ClaimClosedError(_answered(failure, response))
    if not response.is_success:
        raise SnowflakeLicenseError(_answered(failure, response))
    return _data(response, failure)


def _app_key(data: dict[str, object], failure: str) -> AppKey:
    org_key = _text_field(data, "orgKey", failure)
    external_user_id = _text_field(data, "externalUserId", failure)
    organization = data.get("organizationName")
    trial_ends = data.get("trialEndsAt")
    return AppKey(
        org_key,
        external_user_id,
        organization if isinstance(organization, str) else None,
        trial_ends if isinstance(trial_ends, str) else None,
    )


def _data(response: httpx.Response, failure: str) -> dict[str, object]:
    try:
        body = response.json()
    except ValueError as exc:
        msg = f"{failure} The response is not JSON."
        raise SnowflakeLicenseError(msg) from exc
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        msg = f"{failure} The response has no data."
        raise SnowflakeLicenseError(msg)
    return data


def _text_field(data: dict[str, object], name: str, failure: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or value == "":
        msg = f"{failure} The response has no {name}."
        raise SnowflakeLicenseError(msg)
    return value


def snowflake_training_denied(response: httpx.Response) -> bool:
    """Whether a response is Arena's 403 for an org without Snowflake training.

    The run handlers send it as JSON ``{"error": "forbidden", ...}``. The auth
    layer's 403, for an org key that may not call the path, is plain text.

    :param response: Arena response.
    :return: True for the flag-off 403.
    """
    return response.status_code == 403 and _error_code(response) == "forbidden"


def snowflake_trial_ended(response: httpx.Response) -> bool:
    """Whether a response is Arena's 403 for an account whose trial ended unpurchased.

    :param response: Arena response.
    :return: True for 403 ``snowflake_trial_ended``.
    """
    return (
        response.status_code == 403 and _error_code(response) == "snowflake_trial_ended"
    )


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


def _send_with_retries(
    method: str,
    base_url: str,
    headers: Mapping[str, str],
    path: str,
    payload: dict[str, object] | None,
    sleep: Callable[[float], None],
    failure: str,
    retry_budget_s: float = RETRY_BUDGET_S,
    timeout_s: float = USAGE_TIMEOUT_S,
    final_errors: frozenset[str] = frozenset(),
) -> httpx.Response:
    waited = 0.0
    backoff = RETRY_FIRST_S
    while True:
        error: httpx.TransportError | None = None
        try:
            response = _send(method, base_url, headers, path, payload, timeout_s)
        except httpx.TransportError as exc:
            error = exc
            delay = backoff
        else:
            status = response.status_code
            if status != 429 and status < 500:
                return response
            if _error_code(response) in final_errors:
                return response
            retry_after = retry_after_seconds(response)
            # A short Retry-After must not undercut the backoff, or a rate-limited
            # caller retries in a tight loop and keeps tripping the limit.
            delay = (
                backoff
                if retry_after is None
                else min(max(retry_after, backoff), RETRY_MAX_S)
            )
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
    if snowflake_trial_ended(response):
        raise SnowflakeLicenseError(TRIAL_ENDED)
    if response.status_code in {401, 403}:
        raise AuthRejectedError(AUTH_REJECTED)


def _error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    code = body.get("error") if isinstance(body, dict) else None
    return code if isinstance(code, str) else None


def _answered(failure: str, response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = None
    description = body.get("description") if isinstance(body, dict) else None
    if isinstance(description, str) and description.strip():
        return f"{failure} Arena answered {response.status_code}: {description.strip()}"
    return f"{failure} Arena answered {response.status_code}."


def _run_token(response: httpx.Response, failure: str) -> str:
    if not response.is_success:
        raise SnowflakeLicenseError(_answered(failure, response))
    return _text_field(_data(response, failure), "runToken", failure)


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
