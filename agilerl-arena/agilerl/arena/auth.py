from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import stat
import time
import webbrowser
from pathlib import Path
from typing import Any

from agilerl.arena.exceptions import ArenaAuthError, ArenaTimeoutError
from keycloak import KeycloakOpenID
from keycloak.exceptions import KeycloakError

logger = logging.getLogger(__name__)

_DEFAULT_ACCESS_SKEW_SECONDS = 60


def oauth_access_token_expires_at(access_token: str | None) -> float | None:
    """Return JWT ``exp`` claim as Unix timestamp, or ``None`` if not a decodable JWT.

    :param access_token: The access token to decode.
    :type access_token: str | None
    :returns: The Unix timestamp of the JWT ``exp`` claim, or ``None`` if not a decodable JWT.
    :rtype: float | None
    """
    if not access_token or "." not in access_token:
        return None
    try:
        _h, payload_b64, *_rest = access_token.split(".")
        pad = "=" * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        payload = json.loads(raw.decode("utf-8"))
        exp = payload.get("exp")
        if exp is None:
            return None
        return float(exp)
    except (
        json.JSONDecodeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
        KeyError,
    ):
        return None


def is_oauth_access_token_valid(
    access_token: str | None,
    *,
    skew_seconds: int = _DEFAULT_ACCESS_SKEW_SECONDS,
) -> bool:
    """True if *access_token* is a JWT and ``exp`` is after now (plus *skew_seconds*).

    :param access_token: The access token to check.
    :type access_token: str | None
    :param skew_seconds: The number of seconds to add to the current time to allow for clock skew.
    :type skew_seconds: int
    :returns: True if the access token is valid, False otherwise.
    :rtype: bool
    """
    exp = oauth_access_token_expires_at(access_token)
    if exp is None:
        return False
    return exp > time.time() + skew_seconds


def load_credentials_payload(
    credentials_path: Path | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load the raw JSON object from the credentials file (no token validation).

    Used when merging OAuth tokens with other persisted keys.
    Deployment bindings live in ``~/.arena/inference.json``.

    :param credentials_path: The path to the credentials file.
    :type credentials_path: Path | os.PathLike[str] | None
    :returns: The raw JSON object from the credentials file.
    :rtype: dict[str, Any]
    """
    path = (
        Path(os.fspath(credentials_path)).expanduser().resolve()
        if credentials_path is not None
        else Path.home() / ".arena" / "credentials.json"
    )
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_credentials(
    credentials_path: Path | os.PathLike[str] = "~/.arena/credentials.json",
) -> dict[str, Any] | None:
    """Read stored credentials from ``~/.arena/credentials.json``.

    :param credentials_path: The path to the credentials file.
    :type credentials_path: Path | os.PathLike[str]
    :returns: Token dictionary, or ``None`` if absent or malformed.
    :rtype: dict[str, Any] | None
    """
    credentials_path = Path(os.fspath(credentials_path)).expanduser().resolve()
    if not credentials_path.is_file():
        return None
    try:
        data = json.loads(credentials_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "access_token" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


class ArenaOAuth2:
    """Authentication for the Arena RLOps platform.

    Handles authentication with Keycloak. Supports OAuth 2.0 Device Authorization Grant flow.
    """

    CREDENTIALS_DIR = Path.home() / ".arena"
    CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.json"

    KEYCLOAK_URL = "https://auth.arena.agilerl.com"
    REALM = "arena"
    CLIENT_ID = "arena-cli"

    def __init__(self):
        # Create a Keycloak OpenID client with the configured URL, realm, and client ID.
        self.kc = KeycloakOpenID(
            server_url=os.environ.get("ARENA_KEYCLOAK_URL") or self.KEYCLOAK_URL,
            realm_name=self.REALM,
            client_id=self.CLIENT_ID,
        )

    @classmethod
    def configure(
        cls,
        *,
        keycloak_url: str | None = None,
        realm: str | None = None,
        client_id: str | None = None,
        credentials_dir: Path | None = None,
        credentials_file: Path | None = None,
    ) -> type[ArenaOAuth2]:
        """Configure the ArenaOAuth2 instance.

        :param keycloak_url: The URL of the Keycloak server.
        :param realm: The realm to use for authentication.
        :param client_id: The client ID to use for authentication.
        :param credentials_dir: The directory to store the credentials. Defaults to ``~/.arena``.
        :param credentials_file: The file to store the credentials. Defaults to ``~/.arena/credentials.json``.
        :returns: The configured ArenaOAuth2 instance.
        """
        if keycloak_url is not None:
            cls.KEYCLOAK_URL = keycloak_url
        if realm is not None:
            cls.REALM = realm
        if client_id is not None:
            cls.CLIENT_ID = client_id
        if credentials_dir is not None:
            cls.CREDENTIALS_DIR = credentials_dir
        if credentials_file is not None:
            cls.CREDENTIALS_FILE = credentials_file
        return cls

    @classmethod
    def _write_credentials(cls, data: dict[str, Any]) -> None:
        """Persist credentials with owner-only file permissions.

        :param data: The credentials to persist.
        :returns: The credentials.
        """
        cls.CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(cls.CREDENTIALS_DIR, stat.S_IRWXU)
        cls.CREDENTIALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.chmod(cls.CREDENTIALS_FILE, stat.S_IRUSR | stat.S_IWUSR)

    @staticmethod
    def _extract_error(exc: KeycloakError) -> str:
        """Pull the ``error`` field from a Keycloak error response body."""
        try:
            body = getattr(exc, "response_body", None)
            if isinstance(body, bytes):
                body = body.decode("utf-8", errors="replace")
            if isinstance(body, str):
                return json.loads(body).get("error", "")
        except (json.JSONDecodeError, AttributeError):
            pass

        msg = str(exc)
        for known in ("authorization_pending", "slow_down", "expired_token"):
            if known in msg:
                return known
        return ""

    def device_login(self, timeout: int = 300) -> dict[str, Any]:
        """Run the OAuth 2.0 Device Authorization Grant flow.

        Requests a device code from Keycloak, opens the verification URL in a browser,
        then polls until the user authorizes or *timeout* seconds elapse.

        :param timeout: Maximum seconds to wait for user authorization.
        :returns: Token dict with ``access_token``, ``refresh_token``, etc.
        :raises ArenaAuthError: If Keycloak rejects the request.
        :raises ArenaTimeoutError: If the user does not authorize in time.
        """
        try:
            # Request a device code from Keycloak.
            device_resp = self.kc.device(scope="openid profile email")
        except KeycloakError as exc:
            msg = f"Failed to initiate device authorization: {exc}"
            raise ArenaAuthError(msg) from exc

        # Extract the device code and verification URI from the response.
        device_code = device_resp["device_code"]
        verification_uri = device_resp.get(
            "verification_uri_complete", device_resp.get("verification_uri", "")
        )
        interval = device_resp.get("interval", 5)

        logger.info("Opening browser for authentication... %s", verification_uri)
        if not webbrowser.open(verification_uri):
            logger.warning(
                "Could not open browser automatically. Please visit the URL above."
            )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                tokens = self.kc.token(
                    grant_type="urn:ietf:params:oauth:grant-type:device_code",
                    device_code=device_code,
                )
                merged = load_credentials_payload(self.CREDENTIALS_FILE)
                merged.update(tokens)
                ArenaOAuth2._write_credentials(merged)
                return tokens
            except KeycloakError as exc:
                error = ArenaOAuth2._extract_error(exc)
                if error == "authorization_pending":
                    continue
                if error == "slow_down":
                    interval += 5
                    continue
                if error == "expired_token":
                    msg = "Device code expired before authorization was completed."
                    raise ArenaTimeoutError(msg) from exc
                msg = f"Device authorization failed: {exc}"
                raise ArenaAuthError(msg) from exc

        msg = f"Authentication timed out after {timeout}s. No authorization received."
        raise ArenaTimeoutError(msg)

    def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        """Obtain a fresh access token using a refresh token.

        Persists the updated token set to ``~/.arena/credentials.json``.

        :param refresh_token: The stored refresh token.
        :returns: Updated token dict.
        :raises ArenaAuthError: If the refresh is rejected (session expired).
        """
        try:
            tokens = self.kc.refresh_token(refresh_token)
        except KeycloakError as exc:
            msg = "Token refresh failed — your session may have expired."
            raise ArenaAuthError(
                msg,
                sdk_hint="Please run client.login() again.",
                cli_hint="Please run 'arena login' to re-authenticate.",
            ) from exc

        creds = load_credentials_payload(self.CREDENTIALS_FILE)
        creds.update(tokens)
        ArenaOAuth2._write_credentials(creds)
        return tokens

    def revoke(self, refresh_token: str) -> None:
        """Revoke a Keycloak session and delete stored credentials.

        :param refresh_token: The refresh token to revoke.
        """
        try:
            self.kc.logout(refresh_token)
        except KeycloakError:
            logger.debug("Keycloak logout failed (token may already be expired).")

        try:
            self.CREDENTIALS_FILE.unlink(missing_ok=True)
        except OSError:
            pass
