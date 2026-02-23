from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import stat
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from agilerl.arena.exceptions import ArenaAuthError

logger = logging.getLogger(__name__)

_CREDENTIALS_DIR = Path.home() / ".arena"
_CREDENTIALS_FILE = _CREDENTIALS_DIR / "credentials.json"


def _generate_pkce_pair() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for PKCE S256."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _write_credentials(data: dict[str, Any]) -> None:
    """Persist credentials with restricted file permissions (owner-only)."""
    _CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(_CREDENTIALS_DIR, stat.S_IRWXU)

    _CREDENTIALS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(_CREDENTIALS_FILE, stat.S_IRUSR | stat.S_IWUSR)


def load_credentials() -> dict[str, Any] | None:
    """Read stored credentials from ``~/.arena/credentials.json``.

    :return: Token dictionary, or ``None`` if absent / malformed.
    :rtype: dict or None
    """
    if not _CREDENTIALS_FILE.is_file():
        return None
    try:
        data = json.loads(_CREDENTIALS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "access_token" not in data:
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def login(base_url: str, *, port: int = 8400, timeout: int = 300) -> str:
    """Run the browser-based OAuth login flow and return an access token.

    1. Generate a PKCE verifier/challenge and a random *state* token.
    2. Start a temporary HTTP server on ``127.0.0.1:{port}``.
    3. Open the user's browser to the Arena authorization endpoint.
    4. Wait for the callback with an authorization code.
    5. Exchange the code for tokens via the Arena token endpoint.
    6. Persist the tokens to ``~/.arena/credentials.json``.

    :param base_url: Root URL of the Arena platform
        (e.g. ``https://arena.agilerl.com``).
    :type base_url: str
    :param port: Local port for the OAuth callback listener.
    :type port: int
    :param timeout: Seconds to wait for the browser callback before
        raising an error.
    :type timeout: int

    :return: The access token.
    :rtype: str

    :raises ArenaAuthError: If the flow times out, the user denies
        access, the state parameter does not match, or the token
        exchange fails.
    """
    verifier, challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    loopback = "127.0.0.1"
    redirect_uri = f"http://{loopback}:{port}/callback"

    result: dict[str, Any] = {}
    error: list[str] = []

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            params = parse_qs(urlparse(self.path).query)

            if params.get("error"):
                error.append(params["error"][0])
                self._respond(
                    400,
                    "Authentication denied. You may close this tab.",
                )
                return

            received_state = (params.get("state") or [None])[0]
            if received_state != state:
                error.append("state_mismatch")
                self._respond(
                    400,
                    "Security validation failed (state mismatch). "
                    "Please try logging in again.",
                )
                return

            code = (params.get("code") or [None])[0]
            if not code:
                error.append("missing_code")
                self._respond(400, "Missing authorization code.")
                return

            result["code"] = code
            self._respond(
                200,
                "Login successful! You may close this tab and return to your terminal.",
            )

        def _respond(self, status: int, body: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, format, *args):
            pass

    server = HTTPServer((loopback, port), _CallbackHandler)
    server.timeout = timeout

    auth_url = (
        f"{base_url}/auth/authorize"
        f"?response_type=code"
        f"&redirect_uri={redirect_uri}"
        f"&code_challenge={challenge}"
        f"&code_challenge_method=S256"
        f"&state={state}"
    )

    logger.info(
        "Opening browser for Arena login. If it does not open "
        "automatically, visit:\n  %s",
        auth_url,
    )
    webbrowser.open(auth_url)

    shutdown_event = threading.Event()

    def _serve() -> None:
        while not shutdown_event.is_set():
            server.handle_request()
            if result or error:
                break

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    shutdown_event.set()
    server.server_close()

    if error:
        msg = f"Authentication failed: {error[0]}. Please try again."
        raise ArenaAuthError(msg)
    if "code" not in result:
        msg = f"Authentication timed out. No callback received within {timeout}s."
        raise ArenaAuthError(msg)

    tokens = _exchange_code(
        base_url=base_url,
        code=result["code"],
        redirect_uri=redirect_uri,
        code_verifier=verifier,
    )

    _write_credentials(tokens)
    return tokens["access_token"]


def _exchange_code(
    *,
    base_url: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict[str, Any]:
    """Exchange an authorization code for access/refresh tokens."""
    try:
        resp = httpx.post(
            f"{base_url}/auth/token",
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        msg = "Token exchange failed. Please try logging in again."
        raise ArenaAuthError(msg) from exc
    except httpx.HTTPError as exc:
        msg = f"Could not reach Arena auth service at {base_url}."
        raise ArenaAuthError(msg) from exc

    data = resp.json()
    if "access_token" not in data:
        msg = "Unexpected response from token endpoint (no access_token)."
        raise ArenaAuthError(msg)
    return data


def refresh_access_token(base_url: str, refresh_token: str) -> str:
    """Use a refresh token to obtain a new access token.

    Updates ``~/.arena/credentials.json`` on success.

    :param base_url: Root URL of the Arena platform.
    :type base_url: str
    :param refresh_token: The stored refresh token.
    :type refresh_token: str

    :return: The new access token.
    :rtype: str
    """
    try:
        resp = httpx.post(
            f"{base_url}/auth/token",
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            },
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        msg = (
            "Token refresh failed — your session may have expired. "
            "Please run `login()` again."
        )
        raise ArenaAuthError(msg) from exc
    except httpx.HTTPError as exc:
        msg = f"Could not reach Arena auth service at {base_url}."
        raise ArenaAuthError(msg) from exc

    data = resp.json()
    if "access_token" not in data:
        msg = "Unexpected response from refresh endpoint (no access_token)."
        raise ArenaAuthError(msg)

    creds = load_credentials() or {}
    creds.update(data)
    _write_credentials(creds)
    return data["access_token"]


def logout() -> None:
    """Remove stored Arena credentials."""
    try:
        _CREDENTIALS_FILE.unlink(missing_ok=True)
    except OSError:
        pass
