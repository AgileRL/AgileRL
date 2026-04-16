from __future__ import annotations

import json
from typing import Any


class ArenaError(Exception):
    """Base exception for all Arena client errors."""


class ArenaAuthError(ArenaError):
    """Raised when authentication fails or credentials are missing/expired."""


class ArenaTimeoutError(ArenaAuthError):
    """Raised when the device-flow login times out waiting for user authorization."""


_MESSAGE_KEYS = ("description", "message", "detail", "error")
_SKIP_KEYS = frozenset((*_MESSAGE_KEYS, "error_code", "status", "status_code"))


class ArenaAPIError(ArenaError):
    """Raised when the Arena API returns a non-2xx response.

    :param detail: Human-readable error description.
    :param status_code: HTTP status code from the API response.
    :param extras: Supplementary context from the error body.
    """

    _label = "API error"

    def __init__(
        self,
        detail: str,
        *,
        status_code: int = 0,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail
        self.status_code = status_code
        self.extras = extras or {}
        super().__init__(self._format())

    def _format(self) -> str:
        parts = [self.detail]
        for key, value in self.extras.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(value, list):
                parts.append(f"{label}: {', '.join(str(v) for v in value)}")
            else:
                parts.append(f"{label}: {value}")
        msg = "\n".join(parts)
        if self.status_code:
            return f"{self._label} ({self.status_code}): {msg}"
        return f"{self._label}: {msg}"

    @classmethod
    def from_response_body(
        cls,
        raw: str,
        *,
        status_code: int = 0,
    ) -> ArenaAPIError:
        """Build an instance from a raw HTTP response body string.

        Handles plain text, JSON, broken JSON (embedded newlines), nested
        JSON-in-string envelopes, and NDJSON error lines.
        """
        if not raw:
            return cls("No error details", status_code=status_code)

        body = _parse_body(raw)
        if body is None:
            return cls(raw[:500], status_code=status_code)

        primary = ""
        for key in _MESSAGE_KEYS:
            value = body.get(key)
            if value and isinstance(value, str):
                primary = value
                break

        extras = {k: v for k, v in body.items() if k not in _SKIP_KEYS and v}
        return cls(
            detail=primary or raw[:500],
            status_code=status_code,
            extras=extras,
        )


class ArenaValidationError(ArenaAPIError):
    """Raised when server-side environment validation fails."""

    _label = "ValidationError"


def _parse_body(raw: str) -> dict[str, Any] | None:
    """Best-effort parse of an API error body into a dict.

    Handles:
    * Valid JSON
    * JSON with embedded newlines in string values (invalid but common)
    * NDJSON bodies (picks the line containing ``"error"``)
    * Nested JSON-in-string envelopes (one level)
    """
    body: dict[str, Any] | None = None

    for attempt in (raw, raw.replace("\n", " ")):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                body = parsed
                break
        except (json.JSONDecodeError, ValueError):
            continue

    if body is None:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict) and "error" in parsed:
                body = parsed
                break

    if body is None:
        return None

    # Unpack one level of nested JSON-in-string envelopes.
    for key in _MESSAGE_KEYS:
        value = body.get(key)
        if not value or not isinstance(value, str):
            continue
        try:
            inner = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(inner, dict) and any(k in inner for k in _MESSAGE_KEYS):
            return inner

    return body
