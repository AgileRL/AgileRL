from __future__ import annotations


class ArenaError(Exception):
    """Base exception for all Arena client errors."""


class ArenaAuthError(ArenaError):
    """Raised when authentication fails or credentials are missing/expired."""


class ArenaAPIError(ArenaError):
    """Raised when the Arena API returns a non-2xx response.

    :param status_code: HTTP status code from the API response.
    :type status_code: int
    :param detail: Human-readable error description from the server.
    :type detail: str
    """

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Arena API error {status_code}: {detail}")


class ArenaTimeoutError(ArenaAuthError):
    """Raised when the device-flow login times out waiting for user authorization."""


class ArenaValidationError(ArenaError):
    """Raised when server-side environment validation fails.

    :param errors: Structured validation errors returned by the server.
    :type errors: list[dict]
    """

    def __init__(self, errors: list[dict]) -> None:
        self.errors = errors
        summary = "; ".join(e.get("msg", str(e)) for e in errors[:5])
        if len(errors) > 5:
            summary += f" ... and {len(errors) - 5} more"
        super().__init__(f"Environment validation failed: {summary}")
