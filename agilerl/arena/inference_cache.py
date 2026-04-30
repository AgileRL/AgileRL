"""Persist deployment inference endpoints (URL + API key) beside OAuth credentials."""

from __future__ import annotations

from agilerl.arena.auth import ArenaOAuth2, load_credentials_payload

DEPLOYMENT_INFERENCE_KEY = "deployment_inference"


def normalized_deployment_name(name: str) -> str:
    """CLI/cache key for a deployment name."""
    return name.strip()


def load_inference_binding(name: str) -> tuple[str, str] | None:
    """Return ``(url, api_key)`` for *name* if cached, else ``None``."""
    data = load_credentials_payload(ArenaOAuth2.CREDENTIALS_FILE)
    raw = data.get(DEPLOYMENT_INFERENCE_KEY)
    if not isinstance(raw, dict):
        return None
    entry = raw.get(normalized_deployment_name(name))
    if not isinstance(entry, dict):
        return None
    url = entry.get("url")
    api_key = entry.get("api_key")
    if not isinstance(url, str) or not isinstance(api_key, str):
        return None
    if not url.strip() or not api_key.strip():
        return None
    return url.strip(), api_key.strip()


def save_inference_binding(name: str, url: str, api_key: str) -> None:
    """Merge ``deployment_inference[name]`` into ``~/.arena/credentials.json``."""
    data = load_credentials_payload(ArenaOAuth2.CREDENTIALS_FILE)
    di = data.get(DEPLOYMENT_INFERENCE_KEY)
    if not isinstance(di, dict):
        di = {}
    key = normalized_deployment_name(name)
    di[key] = {"url": url.strip(), "api_key": api_key.strip()}
    data[DEPLOYMENT_INFERENCE_KEY] = di
    ArenaOAuth2._write_credentials(data)
