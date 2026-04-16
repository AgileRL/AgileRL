from __future__ import annotations

from dataclasses import dataclass

from agilerl.arena.client import ArenaClient


@dataclass(slots=True)
class CommandConfig:
    api_key: str | None
    base_url: str | None
    keycloak_url: str | None
    realm: str | None
    client_id: str | None
    request_timeout: int
    upload_timeout: int


def build_client(config: CommandConfig) -> ArenaClient:
    ArenaClient.configure(
        base_url=config.base_url,
        keycloak_url=config.keycloak_url,
        realm=config.realm,
        client_id=config.client_id,
    )
    return ArenaClient(
        api_key=config.api_key,
        request_timeout=config.request_timeout,
        upload_timeout=config.upload_timeout,
    )
