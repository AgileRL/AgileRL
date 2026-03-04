from agilerl.arena.client import ArenaClient

if __name__ == "__main__":
    client = ArenaClient.configure(
        base_url="http://localhost:3001",
        keycloak_url="http://localhost:8023",
    )()

    client.login()

    print(f"\nAuthenticated: {client.is_authenticated}")

    current_user = client._request("GET", "/api/users/current")  # noqa: SLF001
    print(current_user)
