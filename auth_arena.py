from agilerl.arena.client import ArenaClient
import json
import os
from pathlib import Path
from uuid import uuid4


def _run_test(name: str, fn):
    try:
        result = fn()
        print(f"[PASS] {name}")
        return True, result
    except Exception as exc:  # pragma: no cover - smoke script helper
        print(f"[FAIL] {name}: {exc}")
        return False, None


def _print_stream_chunk(chunk: str) -> None:
    print(chunk, end="", flush=True)


if __name__ == "__main__":
    client = ArenaClient.configure(
        base_url="http://localhost:3001",
        keycloak_url="http://localhost:8023",
    )()

    client.login()

    print(f"\nAuthenticated: {client.is_authenticated}")

    # checks = [
    #     ("current user", client.get_current_user),
    #     ("user credits", client.get_user_credits),
    # ]

    # passed = 0
    # total = len(checks)
    # for check_name, check_fn in checks:
    #     ok, payload = _run_test(check_name, check_fn)
    #     passed += int(ok)
    #     if ok:
    #         print(payload)

    # env_list_ok, envs = _run_test(
    #     "custom env list (for follow-up checks)",
    #     client.list_custom_environments,
    # )
    # total += 1
    # passed += int(env_list_ok)

    # if env_list_ok and envs:
    #     first_env = envs[0]
    #     env_name = first_env.get("name")
    #     env_version = first_env.get("version", "latest")

    #     if env_name:
    #         exists_ok, exists_payload = _run_test(
    #             f"custom env exists ({env_name}:{env_version})",
    #             lambda: client.custom_environment_exists(env_name, str(env_version)),
    #         )
    #         total += 1
    #         passed += int(exists_ok)
    #         if exists_ok:
    #             print(exists_payload)

    #         entry_ok, entry_payload = _run_test(
    #             f"custom env entrypoints ({env_name}:{env_version})",
    #             lambda: client.list_custom_environment_entrypoints(
    #                 env_name, str(env_version)
    #             ),
    #         )
    #         total += 1
    #         passed += int(entry_ok)
    #         if entry_ok:
    #             print(entry_payload)
    # else:
    #     print(
    #         "[SKIP] No custom environments found; skipping exists/entrypoints checks."
    #     )

    # print(f"\nSmoke test result: {passed}/{total} checks passed.")

    ######
    # Create and validate
    ######

    platform_python = Path("~/dev/agilerl-platform/python").expanduser()
    create_validate_payload = {
        "name": str(uuid4()),
        "version": "ident",
        "file_path": platform_python / "acrobot.tar.gz",
        "env_config_path": platform_python / "env_config.yaml",
        "requirements_path": platform_python / "requirements.txt",
        "multi_agent": False,
        "do_rollouts": True,
        "entrypoint": "acrobot:AcrobotEnv",
    }
    create_validate_ok, create_validate_payload_response = _run_test(
        "create and validate custom env (multipart upload)",
        lambda: client.create_and_validate_custom_environment(
            **create_validate_payload,
            stream=True,
            on_chunk=_print_stream_chunk,
        ),
    )
    print()
    if create_validate_ok:
        print(create_validate_payload_response)

    ######
    # Submit job / experiment
    ######

    run_spec_raw = os.environ.get("ARENA_DQN_RUN_SPEC")
    custom_env_impl_id_raw = os.environ.get("ARENA_CUSTOM_GYM_ENV_IMPL_ID")
    if run_spec_raw and custom_env_impl_id_raw:
        try:
            submit_manifest = json.loads(run_spec_raw)
            custom_env_impl_id = int(custom_env_impl_id_raw)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[FAIL] experiment submit setup: {exc}")
        else:
            submit_ok, submit_response = _run_test(
                "submit experiment job (streaming)",
                lambda: client.submit_experiment_job(
                    custom_gym_env_impl_id=custom_env_impl_id,
                    manifest=submit_manifest,
                    stream=True,
                    on_chunk=_print_stream_chunk,
                ),
            )
            print()
            if submit_ok:
                print(submit_response)
    else:
        print(
            "[SKIP] Set ARENA_DQN_RUN_SPEC and ARENA_CUSTOM_GYM_ENV_IMPL_ID "
            "to run streaming experiment submit."
        )
