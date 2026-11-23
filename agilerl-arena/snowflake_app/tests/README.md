# Arena Native App install test

`install.sql` installs the newest version of `arena_app_pkg` in the provider
account, upgrades it, checks it, and uninstalls it. The upgrade needs two patches
of that version, so run `just snowflake-app-build` twice for a new version. Then
run:

```bash
just snowflake-app-install-test CONNECTION ROLE
```

Use a role other than `ACCOUNTADMIN`, since a consumer installs with one. It needs:

```sql
GRANT CREATE APPLICATION ON ACCOUNT TO ROLE <role>;
GRANT CREATE DATABASE ON ACCOUNT TO ROLE <role>;
GRANT INSTALL, DEVELOP ON APPLICATION PACKAGE arena_app_pkg TO ROLE <role>;
```

The connection's warehouse runs the statements. The script:

1. Creates `arena_app_s6` from the second-newest patch of the newest version.
   This runs the setup script, including the external access integrations and
   the trial-account (509009) block.
2. Checks the three manifest privileges were granted automatically, then
   approves the `hf_access` and `arena_access` specifications.
3. Upgrades to the newest patch, so setup reruns while the app already owns
   both integrations. Checks the upgrade completed, both `AGILERL_APP_ARENA_APP_S6_*`
   integrations exist, and both specifications are still approved.
4. Grants `app_admin` to the role, creates `arena_s6_scratch.public.sft_rows`, and
   binds it to `source_table` through `app_code.register_reference`.
5. Calls `app_code.set_pools(['GPU_NV_S'], 1)`. This is the first import from
   `agilerl_arena.zip`. It checks that all three `AGILERL_APP_ARENA_APP_S6_*` pools exist and
   are suspended.
6. Expects `app_code.status` to refuse `nope` as a malformed run id, and a
   well-formed id with no `job.json` as a run the app does not have.
7. Drops the app and the scratch database, then checks that no `AGILERL_APP_ARENA_APP_S6_%` pool,
   warehouse or integration is left.

A failed check raises. The recipe then drops `arena_app_s6` and
`arena_s6_scratch`. The steps that need an Arena org key (configure, a 1-GPU
SFT submit, status, metrics, publish with serve, endpoint, unpublish) are
commented out near the end of `install.sql`.
