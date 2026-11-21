# Arena Snowflake Native App

Application package files for the Arena Snowflake Native App. The
`arena snowflake setup` CLI (`agilerl/arena/snowflake/`) stays; this app is the
Marketplace path to the same training, publish and serve behaviour. These files
are not part of the `agilerl-arena` wheel or sdist.

Check the manifest without a Snowflake account. Run this from the repository root:

```bash
just snowflake-app-validate
```

## Build

```bash
just snowflake-app-build CONNECTION TAG   # TAG like 38.0.4-py313-cu130
```

`CONNECTION` is a Snowflake CLI connection to the provider account. The
recipe runs the Snowflake CLI with `uvx --from snowflake-cli` and needs
[`crane`](https://github.com/google/go-containerregistry/tree/main/cmd/crane) on
`PATH`; no Docker. It:

1. Copies this directory to `dist/snowflake-app/` and replaces the
   `0.0.0-py313-cu130` placeholder in `manifest.yml` and `app_code/*.py` with `TAG`.
2. Builds the `agilerl-arena` wheel (stamped with the bundle version) and, with
   `bundle.py`, stores it as `agilerl_arena.zip` together with `python-keycloak`
   at its `uv.lock` version, the one dependency the Snowflake Anaconda channel
   lacks. Directory entries are added so `zipimport` finds the `agilerl`
   namespace package. The procedures import `agilerl.arena.snowflake` from it
   instead of copying code.
3. Creates `arena_provider.images.repo` if missing and runs `crane copy` from
   `public.ecr.aws/x7s7d0o8/containers/snowflake-job:TAG` to that repository.
   The registry password is the whole JSON output of
   `snow spcs image-registry token --format JSON`.
4. Runs `snow app version create v<TAG with _ for punctuation>` for the
   `arena_app_pkg` package in `snowflake.yml`.

`snow app bundle --project dist/snowflake-app` checks steps 1 and 2
without a connection.

## Install

The consumer installs from the listing with `ACCOUNTADMIN`, or with a role that
holds `IMPORT SHARE`, `CREATE APPLICATION` and, because the listing is paid,
`PURCHASE DATA EXCHANGE LISTING` on the account. Without the last one the
install is refused.

Some installs print `Reference register_callback 'APP_CODE.REGISTER_REFERENCE(...)'
does not exist or is not granted to an application role` as a second result
row. It is harmless: the setup script creates and grants the procedure, and
binding `source_table` or `source_view` works. `SHOW REFERENCES IN APPLICATION <app>` lists the bindings.

## Connect to Arena

After installing, an `app_admin` runs:

```sql
CALL <app>.app_code.connect_arena();
```

`CONNECT` is a reserved word in Snowflake SQL, so the procedure is `connect_arena`.

It returns a link. Open it, sign up or sign in, and create the Arena
organisation for this Snowflake account. Then run `connect_arena()` again, or wait up
to an hour: the app collects its org key by itself. Nobody sees or types the
key; it goes straight into the app's secrets.

How the app proves it is this install:

1. `connect_arena()` sends Arena the account's region (`CURRENT_REGION()`), locator
   (`CURRENT_ACCOUNT()`), account and organisation names, the app's name, and
   whether the listing is purchased. Arena answers with a claim id and a claim
   secret, which the app saves in `app_state.arena_claim_secret`.
2. Arena writes a one-time code for that region and locator to
   `claims.challenges`, a table shared from the application package. Consumers
   cannot query shared package content, and the app reads only the row for its
   own region and locator, so only this install sees the code. In another
   region the row arrives after a listing refresh Arena starts, so
   `connect_arena()` waits up to two minutes for it.
3. The app sends the code back, and Arena returns the signup link.
4. Once the admin has created the organisation, `connect_arena()` or the hourly task
   collects the key into the three Arena secrets, then checks it with
   `GET /api/cli/v1/capabilities`. Arena issues the key once, so a failed check
   leaves the key saved.

A second `connect_arena()` while the first is still starting does not open
another claim. It asks you to run it again in a moment.

A claim lasts two hours. While a proven claim waits for signup, the hourly
task runs to collect the key; this only happens in the month you ran
`connect_arena()`, which already counts as a month of use. If the code has not
arrived, or Arena cannot see the install yet, `connect_arena()` says so and the next
call carries on with the same claim. Running `connect_arena()` on an app whose key
Arena still accepts changes nothing. After Arena revokes the key (for example
"Disconnect app" in Arena), `connect_arena()` starts a new claim.

The app rotates a key it collected once it is 90 days old, from `submit`,
`publish` with serving, `connect_arena()` or an hourly run that is billing anyway,
and never while a run's services are live: they read the key when they start.
The new key's first use revokes every other key of this install, except the one
it replaced, which keeps working for 5 more minutes.

After an unpurchased trial ends, Arena also refuses run heartbeats and repeat
opens and deployment heartbeats (403 `snowflake_trial_ended`). A run's head
stops renewing and the trainer stops when its token expires; the hourly task
drops a served model's service and closes its deployment, and both log the
trial message.
`CALL <app>.app_code.rotate_arena_key();` rotates it now.

Run and deployment opens send the app's region and its `listing_purchased()`
flag, so Arena sees a purchase at once instead of after its daily sync.

Keys AgileRL issues by hand (tests, enterprise deals) still go in with
`app_code.configure(org_key, external_user_id, base_url)`. The app does not
rotate those.

## Procedures

Python 3.13 stored procedures in `app_code/arena_app.py` (billing and
served-model helpers in `app_code/app_billing.py`, `connect` and key rotation
in `app_code/app_keys.py`, the trial check in `app_code/app_trial.py`), owner's
rights, with
the other `agilerl-arena` dependencies they import (`aiofiles`, `click`,
`cryptography`, `deprecation`, `httpx`, `jwcrypto`, `lazy_loader`, `numpy`,
`pydantic`, `pyyaml`, `requests`, `requests-toolbelt`, `rich`) and
`snowflake-snowpark-python` from the Snowflake Anaconda channel.

Every account-level object the app creates is named `AGILERL_APP_<APP>_<object>`,
where `<APP>` is the application name upper-cased with every character other
than `A-Z`, `0-9` and `_` replaced by `_`. The setup script's
`app_code.object_prefix()` derives it, and the procedures read it from there.
So two installs in one account (or a reinstall under a new name before the old
app is dropped) each get their own warehouse `AGILERL_APP_<APP>_WH`, integrations
`AGILERL_APP_<APP>_HF_EAI` and `AGILERL_APP_<APP>_ARENA_EAI`, and pools, and no
name collides with the CLI's `arena_*` objects. Two app names that map to the
same `<APP>`, such as `"my app"` and `MY_APP`, collide; the second install fails
setup with error -20510.

An install from before per-app names keeps its `agilerl_app_*` warehouse,
integrations and pools after an upgrade, since setup never drops them:

- Setup creates the new integrations and warehouse alongside them, and the
  procedures switch to the new Arena integration.
- Runs and published models keep the pools, warehouse and Hugging Face
  integration recorded in `setup.json`, so the old objects stay in use.
  `set_pools` writes the new pools to `setup.json`, but `setup.json` keeps
  listing `agilerl_app_hf_eai`, and `set_pools` does not drop the old pools.
- Dropping the app drops both sets. Reinstall instead of upgrading to clear
  them.

| Procedure | Role | Does |
|---|---|---|
| `app_code.connect_arena(base_url)` | `app_admin` | Starts or carries on one claim with Arena, reads its code from `claims.challenges`, proves it, and returns the signup link. A second call while a claim is starting is asked to run again. Once the claim is linked, collects the org key into the three secrets, then checks capabilities; a failed check leaves the key saved. See Connect to Arena. With a saved key Arena still accepts, only rotates it when due. Refused after an unpurchased trial. `base_url` defaults to `https://arena.agilerl.com` and must be in `app_state.arena_api_egress`. |
| `app_code.rotate_arena_key()` | `app_admin` | Swaps a key `connect` collected for a new one (`POST /api/cli/v1/snowflake/app-keys/rotate`), saves it, and uses it once; Arena retires the old key 5 minutes later. Refused for keys from `configure` and while a run's services are live. |
| `app_code.configure(org_key, external_user_id, base_url)` | `app_admin` | For keys AgileRL issues by hand. Refuses keys without `arena_org_` and base URLs whose host is not in `app_state.arena_api_egress`; calls `GET /api/cli/v1/capabilities` through `AGILERL_APP_<APP>_ARENA_EAI` and requires `features.snowflakeTraining`; then sets the three secrets and lists `features.snowflakePricedGpuFamilies` when Arena sends it. Secret values are never logged, returned or quoted in errors. |
| `app_code.set_pools(gpu_families, max_gpus, broker_family, head_family)` | `app_admin` | Creates or resizes `AGILERL_APP_<APP>_BROKER_POOL`, `AGILERL_APP_<APP>_HEAD_POOL` and one `AGILERL_APP_<APP>_GPU_<FAMILY>` per listed family (CLI defaults); drops the app's `AGILERL_APP_<APP>_GPU_*` pools of families no longer listed; refuses to shrink or drop under a running service; writes `setup.json`. A GPU family missing from the region, or listed with another GPU count per node than the app sizes (Snowflake reuses the AWS `GPU_R6K_G1_*` names on GCP for nodes with half the GPUs), is refused with the GPU families the app can size there. |
| `app_code.gpu_families()` | `app_admin` | Lists the GPU families `set_pools` can use: returned by `SHOW COMPUTE POOL INSTANCE FAMILIES` with the GPU count per node the app sizes, and in `features.snowflakePricedGpuFamilies` when Arena sends it. Reads capabilities with the three Arena secrets through `AGILERL_APP_<APP>_ARENA_EAI`. |
| `app_code.register_reference(ref_name, operation, ref_or_alias)` | `app_admin` | Reference callback: `ADD`, `REMOVE`, `CLEAR` on `source_table` / `source_view`, and on `export_stage`, which holds one stage. |
| `app_code.submit(manifest, gpus, source_reference, source_alias, gpu_family, reward_file, checkpoint_steps, resume_from)` | `app_admin` | `arena snowflake submit` on the app pools: the same broker, head and worker job services in `app_state`, with the head reading the `app_state` secrets through `AGILERL_APP_<APP>_ARENA_EAI`. `manifest` and `reward_file` are file contents. When capabilities list `features.snowflakePricedGpuFamilies`, an unpriced `gpu_family` is refused and an omitted one is picked from priced pools, before anything starts. A saved pool whose family this region lists with another GPU count per node is refused the same way. That read uses the three Arena secrets through `AGILERL_APP_<APP>_ARENA_EAI`. The job reads `SELECT * FROM reference('<source_reference>', '<source_alias>')`. Returns the run id, the service names and any manifest warnings. |
| `app_code.status(run_id)` | `app_user` | `arena snowflake status`. Refuses run ids with no `job.json` on `@app_state.runs`. |
| `app_code.logs(run_id, role, instance)` | `app_user` | `arena snowflake logs`. |
| `app_code.cancel(run_id)` | `app_admin` | `arena snowflake cancel`, then suspends the app's GPU pools that have no service or job left. |
| `app_code.reconcile_usage()` | none | Collects the key of a linked claim, bills usage, rotates a key `connect` collected once it is 90 days old, suspends the app's GPU pools that have no service or job left, then suspends its task when the app is idle; see Billing. Only the hourly task calls it, so two calls never overlap. |
| `app_code.reconcile_now()` | `app_admin` | Queues a run of `app_state.reconcile_usage_task` with `EXECUTE TASK`, so billing can be run by hand. It starts after any run in progress. |
| `app_code.version_init()` | none | Version initializer: after install, upgrade, or a failed upgrade, points `setup.json` at this version's image (a version runs only its own image, so `submit` and `publish` would otherwise refuse until `set_pools` ran again), then resumes the billing task when the app has work to bill and leaves it suspended otherwise. |
| `app_code.publish(run_id, model_name, version_name, step, serve, gpu_count, service_name, max_model_len)` | `app_admin` | `arena snowflake publish` into the Model Registry in `app_state`, waiting for the job like the CLI. `serve` starts the vLLM service, whose public endpoint uses `BIND SERVICE ENDPOINT`, and bills it as an Arena deployment; see Billing. The app grants the service's `ALL_ENDPOINTS_USAGE` role to `app_user`, after publish and on each hourly run, so account roles holding `app_user` can call the endpoint. |
| `app_code.endpoint(model, version_name)` | `app_user` | `arena snowflake endpoint` for a model in `app_state`. |
| `app_code.unpublish(model, version_name, drop_version)` | `app_admin` | `arena snowflake unpublish` for a model in `app_state`, then closes the service's Arena deployment. |
| `app_code.export_model(run_id, step)` | `app_admin` | Copies a run's merged checkpoint (the files `publish` logs) to `<run_id>/checkpoint_<n>/` on the stage bound to `export_stage`; see Keep your model. |

`app_user` gets the read-only procedures. Starting, cancelling, publishing and
unpublishing spend GPU credits or drop work, so they need `app_admin`.
`export_model` writes outside the app, so it needs `app_admin` too.

## Keep your model

Uninstalling the app drops `@app_state.runs` and the Model Registry models in
`app_state`, and neither application role can read them. To keep a trained
model, an `app_admin` binds one of the consumer's stages to the `export_stage`
reference and calls `export_model` before uninstalling:

```sql
CREATE STAGE my_db.my_schema.arena_models;
CALL <app>.app_code.register_reference('export_stage', 'ADD',
  SYSTEM$REFERENCE('STAGE', 'my_db.my_schema.arena_models', 'PERSISTENT', 'READ', 'WRITE'));
CALL <app>.app_code.export_model('<run_id>');        -- newest merged checkpoint
CALL <app>.app_code.export_model('<run_id>', 1024);  -- the checkpoint of step 1024
LIST @my_db.my_schema.arena_models/<run_id>/;
```

The stage can also be bound in Snowsight on the app's references. Snowflake
requires READ with WRITE on a stage reference
([references](https://docs.snowflake.com/en/developer-guide/native-apps/requesting-refs)).

`export_model` runs `COPY FILES` from the run's `checkpoint_<n>/merged/` to
`@reference('export_stage')/<run_id>/checkpoint_<n>/`. Those are the Hugging
Face files `publish` logs as version `checkpoint_<n>`. It copies nothing else
from the app, overwrites files of the same name, and refuses when no stage is
bound or the run has no complete merged export at that step.

It starts no service and adds no usage item, so Arena bills nothing for it.
`COPY FILES` is serverless and appears on the consumer's own Snowflake bill
([COPY FILES](https://docs.snowflake.com/en/sql-reference/sql/copy-files)),
which also caps each file at 5 GB. Like any call to the app, it counts towards
the month's listing fee (see Idle months). It works after the trial ends
unpurchased, so a consumer who does not buy keeps what they trained.

## Trial

A consumer who installs the paid listing without buying it is on the
listing's 7-day Snowflake trial. The app stops it one hourly task interval
early, 7 days less 60 minutes from install (`TRIAL` in `app_code/app_trial.py`;
the setup script records the install time in `app_state.trial`), because
Snowflake ends the app's access when its trial ends and the app could then no
longer drop its own services. After the cutoff, until the listing is
purchased, `submit` and `publish` refuse with a message to purchase it, and
each hourly run drops every live service in `app_state` (training jobs,
publish jobs, inference services) and every served model's service, before
metering closes their deployments. After purchase, both procedures work again.
`export_model` is not gated: it starts nothing and only copies out a model
trained during the trial.

Purchase is read with `SYSTEM$IS_LISTING_PURCHASED()` through the secure
function `app_code.listing_purchased()`. Called straight from a procedure or
the task, the function returns TRUE on a trial install too. An install not
from a listing reads as purchased, so it is never stopped. The listing's trial
must be a 7-day "Limited time" trial for the cutoff to land in its last hour.
Uninstalling drops `app_state.trial`, so a reinstall restarts the app's clock,
but never past the end of the listing's trial.

## Billing

App runs set `ARENA_SNOWFLAKE_CHANNEL=native_app` in the head spec, so the head
opens the Arena run with `"channel": "native_app"` and Arena bills it through
Snowflake. CLI runs never set it. `submit` adds each run to
`app_state.usage_items` before it starts any job service.

A served model is billed as an Arena deployment. `publish` with `serve`
records the service's model, version, run, GPU family and GPU count in
`app_state.served_models` before it waits for the job. Once the service
exists, it adds `<SERVICE>-<yyyymmddThhmmssZ>` to `usage_items` with the
service name and calls `POST /api/cli/v1/snowflake/deployments`. If Arena
refuses, the service is dropped, so no model serves unbilled. `unpublish`
closes the deployment after dropping the service.

Billing starts when a served model is running and stops when it suspends or
is unpublished. Each hourly run first meters deployments: it heartbeats every
open deployment whose service is running, and closes one whose service is
suspended, gone or ended; a resumed service gets a new deployment. Because
the check is hourly, each suspend can bill up to an hour of idle time, and
each resume can serve up to an hour before billing starts. When Snowflake
training is turned off, the app drops the service and closes the deployment.
The usage row is marked closed only after that close returns, so a failed
close is tried again on the next hour. When Arena has already closed a
deployment (no heartbeat for 2 hours), the item is marked closed and settles
from the usage read, and the live service gets a new deployment. Any live
service in `served_models` with no open deployment, such as one a publish
started after its wait ended, gets one too. A deployment Arena has never
answered for does not count, so a lost open is tried again. A refused open
drops the service. Arena's reason
is logged and stored in `usage_items.last_error` and `last_error_at`.

`app_admin` can `SELECT` from `app_state.usage_items`,
`app_state.billed_hours` and `app_state.served_models` to audit billing,
including why a serve was dropped. A failed usage read or billing call is
logged and its reason stored in the item's `last_error` and `last_error_at`.
`app_user`, which `app_admin` holds, can read `app_state.arena_metrics`.

The serverless task `app_state.reconcile_usage_task` runs every 60 minutes
(XSMALL initial size, never suspended after failures) and calls
`app_code.reconcile_usage()`. That procedure reads the three Arena secrets
through `AGILERL_APP_<APP>_ARENA_EAI` and, for each unsettled item, calls
`GET /api/cli/v1/snowflake/runs/<run_id>/usage` or
`GET /api/cli/v1/snowflake/deployments/<deployment_id>/usage`. For each charged hour it has not
billed yet, it emits one event with `SYSTEM$CREATE_BILLING_EVENTS`
([billable events](https://docs.snowflake.com/en/developer-guide/native-apps/adding-custom-event-billing)):
class `ARENA_USAGE`, subclass `run` or `deployment`, `start_timestamp` = the hour start,
`timestamp` = the hour end or now for a partial hour, `base_charge` = cents ×
0.01 USD, `objects` = the run or deployment id. Billing lags usage by up to about an hour.
Every run logs its summary at INFO (deployments heartbeated, closed, opened
and dropped; items settled, events emitted, failed, missing, skipped until
their daily read, stale, and hours Snowflake never confirmed), or why nothing ran.
The listing's pricing plan carries the event's `billing_quantity` and unit.

`app_state.billed_hours` holds one row per charged hour. A MERGE claims hours
for one call, and `attempted_at` is set just before the billing call, so an
hour is never emitted twice. If Snowflake refuses a batch with one of its
documented messages, the hours are deleted and the next run bills them. An
app not installed from a listing (error 090898) is refused this way on every
run, so it never bills and never counts hours as `unconfirmed`. Any
other failure leaves them attempted with no `emitted_at`; they are never sent
again, and the summary counts those older than an hour as `unconfirmed`, to be
checked against `SNOWFLAKE.DATA_SHARING_USAGE.MARKETPLACE_PAID_USAGE_DAILY`
and logged as a warning. A MERGE that leaves two rows for one hour stops that
item before anything is emitted.

Items are read least recently checked first (`last_checked_at`). The first
failed read other than a 404 (transport error, exhausted retries, rejected
credentials) stops reading for that run. An item settles once Arena reports it
closed and every charged hour has been attempted. A 404 never settles an item
Arena has answered for before (`first_seen_at`) or that has billed hours; it
is logged and kept. An item that has only ever had 404s is read at most daily,
logged on each 404, and settles unbilled 30 days after its first 404
(`first_404_at`). Items Arena has kept open for over 7 days are logged.
Procedure logs, these included, go to the consumer's active event table at the
manifest's `log_level` (`INFO`).

### Idle months

A usage-based plan charges its monthly fee for a calendar month in which a
query references the app or an app compute pool runs
([pricing models](https://docs.snowflake.com/en/collaboration/provider-listings-pricing-model)).
So an idle app runs nothing. The billing task is the app's only scheduled
work. The setup script leaves it suspended; `submit` resumes it after adding
the run to `usage_items`, and `publish` with `serve` after recording the
served model. After billing, each hourly run suspends the task when no
service in `app_state` is live and every usage item has settled, except items
Arena has only answered 404 for, so a run's last hours are billed first. The
check runs again after suspending, so a run submitted meanwhile resumes it.
`connect_arena()` also resumes it after Arena accepts a claim's code, and a proven
claim keeps it resumed until the key is collected or the claim expires two
hours later; key collection and rotation never resume it by themselves.
Pools resume when a service starts and suspend themselves after
`AUTO_SUSPEND_SECS` (300) with no service; the warehouse after 60 seconds.
A GPU pool still waiting for a node never reaches that timer, so `cancel` and
each hourly run suspend the app's GPU pools that have no service or job left.

## Manifest

- `manifest_version: 2`, so Snowflake grants the requested privileges at
  install ([automated granting](https://docs.snowflake.com/en/developer-guide/native-apps/requesting-auto-privs)).
  External access through app specifications requires version 2
  ([request external access](https://docs.snowflake.com/en/developer-guide/native-apps/requesting-app-specs-eai)).
- Privileges: `CREATE COMPUTE POOL`, `CREATE WAREHOUSE`, `BIND SERVICE ENDPOINT`,
  `CREATE EXTERNAL ACCESS INTEGRATION`, `EXECUTE TASK`, `EXECUTE MANAGED TASK`.
  All six are on the auto-grant list, so the app can resume the
  billing task. A version 2 manifest cannot change this list in a patch.
- `lifecycle_callbacks.version_initializer` is `app_code.version_init`
  ([version initializer](https://docs.snowflake.com/en/developer-guide/native-apps/update-app-develop)).
  The setup script suspends the billing task before redefining it; when an
  upgrade fails, Snowflake calls the previous version's initializer, which
  resumes the task when the app has work to bill.
- Image: `/arena_provider/images/repo/snowflake-job:<tag>`, the one trainer
  image the broker, head and workers all run. The repository `arena_provider.images.repo`
  sits in the provider account outside the application package, as
  [Snowflake recommends](https://docs.snowflake.com/en/developer-guide/native-apps/container-containers)
  (a repository inside the package is replicated by Cross-Cloud Auto-Fulfillment).
  The manifest cannot template the tag; `0.0.0-py313-cu130` is a placeholder the
  `snowflake-app-build` recipe replaces with the release tag at build.
- References `source_table` (TABLE) and `source_view` (VIEW), SELECT only. A
  reference has one object type, so table-or-view is two references. Both are
  multi-valued so a consumer can bind several datasets and pick one per run.
  Their `register_callback` is `app_code.register_reference`, written with the
  other procedures.
- Reference `export_stage` (STAGE), READ and WRITE, single-valued, with the
  same callback. `export_model` copies trained models to it.
- Application roles and external hosts are not manifest keys. The setup script
  creates them ([application roles](https://docs.snowflake.com/en/developer-guide/native-apps/creating-setup-script#about-application-roles),
  [app specifications](https://docs.snowflake.com/en/developer-guide/native-apps/requesting-app-specs-eai)):
  `app_admin` and `app_user` (granted to `app_admin`); network rules, EAIs and
  one app specification each for Hugging Face (the 10 hosts in
  `setup_sql.HF_HOSTS`) and `arena.agilerl.com:443`. Hosts in a rule and its
  specification must match exactly.

## Current objects and their app equivalents

| Current object or step | In the app |
|---|---|
| Role `arena_trainer` + grants | Application roles `app_admin` (configure, pools, submit, cancel, publish, unpublish) and `app_user` (status, logs, endpoint). The consumer grants them to account roles; `invite USER` becomes granting `app_admin` to a user who trains, or `app_user` to one who only watches. |
| Service user `arena_job_user` and its key pair | Dropped. Services run as the app, so no user or key exists. |
| Database/schema `arena.training`, stage `runs`, table `arena_metrics`, file format | Inside the app: procedures in the versioned schema `app_code`; stage `runs` (SNOWFLAKE_SSE), table `arena_metrics` and file format `arena_json` in the stateful schema `app_state`, with the CLI's definitions. |
| Image push into `arena.training.images` | Dropped for the consumer. The image is in the provider repository `arena_provider.images.repo` and is listed under `artifacts.container_services.images`. |
| Warehouse `arena_wh` | `CREATE WAREHOUSE` privilege; the setup script creates `AGILERL_APP_<APP>_WH` (XSMALL, auto-suspend 60 s, initially suspended). |
| Resource monitor `arena_monitor` | Dropped. Only `ACCOUNTADMIN` can create a resource monitor ([resource monitors](https://docs.snowflake.com/en/user-guide/resource-monitors)) and `CREATE RESOURCE MONITOR` is not on the app auto-grant list. The consumer may assign their own monitor to the app's warehouse. |
| Budget `arena_gpu_budget` | Dropped from the app; consumer-side step. Creating a custom budget needs `SNOWFLAKE.BUDGET_CREATOR`, which an app cannot be auto-granted. The consumer can add the app to their own budget, which tracks every warehouse and compute pool the app owns; app-owned objects cannot be added one by one ([custom budgets](https://docs.snowflake.com/en/user-guide/budgets/custom-budget)). |
| Compute pools and `pools set` | `CREATE COMPUTE POOL` privilege. No pool exists after install, since an idle pool bills; `app_code.set_pools` creates and resizes them. No budget link (budgets are dropped). `uses_gpu: true` is set because worker pools use GPU families. |
| `BIND SERVICE ENDPOINT ON ACCOUNT` | `BIND SERVICE ENDPOINT` privilege. |
| Network rules + EAIs `arena_hf_eai`, `arena_api_eai` | `CREATE EXTERNAL ACCESS INTEGRATION` privilege; the setup script creates the rules `app_state.hf_egress` and `app_state.arena_api_egress`, the integrations `AGILERL_APP_<APP>_HF_EAI` and `AGILERL_APP_<APP>_ARENA_EAI`, and the `hf_access` and `arena_access` app specifications the consumer approves. Setup fails if either integration name is already in the account and the app does not own it; on upgrade the app keeps the ones it owns. |
| `grant --host` → `arena_env_eai` | Not built yet; follow-up. Planned: an `app_admin` procedure creates a rule, EAI and app specification for each extra host; the consumer approves each specification. |
| `grant --table` (SELECT + USAGE) | References `source_table` / `source_view`, SELECT only. No `--source-query`: custom SQL goes in a view the consumer binds. |
| Secrets `arena_org_key`, `arena_external_user_id`, `arena_base_url` | Same three secrets in `app_state`, created empty by the setup script. `app_code.connect_arena` sets them when Arena issues the key, then checks `GET /api/cli/v1/capabilities`; a failed check leaves them saved. `app_code.configure` sets them only after that check. `AGILERL_APP_<APP>_ARENA_EAI` allows them, so `reconcile_usage` can read them. |
| `setup.json` runtime profile | Same document shape at `@app_state.runs/_arena/setup.json`, written by `app_code.set_pools`. No key material. |
| Trial-account probe EAI (error 509009) | The setup script creates the EAIs in a block that turns error 509009 into a trial-account message. |
| Event table warning for `logs` | Container logs reach the consumer's active event table; consumers may share them with the provider ([event sharing](https://docs.snowflake.com/en/developer-guide/native-apps/event-about)). Containerized apps support only the `SNOWFLAKE$ALL` event definition, and container log levels are set in the service spec, not the manifest ([event definitions](https://docs.snowflake.com/en/developer-guide/native-apps/event-definition)). `logs` reads the consumer event table where the caller can see it. |
| `submit` / `status` / `logs` / `cancel` / `publish` / `endpoint` / `unpublish` | Procedures of the same names (see Procedures). |
| `teardown` | Dropped. Uninstalling the app drops its objects, including models and checkpoints inside it. Copy them out first with `export_model`. |
