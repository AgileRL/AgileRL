# Arena Snowflake Native App

Application package files for the Arena Snowflake Native App, the only way to
run Arena inside a Snowflake account. The procedures call the shared code in
`agilerl/snowflake/`. These files are not part of the `agilerl`
wheel or sdist.

This file is for developers. `app_readme.md` is the README consumers see in
the installed app; the package ships it as `README.md`.

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
`PATH`, plus the AWS CLI; no Docker. It:

1. Copies this directory to `dist/snowflake-app/` and replaces the
   `0.0.0-py313-cu130` placeholder in `manifest.yml` and `app_code/*.py` with `TAG`.
2. Builds the `agilerl` wheel and, with
   `bundle.py`, stores it as `agilerl_arena.zip` together with `python-keycloak`
   at its `uv.lock` version, the one dependency the Snowflake Anaconda channel
   lacks. The procedures import `agilerl.snowflake` from it instead of
   copying code.
3. Creates `arena_provider.images.repo` if missing and runs `crane copy` from
   private ECR, `761296822462.dkr.ecr.us-east-1.amazonaws.com/containers/snowflake-job:TAG`,
   to that repository. It logs crane in to ECR with
   `aws ecr get-login-password --region us-east-1`, so it needs AWS credentials
   that can pull from that repository (the developers IAM group can). The
   Snowflake registry password is the whole JSON output of
   `snow spcs image-registry token --format JSON`. The image is about 12 GB, so
   the copy takes a while.
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
the other `agilerl` dependencies they import (`aiofiles`, `click`,
`cryptography`, `deprecation`, `httpx`, `jwcrypto`, `lazy_loader`, `numpy`,
`pydantic`, `pyyaml`, `requests`, `requests-toolbelt`, `rich`) and
`snowflake-snowpark-python` from the Snowflake Anaconda channel.

Every account-level object the app creates is named `AGILERL_APP_<APP>_<object>`,
where `<APP>` is the application name upper-cased with every character other
than `A-Z`, `0-9` and `_` replaced by `_`. The setup script's
`app_code.object_prefix()` derives it, and the procedures read it from there.
So two installs in one account (or a reinstall under a new name before the old
app is dropped) each get their own warehouse `AGILERL_APP_<APP>_WH`, integrations
`AGILERL_APP_<APP>_HF_EAI` and `AGILERL_APP_<APP>_ARENA_EAI`, and pools. Two app names that map to the
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
| `app_code.set_pools(gpu_families, max_gpus, broker_family, head_family)` | `app_admin` | Creates or resizes `AGILERL_APP_<APP>_BROKER_POOL`, `AGILERL_APP_<APP>_HEAD_POOL` and one `AGILERL_APP_<APP>_GPU_<FAMILY>` per listed family (default broker and head families when omitted); drops the app's `AGILERL_APP_<APP>_GPU_*` pools of families no longer listed; refuses to shrink or drop under a queued or running service, job services included, counting the nodes a service asked for while it waits for one; refuses a `max_gpus` that is not a multiple of a listed family's GPUs per node; writes `setup.json`. A GPU family missing from the region, or listed with another GPU count per node than the app sizes (Snowflake reuses the AWS `GPU_R6K_G1_*` names on GCP for nodes with half the GPUs), is refused with the GPU families the app can size there. |
| `app_code.gpu_families()` | `app_admin` | Lists the GPU families `set_pools` can use: returned by `SHOW COMPUTE POOL INSTANCE FAMILIES` with the GPU count per node the app sizes, and in `features.snowflakePricedGpuFamilies` when Arena sends it. Reads capabilities with the three Arena secrets through `AGILERL_APP_<APP>_ARENA_EAI`. |
| `app_code.register_reference(ref_name, operation, ref_or_alias)` | `app_admin` | Reference callback: `ADD`, `REMOVE`, `CLEAR` on `source_table` / `source_view`, on `export_stage`, which holds one stage, and on `reward_function`, which holds one function. |
| `app_code.submit(manifest, gpus, source_reference, source_alias, gpu_family, reward_file, checkpoint_steps, resume_from)` | `app_admin` | Starts the broker, head and worker job services in `app_state` on the app pools, with the head reading the `app_state` secrets through `AGILERL_APP_<APP>_ARENA_EAI`. `manifest` is the file contents. `reward_file` is refused when set, with the two reward options; see Rewards. So is a manifest with `environment.rubric_file_path`, and one whose `environment.reward` is `snowflake_function` while nothing is bound to `reward_function`. When capabilities list `features.snowflakePricedGpuFamilies`, an unpriced `gpu_family` is refused and an omitted one is picked from priced pools, before anything starts. A saved pool whose family this region lists with another GPU count per node is refused the same way. That read uses the three Arena secrets through `AGILERL_APP_<APP>_ARENA_EAI`. A `source_alias` that `SYSTEM$GET_ALL_REFERENCES` does not list for `source_reference` is refused with the bound aliases and the `register_reference` call that binds one, since `reference()` would fail only once the GPU nodes are up. So is a `gpu_family` the app has no pool for, with the `set_pools` call that adds it to the existing pools, and a GPU count that is not a multiple of the family's GPUs per node or is more than the pool's `MAX_NODES` hold, with the counts the pool can run; that last check reads `setup.json`, so it runs before the run is recorded for billing. A run the memory estimator says needs more GPUs than asked (an RL run on 1 GPU needs a trainer GPU and a rollout GPU) is refused with the count to ask for, and an OPD run with a separate teacher is refused on a family whose node cannot hold the engine and teacher together, since the run packs them onto one node. RL weight sync reads the trainer's GPU buffer with NIXL over UCX TCP, on `lo` within a node and `eth0` between nodes, so a trainer and its rollout engines may sit on different nodes; the job containers set `UCX_TLS=self,sm,cuda_copy,cuda_ipc,tcp`, `UCX_NET_DEVICES=lo,eth0` and `LD_PRELOAD=/usr/local/lib/libnetlink_done.so`, which returns an embedded `NLMSG_DONE` on the next `recv()` or `__recv_chk()`. If `EXECUTE JOB SERVICE` hits the account's node limit (Snowflake error 395034), the services already started are dropped and the refusal says to wait or ask Snowflake to raise the limit. A failed start deletes the run's `usage_items` row, so nothing is billed for it. A manifest that fails validation is refused with one line per field: what is wrong and what to change, without pydantic's formatting; an `algorithm.name` the app does not train is refused with the names it does. Each refusal ends with `Nothing was started.` The job reads `SELECT * FROM reference('<source_reference>', '<source_alias>')`. Returns the run id, the service names and any manifest warnings. |
| `app_code.status(run_id)` | `app_user` | The state of each job service and the run. Refuses run ids with no `job.json` on `@app_state.runs`. `cancel` drops the services, so a cancelled run reads `gone`; `cancelling` and `cancelled` show only while Snowflake still lists a service in that state. |
| `app_code.logs(run_id, role, instance)` | `app_user` | Logs of one role's main container. While the container is still pending, it says so and points to `status`; once it has started but written nothing, it says that too. The job containers set `AGILERL_DEBUG=none`, so the trainer logs at INFO and above, with no DEBUG timing lines. |
| `app_code.cancel(run_id)` | `app_admin` | Drops the run's job services, then suspends the app's GPU pools that have no service or job left. |
| `app_code.reconcile_usage()` | none | Collects the key of a linked claim, bills usage, rotates a key `connect` collected once it is 90 days old, suspends the app's GPU pools that have no service or job left, then suspends its task when the app is idle; see Billing. Only the hourly task calls it, so two calls never overlap. |
| `app_code.reconcile_now()` | `app_admin` | Queues a run of `app_state.reconcile_usage_task` with `EXECUTE TASK`, so billing can be run by hand. It starts after any run in progress. |
| `app_code.version_init()` | none | Version initializer: after install, upgrade, or a failed upgrade, points `setup.json` at this version's image (a version runs only its own image, so `submit` and `publish` would otherwise refuse until `set_pools` ran again), then resumes the billing task when the app has work to bill and leaves it suspended otherwise. |
| `app_code.publish(run_id, model_name, version_name, step, serve, gpu_count, service_name, max_model_len)` | `app_admin` | Logs a run's checkpoint to the Model Registry in `app_state` and waits for the job. `serve` starts the vLLM service with no endpoint outside Snowflake and bills it as an Arena deployment; see Billing. Once the service is up, after publish or on the next hourly run, the app creates the SQL function `app_state.<SERVICE>_CHAT` and grants it, and the service's `INFERENCE_SERVICE_FUNCTION_USAGE` role, to `app_user`; see Call a served agent. |
| `app_code.endpoint(model, version_name)` | `app_user` | Returns the chat function of the served model on a version in `app_state` and a `SELECT` that calls it. |
| `app_code.unpublish(model, version_name, drop_version)` | `app_admin` | Drops a served model's service, or its version, in `app_state`, then drops the service's chat function and closes its Arena deployment. |
| `app_code.export_model(run_id, step)` | `app_admin` | Copies a run's merged checkpoint (the files `publish` logs) to `<run_id>/checkpoint_<n>/` on the stage bound to `export_stage`; see Keep your model. |

`app_user` gets the read-only procedures. Starting, cancelling, publishing and
unpublishing spend GPU credits or drop work, so they need `app_admin`.
`export_model` writes outside the app, so it needs `app_admin` too.

### Models

`submit` trains only models in the Arena catalog (`arena models supported`).
Snowflake does not let a Native App run code from outside its package, so
`submit` refuses a model whose family loads Python from its Hugging Face repo
(`trust_remote_code=True` for vLLM or the trainer). In the catalog that is
`XiaomiMiMo/MiMo-V2.6-Flash-RL`. The model types are listed in
`REMOTE_CODE_MODEL_TYPES` in `agilerl/models/model_info.py`. The policy and
an OPD `teacher.model` are both checked. `submit` also refuses
`algorithm.vllm_engine_args.trust_remote_code: true`, because the rollout
engine forwards that flag to vLLM, which then loads the repo's `auto_map`
modules. `false` does not let a remote-code family through. These checks run
before the app runs any SQL or starts anything. These models still train on
Arena, with the `arena` CLI, and on BYOC.

## Rewards

For the same reason the app takes no reward file. A post-training run over a
table scores each completion in one of two ways, set in the manifest's
`environment.reward`.

**Built-in rewards** run inside the app's image. Each scores the model's final
answer, the text after any reasoning, against the row's `answer` column (map
another column onto it with `environment.columns`), and gives 1 or 0:

| `type` | Scores 1 when | Options |
|---|---|---|
| `exact_match` | the answer equals `answer` | `normalize` (default true: ignore case, extra spaces and a trailing full stop), `answer_pattern` |
| `numeric` | the answer's number is within `tolerance` of `answer` | `tolerance` (default `1e-6`), `answer_pattern`; unset reads the last `\boxed{...}`, else the last number |
| `regex` | the reply matches `pattern` | `pattern`, `full_match` |
| `json_schema` | the reply's first JSON object is valid against `schema` | `schema` |
| `tool_call` | the reply's first tool call (`{"name": ..., "arguments": {...}}`, bare or in `<tool_call>` tags) is the expected one | `name` and `required_arguments`; without `name`, `answer` holds the expected call and the name and arguments must equal it |

`answer_pattern` is a regular expression whose first group is the answer, such
as `'Answer: (.*)'`; the last match counts.

```yaml
environment:
  env_type: rollout
  prompt_template:
    system_0: Solve the problem. End with "Answer: <number>".
    user_1: "{question}"
  reward:
    type: numeric
    answer_pattern: "Answer: (.*)"
```

**Your own reward** is a function in your account, bound to the app's
`reward_function` reference. Snowflake runs it in your account under your
ownership; the app calls it with SQL and never reads its code. It takes the
prompt (the row's `question`), the final answer and the row's `answer` as
text, and returns a FLOAT:

```sql
CREATE OR REPLACE FUNCTION my_db.rewards.score_answer(
    prompt STRING, completion STRING, answer STRING)
  RETURNS FLOAT
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.12'
  HANDLER = 'score'
AS $$
import re

def score(prompt, completion, answer):
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion or "")
    if not numbers or answer is None:
        return 0.0
    return 1.0 if abs(float(numbers[-1]) - float(answer)) < 1e-6 else 0.0
$$;

CALL <app>.app_code.register_reference('reward_function', 'ADD',
  SYSTEM$REFERENCE('FUNCTION', 'my_db.rewards.score_answer(STRING, STRING, STRING)',
    'PERSISTENT', 'USAGE'));
```

Then set `environment.reward` to `{type: snowflake_function}`. The function
can also be bound in Snowsight on the app's references, and binding another
replaces it. Any Snowflake function with that signature works: Python, SQL,
Java, or one that calls `AI_COMPLETE` to grade with an LLM. A SQL function must
cast its result (`... ::FLOAT`); `IFF(..., 1.0, 0.0)` is a NUMBER and the
create fails.

Each training process sends finished groups of completions to the function in
one query on the app's XSMALL warehouse, and each result is that completion's
reward. Groups that finish while a query runs share the next one. A query costs
about the same whatever its size: about 1.3 s for a Python function and 0.15 s
for a SQL one, from 64 to 2,048 rows. The first query after the warehouse
resumes takes about 20 s. Queries are billed to your account like any query on
the app's warehouse. A dropped connection is retried; a function that fails,
returns NULL, or returns another number of rows stops the run with that error.

## Call a served agent

A served model has no endpoint outside Snowflake. Account roles holding
`app_user` call it from SQL, Snowpark or a notebook through its chat function,
which takes the chat messages, a temperature and a token limit, and returns the
OpenAI chat completion as an OBJECT:

```sql
CALL <app>.app_code.publish('<run_id>', 'support_agent', NULL, NULL, TRUE);
CALL <app>.app_code.endpoint('support_agent');  -- the function and an example call
SELECT <app>.app_state.SUPPORT_AGENT_CHECKPOINT_1024_CHAT(
  PARSE_JSON('[{"role": "user", "content": "Hello"}]'),
  0.7,
  256
):choices[0]:message:content::STRING AS reply;
```

Run it over a table of prompts to answer them all in one query. For every chat
option, call the Model Registry's own method on the service, which `app_user`
can also use. It takes all ten arguments in this order: messages, temperature,
max_completion_tokens, stop, n, stream, top_p, frequency_penalty,
presence_penalty, response_format:

```sql
SELECT <app>.app_state.SUPPORT_AGENT_CHECKPOINT_1024!__CALL__(
  PARSE_JSON('[{"role": "user", "content": "Hello"}]'), 0.7, 256, NULL, 1, FALSE, 1.0, 0.0, 0.0, NULL
);
```

The service suspends after 30 minutes without calls and resumes on the next
one. That first call takes about 2 minutes, and longer when the GPU pool has
to start a node too. To serve the agent over HTTPS outside Snowflake, copy it
out with `export_model` (see Keep your model) and serve it yourself.

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
Snowflake. `submit` adds each run to
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
A run is not billed until it records a training step: a row of its
`app_state.arena_metrics` with an `agent_<n>/steps` value above 0. Until then
its hours wait unbilled while Arena has the run open (`waiting` in the
summary). If Arena closes the run first, as when it fails while starting up,
the item settles unbilled with `unbilled_reason` set (`untrained` in the
summary). Once the run has trained, every hour is billed, setup included, up
to the end of the run. A run that already has billed hours is billed as usual.
Every run logs its summary at INFO (deployments heartbeated, closed, opened
and dropped; items settled, events emitted, failed, missing, skipped until
their daily read, stale, hours Snowflake never confirmed, runs waiting for
their first training step, and runs settled unbilled), or why nothing ran.
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
- Privileges: `CREATE COMPUTE POOL`, `CREATE WAREHOUSE`,
  `CREATE EXTERNAL ACCESS INTEGRATION`, `EXECUTE TASK`, `EXECUTE MANAGED TASK`.
  All five are on the auto-grant list, so the app can resume the
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
- Reference `reward_function` (FUNCTION), USAGE, single-valued, with the same
  callback. A run whose `environment.reward` is `snowflake_function` calls it;
  see Rewards.
- Application roles and external hosts are not manifest keys. The setup script
  creates them ([application roles](https://docs.snowflake.com/en/developer-guide/native-apps/creating-setup-script#about-application-roles),
  [app specifications](https://docs.snowflake.com/en/developer-guide/native-apps/requesting-app-specs-eai)):
  `app_admin` and `app_user` (granted to `app_admin`); network rules, EAIs and
  one app specification each for Hugging Face (the 10 hosts in
  `setup_script.sql`) and `arena.agilerl.com:443`. Hosts in a rule and its
  specification must match exactly.
