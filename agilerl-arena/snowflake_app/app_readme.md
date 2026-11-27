# AgileRL Arena

Arena is AgileRL's post-training platform. It turns an open-weight model and your own data into a specialised model or agent for your task.

This app runs post-training and model serving inside your Snowflake account, on Snowpark Container Services GPU compute pools that you control. Your rows, checkpoints and trained models stay in your account. You drive everything with SQL, from a Snowsight worksheet or any Snowflake client.

With the app you can:

- Post-train an open-weight base model from Hugging Face on a table or view you bind to the app. It runs supervised fine-tuning, distillation, preference optimisation and reinforcement learning.
- Score the model's answers with a built-in reward, or with a Snowflake function of yours.
- Publish a trained model to the Snowflake Model Registry and serve it. You call it from SQL through a function the app creates. A served model has no endpoint outside Snowflake.
- Watch runs with SQL: run state, logs, training metrics and billed usage.
- Copy trained model files to a stage of yours at any time.

The examples below call the app `arena_app`. Replace it with the name you gave the app at install.

Full documentation: https://docs.arena.agilerl.com/partners/snowflake-native-app/

## Requirements

- **A paid Snowflake account.** Snowflake trial accounts cannot create external access integrations. On a trial account the install stops with error 509009. Add a payment method, then install again.
- **A region with Snowpark Container Services GPU instance families.** Check with `SHOW COMPUTE POOL INSTANCE FAMILIES;`.
- **A role that can install a paid listing.** Use `ACCOUNTADMIN`, or a role with `IMPORT SHARE`, `CREATE APPLICATION` and `PURCHASE DATA EXCHANGE LISTING` on the account.
- **A role that can approve the external access requests:** `ACCOUNTADMIN`, or a role with `MANAGE APPLICATION SPECIFICATIONS` on the account.
- **A warehouse** in the session you call the procedures from.
- **An email address** to sign up to Arena with, once, when you connect the app.

## Privileges the app requests

Installing as `ACCOUNTADMIN` grants these during install. If you installed with a custom role, run the grants below as `ACCOUNTADMIN`. Until all five succeed, the app has no warehouse, no compute pools and no connection out of your account.

| Privilege | What the app uses it for |
|---|---|
| `CREATE COMPUTE POOL` | Creates the message broker, head and GPU worker pools that training and serving run on. |
| `CREATE WAREHOUSE` | Creates one XSMALL warehouse for exporting the bound table, writing run metrics, and calling a reward function you bind. |
| `CREATE EXTERNAL ACCESS INTEGRATION` | Lets training reach Hugging Face for base model weights, and Arena for licensing. You approve each host. |
| `EXECUTE TASK` | Runs the app's hourly task that bills each finished hour of usage. |
| `EXECUTE MANAGED TASK` | Runs that task on serverless compute, so no warehouse resumes for it. |

```sql
GRANT CREATE COMPUTE POOL ON ACCOUNT TO APPLICATION arena_app;
GRANT CREATE WAREHOUSE ON ACCOUNT TO APPLICATION arena_app;
GRANT CREATE EXTERNAL ACCESS INTEGRATION ON ACCOUNT TO APPLICATION arena_app;
GRANT EXECUTE TASK ON ACCOUNT TO APPLICATION arena_app;
GRANT EXECUTE MANAGED TASK ON ACCOUNT TO APPLICATION arena_app;
```

The app also asks for two external access requests:

| Request | What it is for | Hosts |
|---|---|---|
| `HF_ACCESS` | Downloads base model weights from Hugging Face. | `huggingface.co` and nine Hugging Face download hosts, all on port 443 |
| `ARENA_ACCESS` | Checks the Arena licence and reports run start, heartbeat and end. | `arena.agilerl.com:443` |

The app reads only objects you bind to its references:

| Reference | Object | Privileges | Used for |
|---|---|---|---|
| `source_table` | Table, several allowed | `SELECT` | Training rows. |
| `source_view` | View, several allowed | `SELECT` | Training rows. Put custom SQL in a view and bind it here. |
| `reward_function` | One function | `USAGE` | Your own reward. The app calls it with SQL and never reads its code. |
| `export_stage` | One stage | `READ`, `WRITE` | Where `export_model` copies trained model files. |

## Configuration

Run the steps in this order.

### 1. Install, approve access and grant roles

Install the app from the listing and choose its name. Install creates a warehouse and two external access integrations whose names start with `AGILERL_APP_<APP>_`. It creates no compute pools.

Approve both external access requests on the app's page in Snowsight, or with SQL:

```sql
SHOW SPECIFICATIONS IN APPLICATION arena_app;

-- Use the sequence_number from the SHOW output.
ALTER APPLICATION arena_app APPROVE SPECIFICATION HF_ACCESS SEQUENCE_NUMBER = 1;
ALTER APPLICATION arena_app APPROVE SPECIFICATION ARENA_ACCESS SEQUENCE_NUMBER = 1;
```

The app has two application roles. `app_admin` includes everything `app_user` can do. Grant them to your account roles:

```sql
GRANT APPLICATION ROLE arena_app.app_admin TO ROLE ml_engineer;
GRANT APPLICATION ROLE arena_app.app_user TO ROLE ml_viewer;
```

### 2. Connect to Arena

Each run checks its licence with Arena, so the app needs an Arena key. The app gets the key itself and keeps it in its own secrets. Nobody sees or types it. As a role with `app_admin`:

```sql
CALL arena_app.app_code.connect_arena();
```

It returns a sign-in link. Open it within an hour, sign up or sign in, and create the Arena organisation for this Snowflake account. Then run `connect_arena()` again, or wait up to an hour and the app connects by itself. When it is connected, the call names your organisation, the trial end date and the GPU families Arena has a price for. Calling it again on a connected app changes nothing.

### 3. Create compute pools

List the GPU families the app can use in your region, then create the pools:

```sql
CALL arena_app.app_code.gpu_families();
CALL arena_app.app_code.set_pools(['GPU_NV_S'], 4);
```

`set_pools` creates a broker pool, a head pool and one GPU pool for each family you list. The second argument is the most GPUs each GPU pool can hold. A run uses whole nodes, so its GPU count must be a multiple of the family's GPUs per node, up to that number. Pass every family you want each time: the app drops its GPU pools for families you leave out. Pools start when a job starts and suspend after 5 idle minutes. A suspended pool has no Snowflake compute charge, but a month in which any pool runs counts as a month of use for the monthly fee (see Billing and trial).

GPU nodes come from Snowflake. When Snowflake has little GPU capacity in your region, they can take a long time to start. A run waits until they do: `status` shows its containers as PENDING, and training starts once every GPU node has joined.

### 4. Bind a table or view

Create a table of training rows. Snowflake stores unquoted column names in upper case, so this table has `QUESTION` and `ANSWER`:

```sql
CREATE TABLE my_db.public.math_questions (question STRING, answer STRING);
INSERT INTO my_db.public.math_questions VALUES
  ('What is 12 plus 30?', '42'),
  ('A box holds 6 eggs. How many eggs are in 7 boxes?', '42');
```

Bind it to `source_table`, as a role that holds `app_admin` and can grant `SELECT` on the table:

```sql
CALL arena_app.app_code.register_reference(
  'source_table', 'ADD',
  SYSTEM$REFERENCE('TABLE', 'my_db.public.math_questions', 'PERSISTENT', 'SELECT')
);

SHOW REFERENCES IN APPLICATION arena_app;  -- each binding's alias
```

For a view, use `'source_view'` and `'VIEW'`. Each run names the alias of the table or view it reads. To unbind, pass the alias: `CALL arena_app.app_code.register_reference('source_table', 'REMOVE', '<alias>');`. You can also bind objects in Snowsight, on the app's references.

To score with your own function, bind it to `reward_function`. It takes `prompt STRING, completion STRING, answer STRING` and returns `FLOAT`:

```sql
CALL arena_app.app_code.register_reference(
  'reward_function', 'ADD',
  SYSTEM$REFERENCE('FUNCTION', 'my_db.rewards.score_answer(STRING, STRING, STRING)', 'PERSISTENT', 'USAGE')
);
```

To keep trained models, bind a stage of yours to `export_stage`:

```sql
CREATE STAGE my_db.public.arena_models;
CALL arena_app.app_code.register_reference(
  'export_stage', 'ADD',
  SYSTEM$REFERENCE('STAGE', 'my_db.public.arena_models', 'PERSISTENT', 'READ', 'WRITE')
);
```

## Stored procedures

All procedures are in the app's `app_code` schema. Call them as `arena_app.app_code.<name>(...)`. Arguments with a default can be left out from the end.

### Setup

| Procedure | Role | What it does |
|---|---|---|
| `CONNECT_ARENA(base_url STRING DEFAULT 'https://arena.agilerl.com')` | `app_admin` | Connects the app to a new Arena organisation. Returns a sign-in link, then confirms once the app has its key. |
| `ROTATE_ARENA_KEY()` | `app_admin` | Swaps the app's Arena key for a new one. The app also does this itself once the key is 90 days old. |
| `CONFIGURE(org_key STRING, external_user_id STRING, base_url STRING DEFAULT 'https://arena.agilerl.com')` | `app_admin` | Saves an Arena key that AgileRL gave you directly, after checking it with Arena. Use `connect_arena` unless AgileRL told you otherwise. |
| `GPU_FAMILIES()` | `app_admin` | Lists the GPU families that your region has and that Arena has a price for. |
| `SET_POOLS(gpu_families ARRAY, max_gpus NUMBER DEFAULT NULL, broker_family STRING DEFAULT 'CPU_X64_L', head_family STRING DEFAULT 'CPU_X64_SL')` | `app_admin` | Creates or resizes the broker, head and GPU pools, and drops app GPU pools for families not listed. Refuses to drop or shrink a GPU pool that a queued or running run still needs, and a `max_gpus` that is not a multiple of a listed family's GPUs per node; nothing changes in either case. |
| `REGISTER_REFERENCE(ref_name STRING, operation STRING, ref_or_alias STRING)` | `app_admin` | Binds (`ADD`), unbinds (`REMOVE`) or clears (`CLEAR`) an object on `source_table`, `source_view`, `reward_function` or `export_stage`. |

### Training

| Procedure | Role | What it does |
|---|---|---|
| `SUBMIT(manifest STRING, gpus NUMBER, source_reference STRING DEFAULT NULL, source_alias STRING DEFAULT NULL, gpu_family STRING DEFAULT NULL, reward_file STRING DEFAULT NULL, checkpoint_steps NUMBER DEFAULT NULL, resume_from STRING DEFAULT NULL)` | `app_admin` | Starts a training run from a YAML manifest on the bound table or view. Returns the run ID. Leave `reward_file` as `NULL`: set the reward in the manifest. Before it starts anything, it checks that the alias is bound, that the app has a pool for the GPU family and that the pool can run that many GPUs. If a check fails, or the manifest has a missing, misspelled or wrong-typed field, it names the field, says what to change and starts nothing. |
| `STATUS(run_id STRING)` | `app_user` | Shows the state of the run's services and the run: `running`, `succeeded`, `failed`, `cancelling`, `cancelled` or `gone`. `cancel` drops the run's services, so a run you cancel reads `gone`. |
| `LOGS(run_id STRING, role STRING DEFAULT 'head', instance NUMBER DEFAULT 0)` | `app_user` | Shows the log of one service: `head`, `broker` or `worker`, at INFO level and above. While its GPU node is still starting, it says the container has not started yet. Logs of a finished job need an active event table in your account. |
| `CANCEL(run_id STRING)` | `app_admin` | Stops a run and suspends GPU pools left with nothing on them. |

### Models and serving

| Procedure | Role | What it does |
|---|---|---|
| `PUBLISH(run_id STRING, model_name STRING, version_name STRING DEFAULT NULL, step NUMBER DEFAULT NULL, serve BOOLEAN DEFAULT FALSE, gpu_count NUMBER DEFAULT 1, service_name STRING DEFAULT NULL, max_model_len NUMBER DEFAULT NULL)` | `app_admin` | Logs a run's checkpoint to the Model Registry inside the app. With `serve` set to `TRUE`, also starts an inference service and its chat function. It waits for the job, which can take 20 minutes or more. |
| `ENDPOINT(model STRING, version_name STRING DEFAULT NULL)` | `app_user` | Names the chat function of a served model and gives an example call. |
| `UNPUBLISH(model STRING, version_name STRING DEFAULT NULL, drop_version BOOLEAN DEFAULT FALSE)` | `app_admin` | Drops the inference service and its chat function, which stops serving charges. With `drop_version` set to `TRUE`, also deletes the registry version. |
| `EXPORT_MODEL(run_id STRING, step NUMBER DEFAULT NULL)` | `app_admin` | Copies a run's newest checkpoint, or the one at `step`, to `<run_id>/checkpoint_<step>/` on your `export_stage`. Arena charges nothing for it, and it works after the trial ends. |

### Billing

| Procedure | Role | What it does |
|---|---|---|
| `RECONCILE_NOW()` | `app_admin` | Runs the hourly billing task now, after any run of it in progress. Its summary is logged at INFO level. |

The app's hourly task bills usage by itself. Its procedures cannot be called by any role.

### Chat function of a served model

`publish` with `serve` set to `TRUE` creates one SQL function per served model, in the `app_state` schema:

```text
arena_app.app_state.<service>_CHAT(messages ARRAY, temperature FLOAT, max_completion_tokens INT) RETURNS OBJECT
```

The service is named `<model>_<version>` unless you pass `service_name`. Roles with `app_user` can call it. `messages` are chat messages in the OpenAI format. The result is an OpenAI chat completion, so the reply text is at `choices[0]:message:content`. `unpublish` drops the function.

For the other chat options, call the service's own method, which takes ten arguments in this order: messages, temperature, max_completion_tokens, stop, n, stream, top_p, frequency_penalty, presence_penalty and response_format:

```sql
SELECT arena_app.app_state.math_agent_v1!__CALL__(
  PARSE_JSON('[{"role": "user", "content": "Hello"}]')::ARRAY, 0.7, 256, NULL, 1, FALSE, 1.0, 0.0, 0.0, NULL
);
```

### Tables you can read

| Table | Role | Contents |
|---|---|---|
| `app_state.arena_metrics` | `app_user` | Training metrics for each run. |
| `app_state.usage_items` | `app_admin` | Runs and served models the app bills. `unbilled_reason` says why a run was not charged. |
| `app_state.billed_hours` | `app_admin` | One row per billed usage hour. |
| `app_state.served_models` | `app_admin` | What each inference service serves. |

### Built-in rewards

Set the reward in the manifest's `environment.reward`. The app has five built-in rewards. Each scores the model's final answer 1 or 0:

| Type | Scores 1 when |
|---|---|
| `exact_match` | The answer equals the row's `answer`. Ignores case and extra spaces unless `normalize: false`. |
| `numeric` | The answer is a number within `tolerance` of the row's `answer`. |
| `regex` | The reply matches `pattern`. |
| `json_schema` | The first JSON object in the reply is valid against `schema`. |
| `tool_call` | The first tool call in the reply uses the expected tool and arguments. Use it to train agents. |

`exact_match` and `numeric` take an optional `answer_pattern` whose first capturing group is the answer, such as `"<answer>(.*)</answer>"`. For your own scoring, bind a function to `reward_function` and set `reward: {type: snowflake_function}`. Details and more examples: https://docs.arena.agilerl.com/partners/snowflake-native-app-rewards/

## Example: from training to a served model

### Post-train with CISPO and a built-in reward

CISPO is the recommended default. For GRPO, change `name: CISPO` to `name: GRPO`; the rest of the manifest stays the same. Each answer scores 1 when its number matches the row's `ANSWER`:

```sql
CALL arena_app.app_code.submit(
  '
algorithm:
  name: CISPO
  batch_size: 8
  group_size: 4
environment:
  env_type: rollout
  columns:
    question: QUESTION
    answer: ANSWER
  prompt_template:
    system_0: Think step by step, then give the final number inside <answer></answer>.
    user_1: "{question}"
  reward:
    type: numeric
    answer_pattern: "<answer>(.*)</answer>"
  max_reward: 1.0
network:
  pretrained_model_name_or_path: Qwen/Qwen3.5-0.8B
  max_context_length: 2048
training:
  max_steps: 200
',
  2,                -- GPUs
  'source_table',
  '<alias>',        -- from SHOW REFERENCES
  'GPU_NV_S'
);
-- Returns the run ID. Keep it for the calls below.
```

Reinforcement learning (GRPO, CISPO) needs at least 2 GPUs: one trains while another generates answers. OPD with a separate teacher model needs a GPU family with at least 2 GPUs per node. `submit` refuses a smaller request and says what to ask for.

`columns` maps the names the trainer expects (left) to your column names (right). For SFT, use `name: SFT`, `env_type: dataset`, `objective: sft` and map `prompt` and `target` columns.

### Watch a run

```sql
CALL arena_app.app_code.status('<run_id>');
CALL arena_app.app_code.logs('<run_id>');

SELECT seq, time, metrics
FROM arena_app.app_state.arena_metrics
WHERE run_id = '<run_id>'
ORDER BY seq;

-- To stop it early:
CALL arena_app.app_code.cancel('<run_id>');
```

A run whose head or worker service is gone before the run finished reads `failed`: it cannot make progress. Cancel it to free the nodes its other services still hold.

### Publish, serve and call the model

```sql
CALL arena_app.app_code.publish('<run_id>', 'math_agent', 'v1', NULL, TRUE);
CALL arena_app.app_code.endpoint('math_agent');  -- the function and an example call

SELECT arena_app.app_state.math_agent_v1_chat(
  PARSE_JSON('[{"role": "user", "content": "What is 17 times 23?"}]'),
  0.7,
  256
):choices[0]:message:content::STRING AS reply;
```

Snowflake suspends a service that gets no calls. The next call waits while it starts again, which can take a few minutes.

### Stop serving

```sql
CALL arena_app.app_code.unpublish('math_agent', 'v1');
```

### Export a model

```sql
CALL arena_app.app_code.export_model('<run_id>');
CALL arena_app.app_code.export_model('<run_id>', 200);  -- the checkpoint at step 200
```

The files are the Hugging Face model files that `publish` logs.

## Billing and trial

The listing has one usage-based plan, and Snowflake bills both parts:

- A monthly fee of $12,500, charged only in calendar months you use the app. Any query that references the app counts as use, including a call to `status`. So does a month in which any of the app's compute pools runs, even briefly before it auto-suspends.
- $0.01 for each usage unit your training and serving jobs consume. One usage unit is 1 cent of job cost. Job cost is GPUs × hours × the GPU family's Arena price per GPU-hour.

Training counts from the start of the first GPU node to the end of the run. A run that ends before its first training step, for example because it failed while starting up, is not charged. A run that fails partway through training is charged up to the failure.

Snowflake also charges you as usual for the app's compute pools, warehouse, storage and serverless task.

The first 7 days after install are a free trial. AgileRL charges nothing, and you pay Snowflake only for your own compute. Running jobs and served models stop shortly before the 7 days end. After that, `submit` and `publish` refuse to start anything until you purchase the listing. Reinstalling does not restart the trial.

`app_admin` can audit billed usage with `SELECT * FROM arena_app.app_state.billed_hours;`.

## Data sent to AgileRL

The app sends AgileRL only the metadata needed to connect the app and to record and bill usage:

- When you connect: your account locator, region, account and organisation names, the app's name, whether you have bought the listing, and the app version.
- For each run and each served model: your account locator and region, whether you have bought the listing, the run ID, the GPU count and family, and the start and end times. Each record is labelled as coming from the Snowflake Native App.
- For a served model, also: its deployment ID, model name and version name.

No rows, prompts, metrics or model weights leave your account. The only other connection out is to Hugging Face, to download base model weights.

## Uninstall

Uninstalling deletes everything the app holds: checkpoints, metrics and published models. Copy out what you want to keep first:

```sql
CALL arena_app.app_code.export_model('<run_id>');

CREATE TABLE my_db.public.arena_metrics_copy AS
  SELECT * FROM arena_app.app_state.arena_metrics;
```

Then drop the app as the role that owns it:

```sql
DROP APPLICATION arena_app CASCADE;
```

This drops the app's compute pools, services, warehouse and both external access integrations.

## Support

Email arena@agilerl.com for help with setup, runs, pricing or billing. Include your account locator and the run ID if you have one.
