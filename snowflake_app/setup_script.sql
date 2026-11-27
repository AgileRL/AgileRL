-- Application roles, and the external hosts the consumer approves at install.
CREATE APPLICATION ROLE IF NOT EXISTS app_admin;
CREATE APPLICATION ROLE IF NOT EXISTS app_user;
GRANT APPLICATION ROLE app_user TO APPLICATION ROLE app_admin;

CREATE SCHEMA IF NOT EXISTS app_state;

CREATE OR ALTER NETWORK RULE app_state.hf_egress
  TYPE = HOST_PORT
  MODE = EGRESS
  VALUE_LIST = (
    'huggingface.co:443',
    'cas-server.xethub.hf.co:443',
    'cas-server.xethub-eu.hf.co:443',
    'transfer.xethub.hf.co:443',
    'transfer.xethub-eu.hf.co:443',
    'us.aws.cdn.hf.co:443',
    'us.gcp.cdn.hf.co:443',
    'cdn-lfs.hf.co:443',
    'cdn-lfs-us-1.hf.co:443',
    'cdn-lfs-eu-1.hf.co:443'
  );

CREATE OR ALTER NETWORK RULE app_state.arena_api_egress
  TYPE = HOST_PORT
  MODE = EGRESS
  VALUE_LIST = ('arena.agilerl.com:443');

-- app_code.configure or app_code.connect_arena sets the values. reconcile_usage names the secrets, so they exist from install.
CREATE SECRET IF NOT EXISTS app_state.arena_org_key
  TYPE = GENERIC_STRING
  SECRET_STRING = '';
CREATE SECRET IF NOT EXISTS app_state.arena_external_user_id
  TYPE = GENERIC_STRING
  SECRET_STRING = '';
CREATE SECRET IF NOT EXISTS app_state.arena_base_url
  TYPE = GENERIC_STRING
  SECRET_STRING = '';
-- Secret of the claim app_code.connect_arena has open; blank otherwise.
CREATE SECRET IF NOT EXISTS app_state.arena_claim_secret
  TYPE = GENERIC_STRING
  SECRET_STRING = '';

CREATE OR ALTER VERSIONED SCHEMA app_code;
GRANT USAGE ON SCHEMA app_code TO APPLICATION ROLE app_user;
-- Direct as well as through app_user: app_admin holds the reference callback.
GRANT USAGE ON SCHEMA app_code TO APPLICATION ROLE app_admin;

-- Account-level objects are named per app, so installs in one account do not collide.
-- The job code takes only unquoted identifiers, so a quoted app name maps to A-Z, 0-9 and _.
CREATE OR REPLACE FUNCTION app_code.object_prefix()
  RETURNS STRING
  AS
$$
  'AGILERL_APP_' || REGEXP_REPLACE(UPPER(CURRENT_DATABASE()), '[^A-Z0-9_]', '_') || '_'
$$;

-- Setup reruns on every upgrade, so an integration the app already owns is kept;
-- one it does not own fails setup, because an existing integration keeps its own network rules.
EXECUTE IMMEDIATE $$
DECLARE
  hf_integration STRING;
  arena_integration STRING;
  trial_account EXCEPTION (-20509, 'Snowpark Container Services and external access integrations are not available on trial accounts (Snowflake error 509009). Add a payment method to the account, then install the app again.');
  name_taken EXCEPTION (-20510, 'An integration named AGILERL_APP_<app name>_HF_EAI or AGILERL_APP_<app name>_ARENA_EAI already exists in this account and the app does not own it. Rename or drop it, or install the app under another name.');
BEGIN
  SELECT app_code.object_prefix() || 'HF_EAI', app_code.object_prefix() || 'ARENA_EAI'
    INTO :hf_integration, :arena_integration;
  BEGIN
    CREATE EXTERNAL ACCESS INTEGRATION IDENTIFIER(:hf_integration)
      ALLOWED_NETWORK_RULES = (app_state.hf_egress)
      ENABLED = TRUE;
  EXCEPTION
    WHEN OTHER THEN
      IF (SQLCODE = 509009) THEN
        RAISE trial_account;
      END IF;
      IF (SQLCODE != 2002) THEN
        RAISE;
      END IF;
      BEGIN
        DESCRIBE INTEGRATION IDENTIFIER(:hf_integration);
      EXCEPTION
        WHEN OTHER THEN
          RAISE name_taken;
      END;
  END;
  BEGIN
    CREATE EXTERNAL ACCESS INTEGRATION IDENTIFIER(:arena_integration)
      ALLOWED_NETWORK_RULES = (app_state.arena_api_egress)
      ENABLED = TRUE;
  EXCEPTION
    WHEN OTHER THEN
      IF (SQLCODE != 2002) THEN
        RAISE;
      END IF;
      BEGIN
        DESCRIBE INTEGRATION IDENTIFIER(:arena_integration);
      EXCEPTION
        WHEN OTHER THEN
          RAISE name_taken;
      END;
  END;
  -- A procedure may read only the secrets its integration allows.
  ALTER EXTERNAL ACCESS INTEGRATION IDENTIFIER(:arena_integration) SET
    ALLOWED_AUTHENTICATION_SECRETS = (
      app_state.arena_org_key,
      app_state.arena_external_user_id,
      app_state.arena_base_url,
      app_state.arena_claim_secret
    );
END;
$$;

ALTER APPLICATION SET SPECIFICATION hf_access
  TYPE = EXTERNAL_ACCESS
  LABEL = 'Hugging Face'
  DESCRIPTION = 'Downloads base model weights from Hugging Face'
  HOST_PORTS = (
    'huggingface.co:443',
    'cas-server.xethub.hf.co:443',
    'cas-server.xethub-eu.hf.co:443',
    'transfer.xethub.hf.co:443',
    'transfer.xethub-eu.hf.co:443',
    'us.aws.cdn.hf.co:443',
    'us.gcp.cdn.hf.co:443',
    'cdn-lfs.hf.co:443',
    'cdn-lfs-us-1.hf.co:443',
    'cdn-lfs-eu-1.hf.co:443'
  );

ALTER APPLICATION SET SPECIFICATION arena_access
  TYPE = EXTERNAL_ACCESS
  LABEL = 'Arena'
  DESCRIPTION = 'Checks the Arena license and reports run start, heartbeat and end'
  HOST_PORTS = ('arena.agilerl.com:443');

-- Run stage, metrics table, and file format the job services write to.
CREATE STAGE IF NOT EXISTS app_state.runs
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

CREATE FILE FORMAT IF NOT EXISTS app_state.arena_json
  TYPE = JSON
  STRIP_OUTER_ARRAY = FALSE;

CREATE TABLE IF NOT EXISTS app_state.arena_metrics (
  RUN_ID STRING,
  SEQ NUMBER,
  TIME TIMESTAMP_TZ,
  METRICS VARIANT
);

-- Runs and deployments app_code.reconcile_usage bills until each settles.
CREATE TABLE IF NOT EXISTS app_state.usage_items (
  kind STRING,
  usage_id STRING,
  opened_at TIMESTAMP_TZ,
  settled_at TIMESTAMP_TZ
);
-- Added by ALTER so an upgrade adds them to a table an earlier version created.
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMP_TZ;
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMP_TZ;
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS first_404_at TIMESTAMP_TZ;
-- Service of a deployment item in app_state; NULL for runs.
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS service_name STRING;
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS closed_at TIMESTAMP_TZ;
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS last_error STRING;
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS last_error_at TIMESTAMP_TZ;
-- Why a run settled without billing, such as ending before training started.
ALTER TABLE app_state.usage_items ADD COLUMN IF NOT EXISTS unbilled_reason STRING;

-- What each service app_code.publish started serves, recorded before the service exists.
CREATE TABLE IF NOT EXISTS app_state.served_models (
  service_name STRING,
  model STRING,
  version STRING,
  run_id STRING,
  gpu_family STRING,
  gpu_count NUMBER
);

-- One row per charged usage hour; Snowflake does not enforce the key.
CREATE TABLE IF NOT EXISTS app_state.billed_hours (
  kind STRING,
  usage_id STRING,
  hour_start_ms NUMBER,
  billable_cents NUMBER,
  claim_id STRING,
  attempted_at TIMESTAMP_TZ,
  emitted_at TIMESTAMP_TZ,
  PRIMARY KEY (kind, usage_id, hour_start_ms)
);

-- Where the saved org key came from: 'connect' keys are rotated by the app, 'configure' keys by AgileRL.
CREATE TABLE IF NOT EXISTS app_state.arena_key (
  source STRING,
  issued_at TIMESTAMP_TZ
);

-- The claim app_code.connect_arena has open, at most one row.
-- A row is inserted only while the table is empty, before Arena is called.
CREATE TABLE IF NOT EXISTS app_state.arena_claim (
  claim_id STRING,
  base_url STRING,
  expires_at TIMESTAMP_TZ,
  proven_at TIMESTAMP_TZ
);

-- When the app was installed; app_code.submit and publish refuse an unpurchased listing install once its trial ends.
CREATE TABLE IF NOT EXISTS app_state.trial (
  started_at TIMESTAMP_TZ
);
INSERT INTO app_state.trial (started_at)
  SELECT CURRENT_TIMESTAMP() WHERE NOT EXISTS (SELECT 1 FROM app_state.trial);

-- Read-only grants: app_user reads metrics; app_admin, which holds app_user, also audits billing.
GRANT USAGE ON SCHEMA app_state TO APPLICATION ROLE app_user;
GRANT SELECT ON TABLE app_state.arena_metrics TO APPLICATION ROLE app_user;
GRANT SELECT ON TABLE app_state.usage_items TO APPLICATION ROLE app_admin;
GRANT SELECT ON TABLE app_state.billed_hours TO APPLICATION ROLE app_admin;
GRANT SELECT ON TABLE app_state.served_models TO APPLICATION ROLE app_admin;

-- Compute pools are created by app_code.set_pools, not here: an idle pool bills.
EXECUTE IMMEDIATE $$
DECLARE
  warehouse STRING;
BEGIN
  SELECT app_code.object_prefix() || 'WH' INTO :warehouse;
  CREATE WAREHOUSE IF NOT EXISTS IDENTIFIER(:warehouse)
    WAREHOUSE_SIZE = XSMALL
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;
END;
$$;

-- Called straight from an app procedure or task, SYSTEM$IS_LISTING_PURCHASED is TRUE on a trial
-- install too; only a secure function or view returns FALSE until the consumer buys.
-- An install not from a listing gets TRUE.
CREATE OR REPLACE SECURE FUNCTION app_code.listing_purchased()
  RETURNS BOOLEAN
  AS 'SYSTEM$IS_LISTING_PURCHASED()';

CREATE OR REPLACE PROCEDURE app_code.register_reference(
    ref_name STRING,
    operation STRING,
    ref_or_alias STRING
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.register_reference';
GRANT USAGE ON PROCEDURE app_code.register_reference(STRING, STRING, STRING)
  TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE app_code.configure(
    org_key STRING,
    external_user_id STRING,
    base_url STRING DEFAULT 'https://arena.agilerl.com'
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.configure';
GRANT USAGE ON PROCEDURE app_code.configure(STRING, STRING, STRING)
  TO APPLICATION ROLE app_admin;

-- Connects the app to a new Arena organisation; Arena delivers the key to the app, never to a person.
CREATE OR REPLACE PROCEDURE app_code.connect_arena(base_url STRING DEFAULT 'https://arena.agilerl.com')
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'app_keys.connect_arena';
GRANT USAGE ON PROCEDURE app_code.connect_arena(STRING)
  TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE app_code.rotate_arena_key()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'app_keys.rotate_arena_key';
GRANT USAGE ON PROCEDURE app_code.rotate_arena_key()
  TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE app_code.set_pools(
    gpu_families ARRAY,
    max_gpus NUMBER DEFAULT NULL,
    broker_family STRING DEFAULT 'CPU_X64_L',
    head_family STRING DEFAULT 'CPU_X64_SL'
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.set_pools';
GRANT USAGE ON PROCEDURE app_code.set_pools(ARRAY, NUMBER, STRING, STRING)
  TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE app_code.gpu_families()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.gpu_families';
GRANT USAGE ON PROCEDURE app_code.gpu_families()
  TO APPLICATION ROLE app_admin;

-- Run and publish procedures create services in app_state: a versioned schema cannot hold one.
CREATE OR REPLACE PROCEDURE app_code.submit(
    manifest STRING,
    gpus NUMBER,
    source_reference STRING DEFAULT NULL,
    source_alias STRING DEFAULT NULL,
    gpu_family STRING DEFAULT NULL,
    reward_file STRING DEFAULT NULL,
    checkpoint_steps NUMBER DEFAULT NULL,
    resume_from STRING DEFAULT NULL
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.submit';
GRANT USAGE ON PROCEDURE app_code.submit(STRING, NUMBER, STRING, STRING, STRING, STRING, NUMBER, STRING)
  TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE app_code.status(run_id STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.status';
GRANT USAGE ON PROCEDURE app_code.status(STRING)
  TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_code.logs(
    run_id STRING,
    role STRING DEFAULT 'head',
    instance NUMBER DEFAULT 0
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.logs';
GRANT USAGE ON PROCEDURE app_code.logs(STRING, STRING, NUMBER)
  TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_code.cancel(run_id STRING)
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.cancel';
GRANT USAGE ON PROCEDURE app_code.cancel(STRING)
  TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE app_code.publish(
    run_id STRING,
    model_name STRING,
    version_name STRING DEFAULT NULL,
    step NUMBER DEFAULT NULL,
    serve BOOLEAN DEFAULT FALSE,
    gpu_count NUMBER DEFAULT 1,
    service_name STRING DEFAULT NULL,
    max_model_len NUMBER DEFAULT NULL
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.publish';
GRANT USAGE ON PROCEDURE app_code.publish(STRING, STRING, STRING, NUMBER, BOOLEAN, NUMBER, STRING, NUMBER)
  TO APPLICATION ROLE app_admin;

CREATE OR REPLACE PROCEDURE app_code.endpoint(
    model STRING,
    version_name STRING DEFAULT NULL
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.endpoint';
GRANT USAGE ON PROCEDURE app_code.endpoint(STRING, STRING)
  TO APPLICATION ROLE app_user;

CREATE OR REPLACE PROCEDURE app_code.unpublish(
    model STRING,
    version_name STRING DEFAULT NULL,
    drop_version BOOLEAN DEFAULT FALSE
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.unpublish';
GRANT USAGE ON PROCEDURE app_code.unpublish(STRING, STRING, BOOLEAN)
  TO APPLICATION ROLE app_admin;

-- Copies a run's model to the consumer's export_stage. No services and no usage items, so it is never billed.
CREATE OR REPLACE PROCEDURE app_code.export_model(
    run_id STRING,
    step NUMBER DEFAULT NULL
  )
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.export_model';
GRANT USAGE ON PROCEDURE app_code.export_model(STRING, NUMBER)
  TO APPLICATION ROLE app_admin;

-- Granted to no role: only the hourly task calls it, so two calls never overlap.
CREATE OR REPLACE PROCEDURE app_code.reconcile_usage()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.reconcile_usage';

-- The Arena integration is named per app, so the Arena procedures get it and their secrets here.
EXECUTE IMMEDIATE $$
DECLARE
  arena_access STRING;
  secrets STRING DEFAULT ' SECRETS = (''org_key'' = app_state.arena_org_key, ''external_user_id'' = app_state.arena_external_user_id, ''base_url'' = app_state.arena_base_url)';
  claim_secrets STRING DEFAULT ' SECRETS = (''org_key'' = app_state.arena_org_key, ''external_user_id'' = app_state.arena_external_user_id, ''base_url'' = app_state.arena_base_url, ''claim_secret'' = app_state.arena_claim_secret)';
BEGIN
  SELECT ' SET EXTERNAL_ACCESS_INTEGRATIONS = (' || app_code.object_prefix() || 'ARENA_EAI)'
    INTO :arena_access;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.configure(STRING, STRING, STRING)' || arena_access;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.connect_arena(STRING)' || arena_access || claim_secrets;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.rotate_arena_key()' || arena_access || secrets;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.gpu_families()' || arena_access || secrets;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.submit(STRING, NUMBER, STRING, STRING, STRING, STRING, NUMBER, STRING)' || arena_access || secrets;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.publish(STRING, STRING, STRING, NUMBER, BOOLEAN, NUMBER, STRING, NUMBER)' || arena_access || secrets;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.unpublish(STRING, STRING, BOOLEAN)' || arena_access || secrets;
  EXECUTE IMMEDIATE 'ALTER PROCEDURE app_code.reconcile_usage()' || arena_access || claim_secrets;
END;
$$;

-- Serverless, so no warehouse resumes; a failed run must not suspend billing.
-- Suspended first because an upgrade may change the definition of a started task.
-- Left suspended: version_init, submit and publish resume it only while there is work to bill.
ALTER TASK IF EXISTS app_state.reconcile_usage_task SUSPEND;
CREATE OR ALTER TASK app_state.reconcile_usage_task
  USER_TASK_MANAGED_INITIAL_WAREHOUSE_SIZE = 'XSMALL'
  SCHEDULE = '60 MINUTES'
  SUSPEND_TASK_AFTER_NUM_FAILURES = 0
  AS CALL app_code.reconcile_usage();

-- manifest.yml's version_initializer.
CREATE OR REPLACE PROCEDURE app_code.version_init()
  RETURNS STRING
  LANGUAGE PYTHON
  RUNTIME_VERSION = '3.13'
  PACKAGES = (
    'snowflake-snowpark-python', 'aiofiles', 'click', 'cryptography', 'deprecation',
    'httpx', 'jwcrypto', 'lazy_loader', 'numpy', 'pydantic', 'pyyaml', 'requests',
    'requests-toolbelt', 'rich'
  )
  IMPORTS = ('/app_code/arena_app.py', '/app_code/app_billing.py', '/app_code/app_keys.py', '/app_code/app_trial.py', '/agilerl_arena.zip')
  HANDLER = 'arena_app.version_init';

-- Runs the task rather than reconcile_usage: a task run never overlaps another, and
-- two overlapping calls could each open an Arena deployment for one service.
CREATE OR REPLACE PROCEDURE app_code.reconcile_now()
  RETURNS STRING
  LANGUAGE SQL
  AS
$$
BEGIN
  EXECUTE TASK app_state.reconcile_usage_task;
  RETURN 'Billing run queued after any run in progress. Its summary is logged at INFO.';
END;
$$;
GRANT USAGE ON PROCEDURE app_code.reconcile_now()
  TO APPLICATION ROLE app_admin;
