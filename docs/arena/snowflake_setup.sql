-- Arena in a Snowflake account.
-- Run as ACCOUNTADMIN. Replace the placeholders before running:
-- the Pulsar host, the service user's public key, and the credit quota.
-- Push the trainer image into arena.training.images with a role that has WRITE.
-- The trainer role below only receives READ, which is what the job needs.
-- This script is not executed in CI.

USE ROLE ACCOUNTADMIN;

CREATE ROLE IF NOT EXISTS arena_trainer;

CREATE WAREHOUSE IF NOT EXISTS arena_wh
  WAREHOUSE_SIZE = XSMALL
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- Change CREDIT_QUOTA to the monthly budget for this warehouse.
-- Resource monitors cover warehouses only, not compute pools. The budget
-- below tracks the GPU pool.
CREATE RESOURCE MONITOR IF NOT EXISTS arena_monitor
  WITH CREDIT_QUOTA = 100
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON 80 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND;

ALTER WAREHOUSE arena_wh SET RESOURCE_MONITOR = arena_monitor;

-- Small pool. Raise MAX_NODES to at least the largest --nodes value you submit.
CREATE COMPUTE POOL IF NOT EXISTS arena_gpu_pool
  MIN_NODES = 1
  MAX_NODES = 2
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300
  INITIALLY_SUSPENDED = TRUE;

CREATE DATABASE IF NOT EXISTS arena;
CREATE SCHEMA IF NOT EXISTS arena.training;

-- Monthly credit budget for the GPU pool. A budget sends notifications; it
-- does not suspend the pool. Change the limit, and add a notification email
-- with SET_EMAIL_NOTIFICATIONS.
CREATE SNOWFLAKE.CORE.BUDGET IF NOT EXISTS arena.training.arena_gpu_budget();
CALL arena.training.arena_gpu_budget!SET_SPENDING_LIMIT(500);
CALL arena.training.arena_gpu_budget!ADD_RESOURCE(
  SELECT SYSTEM$REFERENCE('COMPUTE POOL', 'arena_gpu_pool', 'SESSION', 'applybudget'));

CREATE IMAGE REPOSITORY IF NOT EXISTS arena.training.images;

CREATE STAGE IF NOT EXISTS arena.training.runs
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

CREATE FILE FORMAT IF NOT EXISTS arena.training.arena_json
  TYPE = JSON
  STRIP_OUTER_ARRAY = FALSE;

CREATE TABLE IF NOT EXISTS arena.training.arena_metrics (
  RUN_ID STRING,
  SEQ NUMBER,
  TIME TIMESTAMP_TZ,
  METRICS VARIANT
);

-- Hugging Face model download and the storage and CDN hosts it redirects to.
-- Hugging Face publishes the current list at https://huggingface.co/.well-known/meta.json.
CREATE OR REPLACE NETWORK RULE arena.training.hf_egress
  MODE = EGRESS
  TYPE = HOST_PORT
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

-- Replace broker-host with the host of the Pulsar URL passed to arena snowflake submit.
CREATE OR REPLACE NETWORK RULE arena.training.pulsar_egress
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = ('broker-host:6650');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION arena_hf_eai
  ALLOWED_NETWORK_RULES = (arena.training.hf_egress)
  ENABLED = TRUE;

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION arena_pulsar_eai
  ALLOWED_NETWORK_RULES = (arena.training.pulsar_egress)
  ENABLED = TRUE;

-- RL only. Uncomment and set the environment host the job calls.
-- CREATE OR REPLACE NETWORK RULE arena.training.env_egress
--   MODE = EGRESS
--   TYPE = HOST_PORT
--   VALUE_LIST = ('env.example.com:443');
-- CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION arena_env_eai
--   ALLOWED_NETWORK_RULES = (arena.training.env_egress)
--   ENABLED = TRUE;
-- GRANT USAGE ON INTEGRATION arena_env_eai TO ROLE arena_trainer;

-- Paste the public key body: one line, no header, no footer.
CREATE USER IF NOT EXISTS arena_job_user
  TYPE = SERVICE
  DEFAULT_ROLE = arena_trainer
  DEFAULT_WAREHOUSE = arena_wh
  RSA_PUBLIC_KEY = '<paste public key>';

GRANT ROLE arena_trainer TO USER arena_job_user;
GRANT USAGE ON WAREHOUSE arena_wh TO ROLE arena_trainer;
GRANT USAGE ON DATABASE arena TO ROLE arena_trainer;
GRANT USAGE ON SCHEMA arena.training TO ROLE arena_trainer;
GRANT CREATE SERVICE ON SCHEMA arena.training TO ROLE arena_trainer;
-- Publishing logs the model to the registry and serves it on a public endpoint.
GRANT CREATE MODEL ON SCHEMA arena.training TO ROLE arena_trainer;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE arena_trainer;
GRANT READ, WRITE ON STAGE arena.training.runs TO ROLE arena_trainer;
GRANT SELECT, INSERT, UPDATE ON TABLE arena.training.arena_metrics TO ROLE arena_trainer;
GRANT USAGE ON FILE FORMAT arena.training.arena_json TO ROLE arena_trainer;
GRANT READ ON IMAGE REPOSITORY arena.training.images TO ROLE arena_trainer;
GRANT USAGE ON COMPUTE POOL arena_gpu_pool TO ROLE arena_trainer;
GRANT MONITOR ON COMPUTE POOL arena_gpu_pool TO ROLE arena_trainer;
GRANT USAGE ON INTEGRATION arena_hf_eai TO ROLE arena_trainer;
GRANT USAGE ON INTEGRATION arena_pulsar_eai TO ROLE arena_trainer;

-- Training source. Uncomment and name the database, schema, and table or view
-- passed as --source-table, or every object a --source-query reads.
-- GRANT USAGE ON DATABASE source_db TO ROLE arena_trainer;
-- GRANT USAGE ON SCHEMA source_db.source_schema TO ROLE arena_trainer;
-- GRANT SELECT ON TABLE source_db.source_schema.prompts TO ROLE arena_trainer;
