-- Arena in a Snowflake account.
-- Run as ACCOUNTADMIN. Replace the placeholders before running:
-- the image path, the Pulsar host, the service user's public key, and the credit quota.
-- This script is not executed in CI.

USE ROLE ACCOUNTADMIN;

CREATE ROLE IF NOT EXISTS arena_trainer;

CREATE WAREHOUSE IF NOT EXISTS arena_wh
  WAREHOUSE_SIZE = XSMALL
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- Change CREDIT_QUOTA to the monthly budget for this warehouse.
CREATE RESOURCE MONITOR IF NOT EXISTS arena_monitor
  WITH CREDIT_QUOTA = 100
  FREQUENCY = MONTHLY
  START_TIMESTAMP = IMMEDIATELY
  TRIGGERS
    ON 80 PERCENT DO NOTIFY
    ON 100 PERCENT DO SUSPEND;

ALTER WAREHOUSE arena_wh SET RESOURCE_MONITOR = arena_monitor;

-- Small pool. Raise MAX_NODES for a multi-node run.
CREATE COMPUTE POOL IF NOT EXISTS arena_gpu_pool
  MIN_NODES = 1
  MAX_NODES = 2
  INSTANCE_FAMILY = GPU_NV_S
  AUTO_RESUME = TRUE
  AUTO_SUSPEND_SECS = 300;

CREATE DATABASE IF NOT EXISTS arena;
CREATE SCHEMA IF NOT EXISTS arena.training;

CREATE IMAGE REPOSITORY IF NOT EXISTS arena.training.images;

CREATE STAGE IF NOT EXISTS arena.training.payload
  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

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

-- Hugging Face model download, including the CDN hosts the hub redirects to.
CREATE OR REPLACE NETWORK RULE arena.training.hf_egress
  MODE = EGRESS
  TYPE = HOST_PORT
  VALUE_LIST = (
    'huggingface.co:443',
    'www.huggingface.co:443',
    'cdn-lfs.huggingface.co:443',
    'cdn-lfs-us-1.huggingface.co:443',
    'cas-bridge.xethub.hf.co:443'
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

-- After the runtime image is in arena.training.images, register it.
-- Replace the image path. BASE_IMAGE_TYPE must be GPU.
-- CREATE CUSTOM RUNTIME ENVIRONMENT arena_runtime
--   IMAGE_PATH = '/arena/training/images/arena-snowflake:latest'
--   BASE_IMAGE_TYPE = GPU;

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
GRANT READ, WRITE ON STAGE arena.training.payload TO ROLE arena_trainer;
GRANT READ, WRITE ON STAGE arena.training.runs TO ROLE arena_trainer;
GRANT SELECT, INSERT, UPDATE ON TABLE arena.training.arena_metrics TO ROLE arena_trainer;
GRANT USAGE ON FILE FORMAT arena.training.arena_json TO ROLE arena_trainer;
GRANT READ, WRITE ON IMAGE REPOSITORY arena.training.images TO ROLE arena_trainer;
GRANT USAGE, MONITOR ON COMPUTE POOL arena_gpu_pool TO ROLE arena_trainer;
GRANT USAGE ON INTEGRATION arena_hf_eai TO ROLE arena_trainer;
GRANT USAGE ON INTEGRATION arena_pulsar_eai TO ROLE arena_trainer;
GRANT CREATE CUSTOM RUNTIME ENVIRONMENT ON ACCOUNT TO ROLE arena_trainer;
