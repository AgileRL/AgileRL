-- Install test for the Arena Native App, run by `just snowflake-app-install-test`.
-- Installs the newest version of arena_app_pkg at its second-newest patch as
-- arena_app_s6, approves its specifications, upgrades to the newest patch,
-- binds a scratch table, creates suspended pools, checks the unknown-run
-- refusal, then uninstalls. Every check raises on failure.
-- The role (-D role=...) needs CREATE APPLICATION, CREATE DATABASE and INSTALL on the package.

DROP APPLICATION IF EXISTS arena_app_s6 CASCADE;

SHOW VERSIONS IN APPLICATION PACKAGE arena_app_pkg;
SET (version_name, upgrade_patch) = (
  SELECT "version", "patch"
  FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
  WHERE "dropped_on" IS NULL
  ORDER BY "created_on" DESC
  LIMIT 1
);
EXECUTE IMMEDIATE $$
DECLARE
  one_patch EXCEPTION (-20009, 'The newest version has one patch, and the upgrade step needs two. Run snowflake-app-build again.');
BEGIN
  IF ($upgrade_patch < 1) THEN
    RAISE one_patch;
  END IF;
  RETURN 'install patch ' || ($upgrade_patch - 1) || ', upgrade to patch ' || $upgrade_patch;
END;
$$;
SET install_patch = $upgrade_patch - 1;
SET install_sql = 'CREATE APPLICATION arena_app_s6 FROM APPLICATION PACKAGE arena_app_pkg USING VERSION '
  || $version_name || ' PATCH ' || $install_patch;
SET upgrade_sql = 'ALTER APPLICATION arena_app_s6 UPGRADE USING VERSION '
  || $version_name || ' PATCH ' || $upgrade_patch;
-- Runs the setup script, including the EAI block that reports trial accounts (509009).
EXECUTE IMMEDIATE $install_sql;

SHOW GRANTS TO APPLICATION arena_app_s6;
EXECUTE IMMEDIATE $$
DECLARE
  granted INTEGER;
  missing_grant EXCEPTION (-20001, 'The app is missing an automatically granted account privilege.');
BEGIN
  SHOW GRANTS TO APPLICATION arena_app_s6;
  SELECT COUNT(DISTINCT "privilege") INTO :granted
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
    WHERE "granted_on" = 'ACCOUNT'
      AND "privilege" IN (
        'CREATE COMPUTE POOL', 'CREATE WAREHOUSE', 'CREATE EXTERNAL ACCESS INTEGRATION'
      );
  IF (granted <> 3) THEN
    RAISE missing_grant;
  END IF;
  RETURN 'all 3 manifest privileges granted';
END;
$$;

SHOW SPECIFICATIONS IN APPLICATION arena_app_s6;
EXECUTE IMMEDIATE $$
DECLARE
  specs RESULTSET DEFAULT (SHOW SPECIFICATIONS IN APPLICATION arena_app_s6);
  spec_cursor CURSOR FOR specs;
  approved STRING DEFAULT '';
BEGIN
  FOR spec IN spec_cursor DO
    EXECUTE IMMEDIATE 'ALTER APPLICATION arena_app_s6 APPROVE SPECIFICATION '
      || spec."name" || ' SEQUENCE_NUMBER = ' || spec."sequence_number";
    approved := approved || spec."name" || ' ';
  END FOR;
  RETURN 'approved ' || approved;
END;
$$;
EXECUTE IMMEDIATE $$
DECLARE
  pending INTEGER;
  approved INTEGER;
  not_approved EXCEPTION (-20002, 'hf_access and arena_access are not both approved.');
BEGIN
  SHOW SPECIFICATIONS IN APPLICATION arena_app_s6;
  SELECT COUNT_IF("status" <> 'APPROVED'), COUNT_IF("status" = 'APPROVED' AND "name" IN ('HF_ACCESS', 'ARENA_ACCESS'))
    INTO :pending, :approved
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
  IF (pending > 0 OR approved <> 2) THEN
    RAISE not_approved;
  END IF;
  RETURN 'hf_access and arena_access approved';
END;
$$;

-- Setup reruns with both integrations already owned by the app.
EXECUTE IMMEDIATE $upgrade_sql;
EXECUTE IMMEDIATE $$
DECLARE
  upgraded INTEGER;
  integrations INTEGER;
  not_upgraded EXCEPTION (-20007, 'arena_app_s6 is not on the newest patch, or its upgrade did not complete.');
  integrations_missing EXCEPTION (-20008, 'AGILERL_APP_ARENA_APP_S6_HF_EAI and AGILERL_APP_ARENA_APP_S6_ARENA_EAI do not both exist after the upgrade.');
BEGIN
  SHOW APPLICATIONS LIKE 'ARENA_APP_S6';
  SELECT COUNT(*) INTO :upgraded
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
    WHERE "version" = $version_name AND "patch" = $upgrade_patch AND "upgrade_state" = 'COMPLETE';
  IF (upgraded <> 1) THEN
    RAISE not_upgraded;
  END IF;
  SHOW INTEGRATIONS LIKE 'AGILERL_APP_ARENA_APP_S6_%';
  SELECT COUNT(*) INTO :integrations
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
    WHERE "name" IN ('AGILERL_APP_ARENA_APP_S6_HF_EAI', 'AGILERL_APP_ARENA_APP_S6_ARENA_EAI');
  IF (integrations <> 2) THEN
    RAISE integrations_missing;
  END IF;
  RETURN 'upgraded to patch ' || $upgrade_patch || ' with both integrations';
END;
$$;
EXECUTE IMMEDIATE $$
DECLARE
  pending INTEGER;
  approved INTEGER;
  not_approved EXCEPTION (-20002, 'hf_access and arena_access are not both approved.');
BEGIN
  SHOW SPECIFICATIONS IN APPLICATION arena_app_s6;
  SELECT COUNT_IF("status" <> 'APPROVED'), COUNT_IF("status" = 'APPROVED' AND "name" IN ('HF_ACCESS', 'ARENA_ACCESS'))
    INTO :pending, :approved
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
  IF (pending > 0 OR approved <> 2) THEN
    RAISE not_approved;
  END IF;
  RETURN 'hf_access and arena_access approved';
END;
$$;

GRANT APPLICATION ROLE arena_app_s6.app_admin TO ROLE <% role %>;

CREATE OR REPLACE DATABASE arena_s6_scratch;
CREATE TABLE arena_s6_scratch.public.sft_rows (prompt STRING, completion STRING);
INSERT INTO arena_s6_scratch.public.sft_rows VALUES
  ('What is 2 + 2?', '4'),
  ('Name a primary colour.', 'Red'),
  ('Translate "hello" to French.', 'Bonjour');
CALL arena_app_s6.app_code.register_reference(
  'source_table', 'ADD',
  SYSTEM$REFERENCE('TABLE', 'arena_s6_scratch.public.sft_rows', 'PERSISTENT', 'SELECT')
);
SHOW REFERENCES IN APPLICATION arena_app_s6;
EXECUTE IMMEDIATE $$
DECLARE
  bound INTEGER;
  not_bound EXCEPTION (-20003, 'source_table has no binding.');
BEGIN
  SHOW REFERENCES IN APPLICATION arena_app_s6;
  SELECT COUNT(*) INTO :bound
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
    WHERE "name" = 'SOURCE_TABLE' AND "object_name" = 'SFT_ROWS';
  IF (bound < 1) THEN
    RAISE not_bound;
  END IF;
  RETURN 'source_table bound';
END;
$$;

-- First call that imports agilerl.snowflake from agilerl_arena.zip.
CALL arena_app_s6.app_code.set_pools(['GPU_NV_S'], 1);
SHOW COMPUTE POOLS LIKE 'AGILERL_APP_ARENA_APP_S6_%';
EXECUTE IMMEDIATE $$
DECLARE
  pools INTEGER;
  running INTEGER;
  nodes_running EXCEPTION (-20004, 'An AGILERL_APP_ARENA_APP_S6_ pool is missing or not suspended.');
BEGIN
  SHOW COMPUTE POOLS LIKE 'AGILERL_APP_ARENA_APP_S6_%';
  SELECT COUNT(*), COUNT_IF("state" <> 'SUSPENDED' OR "active_nodes" > 0)
    INTO :pools, :running
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
  IF (pools <> 3 OR running > 0) THEN
    RAISE nodes_running;
  END IF;
  RETURN '3 pools, all suspended';
END;
$$;

EXECUTE IMMEDIATE $$
DECLARE
  accepted EXCEPTION (-20005, 'status accepted a malformed run id.');
BEGIN
  BEGIN
    CALL arena_app_s6.app_code.status('nope');
  EXCEPTION
    WHEN OTHER THEN
      IF (CONTAINS(SQLERRM, 'run_id must look like arena-<yyyymmddHHMMSS>-<6 hex>, got \'nope\'')) THEN
        RETURN 'status refused the malformed run id';
      END IF;
      RAISE;
  END;
  RAISE accepted;
END;
$$;
EXECUTE IMMEDIATE $$
DECLARE
  accepted EXCEPTION (-20010, 'status accepted a run the app does not have.');
BEGIN
  BEGIN
    CALL arena_app_s6.app_code.status('arena-20260101000000-abcdef');
  EXCEPTION
    WHEN OTHER THEN
      IF (CONTAINS(SQLERRM, 'This app has no run arena-20260101000000-abcdef.')) THEN
        RETURN 'status refused the unknown run';
      END IF;
      RAISE;
  END;
  RAISE accepted;
END;
$$;

-- Later Arena steps. They need an Arena org key with Snowflake training enabled;
-- uncomment and fill in the placeholders to run them.
-- CALL arena_app_s6.app_code.configure('<arena_org_key>', '<external_user_id>');
-- CALL arena_app_s6.app_code.submit(
--   '<SFT manifest YAML, 1 GPU>', 1, 'source_table', '<alias from SHOW REFERENCES>'
-- );
-- CALL arena_app_s6.app_code.status('<run_id>');  -- repeat until state: COMPLETE
-- SELECT * FROM arena_app_s6.app_state.arena_metrics WHERE run_id = '<run_id>' ORDER BY seq;
-- CALL arena_app_s6.app_code.publish('<run_id>', 'arena_s6_model', NULL, NULL, TRUE);
-- CALL arena_app_s6.app_code.endpoint('arena_s6_model');  -- run the example SELECT it returns
-- CALL arena_app_s6.app_code.unpublish('arena_s6_model', NULL, TRUE);

DROP APPLICATION arena_app_s6 CASCADE;
DROP DATABASE arena_s6_scratch;

EXECUTE IMMEDIATE $$
DECLARE
  pools INTEGER;
  warehouses INTEGER;
  integrations INTEGER;
  left_behind EXCEPTION (-20006, 'An app pool, warehouse or integration survived the uninstall.');
BEGIN
  SHOW COMPUTE POOLS LIKE 'AGILERL_APP_ARENA_APP_S6_%';
  SELECT COUNT(*) INTO :pools FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
  SHOW WAREHOUSES LIKE 'AGILERL_APP_ARENA_APP_S6_%';
  SELECT COUNT(*) INTO :warehouses FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
  SHOW INTEGRATIONS LIKE 'AGILERL_APP_ARENA_APP_S6_%';
  SELECT COUNT(*) INTO :integrations FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
  IF (pools + warehouses + integrations > 0) THEN
    RAISE left_behind;
  END IF;
  RETURN 'uninstall left no app pool, warehouse or integration';
END;
$$;
