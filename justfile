set shell := ["bash", "-cu"]

index := "https://pypi.org/simple"

default:
    @just --list

# ---------------------------------------------------------------------------
# Build / publish
# ---------------------------------------------------------------------------

# Remove dist/.
clean-dist:
    rm -rf dist

# Build agilerl into dist/.
build: clean-dist
    uv build --package agilerl --out-dir dist

check-dist: build
    uv publish --dry-run --check-url {{ index }} dist/*

# Publish agilerl. --check-url skips a version already on the index.
publish: check-dist
    uv publish --check-url {{ index }} dist/*

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Run pytest.
# Usage:
#   just test [no-parallel] [cov] [pytest args...]
test *args:
    bash scripts/run-tests.sh {{args}}

# Lint the Snowflake Native App manifest and setup script (no account needed).
snowflake-app-validate:
    bash scripts/run-tests.sh tests/test_snowflake_app_manifest.py

# Copy the snowflake-job image into the provider repository and add a version
# to the application package. Needs crane, AWS credentials that can pull from
# private ECR in us-east-1, and a Snowflake CLI connection.
# Usage: just snowflake-app-build CONNECTION TAG   (TAG like 38.0.4-py313-cu130)
snowflake-app-build connection tag:
    #!/usr/bin/env bash
    set -euo pipefail
    snow="uvx --from snowflake-cli snow"
    tag="{{ tag }}"
    build="$PWD/dist/snowflake-app"
    rm -rf "$build"
    mkdir -p "$build"
    cp -R snowflake_app/snowflake.yml snowflake_app/manifest.yml \
        snowflake_app/setup_script.sql snowflake_app/app_readme.md \
        snowflake_app/app_code "$build"/
    ML_BUNDLE_CHANNEL=stable ML_BUNDLE_VERSION="${tag%-py313-cu130}" \
        uv build --package agilerl --wheel -o "$build/wheel"
    keycloak="$(uv export --frozen --package agilerl --no-hashes --no-header \
        --no-annotate --no-emit-project | grep '^python-keycloak==' | cut -d' ' -f1)"
    # python-keycloak is the one agilerl dependency the Snowflake channel lacks.
    uv pip install --python 3.13 --target "$build/vendor" --no-deps "$keycloak"
    python3 snowflake_app/bundle.py "$build" "$tag" "$build"/wheel/*.whl "$build/vendor"
    rm -rf "$build/wheel" "$build/vendor"
    $snow sql --connection "{{ connection }}" -q "CREATE DATABASE IF NOT EXISTS arena_provider; CREATE SCHEMA IF NOT EXISTS arena_provider.images; CREATE IMAGE REPOSITORY IF NOT EXISTS arena_provider.images.repo"
    registry="$($snow spcs image-registry url --connection "{{ connection }}")"
    $snow spcs image-registry token --connection "{{ connection }}" --format JSON \
        | crane auth login "$registry" -u 0sessiontoken --password-stdin
    ecr=761296822462.dkr.ecr.us-east-1.amazonaws.com
    aws ecr get-login-password --region us-east-1 | crane auth login "$ecr" -u AWS --password-stdin
    crane copy --platform linux/amd64 "$ecr/containers/snowflake-job:$tag" \
        "$registry/arena_provider/images/repo/snowflake-job:$tag"
    $snow app version create "v${tag//[^A-Za-z0-9]/_}" --project "$build" \
        --connection "{{ connection }}" --skip-git-check --no-interactive --force

# Install the newest arena_app_pkg version as ROLE, check it, then uninstall it.
# Usage: just snowflake-app-install-test CONNECTION ROLE
snowflake-app-install-test connection role:
    #!/usr/bin/env bash
    set -euo pipefail
    snow="uvx --from snowflake-cli snow"
    if ! $snow sql --connection "{{ connection }}" --role "{{ role }}" -D "role={{ role }}" \
        -f snowflake_app/tests/install.sql; then
        $snow sql --connection "{{ connection }}" --role "{{ role }}" \
            -q "DROP APPLICATION IF EXISTS arena_app_s6 CASCADE; DROP DATABASE IF EXISTS arena_s6_scratch"
        exit 1
    fi

# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

# Static type checking with the pinned ty pre-commit hook, as the Type checks
# workflow runs it.
typecheck:
    uv sync --group dev --extra snowflake-job
    uvx pre-commit run ty --all-files

# This tree's .pre-commit-config.yaml (AgileRL/AgileRL git root).
pre-commit *args:
    uvx pre-commit run --all-files {{ args }}
