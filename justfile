set shell := ["bash", "-cu"]

index := "https://pypi.org/simple"

default:
    @just --list

# ---------------------------------------------------------------------------
# Build / publish
# ---------------------------------------------------------------------------

# In a uv workspace, every `uv build` writes to the workspace-root dist/,
# so both packages' artifacts share this one directory and are selected by
# name below (agilerl_arena-* vs agilerl-[0-9]*).

# Remove the shared workspace dist/ directory.
clean-dist:
    rm -rf dist

# Fail publish if a workspace pin drifts, or the all extra misses an extra.
check-extras:
    uv run python scripts/check-extras.py

# Build both workspace packages into dist/.
build: clean-dist
    uv build --package agilerl-arena
    uv build --package agilerl

check-dist: check-extras build
    uv publish --dry-run --check-url {{ index }} dist/*

# Publish order matters: agilerl depends on agilerl-arena, so arena has to
# reach the index first or the release is briefly uninstallable. --check-url
# skips a version already on the index: the two packages are versioned
# independently, so a release often reuses one unchanged.

# Publish agilerl-arena (first: agilerl depends on it).
publish-arena: check-dist
    uv publish --check-url {{ index }} dist/agilerl_arena-*

# Publish agilerl.
publish-core: check-dist
    uv publish --check-url {{ index }} dist/agilerl-[0-9]*

# Publish both, in dependency order.
publish: publish-arena publish-core

# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

# Serve docs locally with auto-rebuild (http://127.0.0.1:8000 by default).
build-docs:
    uv run --group docs sphinx-autobuild docs docs/_build/html

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
    bash scripts/run-tests.sh agilerl-arena/tests/test_snowflake_app_manifest.py

# Copy the snowflake-job image into the provider repository and add a version
# to the application package. Needs crane and a Snowflake CLI connection.
# Usage: just snowflake-app-build CONNECTION TAG   (TAG like 38.0.4-py313-cu130)
snowflake-app-build connection tag:
    #!/usr/bin/env bash
    set -euo pipefail
    snow="uvx --from snowflake-cli snow"
    tag="{{ tag }}"
    build="$PWD/dist/snowflake-app"
    rm -rf "$build"
    mkdir -p "$build"
    cp -R agilerl-arena/snowflake_app/snowflake.yml agilerl-arena/snowflake_app/manifest.yml \
        agilerl-arena/snowflake_app/setup_script.sql agilerl-arena/snowflake_app/app_readme.md \
        agilerl-arena/snowflake_app/app_code "$build"/
    ML_BUNDLE_CHANNEL=stable ML_BUNDLE_VERSION="${tag%-py313-cu130}" \
        uv build --package agilerl-arena --wheel -o "$build/wheel"
    keycloak="$(uv export --frozen --package agilerl-arena --no-hashes --no-header \
        --no-annotate --no-emit-project | grep '^python-keycloak==' | cut -d' ' -f1)"
    # python-keycloak is the one agilerl-arena dependency the Snowflake channel lacks.
    uv pip install --python 3.13 --target "$build/vendor" --no-deps "$keycloak"
    python3 agilerl-arena/snowflake_app/bundle.py "$build" "$tag" "$build"/wheel/*.whl "$build/vendor"
    rm -rf "$build/wheel" "$build/vendor"
    $snow sql --connection "{{ connection }}" -q "CREATE DATABASE IF NOT EXISTS arena_provider; CREATE SCHEMA IF NOT EXISTS arena_provider.images; CREATE IMAGE REPOSITORY IF NOT EXISTS arena_provider.images.repo"
    registry="$($snow spcs image-registry url --connection "{{ connection }}")"
    $snow spcs image-registry token --connection "{{ connection }}" --format JSON \
        | crane auth login "$registry" -u 0sessiontoken --password-stdin
    crane copy --platform linux/amd64 "public.ecr.aws/x7s7d0o8/containers/snowflake-job:$tag" \
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
        -f agilerl-arena/snowflake_app/tests/install.sql; then
        $snow sql --connection "{{ connection }}" --role "{{ role }}" \
            -q "DROP APPLICATION IF EXISTS arena_app_s6 CASCADE; DROP DATABASE IF EXISTS arena_s6_scratch"
        exit 1
    fi

# ---------------------------------------------------------------------------
# Type checking
# ---------------------------------------------------------------------------

# Static type checking with ty (config in pyproject.toml [tool.ty]).
# Runs the pinned ty pre-commit hook so local type-checking stays identical to
# CI, reproducing the two setup steps the Type checks workflow performs first.
typecheck:
    uv sync --all-groups --extra all
    uv run pre-commit run arena-symlink --all-files
    uv run pre-commit run ty --all-files

# This tree's .pre-commit-config.yaml (AgileRL/AgileRL git root).
pre-commit *args:
    uv run pre-commit run --all-files {{ args }}
