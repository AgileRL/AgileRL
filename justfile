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
