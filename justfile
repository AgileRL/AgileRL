set shell := ["bash", "-cu"]

default:
    @just --list

# ---------------------------------------------------------------------------
# Build / publish
# ---------------------------------------------------------------------------

clean-dist:
    rm -rf dist agilerl-arena/dist

build-arena: clean-dist
    uv build --directory agilerl-arena

build-core:
    uv build --directory .

build: build-arena build-core

check-dist: build
    uv publish --dry-run --check-url https://pypi.org/simple agilerl-arena/dist/* dist/*

# Publish order matters: agilerl-arena first, then agilerl.
publish-arena: check-dist
    uv publish agilerl-arena/dist/*

publish-core:
    uv publish dist/*

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
