set shell := ["bash", "-cu"]

default:
    @just --list

# ---------------------------------------------------------------------------
# Build / publish
# ---------------------------------------------------------------------------

# In a uv workspace, every `uv build` writes to the workspace-root dist/,
# so both packages' artifacts share this one directory and are selected by
# name below (agilerl_arena-* vs agilerl-[0-9]*).
clean-dist:
    rm -rf dist

build: clean-dist
    uv build --package agilerl-arena
    uv build --package agilerl

check-dist: build
    uv publish --dry-run --check-url https://pypi.org/simple dist/*

# Publish order matters: agilerl-arena first, then agilerl.
publish-arena: check-dist
    uv publish dist/agilerl_arena-*

publish-core: check-dist
    uv publish dist/agilerl-[0-9]*

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
# Creates the agilerl/arena dev symlink on first run: agilerl.arena is a
# namespace portion shipped by agilerl-arena, and ty resolves it through the
# symlink. Both paths are passed explicitly because ty does not traverse
# symlinked directories during discovery.
typecheck:
    [ -e agilerl/arena ] || ln -s ../agilerl-arena/agilerl/arena agilerl/arena
    uv run ty check agilerl agilerl/arena
