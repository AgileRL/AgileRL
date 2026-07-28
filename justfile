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
# Runs the pinned ty pre-commit hook so local type-checking stays identical to
# CI, reproducing the two setup steps the Type checks workflow performs first.
#
# ty resolves imports from .venv, so without the extras every LLM import reads
# as unresolved. Off Linux the platform-gated ones (vllm, deepspeed,
# liger-kernel, bitsandbytes; box2d on Windows) are skipped by their markers,
# so this resolves on every platform. agilerl.arena is a namespace portion
# shipped by agilerl-arena, which ty only sees through the dev symlink; the
# arena-symlink hook creates it, and `pre-commit run` takes one hook at a time.
typecheck:
    uv sync --all-groups --extra all
    uv run pre-commit run arena-symlink --all-files
    uv run pre-commit run ty --all-files
