# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Exit 0 if the GitHub CI run that gated this commit succeeded."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any

API_VERSION = "2022-11-28"
CI_WORKFLOW = "ci.yml"
MAIN_BRANCH = "main"
SUCCESS = "success"
COMPLETED = "completed"


def env(name: str) -> str:
    """Return a required environment variable.

    :param name: Variable name.
    :return: Non-empty value.
    """
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"error: {name} is not set")
    return value


def github_json(url: str, token: str) -> Any:
    """GET a GitHub REST URL and decode JSON.

    :param url: Absolute API URL.
    :param token: Bearer token.
    :return: Decoded payload.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "agilerl-ci-status",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def commit_tree_sha(payload: object) -> str:
    """Return the git tree SHA from a commit API payload.

    :param payload: `GET /commits/{sha}` JSON.
    :return: Tree SHA.
    """
    if not isinstance(payload, dict):
        raise TypeError("commit JSON must be an object")
    commit = payload.get("commit")
    if not isinstance(commit, dict):
        raise TypeError("commit JSON is missing commit")
    tree = commit.get("tree")
    if not isinstance(tree, dict):
        raise TypeError("commit JSON is missing tree")
    sha = tree.get("sha")
    if not isinstance(sha, str) or not sha:
        raise TypeError("commit JSON is missing tree sha")
    return sha


def merged_main_pull(pulls: object) -> dict[str, Any] | None:
    """Return the latest associated PR merged into main.

    :param pulls: `GET /commits/{sha}/pulls` JSON.
    :return: Pull payload, or None.
    """
    if not isinstance(pulls, list):
        raise TypeError("associated pulls JSON must be an array")
    merged: list[tuple[str, dict[str, Any]]] = []
    for pull in pulls:
        if not isinstance(pull, dict):
            continue
        base = pull.get("base")
        if not isinstance(base, dict) or base.get("ref") != MAIN_BRANCH:
            continue
        merged_at = pull.get("merged_at")
        if not isinstance(merged_at, str) or not merged_at:
            continue
        merged.append((merged_at, pull))
    if not merged:
        return None
    merged.sort(key=lambda item: item[0])
    return merged[-1][1]


def pull_head_sha(pull: dict[str, Any]) -> str | None:
    """Return the PR head SHA.

    :param pull: Pull request payload.
    :return: Head SHA, or None.
    """
    head = pull.get("head")
    if not isinstance(head, dict):
        return None
    sha = head.get("sha")
    if isinstance(sha, str) and sha:
        return sha
    return None


def run_matches_pull(run: dict[str, Any], number: int, head_sha: str | None) -> bool:
    """True if this CI run belongs to the merged PR.

    :param run: Workflow run payload.
    :param number: Pull request number.
    :param head_sha: PR head SHA, if known.
    :return: Whether the run is for that PR.
    """
    if head_sha and run.get("head_sha") == head_sha:
        return True
    for pr in run.get("pull_requests") or []:
        if isinstance(pr, dict) and pr.get("number") == number:
            return True
    return False


def newest_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the run with the highest run_number.

    :param runs: Candidate workflow runs.
    :return: Newest run, or None.
    """
    if not runs:
        return None
    return max(runs, key=lambda run: run.get("run_number") or 0)


def runs_for_tree(
    runs: list[dict[str, Any]],
    commit_tree: str,
    head_trees: dict[str, str],
) -> list[dict[str, Any]]:
    """CI runs whose head commit has this tree.

    :param runs: Workflow run payloads.
    :param commit_tree: Tree SHA of the main commit.
    :param head_trees: Map of run head SHA to tree SHA.
    :return: Matching runs.
    """
    matched: list[dict[str, Any]] = []
    for run in runs:
        head_sha = run.get("head_sha")
        if isinstance(head_sha, str) and head_trees.get(head_sha) == commit_tree:
            matched.append(run)
    return matched


def choose_ci_run(
    associated_pulls: object,
    workflow_runs: object,
    commit_tree: str,
    head_trees: dict[str, str],
) -> dict[str, Any] | None:
    """Select the CI run that gated this commit.

    Merged PR into main first, then a run whose head commit has the same tree
    (`workflow_dispatch` on `hub-ci/<iid>`).

    :param associated_pulls: `GET /commits/{sha}/pulls` JSON.
    :param workflow_runs: `workflow_runs` array from the CI workflow.
    :param commit_tree: Tree SHA of this commit.
    :param head_trees: Map of run head SHA to tree SHA.
    :return: Matching run, or None.
    """
    if not isinstance(workflow_runs, list):
        raise TypeError("workflow runs JSON must be an array")
    runs = [run for run in workflow_runs if isinstance(run, dict)]
    pull = merged_main_pull(associated_pulls)
    if pull is not None:
        number = pull.get("number")
        if isinstance(number, int):
            head_sha = pull_head_sha(pull)
            matched = [run for run in runs if run_matches_pull(run, number, head_sha)]
            chosen = newest_run(matched)
            if chosen is not None:
                return chosen
    if not head_trees:
        return None
    return newest_run(runs_for_tree(runs, commit_tree, head_trees))


def report_ci_run(run: dict[str, Any]) -> int:
    """Print the run URL and return the process exit code.

    :param run: Chosen workflow run.
    :return: 0 if the run succeeded, else 1.
    """
    url = run.get("html_url")
    if isinstance(url, str) and url:
        print(url)
    if run.get("status") == COMPLETED and run.get("conclusion") == SUCCESS:
        return 0
    if run.get("status") == COMPLETED:
        detail = run.get("conclusion")
    else:
        detail = run.get("status")
    print(f"error: CI {detail}", file=sys.stderr)
    return 1


def unique_head_shas(runs: object) -> list[str]:
    """Deduplicated head SHAs from workflow runs.

    :param runs: `workflow_runs` array.
    :return: Head SHAs in first-seen order.
    """
    if not isinstance(runs, list):
        raise TypeError("workflow runs JSON must be an array")
    seen: set[str] = set()
    ordered: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        sha = run.get("head_sha")
        if isinstance(sha, str) and sha and sha not in seen:
            seen.add(sha)
            ordered.append(sha)
    return ordered


def main(argv: list[str] | None = None) -> None:
    """Look up the gating CI run for GITHUB_SHA and exit with its status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    repo = env("GITHUB_REPOSITORY")
    sha = env("GITHUB_SHA")
    token = env("GITHUB_TOKEN")
    api = f"https://api.github.com/repos/{repo}"
    quoted = urllib.parse.quote(sha)
    commit = github_json(f"{api}/commits/{quoted}", token)
    tree = commit_tree_sha(commit)
    pulls = github_json(f"{api}/commits/{quoted}/pulls", token)
    runs_url = f"{api}/actions/workflows/{CI_WORKFLOW}/runs?per_page=100"
    runs_payload = github_json(runs_url, token)
    if not isinstance(runs_payload, dict):
        raise TypeError("workflow runs JSON must be an object")
    runs = runs_payload.get("workflow_runs")
    chosen = choose_ci_run(pulls, runs, tree, {})
    if chosen is None:
        head_trees = {
            head_sha: commit_tree_sha(github_json(f"{api}/commits/{head_sha}", token))
            for head_sha in unique_head_shas(runs)
        }
        chosen = choose_ci_run(pulls, runs, tree, head_trees)
    if chosen is None:
        raise SystemExit("error: no CI run found for this commit")
    raise SystemExit(report_ci_run(chosen))


if __name__ == "__main__":
    main()
