# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Build a GitHub Release title and body from spoke commit messages.

Hub export commits carry the public title and body (spoke-public copy). GitHub
generate-notes only lists merged PRs, so direct-push tags would get an empty
changelog. This formats those commit messages like the historical releases.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

GITHUB_TITLE_MAX = 256
CONVENTIONAL = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|chore|ci|build|revert)"
    r"(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<subject>.+)$"
)
TRAILER = re.compile(
    r"^(Hub-RevId|Spoke-RevId|Spoke-PR|Co-authored-by|Signed-off-by):",
    re.IGNORECASE,
)
BREAKING_FOOTER = re.compile(r"^BREAKING[- ]CHANGE:", re.IGNORECASE | re.MULTILINE)

SECTION_BY_TYPE = {
    "feat": "Features",
    "perf": "Optimizations",
    "fix": "Fixes",
    "docs": "Documentation",
    "test": "Tests",
    "refactor": "Refactoring",
    "style": "Other",
    "chore": "Other",
    "ci": "Other",
    "build": "Other",
    "revert": "Other",
}
SECTION_ORDER = (
    "Features",
    "Optimizations",
    "Fixes",
    "Breaking Changes",
    "Documentation",
    "Tests",
    "Refactoring",
    "Other",
)
TITLE_TYPES = ("feat", "fix", "perf", "refactor", "docs")
TITLE_SKIP_SCOPES = frozenset({"release", "ci", "github"})


@dataclass(frozen=True)
class SpokeCommit:
    """One public spoke commit, trailers stripped."""

    type: str
    scope: str
    subject: str
    body: str
    breaking: bool

    @property
    def section(self) -> str:
        if self.breaking:
            return "Breaking Changes"
        return SECTION_BY_TYPE.get(self.type, "Other")


def _strip_trailers(body: str) -> str:
    kept: list[str] = []
    for line in body.splitlines():
        if TRAILER.match(line.strip()):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def parse_commit_message(message: str) -> SpokeCommit | None:
    """Parse a spoke commit message. Skip empty and merge commits."""
    text = message.strip()
    if not text or text.lower().startswith("merge "):
        return None
    subject, _, rest = text.partition("\n")
    subject = subject.strip()
    body = _strip_trailers(rest)
    match = CONVENTIONAL.fullmatch(subject)
    breaking_footer = bool(BREAKING_FOOTER.search(body))
    if match:
        return SpokeCommit(
            type=match.group("type"),
            scope=(match.group("scope") or "").strip(),
            subject=match.group("subject").strip(),
            body=body,
            breaking=bool(match.group("breaking")) or breaking_footer,
        )
    return SpokeCommit(
        type="",
        scope="",
        subject=subject,
        body=body,
        breaking=breaking_footer,
    )


def commits_from_compare(payload: dict[str, object]) -> list[SpokeCommit]:
    """Read GitHub compare JSON, a {commits: [...]} wrapper, or one commit object."""
    if isinstance(payload.get("commits"), list):
        raw = payload["commits"]
    elif isinstance(payload.get("commit"), dict):
        raw = [payload]
    else:
        return []
    parsed: list[SpokeCommit] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        parents = item.get("parents")
        if isinstance(parents, list) and len(parents) > 1:
            continue
        commit = item.get("commit")
        if not isinstance(commit, dict):
            continue
        message = commit.get("message")
        if not isinstance(message, str):
            continue
        entry = parse_commit_message(message)
        if entry is not None:
            parsed.append(entry)
    return parsed


def release_name_prefix(tag: str) -> str:
    """User-facing tag for the release title."""
    if tag.startswith("agilerl-arena/v"):
        return f"agilerl-arena {tag.removeprefix('agilerl-arena/')}"
    return tag


def _title_rank(commit: SpokeCommit) -> tuple[int, int]:
    if commit.breaking:
        return (0, 0)
    if commit.scope in TITLE_SKIP_SCOPES:
        type_rank = 50
    elif commit.type in TITLE_TYPES:
        type_rank = TITLE_TYPES.index(commit.type)
    else:
        type_rank = 40
    return (1, type_rank)


def release_title(tag: str, commits: Sequence[SpokeCommit]) -> str:
    """`vX.Y.Z: headline` from the highest-signal commit."""
    prefix = release_name_prefix(tag)
    if not commits:
        return prefix
    primary = min(commits, key=_title_rank)
    headline = primary.subject.strip()
    if not headline:
        return prefix
    title = f"{prefix}: {headline}"
    if len(title) <= GITHUB_TITLE_MAX:
        return title
    keep = GITHUB_TITLE_MAX - len(prefix) - 2
    return f"{prefix}: {headline[:keep].rstrip()}"


def _render_entry(commit: SpokeCommit) -> str:
    if not commit.body:
        return f"- {commit.subject}"
    indented = "\n".join(
        f"  {line}" if line else "" for line in commit.body.splitlines()
    )
    return f"- {commit.subject}\n\n{indented}"


def release_body(
    tag: str,
    previous_tag: str,
    commits: Sequence[SpokeCommit],
    *,
    repo: str,
) -> str:
    """Categorized notes plus a compare link."""
    grouped: dict[str, list[SpokeCommit]] = {name: [] for name in SECTION_ORDER}
    for commit in commits:
        grouped[commit.section].append(commit)
    parts: list[str] = []
    for section in SECTION_ORDER:
        entries = grouped[section]
        if not entries:
            continue
        parts.append(f"## {section}\n")
        parts.append("\n\n".join(_render_entry(c) for c in entries))
        parts.append("")
    if previous_tag:
        parts.append(
            f"**Full Changelog**: https://github.com/{repo}/compare/"
            f"{previous_tag}...{tag}"
        )
    return "\n".join(parts).strip() + "\n"


def build_release(
    payload: dict[str, object],
    *,
    tag: str,
    previous_tag: str,
    repo: str,
) -> tuple[str, str]:
    """Return (title, body) for gh release create."""
    total = payload.get("total_commits")
    commits = commits_from_compare(payload)
    listed = payload.get("commits")
    listed_len = len(listed) if isinstance(listed, list) else len(commits)
    if isinstance(total, int) and total > listed_len:
        print(
            f"error: compare {previous_tag}...{tag} truncated "
            f"({listed_len} of {total} commits)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return (
        release_title(tag, commits),
        release_body(tag, previous_tag, commits, repo=repo),
    )


def main(argv: list[str] | None = None) -> None:
    """Read GitHub compare JSON on stdin; write title and notes files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--previous-tag", default="")
    parser.add_argument("--repo", default="AgileRL/AgileRL")
    parser.add_argument("--title-file", required=True)
    parser.add_argument("--notes-file", required=True)
    args = parser.parse_args(argv)
    raw = sys.stdin.read()
    payload: object = json.loads(raw) if raw.strip() else {"commits": []}
    if not isinstance(payload, dict):
        print("error: compare JSON must be an object", file=sys.stderr)
        raise SystemExit(1)
    title, body = build_release(
        payload,
        tag=args.tag,
        previous_tag=args.previous_tag,
        repo=args.repo,
    )
    with Path(args.title_file).open("w", encoding="utf-8") as handle:
        handle.write(title)
    with Path(args.notes_file).open("w", encoding="utf-8") as handle:
        handle.write(body)


if __name__ == "__main__":
    main()
