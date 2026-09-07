# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Create GitHub Releases for each dist a Publish-release plan uploaded."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NoReturn

Run = Callable[..., subprocess.CompletedProcess[str]]


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def artifacts_for(dist_dir: Path, dist_name: str, version: str) -> list[Path]:
    """Return wheel/sdist files for one planned package version."""
    return sorted(dist_dir.glob(f"{dist_name}-{version}*"))


def _default_run(
    args: Sequence[str], **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=kwargs.pop("check", True), **kwargs)


def create_releases(
    plan: dict[str, Any],
    *,
    dist_dir: Path,
    repo: str,
    notes_root: Path,
    run: Run = _default_run,
) -> None:
    """Create a GitHub Release per fetch row; skip tags that already have one."""
    notes_root.mkdir(parents=True, exist_ok=True)
    for row in plan["fetch"]:
        tag = row["git_tag"]
        prev = row["previous_tag"]
        dist_name = row["dist_name"]
        version = row["version"]
        artifacts = artifacts_for(dist_dir, dist_name, version)
        if not artifacts:
            _fail(f"missing dist artifacts for {dist_name}=={version}")
        view = run(
            ["gh", "release", "view", tag],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if view.returncode == 0:
            print(f"GitHub Release {tag} already exists; leaving body unchanged")
            continue
        notes_dir = notes_root / tag.replace("/", "_")
        notes_dir.mkdir(parents=True, exist_ok=True)
        compare_path = notes_dir / "compare.json"
        with compare_path.open("w", encoding="utf-8") as out:
            if prev:
                run(
                    ["gh", "api", f"repos/{repo}/compare/{prev}...{tag}"],
                    stdout=out,
                )
            else:
                run(["gh", "api", f"repos/{repo}/commits/{tag}"], stdout=out)
        title_file = notes_dir / "title.txt"
        notes_file = notes_dir / "notes.md"
        with compare_path.open(encoding="utf-8") as compare:
            run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "spoke_release_notes.py"),
                    "--tag",
                    tag,
                    "--previous-tag",
                    prev,
                    "--repo",
                    repo,
                    "--title-file",
                    str(title_file),
                    "--notes-file",
                    str(notes_file),
                ],
                stdin=compare,
            )
        title = title_file.read_text(encoding="utf-8").rstrip("\n")
        cmd = [
            "gh",
            "release",
            "create",
            tag,
            *[str(path) for path in artifacts],
            "--title",
            title,
            "--notes-file",
            str(notes_file),
        ]
        if tag.startswith("agilerl-arena/"):
            cmd.append("--latest=false")
        run(cmd)


def main(argv: list[str] | None = None) -> None:
    """Create GitHub Releases from plan.json and dist/ after PyPI succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--notes-dir", type=Path)
    args = parser.parse_args(argv)
    notes_root = args.notes_dir
    if notes_root is None:
        notes_root = Path(os.environ["RUNNER_TEMP"]) / "spoke-release"
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    create_releases(
        plan,
        dist_dir=args.dist,
        repo=args.repo,
        notes_root=notes_root,
    )


if __name__ == "__main__":
    main()
