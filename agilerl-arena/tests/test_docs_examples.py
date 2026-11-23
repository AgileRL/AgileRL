# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""The tutorial manifests under ``docs/_static/examples`` submit as written."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agilerl.arena.client import ArenaClient
from agilerl.arena.stream import NDJsonStream

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "docs" / "_static" / "examples"
# Manifests whose tutorial submits them with --reward-file.
REWARD_FILES = {
    "gsm8k_grpo.yaml": EXAMPLES_DIR / "gsm8k-grpo" / "reward.py",
    "manifest.yaml": EXAMPLES_DIR / "gsm8k-judge" / "reward.py",
    "manifest_ruler.yaml": EXAMPLES_DIR / "gsm8k-judge" / "reward.py",
}
MANIFESTS = sorted(EXAMPLES_DIR.rglob("*.yaml"))


@pytest.fixture
def client() -> ArenaClient:
    """ArenaClient with a static API key and a stubbed submit stream."""
    with patch("agilerl.arena.auth.KeycloakOpenID"):
        arena = ArenaClient(api_key="test-key")
    stream = MagicMock(spec=NDJsonStream)
    stream.collect.return_value = {"job_id": 1}
    arena._open_stream = MagicMock(return_value=stream)
    return arena


class TestDocsExampleManifests:
    def test_examples_are_found(self):
        assert {path.name for path in MANIFESTS} >= {"gsm8k_grpo.yaml"}

    @pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda path: path.name)
    def test_example_submits(self, client, manifest):
        reward_file = REWARD_FILES.get(manifest.name)

        result = client.submit_experiment(
            manifest=manifest, project="docs", reward_file=reward_file
        )

        assert result == {"job_id": 1}

    def test_gsm8k_example_targets_registered_dataset(self, client):
        manifest = EXAMPLES_DIR / "gsm8k_grpo.yaml"

        client.submit_experiment(
            manifest=manifest,
            project="docs",
            reward_file=REWARD_FILES[manifest.name],
        )

        files = dict(client._open_stream.call_args[1]["files"])
        environment = json.loads(files["manifest"][1])["environment"]
        assert environment["name"] == "gsm8k"
        assert environment["dataset"] == "gsm8k"
        assert environment["rubric_file_path"] == "reward.py"
        assert files["reward_file"][0] == "reward.py"

    @pytest.mark.parametrize(
        ("manifest_name", "rubric_name", "group_rubric_name"),
        [
            ("manifest.yaml", "RUBRIC", None),
            ("manifest_ruler.yaml", "Correctness", "RULER"),
        ],
    )
    def test_judge_example_submits_its_judge(
        self, client, manifest_name, rubric_name, group_rubric_name
    ):
        manifest = EXAMPLES_DIR / "gsm8k-judge" / manifest_name

        client.submit_experiment(
            manifest=manifest,
            project="docs",
            reward_file=REWARD_FILES[manifest_name],
        )

        files = dict(client._open_stream.call_args[1]["files"])
        environment = json.loads(files["manifest"][1])["environment"]
        assert environment["rubric_name"] == rubric_name
        assert environment.get("group_rubric_name") == group_rubric_name
        assert environment["judge"]["provider"] == "openai"
        assert environment["judge"]["model"] == "gpt-4.1-mini"
        assert environment["judge"]["api_key_env"] == "OPENAI_API_KEY"
        assert environment["judge"]["max_tokens"] == 512
        assert files["reward_file"][0] == "reward.py"
