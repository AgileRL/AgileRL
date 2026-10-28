# Copyright 2026 AgileRL

"""Unset harness image observation keys do not name browser defaults."""

from agilerl.llm_envs.harness import RolloutHarness
from tests.helpers.rollout_doubles import FakeEnvClient, MiniTokenizer


def test_unset_image_observation_fields_read_no_named_keys() -> None:
    harness = RolloutHarness(FakeEnvClient(), MiniTokenizer())

    assert harness._screenshot_field == ""
    assert harness._goal_images_field == ""
