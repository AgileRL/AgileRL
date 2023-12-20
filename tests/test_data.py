from agilerl.data.language_environment import (
    Language_Environment,
    Language_Observation,
    Policy,
)
from agilerl.utils.cache import Cache


def test_language_observation_base_class():
    Language_Observation.__abstractmethods__ = set()
    lang_obs = Language_Observation()
    lang_obs.to_sequence()
    lang_obs.__str__()
    metadata = lang_obs.metadata()

    assert metadata is None


def test_language_environment_base_class():
    Language_Environment.__abstractmethods__ = set()

    lang_env = Language_Environment()
    lang_env.step(None)
    lang_env.reset()
    lang_env.is_terminal()

    assert True


def test_policy_base_class():
    Policy.__abstractmethods__ = set()

    policy = Policy()
    policy.act(None)
    policy.train()
    policy.eval()

    assert isinstance(policy.cache, Cache)
