import gymnasium as gym

from agilerl.wrappers.learning import Skill


# Skill class wraps around an environment
def test_wrap_environment():
    env = gym.make("CartPole-v1")
    skill = Skill(env)

    assert isinstance(skill, Skill)
    assert isinstance(skill.env, gym.Wrapper)


# Skill class returns the skill_reward function output
def test_return_skill_reward():
    env = gym.make("CartPole-v1")

    class CustomSkill(Skill):
        def __init__(self, env):
            super().__init__(env)

        def skill_reward(self, observation, reward, terminated, truncated, info):
            return 0, 1, False, False, "Custom info"

    skill = CustomSkill(env)
    _, _ = env.reset()
    action = env.action_space.sample()
    new_observation, reward, terminated, truncated, info = skill.step(action)

    assert new_observation == 0
    assert reward == 1
    assert terminated is False
    assert truncated is False
    assert info == "Custom info"


# Skill class is initialized with a non-gym environment
def test_non_gym_environment():
    class CustomEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Discrete(2)
            self.action_space = gym.spaces.Discrete(2)

        def reset(self):
            return self.observation_space.sample(), {}

        def step(self, action):
            return 0, 1, False, False, {}

    env = CustomEnv()
    skill = Skill(env)
    _, _ = env.reset()
    action = env.action_space.sample()
    new_observation, reward, terminated, truncated, info = skill.step(action)

    assert new_observation == 0
    assert reward == 1
    assert terminated is False
    assert truncated is False
    assert info == {}
