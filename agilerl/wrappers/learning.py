import gymnasium as gym


class Skill(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """The Skill class, used in curriculum learning to teach agents skills. This class works as a
    wrapper around an environment that alters the reward to encourage learning of a particular skill.

    :param env: Environment to learn in
    :type env: Gymnasium-style environment
    """

    def __init__(self, env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # Use custom reward
        return self.skill_reward(observation, reward, terminated, truncated, info)

    def skill_reward(self, observation, reward, terminated, truncated, info):
        return observation, reward, terminated, truncated, info
