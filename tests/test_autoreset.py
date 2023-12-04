from testing_env import parallel_env

from agilerl.wrappers.pettingzoo_wrappers import PettingZooAutoResetParallelWrapper


def test_autoreset_wrapper_autoreset():
    """Tests the autoreset wrapper actually automatically resets correctly."""
    env = parallel_env(render_mode="human")
    env = PettingZooAutoResetParallelWrapper(env)

    observations, infos = env.reset()
    for _ in range(500):
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        assert list(infos.values())[0] < 100, "Game went on for too long"


if __name__ == "__main__":
    test_autoreset_wrapper_autoreset()
