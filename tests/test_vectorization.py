from pettingzoo.mpe import simple_speaker_listener_v4

from agilerl.wrappers.pettingzoo_wrappers import PettingZooVectorizationParallelWrapper


def test_vectorization_wrapper():
    """
    Test the vectorization wrapper vectorizes and autoresets correctly.
    """
    n_envs = 4
    env = PettingZooVectorizationParallelWrapper(
        simple_speaker_listener_v4.parallel_env(render_mode="human"), n_envs=n_envs
    )
    observations, infos = env.reset()
    # Environment truncates after 100 steps, so we expect 1 reset.
    for ep in range(100):
        # this is where you would insert your policy
        actions = {
            agent: [(env.action_space(agent).sample(),) for n in range(n_envs)]
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        for agent in env.agents:
            assert len(observations[agent]) == n_envs
            assert len(rewards[agent]) == n_envs
            assert len(terminations[agent]) == n_envs
            assert len(truncations[agent]) == n_envs
            assert len(infos[agent]) == n_envs


if __name__ == "__main__":
    test_vectorization_wrapper()
