from pettingzoo.mpe import simple_speaker_listener_v4

env = simple_speaker_listener_v4.parallel_env(render_mode="human")
observations, infos = env.reset()
print("1", observations, infos)
while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print("actions", actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print("2", observations, rewards, terminations, truncations, infos)
print("CLO", env.close())
