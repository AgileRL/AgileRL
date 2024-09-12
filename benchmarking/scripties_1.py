from pettingzoo.mpe import simple_speaker_listener_v4

env = simple_speaker_listener_v4.env(render_mode="human")
print("1", env.reset(seed=42))

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print("2", observation)
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
    print("3", action)
    print("4", env.step(action))
env.close()
