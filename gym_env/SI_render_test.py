from single_inspector_extra_basic import SimpleInspectionEnv

env = SimpleInspectionEnv()
obs = env.reset()

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    env.render()

    print(
        "reward:", reward,
        "done:", done,
        "status:", info["status"],
        "distance:", info["distance"],
    )