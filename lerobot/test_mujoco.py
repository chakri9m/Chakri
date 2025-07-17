import gym
env = gym.make("Pusher-v2", render_mode="human")
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, info, _ = env.step(action)
    env.render()
env.close()

