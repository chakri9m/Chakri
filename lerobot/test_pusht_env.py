import gymnasium as gym

env = gym.make("Pusher-v2", render_mode="human")  # or "Pusher-v4"

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    env.render()

env.close()

