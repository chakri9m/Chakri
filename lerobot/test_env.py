import gymnasium as gym

# Specify render_mode
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Choose a random action
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()


