import gymnasium as gym
import numpy as np

# Initialise the environment
env = gym.make("Pendulum-v1")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    reward = np.expand_dims(reward, axis=(0, 1))
    print('type(observation):', reward.shape)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
