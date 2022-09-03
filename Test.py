import gym


env = gym.make('Swimmer-v4')

env.reset()

for i in range(10000):
    observation, reward, done, info = env.step(env.action_space.sample())

    # print(reward)

    if done:
        print(i)
        observation = env.reset()

env.close()