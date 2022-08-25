import gym


class GymEnvWrapper:
    def __init__(self, name: str):
        self.env = gym.make(name)

    def step(self, action):
        action = action * 2
        observation, reward, done, info = self.env.step(action)
        observation = observation.reshape(1, -1)

        obs = {'obs': observation, 'pos': observation[:, :2]}
        # obs = {'obs': observation}
        reward = {'total_reward': reward}

        return obs, reward, done

    def reset(self):
        observation = self.env.reset()
        observation = observation.reshape(1, -1)

        # obs = {'obs': observation}
        obs = {'obs': observation, 'pos': observation[:, :2]}

        return obs


if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    print(env.action_space)
