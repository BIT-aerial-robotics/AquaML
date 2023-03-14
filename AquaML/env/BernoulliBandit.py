import numpy as np
from AquaML.BaseClass import RLBaseEnv
from AquaML.DataType import DataInfo
from copy import deepcopy


class BernoulliBandit(RLBaseEnv):
    def __init__(self, num_of_bandits, random_freq=0):
        super(BernoulliBandit, self).__init__()
        # 随机生成K个概率，作为K个老虎机的概率
        self.probs = np.random.uniform(size=num_of_bandits, low=0.1, high=0.3) * 0

        self.best_idx = np.random.randint(0, num_of_bandits - 1, 1)  # 获奖概率最大的老虎机的索引
        self.probs[self.best_idx] = 0.95
        self.best_prob = self.probs[self.best_idx]  # 获奖概率最大的老虎机的概率
        self.K = num_of_bandits  # 老虎机的个数

        self.last_action = np.array([0, ]).reshape((1, 1))  # 上一次的动作
        self.last_reward = np.array([0, ]).reshape((1, 1))  # 上一次的奖励

        self.actions = np.zeros((num_of_bandits, 1))  # 存储运行时的动作

        self.times = np.array([0, ]).reshape((1, 1))  # 记录运行的次数

        self.reset_count = 0
        self.random_freq = random_freq

        self._obs_info = DataInfo(
            names=('reward2', 'last_action', 'times', 'best_idx', 'best_pr', 'pr'),
            shapes=((1,), (1,), (1,), (1,), (1,), (1,)),
            dtypes=np.float32
        )

        self.count = 0
        self.current_step_count = 0

    def random_bandit(self):
        self.probs = np.random.uniform(size=self.K, low=0.1, high=0.3) * 0

        self.best_idx = np.random.randint(0, self.K - 1, 1)  # 获奖概率最大的老虎机的索引
        self.probs[self.best_idx] = 0.95

    def reset(self):
        self.reset_count += 1
        hidden_reset = False
        if self.random_freq > 0:

            if self.reset_count % self.random_freq == 0:
                self.times = 0
                self.random_bandit()
                self.last_action = np.array([0, ]).reshape((1, 1))  # 上一次的动作
                self.last_reward = np.array([0, ]).reshape((1, 1))  # 上一次的奖励
                hidden_reset = True

        obs = {'reward2': self.last_reward,
               'last_action': self.last_action,
               'times': np.array([self.times, ]).reshape((1, 1)),
               'best_idx': np.array([self.best_idx, ]).reshape((1, 1)),
               'best_pr': np.array([self.best_prob, ]).reshape((1, 1)),
               'pr': np.array(self.probs[self.last_action]).reshape((1, 1))
               }
        obs = self.initial_obs(obs)
        self.actions = np.zeros((self.K, 1))
        return obs, hidden_reset

    def step(self, action_dict):
        action = action_dict['action']

        rand = np.random.rand()

        if rand < self.probs[action]:
            reward = 1
        else:
            reward = -0.5

        self.actions[action] += 1

        obs = {'reward2': np.array([reward, ]).reshape((1, 1)),
               'last_action': deepcopy(self.last_action.reshape((1, 1))),
               'times': np.array([self.times, ]).reshape((1, 1)),
               'best_idx': np.array([self.best_idx, ]).reshape((1, 1)),
               'best_pr': np.array([self.best_prob, ]).reshape((1, 1)),
               'pr': np.array(self.probs[action]).reshape((1, 1))
               }

        self.last_action = action.numpy()
        self.last_reward = obs['reward2']

        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward}

        self.count += 1

        # if self.count % 200 == 0:
        #     self.display_key_info()

        return obs, reward, False, {}

    def display(self):
        info = {'best`:': self.best_idx, 'best_prob': self.best_prob}

        most_action_chosen = np.argmax(self.actions)

        info['most_prob'] = self.probs[most_action_chosen]

        info['most_action_chosen'] = most_action_chosen
        info['most_action_chosen_prob'] = self.actions[most_action_chosen] / self.actions.sum()
        info['best_action_chosen_prob'] = self.actions[self.best_idx] / self.actions.sum()

        # display info
        print('--------{}----------'.format('validation'))
        for key, value in info.items():
            print(key, value)

    def close(self):
        pass
