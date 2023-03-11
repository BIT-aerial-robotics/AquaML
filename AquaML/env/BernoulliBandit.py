import numpy as np
from AquaML.BaseClass import RLBaseEnv
from AquaML.DataType import DataInfo


class BernoulliBandit(RLBaseEnv):
    def __init__(self, K):
        # 随机生成K个概率，作为K个老虎机的概率
        self.probs = np.random.rand(size=K)
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的老虎机的索引
        self.best_prob = self.probs[self.best_idx]  # 获奖概率最大的老虎机的概率
        self.K = K  # 老虎机的个数

        self._obs_info = DataInfo(
            names=('reward',),
            shapes=((1,),),
            dtypes=np.float32
        )

    def reset(self):
        obs = {'reward': 0}
        obs = self.initial_obs(obs)
        return obs

    def step(self, action_dict):
        action = action_dict['action']

        if np.random.rand() < self.probs[action]:
            reward = 1
        else:
            reward = 0

        obs = {'reward': reward}
        obs = self.check_obs(obs, action_dict)

        reward = {'total_reward': reward}

        return obs, reward, False, {}

    def close(self):
        pass