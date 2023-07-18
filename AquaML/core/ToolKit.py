import numpy as np
from copy import deepcopy
from AquaML.core.DataParser import DataSet
import tensorflow as tf
import os


def mkdir(path: str):
    """
    create a directory in current path.

    Args:
        path (_type_:str): name of directory.

    Returns:
        _type_: str or None: path of directory.
    """
    current_path = os.getcwd()
    # print(current_path)
    path = os.path.join(current_path, path)
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    else:
        None


# def display_dict(dic: dict):
#     for key, value in dic.items():
#         print(key, value)

class SummaryRewardCollector:

    def __init__(self, reward_names):
        self.reward_names = reward_names
        self.reward_dict = {}
        self.summary_dict = {}

    def reset_step(self):
        del self.reward_dict
        self.reward_dict = {}
        for name in self.reward_names:
            self.reward_dict[name] = []

    def store_data(self, reward: dict):
        for name in self.reward_names:
            if name != 'indicate':
                self.reward_dict[name].append(reward[name])

    def summary_episode(self):

        for name in self.reward_names:
            self.summary_dict[name].append(np.sum(self.reward_dict[name]))

        self.reset_step()

    def reset(self):
        self.reset_step()
        for name in self.reward_names:
            self.summary_dict[name] = []
            self.summary_dict['max_' + name] = []
            self.summary_dict['min_' + name] = []

    def get_data(self):

        sunmary_dict = {}

        for name in self.reward_names:
            sunmary_dict['summary_' + name] = np.mean(self.summary_dict[name])
            sunmary_dict['summary_' + name + '_max'] = np.max(self.summary_dict[name])
            sunmary_dict['summary_' + name + '_min'] = np.min(self.summary_dict[name])

        return sunmary_dict


class MDPCollector:
    """
    ThreadCollector
    """

    def __init__(self,
                 obs_names,
                 action_names,
                 reward_names,
                 ):

        self.obs_names = obs_names
        self.action_names = action_names
        self.reward_names = reward_names
        self.next_obs_names = obs_names

        self.obs_dict = {}
        self.action_dict = {}
        self.reward_dict = {}
        self.next_obs_dict = {}

        self.masks = []

    def reset(self):
        """
        reset
        """
        del self.obs_dict
        del self.action_dict
        del self.reward_dict
        del self.next_obs_dict
        del self.masks

        self.obs_dict = {}
        self.action_dict = {}
        self.reward_dict = {}
        self.next_obs_dict = {}

        self.masks = []

        for name in self.obs_names:
            self.obs_dict[name] = []

        for name in self.action_names:
            self.action_dict[name] = []

        for name in self.reward_names:
            self.reward_dict[name] = []

        for name in self.next_obs_names:
            self.next_obs_dict[name] = []

    def store_data(self, obs: dict, action: dict, reward: dict, next_obs: dict, mask: int):
        """
        store data
        """

        for name in self.obs_names:
            self.obs_dict[name].append(obs[name])

        for name in self.action_names:
            self.action_dict[name].append(action[name])

        for name in self.reward_names:
            self.reward_dict[name].append(reward[name])

        for name in self.next_obs_names:
            self.next_obs_dict[name].append(next_obs[name])

        self.masks.append(mask)

    def get_data(self):
        """
        get data

        返回的数据格式为
        """

        obs_dict = {}

        for name in self.obs_names:
            obs_dict[name] = np.vstack(self.obs_dict[name])

        action_dict = {}

        for name in self.action_names:
            action_dict[name] = np.vstack(self.action_dict[name])

        reward_dict = {}

        for name in self.reward_names:
            reward_dict[name] = np.vstack(self.reward_dict[name])

        next_obs_dict = {}

        for name in self.next_obs_names:
            next_obs_dict['next_' + name] = np.vstack(self.next_obs_dict[name])

        mask = np.vstack(self.masks)

        return obs_dict, action_dict, reward_dict, next_obs_dict, mask


class LossTracker:
    def __init__(self):
        self.loss_dict = {}

    def add_data(self, loss_dict: dict, prefix: str = ''):

        if len(prefix) > 0:
            prefix = prefix + '/'

        for key, value in loss_dict.items():

            all_name = prefix + key

            if all_name not in self.loss_dict:
                self.loss_dict[all_name] = []
            if isinstance(value, tf.Tensor):
                self.loss_dict[all_name].append(value.numpy())
            else:
                self.loss_dict[all_name].append(value)

    def reset(self):
        self.loss_dict = {}

    def get_data(self):
        loss_dict = {}
        for key, value in self.loss_dict.items():
            # array = deepcopy(value)
            loss_dict[key + '_max'] = np.max(value)
            loss_dict[key + '_min'] = np.min(value)
            loss_dict[key] = np.mean(value)

            if 'reward' in key:
                loss_dict[key + '_std'] = np.std(value)

            if 'indicate' in key:
                loss_dict[key + '_std'] = np.std(value)

        # 获取data以后自动释放内存
        self.reset()
        return loss_dict


class DataSetTracker:
    """
    用于追踪多线程或者需要分段处理数据，最后将数据进行汇总为标准训练集
    """

    def __init__(self):
        self.data_dict = {}

    def add_data(self, data_dict: dict, prefix: str = ''):

        if len(prefix) > 0:
            prefix = prefix + '/'

        for key, value in data_dict.items():

            all_name = prefix + key

            if all_name not in self.data_dict:
                self.data_dict[all_name] = []
            self.data_dict[all_name].append(value)

    def gett_data(self):
        data_dict = {}
        for key, value in self.data_dict.items():
            data_dict[key] = np.vstack(value)

        return data_dict
