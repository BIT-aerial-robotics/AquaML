'''
这里会提供一些基本的散装工具。

'''
import numpy as np
try:
    import gymnasium as gym
except:
    import gym
# from AquaML.param.DataInfo import DataInfo
# import tensorflow as tf


class BoolFlag:
    '''
    这个类用来管理一些标志位。
    '''

    def __init__(self):
        self.flag = False

    def set(self):
        self.flag = True

    def clear(self):
        self.flag = False

    def get(self):
        return self.flag


class IDFlag:
    '''
    这个类用来管理一些标志位。
    '''

    def __init__(self):
        self.flag = 0

    def set(self, flag):
        self.flag = flag

    def clear(self):
        self.flag = 0

    def get(self):
        return self.flag


class LossTracker:
    def __init__(self):
        self.loss_dict = {}

    def add_data(self, loss_dict: dict, prefix: str = '', from_tf=False):

        # new_dict = {}
        #
        # if from_tf:
        #     for key, value in loss_dict.items():
        #         new_dict[key] = value.numpy()
        # else:
        #     new_dict.update(loss_dict)

        if len(prefix) > 0:
            prefix = prefix + '/'

        for key, value in loss_dict.items():

            all_name = prefix + key

            if all_name not in self.loss_dict:
                self.loss_dict[all_name] = []
            # if not isinstance(value, (int, float, np.float32, np.ndarray)):
            #     if isinstance(value, (tf.Tensor)):
            #         value = np.squeeze(value.numpy)
            #     else:
            #         raise ValueError('loss value must be int, float or np.ndarray')

            self.loss_dict[all_name].append(value)
            # if isinstance(value, tf.Tensor):
            #     self.loss_dict[all_name].append(value.numpy())
            # else:
            #     self.loss_dict[all_name].append(value)
        return loss_dict

    def reset(self):
        self.loss_dict = {}

    def get_data(self):
        loss_dict = {}
        for key, value in self.loss_dict.items():
            # array = deepcopy(value)
            value = np.vstack(value)
            # value = np.squeeze(value)
            loss_dict[key + '_max'] = np.max(value)
            loss_dict[key + '_min'] = np.min(value)
            # print(value)
            if value.shape[0] == 1:
                loss_dict[key] = np.squeeze(value)
            else:
                loss_dict[key] = np.mean(value)

            if 'reward' in key:
                loss_dict[key + '_std'] = np.std(value)

            if 'indicate' in key:
                loss_dict[key + '_std'] = np.std(value)
        # return loss_dict

        # 获取data以后自动释放内存
        self.reset()
        return loss_dict


# def convert_dtype2numpy():


def generate_gym_env_info(env_name: str, env_param: dict = {}):
    '''
    用于生成gym环境的信息。
    
    Args:
        env_name (str): gym环境的名称。
        env_param (dict, optional): gym环境的参数。 Defaults to {}.
    Returns:
        info_element_dict (dict): 数据信息字典。
        rl_state_names (list): 强化学习的状态名称。
        rl_action_name (list): 强化学习的动作名称。
    '''

    info_element_dict = {}

    env = gym.make(env_name, **env_param)

    # 获取状态空间的信息
    observation_space_shape: tuple = env.observation_space.shape

    observation_space_dtype = env.observation_space.dtype

    info_element_dict['observation'] = {
        'name': 'env_obs',
        'dtype': observation_space_dtype.type,  # 目前只支持float32
        'shape': observation_space_shape,
        'size': 1
    }

    # 获取动作空间的信息
    action_space_shape: tuple = env.action_space.shape  # 修改dtype

    action_space_dtype = env.action_space.dtype

    info_element_dict['action'] = {
        'name': 'env_action',
        'dtype': action_space_dtype.type,
        'shape': action_space_shape,
        'size': 1
    }

    # 添加奖励信息
    info_element_dict['reward'] = {
        'name': 'env_reward',
        'dtype': np.float32,
        'shape': (1,),
        'size': 1
    }

    # 添加mask信息
    info_element_dict['mask'] = {
        'name': 'env_mask',
        'dtype': np.bool_,
        'shape': (1,),
        'size': 1
    }

    rl_state_names = ['env_obs']
    rl_action_name = ['env_action']

    rl_reward_names = ['env_reward']

    return info_element_dict, rl_state_names, rl_action_name, rl_reward_names


def dtype2str(dtype):
    '''
    将dtype转换为str。

    Args:
        dtype (np.dtype): 数据类型。

    Returns:
        str: 数据类型的字符串。
    '''
    if dtype == np.float32:
        return 'np.float32'
    elif dtype == np.int32:
        return 'np.int32'
    elif dtype == np.bool_:
        return 'np.bool_'
    elif dtype == np.float64:
        return 'np.float64'
    elif dtype == np.int64:
        return 'np.int64'
    elif dtype == np.uint8:
        return 'np.uint8'
    elif dtype == np.uint16:
        return 'np.uint16'
    elif dtype == np.uint32:
        return 'np.uint32'
    else:
        raise ValueError('dtype not supported: {}'.format(dtype))
