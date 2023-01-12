import abc
from AquaML.data.DataPool import DataPool
from AquaML.DataType import RLIOInfo
from AquaML.data.DataUnit import DataUnit
import numpy as np


class BaseRLalgo(abc.ABC):

    # TODO:统一输入接口
    # TODO:判断是否启动多线程   

    def __init__(self, rl_io_info:RLIOInfo, name:str ,computer_type:str='PC', level:int=0, thread_ID:int=-1):
        """create base for reinforcement learning algorithm.
        This base class provides exploration policy, data pool(multi thread).
        Some tools are also provided for reinforcement learning algorithm such as 
        caculate general advantage estimation. 


        Args:
            rl_io_info (RLIOInfo): reinforcement learning input and output information.
            name (str): reinforcement learning name.
            computer_type (str, optional): 'PC' or 'HPC'. Defaults to 'PC'.
            level (int, optional): thread level. 0 means main thread, 1 means sub thread. Defaults to 0.
            thread_ID (int, optional): ID is given by mpi. -1 means single thread. Defaults to -1.

        Raises:
            ValueError: if thread_ID == -1, it means single thread, then level must be 0.
        """

        self.rl_io_info = rl_io_info
        self._computer_type = computer_type
        self.level = level
        self.thread_ID = thread_ID # if thread_ID == -1, it means single thread
        self.name = name

        # check thread level
        # if thread_ID == -1, it means single thread, then level must be 0
        if self.thread_ID == -1 and self.level != 0:
            raise ValueError('If thread_ID == -1, it means single thread, then level must be 0.')

        # create data pool according to thread level
        self.data_pool = DataPool(name=self.name,level=self.level, computer_type=self._computer_type) # data_pool is a handle

        # multi thread sync
        if self.thread_ID > -1: # multi thread
            self.data_pool.multi_sync(self.rl_io_info.data_info, type='buffer')
        else: # single thread
            self.data_pool.create_buffer_from_dic(self.rl_io_info.data_info.buffer_dict)


    # calculate general advantage estimation
    def calculate_GAE(self, rewards, values, next_values, masks, gamma, lamda):
        """
        calculate general advantage estimation.
        https://arxiv.org/abs/1506.02438
        Args:
            rewards (np.ndarray): rewards.
            values (np.ndarray): values.
            next_values (np.ndarray): next values.
            masks (np.ndarray): dones.
            gamma (float): discount factor.
            lamda (float): general advantage estimation factor.
        Returns:
            np.ndarray: general advantage estimation.
        """
        gae = np.zeros_like(rewards)
        n_steps_target = np.zeros_like(rewards)
        cumulated_advantage = 0.0
        length = len(rewards)
        index = length - 1

        for i in range(length):
            index = index - 1
            delta = rewards[index] + gamma * next_values[index]  - values[index]
            cumulated_advantage = gamma * lamda * masks[index] * cumulated_advantage + delta
            gae[index] = cumulated_advantage
            n_steps_target[index] = gae[index] + values[index]

        return gae, n_steps_target

    # calculate episode reward information
    def cal_episode_info(self):
        pass

    # get action
    def get_action_train(self, obs:dict):
        pass

    # Gaussian exploration policy

            

