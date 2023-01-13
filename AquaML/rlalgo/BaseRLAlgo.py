import abc
import tensorflow as tf
from AquaML.data.DataPool import DataPool
from AquaML.DataType import RLIOInfo
from AquaML.data.DataUnit import DataUnit
from AquaML.rlalgo import ExplorePolicy
import numpy as np

class BaseRLalgo(abc.ABC):

    # TODO:统一输入接口
    # TODO:判断是否启动多线程 (done)  

    def __init__(self, rl_io_info:RLIOInfo, name:str ,computer_type:str='PC', level:int=0, thread_ID:int=-1):
        """create base for reinforcement learning algorithm.
        This base class provides exploration policy, data pool(multi thread).
        Some tools are also provided for reinforcement learning algorithm such as 
        calculate general advantage estimation. 
        
        When you create a reinforcement learning algorithm, you should inherit this class. And do the following things:
        
        And you should run init() function in your __init__ function. The position of init() function is at the end of __init__ function.

        You need to point out which model is the actor, then in __init__ function, you should write:
        "self.actor = actor"
        
        Explore policy is a function which is used to generate action. You should use or create a explore policy. 
        Then in __init__ function, you should write:
        "self.explore_policy = explore_policy"
        You can create a explore policy by inherit ExplorePolicyBase class(AquaML.rlalgo.ExplorePolicy.ExplorePolicyBase). 
        
        
        
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
        
        self.actor = None # actor model. Need point out which model is the actor.

       
            
        self.__explore_init_dict = {} # store explore policy, convenient for multi thread

    
    # initial algorithm
    def init(self):
        """initial algorithm.
        """
        # multi thread initial
        if self.thread_ID > -1: # multi thread
            self.data_pool.multi_init(self.rl_io_info.data_info, type='buffer')
        else: # single thread
            self.data_pool.create_buffer_from_dic(self.rl_io_info.data_info.buffer_dict)
        
        # check some information
        
        # actor model must be given
        if self.actor == None:
            raise ValueError('Actor model must be given.')
        
    # TODO: calculate by multi thread
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

    # calculate discounted reward
    def calculate_discounted_reward(self, rewards, masks, gamma):
        """
        calculate discounted reward.
        Args:
            rewards (np.ndarray): rewards.
            masks (np.ndarray): dones. if done, mask = 0, else mask = 1.
            gamma (float): discount factor.
        Returns:
            np.ndarray: discounted reward.
        """
        discounted_reward = np.zeros_like(rewards)
        cumulated_reward = 0.0
        length = len(rewards)
        index = length - 1

        for i in range(length):
            index = index - 1
            cumulated_reward = rewards[index] + gamma * cumulated_reward * masks[index]
            discounted_reward[index] = cumulated_reward

        return discounted_reward

    # calculate episode reward information
    def cal_episode_info(self):
        """
        calculate episode reward information.

        Returns:
            _type_: dict. summary reward information.
        """
        index_done = np.where(self.data_pool.get_unit_data('mask') == 0)[0] + 1
        start_index = 0
        
        # create list for every reward information
        reward_dict = {}
        for key in self.rl_io_info.reward_info:
            reward_dict[key] = []
            
        # spit reward information according to done
        for end_index in index_done:
            for key in self.rl_io_info.reward_info:
                reward_dict[key].append(np.sum(self.data_pool.get_unit_data(key)[start_index:end_index]))
            start_index = end_index
        
        # summary reward information
        reward_summary = {}
        
        reward_summary['std'] = np.std(reward_dict['total_reward'])
        reward_summary['max_reward'] = np.max(reward_dict['total_reward'])
        reward_summary['min_reward'] = np.min(reward_dict['total_reward'])
        
        for key in self.rl_io_info.reward_info:
            reward_summary[key] = np.mean(reward_dict[key])
            
        # delete list
        del reward_dict
        
        return reward_summary
        

    # get action in the training process
    def get_action_train(self, obs:dict):
        # TODO:根据网络模型的特点需要修改
        
        # get actor input
        input_obs = []
        
        for key in self.rl_io_info.actor_input_info:
            input_obs.append(obs[key])

    # Gaussian exploration policy
    def create_gaussian_exploration_policy(self):
        # verity the style of log_std
        if self.rl_io_info.explore_info == 'self':
            # log_std provided by actor
            # create explore policy
            pass
        elif self.rl_io_info.explore_info == 'auxiliary':
            # log_std provided by auxiliary variable
            # create args by data unit
            self.log_std = DataUnit(name=self.name+'_log_std', data_type=np.float32, shape=self.rl_io_info.action_info['action'].shape, 
                                    level=self.level, computer_type=self._computer_type)
            
            self.log_std.set_value(np.zeros(self.rl_io_info.action_info['action'].shape, dtype=np.float32)-0.5)
            self.tf_log_std = tf.Variable(self.log_std.get_value(), trainable=True)
        
        self.explore_policy = ExplorePolicy(shape = self.rl_io_info.action_info['action'])
        
        # add initial information
        self.rl_io_info.add_info(name='log_std',shape=self.log_std.shape, data_type=self.log_std.data_type)
        self.data_pool.add_unit(name='log_std', data_unit=self.log_std)
        