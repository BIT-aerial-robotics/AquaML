import abc
import tensorflow as tf
from AquaML.data.DataPool import DataPool
from AquaML.DataType import RLIOInfo
from AquaML.data.DataUnit import DataUnit
from AquaML.rlalgo.ExplorePolicy import GaussianExplorePolicy, VoidExplorePolicy, CategoricalExplorePolicy
from AquaML.tool.RLWorker import RLWorker
from AquaML.BaseClass import BaseAlgo
from AquaML.data.ArgsPool import ArgsPool
import numpy as np
from AquaML.tool.Recoder import Recoder
import json
import os


# import time

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


def compute_advantage(gamma, lmbda, td_delta):
    advantage_list = []
    advantage = 0.0

    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return advantage_list


# TODO:model logic has been changed, check the new version
# TODO: 优化命名方式
class BaseRLAlgo(BaseAlgo, abc.ABC):

    # TODO:统一输入接口
    # TODO:判断是否启动多线程 (done)  

    def __init__(self, env,
                 rl_io_info: RLIOInfo,
                 name: str,
                 hyper_parameters,
                 update_interval: int = 0,
                 mini_buffer_size: int = 0,
                 calculate_episodes=5,
                 display_interval=1,
                 prefix_name=None,
                 computer_type: str = 'PC',
                 level: int = 0,
                 thread_ID: int = -1,
                 total_threads: int = 1,
                 policy_type: str = 'off'):
        """create base for reinforcement learning algorithm.
        This base class provides exploration policy, data pool(multi thread).
        Some tools are also provided for reinforcement learning algorithm such as
        calculate general advantage estimation.

        When you create a reinforcement learning algorithm, you should inherit this class. And do the following things:

        1. You should run init() function in your __init__ function. The position of init() function is at the end of __init__ function.

        2. You need to point out which model is the actor, then in __init__ function, you should write:
           "self.actor = actor"

        3. Explore policy is a function which is used to generate action. You should use or create a explore policy.
        Then in __init__ function, you should write:
        "self.explore_policy = explore_policy"
        You can create a explore policy by inherit ExplorePolicyBase class(AquaML.rlalgo.ExplorePolicy.ExplorePolicyBase).

        4. Notice: after optimize the model, you should update optimize_epoch.
           The same as sample_epoch.

        5. Notice: if you want to use multi thread, please specify which model needs to be synchronized by
           setting  self._sync_model_dict


        Some recommends for off-policy algorithm:
        1. mini_buffer_size should have given out.



        Args:
            env (AquaML.rlalgo.EnvBase): reinforcement learning environment.

            rl_io_info (RLIOInfo): reinforcement learning input and output information.

            name (str): reinforcement learning name.

            update_interval (int, optional): update interval. This is an important parameter.
            It determines how many steps to update the model. If update_interval == 1, it means update model after each step.
            if update_interval == 0, it means update model after buffer is full. Defaults to 0.
            That is unusually used in on-policy algorithm.
            if update_interval > 0, it is used in off-policy algorithm.

            mini_buffer_size (int, optional): mini buffer size. Defaults to 0.

            calculate_episodes (int): How many episode will be used to summary. We recommend to use 1 when using multi thread.

            display_interval (int, optional): display interval. Defaults to 1.

            computer_type (str, optional): 'PC' or 'HPC'. Defaults to 'PC'.

            level (int, optional): thread level. 0 means main thread, 1 means sub thread. Defaults to 0.

            thread_ID (int, optional): ID is given by mpi. -1 means single thread. Defaults to -1.

            total_threads (int, optional): total threads. Defaults to 1.

            policy_type (str, optional): 'off' or 'on'. Defaults to 'off'.

        Raises:
            ValueError: if thread_ID == -1, it means single thread, then level must be 0.
            ValueError: if computer_type == 'HPC', then level must be 0.
            ValueError('Sync model must be given.')

        """

        # self.args_pool = None
        self.adjust_parameters = None
        self.rnn_actor_flag = None
        self.expand_dims_idx = ()
        # self.expand_dims_name = None
        self.recoder = None
        self.explore_policy = None
        self.tf_log_std = None
        self.log_std = None
        self.rl_io_info = rl_io_info
        # if thread_ID == -1, it means single thread
        self.name = name
        self.env = env
        self.update_interval = update_interval
        self.mini_buffer_size = mini_buffer_size
        self.display_interval = display_interval
        self.hyper_parameters = hyper_parameters

        self.store_counter = 0

        # meta配置参数
        self.meta_parameter_names = None
        self.args_pool = None

        # parameter of multithread
        self._computer_type = computer_type
        self.level = level
        self.thread_ID = thread_ID
        self.total_threads = total_threads  # main thread is not included
        if self.total_threads == 1:
            self.sample_threads = total_threads
        else:
            self.sample_threads = total_threads - 1
        self.each_thread_summary_episodes = calculate_episodes
        '''
        If it runs in single thread, self.thread_ID == 0, self.total_threads == 1
        '''

        # config cache file
        # self.data_pool_info_path = self.
        if prefix_name is not None:
            name = prefix_name + '/' + name
        self.cache_path = name + '/cache'  # cache path
        self.meta_path = name + '/meta'
        self.log_path = name + '/log'
        self.data_pool_info_file = self.meta_path + '/data_pool_config.json'

        self.args_pool_info_file = self.meta_path + '/args_pool_config.json'

        self.history_model_path = name + '/' + 'history_model'

        mkdir(self.meta_path)
        mkdir(self.cache_path)
        mkdir(self.log_path)

        # create data pool according to thread level
        self.data_pool = DataPool(name=self.name, level=self.level,
                                  computer_type=self._computer_type)  # data_pool is a handle

        self.actor = None  # actor model. Need point out which model is the actor.

        # TODO: 私有变量会出问题，貌似这个没用
        self._explore_dict = {}  # store explore policy, convenient for multi thread

        self.allocate_data_pool()

        # create worker
        if self.total_threads > 1:
            if self.level == 0:
                # self.env = None
                self.worker = None
            else:
                self.worker = RLWorker(self)
        else:
            self.worker = RLWorker(self)

        # initial main thread
        if self.level == 0:
            # resample action
            # TODO: 优化此处命名
            if self.rl_io_info.explore_info == 'self-std':
                self.resample_action = self._resample_action_log_std
                self.resample_log_prob = self._resample_log_prob_with_std
            elif self.rl_io_info.explore_info == 'global-std':
                self.resample_action = self._resample_action_no_log_std
                self.resample_log_prob = self._resample_log_prob_no_std
            elif self.rl_io_info.explore_info == 'void-std':
                self.resample_action = self._resample_action_log_prob
                self.resample_log_prob = None
            elif self.rl_io_info.explore_info == 'discrete':
                self.resample_action = None
                self.resample_log_prob = self._resample_log_prob_discrete

        if self.rl_io_info.explore_info == 'discrete':
            self.create_explore_policy = self.create_categorical_exploration_policy
        else:
            self.create_explore_policy = self.create_gaussian_exploration_policy

        # hyper parameters
        # the hyper parameters is a dictionary
        # you should point out the hyper parameters in your algorithm
        # will be used in optimize function

        # optimizer are created in main thread
        self.optimizer_dict = {}  # store optimizer, convenient search

        self.total_segment = self.sample_threads  # total segment, convenient for multi

        self.sample_epoch = 0  # sample epoch
        self.optimize_epoch = 0  # optimize epoch

        self.policy_type = policy_type  # 'off' or 'on'

        # mini buffer size
        # according to the type of algorithm,

        self._sync_model_dict = {}  # store sync model, convenient for multi thread
        self._sync_explore_dict = {}  # store sync explore policy, convenient for multi thread

        self._all_model_dict = {}  # store all model, convenient to record model

        self._used_characters = ('reward',)

    # initial algorithm
    ############################# key component #############################
    def init(self):
        """initial algorithm.

        This function will be called by starter.
        """
        # multi thread communication

        reward_info_dict = {}

        for name in self.rl_io_info.reward_info:
            reward_info_dict['summary_' + name] = (
                self.total_segment * self.each_thread_summary_episodes, 1)

        # add summary reward information to data pool
        for name, shape in reward_info_dict.items():
            # this must be first level name
            buffer = DataUnit(name=self.name + '_' + name, shape=shape, dtype=np.float32, level=self.level,
                              computer_type=self._computer_type)
            self.rl_io_info.add_info(name=name, shape=shape, dtype=np.float32)
            self.data_pool.add_unit(name=name, data_unit=buffer)

        # TODO:子线程需要等待时间 check
        # multi thread initial
        if self.total_threads > 1:  # multi thread
            # print(self.rl_io_info.data_info)
            self.data_pool.multi_init(self.rl_io_info.data_info, type='buffer')
        else:  # single thread
            self.data_pool.create_buffer_from_dict(self.rl_io_info.data_info)

        # just do in m main thread
        if self.level == 0:
            # initial recoder
            # history_model_path = name + '/' + 'history_model'
            self.recoder = Recoder(log_folder=self.log_path, history_model_log_folder=self.history_model_path, )
            self.data_pool.save_units_info_json(self.data_pool_info_file)

        else:
            self.recoder = None

        # self.init_args_pool()

        # check some information
        # actor model must be given
        if self.actor is None:
            raise ValueError('Actor model must be given.')

    # meta initialize
    def meta_init(self):
        """meta initialize.

        This function will be called by starter.
        """

        reward_info_dict = {}

        for name in self.rl_io_info.reward_info:
            reward_info_dict['summary_' + name] = (
                self.total_segment * self.each_thread_summary_episodes, 1)

        # add summary reward information to data pool
        for name, shape in reward_info_dict.items():
            # this must be first level name
            buffer = DataUnit(name=self.name + '_' + name, shape=shape, dtype=np.float32, level=self.level,
                              computer_type=self._computer_type)
            self.rl_io_info.add_info(name=name, shape=shape, dtype=np.float32)
            self.data_pool.add_unit(name=name, data_unit=buffer)

        # meta need to initialize the data pool by shared memory

        self.data_pool.multi_init(self.rl_io_info.data_info, type='buffer')

        # just do in m main thread
        if self.level == 0:
            # initial recoder
            # history_model_path = self.name + '/' + 'history_model'
            self.recoder = Recoder(log_folder=self.log_path, history_model_log_folder=self.history_model_path, )
            self.data_pool.save_units_info_json(self.data_pool_info_file)
        else:
            self.recoder = None

        self.init_args_pool()

    def init_args_pool(self):
        # summary how many parameters should be automatically adjusted
        adjust_parameters = []
        # hyper parameters
        for name in self.hyper_parameters.adjust_parameters:
            adjust_parameters.append(name)
        # environment parameters
        for name in self.env.adjust_parameters:
            adjust_parameters.append(name)

        self.adjust_parameters = adjust_parameters

        self.args_pool = ArgsPool(
            name=self.name,
            level=self.level,
            computer_type=self._computer_type,
        )

        self.args_pool.create_buffer_from_tuple(self.adjust_parameters)
        if self.level == 0:
            self.args_pool.create_shared_memory()
            self.args_pool.save_units_info_json(self.args_pool_info_file)

    def read_args_pool(self):
        if self.level == 0:
            self.args_pool.read_shared_memory_V2()

    # recreate data pool
    def recreate_data_pool(self):
        """recreate data pool.

        This function will be called by higher level algorithm.

        Before recreate data pool, you should clear the data pool.
        """

        pool_info_dict = json.load(open(self.data_pool_info_file, 'r'))
        self.data_pool.create_buffer_from_dict_direct(pool_info_dict, prefix=self.name)

    def reread_data_pool(self):
        """reread data pool.

        This function will be called by higher level algorithm.
        """
        pool_info_dict = json.load(open(self.data_pool_info_file, 'r'))
        self.data_pool.read_shared_memory_from_dict_direct(pool_info_dict)

    def clear_data_pool(self):
        """clear data pool.

        This function will be called by higher level algorithm.
        """
        self.data_pool.clear()

    def initialize_actor_config(self):
        # initialize sample action
        self.rnn_actor_flag = getattr(self.actor, 'rnn_flag', False)

        if self.rnn_actor_flag:
            self.expand_dims_idx = []

            idx = 0

            names = self.actor.input_name
            for name in names:
                if 'hidden' in name:
                    pass
                else:
                    self.expand_dims_idx.append(idx)
                idx += 1
            self.expand_dims_idx = tuple(self.expand_dims_idx)

            # if self.hyper_parameters.batch_trajectory:
            #     for name in names:
            #         if 'hidden' in name:
            #             raise ValueError('Batch trajectory mode dose not need to input hidden state.')

    def allocate_data_pool(self):

        """allocate data pool.
        Each thread use different part of data pool.
        """

        if self._computer_type == 'PC':
            if self.thread_ID > 0:
                # thread ID start from 0
                self.each_thread_size = int(self.rl_io_info.buffer_size / self.sample_threads)
                self.each_thread_start_index = int((self.thread_ID - 1) * self.each_thread_size)

                self.max_buffer_size = self.each_thread_size * self.sample_threads

                # if mini_buffer_size == 0, it means pre-sample data is disabled
                self.each_thread_mini_buffer_size = int(self.mini_buffer_size / self.sample_threads)
                self.mini_buffer_size = int(self.each_thread_mini_buffer_size * self.sample_threads)

                if self.update_interval == 0:
                    # 这种情形属于将所有buffer填充满以后再更新模型
                    # if update_interval == 0, it means update model after buffer is full
                    self.each_thread_update_interval = self.each_thread_size  # update interval for each thread
                else:
                    # if update_interval != 0, it means update model after each step
                    # then we need to calculate how many steps to update model for each thread
                    # 每个线程更新多少次等待更新模型
                    self.each_thread_update_interval = int(
                        self.update_interval / self.sample_threads)  # update interval for each thread
                if self.level > 0:
                    self.sample_id = self.thread_ID - 1
                else:
                    self.sample_id = 0
            else:
                self.each_thread_size = self.rl_io_info.buffer_size
                self.each_thread_start_index = 0
                self.each_thread_mini_buffer_size = self.mini_buffer_size
                if self.update_interval == 0:
                    self.each_thread_update_interval = self.each_thread_size  # update interval for each thread
                else:
                    self.each_thread_update_interval = self.update_interval  # update interval for each thread

                self.max_buffer_size = self.each_thread_size

                self.thread_ID = 0
                self.sample_id = 0  # sample id is used to identify which thread is sampling data

                # self.each_thread_update_interval = self.update_interval # update interval for each thread

        else:
            # TODO: HPC will implement in the future
            self.each_thread_size = None
            self.each_thread_start_index = None
            self.each_thread_update_interval = None

    def optimize(self):

        # compute current reward information

        optimize_info = self._optimize_()

        # all the information update here
        self.optimize_epoch += 1

        total_steps = self.get_current_steps

        optimize_info['total_steps'] = total_steps

        if self.optimize_epoch % self.display_interval == 0:
            # display information

            epoch = int(self.optimize_epoch / self.display_interval)
            reward_info = self.summary_reward_info()
            print("###############epoch: {}###############".format(epoch))
            self.recoder.display_text(
                reward_info
            )
            self.recoder.display_text(
                optimize_info
            )

            self.recoder.record(reward_info, total_steps, prefix='reward')
            self.recoder.record(optimize_info, self.optimize_epoch, prefix=self.name)
            # record weight
            for key, model in self._all_model_dict.items():
                self.recoder.record_weight(model, total_steps, prefix=key)

            info = {**reward_info, **optimize_info}
            if epoch % self.hyper_parameters.store_model_times == 0:
                self.recoder.recorde_history_model(
                    self._all_model_dict,
                    epoch,
                    info
                )

    def check(self):
        """
        check some information.
        """
        if self.policy_type == 'off':
            if self.mini_buffer_size is None:
                raise ValueError('Mini buffer size must be given.')
        if self._sync_model_dict is None:
            raise ValueError('Sync model must be given.')

        if self.level == 0:
            if len(self._all_model_dict) < 1:
                raise ValueError("Model dictionary is void! Please check _all_model_dict!")

                # check some information
                # actor model must be given
        if self.actor is None:
            raise ValueError('Actor model must be given.')

    ############################# key function #############################
    def store_data(self, obs: dict, action: dict, reward: dict, next_obs: dict, mask: int):
        """
        store data to buffer.

        Args:
            obs (dict): observation. eg. {'obs':np.array([1,2,3])}
            action (dict): action. eg. {'action':np.array([1,2,3])}
            reward (dict): reward. eg. {'reward':np.array([1,2,3])}
            next_obs (dict): next observation. eg. {'next_obs':np.array([1,2,3])}
            mask (int): done. eg. 1 or 0
        """
        # store data to buffer
        # support multi thread

        self.store_counter += 1
        idx = (self.store_counter - 1) % self.each_thread_size
        index = self.each_thread_start_index + idx  # index in each thread

        # store obs to buffer
        self.data_pool.store(obs, index)

        # store next_obs to buffer
        self.data_pool.store(next_obs, index, prefix='next_')

        # store action to buffer
        self.data_pool.store(action, index)

        # store reward to buffer
        self.data_pool.store(reward, index)

        # store mask to buffer
        self.data_pool.data_pool['mask'].store(mask, index)

    @staticmethod
    def copy_weights(model1, model2):
        """
        copy weight from model1 to model2.
        """
        new_weights = []
        target_weights = model1.weights

        for i, weight in enumerate(model2.weights):
            new_weights.append(target_weights[i].numpy())

        model2.set_weights(new_weights)

    @staticmethod
    def soft_update_weights(model1, model2, tau):
        """
        soft update weight from model1 to model2.


        args:
        model1: source model
        model2: target model
        """
        new_weights = []
        source_weights = model1.weights

        for i, weight in enumerate(model2.weights):
            new_weights.append((1 - tau) * weight.numpy() + tau * source_weights[i].numpy())

        model2.set_weights(new_weights)

    def summary_reward_info(self):
        """
        summary reward information.
        """
        # calculate reward information

        summary_reward_info = {}

        for name in self.rl_io_info.reward_info:
            summary_reward_info[name] = np.mean(self.data_pool.get_unit_data('summary_' + name))

        summary_reward_info['std'] = np.std(self.data_pool.get_unit_data('summary_total_reward'))
        summary_reward_info['max_reward'] = np.max(self.data_pool.get_unit_data('summary_total_reward'))
        summary_reward_info['min_reward'] = np.min(self.data_pool.get_unit_data('summary_total_reward'))

        return summary_reward_info

        # calculate episode reward information

        # random sample

    def random_sample(self, batch_size: int):
        """
            random sample data from buffer.

            Args:
                batch_size (int): batch size.

            Returns:
                _type_: dict. data dict.
            """
        # if using multi thread, then sample data from each segment
        # sample data from each segment

        # compute current segment size
        running_step = self.mini_buffer_size + self.optimize_epoch * self.each_thread_update_interval * self.total_segment
        buffer_size = min(self.max_buffer_size, running_step)

        batch_size = min(batch_size, buffer_size)

        sample_index = np.random.choice(range(buffer_size), batch_size, replace=False)

        # index_bias = (sample_index * 1.0 / self.each_thread_size) * self.each_thread_size
        index_bias = sample_index / self.each_thread_size
        index_bias = index_bias.astype(np.int32)
        index_bias = index_bias * self.each_thread_size

        sample_index = sample_index + index_bias
        sample_index = sample_index.astype(np.int32)

        # get data

        data_dict = self.data_pool.get_data_by_indices(sample_index, self.rl_io_info.store_data_name)

        return data_dict

    def cal_episode_info(self):
        """
            calculate episode reward information.

            Returns:
                _type_: dict. summary reward information.
            """

        # data_dict = self.get_current_update_data(('reward', 'mask'))
        # calculate current reward information

        # get done flag
        index_done = np.where(self.data_pool.get_unit_data('mask') == 0)[0] + 1
        index_done_ = index_done / self.each_thread_size
        index_done_ = index_done_.astype(np.int32)

        # config segment
        segment_index = np.arange((0, self.total_segment))
        every_segment_index = []

        # split index_done
        for segment_id in segment_index:
            segment_index_done = np.where(index_done_ == segment_id)[0]
            every_segment_index.append(index_done[segment_index_done])

        reward_dict = {}
        for key in self.rl_io_info.reward_info:
            reward_dict[key] = []

        for each_segment_index in every_segment_index:
            # get index of done
            compute_index = each_segment_index[-self.each_thread_summary_episodes:]
            start_index = compute_index[0]

            for end_index in compute_index[1:]:
                for key in self.rl_io_info.reward_info:
                    reward_dict[key].append(np.sum(self.data_pool.get_unit_data(key)[start_index:end_index]))
                start_index = end_index

        # summary reward information
        reward_summary = {'std': np.std(reward_dict['total_reward']),
                          'max_reward': np.max(reward_dict['total_reward']),
                          'min_reward': np.min(reward_dict['total_reward'])}

        for key in self.rl_io_info.reward_info:
            reward_summary[key] = np.mean(reward_dict[key])

        # delete list
        del reward_dict

        return reward_summary

    def cal_average_batch_dict(self, data_list: list):
        """
            calculate average batch dict.

            Args:
                data_list (list): store data dict list.

            Returns:
                _type_: dict. average batch dict.
            """
        average_batch_dict = {}
        for key in data_list[0]:
            average_batch_dict[key] = []

        for data_dict in data_list:
            for key, value in data_dict.items():
                average_batch_dict[key].append(value)

        # average
        for key, value in average_batch_dict.items():
            if 'gard' in key:
                pass
            else:
                average_batch_dict[key] = np.mean(value)

        return average_batch_dict

    def get_batch_timesteps(self, data_dict: dict):
        """
        Covert data dict to batch timesteps. The data like (batch_size, timesteps, *)
        after processed by this function.
        """

        # get done index
        index_done = np.where(self.data_pool.get_unit_data('mask') == 0)[0] + 1

        # create buffer to store new data
        new_data_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, tuple) or isinstance(value, list):
                list_lists = []
                for idx, data in enumerate(value):
                    list_lists.append([])
                    data_dict[key][idx] = data.numpy()
                new_data_dict[key] = list_lists
            else:
                new_data_dict[key] = []
                data_dict[key] = value.numpy()

        start_index = 0

        for end_index in index_done:
            for key, value in data_dict.items():
                if isinstance(value, tuple) or isinstance(value, list):
                    for idx, data in enumerate(value):
                        buffer = np.expand_dims(data[start_index:end_index], axis=0)
                        new_data_dict[key][idx].append(buffer)
                else:
                    buffer = np.expand_dims(value[start_index:end_index], axis=0)
                    new_data_dict[key].append(buffer)

            start_index = end_index

        # batch timesteps
        for key, value in data_dict.items():
            if isinstance(value, tuple) or isinstance(value, list):
                value = new_data_dict[key]
                for idx, data in enumerate(value):
                    buffer = np.vstack(data)
                    buffer = tf.cast(buffer, dtype=tf.float32)
                    new_data_dict[key][idx] = buffer
            else:
                value = new_data_dict[key]
                buffer = np.vstack(value)
                buffer = tf.cast(buffer, tf.float32)
                new_data_dict[key] = buffer

        return new_data_dict

    # TODO: calculate by multi thread
    ############################# calculate reward information #############################
    # calculate general advantage estimation
    def calculate_GAE(self, rewards, values, next_values, masks, gamma, lamda):
        """
        calculate general advantage estimation.

        Reference:
        ----------
        [1] Schulman J, Moritz P, Levine S, Jordan M, Abbeel P. High-dimensional continuous
        control using generalized advantage estimation. arXiv preprint arXiv:1506.02438. 2015 Jun 8.

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
        index = length

        # td_target = rewards + gamma * next_values * masks
        # td_delta = td_target - values
        # advantage = compute_advantage(gamma, lamda, td_delta)
        for i in range(length):
            index -= 1
            delta = rewards[index] + gamma * next_values[index] - values[index]
            cumulated_advantage = gamma * lamda * masks[index] * cumulated_advantage + delta
            gae[index] = cumulated_advantage
            n_steps_target[index] = gae[index] + values[index]

        # return advantage, td_target

        return gae, n_steps_target

    def calculate_GAEV2(self, rewards: tf.Tensor, values: tf.Tensor, next_values: tf.Tensor,
                        masks: tf.Tensor, gamma: tf.Variable, lamda: tf.Variable):

        """
        为了兼容meta，2.1版本中所有的td之类的必须支持tf的自动微分。
        """
        length = len(rewards)

        # 创建指定大小空list
        gae = [None] * length
        n_steps_target = [None] * length

        cumulated_advantage = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        index = length

        for i in range(length):
            index -= 1
            delta = rewards[index] + gamma * next_values[index] - values[index]
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

    ############################# create function #############################
    # create keras optimizer
    def create_optimizer(self, name: str, optimizer: str, lr: float):
        """
        # TODO: v2.1需要升级改函数操作太麻烦不建议使用
        create keras optimizer for each model.

        Reference:
            https://keras.io/optimizers/

        Args:
            name (str): name of this optimizer, you can call by this name.
            if name is 'actor', then you can call self.actor_optimizer

            optimizer (str): type of optimizer. eg. 'Adam'. For more information,
            please refer to keras.optimizers.

            lr (float): learning rate.
        """

        attribute_name = name + '_optimizer'

        # in main thread, create optimizer
        if self.level == 0:
            # create optimizer
            optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=lr)
        else:
            # None
            optimizer = None

        # set attribute
        setattr(self, attribute_name, optimizer)

        self.optimizer_dict[name] = getattr(self, attribute_name)

    def create_none_optimizer(self, name: str):
        """
        create none optimizer for each model.

        Args:
            name (str): name of this optimizer, you can call by this name.
        """

        attribute_name = name + '_optimizer'
        optimizer = None
        setattr(self, attribute_name, optimizer)

    # Gaussian exploration policy
    def create_gaussian_exploration_policy(self):
        # TODO: sync with tf_std
        # verity the style of log_std
        if self.rl_io_info.explore_info == 'self-std':
            # log_std provided by actor
            # create explore policy
            # self.__explore_dict = None
            self.explore_policy = GaussianExplorePolicy(shape=self.rl_io_info.actor_out_info['action'])

        elif self.rl_io_info.explore_info == 'global-std':
            # log_std provided by auxiliary variable
            # create args by data unit
            self.log_std = DataUnit(name=self.name + '_log_std', dtype=np.float32,
                                    shape=self.rl_io_info.actor_out_info['action'],
                                    level=self.level, computer_type=self._computer_type)
            if self.level == 0:
                self.log_std.set_value(np.zeros(self.rl_io_info.actor_out_info['action'], dtype=np.float32) - 0.5)
            self.tf_log_std = tf.Variable(self.log_std.buffer, trainable=True)
            self._explore_dict = {'log_std': self.tf_log_std}

            self.rl_io_info.add_info(name='log_std', shape=self.log_std.shape, dtype=self.log_std.dtype)
            self.data_pool.add_unit(name='log_std', data_unit=self.log_std)
            self.explore_policy = GaussianExplorePolicy(shape=self.rl_io_info.actor_out_info['action'])
        elif self.rl_io_info.explore_info == 'void-std':
            # log_std is void
            self.explore_policy = VoidExplorePolicy(shape=self.rl_io_info.actor_out_info['action'])

    # create categorical exploration policy
    def create_categorical_exploration_policy(self):

        self.explore_policy = CategoricalExplorePolicy(shape=self.rl_io_info.actor_out_info['action'])

    ############################# get function ################################
    def get_action(self, obs: dict, test_flag: bool = False):
        """

        sample action in the training process.

        Args:
            obs (dict): observation from environment. eg. {'obs':data}.
                        The data must be tensor. And its shape is (batch, feature).

        Returns:
            _type_: _description_
        """

        input_data = []

        # get actor input
        for key in self.actor.input_name:
            input_data.append(tf.cast(obs[key], dtype=tf.float32))

        # expand_dims
        for idx in self.expand_dims_idx:
            input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        actor_out = self.actor(*input_data)  # out is a tuple

        policy_out = dict(zip(self.actor.output_info, actor_out))

        for name, value in self._explore_dict.items():
            policy_out[name] = value

        action, prob = self.explore_policy(policy_out, test_flag=test_flag)

        policy_out['action'] = action
        policy_out['prob'] = prob

        # create return dict according to rl_io_info.actor_out_name
        return_dict = dict()
        for name in self.rl_io_info.actor_out_name:
            return_dict[name] = policy_out[name]

        return return_dict

    # def get_action_rnn(self, obs: dict):
    #     """
    #
    #     sample action with rnn model.
    #
    #     Args:
    #         obs (dict): observation from environment. eg. {'obs':data}.
    #                     The data must be tensor. And its shape is (batch, feature).
    #
    #     Returns:
    #         _type_: _description_
    #     """
    #
    #     input_data = []
    #
    #     # get actor input
    #     for key in self.actor.input_name:
    #         input_data.append(tf.cast(obs[key], dtype=tf.float32))
    #
    #     actor_out = self.actor(*input_data)  # out is a tuple
    #
    #     policy_out = dict(zip(self.actor.output_info, actor_out))
    #
    #     for name, value in self._explore_dict.items():
    #         policy_out[name] = value
    #
    #     action, prob = self.explore_policy(policy_out)
    #
    #     policy_out['action'] = action
    #     policy_out['prob'] = prob
    #
    #     # create return dict according to rl_io_info.actor_out_name
    #     return_dict = dict()
    #     for name in self.rl_io_info.actor_out_name:
    #         return_dict[name] = policy_out[name]
    #
    #     return return_dict

    def get_batch_data(self, data_dict: dict, start_index, end_index):
        """
        Get batch data from data dict.

        The data type stored in data_dict must be tuple or tensor or array.

        Example:
            >>> data_dict = {'obs':(np.array([1,2,3,4,5,6,7,8,9,10]),)}
            >>> start_index = 0
            >>> end_index = 5
            >>> self.get_batch_data(data_dict, start_index, end_index)
            {'obs': (array([1, 2, 3, 4, 5]),)}

        Args:
            data_dict (dict): data dict.
            start_index (int): start index.
            end_index (int): end index.
        Returns:
            batch data. dict.
        """
        batch_data = dict()
        for key, values in data_dict.items():
            if isinstance(values, tuple) or isinstance(values, list):
                buffer = []
                for value in values:
                    buffer.append(value[start_index:end_index])
                batch_data[key] = tuple(buffer)
            else:
                batch_data[key] = values[start_index:end_index]

        return batch_data

    # get trainable actor
    @property
    def get_trainable_actor(self):
        """
        get trainable weights of this model.

        actor model is special, it has two parts, actor and explore policy.
        Maybe in some times, explore policy is independent on actor model.
        """

        train_vars = self.actor.trainable_variables

        for key, value in self._explore_dict.items():
            train_vars += [value]

        return train_vars

    # optimize in the main thread

    def get_corresponding_data(self, data_dict: dict, names: tuple, prefix: str = '', tf_tensor: bool = True):
        """

        Get corresponding data from data dict.

        Args:
            data_dict (dict): data dict.
            names (tuple): name of data.
            prefix (str): prefix of data name.
            tf_tensor (bool): if return tf tensor.
        Returns:
            corresponding data. list or tuple.
        """

        data = []

        for name in names:
            name = prefix + name
            buffer = data_dict[name]
            if tf_tensor:
                buffer = tf.cast(buffer, dtype=tf.float32)
            data.append(buffer)

        return data

    # acquire current update buffer
    def get_current_update_data(self, names: tuple):
        """
        Get current update data.

        Args:
            names (tuple): data name.
        Returns:
            _type_: dict. data dict.
        """
        # running after optimize
        # compute sampling interval
        start_index = (self.optimize_epoch - 1) * self.each_thread_update_interval
        end_index = self.optimize_epoch * self.each_thread_update_interval

        index_bias = np.arange(0, self.total_segment) * self.each_thread_size

        return_dict = {}

        for name in names:
            return_dict[name] = []

        for bias in index_bias:
            start_index = start_index + bias
            end_index = end_index + bias

            data_dict = self.data_pool.get_data_by_indices(np.arange(start_index, end_index).tolist(), names)

            for key, ls in return_dict.items():
                ls.append(data_dict[key])

        # concat data
        for key, ls in return_dict.items():
            return_dict[key] = np.concatenate(ls, axis=0)

        return return_dict

    @property
    def get_all_data(self):
        """
        get all data in buffer.
        """

        return_dict = {}

        for key, unit in self.data_pool.data_pool.items():
            return_dict[key] = unit.buffer

        if self.meta_parameter_names is not None:

            for key in self.meta_parameter_names:
                value = self.args_pool.get_param(key)
                return_dict[key] = np.ones_like(return_dict['total_reward']) * value
                return_dict['next_' + key] = np.ones_like(return_dict['total_reward']) * value

        return return_dict

    @property
    def get_current_buffer_size(self):
        """
        compute current step.
        """
        running_step = self.mini_buffer_size + self.optimize_epoch * self.each_thread_update_interval * self.total_segment
        buffer_size = min(self.max_buffer_size, running_step)
        return buffer_size

    @property
    def get_current_steps(self):
        """
        compute current step.
        """
        running_step = self.mini_buffer_size + self.optimize_epoch * self.each_thread_update_interval * self.sample_threads
        return running_step

    ############################# resample function ################################
    # resample action method

    # The output of resample function is tuple. The last element is the output of actor model.
    @tf.function
    def _resample_action_no_log_std(self, actor_obs: tuple):
        """
        Explore policy in SAC2 is Gaussian  exploration policy.

        _resample_action_no_log_std is used when actor model's out has no log_std.

        The output of actor model is (mu,).

        Args:
            actor_obs (tuple): actor model's input
        Returns:
        action (tf.Tensor): action
        log_pi (tf.Tensor): log_pi
        """

        mu = self.actor(*actor_obs)[0]

        noise, prob = self.explore_policy.noise_and_prob(self.hyper_parameters.batch_size)

        sigma = tf.exp(self.tf_log_std)
        action = mu + noise * sigma
        log_pi = tf.math.log(prob)

        return (action, log_pi)

    @tf.function
    def _resample_action_log_std(self, actor_obs: tuple):
        """
        Explore policy in SAC2 is Gaussian  exploration policy.

        _resample_action_log_std is used when actor model's out has log_std.

        The output of actor model is (mu, log_std).

        Args:
            actor_obs (tuple): actor model's input
        Returns:
        action (tf.Tensor): action
        log_pi (tf.Tensor): log_pi
        """

        out = self.actor(*actor_obs)

        mu, log_std = out[0], out[1]

        noise, prob = self.explore_policy.noise_and_prob(self.hyper_parameters.batch_size)

        sigma = tf.exp(log_std)

        action = mu + noise * sigma

        log_prob = tf.math.log(prob)

        return (action, log_prob)

    @tf.function
    def _resample_action_log_prob(self, actor_obs: tuple):
        """
        Explore policy in SAC2 is Gaussian  exploration policy.

        _resample_action_log_prob is used when actor model's out has log_prob.

        The output of actor model is (mu, log_std).

        Args:
            actor_obs (tuple): actor model's input
        Returns:
        action (tf.Tensor): action
        log_pi (tf.Tensor): log_pi
        """

        action, log_prob = self.actor(*actor_obs)

        return (action, log_prob)

    # @tf.function
    def _resample_log_prob_no_std(self, obs, action):

        """
        Re get log_prob of action.
        The output of actor model is (mu,).
        It is different from resample_action.

        Args:
            obs (tuple): observation.
            action (tf.Tensor): action.
        """

        out = self.actor(*obs)
        mu = out[0]
        std = tf.exp(self.tf_log_std)
        log_prob = self.explore_policy.resample_prob(mu, std, action)

        return (log_prob, *out)

    # def _resample_log_prob_with_std(self, obs, action):

    def _resample_log_prob_with_std(self, obs, action):
        """
        Re get log_prob of action.
        The output of actor model is (mu, log_std,).
        It is different from resample_action.

        """

        out = self.actor(*obs)
        mu = out[0]
        log_std = out[1]
        std = tf.exp(log_std)
        log_prob = self.explore_policy.resample_prob(mu, std, action)

        return (log_prob, *out)

    def _resample_log_prob_discrete(self, obs, action):
        """
        Re get log_prob of action.
        The output of actor model is (log_prob,).
        It is different from resample_action.

        """

        out = self.actor(*obs)
        s_log_prob = out[0]

        log_prob = self.explore_policy.resample_prob(s_log_prob, action)

        return (log_prob,)

    def concat_dict(self, dict_tuple: tuple):
        """
        concat dict.

        Args:
            dict_tuple (tuple): dict tuple.
        Returns:
            _type_: dict. concat dict.
        """
        concat_dict = {}
        for data_dict in dict_tuple:
            for key, value in data_dict.items():
                if key in concat_dict:
                    Warning('key {} is already in concat dict'.format(key))
                else:
                    concat_dict[key] = value

        return concat_dict

    def initialize_model_weights(self, model, expand_dims=False):
        """
        initial model.
        """

        input_data_name = model.input_name

        # create tensor according to input data name
        input_data = []

        for name in input_data_name:
            try:
                shape, _ = self.rl_io_info.get_data_info(name)
            except:
                shape = (1, 1)
            data = tf.zeros(shape=shape, dtype=tf.float32)
            input_data.append(data)
        if expand_dims:
            for idx in self.expand_dims_idx:
                input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        model(*input_data)

    def sync(self):
        """
        sync.
        Used in multi thread.

        """
        if self.level == 0:
            for key, model in self._sync_model_dict.items():
                model.save_weights(self.cache_path + '/' + key + '.h5')
        else:
            for key, model in self._sync_model_dict.items():
                model.load_weights(self.cache_path + '/' + key + '.h5')

        if self.log_std is not None:
            self.sync_log_std()

    def meta_save_model(self):
        pass

    def close(self):
        """
        close.
        """
        self.data_pool.close()
        if self.args_pool is not None:
            self.args_pool.close()

    def sync_log_std(self):
        """
        sync log std.
        """

        if self.level == 0:
            self.log_std.set_value(self.tf_log_std.numpy())  # write log std to shared memory
        else:
            self.tf_log_std = tf.Variable(self.log_std.buffer, trainable=True)  # read log std from shared memory

    # optimize model
    @abc.abstractmethod
    def _optimize_(self, *args, **kwargs):
        """
        optimize model.
        It is a abstract method.

        Recommend when you implement this method, input of this method should be hyperparameters.
        The hyper parameters can be tuned in the training process.

        Returns:
            _type_: dict. Optimizer information. eg. {'loss':data, 'total_reward':data}
        """

    def meta_sync(self):
        self.hyper_parameters.update_meta_parameter_by_args_pool(self.args_pool)
        self.env.update_meta_parameter_by_args_pool(self.args_pool)

