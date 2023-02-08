import abc
import tensorflow as tf
from AquaML.data.DataPool import DataPool
from AquaML.DataType import RLIOInfo
from AquaML.data.DataUnit import DataUnit
from AquaML.rlalgo.ExplorePolicy import GaussianExplorePolicy
from AquaML.tool.RLWorker import RLWorker
from AquaML.BaseClass import BaseAlgo
import numpy as np
from AquaML.tool.Recoder import Recoder


# TODO:model logic has been changed, check the new version
# TODO: 优化命名方式
class BaseRLAlgo(BaseAlgo, abc.ABC):

    # TODO:统一输入接口
    # TODO:判断是否启动多线程 (done)  

    def __init__(self, env, rl_io_info: RLIOInfo, name: str, update_interval: int = 0, mini_buffer_size: int = 0,
                 calculate_episodes=5,
                 display_interval=1,
                 computer_type: str = 'PC',
                 level: int = 0, thread_ID: int = -1, total_threads: int = 1, policy_type: str = 'off'):
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

        self.recoder = None
        self.explore_policy = None
        self.tf_log_std = None
        self.log_std = None
        self.rl_io_info = rl_io_info
        self._computer_type = computer_type
        self.level = level
        self.thread_ID = thread_ID  # if thread_ID == -1, it means single thread
        self.name = name
        self.env = env
        self.update_interval = update_interval
        self.mini_buffer_size = mini_buffer_size
        self.total_threads = total_threads
        self.each_thread_summary_episodes = calculate_episodes
        self.display_interval = display_interval

        self.cache_path = name + '/cache'  # cache path
        self.log_path = name + '/log'

        # self.last_step = 0  # last step in main thread

        # check thread level
        # if thread_ID == -1, it means single thread, then level must be 0
        # TODO: 这个地方有问题
        # if self.thread_ID == -1 and self.level != 0:
        #     raise ValueError('If thread_ID == -1, it means single thread, then level must be 0.')

        # create data pool according to thread level
        self.data_pool = DataPool(name=self.name, level=self.level,
                                  computer_type=self._computer_type)  # data_pool is a handle

        # store thread information for communication
        # self.thread_info = DataPool(name=self.name + '_info', level=self.level, computer_type=self._computer_type)

        self.actor = None  # actor model. Need point out which model is the actor.

        # TODO: 私有变量会出问题，貌似这个没用
        self._explore_dict = {}  # store explore policy, convenient for multi thread

        # TODO: 需要升级为异步执行的方式
        # TODO: 需要确认主线程和子线程得到得硬件不一样是否影响执行速度 
        # allocate start index and size for each thread
        # main thread will part in sample data
        # just used when computer_type == 'PC'
        if self._computer_type == 'PC':
            if self.total_threads > 1:
                # thread ID start from 0
                self.each_thread_size = int(self.rl_io_info.buffer_size / total_threads)
                self.each_thread_start_index = self.thread_ID * self.each_thread_size

                self.max_buffer_size = self.each_thread_size * total_threads

                # if mini_buffer_size == 0, it means pre-sample data is disabled
                self.each_mini_buffer_size = int(self.mini_buffer_size / total_threads)
                self.mini_buffer_size = self.each_mini_buffer_size * total_threads

                if self.update_interval == 0:
                    # 这种情形属于将所有buffer填充满以后再更新模型
                    # if update_interval == 0, it means update model after buffer is full
                    self.each_thread_update_interval = self.each_thread_size  # update interval for each thread
                else:
                    # if update_interval != 0, it means update model after each step
                    # then we need to calculate how many steps to update model for each thread
                    # 每个线程更新多少次等待更新模型
                    self.each_thread_update_interval = int(
                        self.update_interval / total_threads)  # update interval for each thread
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
                # self.each_thread_update_interval = self.update_interval # update interval for each thread

        else:
            # TODO: HPC will implement in the future
            self.each_thread_size = None
            self.each_thread_start_index = None
            self.each_thread_update_interval = None

        # create worker
        self.worker = RLWorker(self)

        # hyper parameters
        # the hyper parameters is a dictionary
        # you should point out the hyper parameters in your algorithm
        # will be used in optimize function
        self.hyper_parameters = None

        # optimizer are created in main thread
        self.optimizer_dict = {}  # store optimizer, convenient search

        self.total_segment = total_threads  # total segment, convenient for multi

        self.sample_epoch = 0  # sample epoch
        self.optimize_epoch = 0  # optimize epoch

        self.policy_type = policy_type  # 'off' or 'on'

        # mini buffer size 
        # according to the type of algorithm,

        self._sync_model_dict = None  # store sync model, convenient for multi thread
        self._sync_explore_dict = None  # store sync explore policy, convenient for multi thread

    # initial algorithm
    def init(self):
        """initial algorithm.
        
        This function will be called by starter.
        """
        # multi thread communication

        reward_info_dict = {}

        for name in self.rl_io_info.reward_info:
            reward_info_dict['summary_' + name] = (self.total_segment * self.each_thread_summary_episodes, 1)

        # add summary reward information to data pool
        for name, shape in reward_info_dict.items():
            buffer = DataUnit(name=name, shape=shape, dtype=np.float32, level=self.level,
                              computer_type=self._computer_type)
            self.data_pool.add_unit(name=name, data_unit=buffer)

        # TODO:子线程需要等待时间 check
        # multi thread initial
        if self.thread_ID > -1:  # multi thread
            self.data_pool.multi_init(self.rl_io_info.data_info, type='buffer')
        else:  # single thread
            self.data_pool.create_buffer_from_dic(self.rl_io_info.data_info)

        # just do in m main thread
        if self.level == 0:
            # initial recoder
            self.recoder = Recoder(log_folder=self.log_path)
        else:
            self.recoder = None

        # check some information
        # actor model must be given
        if self.actor is None:
            raise ValueError('Actor model must be given.')

    def check(self):
        """
        check some information.
        """
        if self.policy_type == 'off':
            if self.mini_buffer_size is None:
                raise ValueError('Mini buffer size must be given.')
        if self._sync_model_dict is None:
            raise ValueError('Sync model must be given.')

    # TODO: calculate by multi thread
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
        index = length - 1

        for i in range(length):
            index = index - 1
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

    # calculate episode reward information
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
        reward_summary = {'std': np.std(reward_dict['total_reward']), 'max_reward': np.max(reward_dict['total_reward']),
                          'min_reward': np.min(reward_dict['total_reward'])}

        for key in self.rl_io_info.reward_info:
            reward_summary[key] = np.mean(reward_dict[key])

        # delete list
        del reward_dict

        return reward_summary

    def sumarry_reward_info(self):
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

    # store data to buffer
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

        idx = (self.worker.step_count - 1) % self.each_thread_size
        index = self.each_thread_start_index + idx  # index in each thread

        # store obs to buffer
        self.data_pool.store(obs, index)

        # store next_obs to buffer
        self.data_pool.store(next_obs, index)

        # store action to buffer
        self.data_pool.store(action, index)

        # store reward to buffer
        self.data_pool.store(reward, index)

        # store mask to buffer
        self.data_pool.data_pool['mask'].store(mask, index)

    def get_action_train(self, obs: dict):
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
            input_data.append(obs[key])

        actor_out = self.actor(*input_data)  # out is a tuple

        policy_out = dict(zip(self.actor.output_info, actor_out))

        for name, value in self._explore_dict.items():
            policy_out[name] = value

        action, prob = self.explore_policy(policy_out)

        policy_out['action'] = action
        policy_out['prob'] = prob

        # create return dict according to rl_io_info.actor_out_name
        return_dict = dict()
        for name in self.rl_io_info.actor_out_name:
            return_dict[name] = policy_out[name]

        return return_dict

    # create keras optimizer
    def create_optimizer(self, name: str, optimizer: str, lr: float):
        """
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
            optimizer = getattr(tf.keras.optimizers, optimizer)(lr=lr)
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
        if self.rl_io_info.explore_info == 'self':
            # log_std provided by actor
            # create explore policy
            # self.__explore_dict = None
            pass
        elif self.rl_io_info.explore_info == 'auxiliary':
            # log_std provided by auxiliary variable
            # create args by data unit
            self.log_std = DataUnit(name=self.name + '_log_std', dtype=np.float32,
                                    shape=self.rl_io_info.actor_out_info['action'],
                                    level=self.level, computer_type=self._computer_type)

            self.log_std.set_value(np.zeros(self.rl_io_info.actor_out_info['action'], dtype=np.float32) - 0.5)
            self.tf_log_std = tf.Variable(self.log_std.buffer, trainable=True)
            self._explore_dict = {'log_std': self.tf_log_std}

            self.rl_io_info.add_info(name='log_std', shape=self.log_std.shape, dtype=self.log_std.dtype)
            self.data_pool.add_unit(name='log_std', data_unit=self.log_std)

        self.explore_policy = GaussianExplorePolicy(shape=self.rl_io_info.actor_out_info['action'])

        # add initial information

    # optimize model
    @abc.abstractmethod
    def _optimize_(self, *args, **kwargs):
        """
        optimize model.
        It is a abstract method.
        
        Recommend when you implement this method, input of this method should be hyperparameters. 
        The hyperparameters can be tuned in the training process.
        
        Returns:
            _type_: dict. Optimizer information. eg. {'loss':data, 'total_reward':data}
        """

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
            train_vars = train_vars + [value]

        return train_vars

    # optimize in the main thread
    def optimize(self):

        # compute current reward information

        optimize_info = self._optimize_()

        # all the information update here
        self.optimize_epoch += 1

        if self.optimize_epoch % self.display_interval == 0:

            # display information

            epoch = int(self.optimize_epoch // self.display_interval)
            reward_info = self.sumarry_reward_info()
            print("###############epoch: {}###############".format(epoch))
            self.recoder.display_text(
                reward_info
            )
            self.recoder.display_text(
                optimize_info
            )

            self.recoder.record(reward_info, epoch, prefix='reward')
            # self.recoder.record(optimize_info, self.optimize_epoch, prefix=self.name)

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
    def current_buffer_size(self):
        """
        compute current step.
        """
        running_step = self.mini_buffer_size + self.optimize_epoch * self.each_thread_update_interval * self.total_segment
        buffer_size = min(self.max_buffer_size, running_step)
        return buffer_size

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

    def initialize_model_weights(self, model):
        """
        initial model.
        """

        input_data_name = model.input_name

        # create tensor according to input data name
        input_data = []

        for name in input_data_name:
            shape, _ = self.rl_io_info.get_data_info(name)
            data = tf.zeros(shape=shape, dtype=tf.float32)
            input_data.append(data)

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

    def close(self):
        """
        close.
        """

        self.data_pool.close()

    def sync_log_std(self):
        """
        sync log std.
        """

        if self.level == 0:
            self.log_std.set_value(self.tf_log_std.numpy())  # write log std to shared memory
        else:
            self.tf_log_std = tf.Variable(self.log_std.buffer, trainable=True)  # read log std from shared memory
