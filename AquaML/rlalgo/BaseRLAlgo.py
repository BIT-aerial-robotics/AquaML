import abc
import tensorflow as tf
from AquaML.data.DataPool import DataPool
from AquaML.DataType import RLIOInfo
from AquaML.data.DataUnit import DataUnit
from AquaML.rlalgo import ExplorePolicy
from AquaML.tool.RLWorker import RLWorker
import numpy as np

class BaseRLAlgo(abc.ABC):

    # TODO:统一输入接口
    # TODO:判断是否启动多线程 (done)  

    def __init__(self,env,rl_io_info:RLIOInfo,name:str, update_interval:int=0, computer_type:str='PC', level:int=0, thread_ID:int=-1, total_threads:int=1,):
        """create base for reinforcement learning algorithm.
        This base class provides exploration policy, data pool(multi thread).
        Some tools are also provided for reinforcement learning algorithm such as 
        calculate general advantage estimation. 
        
        When you create a reinforcement learning algorithm, you should inherit this class. And do the following things:
        
        Point out hyper parameters, use self.hyper_parameters (dict) to store hyper parameters.
        
        And you should run init() function in your __init__ function. The position of init() function is at the end of __init__ function.

        You need to point out which model is the actor, then in __init__ function, you should write:
        "self.actor = actor"
        
        Explore policy is a function which is used to generate action. You should use or create a explore policy. 
        Then in __init__ function, you should write:
        "self.explore_policy = explore_policy"
        You can create a explore policy by inherit ExplorePolicyBase class(AquaML.rlalgo.ExplorePolicy.ExplorePolicyBase). 
        
        
        
        Args:
            env (AquaML.rlalgo.EnvBase): reinforcement learning environment.
            
            rl_io_info (RLIOInfo): reinforcement learning input and output information.
            
            name (str): reinforcement learning name.
            
            update_interval (int, optional): update interval. This is an important parameter. 
            It determines how many steps to update the model. If update_interval == 1, it means update model after each step.
            if update_interval == 0, it means update model after buffer is full. Defaults to 0.
            That is unusually used in on-policy algorithm.
            if update_interval > 0, it is used in off-policy algorithm.
            
            computer_type (str, optional): 'PC' or 'HPC'. Defaults to 'PC'.
            
            level (int, optional): thread level. 0 means main thread, 1 means sub thread. Defaults to 0.
            
            thread_ID (int, optional): ID is given by mpi. -1 means single thread. Defaults to -1.
            
            total_threads (int, optional): total threads. Defaults to 1.

        Raises:
            ValueError: if thread_ID == -1, it means single thread, then level must be 0.
        """

        self.rl_io_info = rl_io_info
        self._computer_type = computer_type
        self.level = level
        self.thread_ID = thread_ID # if thread_ID == -1, it means single thread
        self.name = name
        self.env = env
        self.update_interval = update_interval

        # check thread level
        # if thread_ID == -1, it means single thread, then level must be 0
        if self.thread_ID == -1 and self.level != 0:
            raise ValueError('If thread_ID == -1, it means single thread, then level must be 0.')

        # create data pool according to thread level
        self.data_pool = DataPool(name=self.name,level=self.level, computer_type=self._computer_type) # data_pool is a handle
        
        self.actor = None # actor model. Need point out which model is the actor.

       
            
        self.__explore_dict = {} # store explore policy, convenient for multi thread
        
        # TODO: 需要升级为异步执行的方式
        # allocate start index and size for each thread
        # main thread will part in sample data
        # just used when computer_type == 'PC'
        if self._computer_type == 'PC':
            if self.thread_ID > -1:
                # thread ID start from 0
                self.each_thread_size = int(self.rl_io_info.buffer_size / total_threads)
                self.each_thread_start_index = self.thread_ID * self.each_thread_size
                
                if self.update_interval == 0:
                    # if update_interval == 0, it means update model after buffer is full
                    self.each_thread_update_interval = self.each_thread_size # update interval for each thread
                else:
                    # if update_interval != 0, it means update model after each step
                    # then we need to calculate how many steps to update model for each thread
                    self.each_thread_update_interval = int(self.update_interval / total_threads) # update interval for each thread
            else:
                self.each_thread_size = self.rl_io_info.buffer_size
                self.each_thread_start_index = 0
                self.each_thread_update_interval = self.update_interval # update interval for each thread
                
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
        self.optimizer_dict = {} # store optimizer, convenient search
        

    
    # initial algorithm
    def init(self):
        """initial algorithm.
        
        This function will be called by starter.
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
    
    # store data to buffer
    def store_data(self, obs:dict, action:dict, reward:dict, next_obs:dict, mask:int):
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
        
        id = (self.worker.step_count - 1) % self.each_thread_size
        index = self.each_thread_start_index + id # index in each thread
        
        # store obs to buffer
        self.data_pool.store(obs,index)
        
        # store next_obs to buffer
        self.data_pool.store(next_obs,index)
        
        # store action to buffer
        self.data_pool.store(action,index)
        
        # store reward to buffer
        self.data_pool.store(reward,index)
        
        # store mask to buffer
        self.data_pool.data_pool['mask'].store(mask,index)
        
        
        

    # get action in the training process
    # TODO:根据网络模型的特点需要修改
    def get_action_train(self, obs:dict):
        """
        Get action in the training process.
        This function can be used in all algorithms since 
        the explore policy is given.

        Args:
            obs (dict): observation from environment. eg. {'obs':np.array([1,2,3])}

        Returns:
            _type_: dict. action. eg. {'action':np.array([1,2,3]), 'log_prob':np.array([1,2,3])}
        """
        
        # get actor input
        input_obs = dict()
        
        # Notice: the order of input_obs must be the same as the order of input in actor model
        for key in self.rl_io_info.actor_input_info:
            # if the input is time sequence, data style (batch, timesteps, feature)
            if self.actor.rnn_flag:
                input_obs[key] = np.expand_dims(obs[key], axis=0)
            else:
                input_obs[key] = obs[key]
                
        # call actor model
        actor_out_ = self.actor(*input_obs) # out is a tuple
        
        # TODO: Need to check out
        actor_out = dict(zip(self.rl_io_info.actor_model_out_name, actor_out_))
        
        # explore policy input
        explore_input = dict()
        
        # add actor output to explore input
        for name in self.rl_io_info.actor_out_name:
            explore_input[name] = actor_out[name]
            
        # add __explore_dict to explore_input
        for name, value in self.__explore_dict.items():
            explore_input[name] = value

        # run explore policy
        action, prob = self.explore_policy(explore_input)
        
        # add action and prob to actor_out
        actor_out['action'] = action
        actor_out['prob'] = prob
        
        # create return dict according to rl_io_info.actor_out_name
        return_dict = dict()
        for name in self.rl_io_info.actor_out_name:
            return_dict[name] = actor_out[name]
    
        return return_dict
    
    # create keras optimizer
    def create_optimizer(self,name:str,optimizer:str,lr:float):
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
        
    def create_none_optimizer(self, name:str):
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
        # verity the style of log_std
        if self.rl_io_info.explore_info == 'self':
            # log_std provided by actor
            # create explore policy
            # self.__explore_dict = None
            pass
        elif self.rl_io_info.explore_info == 'auxiliary':
            # log_std provided by auxiliary variable
            # create args by data unit
            self.log_std = DataUnit(name=self.name+'_log_std', data_type=np.float32, shape=self.rl_io_info.action_info['action'].shape, 
                                    level=self.level, computer_type=self._computer_type)
            
            self.log_std.set_value(np.zeros(self.rl_io_info.action_info['action'].shape, dtype=np.float32)-0.5)
            self.tf_log_std = tf.Variable(self.log_std.get_value(), trainable=True)
            self.__explore_dict = {'log_std':self.tf_log_std}
        
        self.explore_policy = ExplorePolicy(shape = self.rl_io_info.action_info['action'])
        
        # add initial information
        self.rl_io_info.add_info(name='log_std',shape=self.log_std.shape, data_type=self.log_std.data_type)
        self.data_pool.add_unit(name='log_std', data_unit=self.log_std)
    
    # optimize model
    @abc.abstractclassmethod
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
            new_weights.append((1-tau)*weight.numpy() + tau*source_weights[i].numpy())
            
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
        
        for key, value in self.__explore_dict.items():
            train_vars = train_vars + [value]
            
        return train_vars
    
    # optimize in the main thread
    def optimize(self):
        
        optimize_info = self._optimize_()
        