"""
high level loop.

MGRL is a meta RL algorithm, the whole name is Meta Gradient Reinforcement Learning.

此次框架进行改进，确定为AuqaML 2.1版本启动器框架。
在2.2版本中将可以直接调用low-level starter

Reference:
1. Meta-Gradient Reinforcement Learning, 2019, https://arxiv.org/abs/1903.03854
"""
from AquaML.data.ArgsPool import ArgsPool
from AquaML.tool.RLWorkerV2 import RLWorker
from AquaML.DataType import RLIOInfo
from AquaML.data.DataUnit import DataUnit
from AquaML.data.DataPool import DataPool
import tensorflow as tf
import numpy as np
from AquaML.meta.parameters import MetaGradientParameter
import atexit


class MGRL:
    def __init__(self,
                 meta_core,  # core algorithm, such as PPO
                 core_hyperparameter,  # core algorithm hyperparameter
                 core_model_class_dict,  # core algorithm model class dict
                 core_env,  # core environment
                 support_env,  # support environment
                 meta_parameter: MetaGradientParameter,  # support environment rollout parameter，
                 mpi_comm=None,
                 name="MGRL",
                 computer_type='PC',
                 ):
        """
        如何使用MGRL:
        1. core algorithm和RL例子里面一样的创建，但是需要注意在core hyperparameter里面使用 add_meta_parameters
        去声明需要使用neta进行调整的参数，注意MGRL只能够调整与reward相关的参数。
        2. 定义support env， 该env可以和core env不一样。
        3. 定义support env参数，注意声明，是否支持多线程，当主线程off的时候仅有meta线程参与采样，否则附属线程采样

        注意：
        1. env创建需要指定出需要调整超参数。
        """
        ################################################################
        # 配置参数汇总
        ################################################################
        self.store_counter = 0
        self.expand_dims_idx = None
        self.rnn_actor_flag = None
        self._computer_type = computer_type
        self.buffer_size = None
        self.each_summary_episodes_start_index = None
        self.each_summary_episodes = None
        self.each_thread_start_index = None
        self.each_thread_size = None

        self.max_steps = meta_parameter.max_steps
        self, name = name

        # 读取meta parameter
        env_meta_parameters = support_env.meta_parameters

        # 将meta parameter添加到core hyperparameter， core hyperparameter集中式管理所有参数
        if len(env_meta_parameters) < 1:
            self.meta_reward_flag = False
        else:
            self.meta_reward_flag = True
            core_hyperparameter.add_meta_parameters(env_meta_parameters)

        # 根据mpi comm分配线程级别
        if mpi_comm is None:
            self.total_threads = 1
            self.thread_id = 0
            self.level = 0
            sample_thread_num = 1

            self.sample_thread_num = sample_thread_num
        else:
            self.total_threads = mpi_comm.Get_size()
            self.thread_id = mpi_comm.Get_rank()
            # self.level = 1
            sample_thread_num = self.total_threads - 1
            self.sample_thread_num = sample_thread_num

            if self.thread_id == 0:
                self.level = 0

            # 确认其余节点是否需要support env
            if meta_parameter.multi_thread_flag:
                # 需要support env不做任何操作
                if self.level == 0:
                    support_env.close()
                    support_env = None
            else:
                # 不需要support env，将support env设置为None
                support_env.close()
                support_env = None

        # 创建args pool集中式管理所有参数，共享内存机制
        self.args_pool = ArgsPool(name=name,
                                  level=self.level,
                                  computer_type=computer_type)

        # 获取meta parameters的name
        meta_parameter_names = core_hyperparameter.meta_parameter_names
        self.meta_parameter_names = meta_parameter_names  # 存储meta parameter name

        # 创建args pool buffer
        self.args_pool.create_buffer_from_tuple(meta_parameter_names)

        # init args pool buffer, and create shared memory
        self.args_pool.multi_init()

        # 更新args pool 里面参数， 这里的参数是meta_parameters存储了所有需要调整的参数起始值
        self.args_pool.set_param_by_dict(core_hyperparameter.meta_parameters)
        # 注意开始时所有线程meta参数都是一样的，不需要额外同步

        # 存储support env
        self.support_env = support_env

        # 检查rollout参数
        support_env_rollout_parameter = self.check_parameter(meta_parameter, sample_thread_num)

        # 检查core hyperparameter
        core_hyperparameter = self.check_parameter(core_hyperparameter, sample_thread_num)

        # 现在开始实例化core algorithm，目前参考RLTaskStarter.py，后期会进行框架调整
        ########################################################################################
        # 在多线程中，id为0的线程为主线程，其余线程为附属线程。主线程上主要负责rl和meta的模型优化算法。
        # 附属线程上主要负责采样，采样的数据会被主线程使用。
        # 所有的参数维护由args pool完成，args pool会将参数存储在共享内存中，所有线程都可以访问。
        # 所有的协调同步工作都由high level算法完成，例如meta算法，meta算法会根据参数的变化来调整采样的策略。
        ########################################################################################

        actor = core_model_class_dict['actor']()
        actor_out_info = actor.output_info  # dict

        # 设置core env和support env的输入输出信息
        core_env.set_action_state_info(actor_out_info, actor.input_name)

        if support_env is not None:
            support_env.set_action_state_info(actor_out_info, actor.input_name)
        del actor

        obs_info = core_env.obs_info

        # 创建core algorithm info
        core_algorithm_info = RLIOInfo(obs_info=obs_info.shape_dict,
                                       obs_type_info=obs_info.type_dict,
                                       actor_out_info=actor_out_info,
                                       reward_info=core_env.reward_info,
                                       buffer_size=core_hyperparameter.buffer_size,
                                       action_space_type=core_hyperparameter.action_space_type,
                                       )

        # create dict for instancing algorithm
        core_parallel_args = {
            'total_threads': self.total_threads,
            'thread_id': self.thread_id,
            'level': self.level,
            'computer_type': computer_type,
        }

        core_algo_args = {
            'env': core_env,
            'rl_io_info': core_algorithm_info,
            'parameters': core_hyperparameter,
            'prefix_name': name,
        }

        core_model_args = core_model_class_dict

        # 汇总所有参数
        core_algo_args = {**core_algo_args, **core_parallel_args, **core_model_args}

        self.core_algorithm = meta_core(**core_algo_args)

        # core算法初始化
        self.core_algorithm.init()

        # 由于房前MGRL不支持off-policy，所以这里只支持on-policy， 不需要设置预采样代码
        # 为core algorithm设置接口
        self.core_algorithm.meta_parameter_names = self.meta_parameter_names  # 检查这部分逻辑
        self.core_algorithm.args_pool = self.args_pool

        ############################################################################################################
        # 下面创建meta algorithm
        ############################################################################################################

        self.allocate_buffer(
            buffer_size=support_env_rollout_parameter.buffer_size,
            sample_thread_num=sample_thread_num,
            thread_id=self.thread_id,
        )

        # 创建meta algorithm info
        self.meta_info = RLIOInfo(
            obs_info=obs_info.shape_dict,
            obs_type_info=obs_info.type_dict,
            actor_out_info=actor_out_info,
            reward_info=core_env.reward_info,
            buffer_size=self.buffer_size,
            action_space_type=core_hyperparameter.action_space_type,
        )

        # 创建meta pool
        self.data_pool = DataPool(
            name=self.name,
            level=self.level,
            computer_type=self._computer_type,
        )

        # 创建trainable variable
        if self.level == 0:
            self.meta_parameters = {}
            for key, value in core_hyperparameter.meta_parameters.items():
                self.meta_parameters[key] = tf.Variable(value, name=key, trainable=True)
        else:
            self.meta_parameters = None

        # 为每一个worker分配buffer size和开始节点标识

        # 创建meta worker
        if self.support_env is not None:
            self.meta_worker = RLWorker(
                env=self.support_env,
                max_steps=self.max_steps,
                update_interval=self.each_thread_size,
                summary_episodes=support_env_rollout_parameter.summary_episodes,  # 重新计算summary_episodes
            )

        # 所有线程直接使用内环actor，这样不需要同步
        self.actor = self.core_algorithm.actor
        self.critic = self.core_algorithm.critic

        self.get_action = self.core_algorithm.get_action  # 使用和core一样的get_action

        # 创建meta optimizer
        if self.level == 0:
            self.meta_optimizer = getattr(tf.keras.optimizers, meta_parameter.optimizer)(
                learning_rate=meta_parameter.learning_rate)
        else:
            self.meta_optimizer = None

        self.actor_ratio = tf.constant(meta_parameter.actor_ratio, dtype=tf.float32)
        self.critic_ratio = tf.constant(meta_parameter.critic_ratio, dtype=tf.float32)

        atexit.register(self.close)

    def store_data(self, obs: dict, action: dict, reward: dict, next_obs: dict, mask: int):
        """
        将数据存储到buffer中
        """
        self.store_counter += 1
        idx = (self.store_counter - 1) % self.each_thread_size
        index = self.start_index + idx

        # 存储到buffer中
        self.data_pool.store(obs, index)

        # store next_obs to buffer
        self.data_pool.store(next_obs, index, prefix='next_')

        # store action to buffer
        self.data_pool.store(action, index)

        # store reward to buffer
        self.data_pool.store(reward, index)

        # store mask to buffer
        self.data_pool.data_pool['mask'].store(mask, index)

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

    def get_all_data(self):
        """
        TODO: v2.1版本中将此函数扩展为标准接口，逐步改进， data_pool作为输入之类的

        该函数会自动搜寻所有pool中的数据，并且匹配格式，返回一个dict
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

    @staticmethod
    def check_parameter(parameters, sample_thread_num):
        """
        每个线程rollout的时候，保证长度一致, 在2.1版本中将全部使用此函数, 这部分功能将逐渐完善。

        2.1中着重更新报错信息，方便更好的配置钻石
        """

        buffer_size = (parameters.buffer_size / sample_thread_num) * sample_thread_num

        summary_episodes = (parameters.summary_episodes / sample_thread_num) * sample_thread_num

        parameters.buffer_size = int(buffer_size)
        parameters.summary_episodes = int(summary_episodes)

        return parameters

    def allocate_buffer(self, buffer_size, sample_thread_num, thread_id):
        """
        为每个buffer分配起始index，以及为每个worker分配任务。

        2.1版本将逐渐替换掉原有的分配策略。
        这个函数也会逐渐扩展到其他算法中。
        """
        self.each_thread_size = int(buffer_size / sample_thread_num)
        self.each_thread_start_index = self.each_thread_size * thread_id
        self.each_summary_episodes = int(self.meta_worker.summary_episodes / sample_thread_num)
        self.each_summary_episodes_start_index = self.each_summary_episodes * thread_id

        self.buffer_size = self.each_thread_size * self.sample_thread_num

    def optimize(self):
        """
        2.1版本中这个为run会调用的函数，这个函数是所有run的接口
        """
        info = self._optimize_()

        print("#######{}########".format('meta loop'))
        for key, value in info.items():
            print("{}: {}".format(key, value))

        for key, value in self.meta_parameters.items():
            print("{}: {}".format(key, value))

    def _optimize_(self):
        """
        2.1版，实现优化算法的地方
        """
        data_dict = self.get_all_data()

        # 从buffer中获取数据
        critic_obs = self.get_corresponding_data(data_dict, self.critic.input_name)
        next_critic_obs = self.get_corresponding_data(data_dict, self.critic.input_name,
                                                      prefix='next_')

        actor_obs = self.get_corresponding_data(data_dict, self.actor.input_name)

        # 获取total reward
        total_reward = data_dict['total_reward']

        # 获取mask
        masks = data_dict['mask']

        # 获取action
        actions = data_dict['action']

        # get old prob
        old_prob = data_dict['prob']

        # 转换成tf tensor
        rewards = tf.cast(total_reward, dtype=tf.float32)
        masks = tf.cast(masks, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        old_prob = tf.convert_to_tensor(old_prob, dtype=tf.float32)
        old_log_prob = tf.math.log(old_prob)

        if 'value' in self.actor.output_info:
            values = data_dict['value']
            next_values = data_dict['next_value']
        else:
            values = self.critic(*critic_obs).numpy()
            next_values = self.critic(*next_critic_obs).numpy()

        with tf.GradientTape() as tape:
            tape.watch(self.meta_parameters.values())
            # 计算GAE
            gae, target = self.core_algorithm.calculate_GAEV2(
                rewards=rewards,
                values=values,
                next_values=next_values,
                masks=masks,
                gamma=self.meta_parameters['gamma'],
                lamda=self.meta_parameters['lamda'],
            )

            # 计算actor loss
            actor_loss = self.core_algorithm.compute_actor_loss(
                actor_obs=actor_obs,
                advantage=gae,
                old_log_prob=old_log_prob,
                actions=actions,
                epsilon=self.core_algorithm.hyper_parameters.epsilon,
                entropy_coefficient=self.core_algorithm.hyper_parameters.entropy_coeff,
            )

            # 计算critic loss
            critic_loss = self.core_algorithm.compute_critic_loss(
                critic_obs=critic_obs,
                target=target,
            )

            # 计算loss
            loss = self.actor_ratio * actor_loss + self.critic_ratio * critic_loss

        # 计算梯度
        grad = tape.gradient(loss, self.meta_parameters.values())
        # 更新梯度
        self.meta_optimizer.apply_gradients(zip(grad, self.meta_parameters.values()))

        return {
            'actor_loss': actor_loss.numpy(),
            'critic_loss': critic_loss.numpy(),
            'loss': loss.numpy(),
        }

    def _run_(self):
        """
        单线程运行
        """
        for _ in range(self.max_steps):
            self.core_algorithm.worker.roll(self.core_algorithm.each_thread_update_interval,
                                            test_flag=False)
            self.core_algorithm.optimize()
            self.meta_worker.roll(self, test_flag=True)
            self.optimize()
            self.sync()

    def sync(self):
        """
        所有同步事件2.1都用此函数
        """
        # 更新args pool 内容
        for key, value in self.meta_parameters.items():
            self.args_pool.set_param_by_name(key, value.numpy())

        # 同步core_algorithm参数
        self.core_algorithm.meta_sync()

    def create_data_pool(self):
        """
        由于在2.1版本中，所有的data pool将逐渐统一这个函数，这个函数也会逐渐改进扩展。
        """

        # 添加summary信息到meta_info中
        reward_info_dict = {}
        for name in self.meta_info.reward_info:
            reward_info_dict['summary_' + name] = (
                self.each_summary_episodes * self.sample_thread_num, 1)

        for name, shape in reward_info_dict.items():
            buffer = DataUnit(
                name=name + '_' + name,
                shape=shape,
                dtype=np.float32,
                level=self.level,
                computer_type=self._computer_type,
            )
            self.meta_info.add_info(
                name=name,
                shape=shape,
                dtype=np.float32,
            )
            self.data_pool.add_unit(
                name=name,
                data_unit=buffer
            )

        # 无论是否使用多线程，我们都会开启共享内存池子，方便管理
        self.data_pool.multi_init(
            self.meta_info.data_info,
            type='buffer'
        )

    def initialize_model_weights(self, model, expand_dims=False):
        """
        该函数
        """

        input_data_name = model.input_name

        # create tensor according to input data name
        input_data = []

        for name in input_data_name:
            shape, _ = self.meta_info.get_data_info(name)
            data = tf.zeros(shape=shape, dtype=tf.float32)
            input_data.append(data)
        if expand_dims:
            for idx in self.expand_dims_idx:
                input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        model(*input_data)

    def set_actor_config(self):
        """
        2.1版本中，这个将成为基本得config函数，当然名字会统一
        """
        self.rnn_actor_flag = getattr(self.actor, 'rnn_flag', False)

        if self.rnn_actor_flag:
            self.expand_dims_idx = []  # 记录输出非hidden state位置

            idx = 0

            names = self.actor.input_name
            for name in names:
                if 'hidden' in name:
                    pass
                else:
                    self.expand_dims_idx.append(idx)
                idx += 1
            self.expand_dims_idx = tuple(self.expand_dims_idx)

    def close(self):
        self.core_algorithm.close()
        self.args_pool.close()
        self.data_pool.close()
