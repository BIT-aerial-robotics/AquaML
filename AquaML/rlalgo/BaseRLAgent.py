from abc import ABC, abstractmethod
from AquaML.core.NetworkTools import *
from AquaML.rlalgo.ExplorePolicy import ExplorePolicyBase
from AquaML.rlalgo.ExplorePolicy import create_explor_policy
from AquaML.core.ToolKit import LossTracker
from AquaML.data.DataUnit import DataUnit
from AquaML.core.DataParser import DataInfo
from AquaML.core.ToolKit import mkdir
import numpy as np
import os


class BaseAgent(ABC):

    @abstractmethod
    def init(self):
        """
        初始化agent。
        """


# TODO: 需要修改非通用agent
class BaseRLAgent(BaseAgent, ABC):
    def __init__(self,
                 name: str,
                 agent_params,
                 level: int = 0,  # 控制是否创建不交互的agent
                 ):
        """
        Agent 基类。所有的Agent都需要继承这个类。
        actor为策略网络的class，value_fn为价值网络的class。

        Args:
            name (str): agent的名字。
            agent_params (AgentParameter): agent的参数。
            level (int, optional): 控制是否创建不交互的agent。Defaults to 0.
        """
        ##############################
        # 基础变量
        ##############################

        self.name = name
        self.agent_params = agent_params
        self.level = level

        ##############################
        # 插件
        # 基础插件
        ##############################

        self.loss_tracker = LossTracker()

        ##############################
        # 接口变量
        ##############################

        # base里面根据配置创建的值，除特殊情况外，不允许修改
        self._explore_dict = {}
        self._tf_explore_dict = {}

        self._network_process_info = {
            'actor': {},
            'critic': {},
        }  # 网络输入数据处理信息

        self._optimizer_pool = {}

        # 子类需要根据需要创建的值
        self.actor = None
        self.critic = None

        # 初始化agent之后需要指定的变量
        self.agent_info = None

        # 实现操作接口

        # sync model
        self._sync_model_dict = {}  # 同步模型字典

        # all model
        self._all_model_dict = {}  # 所有模型字典

        # parame_dict
        self._param_dict = {}  # 全局同步参数字典

        # inidicate dict
        self._indicate_dict = {}

        self._normalization_tuple = []

    # @abstractmethod
    def init(self):
        raise NotImplementedError

    def set_agent_info(self, agent_info):
        """
        设置agent_info。

        Args:
            agent_info (dict): agent_info。
        """
        self.agent_info = agent_info

    def check(self):
        """
        检查agent是否合法。
        """

        # 查看explore_policy是否合法
        if not hasattr(self, 'explore_policy'):
            raise AttributeError(f'{self.__class__.__name__} has no explore_policy attribute')
        else:
            if not issubclass(self.explore_policy, ExplorePolicyBase):
                raise TypeError(f'{self.explore_policy.__class__.__name__} is not a subclass of BaseExplorePolicy')

    def initialize_actor(self):
        """
        用于初始化actor，比如说在rnn系统模型里面，某些输入需要额外处理维度。

        #TODO：在2.1版本中逐步将网络输入配置融入到网络定义中。
        """

        # 判断网络类型
        actor_rnn_flag = getattr(self.actor, 'rnn_flag', False)

        # RNN输入时候维度的处理

        self.actor_expand_dims_idx = []

        if actor_rnn_flag:
            self._network_process_info['actor']['rnn_flag'] = True
            idx = 0
            actor_input_names = self.actor.input_name

            for name in actor_input_names:
                if 'hidden' in name:
                    pass
                else:
                    self.actor_expand_dims_idx.append(idx)
                idx += 1
            self.actor_expand_dims_idx = tuple(self.actor_expand_dims_idx)

        else:
            self._network_process_info['actor']['rnn_flag'] = False

        self.initialize_network(
            model=self.actor,
            expand_dims_idx=self.actor_expand_dims_idx,
        )

    def initialize_critic(self):
        """
        初始化critic。
        """

        self.critic_expand_dims_idx = []

        critic_rnn_flag = getattr(self.critic, 'rnn_flag', False)

        if critic_rnn_flag:
            self._network_process_info['critic']['rnn_flag'] = True

            idx = 0
            critic_input_names = self.critic.input_name

            for name in critic_input_names:
                if 'hidden' in name:
                    pass
                else:
                    self.critic_expand_dims_idx.append(idx)
                idx += 1
            self.critic_expand_dims_idx = tuple(self.critic_expand_dims_idx)

        else:
            self._network_process_info['critic']['rnn_flag'] = False

        self.initialize_network(
            model=self.critic,
            expand_dims_idx=self.critic_expand_dims_idx,
        )

    def initialize_network(self, model, expand_dims_idx=None):
        """

        初始化网络参数。

        Args:
            model (_type_): _description_
            expand_dims_idx (_type_, optional): _description_. Defaults to None.
        """

        input_data_name = model.input_name

        # create tensor according to input data name
        input_data = []

        for name in input_data_name:
            try:
                shape, _ = self.agent_info.get_data_info(name)
            except:
                shape = (1, 1)
            data = tf.zeros(shape=shape, dtype=tf.float32)
            input_data.append(data)
        if expand_dims_idx is not None:
            for idx in expand_dims_idx:
                input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        model(*input_data)

    def copy_weights(self, source_model, target_model):
        """
        将source_model的参数复制到target_model中。

        Args:
            source_model (_type_): _description_
            target_model (_type_): _description_
        """
        new_weights = []

        for idx, weight in enumerate(source_model.get_weights()):
            new_weights.append(weight)

        target_model.set_weights(new_weights)

    def soft_update(self, source_model, target_model, tau):
        """
        将source_model的参数复制到target_model中。

        Args:
            source_model (_type_): _description_
            target_model (_type_): _description_
        """
        new_weights = []

        for idx, weight in enumerate(source_model.get_weights()):
            new_weights.append(weight * tau + target_model.get_weights()[idx] * (1 - tau))

        target_model.set_weights(new_weights)

    def get_action(self, obs, test_flag=False):

        input_data = []

        # 获取输入数据
        for name in self.actor.input_name:
            data = tf.cast(obs[name], dtype=tf.float32)
            input_data.append(data)

        # 数据扩展
        # TODO: 后续版本中需要给出数据处理通用接口 backends

        for idx in self.actor_expand_dims_idx:
            input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        actor_out = self.actor(*input_data)

        # TODO: 这个地方需要优化速度
        policy_out = dict(zip(self.actor.output_info, actor_out))

        for name, value in self._explore_dict.items():
            policy_out[name] = tf.cast(value.buffer, dtype=tf.float32)

        action, prob = self.explore_policy(policy_out, test_flag=test_flag)

        policy_out['action'] = action
        policy_out['prob'] = prob

        # create return dict according to rl_io_info.actor_out_name
        return_dict = dict()
        for name in self.agent_info.actor_out_name:
            return_dict[name] = policy_out[name]

        for name in self.explore_policy.get_aditional_output.keys():
            return_dict[name] = policy_out[name]

        return return_dict

    @property
    def get_action_names(self):
        names = []

        for key in self.actor.output_info.keys():
            names.append(key)

        for key in self.explore_policy.get_aditional_output.keys():
            names.append(key)

        return names

    def create_explorer(self, explore_name, shape, pointed_value={}):

        policy, infos = create_explor_policy(
            explore_policy_name=explore_name,
            shape=shape,
            actor_out_names=self.agent_info.actor_out_name,
        )

        for item in infos:
            name = item['name']

            bu = DataUnit(
                name=self.name + '_' + name,
                dtype=item['dtype'],
                shape=item['shape'],
                level=self.level,
            )

            setattr(self, name, bu)

            self._explore_dict[name] = getattr(self, name)

            init_value = 0

            if name in pointed_value:
                init_value = pointed_value[name]

            if item['trainable']:
                vars = tf.Variable(
                    initial_value=init_value,
                    trainable=True,
                    name=self.name + '_' + name,
                )

                setattr(self, 'tf_' + name, vars)

                self._tf_explore_dict[name] = getattr(self, 'tf_' + name, )

                self._explore_dict[name].set_value(vars.numpy())

                self._param_dict[name] = {
                    'shape': item['shape'],
                    'dtype': item['dtype'],
                    'trainable': True,
                    'init': init_value,
                    'data': self._explore_dict[name],
                }

        self.explore_policy = policy

    def save_param(self, path):

        # store_path = os.path.join(path, self.name)
        # mkdir(store_path)
        for name, value in self._param_dict.items():
            file_name = os.path.join(path, name + '.npy')
            value['data'].save(file_name)

    def load_param(self, path):

        # load_path = os.path.join(path, self.name)

        for name, value in self._param_dict.items():
            file_name = os.path.join(path, name + '.npy')
            value['data'].load(file_name)

            if value['trainable']:
                var = getattr(self, 'tf_' + name)
                var.assign(np.squeeze(value['data'].buffer))

    def update_explorer(self):

        if self.level == 0:
            for name, value in self._tf_explore_dict.items():
                self._explore_dict[name].set_value(value.numpy())

        else:
            for name, value in self._tf_explore_dict.items():
                value.assign(self._explore_dict[name].buffer)

    def create_optimizer(self, optimizer_info, name):

        type = optimizer_info['type']
        args = optimizer_info['args']

        optimizer = getattr(tf.keras.optimizers, type)(**args)

        setattr(self, name, optimizer)

        self._optimizer_pool[name] = {
            'optimizer': optimizer,
            'lr': args['learning_rate'],
        }

        # return optimizer

    @property
    def get_optimizer_pool(self):
        return self._optimizer_pool

    def get_collection_info(self, reward_info: tuple or list, woker_num, obs_norm_flag=False, reward_norm_flag=False,
                            obs_shape_dict=None):
        """
        获取网络参数信息。
        """

        policy_aditional_info = self.explore_policy.get_aditional_output

        for name, value in policy_aditional_info.items():
            self.agent_info.add_info(
                name=name,
                shape=value['shape'],
                dtype=value['dtype'],
            )

        param_names = []
        param_shapes = []
        param_dtypes = []
        # 会在运行中自动同步
        for name, value in self._param_dict.items():
            param_names.append(name)
            param_shapes.append(value['shape'])
            param_dtypes.append(value['dtype'])

        param_info = DataInfo(
            names=param_names,
            shapes=param_shapes,
            dtypes=param_dtypes,
        )

        agent_data_info = self.agent_info.get_info

        # 创建summary reward indicate
        indicate_names = []
        indicate_shapes = []
        indicate_dtypes = []

        for name in reward_info:
            indicate_names.append('summary_' + name)
            indicate_dtypes.append(np.float32)
            indicate_shapes.append((woker_num, 1))

            indicate_names.append('summary_' + name + '_max')
            indicate_dtypes.append(np.float32)
            indicate_shapes.append((woker_num, 1))

            indicate_names.append('summary_' + name + '_min')
            indicate_dtypes.append(np.float32)
            indicate_shapes.append((woker_num, 1))

        if obs_norm_flag:
            if obs_shape_dict is None:
                raise ValueError('obs_shape_dict is None')
            for name, shape in obs_shape_dict.items():
                if len(shape) == 1:
                    shape_ = (woker_num, shape[0])
                else:
                    shape_ = (woker_num, *shape)

                indicate_names.append(name + '_mean')
                indicate_dtypes.append(np.float32)
                indicate_shapes.append(shape_)

                indicate_names.append(name + '_std')
                indicate_dtypes.append(np.float32)
                indicate_shapes.append(shape_)

        if reward_norm_flag:
            indicate_names.append('total_reward_mean')
            indicate_dtypes.append(np.float32)
            indicate_shapes.append((woker_num, 1))

            indicate_names.append('total_reward_std')
            indicate_dtypes.append(np.float32)
            indicate_shapes.append((woker_num, 1))

        indicate_info = DataInfo(
            names=indicate_names,
            shapes=indicate_shapes,
            dtypes=indicate_dtypes,
        )

        return agent_data_info, param_info, indicate_info

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

    @property
    def get_param_dict(self):
        return self._param_dict

    @property
    def get_sync_model_dict(self):
        return self._sync_model_dict

    @property
    def get_all_model_dict(self):
        return self._all_model_dict

    @staticmethod
    @abstractmethod
    def get_algo_name():
        """
        获取算法名称。
        """

    def __del__(self):
        pass
