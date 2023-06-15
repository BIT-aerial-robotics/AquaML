from abc import ABC, abstractmethod
from AquaML.core.NetworkTools import *
from AquaML.rlalgo.ExplorePolicy import ExplorePolicyBase
from AquaML.rlalgo.ExplorePolicy import create_explor_policy
from AquaML.core.AgentIOInfo import AgentIOInfo
from AquaML.data.DataUnit import DataUnit

# TODO: 需要修改非通用agent
class BaseAgent(ABC):
    def __init__(self, name:str,
                 agent_info:AgentIOInfo,
                 agent_params,
                 level:int=0, # 控制是否创建不交互的agent
                 ):
        """
        Agent 基类。所有的Agent都需要继承这个类。
        actor为策略网络的class，value_fn为价值网络的class。

        Args:
            actor (_type_): _description_
        """
        self.name = name
        self.agent_info = agent_info
        self.agent_params = agent_params

        self.level = level

        # 网络输入数据处理信息
        self._network_process_info = {
            'actor': {},
            'critic': {},
        }

        self._explore_dict = {}
        self._tf_explore_dict = {}

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
                shape, _ = self.rl_io_info.get_data_info(name)
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
            policy_out[name] = value

        action, prob = self.explore_policy(policy_out, test_flag=test_flag)

        policy_out['action'] = action
        policy_out['prob'] = prob

        # create return dict according to rl_io_info.actor_out_name
        return_dict = dict()
        for name in self.rl_io_info.actor_out_name:
            return_dict[name] = policy_out[name]

        return return_dict
    
    def create_explorer(self, explore_name, shape, ponited_value={}):
        
        policy, infos = create_explor_policy(
            explore_name=explore_name,
            shape=shape,
            actor_out_names=self.agent_info.actor_out_name,
        )

        for item in infos:
            name = item['name']

            self._explore_dict[name] = DataUnit(
                name=self.name+'_'+name,
                dtype=item['dtype'],
                shape=item['shape'],
                level=self.level,
            )

            if name in ponited_value:
                self._explore_dict[name].set_value(ponited_value[name])

            if item['trainable']:
                vars = tf.Variable(
                    initial_value=self._explore_dict[name].buffer,
                    trainable=True,
                    name=self.name+'_'+name,
                )

                setattr(self,'tf_'+name, vars)

                self._tf_explore_dict[name] = getattr(self,'tf_'+name,)

        self.explore_policy = policy

            
    def update_explorer(self):

        if self.level == 0:
            for name, value in self._tf_explore_dict.items():
                self._explore_dict[name].set_value(value.numpy())

        else:
            for name, value in self._tf_explore_dict.items():
                value.assign(self._explore_dict[name].buffer)

    def create_optimizer(self, optimizer_info):

        type = optimizer_info['type']
        args = optimizer_info['args']

        optimizer = getattr(tf.keras.optimizers, type)(**args)

        return optimizer