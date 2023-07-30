"""
This script can help you to deploy a complete policy for your device.
"""
import tensorflow as tf
import os
import numpy as np
import copy
from AquaML.core.RLToolKit import Normalization


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


class CompletePolicy:
    # Complete policy for your device.
    # TODO: store the input shape of the network
    def __init__(self,
                 actor,
                 obs_shape_dict,
                 checkpoint_path,
                 using_obs_scale=False,
                 ):

        """
        Complete policy for your device.

        args:
            actor (tf.keras.Model): actor model.
            checkpoint_path (str): checkpoint path.
            using_obs_scale (bool): whether using observation scale.
        """

        ############################
        # TODO: store the input shape of the network
        ############################

        self._network_process_info = {
            'actor': {},
            'critic': {},
        }  # 网络输入数据处理信息

        self.obs_dict = obs_shape_dict

        self.actor = actor()

        # self.initialize_actor()

        self.actor_model_path = os.path.join(checkpoint_path, 'actor.h5')



        self.normalizer = Normalization(obs_shape_dict)

        self.using_obs_scale = using_obs_scale

        if using_obs_scale:
            norm_path = os.path.join(checkpoint_path, 'scaler')
            self.normalizer.load(norm_path)


    def initialize_actor(self, obs_io):
        """
        用于初始化actor，比如说在rnn系统模型里面，某些输入需要额外处理维度。

        #TODO：在2.1版本中逐步将网络输入配置融入到网络定义中。
        """

        # 判断网络类型
        actor_rnn_flag = getattr(self.actor, 'rnn_flag', False)

        setattr(self, 'actor_rnn_flag', actor_rnn_flag)

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

        self.initialize_network(
            model=self.actor,
            expand_dims_idx=self.actor_expand_dims_idx,
            obs_io=obs_io,
        )

        self.actor.load_weights(self.actor_model_path)

    def initialize_network(self, model, obs_io,expand_dims_idx=None):
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
            shape = obs_io.shape_dict[name]
            shape = (1,*shape)
            data = tf.zeros(shape=shape, dtype=tf.float32)
            input_data.append(data)

        if expand_dims_idx is not None:
            for idx in expand_dims_idx:
                input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        model(*input_data)

    def get_action(self, obs, test_flag=False):

        input_data = []

        obs = copy.deepcopy(obs)
        # 获取输入数据
        for name in self.actor.input_name:
            data = tf.cast(obs[name], dtype=tf.float32)
            input_data.append(data)

        # 数据扩展
        # TODO: 后续版本中需要给出数据处理通用接口 backends

        for idx in self.actor_expand_dims_idx:
            input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        actor_out = self.actor(*input_data)

        # squeeze

        # TODO: 这个地方需要优化速度
        policy_out = dict(zip(self.actor.output_info, actor_out))

        if self.actor_rnn_flag:
            policy_out['action'] = tf.squeeze(policy_out['action'], axis=1)

        # if self.agent_params.train_fusion:
        #     policy_out['fusion_value'] = tf.squeeze(policy_out['fusion_value'], axis=1)

        # for name, value in self._explore_dict.items():
        #     policy_out[name] = tf.cast(value.buffer, dtype=tf.float32)

        # action, prob = self.explore_policy(policy_out, test_flag=test_flag)

        # policy_out['action'] = action
        # policy_out['prob'] = prob

        # create return dict according to rl_io_info.actor_out_name
        # return_dict = dict()
        # for name in self.agent_info.actor_out_name:
        #     return_dict[name] = policy_out[name]

        # for name in self.explore_policy.get_aditional_output.keys():
        #     return_dict[name] = policy_out[name]

        return policy_out
