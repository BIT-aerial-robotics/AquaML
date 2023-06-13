from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
from AquaML.DataType import DataInfo
from AquaML.data.DataUnit import DataUnit


# TODO: implement the base class when implementing the other classes
class BaseAlgo(ABC):
    def __init__(self):

        self.data_pool = None # 数据池子
        self.sample_thread_num = None # 能够使用的子线程数目
        self.each_summary_episodes = None # 验证数目
        self.algo_info = None

    @property
    def algo_name(self):
        return self.name

    def create_data_pool(self):
        """
        由于在2.1版本中，所有的data pool将逐渐统一这个函数，这个函数也会逐渐改进扩展。
        """

        # 添加summary信息到meta_info中
        reward_info_dict = {}
        for name in self.algo_info.reward_info:
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
            self.algo_info.add_info(
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
            self.algo_info.data_info,
            type='buffer'
        )

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

    def load_weights_from_file(self):
        """
        load weights from file.
        """
        for key, model in self._all_model_dict.items():
            path = getattr(model, 'weight_path', None)
            if path is not None:
                model.load_weights(path)
                print("load {} weight from {}".format(key, path))

    
