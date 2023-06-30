import tensorflow as tf
import numpy as np
from AquaML.core.DataParser import DataSet
class OnPolicyDefaultReplayBuffer:
    """
    RL 第一个经验池，只是单纯的存储经验，对LSTM输入进行部分处理后续将增加更强的经验池。

    该经验池可以在RL算法中可以创建多个，默认中我们将为根据数据集合分别为actor和critic创建两个经验池。

    """

    def __init__(self, concat_flag, batch_normalization: list = []):
        self.data = {}

        self.concat_flag = concat_flag

        self.buffer_size = 0

        self.batch_normalization = batch_normalization

    def add_sample(self, data_set: dict, masks: np.ndarray, plugin_dict: dict = {}):
        """
        添加一个样本。

        Args:
            data_set (dict): 数据集合。
            masks (np.ndarray): 掩码。
            pluging_dict (dict, optional): 插件字典。 Defaults to {}.
            data_format (str, optional): 数据格式。 Defaults to 'list'. 如果是list格式，那么将会对数据进行拆分，
            按照episode进行存储，如果是dict格式，numpy array.
        """

        # if data_format == 'list':
        #     concat_flag = False
        # elif data_format == 'numpy':
        #     concat_flag = True
        # else:
        #     raise ValueError("data_format must be list or numpy")

        concat_flag = self.concat_flag

        buffer_size = masks.shape[0]

        # 操作过程记录
        log = []
        key_data_dict = {
            'mask': masks
        }

        for plugin_name, plugin in plugin_dict.items():
            log.append(plugin.get_name)
            processed_data_set, processed_key_data_dict, buffer_size = plugin(data_set, log, key_data_dict,
                                                                              concat=concat_flag)
            key_data_dict.update(processed_key_data_dict)
            data_set.update(processed_data_set)

        self.buffer_size = buffer_size
        self.data.update(data_set)

    def sample_batch(self, batch_size: int, ):
        """
        从经验池中采样一个batch。

        Args:
            batch_size (int): batch大小。

        Returns:
            dict: batch数据。
        """

        # 打乱数据
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0

        while start_idx < self.buffer_size:
            end_idx = min(start_idx + batch_size, self.buffer_size)
            batch_indices = indices[start_idx:end_idx]

            batch = {}
            for key, val in self.data.items():
                if isinstance(val, list) or isinstance(val, tuple):
                    buffer = []
                    for inner_val in val:
                        buffer.append(tf.cast(inner_val[batch_indices], tf.float32))
                    batch[key] = tuple(buffer)
                else:
                    batch[key] = tf.cast(val[batch_indices], tf.float32)

            # TODO: 创建batch_normalization插件提供基础的batch_normalization功能
            for key in self.batch_normalization:
                batch[key] = (batch[key] - tf.reduce_mean(batch[key])) / (tf.math.reduce_std(batch[key]) + 1e-8)

            yield batch

            start_idx += batch_size