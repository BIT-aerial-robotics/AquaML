import tensorflow as tf
import numpy as np


class RLDataset:
    def __init__(self,
                 data_dict: dict,
                 env_nums: int,
                 rollout_steps: int,
                 ):
        """
        RL算法的数据集。

        Args:
            data_dict (dict): 数据字典。
            env_nums (int): 环境的总数量（所有进程的环境总数）。
            rollout_steps (int): 每次rollout的步数。
        """
        
        self._data_dict = data_dict
        self._env_nums = env_nums
        self._rollout_steps = rollout_steps
        
        self._total_steps = env_nums * rollout_steps
        self._buffer_size = self._total_steps
        
    def concat(self,
               axis: int = 0,
               tf_tensor: bool = False,
               dtype=tf.float32
               ):
        """
        将数据字典中的数据进行拼接。

        Args:
            axis (int, optional): 拼接的轴. Defaults to 0.
        """
        
        for key, value in self._data_dict.items():
            
            buffer = np.concatenate(value, axis=axis)
            
            if tf_tensor:
                tf_buffer = tf.convert_to_tensor(buffer, dtype=dtype)
                self._data_dict[key] = tf_buffer
            else:
                self._data_dict[key] = buffer
                
    def reshape(self,
                shape,
                tf_tensor: bool = False,
                dtype=tf.float32
                ):
        
        for key, value in self._data_dict.items():
                
                buffer = np.reshape(value, shape)
                
                if tf_tensor:
                    tf_buffer = tf.convert_to_tensor(buffer, dtype=dtype)
                    self._data_dict[key] = tf_buffer
                else:
                    self._data_dict[key] = buffer
        
    def get_corresponding_data(self, 
                               names: tuple, 
                               prefix: str = '', 
                               tf_tensor: bool = False,
                               dtype=tf.float32
                               ):
        """
        获取对应的数据。
        
        Args:
            data_dict (dict): 数据字典。
            names (tuple): 数据名称。
            prefix (str): 数据名称前缀。
            concat_axis (int): 拼接的轴。
            tf_tensor (bool): 是否返回tf tensor。
        """

        data = []

        for name in names:
            name = prefix + name
            buffer = self._data_dict[name]
            
            if tf_tensor:
                tf_buffer = tf.convert_to_tensor(buffer, dtype=dtype)
                data.append(tf_buffer)
            else:
                data.append(buffer)

        return data
    
    def add_data(self, name:str, data):
        """
        添加数据。
        
        Args:
            name (str): 数据名称。
            data : 数据。
        """
        
        self._data_dict[name] = data
    
    
    def get_batch(self, batch_size:int):
        """
        获取batch数据。
        
        Args:
            batch_size (int): batch的大小。
        """
        
        indices = np.random.permutation(self._buffer_size)
        
        start_index = 0

        while start_index < self._buffer_size:
            end_index = min(start_index + batch_size, self._buffer_size)
            batch_indices = indices[start_index:end_index]

            batch = {}
            for key, val in self._data_dict.items():
                batch[key] = tf.cast((val[batch_indices]), tf.float32)

            yield RLDataset(data_dict=batch,
                            env_nums=self._env_nums,
                            rollout_steps=self._rollout_steps,
                            )

            start_index = end_index
        
    
    def __call__(self,name:str):
        return self._data_dict[name]
    
    def __getitem__(self,name:str):
        return self._data_dict[name]
        
