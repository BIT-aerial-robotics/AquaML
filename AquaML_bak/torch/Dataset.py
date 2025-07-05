from AquaML import settings, logger
import torch
import numpy as np


class RLDataSet:
    def __init__(self,
                 data_dict: dict,
                 env_nums: int,
                 rollout_steps: int,
                 default_type='numpy', # 'tensor' or 'numpy'
                 default_device=settings.device,
                 ):
        """
        RL算法的数据集。

        Args:
            data_dict (dict): 数据字典.字典中的数据支持两种格式，numpy和torch tensor。
            env_nums (int): 环境的总数量（所有进程的环境总数）。
            rollout_steps (int): 每次rollout的步数。
        """

        self._data_dict = data_dict
        
        self.current_type = default_type
        self.current_device = default_device
        
        # 校验数据的一致性
        if default_type == 'tensor':
            self.to_torch(device=default_device)
        elif default_type == 'numpy':
            self.to_numpy()
        else:
            logger.error(f'Unsupported data type: {default_type}')
            raise TypeError(f'Unsupported data type: {default_type}')
        
        self._env_nums = env_nums
        self._rollout_steps = rollout_steps

        self._total_steps = env_nums * rollout_steps
        self._buffer_size = self._total_steps
        
    def to_numpy(self):
        """
        将数据字典中的数据转换为numpy。
        """
        for key, value in self._data_dict.items():
            if isinstance(value, torch.Tensor):
                self._data_dict[key] = value.cpu().numpy()
        self.current_type = 'numpy'
    
    def to_torch(self, device=settings.device):
        """
        将数据字典中的数据转换为torch tensor。
        """
        for key, value in self._data_dict.items():
            if isinstance(value, np.ndarray):
                self._data_dict[key] = torch.tensor(value, dtype=torch.float32, device=device)
        self.current_type = 'tensor'
        self.current_device = device
    
    def check_device(self,device=settings.device,clean=False):
        """
        检查数据字典中的数据是否为torch tensor，并且是否在指定的设备上。
        如果不是tensor，则转换为tensor，并且放在指定的设备上。
        如果不在指定的设备上，则转移到指定的设备上。
        
        args:
            device (torch.device): 设备。
            clean (bool): 是否清空转换之前的数据。
        """
        
        for key, value in self._data_dict.items():
            if isinstance(value, np.ndarray):
                self._data_dict[key] = torch.tensor(value, device=device)
            elif isinstance(value, torch.Tensor):
                if value.device != device:
                    self._data_dict[key] = value.to(device)
            else:
                logger.error(f'Unsupported data type: {type(value)}')
                raise TypeError(f'Unsupported data type: {type(value)}')
        
        self.current_type = 'tensor'
        self.current_device = device

        if clean:
            # 释放cuda缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                

    
    def concat(self,
               axis: int = 0,
               ):
        """
        将数据字典中的数据进行拼接。

        Args:
            axis (int, optional): 拼接的轴. Defaults to 0.
        """

        for key, value in self._data_dict.items():
            if isinstance(value, torch.Tensor):
                list_value = [value[i] for i in range(value.shape[0])]
                buffer = torch.cat(list_value, axis)
            elif isinstance(value, np.ndarray):
                buffer = np.concatenate(value, axis=axis)
            else:
                logger.error(f'Unsupported data type: {type(value)}')
                raise TypeError(f'Unsupported data type: {type(value)}')
            self._data_dict[key] = buffer

    def reshape(self,
                shape,
                ):
        """
        将数据字典中的数据进行reshape。

        Args:
            shape (tuple): reshape的形状。
        """

        for key, value in self._data_dict.items():
            if isinstance(value, torch.Tensor):
                buffer = value.reshape(shape)
            elif isinstance(value, np.ndarray):
                buffer = np.reshape(value, shape)
            else:
                logger.error(f'Unsupported data type: {type(value)}')
                raise TypeError(f'Unsupported data type: {type(value)}')
            self._data_dict[key] = buffer
            
    def get_corresponding_data(self, 
                               names: tuple, 
                               prefix: str = '', 
                               ):
        """
        获取对应的数据。
        
        在torch中，需要确保数据一致性，要么全为numpy，要么全为torch tensor。

        Args:
            names (tuple): 数据名称。
            prefix (str): 数据名称前缀。
        """

        data = []

        for name in names:
            name = prefix + name
            data.append(self._data_dict[name])

        return data
    
    def add_data(self, name: str, data,device=settings.device):
        """
        添加数据。

        Args:
            name (str): 数据名称。
            data (np.ndarray or torch.Tensor): 数据。
        """
        if self.current_type == 'tensor':
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, device=device)
        elif self.current_type == 'numpy':
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
        else:
            logger.error(f'Unsupported data type: {self.current_type}')
            raise TypeError(f'Unsupported data type: {self.current_type}')
        
        self._data_dict[name] = data
        
    def get_batch(self, batch_size: int):
        """
        获取batch数据。

        Args:
            batch_size (int): batch大小。
        """
        
        indices = np.random.permutation(self._buffer_size)
        
        start_index = 0
        
        while start_index < self._buffer_size:
            end_index = min(start_index + batch_size, self._buffer_size)
            batch_indices = indices[start_index:end_index]
            
            batch_data = {}
            for key, value in self._data_dict.items():
                batch_data[key] = value[batch_indices]
            
            
            yield RLDataSet(
                data_dict=batch_data,
                env_nums=self._env_nums,
                rollout_steps=self._rollout_steps,
                default_type=self.current_type,
                default_device=self.current_device
            )
            
            start_index = end_index
            
    def __getitem__(self, name: str):
        return self._data_dict[name]
    
    def __call__(self, name: str):
        return self._data_dict[name]
        
                
                