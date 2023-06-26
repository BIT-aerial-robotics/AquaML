from typing import Any
import numpy as np
from AquaML.buffer.BaseBuffer import BaseBuffer
from abc import ABC, abstractmethod

def len_filter_trajectory(episode_data: dict, len ,len_threshold: int):
    """
    episode的过滤器函数定义指南：
            1. traj_filter的输入为当前episode的数据，输出为True或者False。
    """
    return len >= len_threshold

class TrajectoryFilterRegister:
    """
    traj_filter: episode的过滤器函数定义指南：
            1. traj_filter的输入为当前episode的数据，输出为True或者False。
    """
    def __init__(self,
                 ):
        
        self.len_filter = len_filter_trajectory

        # args
        self._args_dict = {}

    
    def get_filter(self, filter_name):

        if filter_name is None:

            return lambda data, len : True
        
        filter = getattr(self, filter_name+'_filter', None)

        if filter is None:
            raise NotImplementedError
        
        return filter

    def register(self, filter_name: str, filter):
        setattr(self, filter_name+'_filter', filter)
    
    def add_args(self, filter_name: str, args: dict):
        filter = getattr(self, filter_name+'_filter', None)

        if filter is None:
            raise NotImplementedError
        
        self._args_dict[filter_name] = args

    def update_args(self, filter_name: str, args: dict):
        self.add_args(filter_name, args)

    def get_args(self, filter_name: str):
        if filter_name not in self._args_dict.keys():
            raise ValueError('filter_name not in self._args_dict.keys()')
        
        return self._args_dict[filter_name]

    
class RLBufferPluginBase(ABC):
    """
    RLBuffer的插件基类，用于处理RLBuffer的数据。


    为了保证插件的通用性，插件的输入为data_set_dict（从环境中收集或者网路的batch输出），前面插件处理得到的中间变量（key_data_dict）。

    插件的输出为处理后的data_set_dict和key_data_dict。若该插件输出网络训练数据时候可以为list，
    list里面包含batch_data_dict和batch_key_data_dict。


    """
    def __init__(self,
                 ):
        
        ###############################################
        # 处理接口部分
        ###############################################
        
        # plugin的名称
        self.name = None

        # 需要额外处理的数据
        self._key_data_info = []

        # 需要的前置插件
        self._pre_plugin = []

        # additional parameter
        self._additional_param = {}

    
    def __call__(self, data_set_dict: dict, log ,key_data_dict):
        """
        插件的调用接口，用于处理数据。

        args:
            data_set_dict: 从环境中收集或者网路的batch输出
            log: 前置插件处理名称。
            key_data_dict: 前置插件处理得到的中间变量
        """

        # 检查是否有前置插件
        for pre_plugin_name in self._pre_plugin:
            if pre_plugin_name not in log:
                raise ValueError("pre_plugin_name is not in log")
        
        # 检查额外数据是否存在
        for key_data_name in self._key_data_info:
            if key_data_name not in key_data_dict:
                raise ValueError("key_data_name is not in key_data_dict")
            
        # process
        processed_data_set_dict, processed_key_data_dict = self._process(data_set_dict, key_data_dict, **self._additional_param)

        return processed_data_set_dict, processed_key_data_dict
    
    @abstractmethod
    def _process(self, data_set_dict: dict, key_data_dict, **kwargs)->tuple:
        """
        插件的处理接口，用于处理数据。

        args:
            data_set_dict: 从环境中收集或者网路的batch输出
            log: 前置插件处理名称。
            key_data_dict: 前置插件处理得到的中间变量
        returns:
            data_set_dict: 处理后的数据。
            key_data_dict: 处理后的数据。
        """
 
    
    @property
    def get_name(self):
        if self.name is None:
            raise ValueError("name is None")
        return self.name


class SplitTrajectoryPlugin(RLBufferPluginBase):
    def __init__(self,
                filter_name = None,
                concat: bool = False,
                filter_args: dict = {}
                 ):
        super().__init__()

        from AquaML import traj_filter_register

        self.filter = traj_filter_register.get_filter(filter_name)
        self.filter_args = filter_args

        self.concat = concat
        # if filter_name is None:
        #     self.filter_args = {}
        # else:
        #     self.filter_args = traj_filter_register.get_args(filter_name)
        
        ###############################################
        # 处理接口部分
        ###############################################
        self.name = "split_trajectory"
        self._additional_param["concat"] = concat

        # 需要额外处理的数据
        self._key_data_info.append("mask")
    
    def _process(self, data_set_dict: dict,key_data_dict:dict):
        f"""
        插件的处理接口，用于处理数据。
        由__call__调用.

        


        args:
            data_set_dict: 从环境中收集或者网路的batch输出。
            key_data_dict: 前置插件处理得到的中间变量。
            episode_filter: episode的过滤器, 用于过滤episode,注意这是一个函数。
            concat: 是否将episode拼接成一个numpy数组。

        Returns:
            data_set_dict: 处理后的数据。
            
            key_data_dict: {
                "len_episode": length of episode,
                "start_index": start index of episode,
                "end_index": end index of episode,
            }
            
            if concat:
                the element of data_set_dict is a numpy array.
            else:
                the element of data_set_dict is a list.
        """
        
        # 获取数据
        data_set_dict = data_set_dict.copy()
        mask = key_data_dict["mask"].copy()

        # 拆分轨迹
        
        terminal = np.where(mask == 0)[0] + 1
        start = np.append(np.array(0), terminal[:-1])

        terminal = terminal.tolist()
        start = start.tolist()

        new_start = []
        new_terminal = []

        len_episodes = []

        current_episode = 0

        ret_dict = {}

        for start_index, end_index in zip(start, terminal):

            len_episode = end_index - start_index
            episode_dict = {}

            for name, var in data_set_dict.items():
                episode_dict[name] = var[start_index:end_index]
            
            # 调用filter
            if self.filter(episode_dict, len_episode ,**self.filter_args):

                for name, var in episode_dict.items():
                    if name not in ret_dict:
                        ret_dict[name] = []
                    ret_dict[name].append(var)
                
                len_episodes.append(len_episode)
                new_start.append(current_episode)
                new_terminal.append(current_episode)
                current_episode += 1
        
        if self.concat:
            for name, var in ret_dict.items():
                ret_dict[name] = np.concatenate(var, axis=0)
            
            end_index = np.cumsum(len_episodes).tolist()
            start_index = np.append(np.array(0), end_index[:-1]).tolist()
        
        else:
            start_index = np.array(new_start).tolist()
            end_index = np.array(new_terminal).tolist()

        new_key_data_dict = {}
        new_key_data_dict["len_episodes"] = len_episodes
        new_key_data_dict["start_index"] = start_index
        new_key_data_dict["end_index"] = end_index
    

        return ret_dict, new_key_data_dict

class PaddingPlugin(RLBufferPluginBase):

    def __init__(self,
                 max_steps,
                 concat: bool = False,
                 ):

        super().__init__()
        self.max_steps = max_steps
        self.concat = concat

        ###############################################
        # 处理接口部分
        ###############################################
        self.name = "padding"
        self._additional_param["max_steps"] = max_steps

        # 需要额外处理的数据
        self._key_data_info.append("len_episodes")
        self._key_data_info.append("start_index")
        self._key_data_info.append("end_index")

        # pre plugin
        self._pre_plugin.append("split_trajectory")
    
    def _process(self, data_set_dict: dict,key_data_dict:dict):
        
        # 获取数据
        data_set_dict = data_set_dict.copy()
        len_episodes = key_data_dict["len_episodes"].copy()
        start_index = key_data_dict["start_index"].copy()
        end_index = key_data_dict["end_index"].copy()

        # 拆分轨迹
        ret_dict = {}

        grad_masks = []


        for start, end, len_episode in zip(start_index, end_index, len_episodes):
            for name, var in data_set_dict.items():
                
                if isinstance(var, np.ndarray):
                    episodes = var[start:end]
                elif isinstance(var, list):
                    episodes = var[start]
                else:
                    raise TypeError("data type not support")

                if len_episode < self.max_steps:
                    padding = np.zeros((self.max_steps - len_episode, *episodes.shape[1:]))
                    episodes = np.concatenate([episodes, padding], axis=0)
                
                if name not in ret_dict:
                    ret_dict[name] = []
                ret_dict[name].append(episodes)

                grad_enabled = np.ones((len_episode, 1))
                grad_disabled = np.zeros((self.max_steps - len_episode, 1))
                grad_mask = np.concatenate([grad_enabled, grad_disabled], axis=0)

                grad_masks.append(grad_mask)
        
        if self.concat:
            for name, var in ret_dict.items():
                ret_dict[name] = np.concatenate(var, axis=0)
            
            grad_masks = np.concatenate(grad_masks, axis=0)
            
            start_index = np.ones((len(len_episodes), 1))*self.max_steps
            end_index = np.cumsum(start_index).tolist()
            start_index = np.append(np.array(0), end_index[:-1]).tolist()

        else:
            start_index = np.arange(len(len_episodes)).tolist()
            end_index = np.arange(len(len_episodes)).tolist()      

        new_key_data_dict = {}
        new_key_data_dict["start_index"] = start_index
        new_key_data_dict["end_index"] = end_index
        new_key_data_dict["grad_masks"] = grad_masks

        return ret_dict, new_key_data_dict     
            
            
class RLBufferPluginRegister:
    """
    Register and manage for RLBufferPlugin.

    """

    def __init__(self):
        self.SplitTrajectoryPlugin = SplitTrajectoryPlugin

        ###############################################
        # 插件注册部分
        # 要使用的插件放入这里
        ###############################################
        self._used_plugin_dict = {}
    
    def register(self,name:str,plugin):
        """
        注册新插件。

        Args:
            plugin (RLBufferPluginBase): 插件。
        """
        
        if not isinstance(plugin, RLBufferPluginBase):
            raise ValueError("plugin must be a instance of RLBufferPluginBase")

        setattr(self, name+'Plugin', plugin)
    
    def add_used_plugin(self, name:str, args:dict = {}):
        """
        使用插件。

        Args:
            name (str): 插件名称。
            args (dict, optional): 插件参数。 Defaults to {}.
        """

        Pluguin = getattr(self, name+'Plugin', None)
        if Pluguin is None:
            raise ValueError(f"plugin {name} is not registered")
        
        self._used_plugin_dict[name] = Pluguin(**args)

    def get_used_plugin(self, name:str):
        """
        获取使用的插件。

        Args:
            name (str): 插件名称。

        Returns:
            RLBufferPluginBase: 插件。
        """

        if name not in self._used_plugin_dict:
            raise ValueError(f"plugin {name} is not used")
        
        return self._used_plugin_dict[name]
    
    def get_plugin(self, name:str):
        """
        获取插件。

        Args:
            name (str): 插件名称。

        Returns:
            RLBufferPluginBase: 插件。
        """

        if not hasattr(self, name+'Plugin'):
            raise ValueError(f"plugin {name} is not registered or implemented")
        
        return getattr(self, name+'Plugin')


        
        

class OnPolicyDefaultReplayBuffer:
    """
    RL 第一个经验池，只是单纯的存储经验，对LSTM输入进行部分处理后续将增加更强的经验池。

    该经验池可以在RL算法中可以创建多个，默认中我们将为根据数据集合分别为actor和critic创建两个经验池。

    """

    def __init__(self):
        self.data = {}


    def add_sample(self, data_set: dict, masks: np.ndarray, pluging_dict: dict = {}):
        """
        添加一个样本。

        Args:
            data_set (dict): 数据集合。
            masks (np.ndarray): 掩码。
            pluging_dict (dict, optional): 插件字典。 Defaults to {}.
        """

        # 操作过程记录
        log = []
        key_data_dict = {
            'masks': masks
        }

        for plugin_name, plugin in pluging_dict.items():
            log.append(plugin.get_name)
            processed_data_set, processed_key_data_dict = plugin(data_set, key_data_dict)
            key_data_dict.update(processed_key_data_dict)
            data_set.update(processed_data_set)
        
        self.data.update(data_set)
    



        
