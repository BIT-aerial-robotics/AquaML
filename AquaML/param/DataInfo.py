'''

我实在不知道咋取名字了，暂时用DataInfo这个名字吧。
'''

import yaml
import numpy as np

class Element:
    
    def __init__(self, name, dtype, shape, size=1):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.size = size
    
    def set_size(self, size):
        self.size = size

class DataInfo:
    def __init__(self,scope_name:str):
        self._scope_name = scope_name # 该参数名称
        self.info_dict = {}
        
        # 添加系统状态
        self.add_element(
            name='program_running_state',
            dtype=np.bool_,
            shape=(2,),
            size=1
        )
        
        
        # 强化学习接口
        self._rl_state_names = None
        self._rl_action_names = None
        self._rl_state_dict = None
        self._rl_action_dict = None
        
    def add_element(self, name:str, dtype, shape:tuple, size:int=1):
        """
        添加一个数据元素描述。

        Args:
            name (str): 数据元素名称。
            dtype (_type_): 数据元素类型。
            shape (tuple): 数据元素维度。
            size (int, optional): 数据元素大小。默认为1。 Defaults to 1.
        """
        setattr(self, name, Element(name, dtype, shape, size))
        self.info_dict[name] = getattr(self, name)
    
    def read_from_yaml(self, yaml_path:str):
        """
        从yaml文件中读取数据信息。
        
        我们建议使用yaml去配置数据信息，这样可以方便的进行数据的管理。

        Args:
            yaml_path (str): yaml文件路径。
        """
        with open(yaml_path, 'r') as f:
            data = yaml.load(f)
            
        for name, info in data.items():
            self.add_element(name, info['dtype'], info['shape'])
            
    
    def add_size_to_element(self, name:str, size:int):
        """
        添加数据元素大小。

        Args:
            name (str): 数据元素名称。
            size (int): 数据元素大小。
        """
        self.info_dict[name].set_size(size)
    
    ######################################## 强化学习接口 ########################################‘
    # 设置强化学习的状态
    def set_rl_state(self, state_names:list):
        """
        设置强化学习的状态。

        Args:
            state_names (list): 状态。
        """
        
        # 检测状态是否在info_dict中
        
        for state_name in state_names:
            if state_name not in self.info_dict:
                raise ValueError('State name {} not in info_dict'.format(state_name))

        self._rl_state_names = state_names
        
        self._rl_state_dict = {}
        
        for state_name in state_names:
            self.rl_state_dict[state_name] = self.info_dict[state_name]
            
    def set_rl_rewards(self, reward_names:list):
        
        for reward_name in reward_names:
            if reward_name not in self.info_dict:
                raise ValueError('Reward name {} not in info_dict'.format(reward_name))
        
        self._rl_reward_names = reward_names
        
        self._rl_reward_dict = {}
        
        for reward_name in reward_names:
            self._rl_reward_dict[reward_name] = self.info_dict[reward_name]
    
    # 设置强化学习的动作
    def set_rl_action(self, action_names:list):
        """
        设置强化学习的动作。

        Args:
            action_names (list): 动作。
        """
        
        # 检测动作是否在info_dict中
        
        for action_name in action_names:
            if action_name not in self.info_dict:
                raise ValueError('Action name {} not in info_dict'.format(action_name))

        self._rl_action_names = action_names
        
        self._rl_action_dict = {}
        
        for action_name in action_names:
            self.rl_action_dict[action_name] = self.info_dict[action_name]
            
    def set_rl_action2(self, action_name):
        """
        设置强化学习的动作。
        
        不同于set_rl_action，该函数只设置一个动作。
        
        Args:
            action_name (str): 动作。
        """
        
        self._rl_action_name = action_name
        
        self._rl_action_info = self.info_dict[action_name]
        
    def rl_init(self, policy_num):
        """
        
        运行RL必须要的data_unit信息，以及返回对应的名称。
        """
        
        self.add_element(
            name='env_control_action_flag',
            dtype=np.bool_,
            shape=(1,),
            size=1
        )
        
        self.add_element(
            name='env_control_state_flag',
            dtype=np.bool_,
            shape=(policy_num,),
            size=1
        )
        
        
        ret_dict = {
            'control_action_flag': 'env_control_action_flag',
            'control_state_flag': 'env_control_state_flag'
        }
        
        return ret_dict
    
    # 返回强化学习的状态
    @property
    def rl_state_names(self):
        
        if self._rl_state_names is None:
            raise ValueError('RL state names not set')
        
        return self._rl_state_names
    
    # 返回强化学习的动作
    @property
    def rl_action_names(self):
        
        if self._rl_action_names is None:
            raise ValueError('RL action names not set')
        
        return self._rl_action_names
    
    # 返回强化学习的状态字典
    @property
    def rl_state_dict(self):
        
        if self._rl_state_dict is None:
            raise ValueError('RL state dict not set')
        
        return self._rl_state_dict

    # 返回强化学习的动作字典
    @property
    def rl_action_dict(self):
        
        if self._rl_action_dict is None:
            raise ValueError('RL action dict not set')
        
        return self._rl_action_dict
    
    @property
    def rl_action_name(self):
        
        if self._rl_action_name is None:
            raise ValueError('RL action name not set')
        
        return self._rl_action_name
    
    @property
    def rl_action_info(self):
        
        if self._rl_action_info is None:
            raise ValueError('RL action info not set')
        
        return self._rl_action_info

    @property
    def rl_reward_names(self):

        if self._rl_reward_names is None:
            raise ValueError('RL reward names not set')

        return self._rl_reward_names

    @property
    def rl_reward_dict(self):

        if self._rl_reward_dict is None:
            raise ValueError('RL reward dict not set')

        return self._rl_reward_dict
        