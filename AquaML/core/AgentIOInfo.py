from AquaML.core.DataParser import DataInfo
from abc import ABC, abstractmethod
import numpy as np

class AgentIOInfoBase(ABC):

    def __init__(self,
                    agent_name:str,
                 ):
        self.agent_name = agent_name

    @abstractmethod
    def get_agent_name(self):
        return self.agent_name
    
    @property
    def get_data_info(self):
        return self.data_info


class RLAgentIOInfo(AgentIOInfoBase):
    """
    Information of single agent input and output.
    
    单个agent的输入输出信息。
    
    执行策略输入一般都是observation，critic输入为全部state

    """
    
    def __init__(self, agent_name:str,
                 obs_info:dict,
                 obs_type_info,
                 actor_out_info:dict,
                 reward_name:tuple,
                 buffer_size:int,
                 critic_avalible_name:tuple or None=None,
                 ):
        
        """
        记录agent输入输出信息。
        
        每个agent可以拥有自己的独立的评估方法。

        args:
            agent_name (str): agent name.
            obs_info (dict): observation information.
            obs_type_info (dict or str): observation type information.
            actor_out_info (dict): actor output information.
            reward_info (tuple): reward information.
            buffer_size (int): buffer size.
            critic_avalible_info (tuple or None): critic avalible information. 用于指定critic的输入，当不额外提供时默认为全部state。
        """
        
        super().__init__(
            agent_name=agent_name,
            )
        self.agent_name = agent_name
        
        # insert buffer size into shapes
        def insert_buffer_size(shape):
            shapes = []
            shapes.append(buffer_size)

            if isinstance(shape, tuple):
                for val in shape:
                    shapes.append(val)
            else:
                shapes.append(shape)

            shapes = tuple(shapes)

            return shapes
        
        data_info_dict = dict()
        data_type_info_dict = dict()
        
        # add obs_info to data_info
        for key, shape in obs_info.items():
            data_info_dict[key] = insert_buffer_size(shape)
            if isinstance(obs_type_info, dict):
                data_type_info_dict[key] = obs_type_info[key]
            else:
                data_type_info_dict[key] = obs_type_info
                
        # add next_obs_info to data_info
        for key in obs_info.keys():
            data_info_dict['next_' + key] = data_info_dict[key]
            data_type_info_dict['next_' + key] = data_type_info_dict[key]
            
        # check 'action' whether in actor_out_info
        # if not, rasing error
        if 'action' not in actor_out_info:
            raise ValueError("actor_out_info must have 'action'")
        
        # add mask_info to data_info
        data_info_dict['mask'] = (buffer_size, 1)
        data_type_info_dict['mask'] = np.int32
        
        # add actor_out_info to data_info
        for key, shape in actor_out_info.items():
            data_info_dict[key] = insert_buffer_size(shape)
            data_type_info_dict[key] = np.float32

        # add reward_info to data_info
        for key in reward_name:
            data_info_dict[key] = (buffer_size, 1)
            data_type_info_dict[key] = np.float32
            
        

        #######################################
        # 整合IO数据信息
        #######################################

        self.data_info = DataInfo(
            names=tuple(data_info_dict.keys()),
            shapes=tuple(data_info_dict.values()),
            types=tuple(data_type_info_dict.values()),
        )

        # 记录关键信息的名称
        self.reward_name = reward_name
        self.actor_out_name = actor_out_info.keys()
        self.action_shape = actor_out_info['action']


        # 获取该agent的critic输入信息
        if critic_avalible_name is None:
            self.critic_avalible_name = tuple(obs_info.keys())

        else:
            self.critic_avalible_name = critic_avalible_name

            # 检测critic_avalible_name是否在obs_info中
            for name in critic_avalible_name:
                if name not in obs_info:
                    raise ValueError("critic_avalible_name {} must be in obs_info".format(name))