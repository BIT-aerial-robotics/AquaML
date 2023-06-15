from abc import ABC, abstractmethod
import tensorflow as tf
from AquaML.core.BaseAlgo import BaseAlgo
from AquaML.data.DataPool import DataPool
from AquaML.DataType import DataInfo

class BaseMARL(BaseAlgo,ABC):
    """
    多智能体强化学习的基础类，基类会提供一些基础的函数，方便子类调用。
    
    
    其功能如下：
    1. 为每个Agent创建并维护专属的data pool。
    2. 创建数据记录系统。
    3. 维护哪些数据是全局数据哪些是局部数据。
    
    """
    
    def __init__(self,
                 name:str,
                 ):
        super().__init__()
        
        self.name = name
        
        self.agents_pool = {}
        
    
    def add_agents(self, agents: list, level: int = 0):
        """
        添加多个agent到MARL中。支持动态创建agent。
        
        agents 列表里面的每个元素都是一个agent的class。
        
        Args:
            agents (list): agent列表。
        """
        
        for agent in agents:
            self.agents_pool[agent.name] = agent
            
        # 创建专属的data pool
        for agent in agents:
            data_info = agent.agent_info.data_info
            agent_pool = DataPool(
                name=agent.name,
                level=level,
            )
            agent_pool.create_buffer_from_dict(
                data_info=data_info,
            )
            
            self.agents_pool[agent.name] = agent_pool
            
            # TODO: 创建summary reward buffer
        