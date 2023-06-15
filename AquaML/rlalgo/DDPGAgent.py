'''
AquaML下的多智能体强化学习算法的第一个，用于探索框架设计。  
'''

import numpy as np
import tensorflow as tf

from AquaML.rlalgo.BaseAgent import BaseAgent
from AquaML.AgentInfo import AgentInfo
from AquaML.rlalgo.ExplorePolicy import OrnsteinUhlenbeckExplorePolicy


class DDPGAgent(BaseAgent):

    """
    为了保证可以支持复杂模型，model都是以类的方式传递进来的。
    """

    def __init__(self,
        name:str,
        actor,  
        critic,
        agent_info:AgentInfo,
        level:int=0, # 控制是否创建不交互的agent
    ):
        
        super().__init__(
            name=name,
            agent_info=agent_info,
            level=level,
            )
        


        self.actor = actor() # 注意传入的是类
        self.initialize_actor()

        # 创建target网络和critic网络
        if self.level == 0:
            self.target_actor = actor()
            self.target_critic = critic()
            self.critic = critic()

            # 初始化critic网络
            self.initialize_critic()

            # 初始化target网络
            self.initialize_network(
                model=self.target_actor,
                expand_dims_idx=self.actor_expand_dims_idx,
            )
            self.initialize_network(
                model=self.target_critic,
                expand_dims_idx=self.critic_expand_dims_idx,
            )

            self.copy_weights(
                source=self.actor,
                target=self.target_actor,
            )

            self.copy_weights(
                source=self.critic,
                target=self.target_critic,
            )

            # 创建优化器
            # 检测actor是否包含优化器参数
            if hasattr(self.actor, 'optimizer_args'):
                self.actor_optimizer = tf.keras.optimizers.Adam(**self.actor.optimizer_args)
            else:
                raise AttributeError(f'{self.actor.__class__.__name__} has no optimizer_args attribute')
            
            # 检测critic是否包含优化器参数
            if hasattr(self.critic, 'optimizer_args'):
                self.critic_optimizer = tf.keras.optimizers.Adam(**self.critic.optimizer_args)
            else:
                raise AttributeError(f'{self.critic.__class__.__name__} has no optimizer_args attribute')
            
        
        def train_actor(self,
                        actor_inputs,
                        critic_inputs,
                        mask,
                        ):
            
            with tf.GradientTape() as tape:
                # 监测变量
                tape.watch(self.actor.trainable_variables)
                

    
        

