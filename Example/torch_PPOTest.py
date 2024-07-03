import sys
# sys.path.append('/Users/yangtao/Documents/code.nosync/EvolutionRL')
sys.path.append('/home/yangtao/CODE/AquaML')
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# sys.path.append('C:/Users/29184/Documents/GitHub/AquaML') # TODO： 运行时请修改路径
import AquaML
from AquaML.torch.OnlineRL import PPOAlgo, PPOParam
from AquaML.framework import RL
from AquaML.Tool import GymnasiumMaker
import torch
import numpy as np


# 环境的参数
env_args = {
    'env_name': 'Pendulum-v1',
    'env_args': {}
}

param = PPOParam(
    rollout_steps=200,
    epoch=100,
    batch_size=1000,
    env_num=20,
    envs_args=env_args,
    summary_steps=200,
)

AquaML.init(
    hyper_params=param,
    root_path='test',
    memory_path='test',
    wandb_project='test',
    engine='torch'
)

######################
# 定义模型
######################

class Actor(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        
        self.learning_rate = 3e-4
        self.optimizer_type = 'Adam'
        self.output_info = AquaML.DataInfo(
            names=('action',),
            shapes=((1,),),
            dtypes=(np.float32,)
        )
        
        self.input_names = ('obs',)
        
        self.optimizer_other_args = {
            'eps': 1e-5,
            'clipnorm': 0.5,
        }
        
    def forward(self, obs):
        action = self.linear_relu_stack(obs)
        return (action,)
    
class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        
        self.learning_rate = 3e-4
        self.optimizer_type = 'Adam'
        self.output_info = AquaML.DataInfo(
            names=('value',),
            shapes=((1,),),
            dtypes=(np.float32,)
        )
        
        self.input_names = ('obs',)
        
        self.optimizer_other_args = {
            'eps': 1e-5,
            'clipnorm': 0.5,
        }
        
    def forward(self, obs):
        
        value = self.linear_relu_stack(obs)
        return (value,)
    
    
model_dict = {
    'actor': Actor,
    'critic': Critic
}

rl = RL(
    env_class=GymnasiumMaker,
    algo=PPOAlgo,
    hyper_params=param,
    model_dict=model_dict,
)

rl.run()