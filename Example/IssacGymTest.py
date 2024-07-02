import sys
sys.path.append('/home/yangtao/CODE/EvolutionRL')

import isaacgym
import isaacgymenvs

import AquaML
from AquaML.torch.OnlineRL import PPOAlgo, PPOParam
from AquaML.framework import RL
from AquaML.Tool import IsaacGymMaker
import torch
import numpy as np


# 环境的参数
env_args = {
    'env_name': 'Ant',
    'env_args': {
    'force_render': True
    }
}

param = PPOParam(
    rollout_steps=64,
    epoch=1800,
    batch_size=8196,
    env_num=256,
    envs_args=env_args,
    summary_steps=1000,
    log_std=-0.,
    # ent_coef=0.01,
    # target_kl=None
)

AquaML.init(
    hyper_params=param,
    root_path='test',
    memory_path='test',
    wandb_project='Benchmark',
    engine='torch'
)

######################
# 定义模型
######################

class Actor(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(60, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 8),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 16),
        )
        
        self.learning_rate = 3e-4
        self.optimizer_type = 'Adam'
        self.output_info = AquaML.DataInfo(
            names=('action',),
            shapes=((8,),),
            dtypes=(np.float32,)
        )
        
        self.input_names = ('obs',)
        
        self.optimizer_other_args = {
            'eps': 1e-5,
            'clipnorm': 1,
        }
        
    def forward(self, obs):
        action = self.linear_relu_stack(obs)
        return (action,)
    
class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(60, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, 1),
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
            'clipnorm': 1,
        }
        
    def forward(self, obs):
        
        value = self.linear_relu_stack(obs)
        return (value,)
    
    
model_dict = {
    'actor': Actor,
    'critic': Critic
}

rl = RL(
    env_class=IsaacGymMaker,
    algo=PPOAlgo,
    hyper_params=param,
    model_dict=model_dict,
    env_type='isaacgym',
)

rl.run()