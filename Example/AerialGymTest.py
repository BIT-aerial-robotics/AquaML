import sys
sys.path.append('/home/xi/Documents/AquaML')
sys.path.append("/home/xi/Documents/aerial_gym_simulator_ntnu")
# sys.path.append('/home/xi/Documents')
# sys.path.append('/home/xi/Documents/IsaacGymEnvs-main')

import isaacgym
import isaacgymenvs

import AquaML
from AquaML.torch.OnlineRL import PPOAlgo, PPOParam
from AquaML.framework import RL
from AquaML.Tool import IsaacGymMaker
from AquaML.Tool.AerialGymMaker import AerialGymMaker
import torch
import numpy as np


# 环境的参数
env_args = {
    'env_name': 'model',
    'env_args': {
            'force_render': True,
    }
}

param = PPOParam(
    rollout_steps=64,
    epoch=2000,
    batch_size=256 * 16,
    env_num=256,
    envs_args=env_args,
    summary_steps=1600,
    update_times=4,
    # ent_coef=0.01,
    # target_kl=None
)

AquaML.init(
    hyper_params=param,
    root_path='test3',
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
            torch.nn.Linear(13, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 4),
        )
        
        self.learning_rate = 0.0026
        self.optimizer_type = 'Adam'
        self.output_info = AquaML.DataInfo(
            names=('action',),
            shapes=((4,),),
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
            torch.nn.Linear(13, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        
        
        self.learning_rate = 0.0003
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
    env_class=AerialGymMaker,
    algo=PPOAlgo,
    hyper_params=param,
    model_dict=model_dict,
    env_type='aerialgym',
)

rl.run()