import sys
# sys.path.append('/Users/yangtao/Documents/code.nosync/EvolutionRL')
# sys.path.append('/home/yangtao/CODE/AquaML')
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sys.path.append('/home/aquatao/code/rl_framework/AquaML') # 添加运行路径
import AquaML # 导入AquaML，并且初始化
from AquaML.torch.OnlineRL import PPOAlgo, PPOParam # 导入PPO算法以及超参数
from AquaML.framework import RL # 导入运行RL的框架
from AquaML.Tool import GymnasiumMaker # 导入GymnasiumMaker，用于创建环境
import torch 
import numpy as np


# 环境的参数，这里使用Pendulum-v1
env_args = {
    'env_name': 'Pendulum-v1',
    'env_args': {}
}

# 算法的超参数
param = PPOParam(
    rollout_steps=200,
    epoch=100,
    batch_size=1000,
    env_num=20,
    envs_args=env_args,
    summary_steps=200,
    reward_norm=False
)

# 初始化AquaML
AquaML.init(
    hyper_params=param, # 传入超参数
    root_path='test', # 传入文件夹根路径，用于创建文件夹
    memory_path='test', # 传入共享内存标识符
    # wandb_project='test', # 传入wandb的项目名称
    use_tensorboard=True, # 使用tensorboard
    engine='torch'# 传入引擎名称
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
        
        self.learning_rate = 3e-3 # 学习率
        self.optimizer_type = 'Adam' # 优化器类型
        
        # 模型的输出信息
        self.output_info = AquaML.DataInfo(
            names=('action',),
            shapes=((1,),),
            dtypes=(np.float32,)
        )
        
        # 模型的输入信息
        self.input_names = ('obs',)
        
        # 优化器的其他参数，参考pytorch
        self.optimizer_other_args = {
            'eps': 1e-5,
            'clipnorm': 0.5, # 梯度裁剪，这个会自动创建optimizer step
            'scheduler': {
                'type': torch.optim.lr_scheduler.StepLR, # 学习率调度器
                'args': {
                    'step_size': 100,
                    'gamma': 0.9,
                    # 'verbose': True
                }
            }
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
        
        self.learning_rate = 3e-3
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
            'scheduler': {
                'type': torch.optim.lr_scheduler.StepLR,
                'args': {
                    'step_size': 100,
                    'gamma': 0.9
                }
            }
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
    # checkpoint_path='C:/Users/29184/Documents/GitHub/AquaML/test/history_model/PPO/100', # 检查点路径
    # testing=True, # 测试模式
    # save_trajectory=True # 保存轨迹
)

rl.run() # 运行RL