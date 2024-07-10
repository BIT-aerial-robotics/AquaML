import sys
sys.path.append('/Users/yangtao/Documents/code.nosync/AquaML')
# sys.path.append('/home/yangtao/CODE/AquaML')
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sys.path.append('C:/Users/29184/Documents/GitHub/AquaML') # 添加运行路径
import AquaML # 导入AquaML，并且初始化
from AquaML.torch.OnlineRL.SACAlgo_gpt4o import SACAlgo, SACParam # 导入SAC算法以及超参数
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
param = SACParam(
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
    memory_path='test', # 传入贡献内存标识符
    wandb_project='test', # 传入wandb的项目名称
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
            torch.nn.Linear(64, 2),
        )
        
        self.learning_rate = 3e-4 # 学习率
        self.optimizer_type = 'Adam' # 优化器类型
        
        # 模型的输出信息
        self.output_info = AquaML.DataInfo(
            names=('action', 'log_prob'),
            shapes=((1,), (1,)),
            dtypes=(np.float32, np.float32)
        )
        
        # 模型的输入信息
        self.input_names = ('obs',)
        
        # 优化器的其他参数，参考pytorch
        self.optimizer_other_args = {
            'eps': 1e-5,
            'clipnorm': 0.5, # 梯度裁剪，这个会自动创建optimizer step
        }
        
    def forward(self, obs):
        action = self.linear_relu_stack(obs)
        log_prob = -0.5 * (action.pow(2) + torch.log(2 * np.pi * torch.ones_like(action)))
        return (action, log_prob)
    
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
    'critic_1': Critic,
    'critic_2': Critic
}

rl = RL(
    env_class=GymnasiumMaker,
    algo=SACAlgo,
    hyper_params=param,
    model_dict=model_dict,
    # checkpoint_path='C:/Users/29184/Documents/GitHub/AquaML/test/history_model/SAC/100', # 检查点路径
    # testing=True, # 测试模式
    # save_trajectory=True # 保存轨迹
)

rl.run() # 运行RL