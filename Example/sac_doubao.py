import sys
sys.path.append('/Users/yangtao/Documents/code.nosync/AquaML')
import AquaML  # 导入AquaML，并且初始化
from AquaML.torch.OnlineRL.SAC_doubao import SACAlgo, SACParam  # 导入SAC算法以及超参数
from AquaML.framework import RL  # 导入运行RL的框架
from AquaML.Tool import GymnasiumMaker  # 导入GymnasiumMaker，用于创建环境
import torch
import numpy as np


# 环境的参数，这里使用Pendulum-v1
env_args = {
    'env_name': 'Pendulum-v1',
    'env_args': {}
}

# 算法的超参数
param = SACParam(
    total_timesteps=100000,
    buffer_size=1000000,
    gamma=0.99,
    tau=0.005,
    batch_size=256,
    learning_starts=5000,
    policy_lr=3e-4,
    q_lr=1e-3,
    policy_frequency=2,
    target_network_frequency=1,
    alpha=0.2,
    autotune=True,
    env_num=20,
    envs_args=env_args,
    summary_steps=10000,
    reward_norm=False
)

# 初始化AquaML
AquaML.init(
    hyper_params=param,  # 传入超参数
    root_path='test',  # 传入文件夹根路径，用于创建文件夹
    memory_path='test',  # 传入贡献内存标识符
    wandb_project='test',  # 传入wandb的项目名称
    engine='torch'  # 传入引擎名称
)

######################
# 定义模型
######################

class Actor(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(3, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc_mean = torch.nn.Linear(256, 1)
        self.fc_logstd = torch.nn.Linear(256, 1)

        self.action_scale = torch.tensor([1.0], dtype=torch.float32)
        self.action_bias = torch.tensor([0.0], dtype=torch.float32)

        self.learning_rate = 3e-4  # 学习率
        self.optimizer_type = 'Adam'  # 优化器类型

        # 模型的输出信息
        self.output_info = AquaML.DataInfo(
            names=('action', 'log_pi','mean'),
            shapes=((1,), (1,), (1,)),
            dtypes=(np.float32, np.float32, np.float32)
        )

        # 模型的输入信息
        self.input_names = ('obs',)

        # 优化器的其他参数，参考pytorch
        self.optimizer_other_args = {
            'eps': 1e-5,
            'clipnorm': 0.5,  # 梯度裁剪，这个会自动创建optimizer step
        }

    def forward(self, obs):
        x = torch.nn.functional.relu(self.fc1(obs))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = torch.clamp(log_std, -5, 2)

        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class QFunction(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = torch.nn.Linear(3 + 1, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 1)

        self.learning_rate = 1e-3
        self.optimizer_type = 'Adam'

        # 模型的输出信息
        self.output_info = AquaML.DataInfo(
            names=('q_value',),
            shapes=((1,),),
            dtypes=(np.float32,)
        )

        # 模型的输入信息
        self.input_names = ('obs', 'action')

        # 优化器的其他参数，参考pytorch
        self.optimizer_other_args = {
            'eps': 1e-5,
            'clipnorm': 0.5,
        }

    def forward(self, obs, action):
        x = torch.cat([obs, action], 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


model_dict = {
    'actor': Actor,
    'qf1': QFunction,
    'qf2': QFunction
}

rl = RL(
    env_class=GymnasiumMaker,
    algo=SACAlgo,
    hyper_params=param,
    model_dict=model_dict,
    checkpoint_path='C:/Users/29184/Documents/GitHub/AquaML/test/history_model/SAC/100',  # 检查点路径
    testing=True,  # 测试模式
    save_trajectory=True  # 保存轨迹
)

rl.run()  # 运行RL