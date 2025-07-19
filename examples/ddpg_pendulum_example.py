#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.off_policy.ddpg import DDPG, DDPGCfg
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig

# 自动初始化默认文件系统
from AquaML import coordinator


class PendulumActor(Model):
    """Pendulum环境DDPG策略模型"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # 更深的网络结构
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Pendulum动作空间是[-2, 2]，tanh输出[-1, 1]
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取状态数据
        if "state" in data_dict:
            states = data_dict["state"]
        else:
            states = list(data_dict.values())[0]
        
        # 处理维度
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(-1, states.size(-1))
        
        # 前向传播
        actions = self.net(states)
        
        # 缩放tanh输出到动作空间[-2, 2]
        actions = actions * 2.0
        
        # 确保输出形状正确
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        
        return {"actions": actions}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


class PendulumCritic(Model):
    """Pendulum环境DDPG价值模型"""
    
    def __init__(self, model_cfg: ModelCfg):
        super().__init__(model_cfg)
        
        # 更深的网络结构 - 输入: 状态(3) + 动作(1) = 4
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def compute(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取状态数据
        if "state" in data_dict:
            states = data_dict["state"]
        else:
            states = list(data_dict.values())[0]
        
        # 获取动作数据
        if "taken_actions" in data_dict:
            actions = data_dict["taken_actions"]
        elif "actions" in data_dict:
            actions = data_dict["actions"]
        else:
            raise ValueError("No action found in data_dict for critic")
        
        # 处理维度
        if states.dim() == 1:
            states = states.unsqueeze(0)
        elif states.dim() > 2:
            states = states.view(-1, states.size(-1))
            
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        elif actions.dim() > 2:
            actions = actions.view(-1, actions.size(-1))
        
        # 连接状态和动作
        state_action = torch.cat([states, actions], dim=-1)
        
        # 前向传播
        q_values = self.net(state_action)
        
        # 确保输出形状正确
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(-1)
        
        return {"values": q_values}
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck噪声过程用于探索"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        self.x = torch.ones(self.size) * self.mu
    
    def sample(self, shape):
        dx = self.theta * (self.mu - self.x) * self.dt + \
             self.sigma * torch.sqrt(torch.tensor(self.dt)) * torch.randn_like(self.x)
        self.x = self.x + dx
        return self.x.clone().view(shape)


def main():
    # 1. 简单注册运行器（自动使用当前时间生成名称，自动创建workspace结构）
    runner_name = coordinator.registerRunner()
    print(f"✓ 运行器已注册: {runner_name}")
    
    # 获取文件系统实例（已自动初始化）
    fs = coordinator.getFileSystem()
    
    # 2. 创建环境
    env = GymnasiumWrapper("Pendulum-v1")
    
    # 3. 创建模型配置
    model_cfg = ModelCfg(
        device="cpu",
        inputs_name=["state"],
        concat_dict=False
    )
    
    # 4. 创建DDPG模型
    policy = PendulumActor(model_cfg)
    target_policy = PendulumActor(model_cfg)
    critic = PendulumCritic(model_cfg)
    target_critic = PendulumCritic(model_cfg)
    
    # 5. 创建探索噪声
    exploration_noise = OrnsteinUhlenbeckNoise(size=(1,), sigma=0.3)
    
    # 6. 配置DDPG参数 - 新数据流架构的关键参数
    ddpg_cfg = DDPGCfg()
    ddpg_cfg.device = "cpu"
    ddpg_cfg.memory_size = 10000
    ddpg_cfg.batch_size = 64  # 📊 关键参数：批量大小
    ddpg_cfg.learning_starts = 1000
    ddpg_cfg.gradient_steps = 1
    ddpg_cfg.discount_factor = 0.99
    ddpg_cfg.polyak = 0.005
    ddpg_cfg.actor_learning_rate = 1e-3
    ddpg_cfg.critic_learning_rate = 1e-3
    ddpg_cfg.exploration_noise = exploration_noise
    ddpg_cfg.exploration_initial_scale = 1.0
    ddpg_cfg.exploration_final_scale = 0.1
    ddpg_cfg.exploration_timesteps = 5000
    ddpg_cfg.random_timesteps = 1000
    ddpg_cfg.grad_norm_clip = 0.5
    ddpg_cfg.mixed_precision = False
    
    # 7. 创建DDPG智能体
    models = {
        "policy": policy,
        "target_policy": target_policy,
        "critic": critic,
        "target_critic": target_critic
    }
    agent = DDPG(models, ddpg_cfg)
    
    # 8. 创建训练器配置 - 简化配置，自动从agent读取参数
    trainer_cfg = TrainerConfig(
        timesteps=10000,
        headless=True,
        disable_progressbar=False
    )
    
    # 9. 创建训练器并开始训练 - 使用新数据流架构
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    
    print(f"🌊 开始使用新数据流架构训练DDPG:")
    print(f"  📊 内存大小: {ddpg_cfg.memory_size}")
    print(f"  📦 批量大小: {ddpg_cfg.batch_size}")
    print(f"  🔄 总时间步: {trainer_cfg.timesteps}")
    print(f"  📥 数据缓存格式: (num_env, steps, dims)")
    print(f"  🎯 探索噪声: OU噪声 (sigma={exploration_noise.sigma})")
    
    trainer.train()
    
    # 10. 显示训练统计信息
    status = trainer.get_enhanced_status()
    print(f"\n📈 训练统计:")
    print(f"  收集步数: {status['data_flow_architecture']['collected_steps']}")
    print(f"  训练轮次: {status['data_flow_architecture']['training_episodes']}")
    print(f"  数据效率: {status['data_flow_architecture']['collected_steps']/trainer_cfg.timesteps:.2f}")
    
    # 11. 保存模型到工作目录
    agent.save(fs.getModelPath(runner_name, "trained_ddpg_model.pt"))
    print("✅ 训练完成！模型已保存")
    
    # 12. 验证数据缓存功能
    print(f"\n🔍 验证数据缓存:")
    available_buffers = trainer.list_available_buffers()
    print(f"  可用缓存: {available_buffers}")
    
    # 显示部分缓存数据信息
    for buffer_name in available_buffers[:3]:  # 只显示前3个
        data = trainer.get_collected_buffer_data(buffer_name)
        if data is not None:
            print(f"  {buffer_name}: 形状 {data.shape}")
    
    print(f"\n🌊 新数据流架构DDPG演示完成！")


if __name__ == "__main__":
    main()