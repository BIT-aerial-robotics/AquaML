#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Dict

from AquaML.learning.model import Model
from AquaML.learning.model.model_cfg import ModelCfg
from AquaML.learning.reinforcement.on_policy.ppo import PPO, PPOCfg
from AquaML.learning.model.gaussian import GaussianModel
from AquaML.environment.gymnasium_envs import GymnasiumWrapper
from AquaML.learning.trainers.sequential import SequentialTrainer
from AquaML.learning.trainers.base import TrainerConfig

# 自动初始化默认文件系统
from AquaML import coordinator


class PendulumPolicy(GaussianModel):
    """Pendulum环境策略模型"""
    
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
            nn.Linear(64, 1)
        )
        
        # 学习log标准差参数
        self.log_std_parameter = nn.Parameter(torch.zeros(1))
        
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
        mean = self.net(states)
        log_std = self.log_std_parameter.expand_as(mean)
        
        # 确保输出形状正确
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(0)
        
        return {"mean_actions": mean, "log_std": log_std}


class PendulumValue(Model):
    """Pendulum环境价值模型"""
    
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
            nn.Linear(64, 1)
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
        values = self.net(states)
        
        # 确保输出形状正确
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        
        return {"values": values}
        
    
    def act(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.compute(data_dict)


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
    
    # 4. 创建模型
    policy = PendulumPolicy(model_cfg)
    value = PendulumValue(model_cfg)
    
    # 5. 配置PPO参数 - 新数据流架构的关键参数
    ppo_cfg = PPOCfg()
    ppo_cfg.device = "cpu"
    ppo_cfg.memory_size = 200
    ppo_cfg.rollouts = 32  # 📊 关键参数：每32步触发一次训练
    ppo_cfg.learning_epochs = 4
    ppo_cfg.mini_batches = 2
    ppo_cfg.learning_rate = 3e-4
    ppo_cfg.mixed_precision = False
    
    # 6. 创建PPO智能体
    models = {"policy": policy, "value": value}
    agent = PPO(models, ppo_cfg)
    
    # 7. 创建训练器配置 - 简化配置，自动从agent读取参数
    trainer_cfg = TrainerConfig(
        timesteps=1000,
        headless=True,
        disable_progressbar=False
    )
    # collect_interval自动从PPO的rollouts参数读取，无需手动设置
    
    # 8. 创建训练器并开始训练 - 使用新数据流架构
    trainer = SequentialTrainer(env, agent, trainer_cfg)
    
    print(f"🌊 开始使用新数据流架构训练:")
    print(f"  📊 Rollouts: {ppo_cfg.rollouts} (每{ppo_cfg.rollouts}步训练一次)")
    print(f"  🔄 总时间步: {trainer_cfg.timesteps}")
    print(f"  📥 数据缓存格式: (num_env, steps, dims)")
    
    trainer.train()
    
    # 9. 显示训练统计信息
    status = trainer.get_enhanced_status()
    print(f"\n📈 训练统计:")
    print(f"  收集步数: {status['data_flow_architecture']['collected_steps']}")
    print(f"  训练轮次: {status['data_flow_architecture']['training_episodes']}")
    print(f"  数据效率: {status['data_flow_architecture']['collected_steps']/trainer_cfg.timesteps:.2f}")
    
    # 10. 保存模型到工作目录
    agent.save(fs.getModelPath(runner_name, "trained_model.pt"))
    print("✅ 训练完成！模型已保存")
    
    # 11. 验证数据缓存功能
    print(f"\n🔍 验证数据缓存:")
    available_buffers = trainer.list_available_buffers()
    print(f"  可用缓存: {available_buffers}")
    
    # 显示部分缓存数据信息
    for buffer_name in available_buffers[:3]:  # 只显示前3个
        data = trainer.get_collected_buffer_data(buffer_name)
        if data is not None:
            print(f"  {buffer_name}: 形状 {data.shape}")
    
    print(f"\n🌊 新数据流架构演示完成！")


if __name__ == "__main__":
    main()